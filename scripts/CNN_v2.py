import numpy as np
import torch
import os, sys
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report, roc_auc_score

from kmer_sampling import load_labels, read_parquet, kmerize_and_embed_parquet_dataset




if torch.cuda.is_available(): 
    device = torch.device("cuda")
    labels_path = "/home/projects2/bact_pheno/bacbench_data/labels.csv"
    data_directory = "/home/projects2/bact_pheno/bacbench_data"

# elif torch.backends.mps.is_available(): 
#     device = torch.device("mps")
else: 
    device = torch.device("cpu")
    labels_path = "downloads/labels.csv"
    data_directory = "downloads"


def parse_cli():
    if len(sys.argv) > 1:
        cli_arguments = {arg.split("=")[0].upper() : arg.split("=")[1] for arg in sys.argv[1:]}
        print(cli_arguments)
    else:
        raise ValueError("No arguments was provided!")

    return cli_arguments



cli_arguments = parse_cli()

id = "genome_name"
phenotype = cli_arguments["--PHENOTYPE"] if "--PHENOTYPE" in cli_arguments else "madin_categorical_"
label_dict_literal, label_dict = load_labels(file_path=labels_path, id = id, label = phenotype, sep = ",")


kmer_prefix = cli_arguments["--KMER_PREFIX"] if "--KMER_PREFIX" in cli_arguments else "CGTCAT"
kmer_suffix_size = int(cli_arguments["--K_SIZE"]) if "--K_SIZE" in cli_arguments else 8
dropout = float(cli_arguments["--DROPOUT"]) if "--DROPOUT" in cli_arguments else 0.2

def embed_data():
    # Should return X and y

    if "--REEMBED" in cli_arguments and cli_arguments["--REEMBED"].upper() == "TRUE":

        data_dict = dict()

        file_suffix = ".parquet"
        dir_list = os.listdir(data_directory)
        dir_list = [f'{data_directory}/{file}' for file in dir_list if file_suffix in file]

        print(f'{dir_list=}')

        for path in dir_list:

            parquet_df = read_parquet(parguet_path=path)

            kmerized_sequences = kmerize_and_embed_parquet_dataset(
                df = parquet_df, 
                genome_column= "genome_name", 
                dna_sequence_column= "dna_sequence", 
                kmer_prefix=kmer_prefix, 
                kmer_suffix_size=kmer_suffix_size)

            data_dict.update(kmerized_sequences)
        
        ids = [gid for gid in data_dict.keys()]
        X = [data_dict[gid] for gid in ids]

        if "--PATH" in cli_arguments:
            dataset_path = cli_arguments["--PATH"]
            # Save as object array for variable-length sequences, with ids and y
            X_obj = np.array(X, dtype=object)
            np.savez_compressed(dataset_path, X=X_obj, ids=np.array(ids, dtype=object))

        X = [x for gid, x in zip(ids, X) if gid in label_dict]
        y = np.array([label_dict[gid] for gid in ids if gid in label_dict], dtype=np.int64)
        

    elif "--PATH" in cli_arguments:
        # Don't reembed kmers
        # Load np array instead
        dataset_path = cli_arguments["--PATH"]
        z = np.load(dataset_path, allow_pickle=True)

        X = list(z["X"])  # object array → list of arrays 
        ids = list(z["ids"])  # map labels from current dict
        X = [x for gid, x in zip(ids, X) if gid in label_dict]
        y = np.array([label_dict[gid] for gid in ids if gid in label_dict], dtype=np.int64)
        


        # Select only the rows where y is not None
       
    else:
        raise ValueError("No data was provided! Aborting...")
    

    return X, y




print(f"Using {device} device")


class SequenceDataset(Dataset):
    """Dataset that returns variable-length token sequences and labels.
    If X is 2D padded with pad_id=0, trailing zeros are trimmed per sample.
    If X is an object array/list of 1D arrays, those are returned directly."""
    def __init__(self, X, y, pad_id: int = 0):
        assert len(X) == len(y), "X and y must have same length"
        self.X = X
        self.y = y
        self.pad_id = pad_id

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if isinstance(x, np.ndarray) and x.ndim == 1:
            # If padded, trim trailing pad tokens (== pad_id)
            if x.size > 0 and x.dtype != object:
                if self.pad_id == 0:
                    # Find last non-zero index
                    nz = np.nonzero(x)[0]
                    if nz.size > 0:
                        x = x[: nz[-1] + 1]
                    else:
                        x = x[:1]  # keep at least length 1 to avoid empty
            xi = torch.as_tensor(x, dtype=torch.long)
        else:
            # Row from 2D array
            row = np.asarray(x)
            if self.pad_id == 0:
                nz = np.nonzero(row)[0]
                if nz.size > 0:
                    row = row[: nz[-1] + 1]
                else:
                    row = row[:1]
            xi = torch.as_tensor(row, dtype=torch.long)
        yi = torch.as_tensor(self.y[idx], dtype=torch.long)
        return xi, yi


def pad_collate(batch, pad_id: int = 0):
    """Pad a batch of variable-length 1D LongTensors to the same length and build mask.
    Returns: seqs_padded [B,T], lengths [B], mask [B,T], labels [B]"""
    seqs, labels = zip(*batch)
    seqs = [s if isinstance(s, torch.Tensor) else torch.as_tensor(s, dtype=torch.long) for s in seqs]
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    T = seqs_padded.size(1)
    mask = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)
    labels = torch.stack([torch.as_tensor(y, dtype=torch.long) for y in labels])
    return seqs_padded, lengths, mask, labels


# ----- CNN model: embedding -> Conv1d blocks -> global pool -> classifier -----
class CNNKmerClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, conv_dim = 256, kernel_size = 7, num_classes=2, pad_id=0, use_mask: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        # approximate 'same' padding per conv layer
        self.pad = kernel_size // 2
        self.use_mask = use_mask
        self.head_dropout = nn.Dropout(dropout)
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        # Reduce downsampling to avoid zero-length tensors. Use only two stride-2 stages and no max-pooling.
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, 128, kernel_size=kernel_size, padding=self.pad, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout),

            nn.Conv1d(128, conv_dim, kernel_size=kernel_size, padding=self.pad, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout),

            nn.Conv1d(conv_dim, conv_dim, kernel_size=kernel_size, padding=self.pad, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # → [B, C, 1]
        self.fc = nn.Linear(conv_dim, num_classes)

    def forward(self, token_ids, mask: torch.Tensor | None = None):
        # token_ids: [B, T] Long
        x = self.emb(token_ids)          # [B, T, D]
        x = x.transpose(1, 2)            # [B, D, T] for Conv1d

        z = self.conv(x)                 # [B, C, T']

        if mask is not None:
            # Downsample mask to conv time steps and masked-average pool
            m = mask.float().unsqueeze(1)                       # [B,1,T]
            m = F.adaptive_avg_pool1d(m, output_size=z.size(-1))  # [B,1,T']
            w = (m > 0.5).float()                               # [B,1,T']
            denom = w.sum(dim=-1).clamp_min(1.0)                # [B,1]
            feat = (z * w).sum(dim=-1) / denom                  # [B, C]
        else:
            feat = self.pool(z).squeeze(-1)     # [B, C]
        
        logits = self.fc(self.head_dropout(feat))              # [B, num_classes]
        return logits

# ----- Instantiate loader and model -----
V = (4**kmer_suffix_size)+1      # vocab size; 4**k + 1 (Adding 1 to make space for the padding which is 0)
pad_id = 0          # reserve 0 for padding in tokenizer




batch_size = int(cli_arguments["--BATCH_SIZE"]) if "--BATCH_SIZE" in cli_arguments else 16
learning_rate = float(cli_arguments["--LR"]) if "--LR" in cli_arguments else 1e-3
kernel_size = int(cli_arguments["--KERNEL_SIZE"]) if "--KERNEL_SIZE" in cli_arguments else 7


def fit_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device,
    num_epochs=10,
    learning_rate=0.001,
    class_weight = None):
    

    
    conv_dim = int(cli_arguments["--CONV_DIM"]) if "--CONV_DIM" in cli_arguments else 256
    emb_dim = int(cli_arguments["--EMB_DIM"]) if "--EMB_DIM" in cli_arguments else 128


    #Initialize model, optimizer, loss function
    model = CNNKmerClassifier(vocab_size=V, 
                          emb_dim=emb_dim, 
                          conv_dim=conv_dim, 
                          kernel_size=kernel_size, 
                          num_classes=2, 
                          pad_id=pad_id).to(device)

    weight = None
    if class_weight is not None:
        weight = torch.tensor(class_weight, dtype=torch.float32).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(weight=weight)

    print(model)

    # ----- Training loop -----

    early_stop_counter = 0
    patience = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for xb, lengths, mask, yb in train_loader:
            xb = xb.to(device)
            mask = mask.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            output = model(xb, mask)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        loss = torch.tensor(running_loss / max(len(train_loader), 1))

        model.eval()
        with torch.no_grad():
            val_running = 0.0
            for xb, lengths, mask, yb in val_loader:
                xb = xb.to(device)
                mask = mask.to(device)
                yb = yb.to(device)
                out = model(xb, mask)
                val_running += criterion(out, yb).item()
            val_loss = torch.tensor(val_running / max(len(val_loader), 1))

            if epoch == 0:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
            else:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()
                else:
                    early_stop_counter += 1

                    if early_stop_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        model.load_state_dict(best_model_state)
                        break

        train_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f} | Val loss: {val_loss.item():.4f} | Train Acc: {train_acc:.4f}")

    # Test evaluation after training loop
    model.eval()
    with torch.no_grad():
        outs = []
        for xb, lengths, mask, _ in test_loader:
            xb = xb.to(device)
            mask = mask.to(device)
            out = model(xb, mask)
            outs.append(out.cpu().numpy())
        test_outputs = np.concatenate(outs, axis=0) if outs else np.empty((0, 2), dtype=np.float32)

    return test_outputs


def get_model_performance():
    # ----- Split into train/test only -----


    X, y = embed_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size= 0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state = 42, test_size= 1/8) # Weird with the 1/8th if it should 60, 20, 20

    num_epochs = int(cli_arguments["--EPOCHS"]) if "--EPOCHS" in cli_arguments else 5
    learning_rate = float(cli_arguments["--LR"]) if "--LR" in cli_arguments else 1e-3
    # Build DataLoaders
    bs = int(cli_arguments.get("--BATCH_SIZE", batch_size))
    train_ds = SequenceDataset(X_train, y_train, pad_id=pad_id)
    val_ds = SequenceDataset(X_val, y_val, pad_id=pad_id)
    test_ds = SequenceDataset(X_test, y_test, pad_id=pad_id)


    results_df = pd.DataFrame(
        columns=[
            "phenotype",
            "model_name",
            "f1_score_weighted",
            "f1_score_macro",
            "precision_weighted",
            "precision_macro",
            "precision_weighted",
            "precision_macro",
            "recall_weighted",
            "recall_macro",
            "accuracy",
            "balanced_accuracy",
            "auc_weighted",
            "auc_macro",
            "n_classes",
        ]
    )

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=lambda b: pad_collate(b, pad_id=pad_id))
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=lambda b: pad_collate(b, pad_id=pad_id))
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=lambda b: pad_collate(b, pad_id=pad_id))

    
    class_weight = len(y_train) / (len(np.unique(y_train)) * np.bincount(y_train))
    
    y_test_pred = fit_model(train_loader, val_loader, test_loader,
                            device=device,
                            num_epochs=num_epochs,
                            learning_rate=learning_rate, 
                            class_weight=class_weight)
    
    report = classification_report(y_test, np.argmax(y_test_pred, axis=1), output_dict=True)

    
    y_test_oh = np.eye(len(np.unique(y_train)))[y_test]
    auc_weighted = roc_auc_score(y_test_oh, y_test_pred, average="weighted", multi_class="ovr")
    auc_macro = roc_auc_score(y_test_oh, y_test_pred, average="macro", multi_class="ovr")

    # Calculate balanced accuracy
    balanced_accuracy = balanced_accuracy_score(y_test, np.argmax(y_test_pred, axis=1))

    # Store results
    results_df.loc[len(results_df)] = pd.Series(
        {
            "phenotype": phenotype,
            "model_name": "CNN",
            "f1_score_weighted": report["weighted avg"]["f1-score"],
            "f1_score_macro": report["macro avg"]["f1-score"],
            "precision_weighted": report["weighted avg"]["precision"],
            "precision_macro": report["macro avg"]["precision"],
            "recall_weighted": report["weighted avg"]["recall"],
            "recall_macro": report["macro avg"]["recall"],
            "accuracy": report["accuracy"],
            "balanced_accuracy": balanced_accuracy,
            "auc_weighted": auc_weighted,
            "auc_macro": auc_macro,
            "n_classes": len(np.unique(y_train)),
        }
    )

    return results_df

    
print(get_model_performance())