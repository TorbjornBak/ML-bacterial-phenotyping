import numpy as np
import torch
import os, sys
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report, roc_auc_score
from joblib import Parallel, delayed

from scripts.embeddings import load_labels, kmerize_parquet_joblib
from scripts.models.Transformers_and_S4Ms import TransformerKmerClassifier
from tqdm import tqdm
import re


def parse_cli():
    if len(sys.argv) > 1:
        cli_arguments = {arg.split("=")[0].upper() : arg.split("=")[1] for arg in sys.argv[1:]}
        print(cli_arguments)
    else:
        raise ValueError("No arguments was provided!")

    return cli_arguments




def create_embeddings(input_data_directory, kmer_prefix, kmer_suffix_size, nr_of_cores = 20):

    file_suffix = ".parquet"
    dir_list = os.listdir(input_data_directory)
    dir_list = [f'{input_data_directory}/{file}' for file in dir_list if file_suffix in file]

    X_dict = kmerize_parquet_joblib(dir_list, 
                                       kmer_prefix, 
                                       kmer_suffix_size, 
                                       nr_of_cores = nr_of_cores,
                                       output_type="one-hot")

    return X_dict



def filter_embeddings(X_dict, label_dict):
    X_dict = {gid:x for gid, x in X_dict.items() if gid in label_dict}
    #X = [x for gid, x in X_dict.items() if gid in label_dict]
    y_dict = {gid: label_dict[gid] for gid in X_dict.keys() if gid in label_dict}
    
    print(f'{len(X_dict)=}')
    print(f'{len(y_dict)=}')
    return X_dict, y_dict


def save_embeddings(X_dict, dataset_path):
    print(f"Saving embeddings to: {dataset_path=}")
    np.savez_compressed(dataset_path, X_dict=X_dict)
    return


def load_saved_embeddings(path):
    z = np.load(path, allow_pickle=True)
    X_dict = dict(z["X_dict"].item()) # object array → list of arrays 
    return X_dict

def load_embeddings(input_data_directory, output_data_directory, kmer_prefix, kmer_suffix_size, label_dict, nr_of_cores = 20, store = True):
    file_path = dataset_file_path(output_data_directory, kmer_prefix, kmer_suffix_size, args = "encoding_one-hot")
    if os.path.isfile(file_path):
        X_dict = load_saved_embeddings(file_path)
    else:
        X_dict = create_embeddings(input_data_directory, kmer_prefix, kmer_suffix_size, nr_of_cores)
        
        if store:
            save_embeddings(X_dict, dataset_path=file_path)
    
    X_dict, y_dict = filter_embeddings(X_dict, label_dict)

    return X_dict, y_dict

def dataset_file_path(path, kmer_prefix, kmer_suffix_size, args = "encoding_unknown", file_extension = "npz"):
    return f'{path}/{args}_prefix_{kmer_prefix}_suffix-size_{kmer_suffix_size}.{file_extension}'


class SequenceDataset(Dataset):
    def __init__(self, X_dict, y_dict=None, y_list=None, keys=None, dtype=torch.long):
        assert (y_dict is not None) ^ (y_list is not None), "Provide exactly one of y_dict or y_list"
        self.keys = list(X_dict.keys()) if keys is None else list(keys)
        self.X = X_dict
       
        self.y = y_dict
        print(y_dict)
        self._use_y_dict = True
        
        self.dtype = dtype

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        x = self.X[k]         # np.ndarray (R, C_i)
        y = self.y[k]    
        xi = torch.as_tensor(x, dtype=self.dtype)  # no padding here
        yi = torch.as_tensor(y, dtype=torch.long)
       
        return xi, yi

# REMOVE
# def pad_collate_deprecated(batch, pad_id: int = 0):
#     """Pad a batch of variable-length 1D LongTensors to the same length and build mask.
#     Returns: seqs_padded [B,T], lengths [B], mask [B,T], labels [B]"""
#     seqs, labels = zip(*batch)
#     seqs = [s if isinstance(s, torch.Tensor) else torch.as_tensor(s, dtype=torch.long) for s in seqs]
#     seqs = [s if s.numel() > 0 else torch.tensor([pad_id], dtype=torch.long) for s in seqs]
#     lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
#     seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=pad_id)
#     T = seqs_padded.size(1)
#     mask = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)
#     labels = torch.stack([torch.as_tensor(y, dtype=torch.long) for y in labels])
#     return seqs_padded, lengths, mask, labels

def pad_collate(batch, pad_id: int = 0):
    # batch is list of (xi, yi), with xi shape (R, C_i)
    xs, ys = zip(*batch)
    R = xs[0].shape[0]
    widths = torch.tensor([x.shape[1] for x in xs], dtype=torch.long)
    Cmax = int(widths.max())
    B = len(xs)

    padded = xs[0].new_full((B, R, Cmax), fill_value=pad_id)
    for i, x in enumerate(xs):
        c = x.shape[1]
        padded[i, :, :c] = x

    ys = torch.as_tensor(ys, dtype=torch.long)
    return padded, ys, widths  # widths can be used to build masks


# ----- CNN model: embedding -> Conv1d blocks -> global pool -> classifier -----
class CNNKmerClassifier(nn.Module):
    def __init__(self, emb_dim=128, conv_dim = 256, kernel_size = 7, num_classes=2, pad_id=0, dropout = 0.2):
        super().__init__()
        self.kernel_size = kernel_size
        # approximate 'same' padding per conv layer
        self.pad = kernel_size // 2
        
        self.head_dropout = nn.Dropout(dropout)
        #self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        # Reduce downsampling to avoid zero-length tensors. Use only two stride-2 stages and no max-pooling.
        self.conv = nn.Sequential(
            nn.Conv1d(16, 128, kernel_size=kernel_size, padding=self.pad, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout),

            nn.Conv1d(128, conv_dim, kernel_size=kernel_size, padding=self.pad, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout),

        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # → [B, C, 1]
        self.fc = nn.Linear(conv_dim, num_classes)

    def forward(self, token_ids):
        # token_ids: [B, T] Long
        x = self.emb(token_ids)          # [B, T, D]
        x = x.transpose(1, 2)            # [B, D, T] for Conv1d

        z = self.conv(x)                 # [B, C, T']

        
        feat = self.pool(z).squeeze(-1)     # [B, C]
        
        logits = self.fc(self.head_dropout(feat))              # [B, num_classes]
        return logits




# ----- RNN model: embedding -> BiGRU -> masked global pool -> classifier -----
class RNNKmerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=128,
        rnn_hidden=128,
        num_layers=1,
        bidirectional=True,
        num_classes=2,
        pad_id=0,
        dropout=0.1
    ):
        super().__init__()
       
        self.pad_id = pad_id
        self.bidirectional = bidirectional
        self.head_dropout = nn.Dropout(dropout)

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=rnn_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        feat_dim = rnn_hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, token_ids: torch.Tensor):
        # token_ids: [B, T] Long; mask: [B, T] Bool or 0/1
        B, T = token_ids.shape
        x = self.emb(token_ids)  # [B, T, D]
        x = x.contiguous()

        
        # No packing when mask is absent; GRU will process padded positions but we’ll mask in pooling
        self.gru.flatten_parameters()
        out, _ = self.gru(x)  # [B, T, H*dir] [web:13][web:16]

        
        # Global average over valid timesteps when lengths unknown
        feat = out.mean(dim=1)  # [B, H*dir]

        logits = self.fc(self.head_dropout(feat))  # [B, num_classes]
        return logits




def fit_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device,
    num_epochs=10,
    learning_rate=0.001,
    class_weight = None,
    num_classes = 2, 
    model_type = "CNN"):
    

    
    hidden_dim = 128
    emb_dim = 10
    kernel_size = 7
    dropout = 0.2


    #Initialize model, optimizer, loss function
    if model_type == "CNN":
        model = CNNKmerClassifier(
                            emb_dim=emb_dim, 
                            conv_dim=hidden_dim, 
                            kernel_size=kernel_size, 
                            num_classes=num_classes, 
                            pad_id=pad_id,
                            dropout=dropout,
                            ).to(device)
    
    # elif model_type == "RNN":
    #     model = RNNKmerClassifier(vocab_size=V, 
    #                         emb_dim=emb_dim, 
    #                         rnn_hidden=hidden_dim, 
    #                         num_layers=1,
    #                         bidirectional=True,
    #                         num_classes=num_classes, 
    #                         dropout=dropout,
    #                         pad_id=pad_id).to(device)
    
    # elif model_type == "TRANSFORMER":
    #     model = TransformerKmerClassifier(
    #         vocab_size=V,
    #         emb_dim=emb_dim,
    #         nhead=8,
    #         ff_dim=512,
    #         num_layers=4,
    #         num_classes=num_classes,
    #         pad_id=pad_id,
    #         dropout=dropout,
    #         use_mask=True
    #         ).to(device)
    else:
        raise ValueError("No model type was specified. Aborting...")


    weight = None
    if class_weight is not None:
        weight = torch.tensor(class_weight, dtype=torch.float32).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(weight=weight)

    print(model)

    # ----- Training loop -----

    early_stop_counter = 0
    patience = 15
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for xb, lengths, mask, yb in train_loader:
            print(xb)
            print(yb)
            xb = xb.to(device)
            #mask = mask.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
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
                yb = yb.to(device)
                out = model(xb)
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
            out = model(xb)
            outs.append(out.cpu().numpy())
        test_outputs = np.concatenate(outs, axis=0) if outs else np.empty((0, 2), dtype=np.float32)

    return test_outputs


def get_model_performance(model_type = "CNN", kmer_prefixes = None, kmer_suffix_sizes = None, n_seeds = 5, label_dict = None, nr_of_cores = 2, token_size = 2):
    results_df = pd.DataFrame(
        columns=[
            "phenotype",
            "model_name",
            "kmer_prefix",
            "kmer_suffix_size",
            "seed",
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
    for prefix in kmer_prefixes:
        for suffix_size in kmer_suffix_sizes:
            print(f'Training models with {prefix=} and {suffix_size=}')
            X_dict, y_dict = load_embeddings(input_data_directory=input_data_directory, output_data_directory=input_data_directory, kmer_prefix=prefix, kmer_suffix_size=suffix_size, label_dict=label_dict, nr_of_cores=nr_of_cores, store=True)
            num_classes = len(set(y_dict.values()))
            print(f'{num_classes=}')
            
            try:
                for seed in tqdm(range(n_seeds)):
                    print(f'{seed=}')
                    keys = list(X_dict.keys())
                    X_train, X_test, y_train, y_test = train_test_split(X_dict, y_dict, random_state = seed, test_size= 0.2)
                    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state = 42, test_size= 1/8) # Weird with the 1/8th if it should 60, 20, 20
                    print(f'{len(X_train)=}')
                    print(f'{len(X_val)=}')
                    print(f'{len(X_test)=}')
                    
                    num_epochs =  50
                    learning_rate = 1e-3
                    # Build DataLoaders
                    bs = 25
                    train_ds = SequenceDataset(X_dict, y_dict = y_train, keys = X_train)
                    val_ds = SequenceDataset(X_dict, y_dict = y_val, keys = X_val)
                    test_ds = SequenceDataset(X_dict, y_dict = y_test, keys = X_test)


                    num_workers = 1
                    
                    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, 
                                              collate_fn=pad_collate,
                                              num_workers=num_workers, pin_memory=(device.type=='cuda'),
                                                persistent_workers=True)
                    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn= pad_collate)
                    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn= pad_collate)
                    
                    print(train_loader)
                    
                    binc = np.bincount(y_train, minlength=num_classes).astype(np.float32)
                    # Avoid div by zero if a class is missing in train
                    binc[binc == 0] = 1.0
                    class_weight = (len(y_train) / (num_classes * binc)).astype(np.float32)

                    y_test_pred = fit_model(train_loader, val_loader, test_loader,
                                            device=device,
                                            num_epochs=num_epochs,
                                            learning_rate=learning_rate, 
                                            class_weight=class_weight,
                                            num_classes=num_classes,
                                            model_type=model_type)
                    
                    report = classification_report(y_test, np.argmax(y_test_pred, axis=1), output_dict=True)

                    
                    y_test_oh = np.eye(len(np.unique(y_train)))[y_test]
                    auc_weighted = roc_auc_score(y_test_oh, y_test_pred, average="weighted", multi_class="ovr")
                    auc_macro = roc_auc_score(y_test_oh, y_test_pred, average="macro", multi_class="ovr")

                    # Calculate balanced accuracy
                    balanced_accuracy = balanced_accuracy_score(y_test, np.argmax(y_test_pred, axis=1))

                    # Store results
                    results = pd.Series(
                        {
                            "phenotype": phenotype,
                            "model_name": model_type,
                            "kmer_prefix": prefix,
                            "kmer_suffix_size": suffix_size,
                            "seed": seed,
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
                    dataset_name = f"tmp_result_{model_type}_{phenotype}_{prefix}_{suffix_size}_{seed}"
                    path = f'{output_directory}/{dataset_name}.csv'
                    results.to_csv(path)
                    print(results)
                    results_df.loc[len(results_df)] = results
            
            except torch.OutOfMemoryError as error:
                print(f'''Torch memory error. Parameters for failed training: {model_type=}, {phenotype=}, {prefix=}, {suffix_size=}, {seed=}
                      \nContinuing with next combination of parameters after this error: {error=}''')

            except MemoryError as error:
                print(f'''Memory error: Parameters for failed training: {model_type=}, {phenotype=}, {prefix=}, {suffix_size=}, {seed=}
                      \nContinuing with next combination of parameters after this error: {error=}''')
           
    return results_df


if __name__ == "__main__":

    if torch.cuda.is_available(): 
        device = torch.device("cuda")
        labels_path = "/home/projects2/bact_pheno/bacbench_data/labels.csv"
        input_data_directory = "/home/projects2/bact_pheno/bacbench_data"

    elif torch.backends.mps.is_available(): 
        #device = torch.device("mps")
        device = torch.device("cpu")
        labels_path = "downloads/labels.csv"
        input_data_directory = "downloads"

    else: 
        # On CPU server
        #device = torch.device("cpu")
        device = "cpu"
        labels_path = "/home/projects2/bact_pheno/bacbench_data/labels.csv"
        input_data_directory = "/home/projects2/bact_pheno/bacbench_data"
    print(f"Using {device=}")




    id = "genome_name"
    phenotype = "madin_categorical_gram_stain"
    label_dict_literal, label_dict = load_labels(file_path=labels_path, id = id, label = phenotype, sep = ",")


  
    nr_of_cores = 2
    output_directory = input_data_directory

   
    pad_id = 0 # reserve 0 for padding in tokenizer


   # base_kmer = "CGTCACA"

    #kmer_prefixes = [base_kmer[:i] for i in range(5,len(base_kmer)+1,1)] # Fx. ['CG', 'CGT', 'CGTC', 'CGTCA', 'CGTCAC']
    # kmer_prefixes = ['CGTCACA','CGTCAC','CGTCA', 'CGTC']
    # kmer_suffix_sizes = [8,9,10,11,12]
    kmer_prefixes = ['CGTCAC']
    kmer_suffix_sizes = [4]
    
    
    model_type = "CNN"

    results_df = get_model_performance(model_type=model_type, kmer_prefixes=kmer_prefixes, kmer_suffix_sizes=kmer_suffix_sizes, label_dict=label_dict, nr_of_cores=1, token_size=2)
    dataset_name = f"{model_type}_train_full"
    path = f'{output_directory}/{dataset_name}.csv'
    results_df.to_csv(path_or_buf=path)
    print(results_df)