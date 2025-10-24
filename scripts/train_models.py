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

from embeddings import load_labels, kmerize_parquet_joblib, compress_integer_embeddings
from models.Transformers_and_S4Ms import TransformerKmerClassifier
from models.CNN import CNNKmerClassifier
from models.RNN import RNNKmerClassifier
from tqdm import tqdm




def parse_cli():
    if len(sys.argv) > 1:
        cli_arguments = {arg.split("=")[0].upper() : arg.split("=")[1] for arg in sys.argv[1:]}
        print(cli_arguments)
    else:
        return dict()
        #raise ValueError("No arguments was provided!")

    return cli_arguments



def embed_data(kmer_prefix = None, 
               kmer_suffix_size = None, 
               input_data_directory = None, 
               output_directory = None, 
               reembed = None, 
               no_loading = False, 
               label_dict = None, 
               compress_vocab_space = False):
    # Should return X and y
    if output_directory is None:
        output_directory = input_data_directory

    
    print(f'Embedding dataset with {kmer_prefix=} and {kmer_suffix_size=}')
    dataset_name = f'{kmer_prefix}_{kmer_suffix_size}' 
    dataset_file_path = f'{output_directory}/{dataset_name}.npz'

    if reembed is True:

        data_dict = dict()

        file_suffix = ".parquet"
        dir_list = os.listdir(input_data_directory)
        dir_list = [f'{input_data_directory}/{file}' for file in dir_list if file_suffix in file]

        print(f'{dir_list=}')

        data_dict = kmerize_parquet_joblib(dir_list, kmer_prefix, kmer_suffix_size, nr_of_cores = nr_of_cores)
        
        ids = [gid for gid in data_dict.keys()]
        X = [data_dict[gid] for gid in ids]
            
        X_obj = np.array(X, dtype=object)
        dataset_name = f'{kmer_prefix}_{kmer_suffix_size}' 
        dataset_file_path = f'{output_directory}/{dataset_name}.npz'
        print(f"Saving embeddings to: {dataset_file_path=}")
        np.savez_compressed(dataset_file_path, X=X_obj, ids=np.array(ids, dtype=object))
    
    elif kmer_prefix is not None and kmer_suffix_size is not None:
        #kmer_prefix = prefix
        #kmer_suffix_size = suffix_size

        dataset_name = f'{kmer_prefix}_{kmer_suffix_size}' 
        dataset_file_path = f'{output_directory}/{dataset_name}.npz'

        
        if os.path.isfile(dataset_file_path):
            if no_loading is True:
                return True
            X, ids = load_stored_embeddings(dataset_file_path)
        else: 
            return embed_data(kmer_prefix = kmer_prefix, kmer_suffix_size = kmer_suffix_size, 
                              input_data_directory=input_data_directory, output_directory=output_directory, 
                              label_dict=label_dict, reembed = True,
                              compress_vocab_space=compress_vocab_space)
            
    elif os.path.isfile(dataset_file_path):
        if no_loading is True:
            return True
        # Don't reembed kmers
        # Load np array instead
        X, ids = load_stored_embeddings(dataset_file_path)
       
    else:
        raise FileNotFoundError(f"No npz data file with params {kmer_prefix=} and {kmer_suffix_size=} was found! \nAborting...")

    # Select only the rows where y is not None
    X = [x for gid, x in zip(ids, X) if gid in label_dict]
    
    y = np.array([label_dict[gid] for gid in ids if gid in label_dict], dtype=np.int64)

    if compress_vocab_space is True:
        X, vocab_size = compress_integer_embeddings(X, alphabet_size=4, kmer_suffix_size=kmer_suffix_size)
    else:
        vocab_size = 4**kmer_suffix_size+1
    print(f'{np.unique(y)=}')
    print(f'{len(y)=}')
    print(f'{len(X)=}')
    
    return X, y, vocab_size

def load_stored_embeddings(dataset_file_path):
    print(f"Loading embeddings from: {dataset_file_path=}")
    z = np.load(dataset_file_path, allow_pickle=True)

    X = list(z["X"])  # object array → list of arrays 
    ids = list(z["ids"])  # map labels from current dict
    return X, ids

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
    seqs = [s if s.numel() > 0 else torch.tensor([pad_id], dtype=torch.long) for s in seqs]
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=pad_id).contiguous()
    T = seqs_padded.size(1)
    mask = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)
    labels = torch.stack([torch.as_tensor(y, dtype=torch.long) for y in labels]).contiguous()
    return seqs_padded, lengths, mask, labels


class PadCollate:
    """Top-level callable wrapper to make collate_fn picklable for multiprocessing workers."""
    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def __call__(self, batch):
        return pad_collate(batch, pad_id=self.pad_id)



def fit_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device,
    num_epochs=10,
    learning_rate=0.001,
    class_weight = None,
    num_classes = 2, 
    model_type = "CNN",
    vocab_size = None,
    pad_id=0):
    

    
    hidden_dim = 128
    emb_dim = vocab_size if vocab_size < 16 else 16
    kernel_size = 7
    dropout = 0.2


    #Initialize model, optimizer, loss function
    if model_type == "CNN":
        model = CNNKmerClassifier(vocab_size=vocab_size, 
                            emb_dim=emb_dim, 
                            conv_dim=hidden_dim, 
                            kernel_size=kernel_size, 
                            num_classes=num_classes, 
                            pad_id=pad_id,
                            dropout=dropout,
                            ).to(device)
    
    elif model_type == "RNN":
        model = RNNKmerClassifier(vocab_size=vocab_size, 
                            emb_dim=emb_dim, 
                            rnn_hidden=hidden_dim, 
                            num_layers=1,
                            bidirectional=True,
                            num_classes=num_classes, 
                            dropout=dropout,
                            pad_id=pad_id).to(device)
    
    elif model_type == "TRANSFORMER":
        model = TransformerKmerClassifier(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            nhead=4,
            ff_dim=128,
            num_layers=4,
            num_classes=num_classes,
            pad_id=pad_id,
            dropout=dropout,
            use_mask=True
            ).to(device)
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
    patience = 30
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for xb, lengths, mask, yb in train_loader:
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
                if val_loss < best_val_loss and val_loss is not np.nan:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()
                else:
                    early_stop_counter += 1

                    if early_stop_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        model.load_state_dict(best_model_state)
                        break
                    elif val_loss is np.nan:
                        print(f"Early stopping at epoch {epoch + 1} due to validation loss being nan")
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


def get_model_performance(model_type = "CNN", kmer_prefixes = None, kmer_suffix_sizes = None, n_seeds = 3, label_dict = None, learning_rates = None, compress_vocab_space = False):
    results_df = pd.DataFrame(
        columns=[
            "phenotype",
            "model_name",
            "kmer_prefix",
            "kmer_suffix_size",
            "learning_rate",
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
            "vocab_compression",
            "onehot_encoded",
        ]
    )
    if learning_rates is not None:
        learning_rates = learning_rates
    elif model_type == "CNN":
        learning_rates = [1e-2, 1e-3, 1e-4]
    else:
        learning_rates = [1e-3, 1e-4]
    pad_id = 0
    num_epochs = 150
    for prefix in kmer_prefixes:
        for suffix_size in kmer_suffix_sizes:
            print(f'Training models with {prefix=} and {suffix_size=}')
            X, y, vocab_size = embed_data(kmer_prefix=prefix, kmer_suffix_size=suffix_size, 
                                          input_data_directory=input_data_directory, 
                                          label_dict=label_dict, compress_vocab_space=compress_vocab_space)
            
            onehotencoded = True
            if onehotencoded:
                y = F.one_hot(y, num_classes = 2)
                print(f'Onehot encoded {y=}')

            #vocab_size = (4**suffix_size)+1 
            num_classes = len(np.unique(y))
            #print(f'{vocab_size=}')
            #print(f'{X=}')
            
            
            for lr in learning_rates:
                for seed in tqdm(range(n_seeds)):
                    
                    print(f'Training models with {prefix=}, {suffix_size=}, {lr=}, {seed=}, {compress_vocab_space=}')
                
                    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = seed, test_size= 0.2)
                    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state = 42, test_size=1/8) # Weird with the 1/8th if it should 60, 20, 20, change to 2/8
                    
                    learning_rate = lr
                    # Build DataLoaders
                    bs = 50 if model_type == "CNN" else 25
                    train_ds = SequenceDataset(X_train, y_train, pad_id=pad_id)
                    val_ds = SequenceDataset(X_val, y_val, pad_id=pad_id)
                    test_ds = SequenceDataset(X_test, y_test, pad_id=pad_id)

                    #num_workers = min(8, os.cpu_count() or 2)
                    num_workers = 2
                    
                    train_loader = DataLoader(
                        train_ds,
                        batch_size=bs,
                        shuffle=True,
                        collate_fn=PadCollate(pad_id=pad_id),
                        num_workers=num_workers,
                        pin_memory=False,
                        persistent_workers=(num_workers > 0),
                    )
                    val_loader = DataLoader(
                        val_ds,
                        batch_size=bs,
                        shuffle=False,
                        collate_fn=PadCollate(pad_id=pad_id),
                        num_workers=0,
                    )
                    test_loader = DataLoader(
                        test_ds,
                        batch_size=bs,
                        shuffle=False,
                        collate_fn=PadCollate(pad_id=pad_id),
                        num_workers=0,
                    )
                    
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
                                            model_type=model_type,
                                            vocab_size=vocab_size,
                                            pad_id=pad_id)
                    
                    print(f'{y_test=}')
                    print(f'{y_test_pred=}')
                    report = classification_report(y_test, np.argmax(y_test_pred, axis=1), output_dict=True, zero_division="warn")

                    
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
                            "learning_rate" : lr,
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
                            "vocab_compression": compress_vocab_space,
                            "onehot_encoded" : onehotencoded,
                        }
                    )
                    dataset_name = f"tmp_result_{model_type}_{phenotype}_{"COMPRESSED" if compress_vocab_space else "UNCOMPRESSED"}_{prefix}_{suffix_size}_{seed}_{lr}"
                    path = f'{output_directory}/{dataset_name}.csv'
                    print(f'Finished training model with params:{prefix=}, {suffix_size=}, {lr=}, {seed=}, {compress_vocab_space=}')
                    results.to_csv(path)
                    print(results)
                    results_df.loc[len(results_df)] = results
                    #print()
                    # except torch.OutOfMemoryError as error:
                    #     print(f'''Torch memory error. Parameters for failed training: {model_type=}, {phenotype=}, {prefix=}, {suffix_size=}, {seed=}, {lr=}
                    #         \nContinuing with next combination of parameters after this error: {error=}''')

                    # except MemoryError as error:
                    #     print(f'''Memory error: Parameters for failed training: {model_type=}, {phenotype=}, {prefix=}, {suffix_size=}, {seed=}, {lr=}
                    #         \nContinuing with next combination of parameters after this error: {error=}''')
                    
                    # except ValueError as error:
                    #     print(f'Parameters for failed training: {model_type=}, {phenotype=}, {prefix=}, {suffix_size=}, {seed=}, {lr=}')
                    #     print(f'{error=}')

                    # except error:
                    #     print(f'{error}')
            
    return results_df


if __name__ == "__main__":

    if torch.cuda.is_available(): 
        device = torch.device("cuda")
        labels_path = "/home/projects2/bact_pheno/bacbench_data/labels.csv"
        input_data_directory = "/home/projects2/bact_pheno/bacbench_data"
        output_data_directory = "/home/projects2/bact_pheno/bacbench_data/results/"

    elif torch.backends.mps.is_available(): 
        device = torch.device("mps")
        #device = torch.device("cpu")
        labels_path = "downloads/labels.csv"
        input_data_directory = "downloads"

    else: 
        # On CPU server
        #device = torch.device("cpu")
        device = torch.device("cpu")
        labels_path = "/home/projects2/bact_pheno/bacbench_data/labels.csv"
        input_data_directory = "/home/projects2/bact_pheno/bacbench_data"
    print(f"Using {device=}")


    cli_arguments = parse_cli()

    id = "genome_name"
    phenotype = cli_arguments["--PHENOTYPE"] if "--PHENOTYPE" in cli_arguments else "madin_categorical_gram_stain"
    label_dict_literal, label_dict = load_labels(file_path=labels_path, id = id, label = phenotype, sep = ",")


    kmer_prefixes = cli_arguments["--KMER_PREFIXES"].split(",") if "--KMER_PREFIXES" in cli_arguments else None
    kmer_suffix_sizes = [int(size) for size in cli_arguments["--KMER_SUFFIX_SIZES"].split(",")] if "--KMER_SUFFIX_SIZES" in cli_arguments else None
    nr_of_cores = int(cli_arguments["--CORES"]) if "--CORES" in cli_arguments else 2
    output_directory = cli_arguments["--DATA_OUTPUT"].strip("/") if "--DATA_OUTPUT" in cli_arguments else output_data_directory


    #dataset_name = f'{kmer_prefix}_{kmer_suffix_size}' 
    #dataset_file_path = f'{output_directory}/{dataset_name}.npz'

    embed_only = cli_arguments["--EMBED_ONLY"] == "TRUE" if "--EMBED_ONLY" in cli_arguments else False
    model_type = cli_arguments["--MODEL_ARCH"] if "--MODEL_ARCH" in cli_arguments else "CNN"
    compress_vocab_space = cli_arguments["--COMPRESS"] == "TRUE" if "--COMPRESS" in cli_arguments else False

    learning_rates = [float(lr) for lr in cli_arguments["--LR"].split(",")] if "--LR" in cli_arguments else None
   # base_kmer = "CGTCACA"

    #kmer_prefixes = [base_kmer[:i] for i in range(5,len(base_kmer)+1,1)] # Fx. ['CG', 'CGT', 'CGTC', 'CGTCA', 'CGTCAC']
    
    # kmer_suffix_sizes = [8,9,10,11,12]
    #kmer_prefixes = ['CGT','CG']
    #kmer_suffix_sizes = [1,2]
    

    if embed_only is True:
        #Parallel(n_jobs = 4)(delayed(embed_data)(prefix, suffix_size, no_loading = True) for prefix in kmer_prefixes for suffix_size in kmer_suffix_sizes)
        for prefix in kmer_prefixes:
            for suffix_size in kmer_suffix_sizes:
                        # ----- Instantiate loader and model -----
                     # vocab size; 4**k + 1 (Adding 1 to make space for the padding which is 0)
                pad_id = 0          # reserve 0 for padding in tokenizer

                result = embed_data(prefix=prefix, suffix_size=suffix_size, input_data_directory=input_data_directory, label_dict=label_dict, no_loading=True)
    else:
        

        results_df = get_model_performance(model_type=model_type, kmer_prefixes=kmer_prefixes, kmer_suffix_sizes=kmer_suffix_sizes, label_dict=label_dict, learning_rates=learning_rates, compress_vocab_space=compress_vocab_space)
        dataset_name = f"{model_type}_train_full"
        path = f'{output_directory}/{dataset_name}.csv'
        results_df.to_csv(path_or_buf=path)
        print(results_df)