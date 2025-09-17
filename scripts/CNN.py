import numpy as np
import torch
import os, sys
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

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


label_dict_literal, label_dict_int = load_labels(file_path=labels_path, id = "genome_name", label = "madin_categorical_gram_stain", sep = ",")

def save_npz_dict(path, d):
    keys = list(d.keys())
    payload = {f"k{i}": arr for i, (_, arr) in enumerate(d.items())}
    np.savez_compressed(path, keys=np.array(keys, dtype=object), **payload)

def load_npz_dict(path):
    z = np.load(path, allow_pickle=True)
    keys = list(z["keys"])
    arrays = [z[f"k{i}"] for i in range(len(keys))]
    return dict(zip(keys, arrays))


def parse_cli():
    print(len(sys.argv))
    if len(sys.argv) > 1:
        cli_arguments = {arg.split("=")[0].upper() : arg.split("=")[1] for arg in sys.argv[1:]}
        print(cli_arguments)
    else:
        raise ValueError("No arguments was provided!")

    return cli_arguments

cli_arguments = parse_cli()


if "--REEMBED" in cli_arguments and cli_arguments["--REEMBED"].upper() == "TRUE":

    save_data_dict = True

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
            ids = label_dict_literal.keys(), 
            kmer_prefix="CGTCAT", 
            kmer_suffix_size=8)

        data_dict.update(kmerized_sequences)
    
    if "--PATH" in cli_arguments:
        data_dict_path = cli_arguments["--PATH"]
        save_npz_dict(data_dict_path, data_dict)

elif "--PATH" in cli_arguments:
    # Don't reembed kmers
    # Load np array instead
    data_dict_path = cli_arguments["--PATH"]
    data_dict = load_npz_dict(data_dict_path)


else:
    raise ValueError("No data was provided!")




label_dict = label_dict_int


print(f"Using {device} device")

# ----- PyTorch Dataset wrapping the dict -----
class GenomeKmerDataset(Dataset):
    def __init__(self, seq_dict, label_dict):
        self.ids = list(seq_dict.keys())
        self.seq_dict = seq_dict
        self.label_dict = label_dict

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        gid = self.ids[idx]
        seq_np = self.seq_dict[gid]  # np.array of shape [L]
        y = self.label_dict[gid]
        # Convert to 1D LongTensor of token ids
        seq = torch.from_numpy(seq_np.astype(np.int64))  # [L] Long tensor on CPU
        target = torch.tensor(y, dtype=torch.long)       # scalar Long tensor on CPU
        return seq, target

# ----- Collate: pad sequences to max length in batch and build masks -----
def pad_collate(batch, pad_id=0):
    # batch is list of (seq[L], target)
    seqs, targets = zip(*batch)
    # seqs are torch tensors; use .size(0) to get length
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    # pad on the right to [B, T] with pad_id
    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    # mask True for real tokens
    T = seqs_padded.size(1)
    mask = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)
    targets = torch.stack(targets)  # [B]
    return seqs_padded, lengths, mask, targets

# ----- CNN model: embedding -> Conv1d blocks -> global pool -> classifier -----
class CNNKmerClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, conv_dim = 256, kernel_size = 7, num_classes=2, pad_id=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, 128, kernel_size=kernel_size, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, conv_dim, kernel_size=kernel_size, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=kernel_size, stride=2, padding=3),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # → [B, C, 1]
        self.fc = nn.Linear(conv_dim, num_classes)

    def forward(self, token_ids, lengths=None, mask=None):
        # token_ids: [B, T] Long
        x = self.emb(token_ids)          # [B, T, D]
        x = x.transpose(1, 2)            # [B, D, T] for Conv1d
        z = self.conv(x)                 # [B, C, T']
        z = self.pool(z).squeeze(-1)     # [B, C]
        logits = self.fc(z)              # [B, num_classes]
        return logits

# ----- Instantiate loader and model -----
V = (4**8)+1            # example vocab size; set to 4**k in real use
pad_id = 0          # reserve 0 for padding in your tokenizer


# ----- Split into train/test only -----

ids = list(data_dict.keys())
train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=42, stratify=[label_dict[i] for i in ids])

def subset_dict(d, keys):
    return {k: d[k] for k in keys}

train_dataset = GenomeKmerDataset(subset_dict(data_dict, train_ids), label_dict)
test_dataset = GenomeKmerDataset(subset_dict(data_dict, test_ids), label_dict)


batch_size = int(cli_arguments["--BATCH_SIZE"]) if "--BATCH_SIZE" in cli_arguments else 16
learning_rate = float(cli_arguments["--LR"]) if "--LR" in cli_arguments else 1e-3
kernel_size = int(cli_arguments["--KERNEL_SIZE"]) if "--KERNEL_SIZE" in cli_arguments else 7
conv_dim = int(cli_arguments["--CONV_DIM"]) if "--CONV_DIM" in cli_arguments else 256
emb_dim = int(cli_arguments["--EMB_DIM"]) if "--EMB_DIM" in cli_arguments else 128




train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: pad_collate(b, pad_id))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: pad_collate(b, pad_id))

model = CNNKmerClassifier(vocab_size=V, emb_dim=emb_dim, conv_dim=conv_dim, kernel_size=kernel_size, num_classes=2, pad_id=pad_id).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# ----- Training loop -----
epochs = int(cli_arguments["--EPOCHS"]) if "--EPOCHS" in cli_arguments else 5

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_batches = 0
    correct = 0
    total = 0
    for seqs_padded, lengths, mask, targets in train_loader:
        seqs_padded = seqs_padded.to(device)
        lengths = lengths.to(device)
        mask = mask.to(device)
        targets = targets.to(device)
        logits = model(seqs_padded, lengths, mask)
        loss = criterion(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_batches += 1
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    avg_train_loss = train_loss / train_batches
    train_acc = correct / total if total > 0 else 0
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")

# ----- Test evaluation -----
model.eval()
test_loss = 0.0
test_batches = 0
correct = 0
total = 0
with torch.no_grad():
    for seqs_padded, lengths, mask, targets in test_loader:
        seqs_padded = seqs_padded.to(device)
        lengths = lengths.to(device)
        mask = mask.to(device)
        targets = targets.to(device)
        logits = model(seqs_padded, lengths, mask)
        loss = criterion(logits, targets)
        test_loss += loss.item()
        test_batches += 1
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        
        total += targets.size(0)
avg_test_loss = test_loss / test_batches
test_acc = correct / total if total > 0 else 0

print(f'{correct=}/{total=}')
print(f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.4f}")
