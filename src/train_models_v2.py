from dataclasses import dataclass
#from models.CNN_alternative_version import dataset_file_path
import numpy as np
import torch
import os, sys
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from sklearn.model_selection import train_test_split, GroupKFold, GroupShuffleSplit
from sklearn.metrics import balanced_accuracy_score, classification_report, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

import wandb

from embeddings.tokenization import load_labels, check_id_and_labels_exist
from embeddings.tokenization import KmerTokenizer
from embeddings.integer_embeddings import IntegerEmbeddings, OneHotEmbeddings
from embeddings.esmc_embeddings import ESMcEmbeddings

from models.Transformers_and_S4Ms import TransformerKmerClassifier
from models.CNN import CNNKmerClassifier, CNNKmerClassifier_v2, CNNKmerClassifier_w_embeddings, CNNKmerClassifierLarge
from models.RNN import OneHot_RNN_MLP_KmerClassifier, OneHot_RNN_small, RNN_MLP_KmerClassifier, RNNKmerClassifier
from tqdm import tqdm
from utilities.cliargparser import ArgParser



def embed_data(kmer_prefix = None, 
			   kmer_suffix_size = None, 
			   input_data_directory = None, 
			   output_directory = None, 
			   reembed = False, 
			   no_loading = False, 
			   label_dict = None, 
			   compress_embeddings = True,
			   embedding_class = "IntegerEmbeddings",
			   file_type = "parquet",
			   genome_col = "genome_id",
			   dna_sequence_col = "dna_sequence_col",
			   reverse_complement = True,
			   nr_of_cores = 2,
			   pooling = None,
			   esmc_model = None,
			   device = "cpu",
			   kmer_offset = 0):
	# Should return X and y
	if output_directory is None:
		output_directory = input_data_directory

	print(f'{embedding_class=}')
	
	
	print(f'Embedding dataset with {kmer_prefix=} and {kmer_suffix_size=} as {embedding_class=}.')
	
	if kmer_offset == 0:
		dataset_name = f'{kmer_prefix}_{kmer_suffix_size}_{embedding_class}' 
	else:
		dataset_name = f'{kmer_prefix}_{kmer_suffix_size}_offset{kmer_offset}_{embedding_class}'
	dataset_file_path = f'{output_directory}/{dataset_name}'
	

	if reembed:
		
		tokenizer = KmerTokenizer(
							input_data_directory,
							genome_col=genome_col,
							dna_sequence_col=dna_sequence_col,
							kmer_prefix=kmer_prefix,
							kmer_suffix_size=kmer_suffix_size,
							file_type=file_type,
							reverse_complement=reverse_complement,
							kmer_offset = kmer_offset,
							)
		token_collection = tokenizer.run_tokenizer(nr_of_cores=nr_of_cores)

		if embedding_class == "integer":
			embedder = IntegerEmbeddings(token_collection=token_collection, 
						kmer_suffix_size=kmer_suffix_size,
						compress_embeddings=compress_embeddings
						)
			embeddings = embedder.run_embedder(nr_of_cores=1)
			vocab_size = embedder.vocab_size
		elif embedding_class == "esmc":
			embedder = ESMcEmbeddings(token_collection=token_collection, 
						kmer_suffix_size=kmer_suffix_size,
						compress_embeddings=compress_embeddings,
						esmc_model = esmc_model,
						device = device,
						pooling = pooling
						)
			embeddings = embedder.run_embedder(nr_of_cores=1)
			vocab_size = None
		elif embedding_class == "onehot":
			embedder = OneHotEmbeddings(token_collection=token_collection)
			embeddings = embedder.run_embedder()
			vocab_size = embedder.vocab_size
			
		else:
			raise ValueError(f"Embedding class {embedding_class} not recognized. Aborting...")
		
		

		gid_and_strand_id = [[gid, strand_id] for gid, strands in embeddings.items() for strand_id in strands]

		X = [embeddings[gid][strand_id] for gid, strand_id in gid_and_strand_id]

		
		
		ids = [strand_id for _, strand_id in gid_and_strand_id]
		groups = [gid for gid, _ in gid_and_strand_id]
		print(f'{len(X)=}')
		print(f'{len(ids)=}')
		print(f'{len(groups)=}')

		

		if embedding_class == "integer":
			print(f"Saving embeddings to: {dataset_file_path}.npz")
			np.savez_compressed(f'{dataset_file_path}.npz', 
					  		X=np.array(X, dtype=object), 
					  		ids=np.array(ids, dtype=object), 
							groups=np.array(groups, dtype=object),
							vocab_size = vocab_size)
		elif embedding_class == "onehot":
			print(f"Saving embeddings to: {dataset_file_path}.npz")
			np.savez_compressed(f'{dataset_file_path}.npz', 
					  		X=np.array(X, dtype=object), 
					  		ids=np.array(ids, dtype=object), 
							groups=np.array(groups, dtype=object),
							vocab_size = vocab_size)
		elif embedding_class == "esmc":
			# Convert tensors to numpy before saving
			#X = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in X]
			#print(f'{X[0]=}')
			# X is a torch tensor list
			print(f"Saving embeddings (X) to: {dataset_file_path}.pt \nand metadata (ids and groups) to: {dataset_file_path}.npz")
			torch.save(X, f"{dataset_file_path}.pt")
			np.savez_compressed(f'{dataset_file_path}.npz', 
								ids=np.array(ids, dtype=object), 
								groups=np.array(groups, dtype=object)
								)
	
	elif kmer_prefix is not None and kmer_suffix_size is not None:

		if is_embedding_file(dataset_file_path, embedding_class=embedding_class):
			if no_loading is True:
				return True
			if embedding_class == "esmc":
				torch_device = torch.device(device)
				X, ids, groups = load_stored_embeddings(dataset_file_path, torch_device=torch_device)
				vocab_size = None
			else:
				X, ids, groups, vocab_size = load_stored_embeddings(dataset_file_path)
		else: 
			# Force reembed
			return embed_data(kmer_prefix = kmer_prefix, 
					 		kmer_suffix_size = kmer_suffix_size, 
							input_data_directory=input_data_directory, 
							output_directory=output_directory, 
							reembed = True,
							no_loading = no_loading, 
							label_dict=label_dict, 
							compress_embeddings=compress_embeddings,
							embedding_class=embedding_class,
							file_type = file_type,
							genome_col=genome_col,
							dna_sequence_col=dna_sequence_col,
							reverse_complement=reverse_complement,
							nr_of_cores = nr_of_cores,
							pooling = pooling,
							esmc_model = esmc_model,
							device = device,
			   				kmer_offset = kmer_offset,
						)
	elif is_embedding_file(dataset_file_path, embedding_class=embedding_class):
		if no_loading is True:
			return True
		# Don't reembed kmers
		# Load np array instead
		if embedding_class == "esmc":
			X, ids, groups = load_stored_embeddings(dataset_file_path, torch_device = "cpu")
			vocab_size = None
		else:
			X, ids, groups, vocab_size = load_stored_embeddings(dataset_file_path)
	   
	else:
		raise FileNotFoundError(f"No npz data file with params {kmer_prefix=} and {kmer_suffix_size=} was found! \nAborting...")

	# Select only the rows where y is not None
	if embedding_class == "integer":
		X = np.array([x for gid, x in zip(groups, X) if gid in label_dict], dtype = object)
	elif embedding_class == "esmc":
		X = np.array(
			[
				(x.detach().cpu() if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32))
				for gid, x in zip(groups, X) if gid in label_dict
			],
			dtype=np.float32
		)	
		vocab_size = 960

	elif embedding_class == "onehot":
		X = np.array([x for gid, x in zip(groups, X) if gid in label_dict], dtype = object)
	
		
	y = np.array([label_dict[gid] for gid in groups if gid in label_dict])
	groups = np.array([gid for gid in groups if gid in label_dict])
	
	
	print(f'{np.unique(y)=}')
	print(f'{len(y)=}')
	print(f'{len(X)=}')
	print(f'{X.shape=}')
	print(f'{len(groups)=}')
	print(f'{vocab_size=}')
	
	print(f'{np.array(X[0]).shape=}')
	return X, y, groups, vocab_size

def is_embedding_file(dataset_file_path, embedding_class = "integer"):
	if embedding_class == "integer":
		file_types = [".npz"]
	elif embedding_class == "esmc":
		file_types = [".npz", ".pt"]
	elif embedding_class == "onehot":
		file_types = [".npz"]
	else:
		raise ValueError(f"Embedding class {embedding_class} not recognized. Aborting...")
	
	for type in file_types:
		if not os.path.isfile(f'{dataset_file_path}{type}'):
			return False
			
	return True

def load_stored_embeddings(dataset_file_path, torch_device = None):
	if torch_device is None:
		print(f"Loading embeddings from: {dataset_file_path=}")
		z = np.load(f'{dataset_file_path}.npz', allow_pickle=True)

		X = list(z["X"])  # object array → list of arrays 
		ids = list(z["ids"])  # map labels from current dict
		groups = list(z["groups"])

		vocab_size = int(z["vocab_size"]) if "vocab_size" in z else None
		
		return X, ids, groups, vocab_size
	else:
		print(f"Loading embeddings from: {dataset_file_path=}")
		z = np.load(f'{dataset_file_path}.npz', allow_pickle=True)

		ids = list(z["ids"])  # map labels from current dict
		groups = list(z["groups"])

		X = torch.load(f'{dataset_file_path}.pt', map_location=torch_device)

		vocab_size = int(z["vocab_size"]) if "vocab_size" in z else None
		
		return X, ids, groups

class SequenceDataset(Dataset):
	"""Dataset that returns variable-length token sequences and labels.
	If X is 2D padded with pad_id=0, trailing zeros are trimmed per sample.
	If X is an object array/list of 1D arrays, those are returned directly."""
	def __init__(self, X, y, pad_id: int = 0,):
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
	

class EmbeddingSequenceDataset(Dataset):
	def __init__(self, X, y):
		assert len(X) == len(y)
		self.X = X
		self.y = y
		#print(f'{X[0]=}')
	def __getitem__(self, idx):
		x = self.X[idx]
		x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
		y = torch.as_tensor(self.y[idx], dtype=torch.long)
		return x, y
	def __len__(self): 
		return len(self.X)


class OneHotSequenceDataset(Dataset):
	"""Dataset for one-hot encoded sequences where each sample X[i] is:
	   - either a single 2D array [T, V]
	   - or a list/tuple of 2D arrays [T_j, V] to be concatenated along T.
	   Ensures C-contiguous float32 output.
	"""
	def __init__(self, X, y):
		assert len(X) == len(y), "X and y length mismatch"
		print(f'{X.shape=}')
		print(f'{y.shape=}')
		self.X = X
		self.y = y

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx: int):
		x = self.X[idx]
		# if isinstance(item, (list, tuple)):
		# 	segs = [np.asarray(seg, dtype=np.float32, order="C") for seg in item]
		# 	seq = np.concatenate(segs, axis=0)  # [T, V]
		# else:
		# 	seq = np.asarray(item, dtype=np.float32, order="C")  # [T, V]
		# # Ensure contiguous memory before torch.from_numpy
		# seq = np.ascontiguousarray(seq, dtype=np.float32)
		# print(f'{seq.shape=}')
		
		x = torch.as_tensor(x, dtype=torch.float32)
		y = torch.as_tensor(self.y[idx], dtype=torch.long)
		return x, y


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



def pad_onehot_collate(batch):
	"""Collate for one-hot data.
	batch: list of (x:[T_i,V], y)
	Returns:
	  x_padded: [B, T_max, V]
	  lengths : [B]
	  mask    : [B, T_max] bool
	  labels  : [B]
	"""
	seqs, labels = zip(*batch)
	lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
	padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
	padded = padded.contiguous()  # make sure it's contiguous
	T_max = padded.size(1)
	mask = torch.arange(T_max).unsqueeze(0) < lengths.unsqueeze(1)
	labels = torch.stack(labels)
	return padded, lengths, mask, labels



class PadCollate:
	"""Top-level callable wrapper to make collate_fn picklable for multiprocessing workers."""
	def __init__(self, collate_fn, pad_id: int = 0):
		self.pad_id = pad_id
		self.collate_fn = collate_fn

	def __call__(self, batch):
		if self.collate_fn == "pad_collate":
			return pad_collate(batch, pad_id=self.pad_id)
		# elif self.collate_fn == "pad_embed_collate":
		# 	return pad_embed_collate(batch)
		else:
			raise ValueError(f"Collate fn {self.collate_fn} not recognized")


# def pad_embed_collate(batch):
# 	seqs, labels = zip(*batch)
# 	lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
# 	padded = pad_sequence(seqs, batch_first=True)
# 	mask = torch.arange(padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
# 	labels = torch.stack(labels)
# 	return padded, lengths, mask, labels


# class PadEmbedCollate:
# 	"""Top-level callable wrapper to make pad_embed_collate_fn picklable for multiprocessing workers."""
# 	def __call__(self, batch):
# 		return pad_embed_collate(batch)
	

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
	kmer_suffix_size = None,
	pad_id=0, 
	trace_memory_usage = False,
	dropout = 0.2,
	wandb_run = None,
	patience = 15,
	embedding_class = "integer"):
	
	
	kernel_size = 7

	memory_usage = {"peak_allocated_gib": 0, "peak_reserved_gib" : 0}
	print(f'{vocab_size=}')
	#Initialize model, optimizer, loss function
	if model_type == "CNN":
		emb_dim = vocab_size if (vocab_size is not None and vocab_size < 16) else 16
		model = CNNKmerClassifier(vocab_size=vocab_size, 
							emb_dim=emb_dim, 
							#conv_dim=hidden_dim, 
							kernel_size=kernel_size, 
							num_classes=num_classes, 
							pad_id=pad_id,
							dropout=dropout,
							).to(device)
		weight_decay = 1e-2
		optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

	elif model_type == "CNN_v2":
		emb_dim = vocab_size if (vocab_size is not None and vocab_size < 16) else 16
		model = CNNKmerClassifier_v2(vocab_size=vocab_size, 
							emb_dim=emb_dim, 
							#conv_dim=hidden_dim, 
							kernel_size=kernel_size, 
							num_classes=num_classes, 
							pad_id=pad_id,
							dropout=dropout,
							).to(device)
		weight_decay = 1e-4
		optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

	elif model_type == "CNN_LARGE":
		emb_dim = vocab_size if (vocab_size is not None and vocab_size < 16) else 16
		model = CNNKmerClassifierLarge(vocab_size=vocab_size, 
							emb_dim=emb_dim, 
							#conv_dim=hidden_dim, 
							kernel_size=kernel_size, 
							num_classes=num_classes, 
							pad_id=pad_id,
							dropout=dropout,
							).to(device)
		weight_decay = 1e-2
		optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

	elif model_type == "CNNKmerClassifier_w_embeddings":
		# For ESM-c embeddings
		sample_batch = next(iter(train_loader))[0]  # [B,T,D]
		emb_dim = sample_batch.size(-1)
		model = CNNKmerClassifier_w_embeddings(
			emb_dim=emb_dim,
			kernel_size=kernel_size,
			num_classes=num_classes,
			dropout=dropout,
			).to(device)
		weight_decay = 1e-2
		optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

	elif model_type == "RNN":
		emb_dim = vocab_size if (vocab_size is not None and vocab_size < 16) else 16
		model = RNNKmerClassifier(vocab_size=vocab_size, 
							emb_dim=emb_dim, 
							#rnn_hidden=hidden_dim, 
							num_layers=1,
							bidirectional=True,
							num_classes=num_classes, 
							dropout=dropout,
							pad_id=pad_id).to(device)
		weight_decay = 1e-2
		optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

	elif model_type == "RNN_MLP":
		emb_dim = vocab_size if (vocab_size is not None and vocab_size < 16) else 16
		model = RNN_MLP_KmerClassifier(vocab_size=vocab_size, 
							emb_dim=emb_dim, 
							rnn_hidden=128, 
							num_layers=2,
							bidirectional=True,
							num_classes=num_classes, 
							pad_id=pad_id,
							dropout=dropout,
							emb_dropout=0.1,
							pooling="mean",
							norm="layer",
							).to(device)
		weight_decay = 1e-4
		optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

	elif model_type == "RNN_MLP_ONEHOT":
		
		model = OneHot_RNN_MLP_KmerClassifier(
							vocab_size=vocab_size,
							rnn_hidden=64, 
							num_layers=2,
							bidirectional=True,
							num_classes=num_classes,
							dropout=dropout,
							input_dropout=0.1,
							pooling="mean",
							norm="layer",
							).to(device)
		weight_decay = 1e-4
		optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

	elif model_type == "RNN_MLP_ONEHOT_SMALL":
		
		model = OneHot_RNN_small(
							vocab_size = vocab_size,
							rnn_hidden=64, 
							head_dim=32,
							num_layers=1,
							bidirectional=False,
							num_classes=num_classes,
							dropout=dropout,
							input_dropout=0.1,
							
							).to(device)
		weight_decay = 1e-4
		optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = weight_decay)


	elif model_type == "TRANSFORMER":
		emb_dim = 8
		model = TransformerKmerClassifier(
			vocab_size=vocab_size,
			emb_dim=emb_dim,
			nhead=4,
			ff_dim=128,
			num_layers=2,
			num_classes=num_classes,
			pad_id=pad_id,
			dropout=dropout,
			use_mask=True
			).to(device)
		weight_decay = 1e-2
		optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

	else:
		raise ValueError(f"No valid model type was specified. Model_type was {model_type}. Aborting...")


	weight = None
	if class_weight is not None:
		weight = torch.tensor(class_weight, dtype=torch.float32).to(device)

	
	criterion = nn.CrossEntropyLoss(weight=weight)

	print(model)
	
	

	# ----- Training loop -----

	early_stop_counter = 0
	for epoch in tqdm(range(num_epochs)):
		model.train()
		running_loss = 0.0
		total = 0
		correct = 0

		if embedding_class == "esmc":
			for xb, lengths, mask, yb in train_loader:
				xb = xb.to(device)
				yb = yb.to(device)
				

				optimizer.zero_grad()
				output = model(xb,lengths=lengths.to(device), mask=mask.to(device))
				loss = criterion(output, yb)

				loss.backward()
				optimizer.step()
				running_loss += loss.item()
				preds = output.argmax(dim=1)
				correct += (preds == yb).sum().item()
				total += yb.size(0)
		elif embedding_class == "integer":
			for xb, lengths, mask, yb in train_loader:
				xb = xb.to(device)
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
		elif embedding_class == "onehot":
			for xb, lengths, mask, yb in train_loader:
				xb = xb.to(device)
				yb = yb.to(device)
				lengths = lengths.to(device)
				mask = mask.to(device)

				optimizer.zero_grad()
				output = model(xb, lengths=lengths, mask=mask)
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
			if embedding_class == "esmc":
				for xb, lengths, mask, yb in val_loader:
					xb = xb.to(device)
					yb = yb.to(device)
					out = model(xb,lengths=lengths.to(device), mask=mask.to(device))
					val_running += criterion(out, yb).item()
			elif embedding_class == "integer":
				for xb, lengths, mask, yb in val_loader:
					xb = xb.to(device)
					yb = yb.to(device)
					out = model(xb)
					val_running += criterion(out, yb).item()
			elif embedding_class == "onehot":
				for xb, lengths, mask, yb in val_loader:
					xb = xb.to(device)
					yb = yb.to(device)
					lengths = lengths.to(device)
					mask = mask.to(device)
					out = model(xb, lengths=lengths, mask=mask)
					val_running += criterion(out, yb).item()
			val_loss = torch.tensor(val_running / max(len(val_loader), 1))
			train_acc = correct / total if total > 0 else 0.0
			wandb_run.log({"train_loss": loss, "val_loss": val_loss, "train_acc":train_acc})
			if epoch == 0:
				best_val_loss = val_loss
				best_model_state = model.state_dict()
			else:
				if val_loss < best_val_loss and val_loss is not np.nan:
					best_val_loss = val_loss
					best_model_state = model.state_dict()
					early_stop_counter = 0
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
		
			print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f} | Val loss: {val_loss.item():.4f} | Train Acc: {train_acc:.4f}")
		
		# memory stats
		if trace_memory_usage and torch.device("cuda") == device:
			torch.cuda.synchronize(device)
			peak_alloc_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
			peak_res_gib = torch.cuda.max_memory_reserved()  / (1024 ** 3)
			if peak_alloc_gib > memory_usage["peak_allocated_gib"]:
				memory_usage["peak_allocated_gib"] = peak_alloc_gib
			if peak_res_gib > memory_usage["peak_reserved_gib"]:
				memory_usage["peak_reserved_gib"] = peak_res_gib
			print(f"epoch {epoch}: peak alloc {peak_alloc_gib:.3f} GiB, peak reserved {peak_res_gib:.3f} GiB")
	# Test evaluation after training loop
	model.eval()
	with torch.no_grad():
		outs = []
		if embedding_class == "esmc":
			for xb, lengths, mask, yb in test_loader:
				xb = xb.to(device)
				out = model(xb,lengths=lengths.to(device), mask=mask.to(device))
				outs.append(out.cpu().numpy())
		elif embedding_class == "integer":
			for xb, lengths, mask, yb in test_loader:
				xb = xb.to(device)
				out = model(xb)
				outs.append(out.cpu().numpy())
		elif embedding_class == "onehot":
			for xb, lengths, mask, yb in test_loader:
				lengths = lengths.to(device)
				mask = mask.to(device)
				xb = xb.to(device)
				out = model(xb, lengths=lengths, mask=mask)
				outs.append(out.cpu().numpy())
			
	test_outputs = np.concatenate(outs, axis=0) if outs else np.empty((0, 2), dtype=np.float32)

	fit_model_results = {"test_outputs" : test_outputs, "memory_usage" : memory_usage}
		
	return fit_model_results


def get_model_performance(phenotype = None,
						  model_type = "CNN", 
						  kmer_prefixes = None, 
						  kmer_suffix_sizes = None, 
						  n_seeds = 3, 
						  label_dict = None, 
						  learning_rates = None, 
						  input_data_directory=None, 
						  output_directory = None, 
						  compress_embeddings = False,
						  trace_memory_usage = False,
						  epochs = None,
						  dropout = 0.2,
						  patience = 15,
						  embedding_class = None,
						  reembed = False,
						  device = "cpu",
						  file_type = "parquet",
						  genome_col = None,
						  dna_sequence_col = None,
						  nr_of_cores = 2,
						  pooling = "mean",
						  esmc_model = "esmc_300m",
						  test_val_split = [0.2, 1/8],
						  kmer_offset = 0,
						  reverse_complement = True,
						  ):
	
	num_epochs = epochs
	for prefix in kmer_prefixes:
		for suffix_size in kmer_suffix_sizes:
			print(f'Training models with {prefix=} and {suffix_size=}')
			X, y, groups, vocab_size = embed_data(
											kmer_prefix = prefix, 
											kmer_suffix_size=suffix_size, 
											input_data_directory=input_data_directory, 
											output_directory=output_directory,
											reembed=reembed,
											label_dict=label_dict, 
											compress_embeddings=compress_embeddings,
											embedding_class=embedding_class,
											file_type=file_type,
											genome_col=genome_col,
											dna_sequence_col=dna_sequence_col,
											reverse_complement=reverse_complement,
											nr_of_cores = nr_of_cores,
											pooling = pooling,
											esmc_model = esmc_model,
											device=device,
											kmer_offset=kmer_offset,
											)
			pad_id = 0
			num_classes = len(np.unique(y))

			
			for lr in learning_rates:
				with wandb.init(project="Phenotyping bacteria",
					entity="torbjornbak-technical-university-of-denmark",
					config={
					"learning_rate": lr,
					"architecture": model_type,
					"dataset": phenotype,
					"epochs": epochs,
					"kmer_prefix": prefix,
					"kmer_suffix_size": suffix_size,
					"dropout": dropout,
					"embedding_class": embedding_class,
					"compress_embeddings": compress_embeddings,
					"patience": patience,
					"test_val_split": test_val_split,
					},
					) as run:

					for seed in tqdm(range(n_seeds)):
						# Split in train and test
						gss_test =  GroupShuffleSplit(n_splits = 1, test_size = test_val_split[0], random_state = seed)
						train_val_idx, test_idx = next(gss_test.split(X, y, groups=groups))
						print(f'{train_val_idx=}, {test_idx=}')
						
						X_trainval, y_trainval = X[train_val_idx], np.array(y)[train_val_idx]
						groups_trainval = groups[train_val_idx]
						
						# Split train into train and val
						gss_val = GroupShuffleSplit(n_splits=1, test_size=test_val_split[1], random_state=42)
						train_idx, val_idx = next(gss_val.split(X_trainval, y_trainval, groups=groups_trainval))

					
						print(f'Training models with {prefix=}, {suffix_size=}, {lr=}, {seed=}, {compress_vocab_space=}')
					
						X_train, y_train = X_trainval[train_idx], y_trainval[train_idx]
						X_val,   y_val   = X_trainval[val_idx],   y_trainval[val_idx]
						X_test,  y_test  = X[test_idx],           np.array(y)[test_idx]

						label_encoder = LabelEncoder()

						# Fit on all labels (instead of only on y_train labels), prevents the problem that occasionally occur with small datasets, 
						# where y contains a previously unseen label
						label_encoder.fit(y)

						y_train = label_encoder.transform(y_train)
						y_test = label_encoder.transform(y_test)
						y_val = label_encoder.transform(y_val)

						
						binc = np.bincount(y_train, minlength=num_classes).astype(np.float32)
						# Avoid div by zero if a class is missing in train
						binc[binc == 0] = 1.0
						class_weight = (len(y_train) / (num_classes * binc)).astype(np.float32)


						learning_rate = lr
						# Build DataLoaders
						bs = 50 if model_type == "CNN" else 25
						num_workers = nr_of_cores

						if embedding_class == "esmc":
							print(f'Using EmbeddingSequenceDataset for {model_type=}')
							train_ds = EmbeddingSequenceDataset(X_train, y_train)
							val_ds = EmbeddingSequenceDataset(X_val, y_val)
							test_ds = EmbeddingSequenceDataset(X_test, y_test)
							train_loader = DataLoader(
							train_ds,
							batch_size=bs,
							shuffle=True,
							num_workers=num_workers,
							pin_memory=False,
							collate_fn=pad_onehot_collate,
							persistent_workers=(num_workers > 0),
							)
							val_loader = DataLoader(
								val_ds,
								batch_size=bs,
								shuffle=False,
								num_workers=0,
								collate_fn=pad_onehot_collate,
							)
							test_loader = DataLoader(
								test_ds,
								batch_size=bs,
								shuffle=False,
								num_workers=0,
								collate_fn=pad_onehot_collate,
							)

						# elif embedding_class == "esmc":
						# 	print(f'Using SequenceDataset for {model_type=}')
						# 	train_ds = SequenceDataset(X_train, y_train, pad_id=pad_id)
						# 	val_ds = SequenceDataset(X_val, y_val, pad_id=pad_id)
						# 	test_ds = SequenceDataset(X_test, y_test, pad_id=pad_id)
						# 	pad_collate_fn = "pad_collate"

						# 	train_loader = DataLoader(
						# 	train_ds,
						# 	batch_size=bs,
						# 	shuffle=True,
						# 	collate_fn=PadCollate(pad_collate_fn, pad_id=pad_id),
						# 	num_workers=num_workers,
						# 	pin_memory=False,
						# 	persistent_workers=(num_workers > 0),
						# 	)
						# 	val_loader = DataLoader(
						# 		val_ds,
						# 		batch_size=bs,
						# 		shuffle=False,
						# 		collate_fn=PadCollate(pad_collate_fn, pad_id=pad_id),
						# 		num_workers=0,
						# 	)
						# 	test_loader = DataLoader(
						# 		test_ds,
						# 		batch_size=bs,
						# 		shuffle=False,
						# 		collate_fn=PadCollate(pad_collate_fn, pad_id=pad_id),
						# 		num_workers=0,
						# 	)
						elif embedding_class == "onehot":
							print(f'Using OneHotSequenceDataset for {model_type=}')
							train_ds = OneHotSequenceDataset(X_train, y_train)
							val_ds   = OneHotSequenceDataset(X_val, y_val)
							test_ds  = OneHotSequenceDataset(X_test, y_test)
							pin = (device.type == "cuda")
							train_loader = DataLoader(
								train_ds,
								batch_size=bs,
								shuffle=True,
								num_workers=num_workers,
								pin_memory=pin,
								collate_fn=pad_onehot_collate,
								persistent_workers=(num_workers > 0),
							)
							val_loader = DataLoader(
								val_ds,
								batch_size=bs,
								shuffle=False,
								num_workers=0,
								pin_memory=pin,
								collate_fn=pad_onehot_collate,
							)
							test_loader = DataLoader(
								test_ds,
								batch_size=bs,
								shuffle=False,
								num_workers=0,
								pin_memory=pin,
								collate_fn=pad_onehot_collate,
							)
						else:
							print(f'Using SequenceDataset for {model_type=}')
							train_ds = SequenceDataset(X_train, y_train, pad_id=pad_id)
							val_ds = SequenceDataset(X_val, y_val, pad_id=pad_id)
							test_ds = SequenceDataset(X_test, y_test, pad_id=pad_id)
							

							train_loader = DataLoader(
							train_ds,
							batch_size=bs,
							shuffle=True,
							num_workers=num_workers,
							pin_memory=False,
							persistent_workers=(num_workers > 0),
							)
							val_loader = DataLoader(
								val_ds,
								batch_size=bs,
								shuffle=False,
								num_workers=0,
							)
							test_loader = DataLoader(
								test_ds,
								batch_size=bs,
								shuffle=False,
								num_workers=0,
							)

						#num_workers = min(8, os.cpu_count() or 2)
						if model_type == "TRANSFORMER":
							num_workers = 0
						else:
							num_workers = 2

						
						training_result = fit_model(train_loader, val_loader, test_loader,
												device=device,
												num_epochs=num_epochs,
												learning_rate=learning_rate, 
												class_weight=class_weight,
												num_classes=num_classes,
												model_type=model_type,
												vocab_size=vocab_size,
												pad_id=pad_id, 
												trace_memory_usage=trace_memory_usage,
												dropout = dropout,
												wandb_run = run,
												patience = patience, 
												embedding_class=embedding_class)
						
						y_test_pred, memory_usage = training_result["test_outputs"], training_result["memory_usage"]
						
						print(f'{y_test=}')
						print(f'{y_test_pred=}')
						report = classification_report(y_test, np.argmax(y_test_pred, axis=1), output_dict=True, zero_division="warn")
						conf_matrix = confusion_matrix(y_test, np.argmax(y_test_pred, axis=1), labels = [i for i in int2label.keys()])

						
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
								"kmer_offset": kmer_offset,
								"embedding_class": embedding_class,
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
								"confusion_matrix" : conf_matrix,
								"peak_allocated_gib" : memory_usage["peak_allocated_gib"],
								"peak_reserved_gib": memory_usage["peak_reserved_gib"],
							}
						)
						run.log(results.to_dict())

						dataset_name = f"tmp_result_{model_type}_{phenotype}_{"COMPRESSED" if compress_vocab_space else "UNCOMPRESSED"}_{prefix}_{suffix_size}_{seed}_{lr}_{embedding_class}"
						path = f'{output_directory}/{dataset_name}.csv'
						print(f'Finished training model with params:{prefix=}, {suffix_size=}, {lr=}, {seed=}, {compress_vocab_space=}')
						results.to_csv(path)
						print(f'Saved tmp result to {path=}')
						print(f'{results=}')

	return





if __name__ == "__main__":

	parser = ArgParser(module = "train_models")
	parser = parser.parser
	wandb.login()

	if torch.cuda.is_available(): 
		device = torch.device("cuda")
		
	elif torch.backends.mps.is_available(): 
		device = torch.device("mps")

	else: 
		# On CPU server
		device = torch.device("cpu")
		
	
	print(f"Using {device=}")



	id_column = parser.id_column
	dna_sequence_col = parser.dna_sequence_column
	labels_path = parser.labels_path
	input_directory = parser.input
	phenotypes = parser.phenotype
	

	kmer_prefixes = parser.kmer_prefixes
	kmer_suffix_sizes = parser.kmer_suffix_sizes
	print(f'{kmer_prefixes=}')
	print(f'{kmer_suffix_sizes=}')
	
	nr_of_cores = parser.cores
	output_directory = parser.output


	print(f'{labels_path=}')
	print(f'{input_directory=}')
	print(f'{output_directory=}')

	embed_only = parser.embed_only
	model_type = parser.model_arch
	compress_vocab_space = parser.compress
	trace_memory_usage = parser.trace_memory
	learning_rates = parser.lr
	epochs = parser.epochs
	dropout = parser.dropout
	k_folds = parser.k_folds
	freq_others = parser.freq_others
	patience = parser.patience
	reembed = parser.reembed
	embedding_class = parser.embedding
	file_type = parser.file_type
	esmc_model = parser.esmc_model
	esmc_pooling = parser.esmc_pooling
	test_val_split = parser.test_val_split
	kmer_offset = parser.kmer_offset
	reverse_complement = parser.reverse_complement

	print(f'{trace_memory_usage=}')
	print(f"{learning_rates=}")
	print(f'{compress_vocab_space=}')
	print(f'{test_val_split=}')
   # base_kmer = "CGTCACA"
		
	# if embedding_class == "esmc":
	# 	assert "CNNKmerClassifier_w_embeddings" == model_type, "Currently, ESMc embeddings can only be used with CNNKmerClassifier_w_embeddings model type"



	check_id_and_labels_exist(file_path=labels_path, id = id_column, labels = phenotypes, sep = ",")

	
	if embed_only is True:
		for phenotype in phenotypes:
			labels = load_labels(file_path=labels_path, id = id_column, label = phenotype, sep = ",", freq_others=freq_others)
			label_dict_literal, label_dict, int2label = labels["label_dict"], labels["label_dict_int"], labels["int2label"] 

			for prefix in kmer_prefixes:
				for suffix_size in kmer_suffix_sizes:
					pad_id = 0 # reserve 0 for padding in tokenizer
					result = embed_data(
										kmer_prefix = prefix, 
										kmer_suffix_size=suffix_size, 
										input_data_directory=input_directory, 
										output_directory=output_directory,
										reembed=reembed,
										label_dict=label_dict, 
										compress_embeddings=compress_vocab_space,
										embedding_class=embedding_class,
										file_type=file_type,
										genome_col=id_column,
										dna_sequence_col=dna_sequence_col,
										nr_of_cores = nr_of_cores
										)
	else:
		for phenotype in phenotypes:
			print(f'{phenotype=}')
			labels = load_labels(file_path=labels_path, id = id_column, label = phenotype, sep = ",", freq_others=freq_others)
			label_dict_literal, label_dict, int2label = labels["label_dict"], labels["label_dict_int"], labels["int2label"] 


			get_model_performance(phenotype=phenotype,
									model_type=model_type, 
									kmer_prefixes=kmer_prefixes, 
									kmer_suffix_sizes=kmer_suffix_sizes, 
									n_seeds = k_folds,
									label_dict=label_dict_literal,
									learning_rates=learning_rates, 
									input_data_directory=input_directory, 
									output_directory=output_directory, 
									compress_embeddings=compress_vocab_space,
									trace_memory_usage=trace_memory_usage,
									epochs = epochs,
									dropout = dropout,
									patience=patience,
									embedding_class=embedding_class,
									reembed=reembed,
									device=device,
									file_type=file_type,
									genome_col=id_column,
									dna_sequence_col=dna_sequence_col,
									nr_of_cores = nr_of_cores,
									pooling = esmc_pooling,
									esmc_model = esmc_model,
									test_val_split=test_val_split,
									kmer_offset = kmer_offset,
									reverse_complement=reverse_complement,
									)
