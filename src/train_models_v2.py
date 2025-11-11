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
from embeddings.integer_embeddings import IntegerEmbeddings

from models.Transformers_and_S4Ms import TransformerKmerClassifier
from models.CNN import CNNKmerClassifier, CNNKmerClassifierLarge
from models.RNN import RNNKmerClassifier
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
			   embedding_class = IntegerEmbeddings,
			   file_type = "parquet",
			   genome_col = "genome_id",
			   dna_sequence_col = "dna_sequence_col",
			   reverse_complement = True,
			   nr_of_cores = 2):
	# Should return X and y
	if output_directory is None:
		output_directory = input_data_directory

	print(embedding_class)
	if embedding_class == IntegerEmbeddings:
		emb_str = "IntegerEmbeddings"
	elif embedding_class == ESMcEmbeddings:
		emb_str = "ESMcEmbeddings"
	else:
		raise ValueError(f"Embedding class {embedding_class} does not exist")
	
	print(f'Embedding dataset with {kmer_prefix=} and {kmer_suffix_size=} as {emb_str=}.')
	
	dataset_name = f'{kmer_prefix}_{kmer_suffix_size}_{emb_str}' 
	dataset_file_path = f'{output_directory}/{dataset_name}.npz'
	

	if reembed:
		
		tokenizer = KmerTokenizer(
							input_data_directory,
							genome_col=genome_col,
							dna_sequence_col=dna_sequence_col,
							kmer_prefix=kmer_prefix,
							kmer_suffix_size=kmer_suffix_size,
							file_type=file_type,
							reverse_complement=reverse_complement
							)
		token_collection = tokenizer.run_tokenizer(nr_of_cores=nr_of_cores)

		
		embedder = embedding_class(token_collection=token_collection, 
					kmer_suffix_size=kmer_suffix_size,
					compress_embeddings=compress_embeddings)
		
		embeddings, vocab_size = embedder.run_embedder(nr_of_cores=nr_of_cores)

		gid_and_strand_id = [[gid, strand_id] for gid, strands in embeddings.items() for strand_id in strands]

		X = [embeddings[gid][strand_id] for gid, strand_id in gid_and_strand_id]
		ids = [strand_id for _, strand_id in gid_and_strand_id]
		groups = [gid for gid, _ in gid_and_strand_id]
		print(f'{len(X)=}')
		print(f'{len(ids)=}')
		print(f'{len(groups)=}')

		print(f"Saving embeddings to: {dataset_file_path=}")

		np.savez_compressed(dataset_file_path, 
					  		X=np.array(X, dtype=object), 
					  		ids=np.array(ids, dtype=object), 
							groups=np.array(groups, dtype=object),
							vocab_size = vocab_size)
	
	elif kmer_prefix is not None and kmer_suffix_size is not None:

		if os.path.isfile(dataset_file_path):
			if no_loading is True:
				return True
			X, ids, groups, vocab_size = load_stored_embeddings(dataset_file_path)
		else: 
			return embed_data(kmer_prefix = kmer_prefix, 
					 		kmer_suffix_size = kmer_suffix_size, 
							input_data_directory=input_data_directory, 
							output_directory=output_directory, 
							reembed = True,
							label_dict=label_dict, 
							compress_embeddings=compress_embeddings,
							embedding_class=embedding_class,
							file_type = file_type,
							genome_col=genome_col,
							dna_sequence_col=dna_sequence_col,
							nr_of_cores = nr_of_cores)
			
	elif os.path.isfile(dataset_file_path):
		if no_loading is True:
			return True
		# Don't reembed kmers
		# Load np array instead
		X, ids, groups, vocab_size = load_stored_embeddings(dataset_file_path)
	   
	else:
		raise FileNotFoundError(f"No npz data file with params {kmer_prefix=} and {kmer_suffix_size=} was found! \nAborting...")

	# Select only the rows where y is not None

	X = np.array([x for gid, x in zip(groups, X) if gid in label_dict], dtype = object)
	y = np.array([label_dict[gid] for gid in groups if gid in label_dict])
	groups = np.array([gid for gid in groups if gid in label_dict])
	
	print(f'{np.unique(y)=}')
	print(f'{len(y)=}')
	print(f'{len(X)=}')
	print(f'{len(groups)=}')
	print(f'{vocab_size=}')
	
	
	return X, y, groups, vocab_size

def load_stored_embeddings(dataset_file_path):
	print(f"Loading embeddings from: {dataset_file_path=}")
	z = np.load(dataset_file_path, allow_pickle=True)

	X = list(z["X"])  # object array → list of arrays 
	ids = list(z["ids"])  # map labels from current dict
	groups = list(z["groups"])

	vocab_size = int(z["vocab_size"]) if "vocab_size" in z else None
	
	return X, ids, groups, vocab_size



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
	pad_id=0, 
	trace_memory_usage = False,
	dropout = 0.2,
	wandb_run = None,
	patience = 15):
	

	#hidden_dim = 128
	if model_type == "TRANSFORMER":
		emb_dim = 8
	else:
		emb_dim = vocab_size if vocab_size < 16 else 16
	
	kernel_size = 7

	memory_usage = {"peak_allocated_gib": 0, "peak_reserved_gib" : 0}
	print(f'{vocab_size=}')
	#Initialize model, optimizer, loss function
	if model_type == "CNN":
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

	elif model_type == "CNN_LARGE":
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

	elif model_type == "RNN":
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

	
	elif model_type == "TRANSFORMER":
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
		raise ValueError("No model type was specified. Aborting...")


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
		for xb, lengths, mask, yb in test_loader:
			xb = xb.to(device)
			out = model(xb)
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
						  device = None,
						  file_type = "parquet",
						  genome_col = None,
						  dna_sequence_col = None,
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
											nr_of_cores = nr_of_cores
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
					},
					) as run:

					for seed in tqdm(range(n_seeds)):
						# Split in train and test
						gss_test =  GroupShuffleSplit(n_splits = 1, test_size = 0.2, random_state = seed)
						train_val_idx, test_idx = next(gss_test.split(X, y, groups=groups))
						print(f'{train_val_idx=}, {test_idx=}')
						X_trainval, y_trainval = X[train_val_idx], np.array(y)[train_val_idx]
						groups_trainval = groups[train_val_idx]
						
						# Split train into train and val
						gss_val = GroupShuffleSplit(n_splits=1, test_size=1/8, random_state=42)
						train_idx, val_idx = next(gss_val.split(X_trainval, y_trainval, groups=groups_trainval))

					
						print(f'Training models with {prefix=}, {suffix_size=}, {lr=}, {seed=}, {compress_vocab_space=}')
					
						X_train, y_train = X_trainval[train_idx], y_trainval[train_idx]
						X_val,   y_val   = X_trainval[val_idx],   y_trainval[val_idx]
						X_test,  y_test  = X[test_idx],           np.array(y)[test_idx]

						label_encoder = LabelEncoder()

						# Fit on all labels, prevents the problem that occasionally occur with small datasets, 
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
						train_ds = SequenceDataset(X_train, y_train, pad_id=pad_id)
						val_ds = SequenceDataset(X_val, y_val, pad_id=pad_id)
						test_ds = SequenceDataset(X_test, y_test, pad_id=pad_id)

						#num_workers = min(8, os.cpu_count() or 2)
						if model_type == "TRANSFORMER":
							num_workers = 0
						else:
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
												patience = patience)
						
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
						

						dataset_name = f"tmp_result_{model_type}_{phenotype}_{"COMPRESSED" if compress_vocab_space else "UNCOMPRESSED"}_{prefix}_{suffix_size}_{seed}_{lr}"
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

	print(f'{trace_memory_usage=}')
	print(f"{learning_rates=}")
	print(f'{compress_vocab_space=}')
   
   # base_kmer = "CGTCACA"
	if embedding_class == "integer":
		embedding_class = IntegerEmbeddings 
	elif embedding_class == "esmc":
		raise NotImplementedError
		embedding_class = ESMcEmbeddings

	check_id_and_labels_exist(file_path=labels_path, id = id_column, labels = phenotypes, sep = ",")


  
	if embed_only is True:
		for phenotype in phenotypes:
			labels = load_labels(file_path=labels_path, id = id_column, label = phenotype, sep = ",", freq_others=freq_others)
			label_dict_literal, label_dict, int2label = labels["label_dict"], labels["label_dict_int"], labels["int2label"] 

			for prefix in kmer_prefixes:
				for suffix_size in kmer_suffix_sizes:
					pad_id = 0 # reserve 0 for padding in tokenizer
					result = embed_data(prefix=prefix, suffix_size=suffix_size, input_data_directory=input_directory, label_dict=label_dict, no_loading=True)
					embed_data(
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
									dna_sequence_col=dna_sequence_col)
			