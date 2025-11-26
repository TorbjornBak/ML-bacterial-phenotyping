
from embeddings.esmc_embeddings import ESMcEmbeddings
import torch
from utilities.cliargparser import ArgParser


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report, roc_auc_score
from sklearn.metrics import confusion_matrix

from embeddings.KmerTokenization import KmerTokenizer, load_labels
from embeddings.integer_embeddings import KmerCountsEmbeddings
from dataclasses import dataclass

def load_stored_embeddings(dataset_file_path):
	print(f"Loading embeddings from: {dataset_file_path=}")
	z = np.load(dataset_file_path, allow_pickle=True)

	X = list(z["X"])  # object array → list of arrays 
	ids = list(z["ids"])  # map labels from current dict
	print(f'{len(X)=}')
	print(f'{len(ids)=}')
	return X, ids

def is_embedding_file(dataset_file_path, embedding_class = "frequency"):

	if embedding_class == "frequency":
		file_types = [".npz"]
	elif embedding_class == "counts":
		file_types = [".npz"]
	elif embedding_class == "esmc":
		file_types = [".npz", ".pt"]

	else:
		raise ValueError(f"Embedding class {embedding_class} not recognized. Aborting...")
	
	print(f'Checking for embedding file at: {dataset_file_path} with types: {file_types}')

	for type in file_types:
		if not os.path.isfile(f'{dataset_file_path}{type}'):
			print(f'Embedding file not found: {dataset_file_path}{type}')
			return False
	print(f'Embedding files found.')
	return True


def embed_data(label_dict, 
			   input_data_directory, 
			   output_data_directory,
			   kmer_prefix="CGTCA", 
			   kmer_suffix_size = 4, 
			   kmer_offset = 0,
			   id_column = "genome_name", 
			   sequence_column = "dna_sequence", 
			   embedding_class = "frequency",
			   cores = 4, 
			   file_type = "parquet", 
			   reembed = False, 
			   reverse_complement = False,
			   esmc_model = "esmc_300m",
			   esmc_pooling = "mean",
			   device = "cpu"):

	if embedding_class in ["frequency", "counts"]:

		embedder = KmerCountsEmbeddings(
						kmer_prefix=kmer_prefix,
						kmer_suffix_size=kmer_suffix_size,
						kmer_offset=kmer_offset,
						data_directory=output_data_directory,
						embedding_class=embedding_class,
		)
	elif embedding_class == "esmc":
		embedder = ESMcEmbeddings(
						kmer_prefix=kmer_prefix,
						kmer_suffix_size=kmer_suffix_size,
						kmer_offset=kmer_offset,
						data_directory=output_data_directory,
						esmc_model=esmc_model,
						pooling=esmc_pooling,
						device=device,
		)
	else:
		raise ValueError(f"Embedding class {embedding_class} not recognized. Aborting...")

	if reembed or not is_embedding_file(embedder.file_path, embedding_class=embedder.embedding_class):
		
		tokenizer = KmerTokenizer(
							input_data_directory,
							genome_col=id_column,
							dna_sequence_col=sequence_column,
							kmer_prefix=kmer_prefix,
							kmer_suffix_size=kmer_suffix_size,
							file_type=file_type,
							reverse_complement=reverse_complement,
							kmer_offset = kmer_offset,
							)
		token_collection = tokenizer.run_tokenizer(nr_of_cores=cores)

		embeddings = embedder.run_embedder(token_collection=token_collection)

		gid_and_strand_id = [[gid, strand_id] for gid, strands in embeddings.items() for strand_id in strands]
		print(f'{gid_and_strand_id[:10]=}')
		X = [embeddings[gid][strand_id] for gid, strand_id in gid_and_strand_id]
		ids = [strand_id for _, strand_id in gid_and_strand_id]
		groups = [gid for gid, _ in gid_and_strand_id]

		assert len(X) == len(ids) == len(groups), "Length mismatch in embeddings output!"
		assert len(X) > 0, "No embeddings were created! Aborting..."
		print(f'{len(X)=}')
		print(f'{len(ids)=}')
		print(f'{len(groups)=}')

		print(f'{X.shape=}')
		
		embedder.save_embeddings(X, ids, groups)

	else:
		X, ids, groups, channel_size = embedder.load_stored_embeddings()
		
	if embedder.embedding_class == "esmc":
		if esmc_pooling == "mean":
			X = np.array(
				[
					(x.detach().cpu() if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32))
					for gid, x in zip(groups, X[0]) if gid in label_dict
				],
				dtype=np.float32
			)
		
	else:
		X = [x for gid, x in zip(groups, X) if gid in label_dict]

	y = np.array([label_dict[gid] for gid in groups if gid in label_dict], dtype=np.int64)

	print(f'{len(X)=}')
	print(f'{len(y)=}')
	#print(label_dict)
	

	return X, y




def random_forest_classification(context):
	# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
	context.model_type = "RandomForest"
	for seed in range(context.k_folds):
	
		X_train, X_test, y_train, y_test = train_test_split(context.X, context.y, random_state = seed, test_size= 0.2)
		clf = RandomForestClassifier(max_depth=10, 
							   		random_state=0)
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		
		print(f'{y_test[:100]=}')
		print(f'{y_pred[:100]=}')
		print(f'Accuracy of RandomForest: {clf.score(X_test, y_test)}')
		
		create_classification_report(y_train=y_train, 
							   y_test=y_test, 
							   y_pred=y_pred, 
							   seed=seed, 
							   ctx=context)
	return clf

	
def hist_gradient_boosting_classifier(context):
	# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
	clf = None
	context.model_type = "HistGradientBoosting"
	for seed in range(context.k_folds):
		X_train, X_test, y_train, y_test = train_test_split(context.X, context.y, random_state = seed, test_size= 0.2)
		clf = HistGradientBoostingClassifier(
										loss = 'log_loss', 
										learning_rate=0.01, 
										l2_regularization = 1e-3,
										max_features=0.9,
										class_weight="balanced"
										)
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)


		# print(f'{y_test[:100]=}')
		# print(f'{y_pred[:100]=}')
		print(f'Accuracy of HistGradientBoost: {clf.score(X_test, y_test)}')
		
		create_classification_report(y_train=y_train, 
							   y_test=y_test, 
							   y_pred=y_pred, 
							   seed=seed, 
							   ctx=context)
		
	return clf


def gradient_boosting_classifier(context):
	# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
	clf = None
	context.model_type = "GradientBoosting"
	for seed in range(context.k_folds):
		X_train, X_test, y_train, y_test = train_test_split(context.X, context.y, random_state = seed, test_size= 0.2)
		clf = GradientBoostingClassifier(
										loss = 'log_loss', 
										learning_rate=0.01, 
										max_features=0.9,
										)
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)


		# print(f'{y_test[:100]=}')
		# print(f'{y_pred[:100]=}')
		print(f'Accuracy of GradientBoosting: {clf.score(X_test, y_test)}')
		
		create_classification_report(y_train=y_train, 
							   y_test=y_test, 
							   y_pred=y_pred, 
							   seed=seed, 
							   ctx=context)
		
	return clf

def pca_plot(context, save = True):
	pca = PCA(n_components=2, random_state=0)
	X_pcs = pca.fit_transform(context.X)

	print(pca.explained_variance_ratio_)

	labels = np.unique(context.y)

	label2id = {label: i for i, label in enumerate(labels)}

	color_list = [label2id[l] for l in context.y]

	plt.figure(figsize=(6,5))

	sns.scatterplot(x=X_pcs[:, 0], y=X_pcs[:, 1], hue=color_list,)
	plt.xlabel(f'PC1')
	plt.ylabel(f'PC2')
	plt.title('PCA projection')
	plt.legend(title='Label', frameon=False)
	plt.tight_layout()

	if save:

		
		pca_save_path = f'{context.output_directory}/pca_analysis_{context.phenotype}_prefix_{context.kmer_prefix}_suffix_size_{context.kmer_suffix_size}.jpg'
		plt.savefig(pca_save_path)

		print(f'{pca_save_path=}')
	plt.show()



# def umap_plot(context):
# 	mapper = umap.UMAP().fit(context.X)
# 	ax = umap.plot.points(mapper, labels = context.y)
# 	umap_save_path = f'{context.output_directory}/umap_{context.phenotype}_prefix_{context.kmer_prefix}_suffix_size_{context.kmer_suffix_size}.png'
# 	ax.figure.savefig(umap_save_path)
# 	print(f'{umap_save_path=}')


def kmer_frequency_plot(context):
	plt.hist(context.X, bins = 50)
	plt.xlabel('K-mer Frequency')
	plt.ylabel('Normalized counts')
	frequency_plot_path = f'{context.output_directory}/kmer_frequency_{context.phenotype}_prefix_{context.kmer_prefix}_suffix_size_{context.kmer_suffix_size}.png'
	print(f'{frequency_plot_path=}')
	plt.savefig(frequency_plot_path)


@dataclass
class model_context:
	X: np.array
	y: np.array
	output_directory: str
	phenotype: str
	kmer_prefix: str
	kmer_suffix_size: int
	model_type: str
	int2label: dict
	k_folds: int

def create_classification_report(y_train,
								 y_test, 
								 y_pred, 
								 seed,
								 ctx):

	report = classification_report(y_test, y_pred, output_dict=True, zero_division="warn")
	conf_matrix = confusion_matrix(y_test, y_pred, labels = list(ctx.int2label))

	#y_test_oh = np.eye(len(np.unique(y_train)))[y_test]
	#auc_weighted = roc_auc_score(y_test_oh, y_pred_probabilities, average="weighted", multi_class="ovr")
	#auc_macro = roc_auc_score(y_test_oh, y_pred_probabilities, average="macro", multi_class="ovr")

	# Calculate balanced accuracy
	balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

	# Store results
	results = pd.Series(
		{
			"phenotype": ctx.phenotype,
			"model_name": ctx.model_type,
			"seed" : seed,
			"kmer_prefix": ctx.kmer_prefix,
			"kmer_suffix_size": ctx.kmer_suffix_size,
			"f1_score_weighted": report["weighted avg"]["f1-score"],
			"f1_score_macro": report["macro avg"]["f1-score"],
			"precision_weighted": report["weighted avg"]["precision"],
			"precision_macro": report["macro avg"]["precision"],
			"recall_weighted": report["weighted avg"]["recall"],
			"recall_macro": report["macro avg"]["recall"],
			"accuracy": report["accuracy"],
			"balanced_accuracy": balanced_accuracy,
			"n_classes": len(np.unique(y_train)),
			"confusion_matrix" : conf_matrix,
			"int2label" : ctx.int2label,
		}
		)
	dataset_name = f"tmp_result_{ctx.model_type}_{ctx.phenotype}_{ctx.kmer_prefix}_{ctx.kmer_suffix_size}_{seed}"
	path = f'{ctx.output_directory}/{dataset_name}.csv'
	results.to_csv(path)
	print(f'Saved tmp result to {path=}')
	print(f'{results=}')
	return results



if __name__ == "__main__":

	
	parser = ArgParser(module = "pca_analysis")
	parser = parser.parser
	
	
	phenotypes = parser.phenotype

	if torch.cuda.is_available(): 
		device = torch.device("cuda")
		
	elif torch.backends.mps.is_available(): 
		device = torch.device("mps")

	else: 
		# On CPU server
		device = torch.device("cpu")

	for phenotype in phenotypes:

		label_return = load_labels(file_path=parser.labels_path, id = parser.id_column, label = phenotype, sep = ",")
		label_dict_literal, label_dict, int2label = label_return["label_dict"], label_return["label_dict_int"], label_return["int2label"] 

		kmer_prefix = parser.kmer_prefix
		kmer_suffix_size = parser.kmer_suffix_size
		
		X, y = embed_data(label_dict=label_dict, 
					input_data_directory=parser.input,
					output_data_directory=parser.output,
					kmer_prefix=parser.kmer_prefix, 
					kmer_suffix_size = parser.kmer_suffix_size, 
					kmer_offset=parser.kmer_offset,
					id_column = parser.id_column,
					sequence_column = parser.dna_sequence_column,
					cores = parser.cores, 
					embedding_class = parser.embedding,
					reembed=parser.reembed,
					file_type=parser.file_type,
					esmc_model=parser.esmc_model,
					esmc_pooling=parser.esmc_pooling,
					device=device
					)
		
		print(X)

		reembed = False  # only reembed once per dataset

		ctx = model_context(
							X,
							y, 
							parser.output, 
							phenotype, 
							kmer_prefix, 
							kmer_suffix_size,
							model_type=None,
							int2label=int2label,
							k_folds=5)
		
		
		# kmer_frequency_plot(ctx)
		# # Plotting pca and umap
		pca_plot(ctx)
		# # umap_plot(ctx)

		random_forest_classification(ctx)

		hist_gradient_boosting_classifier(ctx)


		
		

		

		