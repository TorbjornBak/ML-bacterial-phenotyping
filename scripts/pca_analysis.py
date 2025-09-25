from kmer_sampling import find_files_to_kmerize, kmer_sampling_multiple_files, load_labels


import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import os

from kmer_sampling import kmerize_parquet_count_joblib, load_labels
from CNN_v2 import load_stored_embeddings



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


def embed_data(label_dict, dir_list, path = None, kmer_prefix="CGTCA", kmer_suffix_size = 4, cores = 4):
	if path is None:
		data_dict = kmerize_parquet_count_joblib(dir_list, kmer_prefix=kmer_prefix, kmer_suffix_size=kmer_suffix_size, nr_of_cores=cores)
		ids = [gid for gid in data_dict.keys()]
		X = [data_dict[gid] for gid in ids]
	else:
		X, ids = load_stored_embeddings(path)


	X = [x for gid, x in zip(ids, X) if gid in label_dict]
	y = np.array([label_dict[gid] for gid in ids if gid in label_dict], dtype=np.int64)

	return X, y



if __name__ == "__main__":
	# file_names, labels = find_files_to_kmerize(directory="data", prefix = ".fna")
	# #labels = load_labels(file_path="downloads/genome_lineage")

	# X, y = kmer_sampling_multiple_files(directory="data", file_names=file_names, labels = labels)
	# #print()

	# X_np = np.stack(X, axis=0)
	phenotype = "madin_categorical_gram_stain"
	label_dict_literal, label_dict = load_labels(file_path=labels_path, id = "genome_name", label = phenotype, sep = ",")


	file_suffix = ".parquet"
	dir_list = os.listdir(input_data_directory)
	dir_list = [f'{input_data_directory}/{file}' for file in dir_list if file_suffix in file]

	print(f'{dir_list=}')

	kmer_prefix = "CGTCA"
	kmer_suffix_size = 4

	
	X, y = embed_data(label_dict=label_dict, kmer_prefix=kmer_prefix, kmer_suffix_size = kmer_suffix_size, cores = 20)


	pca = PCA(n_components=2, random_state=0)
	X_pcs = pca.fit_transform(X)

	print(pca.explained_variance_ratio_)

	labels = np.unique(y)

	label2id = {label: i for i, label in enumerate(labels) }

	color_list = [label2id[l] for l in y]


	plt.figure(figsize=(6,5))

	plt.scatter(X_pcs[:, 0], X_pcs[:, 1], c=color_list, cmap='coolwarm', edgecolor='k')
	plt.xlabel(f'PC1')
	plt.ylabel(f'PC2')
	plt.title('PCA projection')
	plt.legend(title='Label', frameon=False)
	plt.tight_layout()
	plt.savefig(f'results/pca_analysis_prefix_{kmer_prefix}_suffix_size_{kmer_suffix_size}.jpg')
