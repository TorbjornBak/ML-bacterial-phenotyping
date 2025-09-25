from kmer_sampling import find_files_to_kmerize, kmer_sampling_multiple_files, load_labels


import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

import sys

from kmer_sampling import kmerize_parquet_count_joblib, load_labels


def load_stored_embeddings(dataset_file_path):
    print(f"Loading embeddings from: {dataset_file_path=}")
    z = np.load(dataset_file_path, allow_pickle=True)

    X = list(z["X"])  # object array → list of arrays 
    ids = list(z["ids"])  # map labels from current dict
    return X, ids


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

def parse_cli():
    if len(sys.argv) > 1:
        cli_arguments = {arg.split("=")[0].upper() : arg.split("=")[1] for arg in sys.argv[1:]}
        print(cli_arguments)
    else:
        raise ValueError("No arguments was provided!")

    return cli_arguments

if __name__ == "__main__":

	device = "cpu"
	labels_path = "/home/projects2/bact_pheno/bacbench_data/labels.csv"
	input_data_directory = "/home/projects2/bact_pheno/bacbench_data"

	cli_arguments = parse_cli()
    
	phenotype = cli_arguments["--PHENOTYPE"] if "--PHENOTYPE" in cli_arguments else "madin_categorical_gram_stain"
	label_dict_literal, label_dict = load_labels(file_path=labels_path, id = "genome_name", label = phenotype, sep = ",")


	file_suffix = ".parquet"
	dir_list = os.listdir(input_data_directory)
	dir_list = [f'{input_data_directory}/{file}' for file in dir_list if file_suffix in file]

	print(f'{dir_list=}')

	kmer_prefix = cli_arguments["--KMER"] if "--KMER" in cli_arguments else "CGTCA"
	kmer_suffix_size = int(cli_arguments["--SUFFIX_SIZE"]) if "--SUFFIX_SIZE" in cli_arguments else 4

	
	X, y = embed_data(label_dict=label_dict, dir_list=dir_list, kmer_prefix=kmer_prefix, kmer_suffix_size = kmer_suffix_size, cores = 20)


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
