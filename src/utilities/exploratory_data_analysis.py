import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import os

from embeddings.tokenization import kmerize_parquet_joblib, load_labels


def load_stored_embeddings(dataset_file_path):
    print(f"Loading embeddings from: {dataset_file_path=}")
    z = np.load(dataset_file_path, allow_pickle=True)

    X = list(z["X"])  # object array → list of arrays 
    ids = list(z["ids"])  # map labels from current dict
    return X, ids


def embed_data(label_dict, dir_list, path = None, kmer_prefix="CGTCA", kmer_suffix_size = 4, cores = 4):
	if path is None:
		data_dict = kmerize_parquet_joblib(dir_list, kmer_prefix=kmer_prefix, kmer_suffix_size=kmer_suffix_size, nr_of_cores=cores, output_type="counts")
		ids = [gid for gid in data_dict.keys()]
		X = [data_dict[gid] for gid in ids]
	else:
		X, ids = load_stored_embeddings(path)


	X = [x for gid, x in zip(ids, X) if gid in label_dict]
	y = np.array([label_dict[gid] for gid in ids if gid in label_dict], dtype=np.int64)

	return X, y

	
labels_path = "../downloads/labels.csv"
input_data_directory = "../downloads"
output_data_directory = input_data_directory


phenotype = "madin_categorical_gram_stain"
labels = load_labels(file_path=labels_path, id = "genome_name", label = phenotype, sep = ",")
label_dict_literal, label_dict, int2label = labels["label_dict"], labels["label_dict_int"], labels["int2label"]

file_suffix = ".parquet"
dir_list = os.listdir(input_data_directory)
dir_list = [f'{input_data_directory}/{file}' for file in dir_list if file_suffix in file]

print(f'{dir_list=}')
