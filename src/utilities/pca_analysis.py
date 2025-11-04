
from utilities.cliargparser import ArgParser


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import umap.plot
import os
import numpy as np



from embeddings import kmerize_parquet_joblib, load_labels


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



if __name__ == "__main__":

	# if torch.cuda.is_available(): 
	# 	# On cuda gpu node
	# 	labels_path = "/home/projects2/bact_pheno/bacbench_data/labels.csv"
	# 	input_data_directory = "/home/projects2/bact_pheno/bacbench_data"
	# 	output_data_directory = "/home/projects2/bact_pheno/bacbench_data/results/"

	# elif torch.backends.mps.is_available(): 
	# 	# On Mac
	# 	labels_path = "downloads/labels.csv"
	# 	input_data_directory = "downloads"
	# 	output_data_directory = input_data_directory

	# else: 
	# 	# On CPU node
	# 	labels_path = "/home/projects2/bact_pheno/bacbench_data/labels.csv"
	# 	input_data_directory = "/home/projects2/bact_pheno/bacbench_data"
	# 	output_data_directory = "/home/projects2/bact_pheno/bacbench_data/results/"


	#cli_arguments = parse_cli()
	parser = ArgParser(module = "pca_analysis")
	parser = parser.parser
	
    
	#phenotype = cli_arguments["--PHENOTYPE"] if "--PHENOTYPE" in cli_arguments else "madin_categorical_gram_stain"
	phenotypes = parser.phenotype # is a list 
 
	labels_path = parser.labels_path
	id_column = parser.id_column
	input_data_directory = parser.input
	output_data_directory = parser.output

	for phenotype in phenotypes:

		label_return = load_labels(file_path=labels_path, id = id_column, label = phenotype, sep = ",")
		label_dict_literal, label_dict = label_return["label_dict"], label_return["label_dict_int"]

		file_suffix = ".parquet"
		dir_list = os.listdir(input_data_directory)
		dir_list = [f'{input_data_directory}/{file}' for file in dir_list if file_suffix in file]

		print(f'{dir_list=}')

		#kmer_prefix = cli_arguments["--KMER"] if "--KMER" in cli_arguments else "CGTCA"
		kmer_prefix = parser.kmer_prefix
		#kmer_suffix_size = int(cli_arguments["--SUFFIX_SIZE"]) if "--SUFFIX_SIZE" in cli_arguments else 4
		kmer_suffix_size = parser.kmer_suffix_size
		
		X, y = embed_data(label_dict=label_dict, dir_list=dir_list, kmer_prefix=kmer_prefix, kmer_suffix_size = kmer_suffix_size, cores = parser.cores)


		pca = PCA(n_components=2, random_state=0)
		X_pcs = pca.fit_transform(X)

		print(pca.explained_variance_ratio_)

		labels = np.unique(y)

		label2id = {label: i for i, label in enumerate(labels)}

		color_list = [label2id[l] for l in y]

		plt.figure(figsize=(6,5))

		plt.scatter(X_pcs[:, 0], X_pcs[:, 1], c=color_list, cmap='coolwarm', edgecolor='k')
		plt.xlabel(f'PC1')
		plt.ylabel(f'PC2')
		plt.title('PCA projection')
		plt.legend(title='Label', frameon=False)
		plt.tight_layout()
		pca_save_path = f'{output_data_directory}/pca_analysis_{phenotype}_prefix_{kmer_prefix}_suffix_size_{kmer_suffix_size}.jpg'
		plt.savefig(pca_save_path)

		print(f'{pca_save_path=}')

		mapper = umap.UMAP().fit(X)

		ax = umap.plot.points(mapper, labels = y)
		umap_save_path = f'{output_data_directory}/umap_{phenotype}_prefix_{kmer_prefix}_suffix_size_{kmer_suffix_size}.png'
		ax.figure.savefig(umap_save_path)
		print(f'{umap_save_path=}')