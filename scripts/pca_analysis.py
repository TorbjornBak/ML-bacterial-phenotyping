from kmer_sampling import find_files_to_kmerize, kmer_sampling_multiple_files, load_labels


import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



if __name__ == "__main__":
	file_names, labels = find_files_to_kmerize(directory="data", prefix = ".fna")
	#labels = load_labels(file_path="downloads/genome_lineage")
	
	X, y = kmer_sampling_multiple_files(directory="data", file_names=file_names, labels = labels)
	#print()

	X_np = np.stack(X, axis=0)

	pca = PCA(n_components=2, random_state=0)
	X_pcs = pca.fit_transform(X_np)

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
	plt.savefig('results/pca_analysis.jpg')
