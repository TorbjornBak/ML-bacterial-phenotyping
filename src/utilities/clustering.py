import os
import sourmash
import numpy as np
from sourmash import fig
import pandas as pd
from embeddings.KmerTokenization import KmerTokenizer, load_labels, read_sequence_file
from utilities.cliargparser import ArgParser
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

class SourMashClustering():
	def __init__(self,
				 kmer_suffix_size,
				 target_labels: dict,
				 n = 1000,
				 id_column = None,
				 dna_sequence_column = None,):
		#self.phenotype = phenotype
		self.kmer_suffix_size = kmer_suffix_size
		self.target_labels = target_labels
		self.n = n
		self.id_column = id_column
		self.dna_sequence_column = dna_sequence_column

	def hash_tokens(self, token_dict : dict[dict]):
		
		minhashes = {}

		for id, tokens in token_dict.items():         
		
			for strand, kmers in tokens.items():

				mh = sourmash.MinHash(n=self.n, ksize=self.kmer_suffix_size)

				for record in kmers:

					mh.add_sequence(record, True)

				minhashes[f'{id}_{strand}'] = mh
			
		self.minhashes = minhashes
		return minhashes

	def hash_sequences(self, sequence_df):
		
		minhashes = {}

		for _, row in sequence_df.iterrows():         
			
				mh = sourmash.MinHash(n=self.n, ksize=self.kmer_suffix_size)

				for record in row[self.dna_sequence_column].split(" "):

					mh.add_sequence(record, True)

				minhashes[row[self.id_column]] = mh
			
		self.minhashes = minhashes
		return minhashes



	def jaccard_distance_matrix(self, minhashes : dict):

		distance_matrix = np.zeros((len(minhashes), len(minhashes)))
		labels = list(minhashes.keys())
		for i, label1 in enumerate(labels):
			for j, label2 in enumerate(labels):
				distance = minhashes[label1].jaccard(minhashes[label2])
				distance_matrix[i, j] = distance

		return distance_matrix, labels
	
	def group_clusters(self, distance_matrix, labels, threshold = None, method = "average", nr_of_clusters = None):
		df = pd.DataFrame(distance_matrix, index=labels, columns=labels)
		# Perform hierarchical clustering
		linkage_matrix = sch.linkage(df, method=method, metric='euclidean')

		if nr_of_clusters is None:
			# Form flat clusters based on the threshold
			assert threshold != None, "threshold should be set, cannot be None"
			cluster_groups = sch.fcluster(Z = linkage_matrix, t = threshold, criterion='distance')

			
		
		else:
			print(f'Searching for optimal clustering threshold to get {nr_of_clusters} clusters.')
			# binary search starting threshold
			ceiling = 1
			floor = 0

			assert nr_of_clusters < distance_matrix.shape[0]

			# Desired result = unique_clusters ~ nr_of_clusters

			while True:
				threshold = (ceiling + floor) / 2
				
				cluster_groups = sch.fcluster(Z = linkage_matrix, t = threshold, criterion='distance')

				unique_clusters = len(np.unique(cluster_groups))

				# if unique is within 10 % of desired we say it's good enough overlap
				if unique_clusters * 1.05 >= nr_of_clusters and unique_clusters * 0.95 <= nr_of_clusters:
					break
				
				elif threshold >= 0.99 or threshold <= 0.01:
					break
				
				# if unique clusters is too small, decrease threshold by decreasing the ceiling.
				elif unique_clusters < nr_of_clusters:
					ceiling = threshold
				# if unique clusters is too big, increase threshold by increasing the floor.
				elif unique_clusters > nr_of_clusters:
					floor = threshold
				print(f'{threshold=}, {unique_clusters=}, {ceiling=}, {floor=}')
				#break
			print(f'Final nr of {unique_clusters=}, {threshold=}')
		# Check cluster quality
		
		return cluster_groups



	def plot_composite_matrix(self, distance_matrix, labels, method = "average", title = None, subtitle = None):
		f, reordered_labels, reordered_matrix = fig.plot_composite_matrix(distance_matrix, labels, labels)

		# removes "forward" and "reverse" suffixes from labels
		y = pd.Series([self.target_labels["_".join(label.split("_")[:-1])] for label in labels if "_".join(label.split("_")[:-1]) in self.target_labels], index=labels)

		df = pd.DataFrame(distance_matrix, index=labels, columns=labels)

		unique_labels = y.unique()

		print(f'{unique_labels=}')

		lut = dict(zip(unique_labels, "rbg"))
		col_colors = y.map(lut)
		# see https://seaborn.pydata.org/generated/seaborn.clustermap.html
		linkage = sch.linkage(df, method=method, metric='euclidean')  # precompute linkage

		plot = sns.clustermap(
			df,
			method=method,
			metric='euclidean',

			col_colors=col_colors,
			col_linkage=linkage,
			xticklabels = [],
			yticklabels = [],
			#cbar_pos=(0.02, 0.8, 0.05, 0.18),
			cmap="YlGnBu",
			figsize=(10, 10)
		)
		
		legend_elements = []
		for label in unique_labels:
			legend_elements.append(
				plt.Line2D(
					[0], [0],
					marker='o',
					color='w',
					label=label,
					markerfacecolor=lut[label],
					markersize=10
				)
			)
		plot.ax_col_dendrogram.legend(
			handles=legend_elements)
		
		plot.ax_row_dendrogram.set_visible(False)
		if title:
			plot.ax_col_dendrogram.set_title(title)
		
		if subtitle:
			plot.ax_heatmap.set_title(subtitle)

		return f, plot


def list_files(input_path, file_type):
		dir_list = os.listdir(input_path)
	
		dir_list = [f'{input_path}/{file}' for file in dir_list if file_type == file.split(".")[-1]]
		
		assert len(dir_list) > 0, f'No files with type {file_type} found in {input_path}'
		print(f'Found {len(dir_list)} files with type {file_type} in {input_path}')
		
		return dir_list

if __name__ == "__main__":

	# TODO: filter by phenotype

	parser = ArgParser(module = "baseline")
	parser = parser.parser

	for phenotype in parser.phenotype:

		label_return = load_labels(file_path=parser.labels_path, id = parser.id_column, label = phenotype, sep = ",")
		label_dict_literal, label_dict, int2label = label_return["label_dict"], label_return["label_dict_int"], label_return["int2label"] 
		
		clusterer = SourMashClustering(kmer_suffix_size=parser.kmer_suffix_size, 
									target_labels=label_dict_literal, 
									n = parser.n_minhashes,
									id_column=parser.id_column,
									dna_sequence_column=parser.dna_sequence_column,)

		if parser.hash_full_sequence:
			list_dir = list_files(input_path=parser.input, file_type=parser.file_type)
			
			sequence_df = [read_sequence_file(file_path=file_path, file_type=parser.file_type) for file_path in list_dir]
			sequence_df = pd.concat(sequence_df, ignore_index=True)
			print(sequence_df)

			sequence_df = sequence_df[sequence_df[parser.id_column].isin(label_dict.keys())]
			
			minhashes = clusterer.hash_sequences(sequence_df=sequence_df)
			
		else:
			assert parser.kmer_prefix is not None, "kmer_prefix argument is required when hashing tokenized sequences"
			assert parser.kmer_suffix_size is not None, "kmer_suffix_size argument is required when hashing tokenized sequences"
			tokenizer = KmerTokenizer(input_path=parser.input, 
										genome_col=parser.id_column,
										dna_sequence_col=parser.dna_sequence_column,
										kmer_prefix=parser.kmer_prefix,
										kmer_suffix_size=parser.kmer_suffix_size,
										file_type=parser.file_type,
										reverse_complement=parser.reverse_complement
										)
			
			token_collection = tokenizer.run_tokenizer(nr_of_cores=parser.cores)
			# filter token collection to only include genomes with labels
			print(f'Filtering tokenized genomes to only include those with labels. Original size: {len(token_collection)}')
			token_collection = {k: v for k, v in token_collection.items() if k in label_dict.keys()}
			print(f'Filtered tokenized genomes to only include those with labels. New size: {len(token_collection)}')
			minhashes = clusterer.hash_tokens(token_dict=token_collection)

		print(f'Number of minhashes created: {len(minhashes)}')
		distance_matrix, labels = clusterer.jaccard_distance_matrix(minhashes=minhashes)

		cluster_groups = clusterer.group_clusters(distance_matrix=distance_matrix, labels=labels, threshold=None, nr_of_clusters=30)
		
		print(f'{cluster_groups=}')
		print(f'Number of clusters formed: {len(set(cluster_groups))}')
		print(f'Length of cluster groups: {len(cluster_groups)}')

		smash_plot, sns_plot = clusterer.plot_composite_matrix(distance_matrix=distance_matrix, 
															labels=labels, 
															title = parser.clustermap_title,
															subtitle = parser.clustermap_subtitle)

		if parser.output:
				smash_plot.savefig(f'{parser.output.rstrip("/")}/smash_sourmash_{parser.n_minhashes}_distance_matrix_{phenotype}_{parser.kmer_prefix}_{parser.kmer_suffix_size}.png')

		if parser.output:
				sns_plot.savefig(f'{parser.output.rstrip("/")}/sns_sourmash_{parser.n_minhashes}_distance_matrix_{phenotype}_{parser.kmer_prefix}_{parser.kmer_suffix_size}.png')
				#f.savefig()