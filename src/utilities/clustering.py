import sourmash
from sourmash import MinHash, signature
import numpy as np
from sourmash import fig
import pandas as pd
from embeddings.KmerTokenization import KmerTokenizer, load_labels
from utilities.cliargparser import ArgParser
import seaborn as sns
import matplotlib.pyplot as plt

class SourMashClustering():
	def __init__(self,
				 kmer_suffix_size,
				 phenotype,
				 target_labels: dict):
		self.phenotype = phenotype
		self.kmer_suffix_size = kmer_suffix_size
		self.target_labels = target_labels
		
	def run_clustering(self, token_dict : dict[dict]):
		print(f'Running sourmash clustering on tokenized data')
		self.create_hashes(token_dict=token_dict)
		
		return self.minhashes

	def create_hashes(self, token_dict : dict[dict]):
		
		minhashes = {}

		for id, tokens in token_dict.items():         
		
			for strand, kmers in tokens.items():

				mh = sourmash.MinHash(n=500, ksize=self.kmer_suffix_size)

				for record in kmers:

					mh.add_sequence(record, True)

				minhashes[f'{id}_{strand}'] = mh
			
		self.minhashes = minhashes


	def jaccard_distance_matrix(self, minhashes : dict):

		distance_matrix = np.zeros((len(minhashes), len(minhashes)))
		labels = list(minhashes.keys())
		for i, label1 in enumerate(labels):
			for j, label2 in enumerate(labels):
				distance = minhashes[label1].jaccard(minhashes[label2])
				distance_matrix[i, j] = distance

		return distance_matrix, labels
	

	def plot_composite_matrix(self, distance_matrix, labels, output_path=None):
		f, reordered_labels, reordered_matrix = fig.plot_composite_matrix(distance_matrix, labels, labels)

		
		if output_path:
			f.savefig(f'{output_path.rstrip("/")}/output_smash_sourmash_distance_matrix.png')


		y = pd.Series([self.target_labels[label.split("_")[0]] for label in labels], index=labels)

		df = pd.DataFrame(distance_matrix, index=labels, columns=labels)

		unique_labels = y.unique()

		print(f'{unique_labels=}')

		lut = dict(zip(unique_labels, "rbg"))
		col_colors = y.map(lut)
		# see https://seaborn.pydata.org/generated/seaborn.clustermap.html
		plot = sns.clustermap(
			df,
			method="single",
			metric="euclidean",

			col_colors=col_colors,
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

		if output_path:
			plt.savefig(f'{output_path.rstrip("/")}/output_sns_sourmash_distance_matrix.png')
			#f.savefig()


if __name__ == "__main__":

	parser = ArgParser(module = "baseline")
	parser = parser.parser

	label_return = load_labels(file_path=parser.labels_path, id = parser.id_column, label = parser.phenotype[0], sep = ",")
	label_dict_literal, label_dict, int2label = label_return["label_dict"], label_return["label_dict_int"], label_return["int2label"] 
	
	tokenizer = KmerTokenizer(input_path=parser.input, 
								genome_col=parser.id_column,
								dna_sequence_col=parser.dna_sequence_column,
								kmer_prefix=parser.kmer_prefix,
								kmer_suffix_size=parser.kmer_suffix_size,
								file_type=parser.file_type,
								reverse_complement=parser.reverse_complement
								)
	
	token_collection = tokenizer.run_tokenizer(nr_of_cores=parser.cores)


	clusterer = SourMashClustering(kmer_suffix_size=parser.kmer_suffix_size, phenotype=parser.phenotype, target_labels=label_dict_literal)
	minhashes = clusterer.run_clustering(token_dict=token_collection)

	distance_matrix, labels = clusterer.jaccard_distance_matrix(minhashes=minhashes)

	clusterer.plot_composite_matrix(distance_matrix=distance_matrix, labels=labels, output_path=parser.output)

	