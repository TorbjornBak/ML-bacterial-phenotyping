import os
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio import SeqIO

def read_parquet(parquet_path):
	# Read in as parquet one at a time, kmerize, convert to npy (or parquet?) Then we can stream and potentially use much bigger datasets?
	print(f"Loading parquet dataset: {parquet_path}")
	df = pd.read_parquet(parquet_path, engine = "fastparquet")
	return df

def read_fasta(fasta_path):
	# Assuming standard fasta format and that fasta file contains only one species
	# and that the fasta file name is the genome id
	
	
	sequence = []

	with open(fasta_path, mode = "r") as file:
		for record in SeqIO.parse(file, "fasta"):
			sequence.append(str(record.seq))
	
	df = pd.DataFrame({"genome_id" : [os.path.basename(fasta_path).rsplit(".", 1)[0]],
					   "dna_sequence" : [" ".join(sequence)]})
	
	return df

			

def read_sequence_file(file_path, file_type):
	if file_type == "parquet":
		df = read_parquet(file_path)
		return df
	elif file_type == "fasta":
		df = read_fasta(fasta_path=file_path)
		return df
	

def deprecated_load_labels(file_path, id = "genome_id", label = "class", sep = "\t", freq_others = None):
	assert isinstance(label, str), "label argument needs to be a single string, check that you are passing only one phenotype"
	df = pd.read_csv(file_path, sep = sep)
	print(f'{id=}, {label=}')
	df = df.dropna(subset = [id, label])

	label_dict = dict(zip(df[id].apply(str), df[label])) 

	unique_labels = np.unique(df[label])

	
	if freq_others is not None:
		freq = df[label].value_count(normalize = True)
		bottom_quantile = freq.quantile(q = freq_others)
		least_occuring_labels = freq[freq<=bottom_quantile]
		df.loc[df[label].isin(least_occuring_labels.index.tolist()), label] = "other" 
	
	
	label2int = {label: i for i, label in enumerate(unique_labels)}

	int2label = {i : label for i, label in enumerate(unique_labels)}

	label_dict_int = {id : label2int[label] for id, label in label_dict.items()}

	return_dict = {"label_dict":label_dict, "label_dict_int": label_dict_int, "int2label":int2label}
	
	return return_dict

def load_labels(file_path, id = "genome_id", label = "class", sep = "\t", subset_ratio = None):
	df = pd.read_csv(file_path, sep = sep)
	print(f'{id=}, {label=}')
	df = df.dropna(subset = [id, label])

	label_dict = dict(zip(df[id].apply(str), df[label])) 

	unique_labels = np.unique(df[label])

	if subset_ratio is not None:
		# Downsampling the dataset for specific analysis (if needed)
					
		# Downsample as a % of the full dataset
		print(f'Original dataset size: {len(df)=}')
		print(f'Original class distribution: {np.unique(df[label], return_counts=True)}')
		print(f'Downsampling dataset to {int(subset_ratio*len(df[label]))} samples...')

		selected_ids = np.random.choice(df[id].to_list(), size=int(len(df[id]) * subset_ratio), replace=False)
		unique_ids = np.unique(selected_ids)
		label_dict = {id : label for id, label in label_dict.items() if id in unique_ids}

		print(f'After downsampling: {len(unique_ids)=}')
		print(f'After downsampling class distribution: {np.unique(list(label_dict.values()), return_counts=True)}')
					

	# Creating label mappings
	
	label2int = {label: i for i, label in enumerate(unique_labels)}

	int2label = {i : label for i, label in enumerate(unique_labels)}

	# Creating mapping from id to integer labels

	label_dict_int = {id : label2int[label] for id, label in label_dict.items()}

	return_dict = {"label_dict":label_dict, "label_dict_int": label_dict_int, "int2label":int2label}
	
	return return_dict


class KmerTokenizer():

	def __init__(self, 
			  input_path, 
			  genome_col, 
			  dna_sequence_col, 
			  kmer_prefix,
			  kmer_suffix_size,
			  file_type,
			  reverse_complement,
			  kmer_offset = 0):
		
		self.input_path = input_path.rstrip("/")
		self.genome_col = genome_col
		self.dna_sequence_col = dna_sequence_col
		self.kmer_prefix = kmer_prefix
		self.kmer_suffix_size = kmer_suffix_size
		self.file_type = file_type
		self.reverse_complement = reverse_complement
		self.kmer_offset = kmer_offset

	def run_tokenizer(self, nr_of_cores = 2):
		file_paths = self.list_files()
		print(f'Starting tokenization with {nr_of_cores} cores...')
		if nr_of_cores == 1:
			tokenizer_results = [self.tokenize(file_path) for file_path in file_paths]
		else:
			tokenizer_results = Parallel(n_jobs = nr_of_cores)(delayed(self.tokenize)(file_path) for file_path in file_paths)

		token_collection = dict()
	
		for token_dict in tokenizer_results:
			token_collection.update(token_dict)

		print(f'Finished tokenization. Total genomes tokenized: {len(token_collection)}')

		return token_collection

	def tokenize(self, file_path):
		# load df, loop over sequences
		sequence_dict = self.fetch_sequences(file_path)

		tokens = dict()

		for genome_id, sequences in sequence_dict.items():
			tokens.update(self.tokenize_genome(genome_id, sequences))
		
		return tokens

	def list_files(self):
		dir_list = os.listdir(self.input_path)
	
		dir_list = [f'{self.input_path}/{file}' for file in dir_list if self.file_type == file.split(".")[-1]]
		
		print(f'Found {len(dir_list)} files with type {self.file_type} in {self.input_path}')
		assert len(dir_list) > 0, f'No files with type {self.file_type} found in {self.input_path}'
		
		return dir_list

	def fetch_sequences(self, file_path):
		df = read_sequence_file(file_path=file_path, file_type = self.file_type)
		sequence_dict = {genome_id : dna_sequences.split(" ") for genome_id, dna_sequences in zip(df[self.genome_col], df[self.dna_sequence_col])}

		return sequence_dict
		

	def tokenize_genome(self, genome_id, sequences):
		# Tokenizes one genome
	
		kmer_prefix_size = len(self.kmer_prefix)

		#assert self.kmer_suffix_size % 3 == 0, "For this mode, kmer_suffix_size needs to be divisble by the codon size (of 3)"
		
		kmer_offset = self.kmer_offset # Change this to change the offset 
						
		
		forward_kmers = list()
		if self.reverse_complement:
			reverse_kmers = list()
		
		for sequence in sequences:
			sequence = sequence.replace("\n", "")
			table = str.maketrans("atcgmrykvhdbxnswMRYKVHDBXNSWF","ATCGnnnnnnnnnnnnnnnnnnnnnnnnn")
			sequence = sequence.translate(table)		
			
			# Finds tokens and store them in list
			forward_kmers.extend(self.tokenize_single_sequence(sequence, self.kmer_prefix, self.kmer_suffix_size, kmer_prefix_size, kmer_offset))

			if self.reverse_complement:
				sequence = Seq(sequence)
				reverse_sequence = str(sequence.reverse_complement())

				reverse_kmers.extend(self.tokenize_single_sequence(reverse_sequence, self.kmer_prefix, self.kmer_suffix_size, kmer_prefix_size, kmer_offset))

		if self.reverse_complement:
			return {genome_id : {f"forward" : forward_kmers, f"reverse" : reverse_kmers}}
			
		else:
			return {genome_id: {f"forward" : forward_kmers}}
			


	def tokenize_single_sequence(self, sequence, kmer_prefix, kmer_suffix_size, kmer_prefix_size, kmer_offset, with_unknown_bases = False):
		kmers = list()
		current_kmer_prefix_location = sequence.find(kmer_prefix)

		while current_kmer_prefix_location != -1:

			kmer_suffix_start_location = current_kmer_prefix_location + kmer_prefix_size + kmer_offset
			
			kmer_suffix = sequence[kmer_suffix_start_location : kmer_suffix_start_location + kmer_suffix_size]
			
			
			if len(kmer_suffix) == kmer_suffix_size:
				if 'n' not in kmer_suffix or with_unknown_bases:
				# Converts dna to binary to use for indexing np.array
				
					kmers.append(kmer_suffix)

			current_kmer_prefix_location = sequence.find(kmer_prefix, current_kmer_prefix_location + kmer_prefix_size)
		
		return kmers
