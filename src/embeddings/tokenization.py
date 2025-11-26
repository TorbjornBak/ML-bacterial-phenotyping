import mmap
import sys, os
import numpy as np
import pandas as pd
import random
from joblib import Parallel, delayed
from itertools import product
from utilities import dna_bpe as bpe
from Bio.Seq import Seq
from Bio import SeqIO

os.environ.setdefault("JOBLIB_TEMP_FOLDER", "/tmp")

def kmerize_sequences_prefix_filtering_binary(sequences, kmer_prefix, kmer_suffix_size, array_size, array_type = "np.zeros"):
	# Binary array (0 or 1)

	if array_type == "np.zeros":
		array = np.zeros(array_size, dtype = np.bool)
	elif array_type == "bytearray":
		array = bytearray(array_size)
	
	kmer_prefix_size = len(kmer_prefix)
	kmer_count = 0

	for sequence in sequences:
		
		sequence = sequence.replace(b"\n", b"")
		table = bytes.maketrans(b"atcgmrykvhdbxnswMRYKVHDBXNSW",b"ATCGnnnnnnnnnnnnnnnnnnnnnnnn")
		
		sequence = sequence.translate(table)
		
		
		# Finds kmers and one-hot-encode them
		current_kmer_prefix_location = sequence.find(kmer_prefix)

		while current_kmer_prefix_location != -1:

			kmer_count += 1
			kmer_suffix_start_location = current_kmer_prefix_location + kmer_prefix_size
			
			kmer_suffix = sequence[kmer_suffix_start_location : kmer_suffix_start_location + kmer_suffix_size]
			
			if b'n' not in kmer_suffix:
				# Converts dna to binary to use for indexing np.array
				kmer_suffix_binary = dna_to_binary(kmer_suffix,	kmer_suffix_size)

				array[kmer_suffix_binary] = 1

			current_kmer_prefix_location = sequence.find(kmer_prefix, current_kmer_prefix_location + kmer_prefix_size)
	#print(f'Total kmers: {kmer_count}')
	return array

def kmerize_sequences_prefix_filtering_binary_str(sequences, kmer_prefix, kmer_suffix_size, array_size, array_type = "np.zeros"):
	# Binary array (0 or 1)

	if array_type == "np.zeros":
		array = np.zeros(array_size, dtype = np.bool)
	elif array_type == "bytearray":
		array = bytearray(array_size)
	
	kmer_prefix_size = len(kmer_prefix)
	kmer_count = 0

	for sequence in sequences:
		
		sequence = sequence.replace("\n", "")
		table = str.maketrans("atcgmrykvhdbxnswMRYKVHDBXNSW","ATCGnnnnnnnnnnnnnnnnnnnnnnnn")
		
		sequence = sequence.translate(table)
		
		
		# Finds kmers and one-hot-encode them
		current_kmer_prefix_location = sequence.find(kmer_prefix)

		while current_kmer_prefix_location != -1:

			kmer_count += 1
			kmer_suffix_start_location = current_kmer_prefix_location + kmer_prefix_size
			
			kmer_suffix = sequence[kmer_suffix_start_location : kmer_suffix_start_location + kmer_suffix_size]
			
			if 'n' not in kmer_suffix:
				# Converts dna to binary to use for indexing np.array
				kmer_suffix_binary = dna_to_binary_str(kmer_suffix,	kmer_suffix_size)

				array[kmer_suffix_binary] = 1

			current_kmer_prefix_location = sequence.find(kmer_prefix, current_kmer_prefix_location + kmer_prefix_size)
	#print(f'Total kmers: {kmer_count}')
	return array


def kmerize_sequences_prefix_filtering_count(sequences, kmer_prefix, kmer_suffix_size, array_size, normalize = True):
	# np array with counts of each kmer

	counts = np.zeros(array_size)
	
	kmer_prefix_size = len(kmer_prefix)
	kmer_count = 0

	for sequence in sequences:
		sequence = sequence.replace("\n", "")
		table = str.maketrans("atcgmrykvhdbxnswMRYKVHDBXNSW","ATCGnnnnnnnnnnnnnnnnnnnnnnnn")
		sequence = sequence.translate(table)
		
		# Finds kmers and one-hot-encode them
		current_kmer_prefix_location = sequence.find(kmer_prefix)

		while current_kmer_prefix_location != -1:

			kmer_count += 1
			kmer_suffix_start_location = current_kmer_prefix_location + kmer_prefix_size
			
			kmer_suffix = sequence[kmer_suffix_start_location : kmer_suffix_start_location + kmer_suffix_size]
			
			if 'n' not in kmer_suffix:
				# Converts dna to binary to use for indexing np.array
				kmer_suffix_binary = dna_to_binary_str(kmer_suffix,	kmer_suffix_size)
				
				counts[kmer_suffix_binary] += 1

			current_kmer_prefix_location = sequence.find(kmer_prefix, current_kmer_prefix_location + kmer_prefix_size)
	
	if normalize:
		counts /= kmer_count
			
	return counts

	
def kmerize_sequences_prefix_filtering_return_all(sequences, kmer_prefix, kmer_suffix_size):
	# The same as above, but returns a sequence by finding and creating a list of all the kmers instead of using one-hot-encoded array. 
	kmers = list()
	
	kmer_prefix_size = len(kmer_prefix)
	
	kmer_offset = 0 # Change this to change the offset
	kmer_count = 0
	
	for sequence in sequences:
		sequence = sequence.replace("\n", "")
		table = str.maketrans("atcgmrykvhdbxnswMRYKVHDBXNSW","ATCGnnnnnnnnnnnnnnnnnnnnnnnn")
		sequence = sequence.translate(table)
		
		
		# Finds kmers and one-hot-encode them
		current_kmer_prefix_location = sequence.find(kmer_prefix)

		while current_kmer_prefix_location != -1:

			kmer_count += 1
			kmer_suffix_start_location = current_kmer_prefix_location + kmer_prefix_size + kmer_offset
			
			kmer_suffix = sequence[kmer_suffix_start_location : kmer_suffix_start_location + kmer_suffix_size]
			
			if 'n' not in kmer_suffix and len(kmer_suffix) == kmer_suffix_size:
				# Converts dna to binary to use for indexing np.array
				kmers.append(kmer_suffix)

			current_kmer_prefix_location = sequence.find(kmer_prefix, current_kmer_prefix_location + kmer_prefix_size)
	#print(f'Total kmers: {kmer_count}')
	return kmers

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

		#print(f'{dir_list=}')
		
		return dir_list

	def fetch_sequences(self, file_path):
		df = read_sequence_file(file_path=file_path, file_type = self.file_type)
		sequence_dict = {genome_id : dna_sequences.split(" ") for genome_id, dna_sequences in zip(df[self.genome_col], df[self.dna_sequence_col])}

		return sequence_dict
		

	def tokenize_genome(self, genome_id, sequences,):
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

	



def dna_to_binary_str(dna, kmer_size):
	
	t = 0b11
	t <<= (kmer_size*2 - 2)
	c = 0b01
	c <<= (kmer_size*2 - 2)
	g = 0b10
	g <<= (kmer_size*2 - 2)

	number = 0

	for char in dna:
		number >>= 2
		# A
		if char == "A":
			pass
		# T
		elif char == "T":
			number |= t
		# C
		elif char == "C":
			number |= c
		# G
		elif char == "G":
			number |= g
		
		else:
			print("Illegal base in DNA sequence:", char)
			sys.exit(1)
		
	#print(bin_to_dna(number, 10))
	return number

def dna_to_binary(dna, kmer_size):
	
	t = 0b11
	t <<= (kmer_size*2 - 2)
	c = 0b01
	c <<= (kmer_size*2 - 2)
	g = 0b10
	g <<= (kmer_size*2 - 2)

	number = 0

	for char in dna:
		number >>= 2
		# A
		if char == 65:
			pass
		# T
		elif char == 84:
			number |= t
		# C
		elif char == 67:
			number |= c
		# G
		elif char == 71:
			number |= g
		
		else:
			print("Illegal base in DNA sequence:", char)
			sys.exit(1)
		
	#print(bin_to_dna(number, 10))
	return number


def bin_to_dna(number, kmer_size):
	# Converting bits to individual numbers
	twobits = [(number >> bit) & 0b11 for bit in range(0, kmer_size*2, 2)]
	
	kmer = b""
	
	#and every 2nd bit together with 0b11
	
	for twobit in twobits:
		
		if twobit == 0b00:
			kmer += b"A"
		elif twobit == 0b11:
			kmer += b"T"
		elif twobit == 0b01:
			kmer += b"C"
		else:
			kmer += b"G"
	
	return kmer

def bin_to_dna_str(number, kmer_size):
	# Converting bits to individual numbers
	twobits = [(number >> bit) & 0b11 for bit in range(0, kmer_size*2, 2)]
	
	kmer = ""
	
	#and every 2nd bit together with 0b11
	
	for twobit in twobits:
		
		if twobit == 0b00:
			kmer += "A"
		elif twobit == 0b11:
			kmer += "T"
		elif twobit == 0b01:
			kmer += "C"
		else:
			kmer += "G"
	
	return kmer



def get_array_size(alphabet_size, kmer_size):
	# Load to numpy array, size is calculated based on total possible nr
	# of k-mers for a given kmer size
	return alphabet_size ** kmer_size


	# Similar to dna_to_binary, but for strings instead of binary
def kmer_to_integer(kmer):
	m = {'A':0, 'C':1, 'G':2, 'T':3}
	x = 0
	for ch in kmer:
		x = (x << 2) | m[ch]  # multiply by 4 and add digit
	return x + 1

def int_to_kmer(x: int, k: int) -> str:
	inv = 'ACGT'
	out = []
	x = x - 1
	for _ in range(k):
		out.append(inv[x & 3])  # x % 4
		x >>= 2                 # x //= 4
	return ''.join(reversed(out))


	
def check_id_and_labels_exist(file_path, id, labels : list, sep = "\t"):
	df = pd.read_csv(file_path, sep = sep)
	
	assert id in df, f'{id=} was not found in the columns of the supplied file \n{df.columns=}'

	for label in labels:
		assert label in df, f'{label=} was not found in the columns of the supplied file.\n{df.columns=}'
	
	return



def load_labels(file_path, id = "genome_id", label = "class", sep = "\t", freq_others = None):

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


	

def save_kmerized_files_with_numpy(X, X_file_path, y, y_file_path):
	X = np.stack(X, axis=0)

	np.save(file = X_file_path, arr = X, allow_pickle=True)
	np.save(file = y_file_path, arr=y, allow_pickle=True)
	
	print("Successfully saved X and y to .npy files")

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

	
def kmerize_and_embed_dataset_return_integers(path, genome_col, dna_sequence_col, kmer_prefix = "CGTGAT", kmer_suffix_size = 8, file_type="parquet"):
	# Returns a dict with lists of kmers represented as integers.
	print(f"Kmerizing {path}")
	
	df = read_sequence_file(file_path=path, file_type = file_type)
	
	integer_embeddings = dict()

	if get_array_size(alphabet_size=4, kmer_size = kmer_suffix_size) < 2**16:
		data_type = np.uint16
	else:
		data_type = np.uint32

	for genome_id, dna_sequences in zip(df[genome_col], df[dna_sequence_col]):
		
		dna_sequences = dna_sequences.split(" ")
		kmers = kmerize_sequences_prefix_filtering_return_all(dna_sequences, kmer_prefix, kmer_suffix_size)

		embeddings = [kmer_to_integer(kmer) for kmer in kmers]

		embeddings_np = np.array(embeddings, dtype = data_type)

		integer_embeddings[genome_id] = embeddings_np

	
	return integer_embeddings


def kmerize_and_embed_dataset_return_kmer_str(path, genome_col, dna_sequence_col, kmer_prefix = "CGTGAT", kmer_suffix_size = 8, file_type="parquet"):

	print(f"Kmerizing {path}")
	
	df = read_sequence_file(file_path=path, file_type = file_type)
	
	kmer_embeddings = dict()

	for genome_id, dna_sequences in zip(df[genome_col], df[dna_sequence_col]):
		
		dna_sequences = dna_sequences.split(" ")
		kmers = kmerize_sequences_prefix_filtering_return_all(dna_sequences, kmer_prefix, kmer_suffix_size)

		kmer_embeddings[genome_id] = kmers
	
	return kmer_embeddings



def kmerize_and_embed_dataset_return_tokens(path, 
													genome_col, 
													dna_sequence_col, 
													kmer_prefix = "CGTGAT", 
													kmer_suffix_size = 8, 
													alphabet = "ACTG", 
													token_size = 2,
													file_type = "parquet",
													):

	print(f"Kmerizing {path}")

	assert kmer_suffix_size % token_size == 0, "kmer_suffix size must be divisible by chosen token_size"

	df = read_sequence_file(file_path=path, file_type=file_type)
	
	kmer_embeddings = dict()

	token_list = tokens_given_length(alphabet=alphabet, token_size=token_size)

	for genome_id, dna_sequences in zip(df[genome_col], df[dna_sequence_col]):
		
		dna_sequences = dna_sequences.split(" ")
		kmers = kmerize_sequences_prefix_filtering_return_all(dna_sequences, kmer_prefix, kmer_suffix_size)

		embeddings = [kmer_one_hot(kmer, token_list, token_size) for kmer in kmers]

		
		one_hot_embeddings = np.concatenate(embeddings, axis = 1)
		#print(one_hot_embeddings[0])
		# if results_path is None:
		kmer_embeddings[genome_id] = one_hot_embeddings
		# else:
		# 	dataset_file_path = f'{results_path}/tokens_id_{genome_id}_prefix_{kmer_prefix}_suffix_{kmer_suffix_size}_{token_size}.npz'
		# 	np.savez_compressed(dataset_file_path, one_hot_embeddings = one_hot_embeddings)

		#print(one_hot_embeddings)
		#print(f'{genome_id} : {len(kmers)}')
	
	return kmer_embeddings


def kmer_one_hot(kmer, token_list, token_size = 1):
	dna_tokens = [kmer[slice: slice + token_size] for slice in range(0, len(kmer), token_size)]
	
	# rows, columns
	# rows: nr of total tokens
	# columns:  (len(kmer) / token_size)
	
	matrix = np.zeros((int(len(kmer)/token_size), len(token_list)), dtype = np.bool)

	for row_idx, token in enumerate(dna_tokens):
		
		encoding = get_token_encoding(token_list, token)
		matrix[row_idx, encoding] = True
		#print(f'{encoding=}, {row=}, {matrix[encoding, row]=}')
	return matrix


def get_token_encoding(token_list, token):
	# TODO: Update to log(1)?? 
	return token_list.index(token)


def tokens_given_length(alphabet = "ACTG", token_size = 2):
	return ["".join(t) for t in product(alphabet, repeat=token_size)]


def byte_pair_encoding():
	# https://platform.openai.com/tokenizer
	# https://github.com/openai/tiktoken?tab=readme-ov-file

	pass

# def train_dna_tokenizer(corpus, ):
# 	corpus = [sequence for sequence in read_sequence_file()]
# 	bpe.train_tokenizer(corpus)

def kmerize_and_embed_dataset_count(path, genome_col, dna_sequence_col, kmer_prefix = "CGTGAT", kmer_suffix_size = 8, file_type = "parquet", normalize = True):

	print(f"Kmerizing {path}")
	

	df = read_sequence_file(file_path=path, file_type=file_type)
	kmer_counts = dict()

	for genome_id, dna_sequences in zip(df[genome_col], df[dna_sequence_col]):
		dna_sequences = dna_sequences.split(" ")
		array = kmerize_sequences_prefix_filtering_count(dna_sequences, kmer_prefix, kmer_suffix_size, array_size=get_array_size(4, kmer_suffix_size), normalize=normalize)

		kmer_counts[genome_id] = array

	return kmer_counts



def kmerize_and_embed_dataset_bytearray(path, genome_col, dna_sequence_col, kmer_prefix = "CGTGAT", kmer_suffix_size = 8, file_type="parquet"):

	print(f"Kmerizing {path}")
	

	df = read_sequence_file(file_path=path, file_type=file_type)
	
	kmer_arrays = dict()

	for genome_id, dna_sequences in zip(df[genome_col], df[dna_sequence_col]):
		
		dna_sequences = dna_sequences.split(" ")
		array = kmerize_sequences_prefix_filtering_binary_str(dna_sequences, kmer_prefix, kmer_suffix_size, array_size=get_array_size(4, kmer_suffix_size), array_type="np.zeros")

		kmer_arrays[genome_id] = array
		

		#print(f'{genome_id} : {len(kmers)}')

	
	return kmer_arrays

########################################
########### Deprecated function, still works, use kmerize_joblib instead ####
# def kmerize_parquet_joblib(file_paths, kmer_prefix, kmer_suffix_size, nr_of_cores = 4, output_type = "kmers"):

# 	if output_type == "kmers":
# 		joblib_results = Parallel(n_jobs = nr_of_cores)(delayed(kmerize_and_embed_dataset_return_integers)(path, "genome_name", "dna_sequence", kmer_prefix, kmer_suffix_size, file_type = "parquet") for path in file_paths)
	
# 	elif output_type == "str":
# 		joblib_results = Parallel(n_jobs = nr_of_cores)(delayed(kmerize_and_embed_dataset_return_kmer_str)(path, "genome_name", "dna_sequence", kmer_prefix, kmer_suffix_size, file_type = "parquet") for path in file_paths)

# 	elif output_type == "one-hot":
# 		joblib_results = Parallel(n_jobs = nr_of_cores)(delayed(kmerize_and_embed_dataset_return_tokens)(path, "genome_name", "dna_sequence", kmer_prefix, kmer_suffix_size, file_type = "parquet") for path in file_paths)

# 	elif output_type == "bytearray":
# 			joblib_results = Parallel(n_jobs = nr_of_cores)(delayed(kmerize_and_embed_dataset_bytearray)(path, "genome_name", "dna_sequence", kmer_prefix, kmer_suffix_size, file_type = "parquet") for path in file_paths)

# 	elif output_type == "counts":
# 		joblib_results = Parallel(n_jobs = nr_of_cores)(delayed(kmerize_and_embed_dataset_count)(path, "genome_name", "dna_sequence", kmer_prefix, kmer_suffix_size, file_type = "parquet") for path in file_paths)
	
	
# 	print(f'Processed {len(joblib_results)}/{len(file_paths)} files.')

# 	data_dict = dict()
	
# 	for kmer_dict in joblib_results:
# 		data_dict.update(kmer_dict)

# 	print(f'Nr of sequences in dataset: {len(data_dict.keys())}')
	
# 	return data_dict

def kmerize_joblib(file_paths, kmer_prefix, kmer_suffix_size, id_column = "genome_name", sequence_column = "dna_sequence", nr_of_cores = 4, output_type = "kmers", file_type = "parquet", normalize = True, vocab_size = 500, token_size = 4):
	if output_type == "kmers":
		joblib_results = Parallel(n_jobs = nr_of_cores)(delayed(kmerize_and_embed_dataset_return_integers)(path, id_column, sequence_column, kmer_prefix, kmer_suffix_size, file_type = file_type) for path in file_paths)
	
	elif output_type == "bpe":
		assert vocab_size is not None
		assert token_size is not None
		print(f'Using BPE encoding')
		joblib_results = Parallel(n_jobs = nr_of_cores)(delayed(kmerize_and_embed_dataset_return_kmer_str)(path, id_column, sequence_column, kmer_prefix, kmer_suffix_size, file_type = file_type) for path in file_paths)
		
		print(f'Processed {len(joblib_results)}/{len(file_paths)} files.')
		data_dict = dict()
		for kmer_dict in joblib_results:
			data_dict.update(kmer_dict)

		corpus = ["".join(kmers) for kmers in data_dict.values()]

		print(f'Training bpe tokenizer...')
		tokenizer, _ = bpe.train_tokenizer(corpus=corpus, vocab_size=vocab_size, k = token_size)
		vocab_list, vocab_dict, pad_id, unk_id = bpe.build_vocab_from_merges(corpus, tokenizer, k=token_size)

		print(f'{vocab_list=}')

		tokenizer_results = Parallel(n_jobs = nr_of_cores)(delayed(bpe.tokenize)(id, sequence, tokenizer, vocab_dict, k = token_size) for id, sequence in data_dict.items())
		
		tokenized_sequence_dict = dict()
		for result_dict in tokenizer_results:
			tokenized_sequence_dict.update(result_dict)

		print(f'Nr of sequences in dataset: {len(tokenized_sequence_dict.keys())}')
		print(f'{len(vocab_list)=}')

		return {"joblib_result": tokenized_sequence_dict, 
		  		"vocab_size" : len(vocab_list),
				"pad_id" : pad_id,
				"unk_id" : unk_id}

	elif output_type == "str":
		joblib_results = Parallel(n_jobs = nr_of_cores)(delayed(kmerize_and_embed_dataset_return_kmer_str)(path, id_column, sequence_column, kmer_prefix, kmer_suffix_size, file_type = file_type) for path in file_paths)

	elif output_type == "one-hot":
		joblib_results = Parallel(n_jobs = nr_of_cores)(delayed(kmerize_and_embed_dataset_return_tokens)(path, id_column, sequence_column, kmer_prefix, kmer_suffix_size, file_type = file_type) for path in file_paths)

	elif output_type == "bytearray":
			joblib_results = Parallel(n_jobs = nr_of_cores)(delayed(kmerize_and_embed_dataset_bytearray)(path, id_column, sequence_column, kmer_prefix, kmer_suffix_size, file_type = file_type) for path in file_paths)

	elif output_type == "counts":
		joblib_results = Parallel(n_jobs = nr_of_cores)(delayed(kmerize_and_embed_dataset_count)(path, id_column, sequence_column, kmer_prefix, kmer_suffix_size, file_type = file_type, normalize=normalize) for path in file_paths)

	
	print(f'Processed {len(joblib_results)}/{len(file_paths)} files.')

	data_dict = dict()
	
	for kmer_dict in joblib_results:
		data_dict.update(kmer_dict)

	print(f'Nr of sequences in dataset: {len(data_dict.keys())}')
	
	return {"joblib_result" : data_dict}

def compress_integer_embeddings(integer_embeddings, alphabet_size, kmer_suffix_size):
	# Function for compressing integer embedding vocab space, 
	# Input: dict of the embeddings
	print(f'Compressing embeddings')
	print(f'Initial vocabulary size: {alphabet_size**kmer_suffix_size + 1}')
	
	arr1 = np.zeros(alphabet_size**kmer_suffix_size + 1, dtype = np.uint64)
	# Detect which embeddings are used, which aren't
	for embeddings_np in integer_embeddings:
		unique_embeddings = np.unique(embeddings_np)
		for embedding in unique_embeddings:
			arr1[embedding] = embedding
	
	arr2 = [np.uint64(0)]
	arr2.extend([a for a in arr1[1:] if a != 0]) # Remove zeros from original array
	vocab_size = len(arr2)

	if len(arr1) > len(arr2):
		mapping = {old_embedding:new_embedding for new_embedding, old_embedding in enumerate(arr2)} # Create array for mapping old embeddings to new embeddings
		
		for embeddings_np in integer_embeddings:
			X_compressed  = [np.array([mapping[emb] for emb in embeddings_np]) for embeddings_np in integer_embeddings]
		
		print(f'Compressed vocabulary size: {len(arr2)}')
		
		return X_compressed, vocab_size
	else:
		print(f'Cannot compress vocabulary size')
		vocab_size = len(arr1)

		return integer_embeddings, vocab_size


def parquet_to_fasta(parquet_file_paths, genome_col, dna_sequence_col, kmer_prefix = None, kmer_suffix_size = None, output_path = None, nr_of_cores = 2):
	for file in parquet_file_paths:
		if kmer_prefix is None and kmer_suffix_size is None:
			df = read_sequence_file(file_path=file, file_type="parquet")
			for genome_id, dna_sequences in zip(df[genome_col], df[dna_sequence_col]):
				dna_sequences = dna_sequences.split(" ")
				write_fasta(genome_id, dna_sequences, output_path=output_path)
	else:
		kmerized_seqs_dict = kmerize_joblib(
						file_paths=parquet_file_paths, 
						kmer_prefix=kmer_prefix, 
						kmer_suffix_size=kmer_suffix_size, 
						nr_of_cores=nr_of_cores, 
						output_type="str")
		
		for genome_id, kmer_sequence in kmerized_seqs_dict.items():
			kmer_sequence = "".join(kmer_sequence)
			write_fasta(genome_id, kmer_sequence, output_path=output_path)
	
	return

def write_fasta(id, sequences, output_path, metadata = None):
	# Handles multiple sequences
	file_name = f'{id}_{metadata}' if metadata is not None else f'{id}'

	file_path = f'{output_path.strip("/")}/{file_name}.fna'
	line_width = 80
	with open(file_path, mode = "w") as fasta_file:
		for j, sequence in enumerate([sequences]):
			sequence_identifier = f'>{id}_sect_{j}\n'
			fasta_file.write(sequence_identifier)
			sequence_formatted = "\n".join([sequence[i:i+line_width] for i in range(0,len(sequence), line_width)])
			fasta_file.write(sequence_formatted)
			fasta_file.write("\n")


if __name__ == "__main__":


	parquet_to_fasta(parquet_file_paths=["downloads/train_01.parquet","downloads/train_02.parquet", "downloads/train_03.parquet"], 
				  genome_col= "genome_name", dna_sequence_col= "dna_sequence",
				  kmer_prefix="CGTC",
				  kmer_suffix_size=4,
				  output_path="results/fasta_files")

	# kmerize_parquet_joblib(file_paths=["downloads/train_01.parquet","downloads/train_02.parquet", "downloads/train_03.parquet"],
	# 					kmer_prefix="CGTGA",
	# 					kmer_suffix_size = 8,
	# 					nr_of_cores = 1,fclass
	# 					output_type = "one-hot"
	# 					)

	# kmer_prefix = b"CGTGA"
	# kmer_suffix_size = 8
	# sequences = read_fasta_binary(file_path="data/test/511145.fna")
	# array_size = get_array_size(alphabet_size = 4, kmer_size = kmer_suffix_size)
	# array = kmerize_sequences_prefix_filtering(sequences, kmer_prefix, kmer_suffix_size, array_size)
	# print(f'Unique kmers: {array.sum()}')
	# test_kmer_sampler()
	# kmer_sampling_multiple_files(directory = "data", 
	# 						  genome_ids=[
	# 							  		469009.4,
	# 									1309411.5,
	# 									1123738.3,
	# 									551115.6,
	# 									1856298.3,
	# 									1706000.3,
	# 									28901.2925,
	# 									28901.2926,
	# 									28901.2927,
	# 									28901.2928])
	# #labels = load_labels(file_path="downloads/genome_lineage")
	
	# X, y = kmer_sampling_multiple_files(directory="/home/projects2/s203555/bv-brc-data", file_names=file_names, labels = labels, kmer_prefix = b"CGTGAT", kmer_suffix_size = 8)
	
	# save_kmerized_files_with_numpy(X = X, X_file_path="/home/projects2/s203555/bv-numpy-arrays/X_array.npy", y = y, y_file_path="/home/projects2/s203555/bv-numpy-arrays/y_array.npy")
	
	# print(f'{len(file_names)=}') 
	# print(len(y))
	# print(f'{y=}')

	# label_dict = load_labels(file_path="downloads/labels.csv", id = "genome_name", label = "madin_categorical_gram_stain", sep = ",")

	# data_dict = dict()

	# for path in ["downloads/train_01.parquet","downloads/train_02.parquet", "downloads/train_03.parquet"]:

	# 	parquet_df = read_parquet(parquet_path=path)


	# 	kmerized_sequences = kmerize_and_embed_parquet_dataset(df = parquet_df, genome_column= "genome_name", dna_sequence_column= "dna_sequence", kmer_prefix="CGTCAT", kmer_suffix_size=8)
	
	# 	data_dict.update(kmerized_sequences)
	

	
	# See each base in the kmer as a pixel? Embedding?
	