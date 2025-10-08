import mmap
import sys, os
import numpy as np
import pandas as pd
import random
from joblib import Parallel, delayed

# Reads one fasta file

# One-hot encoding.

def read_fasta_binary(file_path):

	sequences = list()

	with open(file_path, 'r+b') as f:
		
		mm = mmap.mmap(f.fileno(),0, access=mmap.ACCESS_READ)
	
		header_position = 0

		# Find header
		header_position = mm.find(b">")
		
		while header_position > -1:
			
			# moves the pointer to one position after the header start position
			mm.seek(header_position + 1)
			
			# finds end of the header
			header_end = mm.find(b"\n")
			# sequence starts at next position after header ends.
			seq_start = header_end + 1
			

			#header = mm.read(header_end - header_position)
		
			
			
			# Find next header
			header_position = mm.find(b">")
			seq_end = header_position - 1

			current_sequence = mm.read(seq_end - seq_start)
			sequences.append(current_sequence)

	mm.close()		

	return sequences



def kmerize_sequences_prefix_filtering(sequences, kmer_prefix, kmer_suffix_size, array_size):
	# Binary array (0 or 1)
	array = np.zeros(array_size, dtype = np.bool)
	
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

				array[kmer_suffix_binary] = True

			current_kmer_prefix_location = sequence.find(kmer_prefix, current_kmer_prefix_location + kmer_prefix_size)
	#print(f'Total kmers: {kmer_count}')
	return array


def kmerize_sequences_prefix_filtering_count(sequences, kmer_prefix, kmer_suffix_size, array_size):
	# np array with counts of each kmer

	array = np.zeros(array_size, dtype = np.uint16)
	
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
				
				array[kmer_suffix_binary] += 1

			current_kmer_prefix_location = sequence.find(kmer_prefix, current_kmer_prefix_location + kmer_prefix_size)
	#print(f'Total kmers: {kmer_count}')
	return array

	
def kmerize_sequences_prefix_filtering_return_all(sequences, kmer_prefix, kmer_suffix_size):
	# The same as above, but should return a sequence by finding and concatenating all the kmers instead of using one-hot-encoded array. 
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
			
			if 'n' not in kmer_suffix:
				# Converts dna to binary to use for indexing np.array
				kmers.append(kmer_suffix)

			current_kmer_prefix_location = sequence.find(kmer_prefix, current_kmer_prefix_location + kmer_prefix_size)
	#print(f'Total kmers: {kmer_count}')
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

# def test_kmer_sampler(iterations = 1000, file_path = "data/test/511145.fna"):
# 	kmer_prefix = b"CGTGATGT"
# 	kmer_suffix_size = 8
# 	vector = list()
# 	for iteration in range(iterations):
# 		print(f'Iteration nr: {iteration}')
# 		sequences = read_fasta_binary(file_path=file_path)
# 		array_size = get_array_size(alphabet_size = 4, kmer_size = kmer_suffix_size)
# 		array = kmerize_sequences_prefix_filtering(sequences, kmer_prefix, kmer_suffix_size, array_size)

# 		#print(f'Unique kmers: {array.sum()}')
# 		vector.append(array)


def kmer_sampling_multiple_files(directory, genome_ids = None, file_names = None, kmer_prefix = b"CGTGAT", kmer_suffix_size = 8, labels = None, file_suffix = ".fna", sample_nr = None):
	
	arrays = list()
	y_labels = list()
	array_size = get_array_size(alphabet_size = 4, kmer_size = kmer_suffix_size)
	
	if genome_ids is not None:
		file_names = [f'{genome_id}{file_suffix}' for genome_id in genome_ids]


	random.shuffle(file_names)

	print("Sampling kmers from files")
	for i, file_name in enumerate(file_names):
		if i == sample_nr:
			print(f"Stopped after {i} iterations")
			return arrays, y_labels
		file_path = f'{directory}/{file_name}'
		sequences = read_fasta_binary(file_path=file_path)
		
		array = kmerize_sequences_prefix_filtering(sequences, kmer_prefix, kmer_suffix_size, array_size)

		#print(f'Unique kmers: {array.sum()}')
		arrays.append(array)

		genome_id = file_name.strip(file_suffix)

		y_labels.append(labels[genome_id])

	
	return arrays, y_labels
	

def load_labels(file_path = "downloads/genome_lineage", id = "genome_id", label = "class", sep = "\t"):

	df = pd.read_csv(file_path, sep = sep)


	df = df.dropna(subset = [id, label])

	label_dict = dict(zip(df[id].apply(str), df[label])) 

	unique_labels = np.unique(df[label])
	
	label2int = {label: i for i, label in enumerate(unique_labels)}

	label_dict_int = {id : label2int[label] for id, label in label_dict.items()}

	return label_dict, label_dict_int


def find_files_to_kmerize(directory, file_suffix = ".fna", id = "genome_id", label = "class"):

	dir_list = os.listdir(directory)



	dir_list = [file for file in dir_list if file_suffix in file]


	labels = load_labels(id = id, label = label)

	
	dir_list = [dir for dir in dir_list if dir.replace(file_suffix, "") in labels]


	return dir_list, labels
	

def save_kmerized_files_with_numpy(X, X_file_path, y, y_file_path):
	X = np.stack(X, axis=0)

	np.save(file = X_file_path, arr = X, allow_pickle=True)
	np.save(file = y_file_path, arr=y, allow_pickle=True)
	
	print("Successfully saved X and y to .npy files")

def read_parquet(parguet_path):
	# Read in as parquet one at a time, kmerize, convert to npy (or parquet?) Then we can stream and potentially use much bigger datasets?
	print(f"Loading parquet dataset: {parguet_path}")
	df = pd.read_parquet(parguet_path, engine = "fastparquet")
	return df
	
def kmerize_and_embed_parquet_dataset_return_all(path, genome_col, dna_sequence_col, kmer_prefix = "CGTGAT", kmer_suffix_size = 8):

	print(f"Kmerizing {path}")
	
	df = read_parquet(parguet_path=path)
	
	kmer_embeddings = dict()

	if get_array_size(alphabet_size=4, kmer_size = kmer_suffix_size) < 2**16:
		data_type = np.uint16
	else:
		data_type = np.uint32

	for genome_id, dna_sequences in zip(df[genome_col], df[dna_sequence_col]):
		
		dna_sequences = dna_sequences.split(" ")
		kmers = kmerize_sequences_prefix_filtering_return_all(dna_sequences, kmer_prefix, kmer_suffix_size)

		embeddings = [kmer_to_integer(kmer) for kmer in kmers]

		embeddings_np = np.array(embeddings, dtype = data_type)

		kmer_embeddings[genome_id] = embeddings_np

		#print(f'{genome_id} : {len(kmers)}')

	
	return kmer_embeddings



def kmerize_and_embed_parquet_dataset_count(path, genome_col, dna_sequence_col, kmer_prefix = "CGTGAT", kmer_suffix_size = 8):

	print(f"Kmerizing {path}")
	

	df = read_parquet(parguet_path=path)
	
	kmer_counts = dict()

	for genome_id, dna_sequences in zip(df[genome_col], df[dna_sequence_col]):
		
		dna_sequences = dna_sequences.split(" ")
		array = kmerize_sequences_prefix_filtering_count(dna_sequences, kmer_prefix, kmer_suffix_size, array_size=get_array_size(4, kmer_suffix_size))

		kmer_counts[genome_id] = array
		

		#print(f'{genome_id} : {len(kmers)}')

	
	return kmer_counts



def kmerize_and_embed_parquet_dataset_binary(path, genome_col, dna_sequence_col, kmer_prefix = "CGTGAT", kmer_suffix_size = 8):

	print(f"Kmerizing {path}")
	

	df = read_parquet(parguet_path=path)
	
	kmer_counts = dict()

	for genome_id, dna_sequences in zip(df[genome_col], df[dna_sequence_col]):
		
		dna_sequences = dna_sequences.split(" ")
		array = kmerize_sequences_prefix_filtering(dna_sequences, kmer_prefix, kmer_suffix_size, array_size=get_array_size(4, kmer_suffix_size))

		kmer_counts[genome_id] = array
		

		#print(f'{genome_id} : {len(kmers)}')

	
	return kmer_counts

def kmerize_parquet_joblib(file_paths, kmer_prefix, kmer_suffix_size, nr_of_cores = 4):

	joblib_results = Parallel(n_jobs = nr_of_cores)(delayed(kmerize_and_embed_parquet_dataset_return_all)(path, "genome_name", "dna_sequence", kmer_prefix, kmer_suffix_size) for path in file_paths)

	print(f'Processed {len(joblib_results)}/{len(file_paths)} files.')

	data_dict = dict()
	
	for kmer_dict in joblib_results:
		data_dict.update(kmer_dict)

	print(f'Nr of sequences in dataset: {len(data_dict.keys())}')
	
	return data_dict

def kmerize_parquet_count_joblib(file_paths, kmer_prefix, kmer_suffix_size, nr_of_cores = 4):

	joblib_results = Parallel(n_jobs = nr_of_cores)(delayed(kmerize_and_embed_parquet_dataset_count)(path, "genome_name", "dna_sequence", kmer_prefix, kmer_suffix_size) for path in file_paths)

	print(f'Processed {len(joblib_results)}/{len(file_paths)} files.')

	data_dict = dict()
	
	for kmer_counts in joblib_results:
		data_dict.update(kmer_counts)

	print(f'Nr of sequences in dataset: {len(data_dict.keys())}')
	
	return data_dict

if __name__ == "__main__":
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
	# file_names, labels = find_files_to_kmerize(directory="/home/projects2/s203555/bv-brc-data", file_suffix = ".fna")
	# #labels = load_labels(file_path="downloads/genome_lineage")
	
	# X, y = kmer_sampling_multiple_files(directory="/home/projects2/s203555/bv-brc-data", file_names=file_names, labels = labels, kmer_prefix = b"CGTGAT", kmer_suffix_size = 8)
	
	# save_kmerized_files_with_numpy(X = X, X_file_path="/home/projects2/s203555/bv-numpy-arrays/X_array.npy", y = y, y_file_path="/home/projects2/s203555/bv-numpy-arrays/y_array.npy")
	
	# print(f'{len(file_names)=}') 
	# print(len(y))
	# print(f'{y=}')

	label_dict = load_labels(file_path="downloads/labels.csv", id = "genome_name", label = "madin_categorical_gram_stain", sep = ",")

	data_dict = dict()

	for path in ["downloads/train_01.parquet","downloads/train_02.parquet", "downloads/train_03.parquet"]:

		parquet_df = read_parquet(parguet_path=path)


		kmerized_sequences = kmerize_and_embed_parquet_dataset(df = parquet_df, genome_column= "genome_name", dna_sequence_column= "dna_sequence", kmer_prefix="CGTCAT", kmer_suffix_size=8)
	
		data_dict.update(kmerized_sequences)
	

	
	# See each base in the kmer as a pixel? Embedding?
