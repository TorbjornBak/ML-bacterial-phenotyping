import mmap
import sys
import numpy as np
# Reads one fasta file

# One-hot encoding.
# Loop over file, once kmer has been found, don't look for it again.

# Load to numpy array, size should be calculated based on total possible nr
# of k-mers for a given kmer size



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
			

			header = mm.read(header_end - header_position)
		
			
			
			# Find next header
			header_position = mm.find(b">")
			seq_end = header_position - 1

			current_sequence = mm.read(seq_end - seq_start)
			sequences.append(current_sequence)




	mm.close()		

	return sequences



def kmerize_sequences_prefix_filtering(sequences, kmer_prefix, kmer_suffix_size, array_size):

	array = np.zeros(array_size, dtype = int)
	
	kmer_prefix_size = len(kmer_prefix)
	found_count = 0

	for sequence in sequences:
		sequence = sequence.replace(b"\n", b"")
		table = bytes.maketrans(b"atcgmrykvhdbxnswMRYKVHDBXNSW",b"ATCGnnnnnnnnnnnnnnnnnnnnnnnn")
		sequence = sequence.translate(table)
		
		
		# fill out numpy 
		current_kmer_prefix_location = sequence.find(kmer_prefix)

		while current_kmer_prefix_location != -1:

			found_count += 1
			kmer_suffix_start_location = current_kmer_prefix_location + kmer_prefix_size
			
			kmer_suffix = sequence[kmer_suffix_start_location : kmer_suffix_start_location + kmer_suffix_size]
			
			if b'n' not in kmer_suffix:
				kmer_suffix_binary = dna_to_binary(kmer_suffix,	kmer_suffix_size)

				#print(f"{kmer_suffix=}")


				array[kmer_suffix_binary] = 1

			current_kmer_prefix_location = sequence.find(kmer_prefix, current_kmer_prefix_location + kmer_prefix_size)
	print(f'Total kmers: {found_count}')
	return array

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
	return alphabet_size ** kmer_size



def test_kmer_sampler(iterations = 1000, filepath = "data/test/511145.fna"):
	kmer_prefix = b"CGTGATGT"
	kmer_suffix_size = 8
	vector = list()
	for iteration in range(iterations):
		print(f'Iteration nr: {iteration}')
		sequences = read_fasta_binary(file_path="data/test/511145.fna")
		array_size = get_array_size(alphabet_size = 4, kmer_size = kmer_suffix_size)
		array = kmerize_sequences_prefix_filtering(sequences, kmer_prefix, kmer_suffix_size, array_size)

		print(f'Unique kmers: {array.sum()}')
		vector.append(array)


def kmer_sampling_multiple_files(directory, genome_ids):
	kmer_prefix = b"CGTGATGT"
	kmer_suffix_size = 8
	vector = list()
	
	for _ in range(100):

		for id in genome_ids:

			file = f'{directory}/{id}.fna'
			sequences = read_fasta_binary(file_path=file)
			array_size = get_array_size(alphabet_size = 4, kmer_size = kmer_suffix_size)
			array = kmerize_sequences_prefix_filtering(sequences, kmer_prefix, kmer_suffix_size, array_size)

			print(f'Unique kmers: {array.sum()}')
			vector.append(array)


if __name__ == "__main__":
	# kmer_prefix = b"CGTGA"
	# kmer_suffix_size = 8
	# sequences = read_fasta_binary(file_path="data/test/511145.fna")
	# array_size = get_array_size(alphabet_size = 4, kmer_size = kmer_suffix_size)
	# array = kmerize_sequences_prefix_filtering(sequences, kmer_prefix, kmer_suffix_size, array_size)
	# print(f'Unique kmers: {array.sum()}')
	#test_kmer_sampler()
	kmer_sampling_multiple_files(directory = "data", genome_ids=[469009.4,
 1309411.5,
 1123738.3,
 551115.6,
 1856298.3,
 1706000.3,
 28901.2925,
 28901.2926,
 28901.2927,
 28901.2928])
