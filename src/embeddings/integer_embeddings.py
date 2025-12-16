import numpy as np
import os
from itertools import product

#os.environ.setdefault("JOBLIB_TEMP_FOLDER", "/tmp")

class IntegerEmbeddings():
	
	def __init__(self,
			  kmer_prefix,
			  kmer_suffix_size,
			  kmer_offset = 0,
			  data_directory = ".",
			  grouped = False
			  ):
		
		self.kmer_prefix = kmer_prefix
		self.kmer_suffix_size = kmer_suffix_size
		self.kmer_offset = kmer_offset
		self.data_directory = data_directory
		self.embedding_class = "integer"
		self.grouped = grouped

		self.channel_size = 4**self.kmer_suffix_size + 1


	def run_embedder(self, token_collection):
		
		
		embedding_results = [self.embed_tokens(id, tokens) 
					   for id, tokens in token_collection.items()]
		
		embeddings = dict()

		for embedding in embedding_results:
			embeddings.update(embedding)
		
		self.embeddings = embeddings
		
		print(f'Integer embedder finished, channel size: {self.channel_size}')
		return embeddings
	
	def save_embeddings(self, X, strand_ids, groups, genome_ids):
		print(f"Saving embeddings to: {self.file_path}.npz")
		np.savez_compressed(f'{self.file_path}.npz', 
					  		X=np.array(X, dtype=object), 
					  		strand_ids=np.array(strand_ids, dtype=object), 
							groups=np.array(groups, dtype=object),
							genome_ids=np.array(genome_ids, dtype=object),
							channel_size = self.channel_size)
		
		return True

	def load_stored_embeddings(self):
		file_path = self.file_path
		print(f"Loading embeddings from: {file_path=}.npz")
		z = np.load(f'{file_path}.npz', allow_pickle=True)

		X = list(z["X"])  # object array → list of arrays 
		strand_ids = list(z["strand_ids"])  # map labels from current dict
		groups = list(z["groups"])
		genome_ids = list(z["genome_ids"])

		channel_size = int(z["channel_size"]) if "channel_size" in z else None
		
		return X, strand_ids, groups, genome_ids, channel_size
		

	def is_embedding_file(self):
		file_types = [".npz"]
		for type in file_types:
			if not os.path.isfile(f'{self.file_path}{type}'):
				return False
		return True


	@property
	def file_path(self):

		if not hasattr(self, '_file_path'):
			self._file_path = self.build_file_path()
		return self._file_path

	def build_file_path(self):
		dataset_name = f'{self.embedding_class}_embedding_prefix_{self.kmer_prefix}_suffixsize_{self.kmer_suffix_size}'
		dataset_name += "_grouped" if self.grouped else "" 
		dataset_name += f'_offset_{self.kmer_offset}' if self.kmer_offset != 0 else ''
		file_path = f'{self.data_directory.rstrip("/")}/{dataset_name}'

		return file_path
	

	def embed_tokens(self, id, token_dict):
		# {genome_id : {"forward" : forward_tokens, "reverse" : reverse_tokens}}
		#{self.kmer_to_integer(kmer) for strand, token in tokens for kmer in token}
		
		
		integer_embeddings = {id : {strand:[self.kmer_to_integer(kmer) for kmer in kmers] for strand, kmers in token_dict.items()}}
		#integer_embeddings = [self.kmer_to_integer(kmer) for kmer in tokens]

		return integer_embeddings
			
	
	def kmer_to_integer(self, kmer):
		#print(f'{kmer=}')
		m = {'A':0, 'C':1, 'G':2, 'T':3}
		# n = len(alphabet)
		# assert (n & (n-1) == 0 and n != 0), f"Len of alphabet should be a power of two"
		# m = {char : i for i, char in enumerate(alphabet)}

		# Similar to dna_to_binary, but for strings instead of binary
		
		x = 0
		for ch in kmer:
			x = (x << 2) | m[ch]  # multiply by 4 and add digit
		return x + 1 # x + 1 because padding has id : 0

	def int_to_kmer(self, x: int, k: int) -> str:
		inv = 'ACGT'
		out = []
		x = x - 1
		for _ in range(k):
			out.append(inv[x & 3])  # x % 4
			x >>= 2                 # x //= 4
		return ''.join(reversed(out))
	


class OneHotEmbeddings():
	
	def __init__(self, 
				kmer_prefix: int,
			  	kmer_suffix_size: int,
			  	kmer_offset: int = 0,
			  	data_directory: str = ".",
				grouped: bool = False,
				xhot: int | None = None,
			  ):
		
		self.kmer_prefix = kmer_prefix
		self.kmer_suffix_size = kmer_suffix_size
		self.kmer_offset = kmer_offset
		if xhot is None:
			self.embedding_class = "onehot"
		else:
			self.embedding_class = "xhot"
		self.grouped = grouped
		self.xhot = xhot
	
		self.data_directory = data_directory
		

	def run_embedder(self, token_collection):
		
		if self.xhot is not None:
			print(f'Running {self.xhot}-hot embedder')
			embedding_results = [self.x_hot_embedding(id, tokens, x = self.xhot) for id, tokens in token_collection.items()]
		else:	
			print(f'Running one-hot embedder')
			embedding_results = [self.one_hot_embedding(id, tokens) for id, tokens in token_collection.items()]
		
		embeddings = dict()

		for embedding in embedding_results:
			embeddings.update(embedding)

		self.embeddings = embeddings

		
		return embeddings
	
	def save_embeddings(self, X, strand_ids, groups, genome_ids):
		print(f"Saving embeddings to: {self.file_path}.npz")
		np.savez_compressed(f'{self.file_path}.npz', 
					  		X=np.array(X, dtype=object), 
					  		strand_ids=np.array(strand_ids, dtype=object), 
							groups=np.array(groups, dtype=object),
							genome_ids=np.array(genome_ids, dtype=object),
							channel_size = self.channel_size)
		
		return True

	def load_stored_embeddings(self):
		file_path = self.file_path
		print(f"Loading embeddings from: {file_path=}.npz")
		z = np.load(f'{file_path}.npz', allow_pickle=True)

		X = list(z["X"])  # object array → list of arrays 
		strand_ids = list(z["strand_ids"])  # map labels from current dict
		groups = list(z["groups"])
		genome_ids = list(z["genome_ids"])

		channel_size = int(z["channel_size"]) if "channel_size" in z else None
		
		return X, strand_ids, groups, genome_ids, channel_size

	def is_embedding_file(self):
		file_types = [".npz"]
		for type in file_types:
			if not os.path.isfile(f'{self.file_path}{type}'):
				return False
		return True


	@property
	def file_path(self):
		if not hasattr(self, '_file_path'):
			self._file_path = self.build_file_path()
		return self._file_path
		
	
	def build_file_path(self):
		dataset_name = f'{self.embedding_class}_embedding_prefix_{self.kmer_prefix}_suffixsize_{self.kmer_suffix_size}'
		dataset_name += "_grouped" if self.grouped else "" 
		dataset_name += f'_offset_{self.kmer_offset}' if self.kmer_offset != 0 else ''
		dataset_name += f'_xhot_{self.xhot}' if self.xhot is not None else ''
		
		file_path = f'{self.data_directory.rstrip("/")}/{dataset_name}'

		return file_path
	

	def one_hot_embedding(self, id, token_dict):
		# {genome_id : {"forward" : forward_tokens, "reverse" : reverse_tokens}}
		
		embeddings = {id : {strand:
							np.stack([self.kmer_to_k_hot(kmer) for kmer in kmers], axis=0) 
							for strand, kmers in token_dict.items()}}

		return embeddings
	
	def x_combinations(self, alphabet = "ACGTN", x = 2) -> dict[str, int]:
		if not hasattr(self, '_x_combinations'):
			self._x_combinations = {"".join(t) : i for i, t in enumerate(list(product(alphabet, repeat = x)))} # eg. {"AAA" : 0, "AAC" : 1, "AAT" : 2...}
		self.combinations = self._x_combinations
		return self.combinations
	
	def x_hot_embedding(self, id, token_dict, x):

		encoding_size = len(self.x_combinations(alphabet = "ACGTN", x = x))

		embeddings = {id : {strand:
							np.stack([self.kmer_to_x_base_hot(kmer[i:i+x], encoding_size) 
				 			for kmer in kmers 
				 				for i in range(0, len(kmer), x)], axis=0) 
							for strand, kmers in token_dict.items()}
							}
		self.channel_size = encoding_size
		return embeddings
			
	
	def kmer_to_k_hot(self, kmer):
		m = {'A':0, 'C':1, 'G':2, 'T':3, 'n':4}
		  # A, C, G, T, N
		one_hot = np.zeros((len(kmer) * len(m)), dtype=np.float32)
		for i, ch in enumerate(kmer):
			one_hot[i*len(m)+m[ch]] = 1.0

		self.channel_size = len(m) * len(kmer)
		assert len(one_hot) == len(m) * len(kmer), "Length of one hot should be the length of m * length of kmer"
		return one_hot
	
	def kmer_to_x_base_hot(self, token, encoding_size):
		# Encode the token as onehotvectors encoding x bases as one vector
		# A generalization of kmer_to_k_hot, sligthly different as we put each encoding next to eachother. 
		
		encoding_vector = np.zeros(encoding_size, dtype=np.float32)

		encoding_vector[self.combinations[token]] = 1.0
		
		return encoding_vector




		
		

	

	

class KmerCountsEmbeddings():
	
	def __init__(self, 
				kmer_prefix,
			  	kmer_suffix_size,
			  	kmer_offset = 0,
			  	data_directory = ".",
				embedding_class = "frequency",
				grouped = False
			  ):
		
		print(f'Initializing KmerCountsEmbeddings with {embedding_class} embedding.')

		self.kmer_prefix = kmer_prefix
		self.kmer_suffix_size = kmer_suffix_size
		self.kmer_offset = kmer_offset
		self.grouped = grouped

		self.embedding_class = embedding_class
		if self.embedding_class == "frequency":
			self.normalize = True
		else:
			self.normalize = False
	
		self.data_directory = data_directory

		self.channel_size = 4**self.kmer_suffix_size

	def save_embeddings(self, X, strand_ids, groups, genome_ids):
		print(f"Saving embeddings to: {self.file_path}.npz")
		np.savez_compressed(f'{self.file_path}.npz', 
					  		X=np.array(X, dtype=object), 
					  		strand_ids=np.array(strand_ids, dtype=object), 
							groups=np.array(groups, dtype=object),
							genome_ids=np.array(genome_ids, dtype=object),
							channel_size = self.channel_size)
		
		return True

	def load_stored_embeddings(self):
		file_path = self.file_path
		print(f"Loading embeddings from: {file_path=}.npz")
		z = np.load(f'{file_path}.npz', allow_pickle=True)

		X = list(z["X"])  # object array → list of arrays 
		strand_ids = list(z["strand_ids"])  # map labels from current dict
		groups = list(z["groups"])
		genome_ids = list(z["genome_ids"])

		channel_size = int(z["channel_size"]) if "channel_size" in z else None
		
		return X, strand_ids, groups, genome_ids, channel_size
		

	def is_embedding_file(self):
		file_types = [".npz"]
		for type in file_types:
			if not os.path.isfile(f'{self.file_path}{type}'):
				return False
		return True


	@property
	def file_path(self):
		if not hasattr(self, '_file_path'):
			self._file_path = self.build_file_path()
		return self._file_path
		
	
	def build_file_path(self):
		dataset_name = f'{self.embedding_class}_embedding_prefix_{self.kmer_prefix}_suffixsize_{self.kmer_suffix_size}'
		dataset_name += "_grouped" if self.grouped else "" 
		dataset_name += f'_offset_{self.kmer_offset}' if self.kmer_offset != 0 else ''
		
		file_path = f'{self.data_directory.rstrip("/")}/{dataset_name}'

		return file_path
			

	def run_embedder(self, token_collection: dict):
		
		print(f'Running counts embedder')
		embedding_results = [self.counts_embedding(id, tokens) for id, tokens in token_collection.items()]
		
		embeddings = dict()

		for embedding in embedding_results:
			embeddings.update(embedding)

		self.embeddings = embeddings

		return embeddings
	
	def counts_embedding(self, id: str, token_dict: dict):
		# {genome_id : {"forward" : forward_tokens, "reverse" : reverse_tokens}}
		
		counts = {id : {strand:
							self.kmer_frequency_count(kmers) 
							for strand, kmers in token_dict.items()}}

		return counts
	
	def kmer_frequency_count(self, kmers: list[str]) -> dict[str, float]:
		counts = np.zeros(self.channel_size, dtype=np.float32)
		for kmer in kmers:
				if "n" not in kmer:
					counts[self.kmer_to_integer(kmer)] += 1
		
		if self.normalize:
			counts = counts / np.sum(counts)  # Normalize to frequencies
		
		return counts


	def kmer_to_integer(self, kmer: str) -> int:
		m = {'A':0, 'C':1, 'G':2, 'T':3}
		x = 0
		for ch in kmer:
			x = (x << 2) | m[ch]  # multiply by 4 and add digit
		return x

	def integer_to_kmer(self, x: int, k: int) -> str:
		inv = 'ACGT'
		out = []
		for _ in range(k):
			out.append(inv[x & 3])  # x % 4
			x >>= 2                 # x //= 4
		return ''.join(reversed(out))