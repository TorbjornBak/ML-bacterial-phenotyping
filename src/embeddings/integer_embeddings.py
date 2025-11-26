from joblib import Parallel, delayed
import numpy as np
import os

os.environ.setdefault("JOBLIB_TEMP_FOLDER", "/tmp")

class IntegerEmbeddings():
	
	def __init__(self,
			  kmer_prefix,
			  kmer_suffix_size,
			  kmer_offset = 0,
			  data_directory = ".",
			  ):
		
		self.kmer_prefix = kmer_prefix
		self.kmer_suffix_size = kmer_suffix_size
		self.kmer_offset = kmer_offset
		self.data_directory = data_directory
		self.embedding_class = "integer"

		self.channel_size = 4**self.kmer_suffix_size + 1


	def run_embedder(self, token_collection, nr_of_cores = 1):
		
		print(f'Running integer embedder with {nr_of_cores} cores')
		if nr_of_cores == 1:
			embedding_results = [self.embed_tokens(id, tokens) for id, tokens in token_collection.items()]
		else:
			embedding_results = Parallel(n_jobs = nr_of_cores)(delayed(self.embed_tokens)(id, tokens) for id, tokens in token_collection.items())

		embeddings = dict()

		for embedding in embedding_results:
			embeddings.update(embedding)
		
		self.embeddings = embeddings
		
		print(f'Integer embedder finished, channel size: {self.channel_size}')
		return embeddings
	
	def save_embeddings(self, X, ids, groups):
		print(f"Saving embeddings to: {self.file_path}.npz")
		np.savez_compressed(f'{self.file_path}.npz', 
					  		X=np.array(X, dtype=object), 
					  		ids=np.array(ids, dtype=object), 
							groups=np.array(groups, dtype=object),
							channel_size = self.channel_size)
		
		return True

	def load_stored_embeddings(self):
		file_path = self.file_path
		print(f"Loading embeddings from: {file_path=}.npz")
		z = np.load(f'{file_path}.npz', allow_pickle=True)

		X = list(z["X"])  # object array → list of arrays 
		ids = list(z["ids"])  # map labels from current dict
		groups = list(z["groups"])

		channel_size = int(z["channel_size"]) if "channel_size" in z else None
		
		return X, ids, groups, channel_size
		

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
		if self.kmer_offset == 0:
			dataset_name = f'integer_embedding_prefix_{self.kmer_prefix}_suffixsize_{self.kmer_suffix_size}' 
		else:
			dataset_name = f'integer_embedding_prefix_{self.kmer_prefix}_suffixsize_{self.kmer_suffix_size}_offset_{self.kmer_offset}'
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
				kmer_prefix,
			  	kmer_suffix_size,
			  	kmer_offset = 0,
			  	data_directory = "."
			  ):
		
		self.kmer_prefix = kmer_prefix
		self.kmer_suffix_size = kmer_suffix_size
		self.kmer_offset = kmer_offset
		self.embedding_class = "onehot"

	
		self.data_directory = data_directory
			

	def run_embedder(self, token_collection):
		
		print(f'Running one-hot embedder')
		embedding_results = [self.one_hot_embedding(id, tokens) for id, tokens in token_collection.items()]
		
		embeddings = dict()

		for embedding in embedding_results:
			embeddings.update(embedding)

		self.embeddings = embeddings

		
		
		return embeddings
	
	def save_embeddings(self, X, ids, groups):
		print(f"Saving embeddings to: {self.file_path}.npz")
		np.savez_compressed(f'{self.file_path}.npz', 
					  		X=np.array(X, dtype=object), 
					  		ids=np.array(ids, dtype=object), 
							groups=np.array(groups, dtype=object),
							channel_size = self.channel_size)
		
		return True

	def load_stored_embeddings(self):
		file_path = self.file_path
		print(f"Loading embeddings from: {file_path=}.npz")
		z = np.load(f'{file_path}.npz', allow_pickle=True)

		X = list(z["X"])  # object array → list of arrays 
		ids = list(z["ids"])  # map labels from current dict
		groups = list(z["groups"])

		channel_size = int(z["channel_size"]) if "channel_size" in z else None
		
		return X, ids, groups, channel_size
		

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
		if self.kmer_offset == 0:
			dataset_name = f'onehot_embedding_prefix_{self.kmer_prefix}_suffixsize_{self.kmer_suffix_size}' 
		else:
			dataset_name = f'onehot_embedding_prefix_{self.kmer_prefix}_suffixsize_{self.kmer_suffix_size}_offset_{self.kmer_offset}'
		
		file_path = f'{self.data_directory.rstrip("/")}/{dataset_name}'

		return file_path
	

	def one_hot_embedding(self, id, token_dict):
		# {genome_id : {"forward" : forward_tokens, "reverse" : reverse_tokens}}
		
		integer_embeddings = {id : {strand:
							np.stack([self.kmer_to_one_hot(kmer) for kmer in kmers], axis=0) 
							for strand, kmers in token_dict.items()}}

		return integer_embeddings

	
	def kmer_to_one_hot(self, kmer):
		m = {'A':0, 'C':1, 'G':2, 'T':3, 'n':4}
		  # A, C, G, T, N
		one_hot = np.zeros((len(kmer) * len(m)), dtype=np.float32)
		for i, ch in enumerate(kmer):
			one_hot[i*len(m)+m[ch]] = 1.0

		self.channel_size = len(m) * len(kmer)
		return one_hot
	