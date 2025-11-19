from joblib import Parallel, delayed
import numpy as np
import os

os.environ.setdefault("JOBLIB_TEMP_FOLDER", "/tmp")

class IntegerEmbeddings():
	
	def __init__(self, 
			  token_collection,
			  kmer_suffix_size,
			  compress_embeddings):
		self.token_collection = token_collection # Dict with key being an id, value being a list of seq tokens
		self.kmer_suffix_size = kmer_suffix_size
		self.compress_embeddings = compress_embeddings


	def run_embedder(self, nr_of_cores = 2):
		
		print(f'Running integer embedder with {nr_of_cores} cores')
		if nr_of_cores == 1:
			embedding_results = [self.embed_tokens(id, tokens) for id, tokens in self.token_collection.items()]
		else:
			embedding_results = Parallel(n_jobs = nr_of_cores)(delayed(self.embed_tokens)(id, tokens) for id, tokens in self.token_collection.items())

		embeddings = dict()

		for embedding in embedding_results:
			embeddings.update(embedding)

		if self.compress_embeddings:
			embeddings, vocab_size = self.compress_integer_embeddings(embeddings, alphabet_size=4)
			
		else:
			vocab_size = 4**self.kmer_suffix_size + 1
		
		
		self.embeddings = embeddings
		self.vocab_size = vocab_size 
		print(f'Integer embedder done, vocab size: {vocab_size}')
		return embeddings
		

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
			  token_collection):
		self.token_collection = token_collection # Dict with key being an id, value being a list of seq tokens


	def run_embedder(self):
		print(f'Running one-hot embedder')
		embedding_results = [self.one_hot_embedding(id, tokens) for id, tokens in self.token_collection.items()]
		
		embeddings = dict()

		for embedding in embedding_results:
			embeddings.update(embedding)

		self.embeddings = embeddings

		print(f'One-hot embedder done, vocab size: {self.vocab_size}')
		
		return embeddings
	
	
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

		self.vocab_size = len(m) * len(kmer)
		return one_hot
	