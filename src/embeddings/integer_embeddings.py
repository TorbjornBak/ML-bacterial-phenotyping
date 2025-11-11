from joblib import Parallel, delayed
import numpy as np

class IntegerEmbeddings():
	
	def __init__(self, 
			  token_collection,
			  kmer_suffix_size,
			  compress_embeddings):
		self.token_collection = token_collection # Dict with key being an id, value being a list of seq tokens
		self.kmer_suffix_size = kmer_suffix_size
		self.compress_embeddings = compress_embeddings

	def __str__(self):
		return "IntegerEmbeddings"

	def run_embedder(self, nr_of_cores = 2):
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
		return embeddings, vocab_size
		

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
	
	# TODO: Fix this so it works with the new way the tokens are packaged
	def compress_integer_embeddings(self, integer_embeddings, alphabet_size = 4):
		# Function for compressing integer embedding vocab space, 
		# Input: dict of the embeddings
		print(f'Compressing embeddings')
		print(f'Initial vocabulary size: {alphabet_size**self.kmer_suffix_size + 1}')
		
		arr1 = np.zeros(alphabet_size**self.kmer_suffix_size + 1, dtype = np.uint64)
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