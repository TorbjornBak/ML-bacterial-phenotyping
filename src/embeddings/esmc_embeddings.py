from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from esm.tokenization import EsmSequenceTokenizer

from joblib import Parallel, delayed
from Bio.Seq import Seq
from tqdm import tqdm
import torch
import os
os.environ.setdefault("JOBLIB_TEMP_FOLDER", "/tmp")

class ESMcEmbeddings():


	def __init__(self, 
			  token_collection,
			  kmer_suffix_size,
			  compress_embeddings,
			  esmc_model = "esmc_300m",
			  device = "mps",
			  pooling = "mean"):
		self.token_collection = token_collection # Dict with key being an id, value being a list of seq tokens
		self.kmer_suffix_size = kmer_suffix_size
		self.compress_embeddings = compress_embeddings
		self.esmc_model = esmc_model
		self.pooling = pooling
		
		self.client = ESMC.from_pretrained(self.esmc_model).to(device)

	def run_embedder(self, nr_of_cores = 1):
		print(f'Embedding with ESM-c model: {self.esmc_model} using pooling: {self.pooling}')
		if nr_of_cores > 1:
			embedding_results = Parallel(n_jobs = nr_of_cores)(delayed(self.embed_tokens)(id, tokens, self.pooling) for id, tokens in tqdm(self.token_collection.items()))
		else:
			embedding_results = [self.embed_tokens(id, tokens, self.pooling) for id, tokens in tqdm(self.token_collection.items())]

		embeddings = dict()

		for embedding in embedding_results:
			embeddings.update(embedding)
		
		self.embeddings = embeddings
		print(f'Completed embedding of {len(embeddings)} sequences using ESM-c model: {self.esmc_model}')
		
		return embeddings
	
	def embed_tokens(self, id, token_dict, pooling = "mean", join_kmers = True):

		# See https://github.com/facebookresearch/esm/blob/main/esm/tokenization.py#L22
		# or bacformer repo for embedding  details
		
		if pooling == "mean":
			if join_kmers:
			# Join kmers to sequence first, then embed and mean pool
				embeddings = {id : 
							{
							strand:self.embed_kmer("".join(kmers)).mean(axis=1)  # Mean pooling across sequence
							for strand, kmers in token_dict.items()
							}
						}
			else:
				# Embed each kmer individually, then stack and mean pool (slower and would give different result)
				embeddings = {id : 
							{
							strand:torch.hstack([self.embed_kmer(kmer) for kmer in kmers]).mean(axis=1)  # Mean pooling across sequence
							for strand, kmers in token_dict.items()
							}
						}

		elif pooling == "mean_per_token":
			embeddings = {id : 
						{
						strand:[self.embed_kmer(kmer).mean(axis = 1) for kmer in kmers]  # Mean pooling for each token 
					  	for strand, kmers in token_dict.items()
						}
					}
		elif pooling is None:
			# Iterate over tokens, translate to protein sequence, embed using ESM-c
			embeddings = {id : 
						{
						strand:[self.embed_kmer(kmer) for kmer in kmers]
					  	for strand, kmers in token_dict.items()
						}
					}
		return embeddings


	def embed_kmer(self, kmer):
		# Translate kmer to protein sequence
		# Embed using ESM-c
		dna_seq = Seq(kmer)
		protein_seq = dna_seq.translate(table=11, to_stop=True)
		protein_str = ESMProtein(sequence = str(protein_seq))
		
		protein_tensor = self.client.encode(protein_str)
		logits_output = self.client.logits(
   		protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
		)
		return logits_output.embeddings

