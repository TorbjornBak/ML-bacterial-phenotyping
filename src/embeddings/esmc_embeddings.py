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
			  device = "cpu",
			  pooling = "mean_per_token"):
		self.token_collection = token_collection # Dict with key being an id, value being a list of seq tokens
		self.kmer_suffix_size = kmer_suffix_size
		self.compress_embeddings = compress_embeddings
		self.esmc_model = esmc_model
		self.pooling = pooling
		
		self.client = ESMC.from_pretrained(self.esmc_model).to(device)

	def run_embedder(self, nr_of_cores = 2):
		embedding_results = Parallel(n_jobs = nr_of_cores)(delayed(self.embed_tokens)(id, tokens, self.pooling) for id, tokens in tqdm(self.token_collection.items()))
		
		embeddings = dict()

		for embedding in embedding_results:
			embeddings.update(embedding)
		
		self.embeddings = embeddings
		
		return embeddings, None
	
	def embed_tokens(self, id, token_dict, pooling = "mean_per_token"):
		
		if pooling == "mean":
			embeddings = {id : 
						{
						strand:torch.hstack([self.embed_kmer(kmer) for kmer in kmers]).mean(axis=1)  # Mean across embeddings
					  	for strand, kmers in token_dict.items()
						}
					}
		elif pooling == "mean_per_token":
			embeddings = {id : 
						{
						strand:[self.embed_kmer(kmer) for kmer in kmers]  # Mean across embeddings
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

