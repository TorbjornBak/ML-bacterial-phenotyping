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
			  esmc_model = "esmc_300m",
			  device = "mps",
			  pooling = "mean",
			  hidden_state = None,
			  slice = None
			  ):
		self.token_collection = token_collection # Dict with key being an id, value being a list of seq tokens
		self.kmer_suffix_size = kmer_suffix_size
		self.esmc_model = esmc_model
		self.pooling = pooling
		self.hidden_state = hidden_state
		self.slice = slice

		self.client = ESMC.from_pretrained(self.esmc_model).to(device)

		# Selects between returning hidden states or embeddings
		if self.hidden_state is not None:
			self.logits_config = LogitsConfig(sequence=True, return_hidden_states=True)
		else:
			self.logits_config = LogitsConfig(sequence=True, return_embeddings=True)

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
	
	def embed_tokens(self, id, token_dict, pooling = "mean"):

		# See https://github.com/facebookresearch/esm/blob/main/esm/tokenization.py#L22
		# or bacformer repo for embedding  details
		
		if pooling == "mean":
			# Join kmers to sequence first, then embed and mean pool
			embeddings = {id : 
						{
						strand:self.embed_sequence("".join(kmers)).embeddings.mean(axis=1)  # Mean pooling across sequence
						for strand, kmers in token_dict.items()
						}
					}
			


		elif pooling == "mean_per_token":
			
			# Join kmers to sequence first, then embed and mean pool per token
			# embeddings = {id : 
			# 			{
			# 			strand:[self.embed_kmer("".join(kmers[i:i+group])).mean(axis = 1) for i in range(0, len(kmers), group)]  # Mean pooling for each token 
			# 			for strand, kmers in token_dict.items()
			# 			}
			# 		}
			
			# embeddings = {id : 
			# 			{
			# 			strand:self.embed_kmer("".join(kmers))  # Mean pooling across sequence
			# 			for strand, kmers in token_dict.items()
			# 			}
			# 		}
			embeddings = dict()
			protein_mer_size = self.kmer_suffix_size//3
			for strand, kmers in token_dict.items():
				sequence = "".join(kmers)
				embedding_output = self.embed_sequence(sequence)
				
				
				#print(f'{len(kmers)//3=}')
				sequence_embedding = embedding_output.embeddings
				token_embeddings = []
				print(f'{sequence_embedding.shape=}')
				

				# if sequence_embedding.shape[1] <= 2:

					
				for i in range(1, len(kmers)):
					token_embedding = sequence_embedding[0][i:i+protein_mer_size].mean(axis=0)
					
					token_embeddings.append(token_embedding)

				if len(token_embeddings) != len(kmers):
					print(f'{token_embeddings=}, {len(kmers)=}, {sequence_embedding.shape=}, {sequence=}')
				
				embeddings[id] = {strand: torch.vstack(token_embeddings)}

			
		elif pooling == "slice":
			# for strand, kmers in token_dict.items():
			# 	embeddings = []
			# 	for kmer in kmers:
				
			# 		embeddings.append(self.embed_kmer(kmer)[0][self.slice])
			# 	print(embeddings)
			embeddings = {id : 
							{
							strand:torch.vstack([self.embed_sequence(kmer)[0][self.slice] for kmer in kmers])
							for strand, kmers in token_dict.items()
							}
						}
			
		elif pooling is None:
			# Iterate over tokens, translate to protein sequence, embed using ESM-c
			embeddings = {id : 
						{
						strand:[self.embed_sequence(kmer) for kmer in kmers]
					  	for strand, kmers in token_dict.items()
						}
					}
		return embeddings


	def embed_sequence(self, sequence):
		# Translate kmer to protein sequence
		# Embed using ESM-c
		if len(sequence) <= self.kmer_suffix_size*2:
			raise ValueError(f"Sequence too short for embedding.{sequence=}")
		dna_seq = Seq(sequence)
		protein_seq = dna_seq.translate(table=11)
		protein_str = ESMProtein(sequence = str(protein_seq))
		assert len(protein_seq) == len(protein_str.sequence), "Length mismatch between translated protein and ESMProtein sequence"
		
		protein_tensor = self.client.encode(protein_str)
		logits_output = self.client.logits(
   		protein_tensor, self.logits_config
		)
		assert protein_tensor.sequence.shape[0] == logits_output.embeddings.shape[1], "Token length mismatch between input and output embeddings"
		assert logits_output.embeddings.shape[1] == len(protein_seq) + 2, "Output embedding length mismatch"
		if self.hidden_state is not None:
			return logits_output.hidden_states[self.hidden_state]
		else:
			return logits_output

		

