from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from joblib import Parallel, delayed
from Bio.Seq import Seq
from tqdm import tqdm
import torch
import os
import numpy as np
os.environ.setdefault("JOBLIB_TEMP_FOLDER", "/tmp")

class ESMcEmbeddings():


	def __init__(self,
			  kmer_prefix,
			  kmer_suffix_size,
			  esmc_model = "esmc_300m",
			  device = "mps",
			  pooling = "mean",
			  hidden_state = None,
			  slice = None,
			  data_directory = ".",
			  kmer_offset = 0,
			  ):
		 # Dict with key being an id, value being a list of seq tokens
		self.kmer_prefix = kmer_prefix
		self.kmer_suffix_size = kmer_suffix_size
		self.esmc_model = esmc_model
		self.pooling = pooling
		self.hidden_state = hidden_state
		self.slice = slice
		self.device = device
		self.kmer_offset = kmer_offset

		self.embedding_class = "esmc"

		self.data_directory = data_directory


		self.client = ESMC.from_pretrained(self.esmc_model).to(device)

		# Selects between returning hidden states or embeddings
		if self.hidden_state is not None:
			self.logits_config = LogitsConfig(sequence=True, return_hidden_states=True)
		else:
			self.logits_config = LogitsConfig(sequence=True, return_embeddings=True)
		

	def run_embedder(self, token_collection, nr_of_cores = 1):

		print(f'Embedding with ESM-c model: {self.esmc_model} using pooling: {self.pooling}')
		if nr_of_cores > 1:
			embedding_results = Parallel(n_jobs = nr_of_cores)(delayed(self.embed_tokens)(id, tokens, self.pooling) for id, tokens in tqdm(token_collection.items()))
		else:
			embedding_results = [self.embed_tokens(id, tokens, self.pooling) for id, tokens in tqdm(token_collection.items())]

		embeddings = dict()

		for embedding in embedding_results:
			embeddings.update(embedding)
		
		self.embeddings = embeddings
		print(f'Completed embedding of {len(embeddings)} sequences using ESM-c model: {self.esmc_model}')
		
		return embeddings
	
	def save_embeddings(self, X, ids, groups):
		
		file_path = self.file_path()
		print(f"Saving embeddings (X) to: {file_path}.pt \nand metadata (ids and groups) to: {file_path}.npz")
		torch.save(X, f"{file_path}.pt")
		np.savez_compressed(f'{file_path}.npz', 
							ids=np.array(ids, dtype=object), 
							groups=np.array(groups, dtype=object)
							)
		return True
		
	
	def load_stored_embeddings(self):
		print(f"Loading embeddings from: {self.file_path()=}")
		z = np.load(f'{self.file_path()}.npz', allow_pickle=True)

		ids = list(z["ids"])  # map labels from current dict
		groups = list(z["groups"])

		X = torch.load(f'{self.file_path()}.pt', map_location="cpu")

		channel_size = X[0].shape[-1]

		return X, ids, groups, channel_size

	def is_embedding_file(self):

		file_types = [".npz", ".pt"]
		for type in file_types:
			if not os.path.isfile(f'{self.file_path()}{type}'):
				return False
		return True


	def file_path(self):
		pooling_str = self.pooling if self.pooling is not None else "no_pooling"
		hidden_state_str = f"_hiddenstate_{self.hidden_state}" if self.hidden_state is not None else ""
		slice_str = f"_slice_{self.slice[0]}-{self.slice[1]}" if self.slice is not None else ""
		file_path = f"{self.data_directory.rstrip("/")}/{self.esmc_model}_embeddings_{pooling_str}{hidden_state_str}{slice_str}_prefix_{self.kmer_prefix}_suffixsize_{self.kmer_suffix_size}_offset_{self.kmer_offset}"
		return file_path
	
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

		

