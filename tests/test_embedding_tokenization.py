import pytest
from embeddings import tokenization
from embeddings import integer_embeddings, esmc_embeddings
import torch
import numpy as np

from Bio.Seq import Seq




# def test_tokenize_sequence_and_rev_complement():
#     sequences = ["ACTGCT", 
#                     "ACTCGT"]
#     kmer_prefix="AC"
#     kmer_suffix_size=3
	

#     kmers = tokenization.tokenize_sequence_and_rev_complement(sequences, 
#                                             kmer_prefix,
#                                             kmer_suffix_size,
#                                             reverse_complement = True)
	
#     assert "forward_kmers" in kmers
#     assert "reverse_kmers" in kmers

#     assert kmers["forward_kmers"] == [Seq('TGC'), Seq('TCG'),]
#     assert kmers["reverse_kmers"] == [Seq('GAG')]


class TestTokenizer():

	def test_kmer_tokenizer(self):
		kmer_prefix="ACTCGCA"
		kmer_suffix_size=6
		
		tokenizer = tokenization.KmerTokenizer(input_path="downloads/", 
									genome_col="genome_name",
									dna_sequence_col="dna_sequence",
									kmer_prefix=kmer_prefix,
									kmer_suffix_size = kmer_suffix_size,
									file_type="parquet",
									reverse_complement=True
									)
		
		token_collection = tokenizer.run_tokenizer(nr_of_cores=2)

		embedder = integer_embeddings.IntegerEmbeddings(token_collection=token_collection, 
						kmer_suffix_size=kmer_suffix_size,
						compress_embeddings=False)
			
		embeddings = embedder.run_embedder(nr_of_cores=2)
		
		gid_and_strand_id = [[gid, strand_id] for gid, strands in embeddings.items() for strand_id in strands]


	# def test_esmc_embedding():
	# 	kmer_prefix="ACTCGCA"
	# 	kmer_suffix_size=6
		
	# 	tokenizer = tokenization.KmerTokenizer(input_path="downloads/", 
	# 								genome_col="genome_name",
	# 								dna_sequence_col="dna_sequence",
	# 								kmer_prefix=kmer_prefix,
	# 								kmer_suffix_size = kmer_suffix_size,
	# 								file_type="parquet",
	# 								reverse_complement=True
	# 								)
		
	# 	token_collection = tokenizer.run_tokenizer(nr_of_cores=2)

	# 	embedder = tokenization.ESMcEmbeddings(token_collection=token_collection, 
	# 					kmer_suffix_size=kmer_suffix_size,
	# 					compress_embeddings=False)
			
	# 	embeddings = embedder.run_embedder(nr_of_cores=2)
		
	# 	gid_and_strand_id = [[gid, strand_id] for gid, strands in embeddings.items() for strand_id in strands]



	def test_kmer_tokenizer_2(self):
		kmer_prefix="ACTCGCA"
		kmer_suffix_size=7
		
		tokenizer = tokenization.KmerTokenizer(input_path="downloads/", 
									genome_col="genome_name",
									dna_sequence_col="dna_sequence",
									kmer_prefix=kmer_prefix,
									kmer_suffix_size = kmer_suffix_size,
									file_type="parquet",
									reverse_complement=True
									)
		# Should fail
		with pytest.raises(AssertionError):
			token_collection = tokenizer.run_tokenizer(nr_of_cores=2)


class TestDNASequences():

	def test_fasta_reader(self):
		fasta_path = "tests/test_data/562.97505.fasta"
		df = tokenization.read_sequence_file(fasta_path, file_type="fasta")
		print(df)
		
		assert df["genome_id"][0] == "562.97505"
		assert df["dna_sequence"][0].startswith("gagcaccgtattgacgc")


class TestEmbeddings():
	
	def test_esmc_mean_pooling(self):
		kmer_prefix="ACTCGCA"
		kmer_suffix_size=6
		directory = "tests/test_data/EmbeddingTest/"
		
		tokenizer = tokenization.KmerTokenizer(input_path=directory, 
									genome_col="genome_id",
									dna_sequence_col="dna_sequence",
									kmer_prefix=kmer_prefix,
									kmer_suffix_size = kmer_suffix_size,
									file_type="fasta",
									reverse_complement=False
									)
		
		token_collection = tokenizer.run_tokenizer(nr_of_cores=1)

		embedder = esmc_embeddings.ESMcEmbeddings(
											kmer_prefix=kmer_prefix,
										kmer_suffix_size=kmer_suffix_size,
										esmc_model="esmc_300m",
										pooling="mean",
										device="mps",
										data_directory=directory,
										
										)
			
		embeddings = embedder.run_embedder(token_collection=token_collection,nr_of_cores=2)
		
		gid_and_strand_id = [[gid, strand_id] for gid, strands in embeddings.items() for strand_id in strands]
		X = [embeddings[gid][strand_id] for gid, strand_id in gid_and_strand_id]
		groups = [gid for gid, _ in gid_and_strand_id]

		X = np.array(
			[
				(x.detach().cpu() if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32))
				for gid, x in zip(groups, X)
			],
			dtype=np.float32
		)	

		# SHould not fail

	def test_esmc_mean_per_token_pooling(self):
		kmer_prefix="ACTCGCA"
		kmer_suffix_size=6
		directory = "tests/test_data/EmbeddingTest/"
		
		tokenizer = tokenization.KmerTokenizer(input_path=directory, 
									genome_col="genome_id",
									dna_sequence_col="dna_sequence",
									kmer_prefix=kmer_prefix,
									kmer_suffix_size = kmer_suffix_size,
									file_type="fasta",
									reverse_complement=False
									)
		
		token_collection = tokenizer.run_tokenizer(nr_of_cores=1)

		embedder = esmc_embeddings.ESMcEmbeddings(
											kmer_prefix=kmer_prefix,
										kmer_suffix_size=kmer_suffix_size,
										esmc_model="esmc_300m",
										pooling="mean_per_token",
										device="mps",
										data_directory=directory,
										
										)
			
		embeddings = embedder.run_embedder(token_collection=token_collection,nr_of_cores=2)
		
		gid_and_strand_id = [[gid, strand_id] for gid, strands in embeddings.items() for strand_id in strands]
		X = [embeddings[gid][strand_id] for gid, strand_id in gid_and_strand_id]
		groups = [gid for gid, _ in gid_and_strand_id]
		assert len(X[0].shape) == 2, "Expected 2D tensor for mean_per_token pooling"
		print(f'{X[0].shape=}')
		X = np.array(
			[
				(x.detach().cpu() if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32))
				for gid, x in zip(groups, X)
			],
			dtype=object
		)	
		print(f'{X.shape=}')
		
		# Should not fail