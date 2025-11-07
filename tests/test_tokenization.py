import pytest
from embeddings import tokenization
from embeddings import integer_embeddings
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




def test_kmer_tokenizer():
	kmer_prefix="ACTCGCA"
	kmer_suffix_size=3
	
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
		
	embeddings, vocab_size = embedder.run_embedder(nr_of_cores=2)
	
	print(embeddings)