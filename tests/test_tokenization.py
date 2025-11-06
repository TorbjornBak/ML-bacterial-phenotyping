import pytest
from embeddings import tokenization
from Bio.Seq import Seq




def test_tokenize_sequence_and_rev_complement():
    sequences = ["ACTGCT", 
                    "ACTCGT"]
    kmer_prefix="AC"
    kmer_suffix_size=3
    

    kmers = tokenization.tokenize_sequence_and_rev_complement(sequences, 
                                            kmer_prefix,
                                            kmer_suffix_size,
                                            reverse_complement = True)
    
    assert "forward_kmers" in kmers
    assert "reverse_kmers" in kmers

    

    assert kmers["forward_kmers"] == [Seq('TGC'), Seq('TCG'),]
    assert kmers["reverse_kmers"] == [Seq('GAG')]

    print(f'{kmers["forward_kmers"]=}')
    print(f'{kmers["reverse_kmers"]=}')
    #assert kmers["forward_kmers"] == 

        
