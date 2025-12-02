import pytest

from utilities.clustering import SourMashClustering
from embeddings.KmerTokenization import KmerTokenizer

def test_sourmash_clustering():
    kmer_prefix="ATG"
    kmer_suffix_size=12
    
    tokenizer = KmerTokenizer(input_path="tests/test_data/ClusteringTest/", 
                                genome_col="genome_id",
                                dna_sequence_col="dna_sequence",
                                kmer_prefix=kmer_prefix,
                                kmer_suffix_size = kmer_suffix_size,
                                file_type="fasta",
                                reverse_complement=True
                                )
    
    token_collection = tokenizer.run_tokenizer(nr_of_cores=2)

    clusterer = SourMashClustering(kmer_suffix_size=kmer_suffix_size)
    minhashes = clusterer.run_clustering(token_dict=token_collection)
    
    assert len(minhashes) == 9*2  # 9 genomes strands each


    distance_matrix, labels = clusterer.jaccard_distance_matrix(minhashes=minhashes)

    clusterer.plot_composite_matrix(distance_matrix=distance_matrix, labels=labels, output_path="tests/test_data/test_output")

    