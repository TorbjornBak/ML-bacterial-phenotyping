import numpy as np

from utilities.clustering import SourMashClustering


class _MinHash:
    def __init__(self, similarities):
        self.similarities = similarities

    def jaccard(self, other):
        if other is self:
            return 1.0
        return self.similarities[id(other)]


def test_jaccard_distance_matrix_converts_similarity_to_distance():
    left = _MinHash({})
    right = _MinHash({id(left): 0.25})
    left.similarities[id(right)] = 0.25
    matrix, labels = SourMashClustering(kmer_suffix_size=3, target_labels={}).jaccard_distance_matrix(
        {"left": left, "right": right}
    )
    assert labels == ["left", "right"]
    np.testing.assert_allclose(matrix, [[0.0, 0.75], [0.75, 0.0]])
