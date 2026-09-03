import json
from contextlib import contextmanager

import numpy as np
from scipy import sparse

from embeddings.sparse_kmer_counts import (
    build_sparse_kmer_counts,
    decode_kmer,
    encode_kmer,
    load_sparse_kmer_counts,
)


def test_base4_encoding_is_deterministic_and_round_trips():
    assert encode_kmer("AAAA") == 0
    assert encode_kmer("ACGT") == 27
    assert encode_kmer("TTTT") == 255
    assert decode_kmer(27, 4) == "ACGT"


def test_full_counts_have_known_columns_and_raw_token_metadata():
    result = build_sparse_kmer_counts(
        {"g1": "ACGTAC", "g2": "AAAA"}, total_k=3, normalization="raw"
    )

    assert sparse.isspmatrix_csr(result.matrix)
    assert result.feature_space_size == 4**3
    assert result.matrix.shape == (2, 4**3)
    assert result.matrix.toarray()[0, encode_kmer("ACG")] == 1
    assert result.matrix.toarray()[0, encode_kmer("CGT")] == 1
    assert result.matrix.toarray()[0, encode_kmer("GTA")] == 1
    assert result.matrix.toarray()[0, encode_kmer("TAC")] == 1
    assert result.matrix.toarray()[1, encode_kmer("AAA")] == 2
    assert result.raw_token_counts.tolist() == [4, 2]
    assert result.metadata["normalization"] == "raw"


def test_prefix_counts_are_suffix_columns_and_match_full_subset():
    sequences = {"g1": "AACGTA", "g2": "TTACG"}
    full = build_sparse_kmer_counts(sequences, total_k=3, normalization="raw")
    filtered = build_sparse_kmer_counts(
        sequences, total_k=3, prefix="AC", normalization="raw"
    )

    assert filtered.feature_space_size == 4
    assert filtered.metadata["prefix_length"] == 2
    assert filtered.metadata["suffix_length"] == 1
    assert filtered.matrix.toarray()[:, encode_kmer("G")].tolist() == [1, 1]
    for suffix_index in range(4):
        expected = full.matrix[:, [encode_kmer("AC" + decode_kmer(suffix_index, 1))]]
        np.testing.assert_array_equal(
            filtered.matrix[:, suffix_index].toarray().ravel(), expected.toarray().ravel()
        )


def test_prefix_frequencies_use_all_valid_candidate_windows_as_denominator():
    result = build_sparse_kmer_counts(
        {"g1": "AACGTA"}, total_k=3, prefix="AC", normalization="frequency"
    )

    assert result.raw_token_counts.tolist() == [1]
    assert result.valid_window_counts.tolist() == [4]
    assert result.matrix[0, encode_kmer("G")] == 0.25


def test_prefix_scan_preserves_overlaps_and_ambiguous_window_denominator():
    result = build_sparse_kmer_counts(
        {"g1": "AAAAANAAAAA"}, total_k=3, prefix="AA", normalization="frequency"
    )

    assert result.metadata["candidate_window_counts"] == [9]
    assert result.valid_window_counts.tolist() == [6]
    assert result.raw_token_counts.tolist() == [6]
    assert result.matrix[0, encode_kmer("A")] == 1.0


def test_prefix_scan_matches_exhaustive_counts_across_edge_cases():
    rng = np.random.default_rng(20260903)
    alphabet = np.array(list("ACGTNacgtn"))
    genomes = {
        "random": ["".join(rng.choice(alphabet, size=80)) for _ in range(3)],
        "boundaries": ["AC", "GTACGT", "NACGTN", "ACGT"],
        "short": ["A", "NN", "acg"],
    }

    total_k = 4
    prefix = "AC"
    prefix_columns = [
        encode_kmer(prefix + decode_kmer(suffix, total_k - len(prefix)))
        for suffix in range(4 ** (total_k - len(prefix)))
    ]
    for reverse_complement in (False, True):
        full = build_sparse_kmer_counts(
            genomes,
            total_k=total_k,
            normalization="raw",
            reverse_complement=reverse_complement,
        )
        filtered = build_sparse_kmer_counts(
            genomes,
            total_k=total_k,
            prefix=prefix,
            normalization="raw",
            reverse_complement=reverse_complement,
        )

        np.testing.assert_array_equal(
            filtered.matrix.toarray(), full.matrix[:, prefix_columns].toarray()
        )
        np.testing.assert_array_equal(
            filtered.valid_window_counts, full.valid_window_counts
        )
        np.testing.assert_array_equal(
            filtered.raw_token_counts,
            np.asarray(full.matrix[:, prefix_columns].sum(axis=1)).ravel(),
        )
        assert (
            filtered.metadata["candidate_window_counts"]
            == full.metadata["candidate_window_counts"]
        )


def test_lowercase_ambiguous_bases_and_contig_boundaries_are_handled():
    result = build_sparse_kmer_counts(
        {"g1": ["acg", "tac"], "g2": "AANAAA"}, total_k=3, normalization="raw"
    )

    # ACG and TAC are the only complete windows; no k-mer is made across the
    # boundary between ACG and TAC.
    assert result.raw_token_counts.tolist() == [2, 1]
    assert result.matrix[0, encode_kmer("ACG")] == 1
    assert result.matrix[0, encode_kmer("TAC")] == 1
    assert result.matrix[1, encode_kmer("AAA")] == 1
    assert result.matrix[1].nnz == 1


def test_reverse_complement_aggregation_and_empty_rows():
    result = build_sparse_kmer_counts(
        {"g1": "ACG", "empty": "NNNN", "short": "A"},
        total_k=3,
        reverse_complement=True,
        normalization="frequency",
    )

    assert result.raw_token_counts.tolist() == [2, 0, 0]
    assert result.matrix[0, encode_kmer("ACG")] == 0.5
    assert result.matrix[0, encode_kmer("CGT")] == 0.5
    assert result.matrix[1].nnz == 0
    assert result.matrix[2].nnz == 0


def test_save_and_load_round_trip_uses_unambiguous_filename(tmp_path):
    result = build_sparse_kmer_counts(
        {"g1": "ACGT"},
        total_k=3,
        prefix="A",
        normalization="frequency",
        dataset="synthetic",
    )
    path = result.save(tmp_path, dataset="synthetic")

    assert path.name == "synthetic_sparse_kmers_total_k_3_prefix_A_normalization_frequency.npz"
    assert path.exists()
    assert path.with_suffix(".json").exists()
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["dataset"] == "synthetic"
    assert metadata["total_k"] == 3
    assert metadata["prefix"] == "A"

    loaded = load_sparse_kmer_counts(path)
    np.testing.assert_array_equal(loaded.matrix.toarray(), result.matrix.toarray())
    assert loaded.metadata == result.metadata
    np.testing.assert_array_equal(loaded.raw_token_counts, result.raw_token_counts)


def test_stage_callback_separates_counting_from_matrix_construction():
    events = []

    @contextmanager
    def stage(name):
        events.append((name, "start"))
        yield
        events.append((name, "finish"))

    build_sparse_kmer_counts(
        {"g1": "ACGTAC"},
        total_k=3,
        normalization="raw",
        stage_callback=stage,
    )

    assert events == [
        ("tokenization_and_counting", "start"),
        ("tokenization_and_counting", "finish"),
        ("matrix_construction", "start"),
        ("matrix_construction", "finish"),
    ]
