import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from utilities.benchmark_design import (
    assert_split_manifest_identity,
    create_cluster_manifest,
    create_prefix_manifest,
    create_split_manifest,
    get_phenotype_metadata,
    read_prefix_manifest,
    read_split_manifest,
    validate_experiment_matrix,
    validate_split_manifest,
)


def test_prefix_manifest_is_seeded_unique_and_includes_historical_prefix(tmp_path):
    first = create_prefix_manifest(
        total_k_values=[8, 10],
        prefix_lengths=[2, 4],
        n_prefixes=5,
        seed=19,
        historical_prefix="ACATG",
    )
    second = create_prefix_manifest(
        total_k_values=[8, 10],
        prefix_lengths=[2, 4],
        n_prefixes=5,
        seed=19,
        historical_prefix="ACATG",
    )

    pd.testing.assert_frame_equal(first, second)
    sampled = first.loc[first["prefix_source"] == "seeded"].drop_duplicates(
        ["prefix_length", "prefix"]
    )
    assert sampled.groupby("prefix_length")["prefix"].nunique().eq(5).all()
    assert "ACATG" in set(first["prefix"])
    historical = first[first["prefix"] == "ACATG"]
    assert set(historical["prefix_length"]) == {5}
    assert set(historical["suffix_length"]) == {3, 5}

    path = tmp_path / "prefixes.csv"
    create_prefix_manifest(
        total_k_values=[8],
        prefix_lengths=[2],
        n_prefixes=2,
        seed=19,
        output_path=path,
    )
    pd.testing.assert_frame_equal(read_prefix_manifest(path), create_prefix_manifest(
        total_k_values=[8], prefix_lengths=[2], n_prefixes=2, seed=19
    ))


def test_matrix_validation_rejects_impossible_prefixes_and_sample_sizes():
    validate_experiment_matrix([8, 10], [0, 2, 5], n_prefixes=4)
    with pytest.raises(ValueError, match="prefix_length"):
        validate_experiment_matrix([4], [4])
    with pytest.raises(ValueError, match="unique prefixes"):
        validate_experiment_matrix([3], [1], n_prefixes=5)
    with pytest.raises(ValueError, match="historical"):
        validate_experiment_matrix([4], [2], historical_prefix="ACATG")


def test_phenotype_metadata_is_centralized_and_customizable():
    gentamicin = get_phenotype_metadata("Gentamicin resistance")
    assert gentamicin["positive_class"] == 1
    assert gentamicin["n_classes"] == 2

    custom = get_phenotype_metadata(
        "custom", {"custom": {"positive_class": "yes", "classes": ["no", "yes"]}}
    )
    assert custom == {
        "name": "custom",
        "positive_class": "yes",
        "classes": ["no", "yes"],
        "n_classes": 2,
    }


def test_split_fold_reduction_considers_each_phenotype_class():
    genomes = pd.DataFrame(
        {
            "genome_id": ["g0", "g1", "g2", "g3"],
            "cluster_id": ["c0", "c1", "c2", "c3"],
            "trait_a": [0, 1, 0, 1],
            # The positive class occurs in one group only, so two folds
            # cannot both contain it for this phenotype.
            "trait_b": [0, 0, 0, 1],
        }
    )
    manifest = create_split_manifest(
        genomes,
        phenotype=["trait_a", "trait_b"],
        n_folds=2,
        seed=7,
    )

    assert manifest.attrs["n_folds"] == 1
    validate_split_manifest(manifest, phenotype=["trait_a", "trait_b"])


def test_read_prefix_manifest_rejects_ambiguous_bases(tmp_path):
    path = tmp_path / "prefixes.csv"
    pd.DataFrame(
        [
            {
                "total_k": 4,
                "prefix": "AN",
                "prefix_length": 2,
                "suffix_length": 2,
                "prefix_source": "seeded",
                "seed": 0,
            }
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="non-DNA"):
        read_prefix_manifest(path)


def test_cluster_manifest_sketches_contigs_independently_and_is_stable(
    monkeypatch, tmp_path
):
    class FakeMinHash:
        def __init__(self, n, ksize):
            assert n == 100
            assert ksize == 4
            self.kmers = set()

        def add_sequence(self, sequence, is_protein):
            assert is_protein is True
            self.kmers.update(
                sequence[index : index + 4]
                for index in range(len(sequence) - 4 + 1)
            )

        def jaccard(self, other):
            union = self.kmers | other.kmers
            return len(self.kmers & other.kmers) / len(union) if union else 0.0

    monkeypatch.setitem(sys.modules, "sourmash", SimpleNamespace(MinHash=FakeMinHash))
    genomes = {
        "g2": ["AAAA", "CCCC"],
        "g1": ["AAAACCCC"],
        "g3": ["AAAA", "CCCC"],
    }
    manifest = create_cluster_manifest(
        genomes,
        dataset={"g1": "d", "g2": "d", "g3": "d"},
        ksize=4,
        n_hashes=100,
        threshold=0.5,
        output_path=tmp_path / "clusters.csv",
    )
    repeat = create_cluster_manifest(
        dict(reversed(list(genomes.items()))),
        dataset="d",
        ksize=4,
        n_hashes=100,
        threshold=0.5,
    )

    pd.testing.assert_frame_equal(manifest, repeat)
    assert manifest["genome_id"].tolist() == ["g1", "g2", "g3"]
    assert manifest.loc[manifest["genome_id"] == "g2", "cluster_id"].iloc[0] == manifest.loc[
        manifest["genome_id"] == "g3", "cluster_id"
    ].iloc[0]
    assert manifest.loc[manifest["genome_id"] == "g1", "cluster_id"].iloc[0] != manifest.loc[
        manifest["genome_id"] == "g2", "cluster_id"
    ].iloc[0]
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "clusters.csv"), manifest)


def test_similarity_components_join_a_chain_and_include_the_cutoff(monkeypatch):
    class FakeMinHash:
        def __init__(self, n, ksize):
            pass

        def add_sequence(self, sequence, force):
            self.sequence = sequence

        def jaccard(self, other):
            pair = frozenset((self.sequence, other.sequence))
            return {
                frozenset(("AAAA", "CCCC")): 0.96,
                frozenset(("CCCC", "GGGG")): 0.97,
            }.get(pair, 0.91 if pair == frozenset(("AAAA", "GGGG")) else 0.5)

    monkeypatch.setitem(sys.modules, "sourmash", SimpleNamespace(MinHash=FakeMinHash))
    result = create_cluster_manifest(
        {"c": "GGGG", "b": "CCCC", "a": "AAAA", "d": "TTTT"},
        min_similarity=0.96,
    ).set_index("genome_id")
    assert result.loc[["a", "b", "c"], "cluster_id"].nunique() == 1
    assert result["cluster_id"].nunique() == 2


def test_split_manifest_without_labels_is_deterministic_and_missing_labels_are_allowed():
    genomes = pd.DataFrame(
        {
            "genome_id": ["g0", "g1", "g2", "g3"],
            "cluster_id": ["c0", "c1", "c2", "c3"],
            "trait_a": [0, 1, None, None],
            "trait_b": [None, 0, 1, 1],
        }
    )
    manifest = create_split_manifest(
        genomes, phenotype=["trait_a", "trait_b"], n_folds=2, seed=4
    )
    repeat = create_split_manifest(
        genomes.sample(frac=1, random_state=9),
        phenotype=["trait_a", "trait_b"],
        n_folds=2,
        seed=4,
    )
    assert_split_manifest_identity(manifest, repeat)
    validate_split_manifest(manifest, phenotype=["trait_a", "trait_b"])

    unlabeled = genomes.drop(columns=["trait_a", "trait_b"])
    no_label_manifest = create_split_manifest(unlabeled, n_folds=2, seed=4)
    assert no_label_manifest["fold_id"].nunique() == 2


def _labeled_genomes():
    return pd.DataFrame(
        {
            "genome_id": [f"g{i}" for i in range(12)],
            "dataset": ["toy"] * 12,
            # Two genomes per cluster make leakage detectable.
            "cluster_id": ["c0", "c0", "c1", "c1", "c2", "c2", "c3", "c3", "c4", "c4", "c5", "c5"],
            "phenotype": [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2],
        }
    )


def test_split_manifest_is_grouped_leak_free_deterministic_and_round_trips(tmp_path):
    genomes = _labeled_genomes()
    manifest = create_split_manifest(
        genomes, id_column="genome_id", phenotype="phenotype", n_folds=5, seed=7
    )
    repeat = create_split_manifest(
        genomes.sample(frac=1, random_state=3),
        id_column="genome_id", phenotype="phenotype", n_folds=5, seed=7,
    )
    assert manifest["fold_id"].nunique() == 2  # deterministic reduction: 2 clusters/class
    validate_split_manifest(manifest, phenotype="phenotype")
    assert_split_manifest_identity(manifest, repeat)
    for _, group in manifest.groupby(["dataset", "cluster_id"]):
        assert group["fold_id"].nunique() == 1

    path = tmp_path / "splits.csv"
    manifest.to_csv(path, index=False)
    loaded = read_split_manifest(path)
    assert_split_manifest_identity(manifest, loaded)


def test_split_manifest_reduces_folds_deterministically_when_class_coverage_is_limited():
    genomes = pd.DataFrame(
        {
            "genome_id": ["a0", "a1", "b0", "b1", "c0", "c1"],
            "cluster_id": ["a", "a", "b", "b", "c", "c"],
            "label": [0, 0, 1, 1, 1, 1],
        }
    )
    manifest = create_split_manifest(
        genomes, id_column="genome_id", phenotype="label", n_folds=5, seed=0
    )
    assert manifest["fold_id"].nunique() == 1
    assert "fold_reduction" in manifest.attrs["fold_reduction_rule"]
