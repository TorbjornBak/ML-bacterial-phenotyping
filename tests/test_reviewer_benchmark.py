import json
import gzip

import pandas as pd
import pytest

from reviewer_benchmark import (
    build_condition_manifest,
    prepare_split_manifest,
    write_failure_record,
)


def _config():
    return {
        "datasets": {"toy": {"input": "unused", "labels": "unused"}},
        "phenotypes": [
            {
                "name": "trait_a",
                "dataset": "toy",
                "label_column": "trait_a",
                "positive_class": 1,
                "classes": [0, 1],
            },
            {
                "name": "trait_b",
                "dataset": "toy",
                "label_column": "trait_b",
                "positive_class": 1,
                "classes": [0, 1],
            },
        ],
        "total_k": [8, 10],
        "prefix_lengths": [2, 5],
        "prefix_seed": 11,
        "n_prefixes": 2,
        "historical_prefix": "ACATG",
    }


def test_condition_manifest_has_full_and_prespecified_filtered_conditions():
    manifest = build_condition_manifest(_config())

    full = manifest[manifest["representation"] == "full"]
    assert len(full) == 4
    assert full["prefix"].isna().all()

    filtered = manifest[manifest["representation"] == "prefix_filtered"]
    assert set(filtered["dataset"]) == {"toy"}
    assert set(filtered["prefix_source"]) == {"seeded", "historical"}
    assert set(filtered["phenotype"]) == {"trait_a", "trait_b"}
    assert manifest["condition_id"].is_unique
    assert set(manifest["n_classes"]) == {2}
    assert set(manifest["classes_json"]) == {"[0, 1]"}


def test_condition_manifest_rejects_positive_class_outside_declared_classes():
    config = _config()
    config["phenotypes"][0]["positive_class"] = 2

    try:
        build_condition_manifest(config)
    except ValueError as error:
        assert "positive_class" in str(error)
    else:
        raise AssertionError("invalid phenotype orientation was silently accepted")


def test_pilot_manifest_is_one_phenotype_one_k_and_two_length_five_prefixes():
    manifest = build_condition_manifest(_config(), pilot=True)

    assert set(manifest["phenotype"]) == {"trait_a"}
    assert set(manifest["total_k"]) == {8}
    assert len(manifest[manifest["representation"] == "full"]) == 1
    seeded = manifest[manifest["prefix_source"] == "seeded"]
    assert len(seeded) == 2
    assert set(seeded["prefix_length"]) == {5}


def test_failure_records_are_machine_readable(tmp_path):
    condition = pd.Series({"condition_id": "toy__trait-a__full__k8"})
    path = write_failure_record(tmp_path, condition, RuntimeError("out of memory"))

    record = json.loads(path.read_text())
    assert record["condition_id"] == condition["condition_id"]
    assert record["error_type"] == "RuntimeError"
    assert record["message"] == "out of memory"


def test_conflicting_duplicate_labels_are_rejected(tmp_path):
    from reviewer_benchmark import _validate_unique_labels

    labels = pd.DataFrame({"genome_id": ["g1", "g1"], "trait": [0, 1]})
    try:
        _validate_unique_labels(labels, "genome_id", "trait")
    except ValueError as error:
        assert "conflicting duplicate labels" in str(error)
    else:
        raise AssertionError("conflicting duplicate labels were silently accepted")


def test_fasta_loader_reads_mixed_compressed_and_uncompressed_files(tmp_path):
    from reviewer_benchmark import _load_sequences

    (tmp_path / "plain.fasta").write_text(">c1\nAAAA\n")
    with gzip.open(tmp_path / "compressed.fasta.gz", "wt") as stream:
        stream.write(">c1\nCCCC\n")

    sequences, total_bases = _load_sequences(
        {
            "input": str(tmp_path),
            "file_type": "fasta",
            "id_column": "genome_id",
            "sequence_column": "dna_sequence",
        }
    )

    assert sequences == {"compressed": ["CCCC"], "plain": ["AAAA"]}
    assert total_bases == 8


def test_single_condition_writes_tidy_fold_prediction_and_resource_outputs(tmp_path):
    from reviewer_benchmark import run_condition

    genome_directory = tmp_path / "genomes"
    genome_directory.mkdir()
    labels = []
    split_rows = []
    for index in range(8):
        genome_id = f"g{index}"
        label = index % 2
        sequence = "AAAAAAAAAACCCCCCCCCC" if label == 0 else "TTTTTTTTTTGGGGGGGGGG"
        (genome_directory / f"{genome_id}.fasta").write_text(f">contig\n{sequence}\n")
        labels.append({"genome_id": genome_id, "trait": label})
        split_rows.append(
            {
                "genome_id": genome_id,
                "dataset": "toy",
                "cluster_id": f"c{index}",
                "fold_id": index // 4,
                "requested_n_folds": 5,
                "effective_n_folds": 2,
                "fold_reduction_rule": "fold_reduction: requested 5, used 2",
            }
        )
    label_path = tmp_path / "labels.csv"
    split_path = tmp_path / "splits.csv"
    pd.DataFrame(labels).to_csv(label_path, index=False)
    pd.DataFrame(split_rows).to_csv(split_path, index=False)
    config = {
        "datasets": {
            "toy": {
                "input": str(genome_directory),
                "labels": str(label_path),
                "file_type": "fasta",
                "id_column": "genome_id",
                "sequence_column": "dna_sequence",
            }
        },
        "phenotypes": [
            {
                "name": "trait",
                "dataset": "toy",
                "label_column": "trait",
                "positive_class": 1,
                "classes": [0, 1],
            }
        ],
        "total_k": [2],
        "prefix_lengths": [1],
        "prefix_seed": 7,
        "n_prefixes": 1,
        "n_folds": 5,
        "normalization": "frequency",
        "model": {
            "solver": "liblinear",
            "class_weight": "balanced",
            "random_state": 7,
        },
        "output_dir": str(tmp_path / "output"),
    }

    output = run_condition(config, build_condition_manifest(config).iloc[0], split_path)

    metrics = pd.read_parquet(output / "metrics.parquet")
    predictions = pd.read_parquet(output / "predictions.parquet")
    resources = pd.read_parquet(output / "resources.parquet")
    scores = pd.read_parquet(output / "prediction_scores.parquet")
    assert set(metrics["fold_id"]) == {0, 1}
    assert predictions["genome_id"].nunique() == 8
    assert pd.api.types.is_numeric_dtype(predictions["score"])
    assert scores.groupby("genome_id")["score_class"].nunique().eq(2).all()
    assert {
        "loading",
        "tokenization_and_counting",
        "matrix_construction",
        "fitting",
        "prediction",
    }.issubset(resources["stage"])
    assert set(resources["cache_state"]) == {"cold"}

    config["reverse_complement"] = True
    with pytest.raises(ValueError, match="Cached feature metadata"):
        run_condition(
            config,
            build_condition_manifest(config).iloc[0],
            split_path,
            reuse_features=True,
        )


def test_prepare_split_manifest_clusters_full_genomes_and_writes_reusable_folds(
    tmp_path, monkeypatch
):
    genome_directory = tmp_path / "genomes"
    genome_directory.mkdir()
    labels = []
    for index, sequence in enumerate(
        ["AAAAAA", "AAAAAA", "CCCCCC", "CCCCCC", "GGGGGG", "GGGGGG", "TTTTTT", "TTTTTT"]
    ):
        genome_id = f"g{index}"
        (genome_directory / f"{genome_id}.fasta").write_text(f">contig\n{sequence}\n")
        labels.append(
            {
                "genome_id": genome_id,
                "trait_a": index % 2,
                "trait_b": None if index < 2 else index % 2,
            }
        )
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame(labels).to_csv(labels_path, index=False)
    config = {
        "datasets": {
            "toy": {
                "input": str(genome_directory),
                "labels": str(labels_path),
                "file_type": "fasta",
                "id_column": "genome_id",
                "sequence_column": "dna_sequence",
            }
        },
        "phenotypes": [
            {"name": "a", "dataset": "toy", "label_column": "trait_a", "positive_class": 1, "classes": [0, 1]},
            {"name": "b", "dataset": "toy", "label_column": "trait_b", "positive_class": 1, "classes": [0, 1]},
        ],
        "n_folds": 2,
        "split_seed": 13,
        "clustering": {"ksize": 3, "n_hashes": 20, "distance_threshold": 0.1},
        "output_dir": str(tmp_path / "output"),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    output = prepare_split_manifest(config_path)
    manifest = pd.read_csv(output)

    assert output == tmp_path / "output" / "split_manifest.csv"
    assert set(manifest["dataset"]) == {"toy"}
    assert manifest.groupby("cluster_id")["fold_id"].nunique().eq(1).all()
    assert {"trait_a", "trait_b", "requested_n_folds", "effective_n_folds"}.issubset(
        manifest.columns
    )
