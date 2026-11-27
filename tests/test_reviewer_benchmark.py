import json
import gzip

import pandas as pd
import pytest
from scipy.sparse import load_npz

from reviewer_benchmark import (
    build_condition_manifest,
    main,
    prepare_split_manifest,
    write_failure_record,
)


def test_run_conditions_preserves_selection_order_and_worker_count(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"output_dir": str(tmp_path)}))
    manifest = tmp_path / "conditions.csv"
    pd.DataFrame({"condition_id": ["first", "second", "third"]}).to_csv(manifest, index=False)
    calls = []

    def record_run(config, condition, split_manifest, *, workers, progress_position, **kwargs):
        calls.append((condition["condition_id"], workers, progress_position))
        return tmp_path / condition["condition_id"]

    monkeypatch.setattr("reviewer_benchmark.run_condition", record_run)
    assert main([
        "run-conditions", "--config", str(config), "--condition-manifest", str(manifest),
        "--split-manifest", "splits.csv", "--condition-indices", "2", "0", "--workers", "2",
    ]) == 0
    assert calls == [("third", 2, 1), ("first", 2, 1)]


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


def test_fasta_sequence_source_loads_only_the_requested_genome(tmp_path, monkeypatch):
    import reviewer_benchmark

    (tmp_path / "g1.fasta").write_text(">c1\nAAAA\n")
    (tmp_path / "g2.fasta").write_text(">c1\nCCCC\n")
    loaded = []
    original = reviewer_benchmark._read_fasta_contigs

    def recording_reader(path):
        loaded.append(path.name)
        return original(path)

    monkeypatch.setattr(reviewer_benchmark, "_read_fasta_contigs", recording_reader)
    source = reviewer_benchmark._open_sequence_source(
        {
            "input": str(tmp_path),
            "file_type": "fasta",
            "id_column": "genome_id",
        }
    )

    assert list(source) == ["g1", "g2"]
    assert loaded == []
    assert source["g2"] == ["CCCC"]
    assert loaded == ["g2.fasta"]


def test_parquet_sequence_source_loads_ids_before_sequence_shard(tmp_path, monkeypatch):
    import reviewer_benchmark

    path = tmp_path / "genomes.parquet"
    pd.DataFrame(
        {"genome_id": ["g1", "g2"], "dna_sequence": ["AAAA", "CCCC GGGG"]}
    ).to_parquet(path, index=False)
    reads = []
    original = reviewer_benchmark.pd.read_parquet

    def recording_reader(*args, **kwargs):
        reads.append(kwargs.get("columns"))
        return original(*args, **kwargs)

    monkeypatch.setattr(reviewer_benchmark.pd, "read_parquet", recording_reader)
    source = reviewer_benchmark._open_sequence_source(
        {
            "input": str(path),
            "file_type": "parquet",
            "id_column": "genome_id",
            "sequence_column": "dna_sequence",
        }
    )

    assert reads == [["genome_id"]]
    assert list(source) == ["g1", "g2"]
    assert source["g2"] == ["CCCC", "GGGG"]
    assert reads == [["genome_id"], ["dna_sequence"]]


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
    condition = json.loads((output / "condition.json").read_text())
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
    assert condition["prefix"] is None
    assert isinstance(condition["total_k"], int)
    assert isinstance(condition["n_classes"], int)

    parallel_config = {**config, "output_dir": str(tmp_path / "parallel")}
    parallel_output = run_condition(
        parallel_config, build_condition_manifest(config).iloc[0], split_path, workers=2
    )
    serial_matrix = load_npz(next(output.glob("*.npz")))
    parallel_matrix = load_npz(next(parallel_output.glob("*.npz")))
    assert (serial_matrix != parallel_matrix).nnz == 0
    for name, expected in (
        ("metrics", metrics), ("predictions", predictions), ("prediction_scores", scores)
    ):
        pd.testing.assert_frame_equal(pd.read_parquet(parallel_output / f"{name}.parquet"), expected)
    parallel_resources = pd.read_parquet(parallel_output / "resources.parquet")
    assert set(parallel_resources["counting_workers"]) == {2}

    config["reverse_complement"] = True
    with pytest.raises(ValueError, match="Cached feature metadata"):
        run_condition(
            config,
            build_condition_manifest(config).iloc[0],
            split_path,
            reuse_features=True,
        )

    config["reverse_complement"] = False
    prefix_condition = build_condition_manifest(config).query(
        "representation == 'prefix_filtered'"
    ).iloc[0]
    prefix_output = run_condition(config, prefix_condition, split_path)
    hgbm_output = run_condition(
        config,
        prefix_condition,
        split_path,
        reuse_features=True,
        model_name="hist_gradient_boosting",
    )

    assert hgbm_output == prefix_output / "models" / "hist_gradient_boosting"
    hgbm_metrics = pd.read_parquet(hgbm_output / "metrics.parquet")
    hgbm_resources = pd.read_parquet(hgbm_output / "resources.parquet")
    hgbm_manifest = json.loads((hgbm_output / "condition.json").read_text())
    assert set(hgbm_metrics["model_name"]) == {"hist_gradient_boosting"}
    assert "dense_conversion" in set(hgbm_resources["stage"])
    assert set(hgbm_resources["cache_state"]) == {"warm"}
    assert hgbm_manifest["model_name"] == "hist_gradient_boosting"
    assert hgbm_manifest["model_parameters"]["random_state"] == 7


@pytest.mark.parametrize("file_type", ["fasta", "parquet"])
def test_prepare_split_manifest_clusters_full_genomes_and_writes_reusable_folds(
    tmp_path, monkeypatch, file_type
):
    genome_directory = tmp_path / "genomes"
    genome_directory.mkdir()
    labels = []
    genomes = []
    for index, sequence in enumerate(
        ["AAAAAA", "AAAAAA", "CCCCCC", "CCCCCC", "GGGGGG", "GGGGGG", "TTTTTT", "TTTTTT"]
    ):
        genome_id = f"g{index}"
        genomes.append({"genome_id": genome_id, "dna_sequence": sequence})
        if file_type == "fasta":
            (genome_directory / f"{genome_id}.fasta").write_text(f">contig\n{sequence}\n")
        labels.append(
            {
                "genome_id": genome_id,
                "trait_a": index % 2,
                "trait_b": None if index < 2 else index % 2,
            }
        )
    if file_type == "parquet":
        pd.DataFrame(genomes[::2]).to_parquet(genome_directory / "a.parquet", index=False)
        pd.DataFrame(genomes[1::2]).to_parquet(genome_directory / "b.parquet", index=False)
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame(labels).to_csv(labels_path, index=False)
    config = {
        "datasets": {
            "toy": {
                "input": str(genome_directory),
                "labels": str(labels_path),
                "file_type": file_type,
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
        "clustering": {"ksize": 3, "n_hashes": 20, **(
            {"min_similarity": 0.96} if file_type == "parquet" else {"distance_threshold": 0.1}
        )},
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
    assert len(list((output.parent / "sketches").glob("*.sig.gz"))) == 1

    from reviewer_benchmark import _LazyFastaSequences, _LazyParquetSequences

    def unexpected_sequence_read(*args):
        pytest.fail("A warm sketch cache should skip reading genome sequences")

    monkeypatch.setattr(_LazyFastaSequences, "__getitem__", unexpected_sequence_read)
    monkeypatch.setattr(_LazyParquetSequences, "__getitem__", unexpected_sequence_read)
    cached_output = prepare_split_manifest(config_path, tmp_path / "output" / "cached.csv")
    assert output.read_bytes() == cached_output.read_bytes()
