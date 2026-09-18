import numpy as np
import pandas as pd
import torch

from reviewer_rnn_pilot import (
    collate_suffix_codes,
    extract_prefix_suffix_codes,
    run_rnn_pilot,
)


def test_prefix_suffix_codes_preserve_order_and_contig_boundaries():
    codes, raw_count = extract_prefix_suffix_codes(
        ["ATGAAATGCC", "ATGTT"],
        total_k=5,
        prefix="ATG",
        max_tokens=None,
    )

    assert raw_count == 3
    assert codes.tolist() == [0, 5, 15]  # AA, CC, TT in base-four order


def test_prefix_suffix_codes_are_evenly_capped():
    codes, raw_count = extract_prefix_suffix_codes(
        "ATGAAATGACATGAGATGATATGCA",
        total_k=5,
        prefix="ATG",
        max_tokens=3,
    )

    assert raw_count == 5
    assert codes.tolist() == [0, 2, 4]  # first, middle, and last retained token


def test_collate_suffix_codes_builds_flattened_base_one_hot_and_mask():
    batch = [
        (np.array([0, 15], dtype=np.uint32), 1),
        (np.array([6], dtype=np.uint32), 0),
    ]

    inputs, lengths, mask, labels = collate_suffix_codes(batch, suffix_length=2)

    assert inputs.shape == (2, 2, 8)
    assert lengths.tolist() == [2, 1]
    assert mask.tolist() == [[True, True], [True, False]]
    assert labels.tolist() == [1, 0]
    assert torch.equal(inputs[0, 0], torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float32))
    assert torch.equal(inputs[0, 1], torch.tensor([0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float32))
    assert not inputs[1, 1].any()


def test_rnn_pilot_writes_comparable_fold_outputs(tmp_path):
    genomes = tmp_path / "genomes"
    genomes.mkdir()
    label_rows = []
    split_rows = []
    for index in range(10):
        genome_id = f"g{index}"
        label = "Resistant" if index % 2 else "Susceptible"
        suffix = "TT" if label == "Resistant" else "AA"
        (genomes / f"{genome_id}.fasta").write_text(
            f">contig\n{('ATG' + suffix) * 5}\n"
        )
        label_rows.append({"genome_id": genome_id, "trait": label})
        split_rows.append(
            {
                "genome_id": genome_id,
                "dataset": "toy",
                "cluster_id": genome_id,
                "fold_id": index // 2,
            }
        )
    labels = tmp_path / "labels.csv"
    splits = tmp_path / "splits.csv"
    pd.DataFrame(label_rows).to_csv(labels, index=False)
    pd.DataFrame(split_rows).to_csv(splits, index=False)
    config = {
        "datasets": {
            "toy": {
                "input": str(genomes),
                "labels": str(labels),
                "file_type": "fasta",
                "id_column": "genome_id",
                "label_separator": ",",
            }
        },
        "output_dir": str(tmp_path / "output"),
    }
    condition = {
        "condition_id": "toy__trait__k5__prefix-ATG",
        "dataset": "toy",
        "phenotype": "trait",
        "label_column": "trait",
        "positive_class": "Resistant",
        "n_classes": 2,
        "classes_json": '["Susceptible", "Resistant"]',
        "representation": "prefix_filtered",
        "total_k": 5,
        "prefix": "ATG",
        "prefix_length": 3,
        "suffix_length": 2,
        "prefix_source": "historical",
        "seed": 7,
    }

    output = run_rnn_pilot(
        config,
        condition,
        splits,
        fold_ids=[0],
        max_tokens=4,
        epochs=1,
        patience=1,
        batch_size=2,
        hidden_size=4,
        head_dim=4,
        seed=7,
        torch_threads=1,
    )

    metrics = pd.read_parquet(output / "metrics.parquet")
    resources = pd.read_parquet(output / "resources.parquet")
    predictions = pd.read_parquet(output / "predictions.parquet")
    assert metrics["fold_id"].tolist() == [0]
    assert metrics["validation_fold_id"].tolist() == [1]
    assert predictions["genome_id"].nunique() == 2
    assert set(resources["stage"]) >= {"loading", "ordered_tokenization", "fitting", "overall"}
    assert resources["sampled_token_count"].iloc[0] == 40
