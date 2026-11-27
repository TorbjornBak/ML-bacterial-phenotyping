import numpy as np
import pandas as pd
import pytest
import torch

from models.Transformers_and_S4Ms import TransformerKmerClassifier
from reviewer_transformer_pilot import collate_token_codes, run_transformer_pilot


def _model(**overrides):
    torch.manual_seed(0)
    parameters = dict(vocab_size=17, emb_dim=8, nhead=2, ff_dim=16, num_layers=2, dropout=0.0)
    parameters.update(overrides)
    return TransformerKmerClassifier(**parameters).eval()


def test_collate_token_codes_shifts_codes_past_padding():
    batch = [
        (np.array([0, 15], dtype=np.uint32), 1),
        (np.array([6], dtype=np.uint32), 0),
        (np.array([], dtype=np.uint32), 0),
    ]

    tokens, lengths, mask, labels = collate_token_codes(batch)

    assert tokens.tolist() == [[1, 16], [7, 0], [0, 0]]
    assert lengths.tolist() == [2, 1, 1]
    assert mask.tolist() == [[True, True], [True, False], [True, False]]
    assert labels.tolist() == [1, 0, 0]


def test_transformer_output_does_not_depend_on_padding():
    model = _model()
    tokens = torch.tensor([[3, 5, 7, 9]])
    padded = torch.tensor([[3, 5, 7, 9, 0, 0, 0, 0, 0, 0]])

    with torch.no_grad():
        alone = model(tokens)
        with_padding = model(padded, mask=padded != 0)
        from_lengths = model(padded, lengths=torch.tensor([4]))
        from_pad_id = model(padded)

    torch.testing.assert_close(with_padding, alone)
    torch.testing.assert_close(from_lengths, alone)
    torch.testing.assert_close(from_pad_id, alone)


def test_transformer_handles_empty_rows_without_nan():
    model = _model()
    tokens = torch.tensor([[0, 0, 0], [2, 4, 0]])

    with torch.no_grad():
        logits = model(tokens, mask=torch.tensor([[False] * 3, [True, True, False]]))

    assert torch.isfinite(logits).all()


def test_attention_free_ablation_and_no_positions_run():
    tokens = torch.tensor([[1, 2, 3], [4, 0, 0]])

    for model in (_model(num_layers=0), _model(positional_encoding=False)):
        with torch.no_grad():
            assert model(tokens).shape == (2, 2)


def test_transformer_learns_token_identity_task():
    torch.manual_seed(0)
    model = TransformerKmerClassifier(
        vocab_size=5, emb_dim=8, nhead=2, ff_dim=16, num_layers=1, dropout=0.0
    )
    lengths = torch.randint(2, 9, (32,))
    labels = torch.arange(32) % 2
    tokens = torch.zeros((32, 8), dtype=torch.long)
    for row, (length, label) in enumerate(zip(lengths, labels)):
        tokens[row, :length] = 1 + 2 * label + torch.randint(0, 2, (int(length),))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(40):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(tokens, lengths), labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        accuracy = (model(tokens, lengths).argmax(1) == labels).float().mean()
    assert accuracy == 1.0


def _toy_benchmark(tmp_path):
    genomes = tmp_path / "genomes"
    genomes.mkdir()
    label_rows, split_rows = [], []
    for index in range(10):
        genome_id = f"g{index}"
        label = "Resistant" if index % 2 else "Susceptible"
        suffix = "TT" if label == "Resistant" else "AA"
        (genomes / f"{genome_id}.fasta").write_text(f">contig\n{('ATG' + suffix) * 5}\n")
        label_rows.append({"genome_id": genome_id, "trait": label})
        split_rows.append(
            {"genome_id": genome_id, "dataset": "toy", "cluster_id": genome_id, "fold_id": index // 2}
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
    return config, condition, splits


def test_transformer_pilot_writes_comparable_fold_outputs(tmp_path):
    config, condition, splits = _toy_benchmark(tmp_path)

    output = run_transformer_pilot(
        config,
        condition,
        splits,
        fold_ids=[0],
        max_tokens=4,
        epochs=1,
        patience=1,
        batch_size=2,
        emb_dim=8,
        nhead=2,
        ff_dim=16,
        num_layers=1,
        seed=7,
        torch_threads=1,
    )

    assert output.parent.name == "transformer"
    metrics = pd.read_parquet(output / "metrics.parquet")
    resources = pd.read_parquet(output / "resources.parquet")
    predictions = pd.read_parquet(output / "predictions.parquet")
    assert metrics["model_name"].tolist() == ["transformer"]
    assert metrics["fold_id"].tolist() == [0]
    assert metrics["validation_fold_id"].tolist() == [1]
    assert predictions["genome_id"].nunique() == 2
    assert set(resources["stage"]) >= {"loading", "ordered_tokenization", "fitting", "overall"}
    assert resources["sampled_token_count"].iloc[0] == 40


def test_transformer_pilot_rejects_mps(tmp_path):
    config, condition, splits = _toy_benchmark(tmp_path)

    with pytest.raises(ValueError, match="unavailable"):
        run_transformer_pilot(config, condition, splits, fold_ids=[0], device="mps")
