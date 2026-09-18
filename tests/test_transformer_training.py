import numpy as np
import torch
from torch.utils.data import DataLoader

from models.Transformers_and_S4Ms import TransformerKmerClassifier
from train_models_v2 import PadCollate, SequenceDataset, fit_model, resolve_device


def test_sequence_dataset_uniformly_caps_long_sequences():
    dataset = SequenceDataset(
        [np.arange(1, 101, dtype=np.int64)],
        [1],
        max_tokens=5,
    )

    tokens, label = dataset[0]

    assert tokens.tolist() == [1, 26, 51, 75, 100]
    assert label.item() == 1


def test_transformer_ignores_padded_positions():
    torch.manual_seed(0)
    model = TransformerKmerClassifier(
        vocab_size=16,
        emb_dim=8,
        nhead=2,
        ff_dim=16,
        num_layers=1,
        dropout=0.0,
    ).eval()

    short = torch.tensor([[1, 2, 3]])
    padded = torch.tensor([[1, 2, 3, 0, 0]])

    with torch.no_grad():
        short_output = model(short)
        padded_output = model(padded)

    torch.testing.assert_close(short_output, padded_output, atol=1e-6, rtol=1e-6)


def test_transformer_completes_a_cpu_training_epoch():
    sequences = np.array(
        [
            np.array([1, 2, 3, 4, 5]),
            np.array([5, 4, 3, 2]),
            np.array([1, 1, 2]),
            np.array([5, 5, 4, 4, 3, 3]),
        ],
        dtype=object,
    )
    labels = np.array([0, 1, 0, 1])
    dataset = SequenceDataset(sequences, labels, max_tokens=4)
    loader = DataLoader(dataset, batch_size=2, collate_fn=PadCollate("pad_collate"))

    result = fit_model(
        loader,
        loader,
        loader,
        device=torch.device("cpu"),
        num_epochs=1,
        learning_rate=1e-3,
        num_classes=2,
        model_type="TRANSFORMER",
        vocab_size=8,
        dropout=0.0,
        embedding_class="integer",
        transformer_emb_dim=8,
        transformer_heads=2,
        transformer_ff_dim=16,
        transformer_layers=1,
    )

    assert result["test_outputs"].shape == (4, 2)
    assert np.isfinite(result["test_outputs"]).all()


def test_laptop_default_token_budget_completes_backward_pass():
    model = TransformerKmerClassifier(
        vocab_size=1025,
        emb_dim=32,
        nhead=4,
        ff_dim=64,
        num_layers=2,
        dropout=0.0,
    )
    tokens = torch.randint(1, 1025, (2, 2048))
    labels = torch.tensor([0, 1])

    loss = torch.nn.functional.cross_entropy(model(tokens), labels)
    loss.backward()

    assert torch.isfinite(loss)


def test_auto_device_keeps_transformer_off_mps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert resolve_device("auto", model_type="TRANSFORMER") == torch.device("cpu")
