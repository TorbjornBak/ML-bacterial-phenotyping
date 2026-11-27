"""Transformer pilots for prefix-filtered reviewer conditions.

Uses the same ordered suffix tokens, frozen outer folds, metrics, and output
layout as ``reviewer_rnn_pilot``, but embeds each suffix k-mer as one integer
token and classifies with ``TransformerKmerClassifier``.  Attention runs
through fused scaled-dot-product kernels, so memory grows linearly with the
token cap; on CPU a full-length ACATG/k=11 genome (~3-4k tokens) trains with
well under 1 GB of activation memory at batch size 8.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from reviewer_rnn_pilot import condition_from_config, fit_sequence_fold, run_sequence_pilot


def collate_token_codes(batch: Sequence[tuple[np.ndarray, int]]):
    """Pad suffix codes as integer tokens: code ``c`` becomes ``c + 1``, 0 is padding."""

    import torch

    sequences, raw_labels = zip(*batch)
    lengths = torch.tensor([max(1, len(sequence)) for sequence in sequences], dtype=torch.long)
    width = int(lengths.max())
    tokens = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        if len(sequence):
            tokens[row, : len(sequence)] = torch.as_tensor(sequence.astype(np.int64) + 1)
    mask = torch.arange(width).unsqueeze(0) < lengths.unsqueeze(1)
    labels = torch.tensor(raw_labels, dtype=torch.long)
    return tokens, lengths, mask, labels


def _train_fold(
    sequences: Sequence[np.ndarray],
    encoded_labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    suffix_length: int,
    parameters: Mapping[str, Any],
    fold_seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    from models.Transformers_and_S4Ms import TransformerKmerClassifier

    def build_model():
        return TransformerKmerClassifier(
            vocab_size=4**suffix_length + 1,
            emb_dim=int(parameters["emb_dim"]),
            nhead=int(parameters["nhead"]),
            ff_dim=int(parameters["ff_dim"]),
            num_layers=int(parameters["num_layers"]),
            num_classes=int(parameters["n_classes"]),
            pad_id=0,
            dropout=float(parameters["dropout"]),
            positional_encoding=bool(parameters["positional_encoding"]),
        )

    return fit_sequence_fold(
        sequences,
        encoded_labels,
        train_indices,
        validation_indices,
        test_indices,
        collate_fn=collate_token_codes,
        build_model=build_model,
        parameters=parameters,
        fold_seed=fold_seed,
    )


def run_transformer_pilot(
    config: Mapping[str, Any],
    condition: Mapping[str, Any],
    split_manifest_path: str | Path,
    *,
    fold_ids: Sequence[int],
    max_tokens: int = 4096,
    epochs: int = 30,
    patience: int = 6,
    learning_rate: float = 1e-3,
    batch_size: int = 8,
    emb_dim: int = 64,
    nhead: int = 4,
    ff_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
    weight_decay: float = 1e-2,
    positional_encoding: bool = True,
    seed: int = 20260901,
    torch_threads: int = 4,
    device: str = "cpu",
) -> Path:
    """Train selected outer folds of the transformer pilot.

    ``num_layers=0`` gives an attention-free ablation (embedding mean-pool),
    which is useful to tell whether attention adds anything over token
    frequencies.  MPS is rejected: its attention kernel materialises the full
    score matrix and exhausts 16 GB of unified memory at genome lengths.
    """

    import torch

    if device != "cpu" and not (device == "cuda" and torch.cuda.is_available()):
        raise ValueError(f"requested device {device!r} is unavailable")

    parameters = {
        "architecture": "TransformerKmerClassifier",
        "n_classes": len(json.loads(str(condition["classes_json"]))),
        "max_tokens": int(max_tokens),
        "token_cap_strategy": "evenly_spaced",
        "epochs": int(epochs),
        "patience": int(patience),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "emb_dim": int(emb_dim),
        "nhead": int(nhead),
        "ff_dim": int(ff_dim),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "weight_decay": float(weight_decay),
        "gradient_clip": 1.0,
        "positional_encoding": bool(positional_encoding),
        "seed": int(seed),
        "torch_threads": int(torch_threads),
        "device": device,
        "fold_ids": [int(value) for value in fold_ids],
    }
    run_name = (
        f"tokens-{max_tokens}__layers-{num_layers}__dim-{emb_dim}__"
        f"{'' if positional_encoding else 'nopos__'}"
        f"epochs-{epochs}__lr-{learning_rate:g}__"
        f"folds-{'-'.join(map(str, fold_ids))}__seed-{seed}"
    )
    return run_sequence_pilot(
        config,
        condition,
        split_manifest_path,
        fold_ids=fold_ids,
        max_tokens=max_tokens,
        model_name="transformer",
        parameters=parameters,
        train_fold=_train_fold,
        run_name=run_name,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--total-k", required=True, type=int)
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--no-positional-encoding", action="store_true")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = json.loads(Path(args.config).read_text())
    condition = condition_from_config(
        config, prefix=args.prefix, total_k=args.total_k, seed=args.seed
    )
    output = run_transformer_pilot(
        config,
        condition,
        args.split_manifest,
        fold_ids=args.folds,
        max_tokens=args.max_tokens,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        emb_dim=args.emb_dim,
        nhead=args.nhead,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        positional_encoding=not args.no_positional_encoding,
        seed=args.seed,
        torch_threads=args.torch_threads,
        device=args.device,
    )
    print(output)


if __name__ == "__main__":
    main()
