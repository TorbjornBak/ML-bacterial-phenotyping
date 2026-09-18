"""Small, reproducible RNN pilots for prefix-filtered reviewer conditions.

The main reviewer benchmark uses unordered sparse k-mer counts.  This runner
keeps the order of retained suffix tokens and trains the repository's small
one-hot GRU on the same frozen outer folds.  It is intentionally a pilot:
long sequences are deterministically thinned to an explicit token cap and a
caller may select only a subset of folds.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def extract_prefix_suffix_codes(
    genome: Any,
    *,
    total_k: int,
    prefix: str,
    max_tokens: int | None,
) -> tuple[np.ndarray, int]:
    """Return ordered base-four suffix codes and the uncapped token count."""

    from embeddings.sparse_kmer_counts import (
        _DNA_TO_DIGIT,
        _VALID_DNA_RUN,
        _as_contigs,
        encode_kmer,
    )

    prefix = prefix.upper()
    if not prefix or any(base not in _DNA_TO_DIGIT for base in prefix):
        raise ValueError("prefix must contain only A, C, G, and T")
    if total_k <= len(prefix):
        raise ValueError("total_k must be larger than the prefix length")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive or None")

    suffix_length = total_k - len(prefix)
    codes: list[int] = []
    for contig in _as_contigs(genome):
        for match in _VALID_DNA_RUN.finditer(contig):
            run = match.group(0)
            start = run.find(prefix)
            while start >= 0:
                suffix_start = start + len(prefix)
                suffix_end = start + total_k
                if suffix_end <= len(run):
                    codes.append(encode_kmer(run[suffix_start:suffix_end]))
                start = run.find(prefix, start + 1)

    raw_count = len(codes)
    if max_tokens is not None and raw_count > max_tokens:
        indices = np.linspace(0, raw_count - 1, num=max_tokens, dtype=np.int64)
        result = np.asarray(codes, dtype=np.uint32)[indices]
    else:
        result = np.asarray(codes, dtype=np.uint32)
    return result, raw_count


def collate_suffix_codes(
    batch: Sequence[tuple[np.ndarray, int]], *, suffix_length: int
):
    """Pad compact suffix codes and expand them to flattened base one-hot."""

    import torch
    import torch.nn.functional as functional

    if suffix_length <= 0:
        raise ValueError("suffix_length must be positive")
    sequences, raw_labels = zip(*batch)
    lengths = torch.tensor([max(1, len(sequence)) for sequence in sequences], dtype=torch.long)
    width = int(lengths.max())
    padded = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        if len(sequence):
            padded[row, : len(sequence)] = torch.as_tensor(sequence.astype(np.int64))
    mask = torch.arange(width).unsqueeze(0) < lengths.unsqueeze(1)
    powers = torch.tensor(
        [4**position for position in range(suffix_length - 1, -1, -1)],
        dtype=torch.long,
    )
    digits = (padded.unsqueeze(-1) // powers) % 4
    inputs = functional.one_hot(digits, num_classes=4).to(torch.float32)
    inputs = inputs.reshape(len(sequences), width, suffix_length * 4)
    inputs[~mask] = 0
    labels = torch.tensor(raw_labels, dtype=torch.long)
    return inputs.contiguous(), lengths, mask, labels


class _CodeDataset:
    def __init__(self, sequences: Sequence[np.ndarray], labels: np.ndarray, indices: np.ndarray):
        self.sequences = sequences
        self.labels = labels
        self.indices = np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[np.ndarray, int]:
        index = int(self.indices[item])
        return self.sequences[index], int(self.labels[index])


def _loader(
    sequences: Sequence[np.ndarray],
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    collate_fn: Callable,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        _CodeDataset(sequences, labels, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fn,
        generator=generator,
    )


def fit_sequence_fold(
    sequences: Sequence[np.ndarray],
    encoded_labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    collate_fn: Callable,
    build_model: Callable[[], Any],
    parameters: Mapping[str, Any],
    fold_seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Train one fold with class-weighted loss and validation early stopping.

    ``collate_fn`` turns a list of ``(codes, label)`` pairs into
    ``(inputs, lengths, mask, labels)`` and ``build_model`` returns an
    untrained module called as ``model(inputs, lengths, mask=mask)``.
    Returns test-set class probabilities, the test indices, and epochs run.
    """

    import torch

    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)
    torch.set_num_threads(int(parameters["torch_threads"]))
    device = torch.device(str(parameters["device"]))
    loaders = [
        _loader(
            sequences,
            encoded_labels,
            indices,
            collate_fn=collate_fn,
            batch_size=int(parameters["batch_size"]),
            shuffle=shuffle,
            seed=fold_seed,
        )
        for indices, shuffle in (
            (train_indices, True),
            (validation_indices, False),
            (test_indices, False),
        )
    ]
    train_loader, validation_loader, test_loader = loaders
    model = build_model().to(device)
    gradient_clip = parameters.get("gradient_clip")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    counts = np.bincount(
        encoded_labels[train_indices], minlength=int(parameters["n_classes"])
    ).astype(np.float32)
    counts[counts == 0] = 1
    weights = len(train_indices) / (int(parameters["n_classes"]) * counts)
    loss_function = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    epochs_run = 0
    for epoch in range(int(parameters["epochs"])):
        model.train()
        for inputs, lengths, mask, labels in train_loader:
            optimizer.zero_grad()
            logits = model(
                inputs.to(device), lengths.to(device), mask=mask.to(device)
            )
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
            optimizer.step()

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for inputs, lengths, mask, labels in validation_loader:
                logits = model(
                    inputs.to(device), lengths.to(device), mask=mask.to(device)
                )
                validation_loss += float(loss_function(logits, labels.to(device)))
        validation_loss /= max(1, len(validation_loader))
        epochs_run = epoch + 1
        print(f"epoch {epochs_run}: validation loss {validation_loss:.4f}", flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(parameters["patience"]):
                break

    model.load_state_dict(best_state)
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, lengths, mask, _ in test_loader:
            logits = model(inputs.to(device), lengths.to(device), mask=mask.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities), test_indices, epochs_run


TrainFold = Callable[..., tuple[np.ndarray, np.ndarray, int]]


def run_sequence_pilot(
    config: Mapping[str, Any],
    condition: Mapping[str, Any],
    split_manifest_path: str | Path,
    *,
    fold_ids: Sequence[int],
    max_tokens: int | None,
    model_name: str,
    parameters: Mapping[str, Any],
    train_fold: TrainFold,
    run_name: str,
) -> Path:
    """Tokenize one condition in order, train selected outer folds, save outputs.

    ``train_fold`` is called as ``train_fold(sequences, encoded_labels,
    train_indices, validation_indices, test_indices, suffix_length=...,
    parameters=..., fold_seed=...)`` and returns test probabilities, test
    indices, and epochs run.  Outputs are written below
    ``conditions/<condition_id>/models/<model_name>/<run_name>``.
    """

    from reviewer_benchmark import _open_sequence_source, _read_table, _validate_unique_labels
    from utilities.benchmark_design import read_split_manifest
    from utilities.benchmark_metrics import prediction_metrics
    from utilities.resource_monitoring import ResourceMonitor, measure_stage

    if not fold_ids:
        raise ValueError("at least one fold_id is required")

    condition = dict(condition)
    condition_id = str(condition["condition_id"])
    prefix = str(condition["prefix"])
    total_k = int(condition["total_k"])
    suffix_length = total_k - len(prefix)
    seed = int(parameters["seed"])
    gpu_enabled = parameters["device"] == "cuda"
    dataset = config["datasets"][str(condition["dataset"])]
    monitor = ResourceMonitor(condition_id=condition_id, gpu_enabled=gpu_enabled)
    overall = measure_stage(
        "overall", condition_id, gpu_enabled=gpu_enabled, cache_state="cold"
    )
    with overall:
        with monitor.stage("loading", cache_state="cold"):
            source = _open_sequence_source(dataset)
            labels = _read_table(dataset["labels"], dataset.get("label_separator", ","))
            id_column = dataset["id_column"]
            label_column = str(condition["label_column"])
            labels = _validate_unique_labels(
                labels[[id_column, label_column]].dropna().copy(),
                id_column,
                label_column,
            )
            labels[id_column] = labels[id_column].astype(str)
            split_manifest = read_split_manifest(split_manifest_path)
            split_manifest["genome_id"] = split_manifest["genome_id"].astype(str)
            split_manifest = split_manifest[
                split_manifest["dataset"].astype(str) == str(condition["dataset"])
            ]
            usable_ids = [
                genome_id
                for genome_id in source
                if genome_id in set(labels[id_column])
                and genome_id in set(split_manifest["genome_id"])
            ]
            label_by_id = labels.set_index(id_column)[label_column]
            fold_by_id = split_manifest.set_index("genome_id")["fold_id"]
            raw_labels = label_by_id.loc[usable_ids].to_numpy()
            folds = fold_by_id.loc[usable_ids].to_numpy(dtype=int)

        with monitor.stage("ordered_tokenization", cache_state="cold"):
            sequences: list[np.ndarray] = []
            raw_token_counts: list[int] = []
            for genome_id in usable_ids:
                codes, raw_count = extract_prefix_suffix_codes(
                    source[genome_id],
                    total_k=total_k,
                    prefix=prefix,
                    max_tokens=max_tokens,
                )
                sequences.append(codes)
                raw_token_counts.append(raw_count)

        classes = json.loads(str(condition["classes_json"]))
        class_to_index = {value: index for index, value in enumerate(classes)}
        encoded_labels = np.asarray([class_to_index[value] for value in raw_labels])
        available_folds = sorted(np.unique(folds).tolist())
        requested_folds = [int(value) for value in fold_ids]
        if not set(requested_folds).issubset(available_folds):
            raise ValueError("requested fold_ids are not present in the split manifest")

        metric_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        for fold_id in requested_folds:
            fold_position = available_folds.index(fold_id)
            validation_fold = available_folds[(fold_position + 1) % len(available_folds)]
            test_indices = np.flatnonzero(folds == fold_id)
            validation_indices = np.flatnonzero(folds == validation_fold)
            train_indices = np.flatnonzero((folds != fold_id) & (folds != validation_fold))
            with monitor.stage("fitting", cache_state="cold", fold_id=fold_id):
                probabilities, ordered_test_indices, epochs_run = train_fold(
                    sequences,
                    encoded_labels,
                    train_indices,
                    validation_indices,
                    test_indices,
                    suffix_length=suffix_length,
                    parameters=parameters,
                    fold_seed=seed + fold_id,
                )
            predicted_indices = probabilities.argmax(axis=1)
            predicted = np.asarray([classes[index] for index in predicted_indices])
            truth = raw_labels[ordered_test_indices]
            positive_class = condition.get("positive_class")
            positive_index = classes.index(positive_class)
            fold_metrics = prediction_metrics(
                truth,
                predicted,
                scores=probabilities[:, positive_index],
                labels=classes,
                positive_label=positive_class,
            )
            fold_metrics["classes"] = json.dumps(fold_metrics["classes"])
            fold_metrics["confusion_matrix"] = json.dumps(fold_metrics["confusion_matrix"])
            metric_rows.append(
                {
                    **condition,
                    "model_name": model_name,
                    "fold_id": fold_id,
                    "validation_fold_id": validation_fold,
                    "epochs_run": epochs_run,
                    **fold_metrics,
                }
            )
            for row, genome_index in enumerate(ordered_test_indices):
                prediction_rows.append(
                    {
                        **condition,
                        "model_name": model_name,
                        "fold_id": fold_id,
                        "genome_id": usable_ids[genome_index],
                        "true_label": truth[row],
                        "predicted_label": predicted[row],
                        "score": float(probabilities[row, positive_index]),
                    }
                )

    output = (
        Path(config["output_dir"])
        / "conditions"
        / condition_id
        / "models"
        / model_name
        / run_name
    )
    output.mkdir(parents=True, exist_ok=True)
    resource_common = {
        **condition,
        "model_name": model_name,
        "input_genomes": len(usable_ids),
        "retained_token_count": int(sum(raw_token_counts)),
        "sampled_token_count": int(sum(map(len, sequences))),
        "max_tokens": max_tokens,
    }
    resource_rows = [
        {**resource_common, **record} for record in [*monitor.records, overall.record]
    ]
    pd.DataFrame(metric_rows).to_parquet(output / "metrics.parquet", index=False)
    pd.DataFrame(prediction_rows).to_parquet(output / "predictions.parquet", index=False)
    pd.DataFrame(resource_rows).to_parquet(output / "resources.parquet", index=False)
    (output / "condition.json").write_text(
        json.dumps(
            {**condition, "model_name": model_name, "model_parameters": dict(parameters)},
            indent=2,
        )
        + "\n"
    )
    return output


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
    from models.RNN import OneHot_RNN_small

    def build_model():
        return OneHot_RNN_small(
            vocab_size=suffix_length * 4,
            rnn_hidden=int(parameters["hidden_size"]),
            head_dim=int(parameters["head_dim"]),
            num_layers=1,
            bidirectional=False,
            num_classes=int(parameters["n_classes"]),
            dropout=float(parameters["dropout"]),
            input_dropout=float(parameters["input_dropout"]),
        )

    return fit_sequence_fold(
        sequences,
        encoded_labels,
        train_indices,
        validation_indices,
        test_indices,
        collate_fn=partial(collate_suffix_codes, suffix_length=suffix_length),
        build_model=build_model,
        parameters=parameters,
        fold_seed=fold_seed,
    )


def run_rnn_pilot(
    config: Mapping[str, Any],
    condition: Mapping[str, Any],
    split_manifest_path: str | Path,
    *,
    fold_ids: Sequence[int],
    max_tokens: int = 512,
    epochs: int = 5,
    patience: int = 2,
    learning_rate: float = 1e-3,
    batch_size: int = 16,
    hidden_size: int = 64,
    head_dim: int = 32,
    seed: int = 20260901,
    torch_threads: int = 4,
    device: str = "cpu",
) -> Path:
    """Train selected outer folds of the small one-hot RNN pilot."""

    import torch

    if device != "cpu" and not (
        device == "cuda" and torch.cuda.is_available()
    ):
        raise ValueError(f"requested device {device!r} is unavailable")

    parameters = {
        "architecture": "OneHot_RNN_small",
        "n_classes": len(json.loads(str(condition["classes_json"]))),
        "max_tokens": int(max_tokens),
        "token_cap_strategy": "evenly_spaced",
        "epochs": int(epochs),
        "patience": int(patience),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "hidden_size": int(hidden_size),
        "head_dim": int(head_dim),
        "dropout": 0.2,
        "input_dropout": 0.1,
        "weight_decay": 1e-4,
        "seed": int(seed),
        "torch_threads": int(torch_threads),
        "device": device,
        "fold_ids": [int(value) for value in fold_ids],
    }
    run_name = (
        f"tokens-{max_tokens}__epochs-{epochs}__lr-{learning_rate:g}__"
        f"folds-{'-'.join(map(str, fold_ids))}__seed-{seed}"
    )
    return run_sequence_pilot(
        config,
        condition,
        split_manifest_path,
        fold_ids=fold_ids,
        max_tokens=max_tokens,
        model_name="small_rnn",
        parameters=parameters,
        train_fold=_train_fold,
        run_name=run_name,
    )


def condition_from_config(
    config: Mapping[str, Any], *, prefix: str, total_k: int, seed: int
) -> dict[str, Any]:
    """Describe a prefix-filtered condition for the config's first phenotype."""

    phenotype = config["phenotypes"][0]
    prefix = prefix.upper()
    return {
        "condition_id": (
            f"{phenotype['dataset']}__{phenotype['name'].replace('_', '-')}__"
            f"k{total_k}__prefix-{prefix}"
        ),
        "dataset": phenotype["dataset"],
        "phenotype": phenotype["name"],
        "label_column": phenotype["label_column"],
        "positive_class": phenotype["positive_class"],
        "n_classes": len(phenotype["classes"]),
        "classes_json": json.dumps(phenotype["classes"]),
        "representation": "prefix_filtered",
        "total_k": total_k,
        "prefix": prefix,
        "prefix_length": len(prefix),
        "suffix_length": total_k - len(prefix),
        "prefix_source": "historical",
        "seed": seed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--total-k", required=True, type=int)
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = json.loads(Path(args.config).read_text())
    condition = condition_from_config(
        config, prefix=args.prefix, total_k=args.total_k, seed=args.seed
    )
    output = run_rnn_pilot(
        config,
        condition,
        args.split_manifest,
        fold_ids=args.folds,
        max_tokens=args.max_tokens,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        head_dim=args.head_dim,
        seed=args.seed,
        torch_threads=args.torch_threads,
    )
    print(output)


if __name__ == "__main__":
    main()
