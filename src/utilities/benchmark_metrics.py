"""Metrics and uncertainty helpers for the reviewer benchmark.

Functions in this module operate on out-of-fold predictions.  Confidence
intervals use a seeded bootstrap so summaries can be regenerated exactly.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    recall_score,
)


def chance_balanced_accuracy(n_classes: int) -> float:
    """Return the balanced-accuracy chance reference for ``n_classes``."""

    if isinstance(n_classes, bool) or int(n_classes) != n_classes or n_classes < 1:
        raise ValueError("n_classes must be a positive integer")
    return 1.0 / int(n_classes)


def _as_array(values: Any, *, name: str) -> np.ndarray:
    if isinstance(values, pd.Series):
        values = values.to_numpy()
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _classes(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[Any] | None) -> list[Any]:
    if labels is not None:
        classes = list(labels)
        if len(set(classes)) == 0:
            raise ValueError("labels must not be empty")
        unknown = set(np.concatenate([y_true, y_pred])) - set(classes)
        if unknown:
            raise ValueError(f"predictions contain labels not present in labels: {unknown}")
        return classes
    # np.unique is deterministic for numbers and strings and preserves the
    # order expected by sklearn's metric implementations.
    return list(np.unique(np.concatenate([y_true, y_pred])))


def binary_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    positive_label: Any,
    labels: Sequence[Any] | None = None,
) -> dict[str, float | int | Any]:
    """Compute sensitivity/specificity with an explicit positive class.

    Requiring ``positive_label`` prevents a string or encoded-label ordering
    change from silently reversing gentamicin sensitivity and specificity.
    """

    true = _as_array(y_true, name="y_true")
    pred = _as_array(y_pred, name="y_pred")
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have equal length")
    classes = _classes(true, pred, labels)
    if len(classes) != 2 or positive_label not in classes:
        raise ValueError("positive_label must identify one of exactly two classes")
    negative = next(label for label in classes if label != positive_label)
    matrix = confusion_matrix(true, pred, labels=[negative, positive_label])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "positive_label": positive_label,
        "negative_label": negative,
        "sensitivity": float(recall_score(true, pred, pos_label=positive_label, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def prediction_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    scores: Sequence[Any] | np.ndarray | None = None,
    *,
    y_score: Sequence[Any] | np.ndarray | None = None,
    labels: Sequence[Any] | None = None,
    positive_label: Any | None = None,
) -> dict[str, Any]:
    """Compute primary and secondary metrics for one prediction collection."""

    if scores is not None and y_score is not None:
        raise ValueError("pass either scores or y_score, not both")
    score_values = scores if scores is not None else y_score
    true = _as_array(y_true, name="y_true")
    pred = _as_array(y_pred, name="y_pred")
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have equal length")
    classes = _classes(true, pred, labels)
    result: dict[str, Any] = {
        "n_samples": int(len(true)),
        "n_classes": len(classes),
        "classes": classes,
        "chance_balanced_accuracy": chance_balanced_accuracy(len(classes)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, labels=classes, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(true, pred, labels=classes).tolist(),
        "roc_auc": None,
    }
    if len(classes) == 2:
        if positive_label is not None:
            result.update(binary_metrics(true, pred, positive_label=positive_label, labels=classes))
        elif "sensitivity" not in result:
            # Deliberately do not guess orientation.  The generic metrics are
            # still useful, while disease-specific binary metrics need an
            # explicit class declaration.
            result["orientation_validated"] = False
        if score_values is not None:
            score_array = np.asarray(score_values)
            if score_array.ndim == 0 or score_array.shape[0] != len(true):
                raise ValueError("scores must have one row per prediction")
            if score_array.ndim == 2:
                if score_array.shape[1] != 2:
                    raise ValueError("binary scores must have one or two columns")
                # sklearn's predict_proba follows ``classes``.  Respect an
                # explicitly supplied positive class even if labels are not
                # in lexical order.
                score_class = positive_label if positive_label is not None else classes[-1]
                score_array = score_array[:, classes.index(score_class)]
            try:
                result["roc_auc"] = float(roc_auc_score(true, score_array))
            except ValueError:
                result["roc_auc"] = None
    elif score_values is not None:
        score_array = np.asarray(score_values)
        if score_array.ndim == 0 or score_array.shape[0] != len(true):
            raise ValueError("scores must have one row per prediction")
        if score_array.ndim == 2 and score_array.shape == (len(true), len(classes)):
            try:
                result["roc_auc"] = float(roc_auc_score(true, score_array, labels=classes, multi_class="ovr", average="macro"))
            except ValueError:
                result["roc_auc"] = None
    return result


def _values(values: Sequence[float] | pd.DataFrame, value_column: str | None) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        if value_column is None:
            raise ValueError("value_column is required for a DataFrame")
        if value_column not in values:
            raise ValueError(f"missing value column: {value_column}")
        values = values[value_column]
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(result)):
        raise ValueError("values must be finite")
    return result


def _bootstrap_interval(values: np.ndarray, *, confidence: float, seed: int, n_bootstrap: int) -> dict[str, Any]:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    rng = np.random.default_rng(seed)
    # Keep memory bounded for out-of-fold predictions containing many
    # genomes.  A single 10,000 x N index matrix is surprisingly expensive.
    means = np.empty(n_bootstrap, dtype=float)
    cursor = 0
    while cursor < n_bootstrap:
        batch_size = min(1024, n_bootstrap - cursor)
        samples = rng.integers(0, len(values), size=(batch_size, len(values)))
        means[cursor : cursor + batch_size] = values[samples].mean(axis=1)
        cursor += batch_size
    tail = (1.0 - confidence) / 2.0 * 100.0
    lower, upper = np.percentile(means, [tail, 100.0 - tail])
    return {
        "estimate": float(values.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "confidence": float(confidence),
        "n": int(len(values)),
        "seed": int(seed),
        "method": "percentile_bootstrap",
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


def fold_confidence_interval(
    fold_scores: Sequence[float] | pd.DataFrame,
    *,
    value_column: str | None = None,
    confidence: float = 0.95,
    seed: int = 0,
    n_bootstrap: int = 10_000,
) -> dict[str, Any]:
    """Return a reproducible interval for fold-level scores."""

    return _bootstrap_interval(_values(fold_scores, value_column), confidence=confidence, seed=seed, n_bootstrap=n_bootstrap)


def paired_difference_confidence_interval(
    full: Sequence[float] | pd.DataFrame,
    filtered: Sequence[float] | pd.DataFrame,
    *,
    id_column: str | None = None,
    id_col: str | None = None,
    value_column: str | None = None,
    confidence: float = 0.95,
    seed: int = 0,
    n_bootstrap: int = 10_000,
) -> dict[str, Any]:
    """Compute full-minus-filtered CI after ID-safe alignment.

    DataFrames are merged by ``id_column`` and checked for duplicate/missing
    IDs.  Sequence inputs represent already paired fold order.
    """

    if isinstance(full, pd.DataFrame) or isinstance(filtered, pd.DataFrame):
        if not isinstance(full, pd.DataFrame) or not isinstance(filtered, pd.DataFrame):
            raise ValueError("both paired inputs must be DataFrames with id_column")
        id_column = id_column or id_col
        if id_column is None:
            for candidate in ("genome_id", "fold_id", "fold", "id"):
                if candidate in full.columns and candidate in filtered.columns:
                    id_column = candidate
                    break
        if not id_column:
            raise ValueError("both paired inputs must be DataFrames with id_column")
        if value_column is None:
            raise ValueError("value_column is required for DataFrame inputs")
        for frame, label in ((full, "full"), (filtered, "filtered")):
            if id_column not in frame or value_column not in frame:
                raise ValueError(f"{label} is missing {id_column!r} or {value_column!r}")
            if frame[id_column].duplicated().any():
                raise ValueError(f"{label} contains duplicate IDs")
        left = full[[id_column, value_column]].rename(columns={value_column: "_full"})
        right = filtered[[id_column, value_column]].rename(columns={value_column: "_filtered"})
        aligned = left.merge(right, on=id_column, how="inner", validate="one_to_one")
        if len(aligned) != len(left) or len(aligned) != len(right):
            raise ValueError("paired inputs must contain the same IDs")
        differences = aligned["_full"].to_numpy(dtype=float) - aligned["_filtered"].to_numpy(dtype=float)
    else:
        left = _values(full, None)
        right = _values(filtered, None)
        if len(left) != len(right):
            raise ValueError("paired sequences must have equal length")
        differences = left - right
    result = _bootstrap_interval(differences, confidence=confidence, seed=seed, n_bootstrap=n_bootstrap)
    result["comparison"] = "full_minus_filtered"
    return result


RESFINDER_COLUMNS = ("genome_id", "true_label", "predicted_label")


def load_resfinder_predictions(path: str | Path | None) -> pd.DataFrame | None:
    """Load optional ResFinder predictions using the documented CSV schema.

    ``None`` and missing paths return ``None`` so the core benchmark remains
    runnable when no external baseline has been produced.
    """

    if path is None or not Path(path).exists():
        return None
    frame = pd.read_csv(path)
    missing = [column for column in RESFINDER_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"ResFinder CSV missing required columns: {missing}")
    columns = list(RESFINDER_COLUMNS) + (["score"] if "score" in frame.columns else [])
    result = frame[columns].copy()
    if result["genome_id"].duplicated().any():
        raise ValueError("ResFinder CSV contains duplicate genome_id values")
    return result
