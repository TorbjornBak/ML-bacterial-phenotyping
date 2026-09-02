import numpy as np
import pandas as pd
import pytest

from utilities.benchmark_metrics import (
    binary_metrics,
    chance_balanced_accuracy,
    fold_confidence_interval,
    load_resfinder_predictions,
    paired_difference_confidence_interval,
    prediction_metrics,
)


def test_prediction_metrics_reports_binary_metrics_and_auc():
    result = prediction_metrics(
        [0, 0, 1, 1], [0, 1, 1, 1], scores=[0.1, 0.7, 0.8, 0.9], positive_label=1
    )
    assert result["balanced_accuracy"] == pytest.approx(0.75)
    assert result["sensitivity"] == pytest.approx(1.0)
    assert result["specificity"] == pytest.approx(0.5)
    assert result["macro_f1"] == pytest.approx(0.7333333333)
    assert result["roc_auc"] == pytest.approx(1.0)


def test_prediction_metrics_auc_follows_explicit_positive_label_for_probability_columns():
    # Probability columns follow the explicit labels order [1, 0], so the
    # second column is the positive class.  AUC must use that class rather
    # than relying on sklearn's default numeric label ordering.
    result = prediction_metrics(
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        scores=[[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]],
        labels=[1, 0],
        positive_label=0,
    )
    assert result["roc_auc"] == pytest.approx(1.0)


def test_prediction_metrics_accepts_single_binary_probability_column():
    result = prediction_metrics(
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        scores=[[0.1], [0.2], [0.8], [0.9]],
        positive_label=1,
    )
    assert result["roc_auc"] == pytest.approx(1.0)


def test_binary_orientation_is_explicit_and_validated():
    assert binary_metrics(["susceptible", "resistant"], ["resistant", "resistant"], positive_label="resistant")["sensitivity"] == 1
    with pytest.raises(ValueError, match="positive_label"):
        binary_metrics([0, 1], [0, 1], positive_label="resistant")


def test_chance_balanced_accuracy_and_reproducible_fold_interval():
    assert chance_balanced_accuracy(4) == pytest.approx(0.25)
    assert fold_confidence_interval([0.5, 0.6, 0.7, 0.8], seed=17) == fold_confidence_interval([0.5, 0.6, 0.7, 0.8], seed=17)


def test_metrics_reject_empty_predictions_and_invalid_class_count():
    with pytest.raises(ValueError, match="non-empty"):
        prediction_metrics([], [])
    with pytest.raises(ValueError, match="positive integer"):
        chance_balanced_accuracy(0)


def test_paired_interval_aligns_by_genome_id_not_row_order():
    full = pd.DataFrame({"genome_id": ["b", "a", "c"], "score": [0.6, 0.8, 0.7]})
    filtered = pd.DataFrame({"genome_id": ["a", "c", "b"], "score": [0.7, 0.6, 0.5]})
    result = paired_difference_confidence_interval(full, filtered, id_column="genome_id", value_column="score", seed=3)
    assert result["n"] == 3
    assert result["estimate"] == pytest.approx(0.1)


def test_resfinder_csv_is_optional_and_schema_is_checked(tmp_path):
    path = tmp_path / "resfinder.csv"
    pd.DataFrame({"genome_id": ["g1"], "true_label": [1], "predicted_label": [1], "score": [0.9]}).to_csv(path, index=False)
    loaded = load_resfinder_predictions(path)
    assert list(loaded.columns) == ["genome_id", "true_label", "predicted_label", "score"]
    assert load_resfinder_predictions(None) is None
