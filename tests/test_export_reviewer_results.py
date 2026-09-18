import pandas as pd

from export_reviewer_results import export_results


def test_export_results_writes_csv_and_markdown_views(tmp_path):
    result = tmp_path / "conditions" / "toy__trait__k5__prefix-ATG"
    result.mkdir(parents=True)
    pd.DataFrame(
        {
            "condition_id": ["toy", "toy"],
            "dataset": ["toy", "toy"],
            "phenotype": ["trait", "trait"],
            "representation": ["prefix_filtered", "prefix_filtered"],
            "total_k": [5, 5],
            "prefix": ["ATG", "ATG"],
            "prefix_length": [3, 3],
            "suffix_length": [2, 2],
            "prefix_source": ["historical", "historical"],
            "model_name": ["small_rnn", "small_rnn"],
            "fold_id": [0, 1],
            "balanced_accuracy": [0.5, 0.75],
            "roc_auc": [0.6, 0.8],
            "macro_f1": [0.4, 0.7],
        }
    ).to_parquet(result / "metrics.parquet", index=False)
    pd.DataFrame(
        {
            "condition_id": ["toy"],
            "model_name": ["small_rnn"],
            "stage": ["overall"],
            "wall_time_seconds": [12.5],
            "peak_rss_bytes": [2_000_000_000],
        }
    ).to_parquet(result / "resources.parquet", index=False)

    destination = export_results(tmp_path)

    summary = pd.read_csv(destination / "condition_summary.csv")
    assert summary["balanced_accuracy_mean"].tolist() == [0.625]
    assert summary["evaluated_folds"].tolist() == [2]
    assert summary["peak_rss_gb"].tolist() == [2.0]
    assert (destination / "fold_metrics.csv").exists()
    assert (destination / "resources.csv").exists()
    assert "Reviewer benchmark results" in (destination / "README.md").read_text()
