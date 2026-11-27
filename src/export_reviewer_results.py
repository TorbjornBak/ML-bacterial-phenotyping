"""Export binary reviewer benchmark outputs as human-readable CSV and Markdown."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _result_tables(conditions_directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_tables: list[pd.DataFrame] = []
    resource_tables: list[pd.DataFrame] = []
    for path in sorted(conditions_directory.rglob("metrics.parquet")):
        table = pd.read_parquet(path)
        table.insert(0, "result_path", str(path.parent.relative_to(conditions_directory)))
        if "model_name" not in table:
            table["model_name"] = "logistic_regression"
        metric_tables.append(table)
    for path in sorted(conditions_directory.rglob("resources.parquet")):
        table = pd.read_parquet(path)
        table.insert(0, "result_path", str(path.parent.relative_to(conditions_directory)))
        if "model_name" not in table:
            table["model_name"] = "logistic_regression"
        resource_tables.append(table)
    metrics = pd.concat(metric_tables, ignore_index=True, sort=False) if metric_tables else pd.DataFrame()
    resources = (
        pd.concat(resource_tables, ignore_index=True, sort=False)
        if resource_tables
        else pd.DataFrame()
    )
    return metrics, resources


def _summary(metrics: pd.DataFrame, resources: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    identity = [
        column
        for column in (
            "result_path",
            "condition_id",
            "dataset",
            "phenotype",
            "representation",
            "total_k",
            "prefix",
            "prefix_length",
            "suffix_length",
            "prefix_source",
            "model_name",
        )
        if column in metrics
    ]
    grouped = metrics.groupby(identity, dropna=False, sort=True)
    summary = grouped.agg(
        evaluated_folds=("fold_id", "nunique"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_sd=("balanced_accuracy", "std"),
        balanced_accuracy_min=("balanced_accuracy", "min"),
        balanced_accuracy_max=("balanced_accuracy", "max"),
        roc_auc_mean=("roc_auc", "mean"),
        macro_f1_mean=("macro_f1", "mean"),
    ).reset_index()
    if resources.empty or "stage" not in resources:
        return summary
    overall = resources[resources["stage"].eq("overall")].copy()
    keep = [
        column
        for column in (
            "result_path",
            "wall_time_seconds",
            "peak_rss_bytes",
            "input_genomes",
            "retained_token_count",
            "sampled_token_count",
            "matrix_nnz",
            "serialized_size_bytes",
        )
        if column in overall
    ]
    overall = overall[keep].drop_duplicates("result_path", keep="last")
    summary = summary.merge(overall, on="result_path", how="left")
    if "peak_rss_bytes" in summary:
        summary["peak_rss_gb"] = summary["peak_rss_bytes"] / 1e9
    if "serialized_size_bytes" in summary:
        summary["serialized_size_mb"] = summary["serialized_size_bytes"] / 1e6
    return summary


def _markdown(summary: pd.DataFrame) -> str:
    title = "# Reviewer benchmark results\n\n"
    if summary.empty:
        return title + "No completed result files were found.\n"
    columns = [
        "condition_id",
        "model_name",
        "evaluated_folds",
        "balanced_accuracy_mean",
        "balanced_accuracy_sd",
        "roc_auc_mean",
        "peak_rss_gb",
        "wall_time_seconds",
    ]
    columns = [column for column in columns if column in summary]
    headers = [column.replace("_", " ") for column in columns]
    lines = [title.rstrip(), "", "| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in summary[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif column in {
                "balanced_accuracy_mean",
                "balanced_accuracy_sd",
                "roc_auc_mean",
                "peak_rss_gb",
            }:
                values.append(f"{float(value):.4f}")
            elif column == "wall_time_seconds":
                values.append(f"{float(value):.1f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Values are fold means. See `fold_metrics.csv` for individual folds and ",
            "`resources.csv` for stage-level timings and memory measurements.",
            "",
        ]
    )
    return "\n".join(lines)


def export_results(output_root: str | Path) -> Path:
    """Write readable aggregate files and return their directory."""

    root = Path(output_root)
    metrics, resources = _result_tables(root / "conditions")
    summary = _summary(metrics, resources)
    destination = root / "readable"
    destination.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(destination / "fold_metrics.csv", index=False)
    resources.to_csv(destination / "resources.csv", index=False)
    summary.to_csv(destination / "condition_summary.csv", index=False)
    (destination / "README.md").write_text(_markdown(summary))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", nargs="?", default="results/reviewer_benchmark")
    args = parser.parse_args()
    print(export_results(args.output_root))


if __name__ == "__main__":
    main()
