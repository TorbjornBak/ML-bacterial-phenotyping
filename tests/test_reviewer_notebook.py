import json
from pathlib import Path

import pandas as pd


def test_notebook_loader_uses_explicit_filenames_and_ignores_exports(tmp_path, monkeypatch):
    output = tmp_path / "reviewer_benchmark"
    condition = output / "conditions" / "toy"
    readable = output / "readable"
    condition.mkdir(parents=True)
    readable.mkdir()
    pd.DataFrame(
        {"condition_id": ["toy"], "fold_id": [0], "balanced_accuracy": [0.5]}
    ).to_parquet(condition / "metrics.parquet", index=False)
    pd.DataFrame(
        {"condition_id": ["toy"], "phenotype": ["trait"], "representation": ["full"], "total_k": [5]}
    ).to_csv(output / "conditions.csv", index=False)
    pd.DataFrame({"prefix": ["ATG"]}).to_csv(output / "prefix_manifest.csv", index=False)
    pd.DataFrame(
        {"condition_id": ["duplicate"], "fold_id": [0], "balanced_accuracy": [1.0]}
    ).to_csv(readable / "fold_metrics.csv", index=False)
    (condition / "toy_prefix_metadata.json").write_text('{"prefix": "ATG"}')

    monkeypatch.setenv("REVIEWER_BENCHMARK_OUTPUT", str(output))
    notebook_path = Path(__file__).parents[1] / "notebooks" / "reviewer_benchmark.ipynb"
    notebook = json.loads(notebook_path.read_text())
    namespace = {}
    exec("".join(notebook["cells"][1]["source"]), namespace)
    exec("".join(notebook["cells"][3]["source"]), namespace)

    assert namespace["metrics"]["condition_id"].tolist() == ["toy"]
    assert namespace["conditions"]["condition_id"].tolist() == ["toy"]
    assert namespace["prefix_manifest"]["prefix"].tolist() == ["ATG"]


def test_notebook_has_completed_run_overview():
    notebook_path = Path(__file__).parents[1] / "notebooks" / "reviewer_benchmark.ipynb"
    notebook = json.loads(notebook_path.read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "Overview of every completed test run" in source
    assert "completed_run_overview" in source
    assert '"run_overview.csv": run_overview' in source
