"""Command-line runner for the reviewer full-versus-filtered k-mer benchmark."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import traceback
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def _slug(value: object) -> str:
    return "".join(character.lower() if str(character).isalnum() else "-" for character in str(value)).strip("-")


def _code_version() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_scalar(value: Any) -> Any:
    """Convert manifest scalars to strict JSON values without changing types."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def build_condition_manifest(config: Mapping[str, Any], pilot: bool = False) -> pd.DataFrame:
    """Expand configuration into immutable, independently runnable conditions."""
    from utilities.benchmark_design import create_prefix_manifest, get_phenotype_metadata

    phenotypes = list(config["phenotypes"])
    total_k_values = [int(value) for value in config["total_k"]]
    prefix_lengths = [int(value) for value in config["prefix_lengths"]]
    n_prefixes = int(config.get("n_prefixes", 10))

    if pilot:
        phenotypes = phenotypes[:1]
        total_k_values = total_k_values[:1]
        if 5 not in prefix_lengths:
            raise ValueError("Pilot mode requires prefix length 5")
        prefix_lengths = [5]
        n_prefixes = 2

    prefix_manifest = create_prefix_manifest(
        total_k_values=total_k_values,
        prefix_lengths=prefix_lengths,
        n_prefixes=n_prefixes,
        seed=int(config["prefix_seed"]),
        historical_prefix=config.get("historical_prefix"),
    )

    rows: list[dict[str, Any]] = []
    for phenotype in phenotypes:
        if "classes" not in phenotype:
            raise ValueError(
                f"Phenotype {phenotype['name']!r} must declare classes so class orientation is reproducible"
            )
        definition = {
            "classes": phenotype["classes"],
            "n_classes": phenotype.get("n_classes", len(phenotype["classes"])),
            "positive_class": phenotype.get("positive_class"),
        }
        metadata = get_phenotype_metadata(
            phenotype["name"], {phenotype["name"]: definition}
        )
        common = {
            "dataset": phenotype["dataset"],
            "phenotype": metadata["name"],
            "label_column": phenotype.get("label_column", phenotype["name"]),
            "positive_class": metadata["positive_class"],
            "n_classes": metadata["n_classes"],
            "classes_json": json.dumps(metadata["classes"], default=str),
        }
        for total_k in total_k_values:
            rows.append(
                {
                    **common,
                    "representation": "full",
                    "total_k": total_k,
                    "prefix": None,
                    "prefix_length": 0,
                    "suffix_length": total_k,
                    "prefix_source": "full",
                }
            )
        for prefix_row in prefix_manifest.to_dict(orient="records"):
            rows.append(
                {
                    "representation": "prefix_filtered",
                    **prefix_row,
                    **common,
                }
            )

    manifest = pd.DataFrame(rows)
    manifest.insert(
        0,
        "condition_id",
        [
            "__".join(
                [
                    _slug(row.dataset),
                    _slug(row.phenotype),
                    f"k{row.total_k}",
                    "full" if row.representation == "full" else f"prefix-{row.prefix}",
                ]
            )
            for row in manifest.itertuples(index=False)
        ],
    )
    if not manifest["condition_id"].is_unique:
        raise ValueError("Condition manifest contains duplicate conditions")
    return manifest


def write_failure_record(
    output_directory: str | Path,
    condition: Mapping[str, Any] | pd.Series,
    error: BaseException,
) -> Path:
    """Persist a failed condition instead of silently skipping it."""
    output_directory = Path(output_directory)
    failure_directory = output_directory / "failures"
    failure_directory.mkdir(parents=True, exist_ok=True)
    condition_id = str(condition["condition_id"])
    path = failure_directory / f"{condition_id}.json"
    record = {
        "condition_id": condition_id,
        "error_type": type(error).__name__,
        "message": str(error),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "traceback": "".join(traceback.format_exception(error)),
    }
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def _configured_path_base(config: Mapping[str, Any], config_path: Path) -> Path:
    base = Path(config.get("path_base", REPOSITORY_ROOT)).expanduser()
    if not base.is_absolute():
        base = config_path.resolve().parent / base
    return base.resolve()


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _resolve_dataset_paths(
    datasets: Mapping[str, Mapping[str, Any]], base: Path
) -> dict[str, dict[str, Any]]:
    resolved: dict[str, dict[str, Any]] = {}
    for name, dataset in datasets.items():
        item = dict(dataset)
        for field in ("input", "labels"):
            if item.get(field) is not None:
                item[field] = str(_resolve_path(item[field], base))
        resolved[name] = item
    return resolved


def prepare(config_path: str | Path, output_directory: str | Path | None, pilot: bool) -> Path:
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    # A prepared run is commonly submitted later from a different login
    # directory.  Freeze the output location now so the resolved config and
    # every array task agree on where artifacts belong.
    configured_base = _configured_path_base(config, config_path)
    destination = _resolve_path(output_directory or config["output_dir"], configured_base)
    destination.mkdir(parents=True, exist_ok=True)

    manifest = build_condition_manifest(config, pilot=pilot)
    manifest_path = destination / ("conditions.pilot.csv" if pilot else "conditions.csv")
    manifest.to_csv(manifest_path, index=False)

    # Keep the prefix draw as a first-class input to the experiment.  The
    # condition manifest repeats these rows once per phenotype, which is
    # convenient for arrays but is not a useful provenance artifact by
    # itself.  Deduplicating here gives analysts and cluster jobs one stable
    # manifest to archive and inspect before any feature extraction starts.
    prefix_columns = [
        "total_k",
        "prefix",
        "prefix_length",
        "suffix_length",
        "prefix_source",
        "seed",
    ]
    prefix_path = destination / ("prefix_manifest.pilot.csv" if pilot else "prefix_manifest.csv")
    manifest.loc[manifest["representation"].eq("prefix_filtered")].reindex(columns=prefix_columns).drop_duplicates().to_csv(
        prefix_path, index=False
    )

    resolved = dict(config)
    resolved["output_dir"] = str(destination)
    resolved["pilot"] = pilot
    resolved["code_version"] = _code_version()
    resolved["condition_manifest"] = str(manifest_path)
    resolved["prefix_manifest"] = str(prefix_path)
    resolved["condition_manifest_sha256"] = _manifest_digest(manifest_path)
    resolved["prefix_manifest_sha256"] = _manifest_digest(prefix_path)
    # Likewise freeze input paths in the resolved config.  The example config
    # uses repository-root-relative paths. ``path_base`` lets a separate
    # deployment choose another explicit base without depending on CWD.
    resolved["datasets"] = _resolve_dataset_paths(
        config.get("datasets", {}), configured_base
    )
    resolved["prepared_from"] = str(config_path.resolve())
    resolved["path_base"] = str(configured_base)
    (destination / "resolved_run_manifest.json").write_text(json.dumps(resolved, indent=2) + "\n")
    return manifest_path


def prepare_split_manifest(
    config_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Sketch full genomes, cluster them once, and freeze shared grouped folds.

    This is deliberately separate from :func:`prepare`: full-genome Sourmash
    sketching can be expensive, while prefix/condition manifest generation is
    cheap and should be inspectable before compute is submitted.
    """

    from utilities.benchmark_design import create_cluster_manifest, create_split_manifest

    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    configured_base = _configured_path_base(config, config_path)
    config["datasets"] = _resolve_dataset_paths(config["datasets"], configured_base)
    default_output = _resolve_path(config["output_dir"], configured_base) / "split_manifest.csv"
    destination = _resolve_path(output_path, configured_base) if output_path else default_output
    destination.parent.mkdir(parents=True, exist_ok=True)

    clustering = dict(config.get("clustering", {}))
    threshold = clustering.get("distance_threshold")
    n_clusters = clustering.get("n_clusters")
    if (threshold is None) == (n_clusters is None):
        raise ValueError(
            "clustering must declare exactly one of distance_threshold or n_clusters"
        )

    split_frames: list[pd.DataFrame] = []
    for dataset_name, dataset in config["datasets"].items():
        phenotype_definitions = [
            phenotype
            for phenotype in config["phenotypes"]
            if phenotype["dataset"] == dataset_name
        ]
        if not phenotype_definitions:
            continue
        sequences, _ = _load_sequences(dataset)
        id_column = dataset["id_column"]
        label_columns = list(
            dict.fromkeys(
                phenotype.get("label_column", phenotype["name"])
                for phenotype in phenotype_definitions
            )
        )
        labels = _read_table(dataset["labels"], dataset.get("label_separator", ","))
        missing = {id_column, *label_columns}.difference(labels.columns)
        if missing:
            raise ValueError(
                f"Labels for dataset {dataset_name!r} are missing columns: {sorted(missing)}"
            )
        labels = labels[[id_column, *label_columns]].copy()
        labels[id_column] = labels[id_column].astype(str)
        if labels[id_column].duplicated().any():
            conflicts = labels.groupby(id_column, dropna=False)[label_columns].nunique(
                dropna=True
            )
            if conflicts.gt(1).any().any():
                raise ValueError(
                    f"Labels for dataset {dataset_name!r} contain conflicting duplicate genome IDs"
                )
            labels = labels.groupby(id_column, as_index=False, dropna=False).first()
        for phenotype in phenotype_definitions:
            label_column = phenotype.get("label_column", phenotype["name"])
            observed = list(labels[label_column].dropna().unique())
            declared = [
                _resolve_class(value, observed) for value in phenotype.get("classes", [])
            ]
            if not declared or set(declared) != set(observed):
                raise ValueError(
                    f"Configured classes {phenotype.get('classes')!r} do not match observed "
                    f"classes {sorted(observed, key=str)!r} for {label_column!r}"
                )

        cluster_manifest = create_cluster_manifest(
            sequences,
            dataset=dataset_name,
            ksize=int(clustering.get("ksize", 31)),
            n_hashes=int(clustering.get("n_hashes", 1000)),
            threshold=None if threshold is None else float(threshold),
            n_clusters=None if n_clusters is None else int(n_clusters),
        )
        design = cluster_manifest.merge(
            labels,
            left_on="genome_id",
            right_on=id_column,
            how="left",
            validate="one_to_one",
        )
        if id_column != "genome_id":
            design = design.drop(columns=[id_column])
        split_frames.append(
            create_split_manifest(
                design,
                phenotype=label_columns,
                n_folds=int(config.get("n_folds", 5)),
                seed=int(config.get("split_seed", 0)),
            )
        )

    if not split_frames:
        raise ValueError("No configured dataset has a phenotype to split")
    manifest = pd.concat(split_frames, ignore_index=True, sort=False)
    if destination.suffix.lower() in {".parquet", ".pq"}:
        manifest.to_parquet(destination, index=False)
    else:
        manifest.to_csv(destination, index=False)
    return destination


def _read_table(path: str | Path, separator: str = ",") -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, sep=separator)


def _fasta_paths(input_path: Path, file_type: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    extensions = {file_type, "fasta", "fa", "fna"}
    return sorted(
        {
            path
            for extension in extensions
            for pattern in (f"*.{extension}", f"*.{extension}.gz")
            for path in input_path.glob(pattern)
        }
    )


def _read_fasta_contigs(path: Path) -> list[str]:
    from Bio import SeqIO

    handle = gzip.open(path, "rt") if path.suffix.lower() == ".gz" else open(path, "rt")
    with handle as stream:
        return [str(record.seq) for record in SeqIO.parse(stream, "fasta")]


class _LazyFastaSequences(Mapping[str, list[str]]):
    """Index FASTA paths while decoding only the genome currently requested."""

    def __init__(self, paths: list[Path]):
        self._paths: OrderedDict[str, Path] = OrderedDict()
        for path in paths:
            genome_id = Path(path.stem).stem if path.suffix.lower() == ".gz" else path.stem
            self._paths[genome_id] = path

    def __getitem__(self, genome_id: str) -> list[str]:
        return _read_fasta_contigs(self._paths[genome_id])

    def __iter__(self):
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)


def _sequence_value_to_contigs(sequence: Any) -> list[str]:
    if sequence is None or (pd.api.types.is_scalar(sequence) and pd.isna(sequence)):
        return []
    if isinstance(sequence, str):
        return sequence.split()
    return [str(contig) for contig in sequence]


class _LazyParquetSequences(Mapping[str, list[str]]):
    """Index IDs cheaply and retain at most one decoded Parquet shard."""

    def __init__(self, paths: list[Path], id_column: str, sequence_column: str):
        self._id_column = id_column
        self._sequence_column = sequence_column
        self._locations: OrderedDict[str, list[tuple[Path, int]]] = OrderedDict()
        for path in paths:
            identifiers = pd.read_parquet(path, columns=[id_column])
            for row, genome_id in enumerate(identifiers[id_column]):
                self._locations.setdefault(str(genome_id), []).append((path, row))
        self._cached_path: Path | None = None
        self._cached_sequences: pd.Series | None = None

    def _sequences(self, path: Path) -> pd.Series:
        if self._cached_path != path:
            frame = pd.read_parquet(path, columns=[self._sequence_column])
            self._cached_path = path
            self._cached_sequences = frame[self._sequence_column]
        assert self._cached_sequences is not None
        return self._cached_sequences

    def __getitem__(self, genome_id: str) -> list[str]:
        contigs: list[str] = []
        for path, row in self._locations[genome_id]:
            contigs.extend(_sequence_value_to_contigs(self._sequences(path).iloc[row]))
        return contigs

    def __iter__(self):
        return iter(self._locations)

    def __len__(self) -> int:
        return len(self._locations)


class _SelectedSequences(Mapping[str, Any]):
    """Ordered, lazy view over the genomes shared by labels and split data."""

    def __init__(self, source: Mapping[str, Any], genome_ids: list[str]):
        self._source = source
        self._genome_ids = list(genome_ids)
        self._genome_id_set = set(genome_ids)

    def __getitem__(self, genome_id: str) -> Any:
        if genome_id not in self._genome_id_set:
            raise KeyError(genome_id)
        return self._source[genome_id]

    def __iter__(self):
        return iter(self._genome_ids)

    def __len__(self) -> int:
        return len(self._genome_ids)


def _open_sequence_source(dataset: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a source that decodes at most one genome or Parquet shard."""
    input_path = Path(dataset["input"])
    file_type = str(dataset["file_type"]).lower().lstrip(".")
    if file_type.endswith(".gz"):
        file_type = file_type[:-3]
    if file_type in {"fasta", "fa", "fna"}:
        source = _LazyFastaSequences(_fasta_paths(input_path, file_type))
    elif file_type == "parquet":
        paths = [input_path] if input_path.is_file() else sorted(input_path.glob("*.parquet"))
        source = _LazyParquetSequences(
            paths,
            str(dataset["id_column"]),
            str(dataset.get("sequence_column", "dna_sequence")),
        )
    else:
        raise ValueError(f"Unsupported file_type {file_type!r}; use FASTA or Parquet")
    if not source:
        raise ValueError(f"No {file_type} genomes found at {input_path}")
    return source


def _load_sequences(dataset: Mapping[str, Any]) -> tuple[OrderedDict[str, list[str]], int]:
    """Load FASTA (including ``.gz``) or Parquet without joining contigs.

    The repository's gentamicin data stores one compressed FASTA per genome,
    with the genome ID in the filename.  Keeping the contigs as a list is
    important: the sparse counter must never create a k-mer across a contig
    boundary.
    """
    input_path = Path(dataset["input"])
    file_type = str(dataset["file_type"]).lower().lstrip(".")
    if file_type.endswith(".gz"):
        file_type = file_type[:-3]
    id_column = dataset["id_column"]
    sequence_column = dataset.get("sequence_column", "dna_sequence")
    sequences: OrderedDict[str, list[str]] = OrderedDict()

    if file_type in {"fasta", "fa", "fna"}:
        for path in _fasta_paths(input_path, file_type):
            genome_id = Path(path.stem).stem if path.suffix.lower() == ".gz" else path.stem
            sequences[genome_id] = _read_fasta_contigs(path)
    elif file_type == "parquet":
        paths = [input_path] if input_path.is_file() else sorted(input_path.glob("*.parquet"))
        for path in paths:
            frame = pd.read_parquet(path, columns=[id_column, sequence_column])
            for genome_id, sequence in frame[[id_column, sequence_column]].itertuples(index=False):
                contigs = _sequence_value_to_contigs(sequence)
                sequences.setdefault(str(genome_id), []).extend(contigs)
    else:
        raise ValueError(f"Unsupported file_type {file_type!r}; use FASTA or Parquet")

    if not sequences:
        raise ValueError(f"No {file_type} genomes found at {input_path}")
    total_bases = sum(len(contig) for contigs in sequences.values() for contig in contigs)
    return sequences, total_bases


def _manifest_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sequence_digest(genomes: Mapping[str, list[str]]) -> str:
    """Fingerprint genome order, IDs, contents, and contig boundaries."""
    return _sequence_digest_and_total_bases(genomes)[0]


def _sequence_digest_and_total_bases(genomes: Mapping[str, list[str]]) -> tuple[str, int]:
    """Fingerprint a sequence stream and count bases without retaining it."""

    digest = hashlib.sha256()
    total_bases = 0
    for genome_id, contigs in genomes.items():
        digest.update(str(genome_id).encode("utf-8"))
        digest.update(b"\0")
        for contig in contigs:
            total_bases += len(contig)
            digest.update(str(contig).upper().encode("ascii"))
            digest.update(b"\0")
        digest.update(b"\xff")
    return digest.hexdigest(), total_bases


def _fingerprinted_sequence_stream(
    genomes: Mapping[str, list[str]], state: dict[str, Any]
):
    """Yield genomes once while recording the same digest and base count."""
    digest = hashlib.sha256()
    total_bases = 0
    for genome_id, contigs in genomes.items():
        digest.update(str(genome_id).encode("utf-8"))
        digest.update(b"\0")
        for contig in contigs:
            total_bases += len(contig)
            digest.update(str(contig).upper().encode("ascii"))
            digest.update(b"\0")
        digest.update(b"\xff")
        yield contigs
    state["input_sha256"] = digest.hexdigest()
    state["total_bases"] = total_bases


def _validate_unique_labels(frame: pd.DataFrame, id_column: str, label_column: str) -> pd.DataFrame:
    duplicate_rows = frame[frame[id_column].duplicated(keep=False)]
    conflicts = duplicate_rows.groupby(id_column, dropna=False)[label_column].nunique(dropna=False)
    if conflicts.gt(1).any():
        raise ValueError("Labels contain conflicting duplicate labels for one or more genome IDs")
    return frame.drop_duplicates(id_column)


def _resolve_class(value: Any, classes: Any) -> Any:
    """Resolve a JSON-configured class against labels loaded from a table.

    CSV readers commonly infer ``1`` as an integer while JSON necessarily
    stores the same setting as ``"1"``.  Match exact values first, then use a
    string comparison so this harmless serialization difference cannot
    disable the explicitly configured positive-class orientation.
    """

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    classes = list(classes)
    for candidate in classes:
        if value == candidate:
            return candidate
    text = str(value)
    for candidate in classes:
        if str(candidate) == text:
            return candidate
    return value


def _run_condition(
    config: Mapping[str, Any],
    condition: Mapping[str, Any] | pd.Series,
    split_manifest_path: str | Path,
    *,
    reuse_features: bool = False,
    overall_measurement: Any,
) -> Path:
    """Execute one full or prefix-filtered condition and persist tidy outputs."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from embeddings.sparse_kmer_counts import build_sparse_kmer_counts, load_sparse_kmer_counts
    from utilities.benchmark_design import read_split_manifest, validate_split_manifest
    from utilities.benchmark_metrics import prediction_metrics
    from utilities.resource_monitoring import ResourceMonitor, measure_stage

    condition = dict(condition)
    condition_id = str(condition["condition_id"])
    dataset_name = str(condition["dataset"])
    dataset = config["datasets"][dataset_name]
    output_root = Path(config["output_dir"])
    output_directory = output_root / "conditions" / condition_id
    output_directory.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(condition_id=condition_id)
    with monitor.stage("loading"):
        sequences = _open_sequence_source(dataset)
        labels = _read_table(dataset["labels"], dataset.get("label_separator", ","))
        id_column = dataset["id_column"]
        label_column = condition["label_column"]
        if id_column not in labels or label_column not in labels:
            raise ValueError(f"Labels must include {id_column!r} and {label_column!r}")
        labels = labels[[id_column, label_column]].dropna().copy()
        labels[id_column] = labels[id_column].astype(str)
        labels = _validate_unique_labels(labels, id_column, label_column)
        split_manifest = read_split_manifest(split_manifest_path)
        split_manifest["genome_id"] = split_manifest["genome_id"].astype(str)
        split_manifest = split_manifest[split_manifest["dataset"].astype(str) == dataset_name].copy()
        requested_folds = int(
            split_manifest["requested_n_folds"].iloc[0]
            if "requested_n_folds" in split_manifest
            else split_manifest.attrs.get("requested_n_folds", config.get("n_folds", 0))
        )
        if config.get("n_folds") is not None and requested_folds != int(config["n_folds"]):
            raise ValueError(
                f"Split manifest was requested with {requested_folds} folds, not configured {config['n_folds']}"
            )
        effective_folds = int(
            split_manifest["effective_n_folds"].iloc[0]
            if "effective_n_folds" in split_manifest
            else split_manifest.attrs.get("n_folds", split_manifest["fold_id"].nunique())
        )
        validate_split_manifest(
            split_manifest,
            n_folds=effective_folds,
        )

        usable_ids = [
            genome_id
            for genome_id in sequences
            if genome_id in set(labels[id_column]) and genome_id in set(split_manifest["genome_id"])
        ]
        if not usable_ids:
            raise ValueError("No genome IDs are shared by sequences, labels, and split manifest")
        label_by_id = labels.set_index(id_column)[label_column]
        fold_by_id = split_manifest.set_index("genome_id")["fold_id"]
        genomes = _SelectedSequences(sequences, usable_ids)
        y = label_by_id.loc[usable_ids].to_numpy()
        folds = fold_by_id.loc[usable_ids].to_numpy()
        expected_classes = set(y)
        declared_classes = json.loads(str(condition["classes_json"]))
        resolved_classes = [_resolve_class(value, expected_classes) for value in declared_classes]
        if set(resolved_classes) != expected_classes or len(resolved_classes) != len(expected_classes):
            raise ValueError(
                f"Configured classes {declared_classes!r} do not match observed classes "
                f"{sorted(expected_classes, key=str)!r} for {label_column!r}"
            )
        if int(condition["n_classes"]) != len(expected_classes):
            raise ValueError(
                f"Configured n_classes={condition['n_classes']!r} does not match "
                f"{len(expected_classes)} observed classes for {label_column!r}"
            )
        for fold_id in pd.unique(folds):
            if set(y[folds == fold_id]) != expected_classes:
                raise ValueError(f"Fold {fold_id!r} lacks one or more classes for {label_column!r}")

    prefix_value = condition.get("prefix")
    prefix = None if prefix_value is None or pd.isna(prefix_value) or str(prefix_value) == "" else str(prefix_value)
    expected_matrix_name = (
        f"{dataset_name}_sparse_kmers_total_k_{int(condition['total_k'])}_"
        f"prefix_{prefix or 'full'}_normalization_{config.get('normalization', 'frequency')}.npz"
    )
    matrix_path = output_directory / expected_matrix_name
    cache_used = bool(reuse_features and matrix_path.exists() and matrix_path.with_suffix(".json").exists())
    if cache_used:
        input_sha256, total_bases = _sequence_digest_and_total_bases(genomes)
        with monitor.stage("feature_loading", cache_state="warm"):
            features = load_sparse_kmer_counts(matrix_path)
        if features.genome_ids != usable_ids:
            raise ValueError("Cached feature genome order does not match the current benchmark inputs")
        expected_cache_metadata = {
            "total_k": int(condition["total_k"]),
            "prefix": prefix,
            "normalization": config.get("normalization", "frequency"),
            "reverse_complement": bool(config.get("reverse_complement", False)),
            "input_sha256": input_sha256,
        }
        mismatches = {
            key: (features.metadata.get(key), expected)
            for key, expected in expected_cache_metadata.items()
            if features.metadata.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"Cached feature metadata does not match this condition: {mismatches}")
    else:
        stream_state: dict[str, Any] = {}
        with monitor.stage("feature_extraction", cache_state="cold"):
            features = build_sparse_kmer_counts(
                _fingerprinted_sequence_stream(genomes, stream_state),
                total_k=int(condition["total_k"]),
                prefix=prefix,
                normalization=config.get("normalization", "frequency"),
                reverse_complement=bool(config.get("reverse_complement", False)),
                genome_ids=usable_ids,
                dataset=dataset_name,
                stage_callback=lambda stage: monitor.stage(stage, cache_state="cold"),
            )
            input_sha256 = str(stream_state["input_sha256"])
            total_bases = int(stream_state["total_bases"])
            features.metadata["input_sha256"] = input_sha256
        with monitor.stage("matrix_serialization", cache_state="cold"):
            matrix_path = features.save(output_directory, dataset=dataset_name)

    prediction_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    model_config = dict(config.get("model", {}))
    positive_class = _resolve_class(condition.get("positive_class"), expected_classes)
    for fold_id in sorted(np.unique(folds)):
        train = folds != fold_id
        test = folds == fold_id
        if not train.any() or not test.any():
            raise ValueError(f"Fold {fold_id!r} has an empty train or test partition")
        classifier = LogisticRegression(**model_config)
        with monitor.stage("fitting", fold_id=fold_id):
            classifier.fit(features.matrix[train], y[train])
        with monitor.stage("prediction", fold_id=fold_id):
            predicted = classifier.predict(features.matrix[test])
            probabilities = classifier.predict_proba(features.matrix[test])

        if len(classifier.classes_) == 2:
            if positive_class not in set(classifier.classes_):
                raise ValueError(
                    f"Configured positive class {positive_class!r} is not in {classifier.classes_.tolist()}"
                )
            positive_index = list(classifier.classes_).index(positive_class)
            score = probabilities[:, positive_index]
        else:
            score = probabilities
        fold_metrics = prediction_metrics(
            y[test],
            predicted,
            scores=score,
            labels=classifier.classes_,
            positive_label=positive_class if len(classifier.classes_) == 2 else None,
        )
        fold_metrics["classes"] = json.dumps(fold_metrics["classes"], default=str)
        fold_metrics["confusion_matrix"] = json.dumps(fold_metrics["confusion_matrix"])
        metric_rows.append({**condition, "fold_id": fold_id, **fold_metrics})
        for row_index, genome_index in enumerate(np.flatnonzero(test)):
            prediction_rows.append(
                {
                    **condition,
                    "fold_id": fold_id,
                    "genome_id": usable_ids[genome_index],
                    "true_label": y[genome_index],
                    "predicted_label": predicted[row_index],
                    "score": float(score[row_index]) if np.asarray(score).ndim == 1 else None,
                }
            )
            for class_index, score_class in enumerate(classifier.classes_):
                class_probability = (
                    probabilities[row_index, class_index]
                    if probabilities.ndim == 2
                    else float("nan")
                )
                score_rows.append(
                    {
                        **condition,
                        "fold_id": fold_id,
                        "genome_id": usable_ids[genome_index],
                        "score_class": score_class,
                        "predicted_probability": float(class_probability),
                    }
                )

    matrix_size = matrix_path.stat().st_size + matrix_path.with_suffix(".json").stat().st_size
    retained_tokens = int(features.raw_token_counts.sum())
    valid_windows = int(features.valid_window_counts.sum())
    resource_common = {
        **condition,
        "input_genomes": len(usable_ids),
        "total_bases": total_bases,
        "retained_token_count": retained_tokens,
        "retention_fraction": retained_tokens / valid_windows if valid_windows else 0.0,
        "feature_space_size": features.feature_space_size,
        "observed_features": int(np.unique(features.matrix.indices).size),
        "matrix_rows": features.matrix.shape[0],
        "matrix_columns": features.matrix.shape[1],
        "matrix_nnz": int(features.matrix.nnz),
        "serialized_size_bytes": matrix_size,
        "split_manifest_sha256": _manifest_digest(split_manifest_path),
        "cache_state": "warm" if cache_used else "cold",
    }
    overall_measurement.__exit__(None, None, None)
    overall_measurement.record["cache_state"] = resource_common["cache_state"]
    monitor.records.append(overall_measurement.record)
    resource_rows = [{**resource_common, **record} for record in monitor.records]
    pd.DataFrame(metric_rows).to_parquet(output_directory / "metrics.parquet", index=False)
    pd.DataFrame(resource_rows).to_parquet(output_directory / "resources.parquet", index=False)
    pd.DataFrame(prediction_rows).to_parquet(output_directory / "predictions.parquet", index=False)
    pd.DataFrame(score_rows).to_parquet(output_directory / "prediction_scores.parquet", index=False)
    strict_condition = {key: _json_scalar(value) for key, value in condition.items()}
    (output_directory / "condition.json").write_text(
        json.dumps(strict_condition, indent=2, allow_nan=False) + "\n"
    )
    return output_directory


def run_condition(
    config: Mapping[str, Any],
    condition: Mapping[str, Any] | pd.Series,
    split_manifest_path: str | Path,
    *,
    reuse_features: bool = False,
) -> Path:
    """Run one condition and always close the process-wide resource sampler.

    Stage samplers are closed by :class:`ResourceMonitor`; the overall
    sampler spans the whole condition and therefore needs this small wrapper
    around the implementation.  In particular, malformed inputs and model
    failures must not leave a daemon sampler thread alive in a long-lived
    process or test runner.
    """

    from utilities.resource_monitoring import measure_stage

    condition_id = str(condition["condition_id"])
    overall_measurement = measure_stage("overall", condition_id, cache_state="unknown")
    overall_measurement.__enter__()
    try:
        result = _run_condition(
            config,
            condition,
            split_manifest_path,
            reuse_features=reuse_features,
            overall_measurement=overall_measurement,
        )
    except BaseException as error:
        # _run_condition closes the measurement on its successful path before
        # serializing resources.  If it fails earlier, close it with the
        # original exception so the stage has a truthful lifecycle record.
        if not hasattr(overall_measurement, "record"):
            overall_measurement.__exit__(type(error), error, error.__traceback__)
        raise
    else:
        if not hasattr(overall_measurement, "record"):
            overall_measurement.__exit__(None, None, None)
        return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Freeze a condition manifest from JSON configuration")
    prepare_parser.add_argument("--config", required=True)
    prepare_parser.add_argument("--output-dir")
    prepare_parser.add_argument("--pilot", action="store_true")

    split_parser = subparsers.add_parser(
        "prepare-splits",
        help="Sketch full genomes and freeze the reusable cluster/fold manifest",
    )
    split_parser.add_argument("--config", required=True)
    split_parser.add_argument("--output")

    run_parser = subparsers.add_parser("run-condition", help="Run one SLURM-safe condition manifest row")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--condition-manifest", required=True)
    run_parser.add_argument("--condition-index", required=True, type=int)
    run_parser.add_argument("--split-manifest", required=True)
    run_parser.add_argument("--reuse-features", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        print(prepare(args.config, args.output_dir, args.pilot))
        return 0
    if args.command == "prepare-splits":
        print(prepare_split_manifest(args.config, args.output))
        return 0
    if args.command == "run-condition":
        config = json.loads(Path(args.config).read_text())
        manifest = _read_table(args.condition_manifest)
        if args.condition_index < 0 or args.condition_index >= len(manifest):
            raise IndexError(f"condition-index must be between 0 and {len(manifest) - 1}")
        condition = manifest.iloc[args.condition_index]
        try:
            print(run_condition(config, condition, args.split_manifest, reuse_features=args.reuse_features))
        except Exception as error:
            write_failure_record(config["output_dir"], condition, error)
            raise
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
