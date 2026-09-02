"""Design-time manifests for the reviewer benchmark.

The functions in this module deliberately do not extract features or fit models.
They create the immutable inputs shared by all benchmark conditions: prefixes,
phenotype definitions, and grouped genome folds.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DNA_ALPHABET = "ACGT"

# The benchmark uses integer labels for the binary phenotypes unless a caller
# supplies a more specific schema.  Keeping this in one place avoids silently
# reversing sensitivity/specificity in downstream code.
DEFAULT_PHENOTYPE_METADATA: dict[str, dict[str, Any]] = {
    "Gentamicin resistance": {
        "positive_class": 1,
        "classes": [0, 1],
        "n_classes": 2,
    },
    "Motility": {"positive_class": 1, "classes": [0, 1], "n_classes": 2},
    "Beta hemolysis": {
        "positive_class": 1,
        "classes": [0, 1],
        "n_classes": 2,
    },
    "Nitrate-to-nitrite conversion": {
        "positive_class": 1,
        "classes": [0, 1],
        "n_classes": 2,
    },
    "Glucose utilization": {
        "positive_class": 1,
        "classes": [0, 1],
        "n_classes": 2,
    },
}


def _genome_contigs(value: Any) -> list[str]:
    """Normalize one genome to independent, whitespace-free contigs.

    Whitespace-delimited strings are treated as separate contigs, matching
    the sequence convention used by the sparse counter and avoiding windows
    across boundaries in tabular sequence inputs.
    """

    if value is None:
        return []
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        return [part.upper() for part in value.split() if part]
    if not isinstance(value, Iterable):
        raise TypeError("genome sequences must be strings or iterables of contigs")
    contigs: list[str] = []
    for contig in value:
        if contig is None:
            continue
        if isinstance(contig, bytes):
            contig = contig.decode()
        if not isinstance(contig, str):
            raise TypeError("contigs must be strings")
        contigs.extend(part.upper() for part in contig.split() if part)
    return contigs


def create_cluster_manifest(
    genomes: Mapping[Any, Any],
    *,
    dataset: str | Mapping[Any, Any] | Sequence[Any] = "default",
    ksize: int = 31,
    n_hashes: int = 1000,
    threshold: float | None = None,
    n_clusters: int | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Sketch complete genomes and assign deterministic sequence clusters.

    Each mapping value is one sequence or an iterable of independent contigs;
    contigs are added separately to Sourmash so no k-mer crosses a boundary.
    Sourmash Jaccard similarities are explicitly converted to distances before
    hierarchical clustering.  Supply exactly one of ``threshold`` (distance
    cutoff) or ``n_clusters`` (requested flat-cluster count).  Cluster names
    are canonicalized by their lexicographically smallest genome ID, making
    manifests stable when the input mapping order changes.
    """

    if not isinstance(genomes, Mapping):
        raise TypeError("genomes must be a genome_id-to-sequences mapping")
    if not genomes:
        raise ValueError("genomes must not be empty")
    if (
        isinstance(ksize, bool)
        or not isinstance(ksize, (int, np.integer))
        or int(ksize) <= 0
    ):
        raise ValueError("ksize must be a positive integer")
    if (
        isinstance(n_hashes, bool)
        or not isinstance(n_hashes, (int, np.integer))
        or int(n_hashes) <= 0
    ):
        raise ValueError("n_hashes must be a positive integer")
    if (threshold is None) == (n_clusters is None):
        raise ValueError("supply exactly one of threshold or n_clusters")
    if threshold is not None:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.number)):
            raise ValueError("threshold must be a finite non-negative number")
        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be a finite non-negative number")
    if n_clusters is not None:
        if (
            isinstance(n_clusters, bool)
            or not isinstance(n_clusters, (int, np.integer))
            or int(n_clusters) < 1
            or int(n_clusters) > len(genomes)
        ):
            raise ValueError("n_clusters must be between 1 and the number of genomes")
        n_clusters = int(n_clusters)

    raw_genome_ids = [str(genome_id) for genome_id in genomes]
    if len(set(raw_genome_ids)) != len(raw_genome_ids):
        raise ValueError("genome IDs must be unique")
    genome_values = {
        str(genome_id): value for genome_id, value in genomes.items()
    }
    genome_ids = list(genome_values)
    genome_ids.sort()
    if isinstance(dataset, Mapping):
        dataset_values = {str(genome_id): value for genome_id, value in dataset.items()}
        missing = set(genome_ids).difference(dataset_values)
        if missing:
            raise ValueError(f"dataset mapping is missing genome IDs: {sorted(missing)}")
        datasets = [dataset_values[genome_id] for genome_id in genome_ids]
    elif isinstance(dataset, str):
        datasets = [dataset] * len(genome_ids)
    else:
        input_datasets = list(dataset)
        if len(input_datasets) != len(genome_ids):
            raise ValueError("dataset must have one value per genome")
        dataset_values = dict(zip(raw_genome_ids, input_datasets))
        datasets = [dataset_values[genome_id] for genome_id in genome_ids]
    datasets = [str(value) for value in datasets]

    try:
        import sourmash
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
    except ImportError as exc:
        raise ImportError("create_cluster_manifest requires sourmash and scipy") from exc

    sketches = []
    for genome_id in genome_ids:
        sketch = sourmash.MinHash(n=int(n_hashes), ksize=int(ksize))
        for contig in _genome_contigs(genome_values[genome_id]):
            sketch.add_sequence(contig, True)
        sketches.append(sketch)

    distance_matrix = np.zeros((len(sketches), len(sketches)), dtype=float)
    for left in range(len(sketches)):
        for right in range(left):
            similarity = float(sketches[left].jaccard(sketches[right]))
            distance = 1.0 - similarity
            distance_matrix[left, right] = distance
            distance_matrix[right, left] = distance

    if len(sketches) == 1:
        flat_clusters = np.ones(1, dtype=int)
    else:
        linkage_matrix = linkage(squareform(distance_matrix, checks=False), method="average")
        if threshold is not None:
            flat_clusters = fcluster(linkage_matrix, t=threshold, criterion="distance")
        else:
            flat_clusters = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")

    members: dict[int, list[str]] = {}
    for genome_id, flat_cluster in zip(genome_ids, flat_clusters):
        members.setdefault(int(flat_cluster), []).append(genome_id)
    canonical_ids = {
        flat_cluster: f"cluster_{rank:04d}"
        for rank, (flat_cluster, _) in enumerate(
            sorted(members.items(), key=lambda item: min(item[1]))
        )
    }
    manifest = pd.DataFrame(
        {
            "genome_id": genome_ids,
            "dataset": datasets,
            "cluster_id": [canonical_ids[int(cluster)] for cluster in flat_clusters],
        }
    )
    if output_path is not None:
        _write_table(manifest, output_path)
    return manifest


def _write_table(table: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        table.to_parquet(path, index=False)
    elif suffix == ".json":
        table.to_json(path, orient="records", indent=2)
    else:
        table.to_csv(path, index=False)


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path, orient="records")
    return pd.read_csv(path)


def _validate_prefix(prefix: str) -> str:
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("prefix must be a non-empty DNA string")
    prefix = prefix.upper()
    if any(base not in DNA_ALPHABET for base in prefix):
        raise ValueError(f"prefix contains a non-DNA base: {prefix!r}")
    return prefix


def generate_prefixes(
    prefix_length: int,
    n_prefixes: int,
    seed: int = 0,
    historical_prefix: str | None = None,
) -> list[str]:
    """Generate unique, seeded prefixes in a reproducible order.

    Random prefixes are sampled without replacement from lexicographically
    ordered DNA strings.  ``historical_prefix`` is appended only when it was
    not among the random draws, so it can never displace a prespecified random
    sample.
    """

    if not isinstance(prefix_length, (int, np.integer)) or prefix_length <= 0:
        raise ValueError("prefix_length must be a positive integer")
    if not isinstance(n_prefixes, (int, np.integer)) or n_prefixes < 0:
        raise ValueError("n_prefixes must be a non-negative integer")
    population_size = 4**int(prefix_length)
    if n_prefixes > population_size:
        raise ValueError(
            f"cannot draw {n_prefixes} unique prefixes of length {prefix_length}; "
            f"only {population_size} unique prefixes exist"
        )
    historical = None if historical_prefix is None else _validate_prefix(historical_prefix)
    if historical is not None and len(historical) != prefix_length:
        raise ValueError("historical_prefix length must equal prefix_length")

    population = ["".join(chars) for chars in product(DNA_ALPHABET, repeat=int(prefix_length))]
    rng = np.random.default_rng(seed)
    indices = rng.choice(population_size, size=int(n_prefixes), replace=False)
    prefixes = [population[int(index)] for index in indices]
    if historical is not None and historical not in prefixes:
        prefixes.append(historical)
    return prefixes


def validate_experiment_matrix(
    total_k_values: Sequence[int],
    prefix_lengths: Sequence[int],
    n_prefixes: int = 10,
    historical_prefix: str | None = None,
) -> bool:
    """Validate total k-mer and prefix lengths before feature extraction."""

    totals = list(total_k_values)
    lengths = list(prefix_lengths)
    if not totals:
        raise ValueError("total_k_values must not be empty")
    if any(not isinstance(k, (int, np.integer)) or int(k) <= 0 for k in totals):
        raise ValueError("total_k_values must contain positive integers")
    if len(set(map(int, totals))) != len(totals):
        raise ValueError("total_k_values must be unique")
    if any(not isinstance(p, (int, np.integer)) or int(p) < 0 for p in lengths):
        raise ValueError("prefix_length must be a non-negative integer")
    if len(set(map(int, lengths))) != len(lengths):
        raise ValueError("prefix_lengths must be unique")
    if any(int(p) >= int(k) for p in lengths for k in totals):
        raise ValueError("prefix_length must be smaller than every total_k value")
    if not isinstance(n_prefixes, (int, np.integer)) or n_prefixes < 0:
        raise ValueError("n_prefixes must be a non-negative integer")
    for prefix_length in lengths:
        if prefix_length == 0:
            continue
        if n_prefixes > 4 ** int(prefix_length):
            raise ValueError(
                f"requested sample exceeds the number of unique prefixes for length {prefix_length}"
            )
    if historical_prefix is not None:
        historical = _validate_prefix(historical_prefix)
        if len(historical) == 0 or len(historical) >= min(map(int, totals)):
            raise ValueError("historical prefix must be shorter than total_k")
    return True


def create_prefix_manifest(
    total_k_values: Sequence[int],
    prefix_lengths: Sequence[int],
    n_prefixes: int = 10,
    seed: int = 0,
    historical_prefix: str | None = None,
    output_path: str | Path | None = None,
    dataset: str | None = None,
) -> pd.DataFrame:
    """Create the condition manifest shared by all feature extraction jobs."""

    validate_experiment_matrix(total_k_values, prefix_lengths, n_prefixes, historical_prefix)
    totals = [int(k) for k in total_k_values]
    lengths = [int(p) for p in prefix_lengths]
    historical_length = None if historical_prefix is None else len(_validate_prefix(historical_prefix))
    # A historical condition is useful even when a caller's random grid did
    # not include its length.  It remains an explicit, non-random row.
    if historical_length and historical_length not in lengths:
        if any(historical_length >= k for k in totals):
            raise ValueError("historical prefix must be shorter than every total_k value")
        lengths.append(historical_length)

    rows: list[dict[str, Any]] = []
    for prefix_length in lengths:
        if prefix_length == 0:
            prefixes = [""]
            sources = ["full"]
        else:
            random_prefixes = generate_prefixes(prefix_length, n_prefixes, seed)
            prefixes = random_prefixes
            sources = ["seeded"] * len(prefixes)
            if historical_prefix is not None and len(historical_prefix) == prefix_length:
                historical = _validate_prefix(historical_prefix)
                if historical not in prefixes:
                    prefixes.append(historical)
                    sources.append("historical")
                else:
                    sources[prefixes.index(historical)] = "seeded_and_historical"
        for total_k in totals:
            for prefix, source in zip(prefixes, sources):
                rows.append(
                    {
                        "dataset": dataset,
                        "total_k": total_k,
                        "prefix": prefix,
                        "prefix_length": prefix_length,
                        "suffix_length": total_k - prefix_length,
                        "prefix_source": source,
                        "seed": int(seed),
                    }
                )
    columns = [
        "dataset",
        "total_k",
        "prefix",
        "prefix_length",
        "suffix_length",
        "prefix_source",
        "seed",
    ]
    manifest = pd.DataFrame(rows, columns=columns)
    # An unset optional dataset is still part of the schema; object dtype
    # ensures a CSV round-trip preserves the same representation as a named
    # dataset.
    manifest["dataset"] = manifest["dataset"].astype(object)
    if output_path is not None:
        _write_table(manifest, output_path)
    return manifest


def read_prefix_manifest(path: str | Path) -> pd.DataFrame:
    manifest = _read_table(path)
    required = {"total_k", "prefix", "prefix_length", "suffix_length", "prefix_source", "seed"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"prefix manifest missing columns: {sorted(missing)}")
    # CSV encodes the full-representation prefix as an empty field.
    manifest["prefix"] = manifest["prefix"].fillna("")
    if "dataset" in manifest.columns:
        dataset_values = manifest["dataset"].astype(object)
        dataset_values.loc[dataset_values.isna()] = None
        manifest["dataset"] = dataset_values
    if manifest.duplicated(["total_k", "prefix"]).any():
        raise ValueError("prefix manifest contains duplicate conditions")
    if not (manifest["suffix_length"] == manifest["total_k"] - manifest["prefix_length"]).all():
        raise ValueError("suffix_length does not match total_k - prefix_length")
    bad_prefix = manifest.apply(
        lambda row: (row["prefix"] == "" and int(row["prefix_length"]) != 0)
        or (row["prefix"] != "" and len(str(row["prefix"])) != int(row["prefix_length"])),
        axis=1,
    )
    if bad_prefix.any():
        raise ValueError("prefix length does not match prefix_length")
    invalid_prefixes = manifest.loc[manifest["prefix"] != "", "prefix"].map(
        lambda prefix: any(base not in DNA_ALPHABET for base in str(prefix).upper())
    )
    if invalid_prefixes.any():
        raise ValueError("prefix contains a non-DNA base")
    validate_experiment_matrix(
        sorted(manifest["total_k"].dropna().astype(int).unique()),
        sorted(manifest["prefix_length"].astype(int).unique()),
        n_prefixes=0,
    )
    return manifest


def get_phenotype_metadata(
    phenotype: str,
    metadata: Mapping[str, Mapping[str, Any]] | None = None,
    labels: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Return a validated, serializable definition for one phenotype."""

    source = DEFAULT_PHENOTYPE_METADATA if metadata is None else metadata
    if phenotype not in source:
        if labels is None:
            raise KeyError(f"no metadata registered for phenotype {phenotype!r}")
        observed = list(pd.Series(labels).dropna().unique())
        if not observed:
            raise ValueError(f"cannot infer classes for empty phenotype {phenotype!r}")
        source_definition: dict[str, Any] = {"classes": sorted(observed, key=str)}
    else:
        source_definition = dict(source[phenotype])
    classes = list(source_definition.get("classes", []))
    if labels is not None:
        observed = list(pd.Series(labels).dropna().unique())
        classes = list(dict.fromkeys(classes + sorted(observed, key=str)))
    if not classes:
        n_classes = int(source_definition.get("n_classes", 0))
        classes = list(range(n_classes))
    if not classes:
        raise ValueError(f"phenotype {phenotype!r} has no classes")
    n_classes = int(source_definition.get("n_classes", len(classes)))
    if n_classes != len(classes):
        n_classes = len(classes)
    positive_class = source_definition.get("positive_class", classes[-1])
    if positive_class not in classes:
        raise ValueError(
            f"positive_class {positive_class!r} is not present in classes for phenotype {phenotype!r}"
        )
    result = {
        "name": phenotype,
        "positive_class": positive_class,
        "classes": classes,
        "n_classes": n_classes,
    }
    return result


def _as_dataframe(data: Any, id_column: str) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, Mapping):
        return pd.DataFrame({id_column: list(data), "value": list(data.values())})
    return pd.DataFrame(data)


def _normalise_labels(frame: pd.DataFrame, labels: Any, id_column: str, phenotype: str | Sequence[str] | None) -> pd.DataFrame:
    if labels is None:
        return frame
    if isinstance(labels, pd.Series):
        label_frame = labels.rename(phenotype if isinstance(phenotype, str) else "label").reset_index()
        label_frame = label_frame.rename(columns={"index": id_column})
    elif isinstance(labels, Mapping):
        if isinstance(phenotype, Sequence) and not isinstance(phenotype, str):
            # A mapping of genome ID to a mapping of phenotype names is a
            # convenient input for callers constructing several shared folds.
            if not all(isinstance(value, Mapping) for value in labels.values()):
                raise ValueError("mapping labels with multiple phenotypes must map IDs to mappings")
            label_frame = pd.DataFrame.from_dict(labels, orient="index")
            label_frame.index.name = id_column
            label_frame = label_frame.reset_index()
        else:
            label_frame = pd.DataFrame(
                {id_column: list(labels), phenotype or "label": list(labels.values())}
            )
    else:
        label_frame = pd.DataFrame(labels).copy()
    if id_column not in label_frame.columns:
        raise ValueError(f"labels must include {id_column!r}")
    overlap = [c for c in label_frame.columns if c != id_column and c in frame.columns]
    if overlap:
        frame = frame.drop(columns=overlap)
    return frame.merge(label_frame, on=id_column, how="left", validate="one_to_one")


def _assignment_for_groups(groups: pd.DataFrame, classes: list[Any], n_folds: int, seed: int) -> dict[tuple[Any, Any], int]:
    def value(key: tuple[Any, Any], column: str) -> Any:
        # ``DataFrame.loc[(a, b), column]`` treats a tuple as a 2-D indexer;
        # selecting the row first keeps the MultiIndex key unambiguous.
        return groups.loc[[key], column].iloc[0]

    rng = np.random.default_rng(seed)
    keys = list(groups.index)
    random_order = {key: int(value) for key, value in zip(keys, rng.permutation(len(keys)))}
    rarity = {cls: int(sum(cls in values for values in groups["_classes"])) for cls in classes}
    keys.sort(
        key=lambda key: (
            min(rarity[c] for c in value(key, "_classes"))
            if value(key, "_classes")
            else 0,
            random_order[key],
            str(key),
        )
    )
    fold_classes = [set() for _ in range(n_folds)]
    fold_sizes = np.zeros(n_folds, dtype=int)
    assignment: dict[tuple[Any, Any], int] = {}
    for key in keys:
        values = set(value(key, "_classes"))
        scores = []
        for fold in range(n_folds):
            gain = len(values - fold_classes[fold])
            # Strongly prefer coverage; size is only a deterministic tie-breaker.
            score = (gain * 1000, -len(fold_classes[fold] & values), -fold_sizes[fold], -fold)
            scores.append(score)
        fold = max(range(n_folds), key=lambda index: scores[index])
        assignment[key] = fold
        fold_classes[fold].update(values)
        fold_sizes[fold] += int(value(key, "_size"))

    # Repair a missing class where moving a group will not remove any class
    # from its source fold.  This matters for multi-phenotype manifests: a
    # group can carry several (phenotype, class) requirements at once, and a
    # move that repairs one requirement must not break another one.
    for cls in classes:
        for target in range(n_folds):
            if any(cls in value(key, "_classes") and assignment[key] == target for key in keys):
                continue
            candidates = [
                key for key in keys
                if assignment[key] != target and cls in value(key, "_classes")
            ]
            candidates.sort(key=lambda key: (assignment[key], str(key)))
            moved = False
            for key in candidates:
                source = assignment[key]
                values = set(value(key, "_classes"))
                # Preserve every requirement currently covered only by this
                # group in the source fold.
                source_would_lose = any(
                    requirement in values
                    and not any(
                        other != key
                        and assignment[other] == source
                        and requirement in value(other, "_classes")
                        for other in keys
                    )
                    for requirement in value(key, "_classes")
                )
                if source_would_lose:
                    continue
                assignment[key] = target
                moved = True
                break
            if not moved:
                return {}
    if any(
        not any(
            assignment[key] == fold and cls in value(key, "_classes")
            for key in keys
        )
        for fold in range(n_folds)
        for cls in classes
    ):
        return {}
    return assignment


def create_split_manifest(
    genomes: pd.DataFrame,
    labels: Any = None,
    cluster_ids: Mapping[Any, Any] | Sequence[Any] | None = None,
    id_column: str = "genome_id",
    dataset_column: str = "dataset",
    cluster_column: str = "cluster_id",
    phenotype: str | Sequence[str] | None = None,
    n_folds: int = 5,
    seed: int = 0,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Create deterministic grouped folds and return a genome-level manifest."""

    if (
        isinstance(n_folds, bool)
        or not isinstance(n_folds, (int, np.integer))
        or n_folds < 1
    ):
        raise ValueError("n_folds must be a positive integer")
    frame = _as_dataframe(genomes, id_column)
    if id_column not in frame.columns:
        raise ValueError(f"genomes must include {id_column!r}")
    if frame[id_column].duplicated().any():
        raise ValueError("genome IDs must be unique")
    frame = _normalise_labels(frame, labels, id_column, phenotype)
    if dataset_column not in frame.columns:
        frame[dataset_column] = "default"
    if cluster_ids is not None:
        if isinstance(cluster_ids, Mapping):
            frame[cluster_column] = frame[id_column].map(cluster_ids)
        else:
            values = list(cluster_ids)
            if len(values) != len(frame):
                raise ValueError("cluster_ids must have one value per genome")
            frame[cluster_column] = values
    elif cluster_column not in frame.columns:
        frame[cluster_column] = frame[id_column]
    if frame[cluster_column].isna().any():
        raise ValueError("every genome must have a cluster_id")

    if phenotype is None:
        label_columns: list[str] = []
    elif isinstance(phenotype, str):
        label_columns = [phenotype]
    else:
        label_columns = list(phenotype)
    for column in label_columns:
        if column not in frame.columns:
            raise ValueError(f"phenotype column {column!r} not found")

    if frame[dataset_column].isna().any():
        raise ValueError("every genome must have a dataset")

    grouped = frame.groupby([dataset_column, cluster_column], sort=True, dropna=False)
    group_rows = []
    for key, group in grouped:
        # Keep phenotype identity in the coverage requirements.  Taking the
        # union of raw class values would incorrectly treat a class present in
        # phenotype A as coverage for the same-valued class in phenotype B.
        values: set[Any] = set()
        for column in label_columns:
            values.update(
                (column, label)
                for label in group[column].tolist()
                if not pd.isna(label)
            )
        group_rows.append({"_key": key, "_classes": values, "_size": len(group)})
    if not group_rows:
        raise ValueError("cannot create folds from an empty genome set")
    group_frame = pd.DataFrame(group_rows).set_index("_key")
    classes = sorted({item for values in group_frame["_classes"] for item in values}, key=str)
    requested = int(n_folds)
    if classes:
        group_counts = [sum(cls in values for values in group_frame["_classes"]) for cls in classes]
        maximum = min(group_counts)
        effective = min(requested, maximum)
    else:
        effective = min(requested, len(group_frame))
    if effective < 1:
        raise ValueError("cannot create folds from an empty genome set")

    assignment: dict[tuple[Any, Any], int] = {}
    for candidate in range(effective, 0, -1):
        assignment = _assignment_for_groups(group_frame, classes, candidate, seed)
        if assignment:
            effective = candidate
            break
    if not assignment and len(group_frame):
        raise ValueError("unable to assign groups while preserving class coverage")
    frame["fold_id"] = [assignment[(dataset, cluster)] for dataset, cluster in zip(frame[dataset_column], frame[cluster_column])]
    # Use canonical names in the persisted schema while retaining a caller's
    # custom source column when it already has a different name.
    if dataset_column != "dataset":
        frame["dataset"] = frame[dataset_column]
    if cluster_column != "cluster_id":
        frame["cluster_id"] = frame[cluster_column]
    frame.attrs["requested_n_folds"] = requested
    frame.attrs["n_folds"] = effective
    frame.attrs["fold_reduction_rule"] = (
        f"fold_reduction: requested {requested}, used {effective}; "
        "reduce to the largest deterministic count with every class in every fold"
        if effective != requested
        else "no fold reduction"
    )
    frame["requested_n_folds"] = requested
    frame["effective_n_folds"] = effective
    frame["fold_reduction_rule"] = frame.attrs["fold_reduction_rule"]
    if output_path is not None:
        _write_table(frame, output_path)
    return frame


def validate_split_manifest(
    manifest: pd.DataFrame,
    phenotype: str | Sequence[str] | None = None,
    n_folds: int | None = None,
) -> bool:
    """Raise ``ValueError`` if a split manifest can leak or cannot be scored."""

    required = {"genome_id", "dataset", "cluster_id", "fold_id"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"split manifest missing columns: {sorted(missing)}")
    if manifest["genome_id"].duplicated().any():
        raise ValueError("split manifest contains duplicate genome IDs")
    if manifest.groupby(["dataset", "cluster_id"], dropna=False)["fold_id"].nunique().gt(1).any():
        raise ValueError("a cluster crosses fold boundaries")
    if manifest[["dataset", "cluster_id", "fold_id"]].isna().any().any():
        raise ValueError("split manifest contains missing dataset, cluster, or fold values")
    if n_folds is not None and (
        isinstance(n_folds, bool)
        or not isinstance(n_folds, (int, np.integer))
        or int(n_folds) < 1
    ):
        raise ValueError("n_folds must be a positive integer")
    requested = manifest["fold_id"].nunique() if n_folds is None else int(n_folds)
    if requested != manifest["fold_id"].nunique():
        raise ValueError("manifest does not contain the requested number of folds")
    if phenotype is not None:
        columns = [phenotype] if isinstance(phenotype, str) else list(phenotype)
        for column in columns:
            if column not in manifest.columns:
                raise ValueError(f"phenotype column {column!r} not found")
            classes = set(manifest[column].dropna().unique())
            for fold in manifest["fold_id"].unique():
                observed = set(manifest.loc[manifest["fold_id"] == fold, column].dropna().unique())
                if observed != classes:
                    raise ValueError(f"fold {fold} lacks classes for phenotype {column!r}")
    return True


def read_split_manifest(path: str | Path) -> pd.DataFrame:
    manifest = _read_table(path)
    validate_split_manifest(manifest)
    if "requested_n_folds" in manifest.columns:
        manifest.attrs["requested_n_folds"] = int(manifest["requested_n_folds"].iloc[0])
    if "effective_n_folds" in manifest.columns:
        manifest.attrs["n_folds"] = int(manifest["effective_n_folds"].iloc[0])
    if "fold_reduction_rule" in manifest.columns:
        manifest.attrs["fold_reduction_rule"] = str(manifest["fold_reduction_rule"].iloc[0])
    return manifest


def assert_split_manifest_identity(left: pd.DataFrame | str | Path, right: pd.DataFrame | str | Path) -> None:
    """Assert that two conditions use exactly the same genome-level folds."""

    left_frame = read_split_manifest(left) if isinstance(left, (str, Path)) else left
    right_frame = read_split_manifest(right) if isinstance(right, (str, Path)) else right
    columns = ["genome_id", "dataset", "cluster_id", "fold_id"]
    if set(columns).difference(left_frame.columns) or set(columns).difference(right_frame.columns):
        raise ValueError("both split manifests need genome_id, dataset, cluster_id and fold_id")
    left_key = left_frame[columns].sort_values(columns).reset_index(drop=True)
    right_key = right_frame[columns].sort_values(columns).reset_index(drop=True)
    if not left_key.equals(right_key):
        raise AssertionError("split manifests do not have identical genome/cluster/fold assignments")
