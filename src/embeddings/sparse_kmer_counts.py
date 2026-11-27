"""Sparse full and prefix-filtered DNA k-mer count representations.

The public entry point is :func:`build_sparse_kmer_counts`.  It indexes a
complete k-mer (or, for a prefix condition, its suffix) directly with a
base-four integer, so the feature space does not need to be enumerated.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
import json
from multiprocessing import get_context
from itertools import islice
from pathlib import Path
import re
from typing import Any

import numpy as np
from scipy import sparse


_DNA_ALPHABET = "ACGT"
_DNA_TO_DIGIT = {base: digit for digit, base in enumerate(_DNA_ALPHABET)}
_COMPLEMENT = str.maketrans(_DNA_ALPHABET, "TGCA")
_VALID_DNA_RUN = re.compile(r"[ACGT]+")


def encode_kmer(kmer: str) -> int:
    """Return the deterministic base-four index for an uppercase DNA k-mer.

    ``A``, ``C``, ``G`` and ``T`` map to 0, 1, 2 and 3 respectively.  Input
    is accepted case-insensitively, but ambiguous bases are rejected rather
    than silently assigned a feature index.
    """

    if not isinstance(kmer, str) or not kmer:
        raise ValueError("kmer must be a non-empty DNA string")
    index = 0
    for base in kmer.upper():
        try:
            index = index * 4 + _DNA_TO_DIGIT[base]
        except KeyError as exc:
            raise ValueError(f"kmer contains non-DNA base {base!r}") from exc
    return index


def decode_kmer(index: int, length: int) -> str:
    """Decode a base-four feature index into a DNA k-mer of ``length``."""

    if not isinstance(index, (int, np.integer)) or index < 0:
        raise ValueError("index must be a non-negative integer")
    if not isinstance(length, (int, np.integer)) or length <= 0:
        raise ValueError("length must be a positive integer")
    if index >= 4**length:
        raise ValueError("index does not fit in the requested k-mer length")
    chars = ["A"] * length
    for position in range(length - 1, -1, -1):
        chars[position] = _DNA_ALPHABET[index % 4]
        index //= 4
    return "".join(chars)


def _normalise_normalization(normalization: str | bool | None) -> str:
    if normalization is True or normalization is None:
        return "frequency" if normalization is not False else "raw"
    if normalization is False:
        return "raw"
    if not isinstance(normalization, str):
        raise ValueError("normalization must be 'frequency', 'raw', or a bool")
    value = normalization.lower().strip()
    if value in {"frequency", "frequencies", "freq"}:
        return "frequency"
    if value in {"raw", "count", "counts", "none"}:
        return "raw"
    raise ValueError("normalization must be 'frequency' or 'raw'")


def _as_contigs(genome: Any) -> list[str]:
    """Convert one genome value to independent, uppercase contig strings."""

    if genome is None:
        return []
    if isinstance(genome, bytes):
        genome = genome.decode()
    if isinstance(genome, str):
        # Whitespace-delimited input is common in the repository's parquet
        # files.  Splitting it also makes accidental cross-contig windows
        # impossible; callers with a single contig need not add whitespace.
        return [part.upper() for part in genome.split() if part]
    if not isinstance(genome, Iterable):
        raise TypeError("each genome must be a sequence string or an iterable of contigs")
    contigs: list[str] = []
    for contig in genome:
        if contig is None:
            continue
        if isinstance(contig, bytes):
            contig = contig.decode()
        if not isinstance(contig, str):
            raise TypeError("contigs must be strings")
        contigs.extend(part.upper() for part in contig.split() if part)
    return contigs


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(_COMPLEMENT)[::-1]


@dataclass
class SparseKmerCounts:
    """A CSR count matrix and the metadata needed to interpret its columns."""

    matrix: sparse.csr_matrix
    metadata: dict[str, Any]

    @property
    def feature_space_size(self) -> int:
        return int(self.metadata["feature_space_size"])

    @property
    def total_k(self) -> int:
        return int(self.metadata["total_k"])

    @property
    def prefix(self) -> str | None:
        return self.metadata["prefix"]

    @property
    def suffix_length(self) -> int:
        return int(self.metadata["suffix_length"])

    @property
    def genome_ids(self) -> list[str]:
        return list(self.metadata["genome_ids"])

    @property
    def raw_token_counts(self) -> np.ndarray:
        """Number of retained valid tokens per genome, before normalization."""

        return np.asarray(self.metadata["raw_token_counts"], dtype=np.int64)

    @property
    def valid_window_counts(self) -> np.ndarray:
        return np.asarray(self.metadata["valid_window_counts"], dtype=np.int64)

    def save(self, path: str | Path, dataset: str | None = None) -> Path:
        """Save the CSR matrix and JSON metadata, returning the NPZ path.

        Passing a directory creates a descriptive filename containing the
        dataset, total k, prefix condition and normalization.  A path ending
        in ``.npz`` may be supplied when a caller needs an explicit location;
        its adjacent JSON file remains part of the save format.
        """

        destination = Path(path)
        dataset_name = dataset or str(self.metadata.get("dataset", "dataset"))
        if destination.suffix.lower() != ".npz":
            destination.mkdir(parents=True, exist_ok=True)
            prefix_name = self.prefix if self.prefix is not None else "full"
            destination = destination / (
                f"{_safe_component(dataset_name)}_sparse_kmers_"
                f"total_k_{self.total_k}_prefix_{_safe_component(prefix_name)}_"
                f"normalization_{self.metadata['normalization']}.npz"
            )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)

        metadata = dict(self.metadata)
        metadata["dataset"] = dataset_name
        sparse.save_npz(destination, self.matrix, compressed=True)
        destination.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "SparseKmerCounts":
        return load_sparse_kmer_counts(path)


def _safe_component(value: Any) -> str:
    value = str(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value) or "none"


def _count_genome(
    contigs: list[str], total_k: int, prefix: str | None, reverse_complement: bool
) -> tuple[dict[int, int], int, int, int]:
    row_counts: dict[int, int] = {}
    raw_count = valid_count = candidate_count = 0
    strands = list(contigs)
    if reverse_complement:
        strands.extend(_reverse_complement(contig) for contig in contigs)
    for sequence in strands:
        candidate_count += max(0, len(sequence) - total_k + 1)
        if prefix is None:
            for start in range(max(0, len(sequence) - total_k + 1)):
                kmer = sequence[start : start + total_k]
                if any(base not in _DNA_TO_DIGIT for base in kmer):
                    continue
                valid_count += 1
                raw_count += 1
                column = encode_kmer(kmer)
                row_counts[column] = row_counts.get(column, 0) + 1
            continue

        # Prefix-filtered conditions retain only ~4**-len(prefix) of
        # all windows. Find prefix occurrences inside valid DNA runs
        # in C-backed string/regex operations instead of constructing
        # and checking every candidate k-mer in Python.
        for match in _VALID_DNA_RUN.finditer(sequence):
            valid_run = match.group(0)
            last_start = len(valid_run) - total_k
            valid_count += max(0, last_start + 1)
            start = valid_run.find(prefix)
            while start != -1 and start <= last_start:
                raw_count += 1
                suffix = valid_run[
                    start + len(prefix) : start + total_k
                ]
                column = encode_kmer(suffix)
                row_counts[column] = row_counts.get(column, 0) + 1
                start = valid_run.find(prefix, start + 1)
    return row_counts, raw_count, valid_count, candidate_count


def _count_genomes(
    genomes: Iterable[Any],
    total_k: int,
    prefix: str | None,
    reverse_complement: bool,
    workers: int,
) -> Iterable[tuple[dict[int, int], int, int, int]]:
    contigs = (_as_contigs(genome) for genome in genomes)
    if workers == 1:
        for genome in contigs:
            yield _count_genome(genome, total_k, prefix, reverse_complement)
        return

    # Spawn rather than fork because the benchmark samples resources in threads.
    # Limit the queue so streaming input does not accumulate genomes in RAM.
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        pending = deque(
            pool.submit(_count_genome, genome, total_k, prefix, reverse_complement)
            for genome in islice(contigs, workers)
        )
        while pending:
            result = pending.popleft().result()
            genome = next(contigs, None)
            if genome is not None:
                pending.append(pool.submit(_count_genome, genome, total_k, prefix, reverse_complement))
            yield result


def build_sparse_kmer_counts(
    genomes: Mapping[Any, Any] | Sequence[Any] | Iterable[Any],
    total_k: int,
    prefix: str | None = None,
    *,
    normalization: str | bool | None = "frequency",
    reverse_complement: bool = False,
    genome_ids: Sequence[Any] | None = None,
    dataset: str = "dataset",
    stage_callback: Callable[[str], Any] | None = None,
    genome_counted: Callable[[], Any] | None = None,
    workers: int = 1,
) -> SparseKmerCounts:
    """Build full or prefix-filtered sparse k-mer counts.

    ``genomes`` is either an ID-to-sequences mapping or an ordered iterable of
    genome values.  A genome value can be one string or an iterable of contig
    strings.  Contigs are never concatenated.  A prefix condition stores only
    the suffix of each matching complete k-mer, making its feature space
    ``4 ** (total_k - len(prefix))``.  With ``reverse_complement=True``,
    forward and reverse-complement strand windows are both counted.

    ``stage_callback``, when supplied, must be a callable accepting a stage
    name and returning a context manager. It is entered separately for
    ``tokenization_and_counting`` and ``matrix_construction`` so callers can
    instrument those phases without reimplementing the counter.
    With more than one worker, counting runs in separate processes and results
    are assembled in input order.
    """

    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")

    if not isinstance(total_k, (int, np.integer)) or isinstance(total_k, bool) or total_k <= 0:
        raise ValueError("total_k must be a positive integer")
    total_k = int(total_k)
    if prefix is not None:
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a DNA string or None")
        prefix = prefix.upper()
        if not prefix:
            prefix = None
        elif len(prefix) >= total_k:
            raise ValueError("prefix length must be smaller than total_k")
        elif any(base not in _DNA_TO_DIGIT for base in prefix):
            raise ValueError("prefix must contain only A, C, G, and T")

    mode = _normalise_normalization(normalization)
    expected_rows: int | None
    dynamic_ids = False
    if isinstance(genomes, Mapping):
        if genome_ids is not None:
            raise ValueError("genome_ids cannot be supplied when genomes is a mapping")
        genome_keys = list(genomes.keys())
        ids = [str(genome_id) for genome_id in genome_keys]
        genome_values = (genomes[genome_id] for genome_id in genome_keys)
        expected_rows = len(ids)
    elif isinstance(genomes, (str, bytes)):
        # Treat one sequence string as one genome rather than iterating over
        # its individual bases.
        genome_values = iter([genomes])
        if genome_ids is None:
            ids = ["0"]
        else:
            if len(genome_ids) != 1:
                raise ValueError("genome_ids must have one entry per genome")
            ids = [str(genome_ids[0])]
        expected_rows = 1
    else:
        genome_values = iter(genomes)
        if genome_ids is None:
            ids = []
            expected_rows = None
            dynamic_ids = True
        else:
            ids = [str(genome_id) for genome_id in genome_ids]
            expected_rows = len(ids)

    suffix_length = total_k if prefix is None else total_k - len(prefix)
    feature_space_size = 4**suffix_length
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[int] = []
    raw_token_counts: list[int] = []
    valid_window_counts: list[int] = []
    candidate_window_counts: list[int] = []

    count_context = (
        stage_callback("tokenization_and_counting")
        if stage_callback is not None
        else nullcontext()
    )
    row_count = 0
    with count_context:
        for row, (row_counts, raw_count, valid_count, candidate_count) in enumerate(
            _count_genomes(genome_values, total_k, prefix, reverse_complement, workers)
        ):
            if expected_rows is not None and row >= expected_rows:
                raise ValueError("genome_ids must have one entry per genome")
            if dynamic_ids:
                ids.append(str(row))
            for column, count in row_counts.items():
                row_indices.append(row)
                column_indices.append(column)
                values.append(count)
            raw_token_counts.append(raw_count)
            valid_window_counts.append(valid_count)
            candidate_window_counts.append(candidate_count)
            row_count = row + 1
            if genome_counted is not None:
                genome_counted()

    if expected_rows is not None and row_count != expected_rows:
        raise ValueError("genome_ids must have one entry per genome")

    data_type = np.float32 if mode == "frequency" else np.int64
    matrix_context = (
        stage_callback("matrix_construction")
        if stage_callback is not None
        else nullcontext()
    )
    with matrix_context:
        matrix = sparse.coo_matrix(
            (np.asarray(values, dtype=data_type), (row_indices, column_indices)),
            shape=(row_count, feature_space_size),
            dtype=data_type,
        ).tocsr()
        if mode == "frequency" and matrix.nnz:
            # Use every valid candidate window as the common denominator.  This
            # keeps full and filtered representations directly comparable and
            # leaves prefix retention visible in feature magnitude.
            denominators = np.asarray(valid_window_counts, dtype=np.float32)
            nonempty_rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
            matrix.data /= denominators[nonempty_rows]

    metadata = {
        "dataset": str(dataset),
        "genome_ids": ids,
        "total_k": total_k,
        "prefix": prefix,
        "prefix_length": 0 if prefix is None else len(prefix),
        "suffix_length": suffix_length,
        "feature_space_size": feature_space_size,
        "normalization": mode,
        "reverse_complement": bool(reverse_complement),
        "raw_token_counts": raw_token_counts,
        "retained_token_counts": raw_token_counts,
        "valid_window_counts": valid_window_counts,
        "candidate_window_counts": candidate_window_counts,
        "retention_fraction": [
            (raw / valid) if valid else 0.0
            for raw, valid in zip(raw_token_counts, valid_window_counts)
        ],
        "normalization_denominator": valid_window_counts,
        "matrix_shape": list(matrix.shape),
        "nnz": int(matrix.nnz),
    }
    return SparseKmerCounts(matrix=matrix, metadata=metadata)


def save_sparse_kmer_counts(
    result: SparseKmerCounts, path: str | Path, dataset: str | None = None
) -> Path:
    """Functional counterpart to :meth:`SparseKmerCounts.save`."""

    return result.save(path, dataset=dataset)


def load_sparse_kmer_counts(path: str | Path) -> SparseKmerCounts:
    """Load a matrix and its adjacent JSON metadata sidecar."""

    matrix_path = Path(path)
    if matrix_path.is_dir():
        matches = sorted(matrix_path.glob("*.npz"))
        if len(matches) != 1:
            raise ValueError("a directory load requires exactly one sparse k-mer NPZ")
        matrix_path = matches[0]
    if matrix_path.suffix.lower() != ".npz":
        matrix_path = matrix_path.with_suffix(".npz")
    metadata_path = matrix_path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"sparse k-mer metadata not found: {metadata_path}")
    matrix = sparse.load_npz(matrix_path).tocsr()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return SparseKmerCounts(matrix=matrix, metadata=metadata)


__all__ = [
    "SparseKmerCounts",
    "build_sparse_kmer_counts",
    "decode_kmer",
    "encode_kmer",
    "load_sparse_kmer_counts",
    "save_sparse_kmer_counts",
]
