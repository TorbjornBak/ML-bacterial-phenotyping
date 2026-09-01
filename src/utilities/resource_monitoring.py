"""Small, process-safe resource measurements for benchmark stages.

The benchmark is intentionally CPU-first.  GPU values are therefore ``None``
unless a caller explicitly enables GPU measurement and a CUDA device is
available.  All memory values are bytes and all durations are seconds.
"""

from __future__ import annotations

import contextlib
import datetime as _datetime
import math
import os
import re
import resource
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


RESOURCE_FIELDS = (
    "stage",
    "condition_id",
    "wall_time_seconds",
    "cpu_time_seconds",
    "peak_rss_bytes",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "started_at",
    "finished_at",
)


def _self_rss_bytes() -> int:
    """Return the current process' RSS, in bytes, on macOS and Linux."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes, Linux reports KiB.
    return value if os.sys.platform == "darwin" else value * 1024


def _children_max_rss_bytes() -> int:
    """Return the maximum RSS reported for reaped children by the OS."""

    value = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    return value if os.sys.platform == "darwin" else value * 1024


def _cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def _cuda_peaks(enabled: bool) -> tuple[int | None, int | None]:
    if not enabled:
        return None, None
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())
    except (ImportError, RuntimeError, AttributeError):
        return None, None


def _reset_cuda_peaks(enabled: bool) -> None:
    if not enabled:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError, AttributeError):
        return


def _descendant_rss_bytes(pid: int) -> int:
    """Sample a process tree when psutil is installed, with a safe fallback."""

    try:
        import psutil  # optional: avoid making monitoring a hard dependency

        process = psutil.Process(pid)
        return int(process.memory_info().rss + sum(child.memory_info().rss for child in process.children(recursive=True)))
    except (ImportError, OSError, RuntimeError):
        # ``RUSAGE_CHILDREN`` is less precise than live psutil sampling but
        # still captures a worker that has already exited before the next
        # sample.  This keeps process-tree accounting useful without adding a
        # mandatory dependency to the project.
        return max(_self_rss_bytes(), _children_max_rss_bytes())


@dataclass
class StageMeasurement:
    """Context manager result containing one normalized stage record."""

    stage: str
    condition_id: str | None = None
    gpu_enabled: bool = False
    sample_interval_seconds: float = 0.05
    extra: Mapping[str, Any] = field(default_factory=dict)
    record: dict[str, Any] = field(init=False)
    _started: float = field(init=False, repr=False)
    _started_at: str = field(init=False, repr=False)
    _cpu_started: float = field(init=False, repr=False)
    _peak_rss: int = field(init=False, repr=False)
    _stop: threading.Event = field(init=False, repr=False)
    _sampler: threading.Thread | None = field(init=False, default=None, repr=False)

    def __enter__(self) -> "StageMeasurement":
        self._started = time.perf_counter()
        self._started_at = _datetime.datetime.now(_datetime.timezone.utc).isoformat()
        self._cpu_started = _cpu_seconds()
        self._peak_rss = _descendant_rss_bytes(os.getpid())
        _reset_cuda_peaks(self.gpu_enabled)
        self._stop = threading.Event()

        def sample() -> None:
            while not self._stop.wait(max(float(self.sample_interval_seconds), 0.001)):
                self._peak_rss = max(self._peak_rss, _descendant_rss_bytes(os.getpid()))

        # Even when psutil is absent this fallback samples the current process.
        self._sampler = threading.Thread(target=sample, name="resource-monitor", daemon=True)
        self._sampler.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop.set()
        if self._sampler is not None:
            self._sampler.join(timeout=max(self.sample_interval_seconds * 2, 0.1))
        self._peak_rss = max(self._peak_rss, _descendant_rss_bytes(os.getpid()), _self_rss_bytes())
        gpu_allocated, gpu_reserved = _cuda_peaks(self.gpu_enabled)
        finished = time.perf_counter()
        record: dict[str, Any] = {
            "stage": self.stage,
            "condition_id": self.condition_id,
            "wall_time_seconds": float(finished - self._started),
            "cpu_time_seconds": max(0.0, _cpu_seconds() - self._cpu_started),
            "peak_rss_bytes": int(self._peak_rss),
            "gpu_peak_allocated_bytes": gpu_allocated,
            "gpu_peak_reserved_bytes": gpu_reserved,
            "started_at": self._started_at,
            "finished_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        }
        record.update(self.extra)
        self.record = record
        return None

    def __getitem__(self, key: str) -> Any:
        """Allow a completed measurement to be used like its record mapping."""

        return self.record[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.record.get(key, default)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.record)


def measure_stage(
    stage: str,
    condition_id: str | None = None,
    *,
    gpu_enabled: bool = False,
    sample_interval_seconds: float = 0.05,
    **metadata: Any,
) -> StageMeasurement:
    """Measure one stage and expose its record after the context exits."""

    if not stage or not isinstance(stage, str):
        raise ValueError("stage must be a non-empty string")
    return StageMeasurement(stage, condition_id, gpu_enabled, sample_interval_seconds, metadata)


class ResourceMonitor:
    """Collect stage records for one benchmark condition."""

    def __init__(self, condition_id: str | None = None, *, gpu_enabled: bool = False, sample_interval_seconds: float = 0.05):
        self.condition_id = condition_id
        self.gpu_enabled = gpu_enabled
        self.sample_interval_seconds = sample_interval_seconds
        self.records: list[dict[str, Any]] = []

    @contextlib.contextmanager
    def stage(self, stage: str, **metadata: Any) -> Iterator[StageMeasurement]:
        measurement = measure_stage(
            stage,
            self.condition_id,
            gpu_enabled=self.gpu_enabled,
            sample_interval_seconds=self.sample_interval_seconds,
            **metadata,
        )
        try:
            with measurement:
                yield measurement
        finally:
            # Keep a completed (or failed) stage available to a caller that
            # turns failures into machine-readable benchmark records.
            if hasattr(measurement, "record"):
                self.records.append(measurement.record)

    def as_records(self) -> list[dict[str, Any]]:
        return list(self.records)


_MEMORY_UNITS = {"B": 1, "K": 1024, "KB": 1024, "M": 1024**2, "MB": 1024**2, "G": 1024**3, "GB": 1024**3, "T": 1024**4, "TB": 1024**4}


def parse_slurm_max_rss(value: str | int | float | None) -> int | None:
    """Parse Slurm ``MaxRSS`` (e.g. ``1536K``, ``1.5G``) into bytes."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?b?)?\s*", str(value), re.IGNORECASE)
    if not match:
        raise ValueError(f"invalid Slurm MaxRSS value: {value!r}")
    unit = (match.group(2) or "B").upper()
    return int(float(match.group(1)) * _MEMORY_UNITS[unit])


def parse_slurm_elapsed(value: str | int | float | None) -> float | None:
    """Parse Slurm elapsed time (``D-HH:MM:SS``, ``HH:MM:SS``) to seconds."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.upper() in {"", "N/A", "NA", "NONE", "UNKNOWN", "UNLIMITED"}:
        return None
    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        days = int(day_text)
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"invalid Slurm elapsed value: {value!r}")
    hours, minutes, seconds = parts
    return float(days * 86400 + int(hours) * 3600 + int(minutes) * 60 + float(seconds))


def ingest_slurm_record(
    elapsed: str | int | float | None,
    max_rss: str | int | float | None,
    *,
    job_id: str | int | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    """Normalize completed ``sacct`` values for joining to benchmark output."""

    row = {
        "slurm_job_id": None if job_id is None else str(job_id),
        "slurm_elapsed_seconds": parse_slurm_elapsed(elapsed),
        "slurm_max_rss_bytes": parse_slurm_max_rss(max_rss),
    }
    row.update(metadata)
    return row


def ingest_slurm_accounting(record: Mapping[str, Any]) -> dict[str, Any]:
    """Ingest a mapping returned by ``sacct`` using common column spellings."""

    elapsed = record.get("Elapsed", record.get("elapsed", record.get("elapsed_time")))
    max_rss = record.get("MaxRSS", record.get("maxrss", record.get("max_rss")))
    job_id = record.get("JobID", record.get("job_id", record.get("slurm_job_id")))
    return ingest_slurm_record(elapsed, max_rss, job_id=job_id)


# Concise aliases for callers writing a post-job ingestion script.
ingest_slurm = ingest_slurm_accounting

# Compatibility spellings make the post-processing API easy to discover.
parse_slurm_maxrss = parse_slurm_max_rss
monitor_stage = measure_stage
ResourceMeasurement = StageMeasurement
StageResourceMonitor = ResourceMonitor
