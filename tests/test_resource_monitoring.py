import time

from utilities.resource_monitoring import (
    RESOURCE_FIELDS,
    ResourceMonitor,
    ingest_slurm_record,
    measure_stage,
)


def test_stage_measurement_has_consistent_cpu_and_memory_schema():
    with measure_stage("counting", condition_id="full-k8") as stage:
        time.sleep(0.002)

    record = stage.record
    assert set(RESOURCE_FIELDS).issubset(record)
    assert record["stage"] == "counting"
    assert record["condition_id"] == "full-k8"
    assert record["wall_time_seconds"] >= 0
    assert record["cpu_time_seconds"] >= 0
    assert record["peak_rss_bytes"] > 0
    assert record["gpu_peak_allocated_bytes"] is None
    assert record["gpu_peak_reserved_bytes"] is None


def test_resource_monitor_can_record_multiple_stages_and_export_records():
    monitor = ResourceMonitor(condition_id="pilot")
    with monitor.stage("loading"):
        pass
    with monitor.stage("fitting"):
        pass
    assert [row["stage"] for row in monitor.records] == ["loading", "fitting"]
    assert all(row["condition_id"] == "pilot" for row in monitor.records)


def test_slurm_ingestion_normalizes_elapsed_and_maxrss():
    row = ingest_slurm_record("00:01:02", "1.5G", job_id="123")
    assert row["slurm_job_id"] == "123"
    assert row["slurm_elapsed_seconds"] == 62
    assert row["slurm_max_rss_bytes"] == int(1.5 * 1024**3)
