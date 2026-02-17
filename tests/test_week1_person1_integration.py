# tests/test_week1_person1_integration.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# If dependencies are missing, skip the whole file.
pytest.importorskip("stim")
pytest.importorskip("pymatching")

from src.pipelines.week1_person1_pipeline import (
    Week1Person1PipelineConfig,
    run_and_save_week1_person1_pipeline,
    run_week1_person1_pipeline,
    save_pipeline_result,
)


def test_pipeline_e2e_output_contract() -> None:
    """
    Main E2E test:
    - Pipeline runs end-to-end
    - Returns expected structure
    - Metrics stay within valid ranges
    """
    cfg = Week1Person1PipelineConfig(
        distance=3,
        rounds=2,
        noise_model="depolarizing",
        p=0.01,
        logical_basis="x",
        shots=40,
        keep_soft_info_samples=15,
        reference_ler_shots=40,
    )

    result = run_week1_person1_pipeline(cfg)

    assert isinstance(result, dict)
    assert result.get("status") == "ok"

    # Main blocks
    assert "metadata" in result
    assert "config" in result
    assert "circuit_summary" in result
    assert "benchmark" in result
    assert "reference" in result

    # Metadata
    metadata = result["metadata"]
    assert metadata["pipeline"] == "week1_person1_pipeline"
    assert "timestamp_utc" in metadata

    # Echoed config
    config_out = result["config"]
    assert config_out["distance"] == 3
    assert config_out["rounds"] == 2
    assert config_out["shots"] == 40

    # Circuit summary
    csum = result["circuit_summary"]
    assert csum["num_qubits"] > 0
    assert csum["num_measurements"] > 0
    assert csum["num_detectors"] > 0
    assert csum["num_observables"] >= 1

    # Benchmark
    bench = result["benchmark"]
    expected_bench_keys = {
        "shots",
        "num_detectors",
        "num_observables",
        "error_rate",
        "avg_decode_time",
        "soft_info_samples",
    }
    assert expected_bench_keys.issubset(bench.keys())

    assert bench["shots"] == 40
    assert isinstance(bench["avg_decode_time"], float)
    assert bench["avg_decode_time"] >= 0.0
    assert isinstance(bench["soft_info_samples"], list)
    assert len(bench["soft_info_samples"]) <= 15

    er = bench["error_rate"]
    assert isinstance(er, float)
    assert (0.0 <= er <= 1.0) or np.isnan(er)

    # Reference
    ref = result["reference"]
    assert "ler_mwpm_helper" in ref
    assert "reference_ler_shots" in ref
    assert ref["reference_ler_shots"] == 40
    assert 0.0 <= ref["ler_mwpm_helper"] <= 1.0


def test_pipeline_with_noise_dict_runs() -> None:
    """
    Verify compatibility with dict-type noise_model (Stim kwargs).
    """
    cfg = Week1Person1PipelineConfig(
        distance=3,
        rounds=2,
        noise_model={"after_clifford_depolarization": 0.01},
        p=0.01,
        logical_basis="x",
        shots=20,
        keep_soft_info_samples=10,
        reference_ler_shots=20,
    )

    result = run_week1_person1_pipeline(cfg)
    assert result["status"] == "ok"
    assert result["benchmark"]["shots"] == 20


def test_pipeline_ideal_noise_is_low_error() -> None:
    """
    In the ideal case (no noise), logical error rate should be very low/near zero.
    Keep a small margin for environment robustness.
    """
    cfg = Week1Person1PipelineConfig(
        distance=3,
        rounds=3,
        noise_model="none",
        p=0.0,
        logical_basis="x",
        shots=50,
        keep_soft_info_samples=20,
        reference_ler_shots=50,
    )

    result = run_week1_person1_pipeline(cfg)
    bench_er = result["benchmark"]["error_rate"]
    ref_er = result["reference"]["ler_mwpm_helper"]

    # In ideal conditions this should be near 0
    assert bench_er <= 0.05
    assert ref_er <= 0.05


def test_pipeline_invalid_distance_raises() -> None:
    """
    Even distance should propagate ValueError from generate_xzzx_circuit.
    """
    cfg = Week1Person1PipelineConfig(
        distance=4,  # invalid (must be odd >= 3)
        rounds=2,
        noise_model="depolarizing",
        p=0.01,
        logical_basis="x",
        shots=10,
        keep_soft_info_samples=5,
        reference_ler_shots=10,
    )

    with pytest.raises(ValueError):
        _ = run_week1_person1_pipeline(cfg)


def test_save_pipeline_result_creates_json(tmp_path: Path) -> None:
    """
    save_pipeline_result must create a valid JSON file on disk.
    """
    cfg = Week1Person1PipelineConfig(
        distance=3,
        rounds=2,
        noise_model="depolarizing",
        p=0.01,
        logical_basis="x",
        shots=15,
        keep_soft_info_samples=5,
        reference_ler_shots=15,
    )
    result = run_week1_person1_pipeline(cfg)

    out_path = tmp_path / "results" / "week1_person1_result.json"
    saved = save_pipeline_result(result, out_path)

    assert saved.exists()
    assert saved.is_file()

    with saved.open("r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["status"] == "ok"
    assert loaded["config"]["distance"] == 3
    assert "benchmark" in loaded


def test_run_and_save_pipeline_wrapper(tmp_path: Path) -> None:
    """
    run_and_save_week1_person1_pipeline must run and save in one call.
    """
    cfg = Week1Person1PipelineConfig(
        distance=3,
        rounds=2,
        noise_model="depolarizing",
        p=0.01,
        logical_basis="x",
        shots=12,
        keep_soft_info_samples=4,
        reference_ler_shots=12,
    )

    out_path = tmp_path / "out" / "pipeline.json"
    result = run_and_save_week1_person1_pipeline(cfg, out_path)

    assert result["status"] == "ok"
    assert out_path.exists()

    with out_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["config"]["shots"] == 12
    assert loaded["benchmark"]["shots"] == 12
