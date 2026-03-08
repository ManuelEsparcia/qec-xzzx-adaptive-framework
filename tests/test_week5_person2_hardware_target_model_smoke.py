from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

pytest.importorskip("stim")
pytest.importorskip("pymatching")

TMP_ROOT = Path(".tmp_week5_tests")
TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _out_path(stem: str, suffix: str = ".json") -> Path:
    return TMP_ROOT / f"{stem}_{uuid.uuid4().hex}{suffix}"


def _run_cmd(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    tmp = str(TMP_ROOT.resolve())
    env["TMP"] = tmp
    env["TEMP"] = tmp
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )


def test_script_exists() -> None:
    script = Path("scripts/run_week5_person2_hardware_target_model.py")
    assert script.exists(), f"Script does not exist: {script}"


def test_week5_hardware_target_model_smoke() -> None:
    output_json = _out_path("week5_hardware_target_model_smoke")
    output_fig = _out_path("week5_hardware_target_model_smoke", ".png")
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_week5_person2_hardware_target_model",
        "--benchmark-distances",
        "3,5",
        "--distances",
        "3,5",
        "--rounds-mode",
        "fixed",
        "--rounds",
        "2",
        "--adaptive-thresholds",
        "0.20,0.50",
        "--include-bm",
        "--shots",
        "20",
        "--repeats",
        "1",
        "--seed",
        "12345",
        "--noise-model",
        "depolarizing",
        "--p-phys",
        "0.01",
        "--adaptive-fast-mode",
        "--adaptive-fast-backend",
        "bm",
        "--adaptive-min-switch-weight",
        "none",
        "--output",
        str(output_json),
        "--figure-output",
        str(output_fig),
    ]
    result = _run_cmd(cmd, timeout=420)
    assert result.returncode == 0, (
        "Script failed.\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )

    assert output_json.exists(), "Output JSON was not created."
    assert output_fig.exists(), "Output figure was not created."

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    md = payload.get("metadata", {})
    cfg = payload.get("config", {})
    comp = payload.get("compatibility", {})

    assert md.get("report_name") == "week5_person2_hardware_target_model"
    assert cfg.get("adaptive_fast_backend") == "bm"
    assert cfg.get("benchmark_distances") == [3, 5]
    assert cfg.get("distances") == [3, 5]
    assert cfg.get("include_bm") is True

    row_labels = comp.get("decoder_labels", [])
    assert "mwpm" in row_labels
    assert "uf" in row_labels
    assert "bm" in row_labels
    assert "adaptive_g0.20" in row_labels
    assert "adaptive_g0.50" in row_labels

    archs = comp.get("architectures", [])
    assert len(archs) == 4
    for arch in archs:
        assert "python_runtime_matrix" in arch
        assert "target_model_matrix" in arch
        assert "python_runtime_ratio" in arch
        assert "target_model_ratio" in arch


def test_week5_hardware_target_model_trace_calibration_smoke() -> None:
    output_json = _out_path("week5_hardware_target_model_trace_smoke")
    output_fig = _out_path("week5_hardware_target_model_trace_smoke", ".png")
    trace_json = _out_path("trace_rows")
    trace_rows = [
        {
            "architecture": "GoogleSuperconducting",
            "backend": "mwpm",
            "distance": 3,
            "observed_decode_time_sec": 2.1e-6,
        },
        {
            "architecture": "GoogleSuperconducting",
            "backend": "uf",
            "distance": 3,
            "observed_decode_time_sec": 1.6e-6,
        },
        {
            "architecture": "GoogleSuperconducting",
            "backend": "mwpm",
            "distance": 5,
            "observed_decode_time_sec": 3.3e-6,
        },
        {
            "architecture": "GoogleSuperconducting",
            "backend": "adaptive",
            "distance": 5,
            "switch_rate": 0.3,
            "observed_decode_time_sec": 2.7e-6,
        },
    ]
    trace_json.write_text(json.dumps(trace_rows, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "scripts.run_week5_person2_hardware_target_model",
        "--benchmark-distances",
        "3,5",
        "--distances",
        "3,5",
        "--rounds-mode",
        "fixed",
        "--rounds",
        "2",
        "--adaptive-thresholds",
        "0.40",
        "--shots",
        "16",
        "--repeats",
        "1",
        "--seed",
        "12345",
        "--noise-model",
        "depolarizing",
        "--p-phys",
        "0.01",
        "--adaptive-fast-mode",
        "--adaptive-fast-backend",
        "uf",
        "--trace-input",
        str(trace_json),
        "--output",
        str(output_json),
        "--figure-output",
        str(output_fig),
    ]
    result = _run_cmd(cmd, timeout=420)
    assert result.returncode == 0, (
        "Script failed with trace calibration.\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    cfg = payload.get("config", {})
    trace_observations = payload.get("trace_observations", [])
    trace_cal = payload.get("trace_calibration", {})
    arch_rows = trace_cal.get("architectures", [])

    assert cfg.get("trace_input") == str(trace_json)
    assert len(trace_observations) == len(trace_rows)
    assert trace_cal.get("enabled") is True
    assert trace_cal.get("num_trace_rows") == len(trace_rows)
    google_row = next(
        row for row in arch_rows if row.get("architecture") == "GoogleSuperconducting"
    )
    assert google_row.get("calibrated") is True
    assert any(bool(row.get("calibrated")) for row in arch_rows)


def test_invalid_ops_factor_fails() -> None:
    output_json = _out_path("invalid_ops_factor")
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_week5_person2_hardware_target_model",
        "--ops-factor-mwpm",
        "0",
        "--output",
        str(output_json),
    ]
    result = _run_cmd(cmd, timeout=180)
    assert result.returncode != 0, (
        "Expected failure with ops-factor-mwpm=0, but got return code 0.\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )
