from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("stim")
pytest.importorskip("pymatching")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_week4_hardware_compatibility.py"


def _run_cmd(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _tmp_output_path(stem: str, suffix: str) -> Path:
    out_dir = REPO_ROOT / "results" / "_tmp_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}_{uuid4().hex}{suffix}"


def test_script_exists() -> None:
    assert SCRIPT.exists(), f"Script does not exist: {SCRIPT}"


def test_week4_hardware_compatibility_smoke() -> None:
    output_json = _tmp_output_path("week4_hardware_compatibility_smoke", ".json")
    output_fig = _tmp_output_path("week4_hardware_compatibility_smoke", ".png")
    cmd = [
        sys.executable,
        str(SCRIPT),
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
        "--output",
        str(output_json),
        "--figure-output",
        str(output_fig),
    ]
    result = _run_cmd(cmd, timeout=360)
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

    assert md.get("report_name") == "week4_hardware_compatibility"
    assert md.get("schema_version") == "week4_hw_v2"
    assert "coverage" in md
    assert "latency_source_mode" in md
    assert cfg.get("adaptive_fast_mode") is True
    assert cfg.get("benchmark_distances") == [3, 5]
    assert cfg.get("distances") == [3, 5]

    row_labels = comp.get("decoder_labels", [])
    assert "mwpm" in row_labels
    assert "uf" in row_labels
    assert "adaptive_g0.20" in row_labels
    assert "adaptive_g0.50" in row_labels

    archs = comp.get("architectures", [])
    assert len(archs) == 4
    for arch in archs:
        assert "matrix" in arch
        assert "compatibility_ratio" in arch

    latency_source = payload.get("latency_source", {})
    assert latency_source.get("mode") == "benchmarked_plus_fitted_extrapolation"
    assert isinstance(latency_source.get("fit_provenance_summary", []), list)

