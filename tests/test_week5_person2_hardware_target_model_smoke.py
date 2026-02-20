from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_cmd(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_script_exists() -> None:
    script = Path("scripts/run_week5_person2_hardware_target_model.py")
    assert script.exists(), f"Script does not exist: {script}"


def test_week5_hardware_target_model_smoke(tmp_path: Path) -> None:
    output_json = tmp_path / "week5_hardware_target_model_smoke.json"
    output_fig = tmp_path / "week5_hardware_target_model_smoke.png"
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


def test_invalid_ops_factor_fails(tmp_path: Path) -> None:
    output_json = tmp_path / "invalid_ops_factor.json"
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
