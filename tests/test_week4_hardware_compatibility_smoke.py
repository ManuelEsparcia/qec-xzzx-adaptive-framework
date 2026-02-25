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
    script = Path("scripts/run_week4_hardware_compatibility.py")
    assert script.exists(), f"Script does not exist: {script}"


def test_week4_hardware_compatibility_smoke(tmp_path: Path) -> None:
    output_json = tmp_path / "week4_hardware_compatibility_smoke.json"
    output_fig = tmp_path / "week4_hardware_compatibility_smoke.png"
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_week4_hardware_compatibility",
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

