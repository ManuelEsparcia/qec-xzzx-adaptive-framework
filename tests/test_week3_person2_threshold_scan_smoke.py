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
    script = Path("scripts/run_week3_person2_threshold_scan.py")
    assert script.exists(), f"Script does not exist: {script}"


def test_week3_person2_threshold_scan_smoke(tmp_path: Path) -> None:
    output = tmp_path / "week3_person2_threshold_scan_smoke.json"
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_week3_person2_threshold_scan",
        "--distances",
        "3,5",
        "--rounds",
        "2",
        "--p-values",
        "0.005,0.01",
        "--decoders",
        "mwpm,adaptive",
        "--noise-models",
        "depolarizing,biased_eta10",
        "--shots",
        "20",
        "--repeats",
        "2",
        "--seed",
        "12345",
        "--g-threshold",
        "0.35",
        "--checkpoint-every",
        "2",
        "--adaptive-fast-mode",
        "--output",
        str(output),
    ]
    result = _run_cmd(cmd, timeout=300)
    assert result.returncode == 0, (
        "Script failed.\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )
    assert output.exists(), "Output JSON was not created."

    payload = json.loads(output.read_text(encoding="utf-8"))
    md = payload.get("metadata", {})
    cfg = payload.get("config", {})
    assert md.get("report_name") == "week3_person2_threshold_scan"
    assert cfg.get("adaptive_fast_mode") is True
    assert cfg.get("repeats") == 2

    points = payload.get("points", [])
    assert isinstance(points, list) and points
    p0 = points[0]
    for k in ("decoder", "noise_model", "distance", "p_phys", "error_rate", "avg_decode_time_sec"):
        assert k in p0
    for k in (
        "repeats",
        "repeat_runs",
        "error_rate_std",
        "error_rate_ci95_half_width",
        "avg_decode_time_sec_std",
        "avg_decode_time_sec_ci95_half_width",
    ):
        assert k in p0
    assert int(p0["repeats"]) == 2
    assert isinstance(p0["repeat_runs"], list)
    assert len(p0["repeat_runs"]) == 2

    # threshold estimates may be empty for very small stochastic runs,
    # but the key must exist by contract.
    assert "threshold_estimates" in payload
    assert "aggregates" in payload


def test_invalid_decoder_fails(tmp_path: Path) -> None:
    output = tmp_path / "week3_person2_threshold_scan_invalid.json"
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_week3_person2_threshold_scan",
        "--decoders",
        "mwpm,decoder_inexistente",
        "--shots",
        "20",
        "--output",
        str(output),
    ]
    result = _run_cmd(cmd, timeout=180)
    assert result.returncode != 0, (
        "Expected failure with invalid decoder, but got return code 0.\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )


def test_invalid_repeats_fails(tmp_path: Path) -> None:
    output = tmp_path / "week3_person2_threshold_scan_invalid_repeats.json"
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_week3_person2_threshold_scan",
        "--repeats",
        "0",
        "--shots",
        "20",
        "--output",
        str(output),
    ]
    result = _run_cmd(cmd, timeout=180)
    assert result.returncode != 0, (
        "Expected failure with invalid repeats, but got return code 0.\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )
