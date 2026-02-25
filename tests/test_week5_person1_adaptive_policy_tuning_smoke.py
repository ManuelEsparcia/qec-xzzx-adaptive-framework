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
    script = Path("scripts/run_week5_person1_adaptive_policy_tuning.py")
    assert script.exists(), f"Script does not exist: {script}"


def test_week5_adaptive_policy_tuning_smoke(tmp_path: Path) -> None:
    output = tmp_path / "week5_adaptive_policy_tuning_smoke.json"
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_week5_person1_adaptive_policy_tuning",
        "--distances",
        "3,5",
        "--rounds-mode",
        "distance",
        "--noise-model",
        "depolarizing",
        "--p-phys",
        "0.01",
        "--shots",
        "20",
        "--repeats",
        "1",
        "--seed",
        "12345",
        "--g-thresholds",
        "0.35,0.65",
        "--fast-backends",
        "uf",
        "--min-switch-weights",
        "none,2",
        "--mode",
        "fast",
        "--time-metric",
        "core",
        "--max-delta-error",
        "0.05",
        "--min-speedup",
        "0.5",
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
    assert payload.get("metadata", {}).get("report_name") == "week5_person1_adaptive_policy_tuning"
    assert payload.get("config", {}).get("time_metric") == "core"

    rows = payload.get("rows", [])
    assert isinstance(rows, list) and rows
    assert len(rows) == 8

    policy_summary = payload.get("policy_summary", [])
    assert isinstance(policy_summary, list) and policy_summary

    best_global = payload.get("best_global_policy")
    assert isinstance(best_global, dict)
    assert "policy_label" in best_global


def test_invalid_min_speedup_fails(tmp_path: Path) -> None:
    output = tmp_path / "invalid_min_speedup.json"
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_week5_person1_adaptive_policy_tuning",
        "--shots",
        "10",
        "--repeats",
        "1",
        "--min-speedup",
        "0",
        "--output",
        str(output),
    ]
    result = _run_cmd(cmd, timeout=180)
    assert result.returncode != 0, (
        "Expected failure with min-speedup=0, but got return code 0.\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )
