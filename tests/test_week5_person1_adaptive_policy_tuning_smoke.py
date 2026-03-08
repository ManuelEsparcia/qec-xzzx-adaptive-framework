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
    script = Path("scripts/run_week5_person1_adaptive_policy_tuning.py")
    assert script.exists(), f"Script does not exist: {script}"


def test_week5_adaptive_policy_tuning_smoke() -> None:
    output = _out_path("week5_adaptive_policy_tuning_smoke")
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


def test_invalid_min_speedup_fails() -> None:
    output = _out_path("invalid_min_speedup")
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
