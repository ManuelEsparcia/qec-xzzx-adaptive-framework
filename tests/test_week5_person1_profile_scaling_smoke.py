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
    script = Path("scripts/run_week5_person1_profile_scaling.py")
    assert script.exists(), f"Script does not exist: {script}"


def test_week5_profile_scaling_smoke() -> None:
    output = _out_path("week5_profile_scaling_smoke")
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_week5_person1_profile_scaling",
        "--distances",
        "3,5",
        "--rounds-mode",
        "distance",
        "--decoders",
        "mwpm,adaptive",
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
        "--g-threshold",
        "0.35",
        "--profile-decoder",
        "adaptive",
        "--profile-distance",
        "5",
        "--profile-shots",
        "15",
        "--profile-top-n",
        "5",
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
    assert payload.get("metadata", {}).get("report_name") == "week5_person1_profile_scaling"

    rows = payload.get("rows", [])
    assert isinstance(rows, list) and rows
    assert len(rows) == 4

    first_metrics = rows[0].get("metrics", {})
    assert "avg_decode_time_sec" in first_metrics
    assert "memory_peak_bytes" in first_metrics
    assert first_metrics["memory_peak_bytes"]["mean"] >= 0.0

    models = payload.get("scaling_models", [])
    assert isinstance(models, list) and len(models) == 2
    assert {"mwpm", "adaptive"} == {m.get("decoder") for m in models}

    profile = payload.get("profile_hotspots", {}).get("profile", {})
    top = profile.get("top_cumulative", [])
    assert isinstance(top, list) and top


def test_invalid_profile_distance_fails() -> None:
    output = _out_path("invalid_profile_distance")
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_week5_person1_profile_scaling",
        "--distances",
        "3,5",
        "--profile-distance",
        "4",
        "--shots",
        "10",
        "--repeats",
        "1",
        "--output",
        str(output),
    ]
    result = _run_cmd(cmd, timeout=180)
    assert result.returncode != 0, (
        "Expected failure with profile-distance=4, but got return code 0.\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )
