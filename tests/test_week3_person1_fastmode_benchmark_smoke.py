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
    script = Path("scripts/run_week3_person1_fastmode_benchmark.py")
    assert script.exists(), f"Script does not exist: {script}"


def test_week3_fastmode_benchmark_smoke(tmp_path: Path) -> None:
    output = tmp_path / "week3_fastmode_benchmark_smoke.json"
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_week3_person1_fastmode_benchmark",
        "--shots",
        "40",
        "--repeats",
        "2",
        "--keep-samples",
        "5",
        "--g-threshold",
        "0.35",
        "--seed",
        "12345",
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
    assert isinstance(payload, dict)
    md = payload.get("metadata", {})
    assert md.get("report_name") == "week3_person1_fastmode_benchmark"
    assert int(md.get("shots", -1)) == 40
    assert int(md.get("repeats", -1)) == 2

    rows = payload.get("cases_summary", [])
    assert isinstance(rows, list) and rows
    first = rows[0]
    assert "means" in first
    assert "repeats_summary" in first
    reps = first["repeats_summary"]
    assert isinstance(reps, list) and reps
    rep0 = reps[0]
    assert rep0["standard"]["fast_mode"] is False
    assert rep0["fast"]["fast_mode"] is True


def test_invalid_repeats_fail(tmp_path: Path) -> None:
    output = tmp_path / "invalid_repeats.json"
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_week3_person1_fastmode_benchmark",
        "--shots",
        "20",
        "--repeats",
        "0",
        "--output",
        str(output),
    ]
    result = _run_cmd(cmd, timeout=180)
    assert result.returncode != 0, (
        "Expected failure with repeats=0, but got return code 0.\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )
