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
    script = Path("scripts/run_week3_person1_scaling_benchmark.py")
    assert script.exists(), f"Script does not exist: {script}"


def test_week3_scaling_benchmark_smoke(tmp_path: Path) -> None:
    output = tmp_path / "week3_scaling_benchmark_smoke.json"
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_week3_person1_scaling_benchmark",
        "--shots",
        "40",
        "--repeats",
        "1",
        "--keep-soft",
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
    assert payload.get("metadata", {}).get("report_name") == "week3_person1_scaling_benchmark"
    rows = payload.get("cases_summary", [])
    assert isinstance(rows, list) and rows
    means = rows[0].get("means", {})
    assert "speedup_fast_vs_standard_decode_time" in means
    assert "speedup_fast_vs_standard_wall_time" in means


def test_invalid_shots_fails(tmp_path: Path) -> None:
    output = tmp_path / "invalid_shots.json"
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_week3_person1_scaling_benchmark",
        "--shots",
        "0",
        "--output",
        str(output),
    ]
    result = _run_cmd(cmd, timeout=180)
    assert result.returncode != 0, (
        "Expected failure with shots=0, but got return code 0.\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )
