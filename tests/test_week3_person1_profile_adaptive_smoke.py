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
    script = Path("scripts/run_week3_person1_profile_adaptive.py")
    assert script.exists(), f"Script does not exist: {script}"


def test_week3_profile_adaptive_smoke(tmp_path: Path) -> None:
    output = tmp_path / "week3_profile_adaptive_smoke.json"
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_week3_person1_profile_adaptive",
        "--distance",
        "5",
        "--rounds",
        "3",
        "--p",
        "0.01",
        "--shots",
        "30",
        "--keep-samples",
        "3",
        "--g-threshold",
        "0.35",
        "--seed",
        "12345",
        "--top-n",
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
    assert payload.get("metadata", {}).get("report_name") == "week3_person1_adaptive_profile"
    assert payload.get("modes", {}).get("standard", {}).get("summary", {}).get("fast_mode") is False
    assert payload.get("modes", {}).get("fast", {}).get("summary", {}).get("fast_mode") is True
    assert "comparisons" in payload


def test_invalid_top_n_fails(tmp_path: Path) -> None:
    output = tmp_path / "invalid_top_n.json"
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_week3_person1_profile_adaptive",
        "--shots",
        "20",
        "--top-n",
        "0",
        "--output",
        str(output),
    ]
    result = _run_cmd(cmd, timeout=180)
    assert result.returncode != 0, (
        "Expected failure with top-n=0, but got return code 0.\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )
