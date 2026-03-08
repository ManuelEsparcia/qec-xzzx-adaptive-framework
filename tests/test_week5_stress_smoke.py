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


def _run_cmd(cmd: list[str], timeout: int = 420) -> subprocess.CompletedProcess[str]:
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


def test_week5_stress_smoke() -> None:
    output = _out_path("week5_stress_smoke")
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_week5_stress_test",
        "--distances",
        "11,13",
        "--decoders",
        "mwpm,uf,bm,adaptive",
        "--shots",
        "20",
        "--repeats",
        "1",
        "--seed",
        "12345",
        "--output",
        str(output),
    ]
    result = _run_cmd(cmd, timeout=600)
    assert result.returncode == 0, (
        "Stress script failed.\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )
    assert output.exists(), "Stress output JSON was not created."

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload.get("metadata", {}).get("report_name") == "week5_stress_test"
    assert isinstance(payload.get("schema_version"), str) and payload["schema_version"]

    rows = payload.get("rows", [])
    assert isinstance(rows, list) and rows

    distances = {int(r.get("distance")) for r in rows}
    decoders = {str(r.get("decoder")) for r in rows}
    assert distances == {11, 13}
    assert {"mwpm", "uf", "bm", "adaptive"} == decoders

    for row in rows:
        assert "status" in row
        assert "pass" in row
        metrics = row.get("metrics", {})
        assert "avg_decode_time_sec" in metrics
        assert "memory_peak_bytes" in metrics
        assert "error_rate" in metrics
