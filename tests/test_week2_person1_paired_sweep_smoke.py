# tests/test_week2_person1_paired_sweep_smoke.py
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("stim")
pytest.importorskip("pymatching")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_week2_person1_paired_threshold_sweep.py"


def test_script_exists() -> None:
    assert SCRIPT.exists(), f"Expected script does not exist: {SCRIPT}"


def test_paired_sweep_script_smoke(tmp_path: Path) -> None:
    out = tmp_path / "paired_sweep.json"

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--shots",
        "40",
        "--thresholds",
        "0.2,0.6",
        "--output",
        str(out),
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, (
        f"Script failed.\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
    )
    assert out.exists(), "Output JSON was not generated."

    data = json.loads(out.read_text(encoding="utf-8"))

    # Metadata
    md = data.get("metadata", {})
    assert md.get("report_name") == "week2_person1_paired_threshold_sweep"
    assert md.get("shots_per_case") == 40
    assert md.get("thresholds") == [0.2, 0.6]

    # Cases
    cases = data.get("cases_summary", [])
    assert len(cases) == 3

    for c in cases:
        assert c.get("status") == "ok"
        assert "mwpm" in c and "uf" in c and "adaptive_by_threshold" in c

        mwpm_er = float(c["mwpm"]["error_rate"])
        uf_er = float(c["uf"]["error_rate"])
        assert 0.0 <= mwpm_er <= 1.0
        assert 0.0 <= uf_er <= 1.0

        adp_rows = c["adaptive_by_threshold"]
        assert len(adp_rows) == 2
        for r in adp_rows:
            g = float(r["g_threshold"])
            er = float(r["error_rate"])
            t = float(r["avg_decode_time_sec"])
            sw = float(r["switch_rate"])

            assert 0.0 <= g <= 1.0
            assert 0.0 <= er <= 1.0
            assert t >= 0.0
            assert 0.0 <= sw <= 1.0

    # Minimum aggregates
    agg = data.get("aggregates", {})
    assert "mean_mwpm_error_rate" in agg
    assert "mean_uf_error_rate" in agg
    assert "adaptive_means_by_threshold" in agg
    assert len(agg["adaptive_means_by_threshold"]) == 2


def test_invalid_thresholds_fail(tmp_path: Path) -> None:
    out = tmp_path / "bad.json"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--shots",
        "10",
        "--thresholds",
        "1.2",  # invalid
        "--output",
        str(out),
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
