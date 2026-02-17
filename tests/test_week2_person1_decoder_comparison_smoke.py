from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("stim")
pytest.importorskip("pymatching")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_week2_person1_decoder_comparison.py"


def test_script_exists() -> None:
    assert SCRIPT.exists(), f"Expected script does not exist: {SCRIPT}"


def test_decoder_comparison_script_smoke(tmp_path: Path) -> None:
    out = tmp_path / "decoder_comparison.json"

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--shots",
        "40",
        "--keep-soft",
        "10",
        "--g-threshold",
        "0.65",
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

    md = data.get("metadata", {})
    assert md.get("report_name") == "week2_person1_decoder_comparison"
    assert md.get("shots_per_case") == 40

    cases = data.get("cases_summary", [])
    assert len(cases) == 3

    for c in cases:
        assert c.get("status") == "ok"
        for k in (
            "mwpm_error_rate",
            "uf_error_rate",
            "bm_error_rate",
            "adaptive_error_rate",
            "mwpm_avg_decode_time_sec",
            "uf_avg_decode_time_sec",
            "bm_avg_decode_time_sec",
            "adaptive_avg_decode_time_sec",
            "uf_speedup_vs_mwpm",
            "bm_speedup_vs_mwpm",
            "adaptive_speedup_vs_mwpm",
            "adaptive_switch_rate",
        ):
            assert k in c, f"Missing key {k} in case {c.get('case_name')}"

        assert 0.0 <= float(c["mwpm_error_rate"]) <= 1.0
        assert 0.0 <= float(c["uf_error_rate"]) <= 1.0
        assert 0.0 <= float(c["bm_error_rate"]) <= 1.0
        assert 0.0 <= float(c["adaptive_error_rate"]) <= 1.0
        assert float(c["mwpm_avg_decode_time_sec"]) >= 0.0
        assert float(c["uf_avg_decode_time_sec"]) >= 0.0
        assert float(c["bm_avg_decode_time_sec"]) >= 0.0
        assert float(c["adaptive_avg_decode_time_sec"]) >= 0.0
        assert 0.0 <= float(c["adaptive_switch_rate"]) <= 1.0

    agg = data.get("aggregates", {})
    for k in (
        "mean_mwpm_error_rate",
        "mean_uf_error_rate",
        "mean_bm_error_rate",
        "mean_adaptive_error_rate",
        "mean_mwpm_avg_decode_time_sec",
        "mean_uf_avg_decode_time_sec",
        "mean_bm_avg_decode_time_sec",
        "mean_adaptive_avg_decode_time_sec",
        "mean_uf_speedup_vs_mwpm",
        "mean_bm_speedup_vs_mwpm",
        "mean_adaptive_speedup_vs_mwpm",
        "mean_adaptive_switch_rate",
    ):
        assert k in agg, f"Falta agregado {k}"
