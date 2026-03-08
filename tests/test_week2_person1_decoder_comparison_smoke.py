from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("stim")
pytest.importorskip("pymatching")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_week2_person1_decoder_comparison.py"


def _tmp_output_path(stem: str) -> Path:
    out_dir = REPO_ROOT / "results" / "_tmp_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}_{uuid4().hex}.json"


def test_script_exists() -> None:
    assert SCRIPT.exists(), f"Expected script does not exist: {SCRIPT}"


def test_decoder_comparison_script_smoke() -> None:
    out = _tmp_output_path("decoder_comparison")

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
    assert md.get("schema_version") == "week2_cmp_v2"

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
            "bm_backend_matcher_build_mode",
            "bm_backend_decode_mode",
            "bm_is_bp_backend",
            "bm_beliefmatching_available",
            "bm_bp_diagnostics_policy",
            "bm_convergence_rate",
            "bm_avg_num_iterations",
            "bm_avg_residual_error",
            "bm_bp_diagnostics_source",
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
        assert isinstance(c["bm_backend_matcher_build_mode"], str)
        assert isinstance(c["bm_backend_decode_mode"], str)
        assert isinstance(c["bm_is_bp_backend"], bool)
        assert isinstance(c["bm_beliefmatching_available"], bool)
        assert isinstance(c["bm_bp_diagnostics_policy"], str)
        assert 0.0 <= float(c["bm_convergence_rate"]) <= 1.0 or float(c["bm_convergence_rate"]) != float(c["bm_convergence_rate"])
        assert float(c["bm_avg_num_iterations"]) >= 0.0 or float(c["bm_avg_num_iterations"]) != float(c["bm_avg_num_iterations"])
        assert 0.0 <= float(c["bm_avg_residual_error"]) <= 1.0 or float(c["bm_avg_residual_error"]) != float(c["bm_avg_residual_error"])
        assert isinstance(c["bm_bp_diagnostics_source"], str)
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
        "mean_bm_convergence_rate",
        "mean_bm_avg_num_iterations",
        "mean_bm_avg_residual_error",
        "mean_adaptive_speedup_vs_mwpm",
        "mean_adaptive_switch_rate",
    ):
        assert k in agg, f"Falta agregado {k}"
