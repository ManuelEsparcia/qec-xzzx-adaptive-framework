from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL = REPO_ROOT / "results" / "week3_person2_threshold_scan_full.json"


def test_canonical_artifact_exists() -> None:
    assert CANONICAL.exists(), f"Canonical Week 3 artifact missing: {CANONICAL}"


def test_canonical_artifact_schema_current() -> None:
    payload = json.loads(CANONICAL.read_text(encoding="utf-8"))
    md = payload.get("metadata", {})
    cfg = payload.get("config", {})
    points = payload.get("points", [])
    thresholds = payload.get("threshold_estimates", [])

    assert md.get("report_name") == "week3_person2_threshold_scan"
    assert md.get("schema_version") == "week3_scan_v2"
    assert "grid_expected_points" in md
    assert "grid_completed_points" in md
    assert "covered_distances" in md
    assert "covered_decoders" in md
    assert "covered_noise_models" in md

    distances = sorted(int(x) for x in cfg.get("distances", []))
    decoders = sorted(str(x) for x in cfg.get("decoders", []))
    noise_models = sorted(str(x) for x in cfg.get("noise_models", []))
    p_values = [float(x) for x in cfg.get("p_values", [])]

    assert distances == [3, 5, 7, 9, 11, 13]
    assert decoders == ["adaptive", "bm", "mwpm", "uf"]
    assert noise_models == [
        "biased_eta10",
        "biased_eta100",
        "biased_eta500",
        "circuit_level",
        "correlated",
        "depolarizing",
    ]
    assert 8 <= len(p_values) <= 10
    assert max(p_values) >= 0.03
    assert cfg.get("threshold_method") in {"crossing", "crossing_then_fit", "fit_only"}

    assert isinstance(points, list) and points
    p0 = points[0]
    for key in (
        "decoder",
        "noise_model",
        "distance",
        "rounds",
        "p_phys",
        "shots",
        "seed",
        "repeats",
        "repeat_runs",
        "error_rate",
        "avg_decode_time_sec",
        "switch_rate",
        "error_rate_ci95_half_width",
        "avg_decode_time_sec_ci95_half_width",
        "status",
    ):
        assert key in p0, f"Missing point key: {key}"

    assert isinstance(thresholds, list) and thresholds
    t0 = thresholds[0]
    for key in (
        "decoder",
        "noise_model",
        "distance_pair",
        "pair_scope",
        "crossing_detected",
        "threshold_method_preference",
        "method",
        "fit_method",
        "p_threshold_estimate",
        "fallback_used",
        "fallback_reason",
        "quality_level",
        "threshold_quality",
    ):
        assert key in t0, f"Missing threshold-estimate key: {key}"
