from __future__ import annotations

import json
from pathlib import Path


def test_week5_profile_schema_regression() -> None:
    artifact = Path("results/week5_person1_profile_scaling.json")
    assert artifact.exists(), f"Missing canonical artifact: {artifact}"

    payload = json.loads(artifact.read_text(encoding="utf-8"))

    assert isinstance(payload.get("schema_version"), str) and payload["schema_version"]
    provenance = payload.get("provenance", {})
    assert isinstance(provenance, dict) and provenance
    assert "generator_script" in provenance
    assert "memory_measurement_mode" in provenance
    assert "timing_measurement_mode" in provenance
    assert "fit_method" in provenance
    assert "memory_limitations" in provenance

    rows = payload.get("rows", [])
    assert isinstance(rows, list) and rows

    decoders = {str(r.get("decoder")) for r in rows}
    assert {"mwpm", "uf", "bm", "adaptive"}.issubset(decoders)

    distances = {int(r.get("distance")) for r in rows}
    assert distances == {3, 5, 7, 9, 11, 13}

    for row in rows:
        metrics = row.get("metrics", {})
        assert "avg_decode_time_sec" in metrics
        assert "wall_time_sec" in metrics
        assert "memory_peak_bytes" in metrics
        assert "memory_current_bytes" in metrics

    scaling_models = payload.get("scaling_models", [])
    assert isinstance(scaling_models, list) and scaling_models
    model_decoders = {str(m.get("decoder")) for m in scaling_models}
    assert {"mwpm", "uf", "bm", "adaptive"}.issubset(model_decoders)
    for model in scaling_models:
        assert model.get("fit_method") == "loglog_power_law_polyfit"
        assert model.get("fit_quality_field") == "r2"
        for fit_key in ("time_fit", "memory_fit"):
            fit = model.get(fit_key, {})
            assert "coefficient" in fit
            assert "exponent" in fit
            assert "r2" in fit
