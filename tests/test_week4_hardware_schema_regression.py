from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_JSON = REPO_ROOT / "results" / "week4_hardware_compatibility.json"
CANONICAL_FIG = REPO_ROOT / "figures" / "week4_hardware_compatibility_heatmaps.png"


def test_week4_canonical_artifacts_exist() -> None:
    assert CANONICAL_JSON.exists(), f"Missing Week-4 JSON artifact: {CANONICAL_JSON}"
    assert CANONICAL_FIG.exists(), f"Missing Week-4 heatmap artifact: {CANONICAL_FIG}"
    assert CANONICAL_FIG.stat().st_size > 0


def test_week4_schema_and_provenance_contract() -> None:
    payload = json.loads(CANONICAL_JSON.read_text(encoding="utf-8"))

    md = payload.get("metadata", {})
    cfg = payload.get("config", {})
    src = payload.get("latency_source", {})
    comp = payload.get("compatibility", {})

    assert md.get("report_name") == "week4_hardware_compatibility"
    assert md.get("schema_version") == "week4_hw_v2"
    assert isinstance(md.get("coverage"), dict)
    assert "latency_source_mode" in md
    assert "benchmark_distances_used" in md
    assert "limitations" in md

    assert cfg.get("distances") == [3, 5, 7, 9, 11, 13]

    decoder_labels = [str(x) for x in comp.get("decoder_labels", [])]
    distances = [int(x) for x in comp.get("distances", [])]
    archs = comp.get("architectures", [])

    assert distances == [3, 5, 7, 9, 11, 13]
    assert "mwpm" in decoder_labels
    assert "uf" in decoder_labels
    adaptive_rows = [x for x in decoder_labels if x.startswith("adaptive_g")]
    assert adaptive_rows, "Expected adaptive rows in compatibility report."

    arch_names = {str(a.get("architecture")) for a in archs}
    assert arch_names == {
        "GoogleSuperconducting",
        "IBMEagle",
        "IonQForte",
        "PsiQuantumPhotonic",
    }

    row_count = len(decoder_labels)
    col_count = len(distances)
    for a in archs:
        matrix = a.get("matrix", [])
        assert len(matrix) == row_count
        assert all(len(r) == col_count for r in matrix)
        assert int(a.get("total_cells", -1)) == row_count * col_count
        compat_cells = int(a.get("compatible_cells", -1))
        assert 0 <= compat_cells <= row_count * col_count

    coverage = md["coverage"]
    assert int(coverage.get("architecture_count", -1)) == len(archs)
    assert int(coverage.get("decoder_row_count", -1)) == row_count
    assert coverage.get("distances_covered") == distances
    assert isinstance(coverage.get("adaptive_thresholds_covered"), list)

    assert isinstance(src, dict)
    assert src.get("mode") == md.get("latency_source_mode")
    assert src.get("benchmark_distances_used") == md.get("benchmark_distances_used")
    assert "fit_provenance_summary" in src
    assert "hardware_trace_calibrated" in src
