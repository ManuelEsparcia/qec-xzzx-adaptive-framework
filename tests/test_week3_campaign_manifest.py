from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "generate_week3_campaign_manifest.py"


def _tmp_manifest_path(stem: str) -> Path:
    out_dir = REPO_ROOT / "results" / "_tmp_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}_{uuid4().hex}.json"


def test_script_exists() -> None:
    assert SCRIPT.exists(), f"Expected script does not exist: {SCRIPT}"


def test_manifest_has_no_duplicates_and_expected_coverage() -> None:
    manifest_path = _tmp_manifest_path("week3_manifest")
    chunks_dir = (REPO_ROOT / "results" / "_tmp_smoke" / f"week3_chunks_{uuid4().hex}").as_posix()
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--distances",
        "3,5",
        "--p-values",
        "0.005,0.01",
        "--decoders",
        "mwpm,uf",
        "--noise-models",
        "depolarizing,biased_eta10",
        "--shots",
        "20",
        "--repeats",
        "1",
        "--output-dir",
        chunks_dir,
        "--output",
        str(manifest_path),
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert proc.returncode == 0, (
        f"Manifest generation failed.\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
    )
    assert manifest_path.exists(), "Manifest was not created."

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    md = payload.get("metadata", {})
    jobs = payload.get("jobs", [])

    expected_jobs = 2 * 2 * 2  # distances * noise * decoders
    expected_points = expected_jobs * 2  # p-values

    assert md.get("report_name") == "week3_campaign_manifest"
    assert md.get("schema_version") == "week3_manifest_v1"
    assert int(md.get("num_jobs", -1)) == expected_jobs
    assert int(md.get("expected_total_points", -1)) == expected_points
    assert len(jobs) == expected_jobs

    job_ids = [str(j["job_id"]) for j in jobs]
    job_keys = [str(j["job_key"]) for j in jobs]
    outputs = [str(j["output"]) for j in jobs]
    assert len(job_ids) == len(set(job_ids)), "Duplicate job_id in manifest."
    assert len(job_keys) == len(set(job_keys)), "Duplicate job_key in manifest."
    assert len(outputs) == len(set(outputs)), "Duplicate output paths in manifest."

    for j in jobs:
        assert int(j["distance"]) in {3, 5}
        assert str(j["decoder"]) in {"mwpm", "uf"}
        assert str(j["noise_model"]) in {"depolarizing", "biased_eta10"}
        cmd = j.get("command", [])
        assert isinstance(cmd, list) and cmd
        assert "--distances" in cmd
        assert "--output" in cmd
