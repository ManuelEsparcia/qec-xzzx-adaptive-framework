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
SCRIPT = REPO_ROOT / "scripts" / "run_week3_person2_threshold_scan.py"


def _tmp_output_path(stem: str) -> Path:
    out_dir = REPO_ROOT / "results" / "_tmp_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}_{uuid4().hex}.json"


def _run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _point_key(row: dict) -> tuple:
    return (
        str(row["decoder"]),
        str(row["noise_model"]),
        int(row["distance"]),
        int(row["rounds"]),
        float(row["p_phys"]),
        int(row["shots"]),
    )


def _normalize_points(points: list[dict]) -> list[dict]:
    norm: list[dict] = []
    for row in points:
        rr = sorted(row.get("repeat_runs", []), key=lambda x: int(x["repeat_index"]))
        norm.append(
            {
                "key": _point_key(row),
                "error_rate": float(row["error_rate"]),
                "switch_rate": float(row.get("switch_rate", 0.0)),
                "repeat_runs": [
                    {
                        "repeat_index": int(x["repeat_index"]),
                        "seed": int(x["seed"]),
                        "error_rate": float(x["error_rate"]),
                        "switch_rate": float(x.get("switch_rate", 0.0)),
                    }
                    for x in rr
                ],
            }
        )
    return sorted(norm, key=lambda x: x["key"])


def test_script_exists() -> None:
    assert SCRIPT.exists(), f"Expected script does not exist: {SCRIPT}"


def test_resume_repeats_matches_clean_and_is_idempotent() -> None:
    clean_out = _tmp_output_path("week3_scan_clean")
    partial_out = _tmp_output_path("week3_scan_partial")
    resumed_out = _tmp_output_path("week3_scan_resumed")
    resumed_again_out = _tmp_output_path("week3_scan_resumed_again")

    base_args = [
        sys.executable,
        str(SCRIPT),
        "--distances",
        "3,5",
        "--rounds",
        "2",
        "--p-values",
        "0.005,0.01",
        "--decoders",
        "mwpm",
        "--noise-models",
        "depolarizing",
        "--shots",
        "20",
        "--seed",
        "777",
        "--threshold-method",
        "crossing_then_fit",
        "--checkpoint-every",
        "1",
    ]

    clean_cmd = [*base_args, "--repeats", "2", "--output", str(clean_out)]
    partial_cmd = [*base_args, "--repeats", "1", "--output", str(partial_out)]
    resume_cmd = [
        *base_args,
        "--repeats",
        "2",
        "--resume-from",
        str(partial_out),
        "--output",
        str(resumed_out),
    ]
    resume_again_cmd = [
        *base_args,
        "--repeats",
        "2",
        "--resume-from",
        str(resumed_out),
        "--output",
        str(resumed_again_out),
    ]

    clean_proc = _run(clean_cmd)
    assert clean_proc.returncode == 0, (
        f"Clean run failed.\nSTDOUT:\n{clean_proc.stdout}\n\nSTDERR:\n{clean_proc.stderr}"
    )

    partial_proc = _run(partial_cmd)
    assert partial_proc.returncode == 0, (
        f"Partial run failed.\nSTDOUT:\n{partial_proc.stdout}\n\nSTDERR:\n{partial_proc.stderr}"
    )

    resumed_proc = _run(resume_cmd)
    assert resumed_proc.returncode == 0, (
        f"Resume run failed.\nSTDOUT:\n{resumed_proc.stdout}\n\nSTDERR:\n{resumed_proc.stderr}"
    )

    resumed_again_proc = _run(resume_again_cmd)
    assert resumed_again_proc.returncode == 0, (
        "Second resume (idempotence check) failed.\n"
        f"STDOUT:\n{resumed_again_proc.stdout}\n\nSTDERR:\n{resumed_again_proc.stderr}"
    )

    clean_payload = json.loads(clean_out.read_text(encoding="utf-8"))
    resumed_payload = json.loads(resumed_out.read_text(encoding="utf-8"))
    resumed_again_payload = json.loads(resumed_again_out.read_text(encoding="utf-8"))

    clean_points = _normalize_points(clean_payload.get("points", []))
    resumed_points = _normalize_points(resumed_payload.get("points", []))
    resumed_again_points = _normalize_points(resumed_again_payload.get("points", []))

    assert clean_points == resumed_points
    assert resumed_points == resumed_again_points

    expected_points = 2 * 1 * 1 * 2  # distances * noise * decoders * p_values
    assert len(resumed_points) == expected_points
    keys = [p["key"] for p in resumed_points]
    assert len(keys) == len(set(keys)), "Duplicate points found after resume."


def test_resume_from_missing_file_fails() -> None:
    out = _tmp_output_path("week3_scan_missing_resume")
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--distances",
        "3",
        "--p-values",
        "0.01",
        "--decoders",
        "mwpm",
        "--noise-models",
        "depolarizing",
        "--shots",
        "10",
        "--resume-from",
        str(REPO_ROOT / "results" / "_tmp_smoke" / "missing_week3_resume.json"),
        "--output",
        str(out),
    ]
    proc = _run(cmd, timeout=180)
    assert proc.returncode != 0
