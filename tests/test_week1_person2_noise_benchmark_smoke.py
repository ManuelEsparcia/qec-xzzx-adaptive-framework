# tests/test_week1_person2_noise_benchmark_smoke.py
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _script_path() -> Path:
    return _repo_root() / "scripts" / "run_week1_person2_noise_benchmark.py"


def _run_cmd(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    # Reinforce imports like "from src...."
    env["PYTHONPATH"] = str(_repo_root()) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        args,
        cwd=_repo_root(),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_script_exists() -> None:
    script = _script_path()
    assert script.exists(), f"Script does not exist: {script}"


def test_noise_benchmark_script_smoke(tmp_path: Path) -> None:
    """
    Smoke test:
    - Run benchmark with few shots
    - Verify successful completion
    - Verify minimum output JSON contract
    """
    output = tmp_path / "week1_person2_noise_benchmark_smoke.json"

    cmd = [
        sys.executable,
        "-m",
        "scripts.run_week1_person2_noise_benchmark",
        "--shots",
        "40",
        "--logical-basis",
        "x",
        "--seed",
        "12345",
        "--output",
        str(output),
    ]
    result = _run_cmd(cmd, timeout=240)

    assert result.returncode == 0, (
        "Script failed.\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )

    assert output.exists(), "Output JSON was not generated."

    with output.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Top-level contract
    assert isinstance(data, dict)
    for k in ("metadata", "cases_summary", "aggregates", "status"):
        assert k in data, f"Missing top-level key: {k}"

    assert data["status"] == "ok"

    meta = data["metadata"]
    assert meta.get("report_name") == "week1_person2_noise_benchmark"
    assert isinstance(meta.get("shots_per_model"), int)
    assert meta.get("num_cases") == 3

    cases = data["cases_summary"]
    assert isinstance(cases, list)
    assert len(cases) == 3

    expected_models = {
        "depolarizing",
        "biased",
        "circuit_level",
        "phenomenological",
        "correlated",
    }

    for c in cases:
        assert c.get("status") == "ok"
        for key in (
            "case_name",
            "distance",
            "rounds",
            "p",
            "shots",
            "logical_basis",
            "num_qubits",
            "models",
            "delta_ler_correlated_minus_depolarizing",
        ):
            assert key in c, f"Missing key in case {c.get('case_name')}: {key}"

        models = c["models"]
        assert isinstance(models, list)
        assert len(models) == 5

        model_names = {m.get("model_name") for m in models}
        assert expected_models.issubset(model_names)

        for m in models:
            for mk in (
                "model_name",
                "model_spec",
                "seed",
                "ler",
                "avg_decode_time_sec",
                "decode_total_time_sec",
                "num_detectors",
                "num_observables",
            ):
                assert mk in m, f"Missing key '{mk}' in model {m.get('model_name')}"

            assert 0.0 <= float(m["ler"]) <= 1.0
            assert float(m["avg_decode_time_sec"]) >= 0.0
            assert int(m["num_detectors"]) > 0
            assert int(m["num_observables"]) >= 1

    aggr = data["aggregates"]
    assert "mean_ler_by_model" in aggr
    assert "mean_avg_decode_time_sec_by_model" in aggr
    assert "mean_delta_ler_correlated_minus_depolarizing" in aggr

    mean_ler_by_model = aggr["mean_ler_by_model"]
    assert expected_models.issubset(set(mean_ler_by_model.keys()))


def test_invalid_shots_fail(tmp_path: Path) -> None:
    """
    Must fail with invalid shots (<=0).
    """
    output = tmp_path / "should_not_exist.json"
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_week1_person2_noise_benchmark",
        "--shots",
        "0",
        "--output",
        str(output),
    ]
    result = _run_cmd(cmd)

    assert result.returncode != 0, "Expected error with shots=0, but command did not fail."

    merged = (result.stdout + "\n" + result.stderr).lower()
    assert ("shots" in merged) or ("valueerror" in merged), (
        "Expected error message not found.\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )
