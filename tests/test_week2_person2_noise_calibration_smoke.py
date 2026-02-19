# tests/test_week2_person2_noise_calibration_smoke.py
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_cmd(cmd: list[str], timeout: int = 240) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_script_exists() -> None:
    script = Path("scripts/run_week2_person2_noise_calibration.py")
    assert script.exists(), f"Script does not exist: {script}"


def test_noise_calibration_script_smoke(tmp_path: Path) -> None:
    """
    Smoke test:
    - Run script with a small configuration.
    - Verify successful completion.
    - Verify minimum output JSON contract.
    """
    output = tmp_path / "week2_person2_noise_calibration_smoke.json"

    cmd = [
        sys.executable,
        "-m",
        "scripts.run_week2_person2_noise_calibration",
        "--shots",
        "40",
        "--seed",
        "12345",
        "--logical-basis",
        "x",
        "--models",
        "depolarizing,biased",
        "--fast",
        "--output",
        str(output),
    ]
    result = _run_cmd(cmd, timeout=300)

    assert result.returncode == 0, (
        "Script failed.\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )

    assert output.exists(), "Output JSON was not created."

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), "JSON must be an object."

    # Robust minimum contract
    metadata = payload.get("metadata", {})
    assert isinstance(metadata, dict), "metadata must be dict."
    assert metadata.get("script") == "run_week2_person2_noise_calibration.py"
    assert int(metadata.get("shots", -1)) == 40

    assert "selected_sweep_templates" in payload
    assert isinstance(payload["selected_sweep_templates"], list)
    assert len(payload["selected_sweep_templates"]) >= 1

    # Accept any of these structures depending on internal implementation
    has_summary = isinstance(payload.get("sweeps_summary"), list)
    has_reports = isinstance(payload.get("sweeps_reports"), list)
    has_raw = "raw_report" in payload
    assert has_summary or has_reports or has_raw, (
        "Falta estructura esperada: sweeps_summary / sweeps_reports / raw_report"
    )


def test_invalid_shots_fail(tmp_path: Path) -> None:
    """
    If shots <= 0, script must fail with return code != 0.
    """
    output = tmp_path / "should_not_exist.json"

    cmd = [
        sys.executable,
        "-m",
        "scripts.run_week2_person2_noise_calibration",
        "--shots",
        "0",
        "--output",
        str(output),
    ]
    result = _run_cmd(cmd, timeout=120)

    assert result.returncode != 0, (
        "Expected failure with shots=0, but returned code 0.\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )


def test_invalid_models_fail(tmp_path: Path) -> None:
    """
    Unknown models must fail during parsing/validation.
    """
    output = tmp_path / "should_not_exist_2.json"

    cmd = [
        sys.executable,
        "-m",
        "scripts.run_week2_person2_noise_calibration",
        "--shots",
        "20",
        "--models",
        "depolarizing,model_inexistente",
        "--output",
        str(output),
    ]
    result = _run_cmd(cmd, timeout=120)

    assert result.returncode != 0, (
        "Expected failure for invalid model, but returned code 0.\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )


def test_objective_min_time_is_applied_to_sweeps(tmp_path: Path) -> None:
    """
    Regression check:
    --objective must be propagated to each sweep definition.
    """
    output = tmp_path / "week2_person2_noise_calibration_min_time.json"

    cmd = [
        sys.executable,
        "-m",
        "scripts.run_week2_person2_noise_calibration",
        "--shots",
        "20",
        "--seed",
        "12345",
        "--logical-basis",
        "x",
        "--models",
        "depolarizing",
        "--objective",
        "min_time",
        "--fast",
        "--output",
        str(output),
    ]
    result = _run_cmd(cmd, timeout=300)

    assert result.returncode == 0, (
        "Script failed.\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )
    assert output.exists(), "Output JSON was not created."

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload.get("metadata", {}).get("objective") == "min_time"

    sweeps_reports = payload.get("sweeps_reports", [])
    assert isinstance(sweeps_reports, list) and sweeps_reports, (
        "Expected non-empty sweeps_reports."
    )
    for rep in sweeps_reports:
        assert rep.get("sweep", {}).get("objective") == "min_time"
