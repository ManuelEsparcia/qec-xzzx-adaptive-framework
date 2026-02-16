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
    assert script.exists(), f"No existe el script: {script}"


def test_noise_calibration_script_smoke(tmp_path: Path) -> None:
    """
    Smoke test:
    - Ejecuta el script con una configuración pequeña.
    - Verifica que termina correctamente.
    - Verifica contrato mínimo del JSON de salida.
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
        "El script falló.\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )

    assert output.exists(), "No se creó el JSON de salida."

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), "El JSON debe ser un objeto."

    # Contrato mínimo robusto
    metadata = payload.get("metadata", {})
    assert isinstance(metadata, dict), "metadata debe ser dict."
    assert metadata.get("script") == "run_week2_person2_noise_calibration.py"
    assert int(metadata.get("shots", -1)) == 40

    assert "selected_sweep_templates" in payload
    assert isinstance(payload["selected_sweep_templates"], list)
    assert len(payload["selected_sweep_templates"]) >= 1

    # Aceptamos cualquiera de estas estructuras según implementación interna
    has_summary = isinstance(payload.get("sweeps_summary"), list)
    has_reports = isinstance(payload.get("sweeps_reports"), list)
    has_raw = "raw_report" in payload
    assert has_summary or has_reports or has_raw, (
        "Falta estructura esperada: sweeps_summary / sweeps_reports / raw_report"
    )


def test_invalid_shots_fail(tmp_path: Path) -> None:
    """
    Si shots <= 0, el script debe fallar con código != 0.
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
        "Se esperaba fallo con shots=0, pero terminó con código 0.\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )


def test_invalid_models_fail(tmp_path: Path) -> None:
    """
    Modelos desconocidos deben fallar en parseo/validación.
    """
    output = tmp_path / "should_not_exist_2.json"

    cmd = [
        sys.executable,
        "-m",
        "scripts.run_week2_person2_noise_calibration",
        "--shots",
        "20",
        "--models",
        "depolarizing,modelo_inexistente",
        "--output",
        str(output),
    ]
    result = _run_cmd(cmd, timeout=120)

    assert result.returncode != 0, (
        "Se esperaba fallo por modelo inválido, pero terminó con código 0.\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )
