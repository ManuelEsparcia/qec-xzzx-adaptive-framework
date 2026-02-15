# tests/test_xzzx_code.py
from __future__ import annotations

import pytest

# Si faltan dependencias, se salta el archivo completo de tests.
stim = pytest.importorskip("stim")
pytest.importorskip("pymatching")

from src.codes.xzzx_code import (
    circuit_summary,
    generate_xzzx_circuit,
    logical_error_rate_mwpm,
)


@pytest.mark.parametrize(
    "distance,rounds,logical_basis",
    [
        (3, 2, "x"),
        (3, 2, "z"),
        (5, 2, "x"),
    ],
)
def test_circuit_generation(distance: int, rounds: int, logical_basis: str) -> None:
    """
    Test 1: el circuito se genera correctamente.
    """
    circuit = generate_xzzx_circuit(
        distance=distance,
        rounds=rounds,
        noise_model="none",
        p=0.0,
        logical_basis=logical_basis,
    )

    assert isinstance(circuit, stim.Circuit)

    summary = circuit_summary(circuit)
    assert summary["num_qubits"] > 0
    assert summary["num_measurements"] > 0
    assert summary["num_detectors"] > 0
    assert summary["num_observables"] >= 1


def test_detector_count_consistency() -> None:
    """
    Test 2: consistencia del número de detectores.
    No asumimos una fórmula cerrada (depende del task), pero sí consistencia interna.
    """
    c = generate_xzzx_circuit(distance=5, rounds=3, noise_model="none", p=0.0, logical_basis="x")
    dem = c.detector_error_model(decompose_errors=True)

    # Consistencia entre circuito y DEM
    assert c.num_detectors == dem.num_detectors
    assert c.num_detectors > 0

    # Monotonicidad simple al aumentar rounds
    c_more_rounds = generate_xzzx_circuit(distance=5, rounds=4, noise_model="none", p=0.0, logical_basis="x")
    assert c_more_rounds.num_detectors >= c.num_detectors


def test_perfect_decoding_no_noise() -> None:
    """
    Test 3: sin ruido, el decoding debe ser perfecto (LER ~ 0).
    """
    c = generate_xzzx_circuit(distance=3, rounds=3, noise_model="none", p=0.0, logical_basis="x")
    ler = logical_error_rate_mwpm(c, shots=500)

    assert isinstance(ler, float)
    assert ler == pytest.approx(0.0, abs=1e-12)


def test_invalid_inputs() -> None:
    """
    Validaciones básicas de entrada.
    """
    with pytest.raises(ValueError):
        generate_xzzx_circuit(distance=4, rounds=2, noise_model="none", p=0.0)  # distance par

    with pytest.raises(ValueError):
        generate_xzzx_circuit(distance=3, rounds=0, noise_model="none", p=0.0)  # rounds inválido

    with pytest.raises(ValueError):
        generate_xzzx_circuit(distance=3, rounds=2, noise_model="none", p=-0.1)  # p fuera de rango

    with pytest.raises(ValueError):
        generate_xzzx_circuit(distance=3, rounds=2, noise_model="none", p=0.1, logical_basis="y")
