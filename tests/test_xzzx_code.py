# tests/test_xzzx_code.py
from __future__ import annotations

import pytest

# If dependencies are missing, skip the whole file de tests.
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
    Test 1: el circuit is genera correctamente.
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
    Test 2: consistency del número de detectores.
    Not asumimos una fórmula cerrada (depende del task), pero sí internal consistency.
    """
    c = generate_xzzx_circuit(distance=5, rounds=3, noise_model="none", p=0.0, logical_basis="x")
    dem = c.detector_error_model(decompose_errors=True)

    # Consistency entre circuit and DEM
    assert c.num_detectors == dem.num_detectors
    assert c.num_detectors > 0

    # Monotonicidad simple al aumentar rounds
    c_more_rounds = generate_xzzx_circuit(distance=5, rounds=4, noise_model="none", p=0.0, logical_basis="x")
    assert c_more_rounds.num_detectors >= c.num_detectors


def test_perfect_decoding_no_noise() -> None:
    """
    Test 3: without noise, el decoding must be perfecto (LER ~ 0).
    """
    c = generate_xzzx_circuit(distance=3, rounds=3, noise_model="none", p=0.0, logical_basis="x")
    ler = logical_error_rate_mwpm(c, shots=500)

    assert isinstance(ler, float)
    assert ler == pytest.approx(0.0, abs=1e-12)


def test_invalid_inputs() -> None:
    """
    Validation básicas de input.
    """
    with pytest.raises(ValueError):
        generate_xzzx_circuit(distance=4, rounds=2, noise_model="none", p=0.0)  # distance par

    with pytest.raises(ValueError):
        generate_xzzx_circuit(distance=3, rounds=0, noise_model="none", p=0.0)  # invalid rounds

    with pytest.raises(ValueError):
        generate_xzzx_circuit(distance=3, rounds=2, noise_model="none", p=-0.1)  # p out of range

    with pytest.raises(ValueError):
        generate_xzzx_circuit(distance=3, rounds=2, noise_model="none", p=0.1, logical_basis="y")


def test_custom_noise_object_accepts_apply_to_circuit_with_one_arg() -> None:
    class OneArgNoise:
        def __init__(self) -> None:
            self.called = False

        def apply_to_circuit(self, circuit: "stim.Circuit") -> "stim.Circuit":
            self.called = True
            return circuit

    model = OneArgNoise()
    circuit = generate_xzzx_circuit(
        distance=3,
        rounds=2,
        noise_model=model,
        p=0.02,
        logical_basis="x",
    )

    assert isinstance(circuit, stim.Circuit)
    assert model.called is True


def test_custom_noise_object_accepts_apply_to_circuit_with_two_args() -> None:
    class TwoArgNoise:
        def __init__(self) -> None:
            self.called = False
            self.last_p = None

        def apply_to_circuit(self, circuit: "stim.Circuit", p: float) -> "stim.Circuit":
            self.called = True
            self.last_p = float(p)
            return circuit

    model = TwoArgNoise()
    circuit = generate_xzzx_circuit(
        distance=3,
        rounds=2,
        noise_model=model,
        p=0.03,
        logical_basis="x",
    )

    assert isinstance(circuit, stim.Circuit)
    assert model.called is True
    assert model.last_p == pytest.approx(0.03)
