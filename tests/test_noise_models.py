# tests/test_noise_models.py
from __future__ import annotations

from typing import List, Set, Tuple

import pytest
import stim

from src.noise.noise_models import (
    BiasedNoise,
    CircuitLevelNoise,
    CorrelatedNoise,
    DepolarizingNoise,
    NoiseModel,
    PhenomenologicalNoise,
    apply_noise_model,
    build_noise_model,
)


def _instruction_names(circuit: stim.Circuit) -> List[str]:
    """Devuelve nombres de instrucciones aplanando bloques REPEAT."""
    names: List[str] = []

    def _walk(c: stim.Circuit) -> None:
        for op in c:
            if hasattr(op, "body_copy"):  # CircuitRepeatBlock
                _walk(op.body_copy())
            else:  # CircuitInstruction
                names.append(op.name)

    _walk(circuit)
    return names


@pytest.fixture
def sample_circuit() -> stim.Circuit:
    """
    Circuit pequeño with:
      - reset
      - gate 1q
      - gate 2q
      - tick
      - medida
    """
    c = stim.Circuit()
    c.append("R", [2])
    c.append("H", [0])
    c.append("CX", [0, 1])
    c.append("TICK")
    c.append("M", [0, 1, 2])
    return c


def test_build_noise_model_from_string_and_aliases() -> None:
    assert isinstance(build_noise_model("depolarizing", p=0.01), DepolarizingNoise)
    assert isinstance(build_noise_model("depol", p=0.01), DepolarizingNoise)
    assert isinstance(build_noise_model("biased", p=0.01, eta=50), BiasedNoise)
    assert isinstance(build_noise_model("circuit_level", p=0.01), CircuitLevelNoise)
    assert isinstance(build_noise_model("phenomenological", p=0.01), PhenomenologicalNoise)
    assert isinstance(
        build_noise_model(
            "correlated",
            p=0.01,
            correlation_length=2,
            correlation_strength=0.4,
        ),
        CorrelatedNoise,
    )


def test_build_noise_model_from_dict_spec() -> None:
    model = build_noise_model(
        {
            "type": "biased",
            "p": 0.02,
            "eta": 100.0,
        }
    )
    assert isinstance(model, BiasedNoise)
    assert model.p == pytest.approx(0.02)
    assert model.eta == pytest.approx(100.0)


def test_apply_noise_model_helper_returns_circuit(sample_circuit: stim.Circuit) -> None:
    noisy = apply_noise_model(sample_circuit, model="depolarizing", p=0.01)
    assert isinstance(noisy, stim.Circuit)
    names = set(_instruction_names(noisy))
    assert "DEPOLARIZE1" in names or "DEPOLARIZE2" in names


@pytest.mark.parametrize(
    "model,expected_gates",
    [
        (DepolarizingNoise(p=0.01), {"DEPOLARIZE1", "DEPOLARIZE2"}),
        (BiasedNoise(p=0.01, eta=100.0), {"PAULI_CHANNEL_1"}),
        (CircuitLevelNoise(p=0.01), {"X_ERROR", "DEPOLARIZE2"}),
        (PhenomenologicalNoise(p=0.01), {"X_ERROR", "Z_ERROR"}),
        (
            CorrelatedNoise(
                p=0.01,
                correlation_length=2,
                correlation_strength=0.5,
                topology="line",
                pauli="Z",
            ),
            {"Z_ERROR", "CORRELATED_ERROR"},
        ),
    ],
)
def test_each_model_injects_expected_noise(
    sample_circuit: stim.Circuit,
    model: NoiseModel,
    expected_gates: Set[str],
) -> None:
    noisy = model.apply_to_circuit(sample_circuit)
    names = set(_instruction_names(noisy))

    # At least one expected gate must appear; in practice they usually all appear.
    assert any(g in names for g in expected_gates), (
        f"Expected noise gates were not detected: {expected_gates}. "
        f"Encontradas: {sorted(names)}"
    )


def test_meta_instructions_preserved_and_no_crash() -> None:
    c = stim.Circuit(
        """
        R 0
        M 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )
    noisy = DepolarizingNoise(p=0.01).apply_to_circuit(c)
    names = _instruction_names(noisy)
    assert "DETECTOR" in names
    assert "OBSERVABLE_INCLUDE" in names


def test_repeat_block_supported() -> None:
    c = stim.Circuit(
        """
        REPEAT 3 {
            H 0
            M 0
        }
        """
    )
    noisy = DepolarizingNoise(p=0.01).apply_to_circuit(c)
    names = _instruction_names(noisy)
    assert "DEPOLARIZE1" in names  # noise insertado dentro del bloque


def test_correlated_neighbors_line() -> None:
    model = CorrelatedNoise(
        p=0.01,
        correlation_length=2,
        correlation_strength=0.3,
        topology="line",
    )
    neighbors = model.get_neighbors(qubit_idx=3, max_distance=2, n_qubits=6)
    assert set(neighbors) == {(2, 1), (4, 1), (1, 2), (5, 2)}


def test_correlated_neighbors_grid_center_distance_1() -> None:
    model = CorrelatedNoise(
        p=0.01,
        correlation_length=1,
        correlation_strength=0.3,
        topology="grid",
    )
    # n_qubits=9 => grid 3x3, centro idx=4
    neighbors = model.get_neighbors(qubit_idx=4, max_distance=1, n_qubits=9)
    idxs = {q for q, d in neighbors if d == 1}
    assert idxs == {1, 3, 5, 7}


def test_invalid_probabilities_and_params_raise() -> None:
    with pytest.raises(ValueError):
        DepolarizingNoise(p=-0.1)

    with pytest.raises(ValueError):
        DepolarizingNoise(p=1.1)

    with pytest.raises(ValueError):
        BiasedNoise(p=0.01, eta=0.0)

    with pytest.raises(ValueError):
        CorrelatedNoise(p=0.01, correlation_length=0, correlation_strength=0.3)

    with pytest.raises(ValueError):
        CorrelatedNoise(p=0.01, correlation_length=2, correlation_strength=1.2)

    with pytest.raises(ValueError):
        CorrelatedNoise(
            p=0.01,
            correlation_length=2,
            correlation_strength=0.3,
            topology="ring",
        )

    with pytest.raises(ValueError):
        CorrelatedNoise(
            p=0.01,
            correlation_length=2,
            correlation_strength=0.3,
            pauli="A",
        )


def test_build_noise_model_unknown_raises() -> None:
    with pytest.raises(ValueError):
        build_noise_model("unknown_model", p=0.01)


def test_build_noise_model_dict_without_type_raises() -> None:
    with pytest.raises(ValueError):
        build_noise_model({"p": 0.01})  # falta "type"


def test_apply_noise_model_type_error() -> None:
    with pytest.raises(TypeError):
        apply_noise_model("not_a_circuit", model="depolarizing", p=0.01)  # type: ignore[arg-type]
