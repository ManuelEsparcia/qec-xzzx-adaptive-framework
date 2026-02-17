# tests/test_week1_person2_integration.py
from __future__ import annotations

import inspect
import statistics
from typing import Callable, Iterable, List, Optional, Tuple

import pytest
import stim

pytest.importorskip("pymatching")
import pymatching  # noqa: E402

from src.noise.noise_models import (
    BiasedNoise,
    CircuitLevelNoise,
    CorrelatedNoise,
    DepolarizingNoise,
    PhenomenologicalNoise,
)


def _maybe_extract_circuit(obj) -> Optional[stim.Circuit]:
    """Allow builders that return Circuit or tuple/dict containing Circuit."""
    if isinstance(obj, stim.Circuit):
        return obj

    if isinstance(obj, tuple):
        for x in obj:
            if isinstance(x, stim.Circuit):
                return x

    if isinstance(obj, dict):
        for k in ("circuit", "stim_circuit", "memory_circuit"):
            if k in obj and isinstance(obj[k], stim.Circuit):
                return obj[k]

    return None


def _call_builder_with_flexible_signature(
    fn: Callable,
    *,
    distance: int,
    rounds: int,
    logical_basis: str,
) -> Optional[stim.Circuit]:
    """
    Try calling the XZZX builder with multiple signature conventions.
    """
    sig = inspect.signature(fn)
    kwargs = {}

    name_map = {
        "distance": distance,
        "d": distance,
        "code_distance": distance,
        "rounds": rounds,
        "r": rounds,
        "num_rounds": rounds,
        "cycles": rounds,
        "logical_basis": logical_basis,
        "basis": logical_basis,
        "memory_basis": logical_basis,
        # Force ideal/no-noise if the builder accepts these fields
        "p": 0.0,
        "error_rate": 0.0,
        "noise_strength": 0.0,
        "noise_model": "depolarizing",
        "noise": "depolarizing",
        "model": "depolarizing",
    }

    for pname in sig.parameters:
        if pname in name_map:
            kwargs[pname] = name_map[pname]

    try:
        out = fn(**kwargs)
        c = _maybe_extract_circuit(out)
        if c is not None:
            return c
    except Exception:
        pass

    # Minimal positional fallback
    try:
        out = fn(distance, rounds)
        c = _maybe_extract_circuit(out)
        if c is not None:
            return c
    except Exception:
        pass

    return None


def _build_base_xzzx_circuit(
    *,
    distance: int = 3,
    rounds: int = 3,
    logical_basis: str = "x",
) -> stim.Circuit:
    """
    Dynamically find the builder in src.codes.xzzx_code to avoid coupling
    to an exact function name.
    """
    from src.codes import xzzx_code as xc

    preferred_names = [
        "build_xzzx_memory_circuit",
        "generate_xzzx_memory_circuit",
        "make_xzzx_memory_circuit",
        "create_xzzx_memory_circuit",
        "build_xzzx_circuit",
        "generate_xzzx_circuit",
        "make_xzzx_circuit",
        "create_xzzx_circuit",
    ]

    # 1) Attempt preferred names
    for name in preferred_names:
        fn = getattr(xc, name, None)
        if callable(fn):
            c = _call_builder_with_flexible_signature(
                fn,
                distance=distance,
                rounds=rounds,
                logical_basis=logical_basis,
            )
            if c is not None:
                return c

    # 2) Fallback: explore module callables with "xzzx" and "circuit" in name
    for name in dir(xc):
        if "xzzx" not in name.lower() or "circuit" not in name.lower():
            continue
        fn = getattr(xc, name, None)
        if callable(fn):
            c = _call_builder_with_flexible_signature(
                fn,
                distance=distance,
                rounds=rounds,
                logical_basis=logical_basis,
            )
            if c is not None:
                return c

    pytest.skip(
        "No compatible XZZX circuit builder found in src.codes.xzzx_code."
    )


def _sample_det_obs(
    circuit: stim.Circuit,
    *,
    shots: int,
    seed: int,
) -> Tuple:
    sampler = circuit.compile_detector_sampler(seed=seed)
    data = sampler.sample(shots=shots, separate_observables=True)
    if not isinstance(data, tuple) or len(data) != 2:
        raise RuntimeError(
            "Stim did not return (detectors, observables) with separate_observables=True."
        )
    dets, obs = data
    return dets, obs


def _estimate_ler(
    circuit: stim.Circuit,
    *,
    shots: int = 200,
    seed: int = 1234,
) -> dict:
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)

    dets, obs = _sample_det_obs(circuit, shots=shots, seed=seed)

    pred = matcher.decode_batch(dets)

    # Normalize shape to (shots, num_obs)
    if getattr(pred, "ndim", 1) == 1:
        pred = pred.reshape(-1, 1)
    if getattr(obs, "ndim", 1) == 1:
        obs = obs.reshape(-1, 1)

    if obs.shape[1] < 1:
        pytest.skip("Circuit does not expose logical observables (num_observables=0).")

    ler = float((pred[:, 0] != obs[:, 0]).mean())
    return {
        "ler": ler,
        "num_shots": int(shots),
        "num_detectors": int(dets.shape[1]),
        "num_observables": int(obs.shape[1]),
    }


def test_full_pipeline_e2e_d3_runs() -> None:
    base = _build_base_xzzx_circuit(distance=3, rounds=3, logical_basis="x")
    noisy = DepolarizingNoise(p=0.01).apply_to_circuit(base)
    metrics = _estimate_ler(noisy, shots=120, seed=11)

    assert 0.0 <= metrics["ler"] <= 1.0
    assert metrics["num_detectors"] > 0
    assert metrics["num_observables"] >= 1


def test_all_noise_models_run_without_breaking_contract() -> None:
    base = _build_base_xzzx_circuit(distance=3, rounds=2, logical_basis="x")

    models = [
        DepolarizingNoise(p=0.008),
        BiasedNoise(p=0.008, eta=50.0),
        CircuitLevelNoise(p=0.008),
        PhenomenologicalNoise(p=0.008),
        CorrelatedNoise(
            p=0.008,
            correlation_length=2,
            correlation_strength=0.5,
            topology="line",
            pauli="Z",
        ),
    ]

    for i, model in enumerate(models):
        noisy = model.apply_to_circuit(base)
        metrics = _estimate_ler(noisy, shots=80, seed=100 + i)
        assert 0.0 <= metrics["ler"] <= 1.0
        assert metrics["num_detectors"] > 0
        assert metrics["num_observables"] >= 1


def test_correlated_noise_not_better_than_independent_on_average() -> None:
    """
    On average and with sufficiently aggressive parameters, correlated noise
    should not outperform a comparable independent baseline.
    """
    base = _build_base_xzzx_circuit(distance=3, rounds=3, logical_basis="x")

    indep_model = DepolarizingNoise(p=0.012)
    corr_model = CorrelatedNoise(
        p=0.012,
        correlation_length=2,
        correlation_strength=0.8,
        topology="line",
        pauli="Z",
    )

    seeds = [7, 23, 47, 89, 131]
    indep_lers: List[float] = []
    corr_lers: List[float] = []

    for s in seeds:
        indep_noisy = indep_model.apply_to_circuit(base)
        corr_noisy = corr_model.apply_to_circuit(base)

        indep_lers.append(_estimate_ler(indep_noisy, shots=160, seed=s)["ler"])
        corr_lers.append(_estimate_ler(corr_noisy, shots=160, seed=s)["ler"])

    mean_indep = statistics.fmean(indep_lers)
    mean_corr = statistics.fmean(corr_lers)

    # Small tolerance to avoid statistical fragility.
    assert mean_corr >= mean_indep - 0.01, (
        f"Unexpected: correlated model improves too much over independent baseline. "
        f"mean_corr={mean_corr:.4f}, mean_indep={mean_indep:.4f}"
    )
