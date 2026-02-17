# src/codes/xzzx_code.py
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Union

import numpy as np

try:
    import stim
except Exception as exc:  # pragma: no cover
    stim = None  # type: ignore[assignment]
    _STIM_IMPORT_ERROR = exc
else:
    _STIM_IMPORT_ERROR = None

try:
    import pymatching
except Exception as exc:  # pragma: no cover
    pymatching = None  # type: ignore[assignment]
    _PYMATCHING_IMPORT_ERROR = exc
else:
    _PYMATCHING_IMPORT_ERROR = None


# -----------------------------
# Types
# -----------------------------
NoiseDict = Dict[str, float]
NoiseCallable = Callable[["stim.Circuit", float], Optional["stim.Circuit"]]


class NoiseApplierProtocol(Protocol):
    def apply_to_circuit(self, circuit: "stim.Circuit", p: float = ...) -> Optional["stim.Circuit"]:
        ...


NoiseModel = Union[str, None, NoiseDict, NoiseCallable, NoiseApplierProtocol]


@dataclass(frozen=True)
class _InputConfig:
    distance: int
    rounds: int
    p: float
    logical_basis: str


# -----------------------------
# Internal helpers
# -----------------------------
def _require_stim() -> None:
    if stim is None:
        raise ImportError(
            "stim is not installed or failed to import. "
            "Install with: pip install stim"
        ) from _STIM_IMPORT_ERROR


def _require_pymatching() -> None:
    if pymatching is None:
        raise ImportError(
            "pymatching is not installed or failed to import. "
            "Install with: pip install pymatching"
        ) from _PYMATCHING_IMPORT_ERROR


def _validate_inputs(
    distance: int,
    rounds: int,
    p: float,
    logical_basis: str,
) -> _InputConfig:
    if not isinstance(distance, int):
        raise TypeError(f"distance must be int, received: {type(distance).__name__}")
    if distance < 3 or distance % 2 == 0:
        raise ValueError(f"distance must be odd and >= 3. Received: {distance}")

    if not isinstance(rounds, int):
        raise TypeError(f"rounds must be int, received: {type(rounds).__name__}")
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1. Received: {rounds}")

    if not isinstance(p, (int, float)):
        raise TypeError(f"p must be float/int, received: {type(p).__name__}")
    p_float = float(p)
    if not (0.0 <= p_float <= 1.0):
        raise ValueError(f"p must be in [0, 1]. Received: {p_float}")

    if not isinstance(logical_basis, str):
        raise TypeError(f"logical_basis must be str, received: {type(logical_basis).__name__}")
    lb = logical_basis.lower().strip()
    if lb not in {"x", "z"}:
        raise ValueError(f"logical_basis must be 'x' or 'z'. Received: {logical_basis}")

    return _InputConfig(
        distance=distance,
        rounds=rounds,
        p=p_float,
        logical_basis=lb,
    )


def _task_name(logical_basis: str) -> str:
    """
    Use Stim's built-in rotated memory code generator.
    For Week 1, this is the most robust startup path.
    """
    if logical_basis == "x":
        return "surface_code:rotated_memory_x"
    return "surface_code:rotated_memory_z"


def _stim_noise_kwargs(noise_model: NoiseModel, p: float) -> Dict[str, float]:
    """
    Traduce noise_model -> kwargs compatibles with stim.Circuit.generated(...).

    Soportado:
      - None / "none" / "ideal": without noise
      - "depolarizing": simple uniform noise
      - dict: is pasa tal cual (validando tipo numérico)
    """
    if noise_model is None:
        return {}

    if isinstance(noise_model, str):
        nm = noise_model.lower().strip()
        if nm in {"none", "ideal", "no_noise"}:
            return {}
        if nm in {"depolarizing", "depolarising", "uniform"}:
            # Ajuste simple for Week 1:
            # - after_clifford_depolarization controla la tasa principal.
            # Puedes endurecer/afinar después.
            return {"after_clifford_depolarization": float(p)}
        raise ValueError(
            f"Unsupported noise_model string: {noise_model!r}. "
            "Use 'none', 'depolarizing', dict, callable, or an object with apply_to_circuit."
        )

    if isinstance(noise_model, dict):
        clean: Dict[str, float] = {}
        for k, v in noise_model.items():
            if not isinstance(k, str):
                raise TypeError(f"Invalid noise dict key: {k!r} (must be str)")
            if not isinstance(v, (int, float)):
                raise TypeError(f"Invalid noise dict value at {k!r}: {v!r} (must be numeric)")
            clean[k] = float(v)
        return clean

    # callable or object with apply_to_circuit is handled after base construction
    return {}


def _is_custom_noise_model(noise_model: NoiseModel) -> bool:
    if callable(noise_model):
        return True
    return hasattr(noise_model, "apply_to_circuit")


def _apply_noise_object_compat(
    noise_model: NoiseApplierProtocol,
    circuit: "stim.Circuit",
    p: float,
) -> Optional["stim.Circuit"]:
    """
    Compatibility with two common contracts:
      - apply_to_circuit(circuit, p)
      - apply_to_circuit(circuit)
    """
    apply_fn = noise_model.apply_to_circuit

    try:
        sig = inspect.signature(apply_fn)
    except (TypeError, ValueError):
        # Fallback defensivo cuando la firma not es introspectable
        try:
            return apply_fn(circuit, p)
        except TypeError:
            return apply_fn(circuit)

    try:
        sig.bind(circuit, p)
        return apply_fn(circuit, p)
    except TypeError:
        pass

    try:
        sig.bind(circuit)
        return apply_fn(circuit)
    except TypeError as exc:
        raise TypeError(
            "noise_model object must expose apply_to_circuit(circuit) "
            "or apply_to_circuit(circuit, p)."
        ) from exc


# -----------------------------
# Public API
# -----------------------------
def generate_xzzx_circuit(
    distance: int,
    rounds: int,
    noise_model: NoiseModel = "none",
    p: float = 0.0,
    logical_basis: str = "x",
) -> "stim.Circuit":
    """
    Generate XZZX circuit (robust Week 1 startup using Stim rotated_memory_* templates).

    Parameters
    ----------
    distance : int
        Code distance (odd >= 3).
    rounds : int
        Number of correction rounds.
    noise_model : NoiseModel
        Can be:
          - "none" / "ideal"
          - "depolarizing"
          - dict de kwargs for stim.Circuit.generated
          - callable(circuit, p) -> Optional[circuit]
          - object with apply_to_circuit(circuit[, p]) -> Optional[circuit]
    p : float
        Physical error probability.
    logical_basis : str
        'x' or 'z', chooses rotated_memory_x or rotated_memory_z.

    Returns
    -------
    stim.Circuit
    """
    _require_stim()
    cfg = _validate_inputs(distance=distance, rounds=rounds, p=p, logical_basis=logical_basis)

    task = _task_name(cfg.logical_basis)

    # 1) Base construction (without custom noise)
    if _is_custom_noise_model(noise_model):
        base = stim.Circuit.generated(
            task,
            distance=cfg.distance,
            rounds=cfg.rounds,
        )
        # 2) Custom application
        if callable(noise_model):
            out = noise_model(base, cfg.p)
        else:
            # object with apply_to_circuit(circuit[, p])
            out = _apply_noise_object_compat(noise_model, base, cfg.p)

        if out is None:
            return base
        if not isinstance(out, stim.Circuit):
            raise TypeError(
                "El noise model custom must devolver stim.Circuit o None."
            )
        return out

    # Standard noise_model (none/depolarizing/dict)
    kwargs = _stim_noise_kwargs(noise_model, cfg.p)
    if kwargs:
        return stim.Circuit.generated(
            task,
            distance=cfg.distance,
            rounds=cfg.rounds,
            **kwargs,
        )
    return stim.Circuit.generated(
        task,
        distance=cfg.distance,
        rounds=cfg.rounds,
    )


def build_mwpm_matcher(circuit: "stim.Circuit") -> "pymatching.Matching":
    """
    Build an MWPM matcher from the circuit detector error model.
    """
    _require_stim()
    _require_pymatching()

    if not isinstance(circuit, stim.Circuit):
        raise TypeError(f"circuit must be stim.Circuit, received: {type(circuit).__name__}")

    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    return matcher


def _decode_batch_compat(
    matcher: "pymatching.Matching",
    dets: np.ndarray,
) -> np.ndarray:
    """
    Compatibility with different PyMatching versions:
    - if decode_batch exists, use it;
    - if not, fallback by shot.
    Returns an array with shape (shots, n_obs_pred).
    """
    dets = np.asarray(dets, dtype=np.uint8)

    if dets.ndim != 2:
        raise ValueError(f"dets must have shape (shots, num_detectors). Received: {dets.shape}")

    # Preferred path (fast)
    if hasattr(matcher, "decode_batch"):
        preds = matcher.decode_batch(dets)
        preds = np.asarray(preds, dtype=np.uint8)
    else:
        # Fallback compatible
        rows = []
        for i in range(dets.shape[0]):
            pred_i = matcher.decode(dets[i])
            rows.append(np.asarray(pred_i, dtype=np.uint8))
        preds = np.asarray(rows, dtype=np.uint8)

    # Normalize shape to 2D
    if preds.ndim == 1:
        preds = preds[:, np.newaxis]
    return preds


def logical_error_rate_mwpm(
    circuit: "stim.Circuit",
    shots: int = 1000,
) -> float:
    """
    Estimate logical error rate using MWPM.

    Requires Stim to sample observables separately:
      sampler.sample(shots, separate_observables=True) -> (dets, obs)
    """
    _require_stim()
    _require_pymatching()

    if not isinstance(circuit, stim.Circuit):
        raise TypeError(f"circuit must be stim.Circuit, received: {type(circuit).__name__}")
    if not isinstance(shots, int) or shots <= 0:
        raise ValueError(f"shots must be int > 0. Received: {shots}")

    matcher = build_mwpm_matcher(circuit)
    sampler = circuit.compile_detector_sampler()

    try:
        sampled = sampler.sample(shots=shots, separate_observables=True)
    except TypeError as exc:
        raise RuntimeError(
            "Your Stim version does not support sample(..., separate_observables=True). "
            "Update stim to compute LER directly."
        ) from exc

    if not isinstance(sampled, tuple) or len(sampled) != 2:
        raise RuntimeError(
            "Expected Stim to return (detector_samples, observable_flips)."
        )

    dets, obs = sampled
    dets = np.asarray(dets, dtype=np.uint8)
    obs = np.asarray(obs, dtype=np.uint8)

    if obs.ndim == 1:
        obs = obs[:, np.newaxis]

    preds = _decode_batch_compat(matcher, dets)

    if preds.shape[0] != obs.shape[0]:
        raise RuntimeError(
            f"Shot mismatch between preds ({preds.shape[0]}) and obs ({obs.shape[0]})."
        )

    n = min(preds.shape[1], obs.shape[1])
    if n == 0:
        # No declared observables -> standard LER cannot be measured.
        raise RuntimeError(
            "The circuit does not expose logical observables needed to evaluate logical error rate."
        )

    logical_fail = np.any((preds[:, :n] & 1) != (obs[:, :n] & 1), axis=1)
    return float(np.mean(logical_fail))


def circuit_summary(circuit: "stim.Circuit") -> Dict[str, Any]:
    """
    Quick circuit summary for debugging/validation.
    """
    _require_stim()

    if not isinstance(circuit, stim.Circuit):
        raise TypeError(f"circuit must be stim.Circuit, received: {type(circuit).__name__}")

    summary: Dict[str, Any] = {
        "num_qubits": int(getattr(circuit, "num_qubits", -1)),
        "num_measurements": int(getattr(circuit, "num_measurements", -1)),
        "num_detectors": int(getattr(circuit, "num_detectors", -1)),
        "num_observables": int(getattr(circuit, "num_observables", -1)),
        "stim_version": getattr(stim, "__version__", "unknown"),
    }

    # Detector Error Model stats (if available)
    try:
        dem = circuit.detector_error_model(decompose_errors=True)
        summary["dem_num_detectors"] = int(getattr(dem, "num_detectors", -1))
        summary["dem_num_observables"] = int(getattr(dem, "num_observables", -1))
    except Exception as exc:  # pragma: no cover
        summary["dem_error"] = f"{type(exc).__name__}: {exc}"

    return summary


__all__ = [
    "generate_xzzx_circuit",
    "build_mwpm_matcher",
    "logical_error_rate_mwpm",
    "circuit_summary",
]
