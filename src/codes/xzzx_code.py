# src/codes/xzzx_code.py
from __future__ import annotations

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
    def apply_to_circuit(self, circuit: "stim.Circuit", p: float) -> Optional["stim.Circuit"]:
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
            "stim no está instalado o falló su importación. "
            "Instala con: pip install stim"
        ) from _STIM_IMPORT_ERROR


def _require_pymatching() -> None:
    if pymatching is None:
        raise ImportError(
            "pymatching no está instalado o falló su importación. "
            "Instala con: pip install pymatching"
        ) from _PYMATCHING_IMPORT_ERROR


def _validate_inputs(
    distance: int,
    rounds: int,
    p: float,
    logical_basis: str,
) -> _InputConfig:
    if not isinstance(distance, int):
        raise TypeError(f"distance debe ser int, recibido: {type(distance).__name__}")
    if distance < 3 or distance % 2 == 0:
        raise ValueError(f"distance debe ser impar y >= 3. Recibido: {distance}")

    if not isinstance(rounds, int):
        raise TypeError(f"rounds debe ser int, recibido: {type(rounds).__name__}")
    if rounds < 1:
        raise ValueError(f"rounds debe ser >= 1. Recibido: {rounds}")

    if not isinstance(p, (int, float)):
        raise TypeError(f"p debe ser float/int, recibido: {type(p).__name__}")
    p_float = float(p)
    if not (0.0 <= p_float <= 1.0):
        raise ValueError(f"p debe estar en [0, 1]. Recibido: {p_float}")

    if not isinstance(logical_basis, str):
        raise TypeError(f"logical_basis debe ser str, recibido: {type(logical_basis).__name__}")
    lb = logical_basis.lower().strip()
    if lb not in {"x", "z"}:
        raise ValueError(f"logical_basis debe ser 'x' o 'z'. Recibido: {logical_basis}")

    return _InputConfig(
        distance=distance,
        rounds=rounds,
        p=p_float,
        logical_basis=lb,
    )


def _task_name(logical_basis: str) -> str:
    """
    Usa el generador builtin de Stim para rotated memory code.
    Para Semana 1, esta es la vía más robusta para arrancar.
    """
    if logical_basis == "x":
        return "surface_code:rotated_memory_x"
    return "surface_code:rotated_memory_z"


def _stim_noise_kwargs(noise_model: NoiseModel, p: float) -> Dict[str, float]:
    """
    Traduce noise_model -> kwargs compatibles con stim.Circuit.generated(...).

    Soportado:
      - None / "none" / "ideal": sin ruido
      - "depolarizing": ruido sencillo uniforme
      - dict: se pasa tal cual (validando tipo numérico)
    """
    if noise_model is None:
        return {}

    if isinstance(noise_model, str):
        nm = noise_model.lower().strip()
        if nm in {"none", "ideal", "no_noise"}:
            return {}
        if nm in {"depolarizing", "depolarising", "uniform"}:
            # Ajuste simple para Semana 1:
            # - after_clifford_depolarization controla la tasa principal.
            # Puedes endurecer/afinar después.
            return {"after_clifford_depolarization": float(p)}
        raise ValueError(
            f"noise_model string no soportado: {noise_model!r}. "
            "Usa 'none', 'depolarizing', dict, callable, o objeto con apply_to_circuit."
        )

    if isinstance(noise_model, dict):
        clean: Dict[str, float] = {}
        for k, v in noise_model.items():
            if not isinstance(k, str):
                raise TypeError(f"Clave de noise dict inválida: {k!r} (debe ser str)")
            if not isinstance(v, (int, float)):
                raise TypeError(f"Valor de noise dict inválido en {k!r}: {v!r} (debe ser numérico)")
            clean[k] = float(v)
        return clean

    # callable u objeto con apply_to_circuit se manejan fuera (post construcción base)
    return {}


def _is_custom_noise_model(noise_model: NoiseModel) -> bool:
    if callable(noise_model):
        return True
    return hasattr(noise_model, "apply_to_circuit")


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
    Genera circuito XZZX (arranque robusto de Semana 1 usando template rotated_memory_* de Stim).

    Parámetros
    ----------
    distance : int
        Distancia del código (impar >= 3).
    rounds : int
        Número de rondas de corrección.
    noise_model : NoiseModel
        Puede ser:
          - "none" / "ideal"
          - "depolarizing"
          - dict de kwargs para stim.Circuit.generated
          - callable(circuit, p) -> Optional[circuit]
          - objeto con apply_to_circuit(circuit, p) -> Optional[circuit]
    p : float
        Probabilidad de error físico.
    logical_basis : str
        'x' o 'z', elige template rotated_memory_x o rotated_memory_z.

    Returns
    -------
    stim.Circuit
    """
    _require_stim()
    cfg = _validate_inputs(distance=distance, rounds=rounds, p=p, logical_basis=logical_basis)

    task = _task_name(cfg.logical_basis)

    # 1) Construcción base (sin ruido custom)
    if _is_custom_noise_model(noise_model):
        base = stim.Circuit.generated(
            task,
            distance=cfg.distance,
            rounds=cfg.rounds,
        )
        # 2) Aplicación custom
        if callable(noise_model):
            out = noise_model(base, cfg.p)
        else:
            # objeto con apply_to_circuit
            out = noise_model.apply_to_circuit(base, cfg.p)  # type: ignore[attr-defined]

        if out is None:
            return base
        if not isinstance(out, stim.Circuit):
            raise TypeError(
                "El noise model custom debe devolver stim.Circuit o None."
            )
        return out

    # noise_model estándar (none/depolarizing/dict)
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
    Construye matcher MWPM desde detector error model del circuito.
    """
    _require_stim()
    _require_pymatching()

    if not isinstance(circuit, stim.Circuit):
        raise TypeError(f"circuit debe ser stim.Circuit, recibido: {type(circuit).__name__}")

    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    return matcher


def _decode_batch_compat(
    matcher: "pymatching.Matching",
    dets: np.ndarray,
) -> np.ndarray:
    """
    Compatibilidad con distintas versiones de PyMatching:
    - si existe decode_batch, lo usa;
    - si no, fallback por shot.
    Devuelve array shape (shots, n_obs_pred).
    """
    dets = np.asarray(dets, dtype=np.uint8)

    if dets.ndim != 2:
        raise ValueError(f"dets debe tener shape (shots, num_detectors). Recibido: {dets.shape}")

    # Ruta preferida (rápida)
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

    # Normalizar forma a 2D
    if preds.ndim == 1:
        preds = preds[:, np.newaxis]
    return preds


def logical_error_rate_mwpm(
    circuit: "stim.Circuit",
    shots: int = 1000,
) -> float:
    """
    Estima logical error rate usando MWPM.

    Requiere que Stim pueda muestrear observables separadamente:
      sampler.sample(shots, separate_observables=True) -> (dets, obs)
    """
    _require_stim()
    _require_pymatching()

    if not isinstance(circuit, stim.Circuit):
        raise TypeError(f"circuit debe ser stim.Circuit, recibido: {type(circuit).__name__}")
    if not isinstance(shots, int) or shots <= 0:
        raise ValueError(f"shots debe ser int > 0. Recibido: {shots}")

    matcher = build_mwpm_matcher(circuit)
    sampler = circuit.compile_detector_sampler()

    try:
        sampled = sampler.sample(shots=shots, separate_observables=True)
    except TypeError as exc:
        raise RuntimeError(
            "Tu versión de Stim no soporta sample(..., separate_observables=True). "
            "Actualiza stim para calcular LER de forma directa."
        ) from exc

    if not isinstance(sampled, tuple) or len(sampled) != 2:
        raise RuntimeError(
            "Se esperaba que Stim devolviera (detector_samples, observable_flips)."
        )

    dets, obs = sampled
    dets = np.asarray(dets, dtype=np.uint8)
    obs = np.asarray(obs, dtype=np.uint8)

    if obs.ndim == 1:
        obs = obs[:, np.newaxis]

    preds = _decode_batch_compat(matcher, dets)

    if preds.shape[0] != obs.shape[0]:
        raise RuntimeError(
            f"Mismatch de shots entre preds ({preds.shape[0]}) y obs ({obs.shape[0]})."
        )

    n = min(preds.shape[1], obs.shape[1])
    if n == 0:
        # No observables declarados -> no se puede medir LER de forma estándar.
        raise RuntimeError(
            "El circuito no parece exponer observables lógicos para evaluar logical error rate."
        )

    logical_fail = np.any((preds[:, :n] & 1) != (obs[:, :n] & 1), axis=1)
    return float(np.mean(logical_fail))


def circuit_summary(circuit: "stim.Circuit") -> Dict[str, Any]:
    """
    Resumen rápido del circuito para debugging/validación.
    """
    _require_stim()

    if not isinstance(circuit, stim.Circuit):
        raise TypeError(f"circuit debe ser stim.Circuit, recibido: {type(circuit).__name__}")

    summary: Dict[str, Any] = {
        "num_qubits": int(getattr(circuit, "num_qubits", -1)),
        "num_measurements": int(getattr(circuit, "num_measurements", -1)),
        "num_detectors": int(getattr(circuit, "num_detectors", -1)),
        "num_observables": int(getattr(circuit, "num_observables", -1)),
        "stim_version": getattr(stim, "__version__", "unknown"),
    }

    # Detector Error Model stats (si están disponibles)
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
