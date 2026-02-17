# src/noise/noise_calibration.py
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pymatching
import stim

from src.noise.noise_models import build_noise_model


# =============================================================================
# Public dataclasses
# =============================================================================
@dataclass(frozen=True)
class CalibrationCase:
    """
    Defines a base calibration case over which noise parameters are swept.
    """
    case_name: str
    distance: int
    rounds: int
    p: float
    logical_basis: str = "x"

    def __post_init__(self) -> None:
        if self.distance < 3 or self.distance % 2 == 0:
            raise ValueError(
                f"distance must be odd and >= 3. Received: {self.distance}"
            )
        if self.rounds <= 0:
            raise ValueError(f"rounds must be > 0. Received: {self.rounds}")
        if not (0.0 <= self.p <= 1.0):
            raise ValueError(f"p must be in [0,1]. Received: {self.p}")
        if self.logical_basis not in {"x", "z"}:
            raise ValueError(
                f"logical_basis must be 'x' or 'z'. Received: {self.logical_basis}"
            )


@dataclass(frozen=True)
class SweepSpec:
    """
    Specifies a 1D sweep of one parameter over a noise model.
    """
    model_type: str
    param_name: str
    values: Tuple[float, ...]
    base_params: Dict[str, Any] = field(default_factory=dict)
    objective: str = "min_ler"  # opciones: min_ler, min_time

    def __post_init__(self) -> None:
        if not self.model_type or not isinstance(self.model_type, str):
            raise ValueError("model_type must be a non-empty string.")
        if not self.param_name or not isinstance(self.param_name, str):
            raise ValueError("param_name must be a non-empty string.")
        if not self.values:
            raise ValueError("values cannot be empty.")
        if self.objective not in {"min_ler", "min_time"}:
            raise ValueError(
                f"Invalid objective: {self.objective}. Use 'min_ler' or 'min_time'."
            )


@dataclass(frozen=True)
class CalibrationConfig:
    """
    Global calibration config.
    """
    shots: int = 500
    seed: int = 20260216
    keep_soft_info_samples: int = 0

    def __post_init__(self) -> None:
        if self.shots <= 0:
            raise ValueError(f"shots must be > 0. Received: {self.shots}")
        if self.keep_soft_info_samples < 0:
            raise ValueError(
                "keep_soft_info_samples must be >= 0. "
                f"Received: {self.keep_soft_info_samples}"
            )


# =============================================================================
# Flexible base XZZX circuit builder
# =============================================================================
def _maybe_extract_circuit(obj: Any) -> Optional[stim.Circuit]:
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
    fn: Callable[..., Any],
    *,
    distance: int,
    rounds: int,
    logical_basis: str,
) -> Optional[stim.Circuit]:
    import inspect

    sig = inspect.signature(fn)
    kwargs: Dict[str, Any] = {}

    name_map: Dict[str, Any] = {
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
        # Intento ideal/not-noise if el builder lo expone:
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

    # Intento with kwargs compatibles
    try:
        out = fn(**kwargs)
        c = _maybe_extract_circuit(out)
        if c is not None:
            return c
    except Exception:
        pass

    # Fallback mínimo posicional
    try:
        out = fn(distance, rounds)
        c = _maybe_extract_circuit(out)
        if c is not None:
            return c
    except Exception:
        pass

    return None


def build_base_xzzx_circuit(
    *,
    distance: int,
    rounds: int,
    logical_basis: str = "x",
) -> stim.Circuit:
    """
    Build el circuit base XZZX using discovery flexible sobre src.codes.xzzx_code.
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

    # 1) Attempt expected names
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

    # 2) Exploración amplia
    for name in dir(xc):
        lname = name.lower()
        if "xzzx" not in lname or "circuit" not in lname:
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

    raise RuntimeError(
        "No compatible XZZX circuit builder found in src.codes.xzzx_code."
    )


# =============================================================================
# Simulation / decoding core
# =============================================================================
def _stable_name_seed(name: str) -> int:
    """
    Stable hash (independent of Python hash randomization).
    """
    return sum((i + 1) * ord(ch) for i, ch in enumerate(name))


def _validate_probability_like(name: str, value: Any) -> None:
    if name in {"p", "p_meas", "p_idle", "correlation_strength"}:
        try:
            v = float(value)
        except Exception as exc:
            raise ValueError(f"{name} must be numeric. Received: {value!r}") from exc
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0,1]. Received: {v}")


def estimate_ler_and_time(
    circuit: stim.Circuit,
    *,
    shots: int,
    seed: int,
    keep_soft_info_samples: int = 0,
) -> Dict[str, Any]:
    """
    Estimate logical error rate (LER) and mean decode time.
    """
    if shots <= 0:
        raise ValueError(f"shots must be > 0. Received: {shots}")
    if keep_soft_info_samples < 0:
        raise ValueError(
            "keep_soft_info_samples must be >= 0. "
            f"Received: {keep_soft_info_samples}"
        )

    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)

    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots=shots, separate_observables=True)

    import time

    t0 = time.perf_counter()
    pred = matcher.decode_batch(dets)
    decode_total = time.perf_counter() - t0

    if getattr(pred, "ndim", 1) == 1:
        pred = pred.reshape(-1, 1)
    if getattr(obs, "ndim", 1) == 1:
        obs = obs.reshape(-1, 1)

    if obs.shape[1] < 1:
        raise RuntimeError("Circuit has no logical observables (OBSERVABLE_INCLUDE).")

    ler = float((pred[:, 0] != obs[:, 0]).mean())

    out: Dict[str, Any] = {
        "ler": ler,
        "num_detectors": int(dets.shape[1]),
        "num_observables": int(obs.shape[1]),
        "decode_total_time_sec": float(decode_total),
        "avg_decode_time_sec": float(decode_total / shots),
    }

    if keep_soft_info_samples > 0:
        k = min(keep_soft_info_samples, shots)
        samples: List[Dict[str, float]] = []
        # Simple and cheap soft proxy (without touching decoder internals):
        # syndrome weight = number of active detectors
        for i in range(k):
            sw = float(dets[i].sum())
            samples.append({"syndrome_weight": sw})
        out["soft_info_samples"] = samples

    return out


def _build_model_spec(
    *,
    model_type: str,
    case_p: float,
    base_params: Optional[Mapping[str, Any]],
    sweep_param_name: str,
    sweep_value: Any,
) -> Dict[str, Any]:
    spec: Dict[str, Any] = {"type": model_type, "p": case_p}
    if base_params:
        spec.update(dict(base_params))

    spec[sweep_param_name] = sweep_value

    # Light validation of probability-like parameters
    for k, v in spec.items():
        _validate_probability_like(k, v)

    return spec


def run_single_calibration_point(
    *,
    case: CalibrationCase,
    model_type: str,
    sweep_param_name: str,
    sweep_value: Any,
    config: CalibrationConfig,
    base_params: Optional[Mapping[str, Any]] = None,
    experiment_seed: int,
    base_circuit: Optional[stim.Circuit] = None,
) -> Dict[str, Any]:
    """
    Run one calibration point:
      (case, model_type, sweep_param_name=sweep_value).
    """
    if base_circuit is None:
        base_circuit = build_base_xzzx_circuit(
            distance=case.distance,
            rounds=case.rounds,
            logical_basis=case.logical_basis,
        )

    model_spec = _build_model_spec(
        model_type=model_type,
        case_p=case.p,
        base_params=base_params,
        sweep_param_name=sweep_param_name,
        sweep_value=sweep_value,
    )

    model = build_noise_model(model_spec)
    noisy = model.apply_to_circuit(base_circuit)

    metrics = estimate_ler_and_time(
        noisy,
        shots=config.shots,
        seed=experiment_seed,
        keep_soft_info_samples=config.keep_soft_info_samples,
    )

    return {
        "case_name": case.case_name,
        "distance": case.distance,
        "rounds": case.rounds,
        "logical_basis": case.logical_basis,
        "model_type": model_type,
        "model_spec": model_spec,
        "sweep_param_name": sweep_param_name,
        "sweep_value": sweep_value,
        "shots": config.shots,
        "seed": int(experiment_seed),
        **metrics,
        "status": "ok",
    }


def _select_best_point(
    points: Sequence[Dict[str, Any]],
    *,
    objective: str,
) -> Dict[str, Any]:
    if not points:
        raise ValueError("No points available to select best_point.")

    if objective == "min_time":
        # Tie-break by LER
        return min(points, key=lambda x: (float(x["avg_decode_time_sec"]), float(x["ler"])))

    # Default: min_ler, tie-break by time
    return min(points, key=lambda x: (float(x["ler"]), float(x["avg_decode_time_sec"])))


def _mean(values: Iterable[float]) -> Optional[float]:
    arr = list(values)
    if not arr:
        return None
    return float(sum(arr) / len(arr))


def run_noise_sweep(
    *,
    cases: Sequence[CalibrationCase],
    sweep: SweepSpec,
    config: CalibrationConfig,
) -> Dict[str, Any]:
    """
    Run a 1D sweep of a noise parameter for one model.

    Returns a JSON-friendly report with:
    - results by case and value,
    - best value by case,
    - global aggregates by value,
    - global best value.
    """
    if not cases:
        raise ValueError("cases cannot be empty.")

    # Build base circuits once per case (avoids overhead).
    base_by_case: Dict[str, stim.Circuit] = {}
    for c in cases:
        base_by_case[c.case_name] = build_base_xzzx_circuit(
            distance=c.distance,
            rounds=c.rounds,
            logical_basis=c.logical_basis,
        )

    case_results: List[Dict[str, Any]] = []
    value_buckets: Dict[Any, List[Dict[str, Any]]] = {v: [] for v in sweep.values}

    base_seed = config.seed + _stable_name_seed(sweep.model_type) + _stable_name_seed(sweep.param_name)

    for ci, case in enumerate(cases):
        points: List[Dict[str, Any]] = []

        for vi, value in enumerate(sweep.values):
            exp_seed = base_seed + ci * 100_000 + vi * 997
            point = run_single_calibration_point(
                case=case,
                model_type=sweep.model_type,
                sweep_param_name=sweep.param_name,
                sweep_value=value,
                config=config,
                base_params=sweep.base_params,
                experiment_seed=exp_seed,
                base_circuit=base_by_case[case.case_name],
            )
            points.append(point)
            value_buckets[value].append(point)

        best = _select_best_point(points, objective=sweep.objective)

        case_results.append(
            {
                "case": asdict(case),
                "points": points,
                "best_point": best,
                "status": "ok",
            }
        )

    # Aggregates by sweep value
    per_value_summary: List[Dict[str, Any]] = []
    for value in sweep.values:
        pts = value_buckets[value]
        per_value_summary.append(
            {
                "sweep_value": value,
                "num_points": len(pts),
                "mean_ler": _mean(float(p["ler"]) for p in pts),
                "mean_avg_decode_time_sec": _mean(float(p["avg_decode_time_sec"]) for p in pts),
            }
        )

    if sweep.objective == "min_time":
        global_best = min(
            per_value_summary,
            key=lambda x: (float(x["mean_avg_decode_time_sec"]), float(x["mean_ler"])),
        )
    else:
        global_best = min(
            per_value_summary,
            key=lambda x: (float(x["mean_ler"]), float(x["mean_avg_decode_time_sec"])),
        )

    report: Dict[str, Any] = {
        "metadata": {
            "report_name": "noise_calibration_sweep",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "sweep": {
            "model_type": sweep.model_type,
            "param_name": sweep.param_name,
            "values": list(sweep.values),
            "base_params": dict(sweep.base_params),
            "objective": sweep.objective,
        },
        "config": asdict(config),
        "cases_summary": case_results,
        "aggregates": {
            "num_cases": len(cases),
            "per_value_summary": per_value_summary,
            "global_best": global_best,
            "mean_best_case_ler": _mean(
                float(c["best_point"]["ler"]) for c in case_results
            ),
            "mean_best_case_avg_decode_time_sec": _mean(
                float(c["best_point"]["avg_decode_time_sec"]) for c in case_results
            ),
        },
        "status": "ok",
    }
    return report


def run_multi_model_calibration(
    *,
    cases: Sequence[CalibrationCase],
    sweeps: Sequence[SweepSpec],
    config: CalibrationConfig,
) -> Dict[str, Any]:
    """
    Run multiple sweeps (possibly with different models/parameters)
    and return a unified report.
    """
    if not sweeps:
        raise ValueError("sweeps cannot be empty.")

    sweep_reports: List[Dict[str, Any]] = []
    for sw in sweeps:
        sweep_reports.append(
            run_noise_sweep(
                cases=cases,
                sweep=sw,
                config=config,
            )
        )

    # Compact summary
    compact: List[Dict[str, Any]] = []
    for rep in sweep_reports:
        sw = rep["sweep"]
        gb = rep["aggregates"]["global_best"]
        compact.append(
            {
                "model_type": sw["model_type"],
                "param_name": sw["param_name"],
                "objective": sw["objective"],
                "best_value": gb["sweep_value"],
                "mean_ler_at_best": gb["mean_ler"],
                "mean_avg_decode_time_sec_at_best": gb["mean_avg_decode_time_sec"],
            }
        )

    return {
        "metadata": {
            "report_name": "noise_calibration_multi_model",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "config": asdict(config),
        "num_sweeps": len(sweep_reports),
        "sweeps_summary": compact,
        "sweeps_reports": sweep_reports,
        "status": "ok",
    }


# =============================================================================
# I/O utilities
# =============================================================================
def save_calibration_report(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """
    Save a report (JSON-friendly dict) to disk.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    import json

    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return path


def load_calibration_report(input_path: str | Path) -> Dict[str, Any]:
    """
    Load a calibration JSON report.
    """
    path = Path(input_path)
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


__all__ = [
    "CalibrationCase",
    "SweepSpec",
    "CalibrationConfig",
    "build_base_xzzx_circuit",
    "estimate_ler_and_time",
    "run_single_calibration_point",
    "run_noise_sweep",
    "run_multi_model_calibration",
    "save_calibration_report",
    "load_calibration_report",
]
