# scripts/run_week1_person2_noise_benchmark.py
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# Bootstrap de ruta para permitir ejecutar:
#   py -3.10 scripts/run_week1_person2_noise_benchmark.py
# además de:
#   py -3.10 -m scripts.run_week1_person2_noise_benchmark
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pymatching
import stim

from src.noise.noise_models import build_noise_model


def _configure_stdio_utf8() -> None:
    """
    Evita errores de impresión Unicode en Windows (cp1252).
    """
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        # Si no se puede reconfigurar, continuamos igualmente.
        pass


# ---------------------------------------------------------------------
# Utilidades de descubrimiento de builder XZZX
# ---------------------------------------------------------------------
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
        # Intenta forzar ideal/no-noise si el builder lo admite:
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

    # Llamada con kwargs compatibles
    try:
        out = fn(**kwargs)
        c = _maybe_extract_circuit(out)
        if c is not None:
            return c
    except Exception:
        pass

    # Fallback posicional mínimo
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

    # 1) Intento por nombres preferidos
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

    # 2) Exploración amplia de callables del módulo
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
        "No se encontró un builder de circuito XZZX compatible en src.codes.xzzx_code."
    )


# ---------------------------------------------------------------------
# Métricas de benchmark
# ---------------------------------------------------------------------
def estimate_ler_and_time(
    circuit: stim.Circuit,
    *,
    shots: int,
    seed: int,
) -> Dict[str, Any]:
    if shots <= 0:
        raise ValueError(f"shots debe ser > 0. Recibido: {shots}")

    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)

    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots=shots, separate_observables=True)

    t0 = time.perf_counter()
    pred = matcher.decode_batch(dets)
    decode_total = time.perf_counter() - t0

    if getattr(pred, "ndim", 1) == 1:
        pred = pred.reshape(-1, 1)
    if getattr(obs, "ndim", 1) == 1:
        obs = obs.reshape(-1, 1)

    if obs.shape[1] < 1:
        raise RuntimeError("El circuito no contiene observables lógicos para medir LER.")

    ler = float((pred[:, 0] != obs[:, 0]).mean())

    return {
        "ler": ler,
        "num_detectors": int(dets.shape[1]),
        "num_observables": int(obs.shape[1]),
        "decode_total_time_sec": float(decode_total),
        "avg_decode_time_sec": float(decode_total / shots),
    }


def default_noise_specs(p: float) -> List[Dict[str, Any]]:
    return [
        {"name": "depolarizing", "type": "depolarizing", "p": p},
        {"name": "biased", "type": "biased", "p": p, "eta": 100.0},
        {"name": "circuit_level", "type": "circuit_level", "p": p},
        {"name": "phenomenological", "type": "phenomenological", "p": p},
        {
            "name": "correlated",
            "type": "correlated",
            "p": p,
            "correlation_length": 2,
            "correlation_strength": 0.8,
            "topology": "line",
            "pauli": "Z",
        },
    ]


def run_case(
    *,
    case_name: str,
    distance: int,
    rounds: int,
    p: float,
    shots: int,
    logical_basis: str,
    base_seed: int,
) -> Dict[str, Any]:
    base_circuit = build_base_xzzx_circuit(
        distance=distance,
        rounds=rounds,
        logical_basis=logical_basis,
    )

    models_out: List[Dict[str, Any]] = []
    specs = default_noise_specs(p)

    for idx, spec in enumerate(specs):
        seed = base_seed + idx * 97
        model_name = str(spec["name"])

        # build_noise_model no acepta "name"; se usa solo como etiqueta de reporte
        model_spec = {k: v for k, v in spec.items() if k != "name"}

        model = build_noise_model(model_spec)
        noisy = model.apply_to_circuit(base_circuit)
        metrics = estimate_ler_and_time(noisy, shots=shots, seed=seed)

        models_out.append(
            {
                "model_name": model_name,
                "model_spec": model_spec,
                "seed": int(seed),
                "ler": metrics["ler"],
                "avg_decode_time_sec": metrics["avg_decode_time_sec"],
                "decode_total_time_sec": metrics["decode_total_time_sec"],
                "num_detectors": metrics["num_detectors"],
                "num_observables": metrics["num_observables"],
            }
        )

    # Delta correlated vs depolarizing
    dep = next((m for m in models_out if m["model_name"] == "depolarizing"), None)
    cor = next((m for m in models_out if m["model_name"] == "correlated"), None)
    delta_corr_vs_dep = None
    if dep is not None and cor is not None:
        delta_corr_vs_dep = float(cor["ler"] - dep["ler"])

    return {
        "case_name": case_name,
        "distance": int(distance),
        "rounds": int(rounds),
        "p": float(p),
        "shots": int(shots),
        "logical_basis": logical_basis,
        "num_qubits": int(base_circuit.num_qubits),
        "models": models_out,
        "delta_ler_correlated_minus_depolarizing": delta_corr_vs_dep,
        "status": "ok",
    }


def aggregate_results(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_model: Dict[str, List[float]] = {}
    by_model_time: Dict[str, List[float]] = {}
    corr_minus_dep: List[float] = []

    for case in cases:
        for m in case["models"]:
            name = str(m["model_name"])
            by_model.setdefault(name, []).append(float(m["ler"]))
            by_model_time.setdefault(name, []).append(float(m["avg_decode_time_sec"]))

        d = case.get("delta_ler_correlated_minus_depolarizing")
        if d is not None:
            corr_minus_dep.append(float(d))

    mean_ler_by_model = {
        k: (sum(v) / len(v) if v else None) for k, v in by_model.items()
    }
    mean_time_by_model = {
        k: (sum(v) / len(v) if v else None) for k, v in by_model_time.items()
    }

    return {
        "mean_ler_by_model": mean_ler_by_model,
        "mean_avg_decode_time_sec_by_model": mean_time_by_model,
        "mean_delta_ler_correlated_minus_depolarizing": (
            sum(corr_minus_dep) / len(corr_minus_dep) if corr_minus_dep else None
        ),
    }


def print_table(cases: List[Dict[str, Any]]) -> None:
    print("\n=== Week 1 Person 2 - Noise Benchmark ===")
    print(f"{'CASE':<16} {'MODEL':<16} {'LER':>8} {'avg_t(s)':>12} {'det':>6} {'obs':>6}")
    print("-" * 72)

    for case in cases:
        case_name = str(case["case_name"])
        for m in case["models"]:
            print(
                f"{case_name:<16} "
                f"{m['model_name']:<16} "
                f"{float(m['ler']):>8.4f} "
                f"{float(m['avg_decode_time_sec']):>12.6f} "
                f"{int(m['num_detectors']):>6d} "
                f"{int(m['num_observables']):>6d}"
            )

        d = case.get("delta_ler_correlated_minus_depolarizing")
        if d is not None:
            print(
                f"{'':<16} {'delta corr-depol':<16} {float(d):>8.4f} {'':>12} {'':>6} {'':>6}"
            )

        print("-" * 72)


def _positive_int(value: str) -> int:
    iv = int(value)
    if iv <= 0:
        raise argparse.ArgumentTypeError("Debe ser un entero > 0.")
    return iv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week1 Person2 benchmark de modelos de ruido."
    )
    parser.add_argument(
        "--shots",
        type=_positive_int,
        default=400,
        help="Número de shots por caso/modelo (default: 400).",
    )
    parser.add_argument(
        "--logical-basis",
        type=str,
        default="x",
        choices=["x", "z"],
        help="Base lógica para el circuito XZZX (default: x).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/week1_person2_noise_benchmark.json",
        help="Ruta JSON de salida.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260216,
        help="Semilla base para muestreo.",
    )
    return parser.parse_args()


def main() -> None:
    _configure_stdio_utf8()
    args = parse_args()

    cases_cfg = [
        {"case_name": "d3_r2_p0.005", "distance": 3, "rounds": 2, "p": 0.005},
        {"case_name": "d3_r3_p0.010", "distance": 3, "rounds": 3, "p": 0.010},
        {"case_name": "d5_r3_p0.010", "distance": 5, "rounds": 3, "p": 0.010},
    ]

    cases_out: List[Dict[str, Any]] = []
    for i, c in enumerate(cases_cfg):
        case_res = run_case(
            case_name=str(c["case_name"]),
            distance=int(c["distance"]),
            rounds=int(c["rounds"]),
            p=float(c["p"]),
            shots=int(args.shots),
            logical_basis=str(args.logical_basis),
            base_seed=int(args.seed) + i * 1000,
        )
        cases_out.append(case_res)

    aggregates = aggregate_results(cases_out)
    print_table(cases_out)

    print("\n--- Aggregates ---")
    for key, value in aggregates.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subk, subv in value.items():
                if isinstance(subv, float):
                    print(f"  - {subk}: {subv:.6f}")
                else:
                    print(f"  - {subk}: {subv}")
        else:
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")

    payload: Dict[str, Any] = {
        "metadata": {
            "report_name": "week1_person2_noise_benchmark",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "shots_per_model": int(args.shots),
            "logical_basis": str(args.logical_basis),
            "seed": int(args.seed),
            "num_cases": len(cases_out),
        },
        "cases_summary": cases_out,
        "aggregates": aggregates,
        "status": "ok",
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nJSON guardado en: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
