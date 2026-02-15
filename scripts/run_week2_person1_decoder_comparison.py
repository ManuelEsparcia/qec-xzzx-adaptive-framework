# scripts/run_week2_person1_decoder_comparison.py
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

# Asegura import de "src" al ejecutar como script suelto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.codes.xzzx_code import generate_xzzx_circuit  # noqa: E402
from src.decoders.mwpm_decoder import MWPMDecoderWithSoftInfo  # noqa: E402
from src.decoders.union_find_decoder import UnionFindDecoderWithSoftInfo  # noqa: E402
from src.switching.adaptive_decoder import AdaptiveConfig, AdaptiveDecoder  # noqa: E402


@dataclass(frozen=True)
class ComparisonCase:
    name: str
    distance: int
    rounds: int
    p: float
    noise_model: Any = "depolarizing"
    logical_basis: str = "x"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_cases() -> List[ComparisonCase]:
    # Base robusta y rápida para evidencia de Persona 1 en Semana 2
    return [
        ComparisonCase(name="d3_r2_p0.005", distance=3, rounds=2, p=0.005),
        ComparisonCase(name="d3_r3_p0.010", distance=3, rounds=3, p=0.010),
        ComparisonCase(name="d5_r3_p0.010", distance=5, rounds=3, p=0.010),
    ]


def safe_speedup(ref_time: float, candidate_time: float) -> float:
    if candidate_time <= 0:
        return float("nan")
    return float(ref_time / candidate_time)


def run_one_case(
    case: ComparisonCase,
    shots: int,
    keep_soft: int,
    g_threshold: float,
) -> Dict[str, Any]:
    circuit = generate_xzzx_circuit(
        distance=case.distance,
        rounds=case.rounds,
        noise_model=case.noise_model,
        p=case.p,
        logical_basis=case.logical_basis,
    )

    mwpm = MWPMDecoderWithSoftInfo(circuit)
    uf = UnionFindDecoderWithSoftInfo(circuit, prefer_union_find=True)
    adaptive = AdaptiveDecoder(
        circuit=circuit,
        fast_decoder=uf,
        accurate_decoder=mwpm,
        config=AdaptiveConfig(
            g_threshold=g_threshold,
            compare_against_mwpm_in_benchmark=False,  # referencia MWPM ya la medimos aparte
        ),
    )

    # Benchmarks independientes
    mwpm_res = mwpm.benchmark(shots=shots, keep_soft_info_samples=keep_soft)
    uf_res = uf.benchmark(shots=shots, keep_soft_info_samples=keep_soft)
    adp_res = adaptive.benchmark_adaptive(
        shots=shots,
        g_threshold=g_threshold,
        keep_samples=keep_soft,
        compare_against_mwpm=False,
    )

    mwpm_er = float(mwpm_res["error_rate"])
    mwpm_t = float(mwpm_res["avg_decode_time"])

    uf_er = float(uf_res["error_rate"])
    uf_t = float(uf_res["avg_decode_time"])

    adp_er = float(adp_res["error_rate_adaptive"])
    adp_t = float(adp_res["avg_decode_time_adaptive"])
    switch_rate = float(adp_res["switch_rate"])

    row: Dict[str, Any] = {
        "case_name": case.name,
        "distance": case.distance,
        "rounds": case.rounds,
        "p": case.p,
        "noise_model": case.noise_model,
        "logical_basis": case.logical_basis,
        "shots": shots,

        "mwpm_error_rate": mwpm_er,
        "mwpm_avg_decode_time_sec": mwpm_t,

        "uf_error_rate": uf_er,
        "uf_avg_decode_time_sec": uf_t,
        "uf_speedup_vs_mwpm": safe_speedup(mwpm_t, uf_t),

        "adaptive_error_rate": adp_er,
        "adaptive_avg_decode_time_sec": adp_t,
        "adaptive_speedup_vs_mwpm": safe_speedup(mwpm_t, adp_t),
        "adaptive_switch_rate": switch_rate,
        "g_threshold": g_threshold,

        "num_detectors": int(adp_res["num_detectors"]),
        "num_observables": int(adp_res["num_observables"]),
        "status": "ok",
    }

    # Guardamos también trozos útiles de trazabilidad
    row["backend_info_uf"] = uf_res.get("backend_info", {})
    row["samples_adaptive"] = adp_res.get("samples", [])
    return row


def run_comparison(
    cases: List[ComparisonCase],
    shots: int,
    keep_soft: int,
    g_threshold: float,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    for c in cases:
        rows.append(
            run_one_case(
                case=c,
                shots=shots,
                keep_soft=keep_soft,
                g_threshold=g_threshold,
            )
        )

    agg: Dict[str, Any] = {
        "mean_mwpm_error_rate": mean([r["mwpm_error_rate"] for r in rows]) if rows else float("nan"),
        "mean_uf_error_rate": mean([r["uf_error_rate"] for r in rows]) if rows else float("nan"),
        "mean_adaptive_error_rate": mean([r["adaptive_error_rate"] for r in rows]) if rows else float("nan"),

        "mean_mwpm_avg_decode_time_sec": mean([r["mwpm_avg_decode_time_sec"] for r in rows]) if rows else float("nan"),
        "mean_uf_avg_decode_time_sec": mean([r["uf_avg_decode_time_sec"] for r in rows]) if rows else float("nan"),
        "mean_adaptive_avg_decode_time_sec": mean([r["adaptive_avg_decode_time_sec"] for r in rows]) if rows else float("nan"),

        "mean_uf_speedup_vs_mwpm": mean([r["uf_speedup_vs_mwpm"] for r in rows]) if rows else float("nan"),
        "mean_adaptive_speedup_vs_mwpm": mean([r["adaptive_speedup_vs_mwpm"] for r in rows]) if rows else float("nan"),
        "mean_adaptive_switch_rate": mean([r["adaptive_switch_rate"] for r in rows]) if rows else float("nan"),
    }

    report = {
        "metadata": {
            "report_name": "week2_person1_decoder_comparison",
            "timestamp_utc": utc_now_iso(),
            "num_cases": len(cases),
            "shots_per_case": shots,
            "keep_soft_info_samples": keep_soft,
            "g_threshold": g_threshold,
        },
        "cases_summary": rows,
        "aggregates": agg,
    }
    return report


def save_json(data: Dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return output_path


def print_table(report: Dict[str, Any]) -> None:
    rows = report.get("cases_summary", [])
    print("\n=== Week 2 Person 1 - Decoder Comparison (MWPM vs UF vs Adaptive) ===")
    if not rows:
        print("No hay casos para mostrar.")
        return

    header = (
        f"{'CASE':<15} {'d':>2} {'r':>2} {'p':>7} "
        f"{'ER_MWPM':>9} {'ER_UF':>9} {'ER_ADP':>9} "
        f"{'t_MWPM':>10} {'t_UF':>10} {'t_ADP':>10} "
        f"{'spd_UF':>8} {'spd_ADP':>8} {'sw%':>7}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['case_name']:<15} "
            f"{r['distance']:>2d} "
            f"{r['rounds']:>2d} "
            f"{r['p']:>7.4f} "
            f"{r['mwpm_error_rate']:>9.4f} "
            f"{r['uf_error_rate']:>9.4f} "
            f"{r['adaptive_error_rate']:>9.4f} "
            f"{r['mwpm_avg_decode_time_sec']:>10.6f} "
            f"{r['uf_avg_decode_time_sec']:>10.6f} "
            f"{r['adaptive_avg_decode_time_sec']:>10.6f} "
            f"{r['uf_speedup_vs_mwpm']:>8.3f} "
            f"{r['adaptive_speedup_vs_mwpm']:>8.3f} "
            f"{100.0 * r['adaptive_switch_rate']:>6.2f}%"
        )

    agg = report.get("aggregates", {})
    print("\n--- Aggregates ---")
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"{k:<36}: {v:.6f}")
        else:
            print(f"{k:<36}: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Week 2 Person 1 decoder comparison: MWPM vs UF vs Adaptive"
    )
    parser.add_argument("--shots", type=int, default=300, help="Shots por caso (default: 300)")
    parser.add_argument("--keep-soft", type=int, default=80, help="Muestras soft-info guardadas (default: 80)")
    parser.add_argument("--g-threshold", type=float, default=0.65, help="Umbral adaptativo (default: 0.65)")
    parser.add_argument(
        "--output",
        type=str,
        default="results/week2_person1_decoder_comparison.json",
        help="Ruta JSON de salida",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.shots <= 0:
        raise ValueError(f"--shots debe ser > 0. Recibido: {args.shots}")
    if args.keep_soft < 0:
        raise ValueError(f"--keep-soft debe ser >= 0. Recibido: {args.keep_soft}")
    if not (0.0 <= args.g_threshold <= 1.0):
        raise ValueError(f"--g-threshold debe estar en [0,1]. Recibido: {args.g_threshold}")

    report = run_comparison(
        cases=default_cases(),
        shots=args.shots,
        keep_soft=args.keep_soft,
        g_threshold=args.g_threshold,
    )
    saved = save_json(report, Path(args.output))
    print_table(report)
    print(f"\nJSON guardado en: {saved}")


if __name__ == "__main__":
    main()
