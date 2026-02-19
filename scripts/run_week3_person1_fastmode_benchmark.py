from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

# Ensure "src" import works when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.codes.xzzx_code import generate_xzzx_circuit  # noqa: E402
from src.decoders.mwpm_decoder import MWPMDecoderWithSoftInfo  # noqa: E402
from src.decoders.union_find_decoder import UnionFindDecoderWithSoftInfo  # noqa: E402
from src.switching.adaptive_decoder import AdaptiveConfig, AdaptiveDecoder  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    distance: int
    rounds: int
    p: float
    noise_model: Any = "depolarizing"
    logical_basis: str = "x"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_cases() -> List[Case]:
    return [
        Case(name="d3_r2_p0.005", distance=3, rounds=2, p=0.005),
        Case(name="d3_r3_p0.010", distance=3, rounds=3, p=0.010),
        Case(name="d5_r3_p0.010", distance=5, rounds=3, p=0.010),
    ]


def run_mode(
    adaptive: AdaptiveDecoder,
    *,
    shots: int,
    keep_samples: int,
    g_threshold: float,
    sampler_seed: int,
    fast_mode: bool,
) -> Dict[str, Any]:
    adaptive.sampler = adaptive.circuit.compile_detector_sampler(seed=int(sampler_seed))
    return adaptive.benchmark_adaptive(
        shots=shots,
        g_threshold=g_threshold,
        keep_samples=keep_samples,
        compare_against_mwpm=False,
        fast_mode=fast_mode,
    )


def run_one_case(
    case: Case,
    *,
    shots: int,
    keep_samples: int,
    g_threshold: float,
    repeats: int,
    base_seed: int,
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
            compare_against_mwpm_in_benchmark=False,
        ),
    )

    reps: List[Dict[str, Any]] = []
    for i in range(repeats):
        sampler_seed = int(base_seed + i * 10007 + case.distance * 997 + case.rounds * 97)
        standard = run_mode(
            adaptive,
            shots=shots,
            keep_samples=keep_samples,
            g_threshold=g_threshold,
            sampler_seed=sampler_seed,
            fast_mode=False,
        )
        fast = run_mode(
            adaptive,
            shots=shots,
            keep_samples=keep_samples,
            g_threshold=g_threshold,
            sampler_seed=sampler_seed,
            fast_mode=True,
        )
        t_std = float(standard["avg_decode_time_adaptive"])
        t_fast = float(fast["avg_decode_time_adaptive"])
        speedup = float(t_std / t_fast) if t_fast > 0 else float("nan")
        reps.append(
            {
                "repeat_index": i,
                "sampler_seed": sampler_seed,
                "standard": {
                    "error_rate": float(standard["error_rate_adaptive"]),
                    "avg_decode_time_sec": t_std,
                    "switch_rate": float(standard["switch_rate"]),
                    "fast_mode": bool(standard.get("fast_mode", False)),
                },
                "fast": {
                    "error_rate": float(fast["error_rate_adaptive"]),
                    "avg_decode_time_sec": t_fast,
                    "switch_rate": float(fast["switch_rate"]),
                    "fast_mode": bool(fast.get("fast_mode", True)),
                },
                "time_speedup_fast_vs_standard": speedup,
                "delta_error_rate_fast_minus_standard": float(
                    fast["error_rate_adaptive"] - standard["error_rate_adaptive"]
                ),
                "delta_switch_rate_fast_minus_standard": float(
                    fast["switch_rate"] - standard["switch_rate"]
                ),
            }
        )

    mean_std_t = mean(r["standard"]["avg_decode_time_sec"] for r in reps)
    mean_fast_t = mean(r["fast"]["avg_decode_time_sec"] for r in reps)
    mean_std_er = mean(r["standard"]["error_rate"] for r in reps)
    mean_fast_er = mean(r["fast"]["error_rate"] for r in reps)
    mean_std_sw = mean(r["standard"]["switch_rate"] for r in reps)
    mean_fast_sw = mean(r["fast"]["switch_rate"] for r in reps)

    return {
        "case_name": case.name,
        "distance": case.distance,
        "rounds": case.rounds,
        "p": case.p,
        "shots": shots,
        "repeats": repeats,
        "g_threshold": g_threshold,
        "num_detectors": int(getattr(circuit, "num_detectors", 0)),
        "num_observables": int(getattr(circuit, "num_observables", 0)),
        "means": {
            "standard_error_rate": float(mean_std_er),
            "fast_error_rate": float(mean_fast_er),
            "standard_avg_decode_time_sec": float(mean_std_t),
            "fast_avg_decode_time_sec": float(mean_fast_t),
            "standard_switch_rate": float(mean_std_sw),
            "fast_switch_rate": float(mean_fast_sw),
            "time_speedup_fast_vs_standard": float(mean_std_t / mean_fast_t)
            if mean_fast_t > 0
            else float("nan"),
            "delta_error_rate_fast_minus_standard": float(mean_fast_er - mean_std_er),
            "delta_switch_rate_fast_minus_standard": float(mean_fast_sw - mean_std_sw),
        },
        "repeats_summary": reps,
        "status": "ok",
    }


def aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    return {
        "mean_standard_error_rate": mean(r["means"]["standard_error_rate"] for r in rows),
        "mean_fast_error_rate": mean(r["means"]["fast_error_rate"] for r in rows),
        "mean_standard_avg_decode_time_sec": mean(
            r["means"]["standard_avg_decode_time_sec"] for r in rows
        ),
        "mean_fast_avg_decode_time_sec": mean(r["means"]["fast_avg_decode_time_sec"] for r in rows),
        "mean_standard_switch_rate": mean(r["means"]["standard_switch_rate"] for r in rows),
        "mean_fast_switch_rate": mean(r["means"]["fast_switch_rate"] for r in rows),
        "mean_time_speedup_fast_vs_standard": mean(
            r["means"]["time_speedup_fast_vs_standard"] for r in rows
        ),
        "mean_delta_error_rate_fast_minus_standard": mean(
            r["means"]["delta_error_rate_fast_minus_standard"] for r in rows
        ),
        "mean_delta_switch_rate_fast_minus_standard": mean(
            r["means"]["delta_switch_rate_fast_minus_standard"] for r in rows
        ),
    }


def save_json(data: Dict[str, Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return output


def print_summary(report: Dict[str, Any]) -> None:
    rows = report.get("cases_summary", [])
    print("\n=== Week 3 Person 1 - Adaptive fast_mode Benchmark ===")
    if not rows:
        print("No cases to show.")
        return

    header = (
        f"{'CASE':<15} {'d':>2} {'r':>2} {'p':>7} "
        f"{'ER_STD':>9} {'ER_FAST':>9} {'dER':>9} "
        f"{'t_STD':>10} {'t_FAST':>10} {'spd':>8} "
        f"{'sw_STD':>8} {'sw_FAST':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        m = r["means"]
        print(
            f"{r['case_name']:<15} "
            f"{r['distance']:>2d} "
            f"{r['rounds']:>2d} "
            f"{r['p']:>7.4f} "
            f"{m['standard_error_rate']:>9.4f} "
            f"{m['fast_error_rate']:>9.4f} "
            f"{m['delta_error_rate_fast_minus_standard']:>9.4f} "
            f"{m['standard_avg_decode_time_sec']:>10.6f} "
            f"{m['fast_avg_decode_time_sec']:>10.6f} "
            f"{m['time_speedup_fast_vs_standard']:>8.3f} "
            f"{100.0 * m['standard_switch_rate']:>7.2f}% "
            f"{100.0 * m['fast_switch_rate']:>7.2f}%"
        )

    agg = report.get("aggregates", {})
    print("\n--- Aggregates ---")
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"{k:<45}: {v:.6f}")
        else:
            print(f"{k:<45}: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week 3 Person 1 benchmark: compare Adaptive fast_mode=False vs fast_mode=True."
    )
    parser.add_argument("--shots", type=int, default=400, help="Shots per case and repeat.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of paired repeats per case.")
    parser.add_argument("--seed", type=int, default=2026, help="Base seed for reproducibility.")
    parser.add_argument("--keep-samples", type=int, default=40, help="Samples to keep in each mode run.")
    parser.add_argument("--g-threshold", type=float, default=0.35, help="Adaptive threshold.")
    parser.add_argument(
        "--output",
        type=str,
        default="results/week3_person1_fastmode_benchmark.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shots <= 0:
        raise ValueError("--shots must be > 0")
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")
    if args.keep_samples < 0:
        raise ValueError("--keep-samples must be >= 0")
    if not (0.0 <= args.g_threshold <= 1.0):
        raise ValueError("--g-threshold must be in [0,1]")

    rows = [
        run_one_case(
            c,
            shots=args.shots,
            keep_samples=args.keep_samples,
            g_threshold=args.g_threshold,
            repeats=args.repeats,
            base_seed=args.seed + i * 1000003,
        )
        for i, c in enumerate(default_cases())
    ]

    report = {
        "metadata": {
            "report_name": "week3_person1_fastmode_benchmark",
            "timestamp_utc": utc_now_iso(),
            "shots": int(args.shots),
            "repeats": int(args.repeats),
            "seed": int(args.seed),
            "g_threshold": float(args.g_threshold),
            "num_cases": len(rows),
            "keep_samples": int(args.keep_samples),
        },
        "cases_summary": rows,
        "aggregates": aggregate_rows(rows),
    }

    saved = save_json(report, Path(args.output))
    print_summary(report)
    print(f"\nJSON saved at: {saved}")


if __name__ == "__main__":
    main()
