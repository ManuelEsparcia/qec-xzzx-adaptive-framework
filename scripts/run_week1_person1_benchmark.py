# scripts/run_week1_person1_benchmark.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

# Ensure "src" import works when running this script directly.
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.week1_person1_pipeline import (  # noqa: E402
    Week1Person1PipelineConfig,
    run_week1_person1_pipeline,
)


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    distance: int
    rounds: int
    p: float
    noise_model: Any = "depolarizing"
    logical_basis: str = "x"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_cases() -> List[BenchmarkCase]:
    """
    Recommended base cases for Week 1 evidence.
    """
    return [
        BenchmarkCase(name="case_d3_r2_p0.005", distance=3, rounds=2, p=0.005),
        BenchmarkCase(name="case_d3_r3_p0.01", distance=3, rounds=3, p=0.01),
        BenchmarkCase(name="case_d5_r3_p0.01", distance=5, rounds=3, p=0.01),
    ]


def run_cases(
    cases: List[BenchmarkCase],
    shots: int,
    ref_shots: int,
    keep_soft_info_samples: int,
) -> Dict[str, Any]:
    summary_rows: List[Dict[str, Any]] = []
    full_results: List[Dict[str, Any]] = []

    for c in cases:
        cfg = Week1Person1PipelineConfig(
            distance=c.distance,
            rounds=c.rounds,
            noise_model=c.noise_model,
            p=c.p,
            logical_basis=c.logical_basis,
            shots=shots,
            keep_soft_info_samples=keep_soft_info_samples,
            reference_ler_shots=ref_shots,
        )

        result = run_week1_person1_pipeline(cfg)
        full_results.append(result)

        bench = result["benchmark"]
        ref = result["reference"]
        csum = result["circuit_summary"]

        bench_er = float(bench["error_rate"])
        ref_er = float(ref["ler_mwpm_helper"])
        delta = abs(bench_er - ref_er)

        row = {
            "case_name": c.name,
            "distance": c.distance,
            "rounds": c.rounds,
            "p": c.p,
            "logical_basis": c.logical_basis,
            "shots": int(bench["shots"]),
            "num_qubits": int(csum["num_qubits"]),
            "num_detectors": int(csum["num_detectors"]),
            "num_observables": int(csum["num_observables"]),
            "benchmark_error_rate": bench_er,
            "reference_ler_helper": ref_er,
            "abs_delta_benchmark_vs_reference": delta,
            "avg_decode_time_sec": float(bench["avg_decode_time"]),
            "soft_info_samples_kept": len(bench["soft_info_samples"]),
            "status": result.get("status", "unknown"),
        }
        summary_rows.append(row)

    # Global aggregates
    mean_bench_er = mean([r["benchmark_error_rate"] for r in summary_rows]) if summary_rows else float("nan")
    mean_ref_er = mean([r["reference_ler_helper"] for r in summary_rows]) if summary_rows else float("nan")
    mean_delta = mean([r["abs_delta_benchmark_vs_reference"] for r in summary_rows]) if summary_rows else float("nan")
    mean_decode_time = mean([r["avg_decode_time_sec"] for r in summary_rows]) if summary_rows else float("nan")

    report: Dict[str, Any] = {
        "metadata": {
            "report_name": "week1_person1_baseline",
            "timestamp_utc": utc_now_iso(),
            "num_cases": len(cases),
            "shots_per_case": shots,
            "reference_shots_per_case": ref_shots,
            "keep_soft_info_samples": keep_soft_info_samples,
        },
        "cases_summary": summary_rows,
        "aggregates": {
            "mean_benchmark_error_rate": mean_bench_er,
            "mean_reference_ler_helper": mean_ref_er,
            "mean_abs_delta_benchmark_vs_reference": mean_delta,
            "mean_avg_decode_time_sec": mean_decode_time,
        },
        # Store full results for Week 1 traceability:
        "full_results": full_results,
    }
    return report


def save_json(data: Dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return output_path


def print_summary_table(report: Dict[str, Any]) -> None:
    rows = report.get("cases_summary", [])
    print("\n=== Week 1 Person 1 - Baseline Benchmark ===")
    if not rows:
        print("No cases to show.")
        return

    header = (
        f"{'CASE':<20} {'d':>2} {'r':>2} {'p':>8} "
        f"{'bench_ER':>10} {'ref_LER':>10} {'|Δ|':>10} {'avg_t(s)':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['case_name']:<20} "
            f"{r['distance']:>2d} "
            f"{r['rounds']:>2d} "
            f"{r['p']:>8.4f} "
            f"{r['benchmark_error_rate']:>10.6f} "
            f"{r['reference_ler_helper']:>10.6f} "
            f"{r['abs_delta_benchmark_vs_reference']:>10.6f} "
            f"{r['avg_decode_time_sec']:>10.6f}"
        )

    agg = report.get("aggregates", {})
    print("\n--- Aggregates ---")
    print(f"mean_benchmark_error_rate           : {agg.get('mean_benchmark_error_rate', float('nan')):.6f}")
    print(f"mean_reference_ler_helper           : {agg.get('mean_reference_ler_helper', float('nan')):.6f}")
    print(f"mean_abs_delta_benchmark_vs_ref     : {agg.get('mean_abs_delta_benchmark_vs_reference', float('nan')):.6f}")
    print(f"mean_avg_decode_time_sec            : {agg.get('mean_avg_decode_time_sec', float('nan')):.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Week 1 Person 1 baseline benchmark (XZZX + MWPM soft-info)."
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=300,
        help="Shots per case for benchmark (default: 300).",
    )
    parser.add_argument(
        "--ref-shots",
        type=int,
        default=300,
        help="Shots per case for reference helper LER (default: 300).",
    )
    parser.add_argument(
        "--keep-soft",
        type=int,
        default=100,
        help="Number of soft_info_samples to keep per case (default: 100).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/week1_person1_baseline.json",
        help="Output JSON path (default: results/week1_person1_baseline.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.shots <= 0:
        raise ValueError(f"--shots must be > 0. Received: {args.shots}")
    if args.ref_shots <= 0:
        raise ValueError(f"--ref-shots must be > 0. Received: {args.ref_shots}")
    if args.keep_soft < 0:
        raise ValueError(f"--keep-soft must be >= 0. Received: {args.keep_soft}")

    cases = default_cases()
    report = run_cases(
        cases=cases,
        shots=args.shots,
        ref_shots=args.ref_shots,
        keep_soft_info_samples=args.keep_soft,
    )

    output_path = Path(args.output)
    saved = save_json(report, output_path)

    print_summary_table(report)
    print(f"\nJSON saved at: {saved}")


if __name__ == "__main__":
    main()
