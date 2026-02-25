from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
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
        Case(name="d5_r3_p0.010", distance=5, rounds=3, p=0.010),
        Case(name="d7_r5_p0.010", distance=7, rounds=5, p=0.010),
        Case(name="d9_r7_p0.010", distance=9, rounds=7, p=0.010),
    ]


def _run_mwpm(mwpm: MWPMDecoderWithSoftInfo, *, shots: int, keep_soft: int, seed: int) -> Dict[str, Any]:
    mwpm.sampler = mwpm.circuit.compile_detector_sampler(seed=int(seed))
    t0 = perf_counter()
    res = mwpm.benchmark(shots=shots, keep_soft_info_samples=keep_soft)
    wall = float(perf_counter() - t0)
    return {
        "error_rate": float(res["error_rate"]),
        "avg_decode_time_sec": float(res["avg_decode_time"]),
        "wall_time_sec": wall,
    }


def _run_uf(
    uf: UnionFindDecoderWithSoftInfo,
    *,
    shots: int,
    keep_soft: int,
    seed: int,
) -> Dict[str, Any]:
    uf.sampler = uf.circuit.compile_detector_sampler(seed=int(seed))
    t0 = perf_counter()
    res = uf.benchmark(shots=shots, keep_soft_info_samples=keep_soft)
    wall = float(perf_counter() - t0)
    return {
        "error_rate": float(res["error_rate"]),
        "avg_decode_time_sec": float(res["avg_decode_time"]),
        "wall_time_sec": wall,
        "backend_info": res.get("backend_info", {}),
    }


def _run_adaptive(
    adaptive: AdaptiveDecoder,
    *,
    shots: int,
    keep_samples: int,
    g_threshold: float,
    seed: int,
    fast_mode: bool,
) -> Dict[str, Any]:
    adaptive.sampler = adaptive.circuit.compile_detector_sampler(seed=int(seed))
    t0 = perf_counter()
    res = adaptive.benchmark_adaptive(
        shots=shots,
        g_threshold=g_threshold,
        keep_samples=keep_samples,
        compare_against_mwpm=False,
        fast_mode=fast_mode,
    )
    wall = float(perf_counter() - t0)
    return {
        "error_rate": float(res["error_rate_adaptive"]),
        "avg_decode_time_sec": float(res["avg_decode_time_adaptive"]),
        "switch_rate": float(res["switch_rate"]),
        "wall_time_sec": wall,
        "fast_mode": bool(res.get("fast_mode", fast_mode)),
    }


def run_one_case(
    case: Case,
    *,
    shots: int,
    keep_soft: int,
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
        seed_i = int(base_seed + i * 10007 + case.distance * 997 + case.rounds * 97)
        run_mwpm = _run_mwpm(mwpm, shots=shots, keep_soft=keep_soft, seed=seed_i + 1)
        run_uf = _run_uf(uf, shots=shots, keep_soft=keep_soft, seed=seed_i + 2)
        run_adp_std = _run_adaptive(
            adaptive,
            shots=shots,
            keep_samples=keep_soft,
            g_threshold=g_threshold,
            seed=seed_i + 3,
            fast_mode=False,
        )
        run_adp_fast = _run_adaptive(
            adaptive,
            shots=shots,
            keep_samples=keep_soft,
            g_threshold=g_threshold,
            seed=seed_i + 3,
            fast_mode=True,
        )

        reps.append(
            {
                "repeat_index": i,
                "seed_base": seed_i,
                "mwpm": run_mwpm,
                "uf": run_uf,
                "adaptive_standard": run_adp_std,
                "adaptive_fast": run_adp_fast,
                "speedup_fast_vs_standard_decode_time": float(
                    run_adp_std["avg_decode_time_sec"] / run_adp_fast["avg_decode_time_sec"]
                )
                if run_adp_fast["avg_decode_time_sec"] > 0
                else float("nan"),
                "speedup_fast_vs_standard_wall_time": float(
                    run_adp_std["wall_time_sec"] / run_adp_fast["wall_time_sec"]
                )
                if run_adp_fast["wall_time_sec"] > 0
                else float("nan"),
            }
        )

    mean_mwpm_t = mean(r["mwpm"]["avg_decode_time_sec"] for r in reps)
    mean_uf_t = mean(r["uf"]["avg_decode_time_sec"] for r in reps)
    mean_std_t = mean(r["adaptive_standard"]["avg_decode_time_sec"] for r in reps)
    mean_fast_t = mean(r["adaptive_fast"]["avg_decode_time_sec"] for r in reps)
    mean_std_wall = mean(r["adaptive_standard"]["wall_time_sec"] for r in reps)
    mean_fast_wall = mean(r["adaptive_fast"]["wall_time_sec"] for r in reps)

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
            "mwpm_error_rate": float(mean(r["mwpm"]["error_rate"] for r in reps)),
            "uf_error_rate": float(mean(r["uf"]["error_rate"] for r in reps)),
            "adaptive_standard_error_rate": float(mean(r["adaptive_standard"]["error_rate"] for r in reps)),
            "adaptive_fast_error_rate": float(mean(r["adaptive_fast"]["error_rate"] for r in reps)),
            "mwpm_avg_decode_time_sec": float(mean_mwpm_t),
            "uf_avg_decode_time_sec": float(mean_uf_t),
            "adaptive_standard_avg_decode_time_sec": float(mean_std_t),
            "adaptive_fast_avg_decode_time_sec": float(mean_fast_t),
            "adaptive_standard_wall_time_sec": float(mean_std_wall),
            "adaptive_fast_wall_time_sec": float(mean_fast_wall),
            "adaptive_standard_switch_rate": float(mean(r["adaptive_standard"]["switch_rate"] for r in reps)),
            "adaptive_fast_switch_rate": float(mean(r["adaptive_fast"]["switch_rate"] for r in reps)),
            "speedup_fast_vs_standard_decode_time": float(mean_std_t / mean_fast_t)
            if mean_fast_t > 0
            else float("nan"),
            "speedup_fast_vs_standard_wall_time": float(mean_std_wall / mean_fast_wall)
            if mean_fast_wall > 0
            else float("nan"),
            "speedup_uf_vs_mwpm_decode_time": float(mean_mwpm_t / mean_uf_t)
            if mean_uf_t > 0
            else float("nan"),
            "speedup_adaptive_fast_vs_mwpm_decode_time": float(mean_mwpm_t / mean_fast_t)
            if mean_fast_t > 0
            else float("nan"),
        },
        "repeats_summary": reps,
        "status": "ok",
    }


def aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    return {
        "mean_mwpm_error_rate": mean(r["means"]["mwpm_error_rate"] for r in rows),
        "mean_uf_error_rate": mean(r["means"]["uf_error_rate"] for r in rows),
        "mean_adaptive_standard_error_rate": mean(
            r["means"]["adaptive_standard_error_rate"] for r in rows
        ),
        "mean_adaptive_fast_error_rate": mean(r["means"]["adaptive_fast_error_rate"] for r in rows),
        "mean_mwpm_avg_decode_time_sec": mean(r["means"]["mwpm_avg_decode_time_sec"] for r in rows),
        "mean_uf_avg_decode_time_sec": mean(r["means"]["uf_avg_decode_time_sec"] for r in rows),
        "mean_adaptive_standard_avg_decode_time_sec": mean(
            r["means"]["adaptive_standard_avg_decode_time_sec"] for r in rows
        ),
        "mean_adaptive_fast_avg_decode_time_sec": mean(
            r["means"]["adaptive_fast_avg_decode_time_sec"] for r in rows
        ),
        "mean_adaptive_standard_wall_time_sec": mean(
            r["means"]["adaptive_standard_wall_time_sec"] for r in rows
        ),
        "mean_adaptive_fast_wall_time_sec": mean(
            r["means"]["adaptive_fast_wall_time_sec"] for r in rows
        ),
        "mean_speedup_fast_vs_standard_decode_time": mean(
            r["means"]["speedup_fast_vs_standard_decode_time"] for r in rows
        ),
        "mean_speedup_fast_vs_standard_wall_time": mean(
            r["means"]["speedup_fast_vs_standard_wall_time"] for r in rows
        ),
    }


def save_json(data: Dict[str, Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return output


def print_summary(report: Dict[str, Any]) -> None:
    rows = report.get("cases_summary", [])
    print("\n=== Week 3 Person 1 - Scaling Benchmark ===")
    if not rows:
        print("No cases to show.")
        return

    header = (
        f"{'CASE':<15} {'d':>2} {'r':>2} "
        f"{'t_MWPM':>10} {'t_UF':>10} {'t_ADP':>10} {'t_ADP_F':>10} "
        f"{'spd_F/ADP':>10} {'spd_Fwall':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        m = r["means"]
        print(
            f"{r['case_name']:<15} {r['distance']:>2d} {r['rounds']:>2d} "
            f"{m['mwpm_avg_decode_time_sec']:>10.6f} "
            f"{m['uf_avg_decode_time_sec']:>10.6f} "
            f"{m['adaptive_standard_avg_decode_time_sec']:>10.6f} "
            f"{m['adaptive_fast_avg_decode_time_sec']:>10.6f} "
            f"{m['speedup_fast_vs_standard_decode_time']:>10.3f} "
            f"{m['speedup_fast_vs_standard_wall_time']:>10.3f}"
        )

    print("\n--- Aggregates ---")
    for k, v in report.get("aggregates", {}).items():
        if isinstance(v, float):
            print(f"{k:<45}: {v:.6f}")
        else:
            print(f"{k:<45}: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week 3 Person 1 scaling benchmark on larger distances/rounds."
    )
    parser.add_argument("--shots", type=int, default=300, help="Shots per case per repeat.")
    parser.add_argument("--repeats", type=int, default=2, help="Paired repeats per case.")
    parser.add_argument("--keep-soft", type=int, default=20, help="Soft sample cap for decoder benchmarks.")
    parser.add_argument("--seed", type=int, default=2026, help="Base seed.")
    parser.add_argument("--g-threshold", type=float, default=0.35, help="Adaptive threshold.")
    parser.add_argument(
        "--output",
        type=str,
        default="results/week3_person1_scaling_benchmark.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shots <= 0:
        raise ValueError("--shots must be > 0")
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")
    if args.keep_soft < 0:
        raise ValueError("--keep-soft must be >= 0")
    if not (0.0 <= args.g_threshold <= 1.0):
        raise ValueError("--g-threshold must be in [0,1]")

    rows = [
        run_one_case(
            c,
            shots=args.shots,
            keep_soft=args.keep_soft,
            g_threshold=args.g_threshold,
            repeats=args.repeats,
            base_seed=args.seed + i * 1000003,
        )
        for i, c in enumerate(default_cases())
    ]

    report = {
        "metadata": {
            "report_name": "week3_person1_scaling_benchmark",
            "timestamp_utc": utc_now_iso(),
            "shots": int(args.shots),
            "repeats": int(args.repeats),
            "seed": int(args.seed),
            "g_threshold": float(args.g_threshold),
            "keep_soft": int(args.keep_soft),
            "num_cases": len(rows),
        },
        "cases_summary": rows,
        "aggregates": aggregate_rows(rows),
    }
    saved = save_json(report, Path(args.output))
    print_summary(report)
    print(f"\nJSON saved at: {saved}")


if __name__ == "__main__":
    main()
