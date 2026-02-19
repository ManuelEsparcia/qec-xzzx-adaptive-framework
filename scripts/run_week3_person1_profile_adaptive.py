from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Tuple

# Ensure "src" import works when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.codes.xzzx_code import generate_xzzx_circuit  # noqa: E402
from src.decoders.mwpm_decoder import MWPMDecoderWithSoftInfo  # noqa: E402
from src.decoders.union_find_decoder import UnionFindDecoderWithSoftInfo  # noqa: E402
from src.switching.adaptive_decoder import AdaptiveConfig, AdaptiveDecoder  # noqa: E402


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _profile_one_mode(
    adaptive: AdaptiveDecoder,
    *,
    shots: int,
    keep_samples: int,
    g_threshold: float,
    seed: int,
    fast_mode: bool,
    top_n: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    adaptive.sampler = adaptive.circuit.compile_detector_sampler(seed=int(seed))

    prof = cProfile.Profile()
    t0 = perf_counter()
    prof.enable()
    bench = adaptive.benchmark_adaptive(
        shots=shots,
        g_threshold=g_threshold,
        keep_samples=keep_samples,
        compare_against_mwpm=False,
        fast_mode=fast_mode,
    )
    prof.disable()
    wall = float(perf_counter() - t0)

    st = pstats.Stats(prof)
    entries: List[Dict[str, Any]] = []
    for (filename, lineno, funcname), stat in st.stats.items():
        cc, nc, tt, ct, _callers = stat
        entries.append(
            {
                "file": str(filename),
                "line": int(lineno),
                "function": str(funcname),
                "primitive_calls": int(cc),
                "total_calls": int(nc),
                "tottime_sec": float(tt),
                "cumtime_sec": float(ct),
            }
        )

    entries.sort(key=lambda x: x["cumtime_sec"], reverse=True)
    top = entries[:top_n]

    summary = {
        "fast_mode": bool(fast_mode),
        "seed": int(seed),
        "shots": int(shots),
        "wall_time_sec": wall,
        "benchmark": {
            "error_rate_adaptive": float(bench["error_rate_adaptive"]),
            "avg_decode_time_adaptive": float(bench["avg_decode_time_adaptive"]),
            "switch_rate": float(bench["switch_rate"]),
            "num_detectors": int(bench["num_detectors"]),
            "num_observables": int(bench["num_observables"]),
        },
        "cprofile": {
            "total_calls": int(st.total_calls),
            "prim_calls": int(st.prim_calls),
            "total_tt_sec": float(st.total_tt),
        },
    }
    return summary, top


def save_json(data: Dict[str, Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return output


def print_summary(report: Dict[str, Any]) -> None:
    std = report["modes"]["standard"]
    fast = report["modes"]["fast"]

    print("\n=== Week 3 Person 1 - Adaptive Profile (cProfile) ===")
    print(f"Case: d={report['case']['distance']}, r={report['case']['rounds']}, p={report['case']['p']}")
    print(f"Shots: {report['config']['shots']} | g_threshold={report['config']['g_threshold']}")

    print("\n--- Mode summary ---")
    print(
        f"standard: wall={std['summary']['wall_time_sec']:.6f}s | "
        f"avg_decode={std['summary']['benchmark']['avg_decode_time_adaptive']:.6f}s | "
        f"switch={100.0*std['summary']['benchmark']['switch_rate']:.2f}%"
    )
    print(
        f"fast    : wall={fast['summary']['wall_time_sec']:.6f}s | "
        f"avg_decode={fast['summary']['benchmark']['avg_decode_time_adaptive']:.6f}s | "
        f"switch={100.0*fast['summary']['benchmark']['switch_rate']:.2f}%"
    )
    wall_spd = report["comparisons"]["wall_time_speedup_fast_vs_standard"]
    dec_spd = report["comparisons"]["avg_decode_speedup_fast_vs_standard"]
    print(f"speedup (wall fast/std): {wall_spd:.6f}")
    print(f"speedup (avg_decode fast/std): {dec_spd:.6f}")

    print("\n--- Top cumulative functions (standard) ---")
    for row in std["top_cumulative"]:
        print(
            f"{row['cumtime_sec']:.6f}s | {row['function']} | "
            f"{row['file']}:{row['line']}"
        )

    print("\n--- Top cumulative functions (fast) ---")
    for row in fast["top_cumulative"]:
        print(
            f"{row['cumtime_sec']:.6f}s | {row['function']} | "
            f"{row['file']}:{row['line']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week 3 Person 1: cProfile analysis for AdaptiveDecoder benchmark."
    )
    parser.add_argument("--distance", type=int, default=7, help="Code distance.")
    parser.add_argument("--rounds", type=int, default=5, help="Rounds.")
    parser.add_argument("--p", type=float, default=0.01, help="Physical noise rate.")
    parser.add_argument("--logical-basis", type=str, default="x", choices=["x", "z"], help="Logical basis.")
    parser.add_argument("--noise-model", type=str, default="depolarizing", help="Noise model.")
    parser.add_argument("--shots", type=int, default=500, help="Shots per mode.")
    parser.add_argument("--keep-samples", type=int, default=20, help="Samples kept from benchmark output.")
    parser.add_argument("--g-threshold", type=float, default=0.35, help="Adaptive threshold.")
    parser.add_argument("--seed", type=int, default=2026, help="Base seed.")
    parser.add_argument("--top-n", type=int, default=15, help="Top cProfile rows by cumulative time.")
    parser.add_argument(
        "--output",
        type=str,
        default="results/week3_person1_adaptive_profile.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.distance < 3 or args.distance % 2 == 0:
        raise ValueError("--distance must be odd and >= 3")
    if args.rounds <= 0:
        raise ValueError("--rounds must be > 0")
    if not (0.0 <= args.p <= 1.0):
        raise ValueError("--p must be in [0,1]")
    if args.shots <= 0:
        raise ValueError("--shots must be > 0")
    if args.keep_samples < 0:
        raise ValueError("--keep-samples must be >= 0")
    if not (0.0 <= args.g_threshold <= 1.0):
        raise ValueError("--g-threshold must be in [0,1]")
    if args.top_n <= 0:
        raise ValueError("--top-n must be > 0")

    circuit = generate_xzzx_circuit(
        distance=args.distance,
        rounds=args.rounds,
        noise_model=args.noise_model,
        p=args.p,
        logical_basis=args.logical_basis,
    )
    mwpm = MWPMDecoderWithSoftInfo(circuit)
    uf = UnionFindDecoderWithSoftInfo(circuit, prefer_union_find=True)
    adaptive = AdaptiveDecoder(
        circuit=circuit,
        fast_decoder=uf,
        accurate_decoder=mwpm,
        config=AdaptiveConfig(
            g_threshold=args.g_threshold,
            compare_against_mwpm_in_benchmark=False,
        ),
    )

    std_summary, std_top = _profile_one_mode(
        adaptive,
        shots=args.shots,
        keep_samples=args.keep_samples,
        g_threshold=args.g_threshold,
        seed=args.seed + 11,
        fast_mode=False,
        top_n=args.top_n,
    )
    fast_summary, fast_top = _profile_one_mode(
        adaptive,
        shots=args.shots,
        keep_samples=args.keep_samples,
        g_threshold=args.g_threshold,
        seed=args.seed + 11,
        fast_mode=True,
        top_n=args.top_n,
    )

    std_t = float(std_summary["benchmark"]["avg_decode_time_adaptive"])
    fast_t = float(fast_summary["benchmark"]["avg_decode_time_adaptive"])
    std_wall = float(std_summary["wall_time_sec"])
    fast_wall = float(fast_summary["wall_time_sec"])

    report = {
        "metadata": {
            "report_name": "week3_person1_adaptive_profile",
            "timestamp_utc": utc_now_iso(),
        },
        "case": {
            "distance": int(args.distance),
            "rounds": int(args.rounds),
            "p": float(args.p),
            "logical_basis": args.logical_basis,
            "noise_model": args.noise_model,
        },
        "config": {
            "shots": int(args.shots),
            "keep_samples": int(args.keep_samples),
            "g_threshold": float(args.g_threshold),
            "seed": int(args.seed),
            "top_n": int(args.top_n),
        },
        "modes": {
            "standard": {
                "summary": std_summary,
                "top_cumulative": std_top,
            },
            "fast": {
                "summary": fast_summary,
                "top_cumulative": fast_top,
            },
        },
        "comparisons": {
            "avg_decode_speedup_fast_vs_standard": float(std_t / fast_t) if fast_t > 0 else float("nan"),
            "wall_time_speedup_fast_vs_standard": float(std_wall / fast_wall)
            if fast_wall > 0
            else float("nan"),
            "delta_error_rate_fast_minus_standard": float(
                fast_summary["benchmark"]["error_rate_adaptive"]
                - std_summary["benchmark"]["error_rate_adaptive"]
            ),
            "delta_switch_rate_fast_minus_standard": float(
                fast_summary["benchmark"]["switch_rate"]
                - std_summary["benchmark"]["switch_rate"]
            ),
        },
    }

    saved = save_json(report, Path(args.output))
    print_summary(report)
    print(f"\nJSON saved at: {saved}")


if __name__ == "__main__":
    main()
