from __future__ import annotations

import argparse
import cProfile
import gc
import json
import math
import pstats
import sys
import tracemalloc
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Ensure "src" import works when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.codes.xzzx_code import generate_xzzx_circuit  # noqa: E402
from src.decoders.belief_matching_decoder import BeliefMatchingDecoderWithSoftInfo  # noqa: E402
from src.decoders.mwpm_decoder import MWPMDecoderWithSoftInfo  # noqa: E402
from src.decoders.union_find_decoder import UnionFindDecoderWithSoftInfo  # noqa: E402
from src.noise.noise_models import apply_noise_model  # noqa: E402
from src.switching.adaptive_decoder import AdaptiveConfig, AdaptiveDecoder  # noqa: E402


DECODERS_ALLOWED = {"mwpm", "uf", "bm", "adaptive"}
NOISE_ALLOWED = {
    "depolarizing",
    "biased_eta10",
    "biased_eta100",
    "biased_eta500",
    "circuit_level",
    "correlated",
}


@dataclass(frozen=True)
class Case:
    distance: int
    rounds: int
    p: float
    noise_model: str
    logical_basis: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_csv_ints(raw: str) -> List[int]:
    vals: List[int] = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        vals.append(int(t))
    if not vals:
        raise ValueError("empty integer list")
    return vals


def parse_decoders_csv(raw: str) -> List[str]:
    vals = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("empty decoder list")
    bad = [x for x in vals if x not in DECODERS_ALLOWED]
    if bad:
        raise ValueError(f"invalid decoders: {bad}; allowed: {sorted(DECODERS_ALLOWED)}")
    out: List[str] = []
    seen = set()
    for x in vals:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def noise_spec_from_name(noise_name: str, p_phys: float) -> Dict[str, Any]:
    if noise_name == "depolarizing":
        return {"type": "depolarizing", "p": p_phys}
    if noise_name == "biased_eta10":
        return {"type": "biased", "p": p_phys, "eta": 10.0}
    if noise_name == "biased_eta100":
        return {"type": "biased", "p": p_phys, "eta": 100.0}
    if noise_name == "biased_eta500":
        return {"type": "biased", "p": p_phys, "eta": 500.0}
    if noise_name == "circuit_level":
        return {"type": "circuit_level", "p": p_phys}
    if noise_name == "correlated":
        return {"type": "correlated", "p": p_phys, "correlation_strength": 0.35}
    raise ValueError(f"unknown noise model: {noise_name}")


def _summary(values: Sequence[float], *, clamp_01: bool = False) -> Dict[str, float]:
    if not values:
        raise ValueError("values must be non-empty")
    vals = [float(v) for v in values]
    n = len(vals)
    m = float(mean(vals))
    sd = float(stdev(vals)) if n > 1 else 0.0
    ci = float(1.96 * sd / math.sqrt(n)) if n > 1 else 0.0
    lo = float(m - ci)
    hi = float(m + ci)
    if clamp_01:
        lo = float(max(0.0, lo))
        hi = float(min(1.0, hi))
    return {
        "n": float(n),
        "mean": m,
        "std": sd,
        "ci95_half_width": ci,
        "ci95_low": lo,
        "ci95_high": hi,
    }


def _build_circuit(case: Case) -> Any:
    base = generate_xzzx_circuit(
        distance=case.distance,
        rounds=case.rounds,
        noise_model="none",
        p=0.0,
        logical_basis=case.logical_basis,
    )
    noisy = apply_noise_model(base, model=noise_spec_from_name(case.noise_model, case.p))
    return noisy


def _run_mwpm(circuit: Any, *, shots: int, seed: int) -> Dict[str, float]:
    dec = MWPMDecoderWithSoftInfo(circuit)
    dec.sampler = circuit.compile_detector_sampler(seed=int(seed))
    t0 = perf_counter()
    res = dec.benchmark(shots=int(shots), keep_soft_info_samples=0)
    wall = float(perf_counter() - t0)
    return {
        "error_rate": float(res["error_rate"]),
        "avg_decode_time_sec": float(res["avg_decode_time"]),
        "switch_rate": 0.0,
        "wall_time_sec": wall,
    }


def _run_uf(circuit: Any, *, shots: int, seed: int) -> Dict[str, float]:
    dec = UnionFindDecoderWithSoftInfo(circuit, prefer_union_find=True)
    dec.sampler = circuit.compile_detector_sampler(seed=int(seed))
    t0 = perf_counter()
    res = dec.benchmark(shots=int(shots), keep_soft_info_samples=0)
    wall = float(perf_counter() - t0)
    return {
        "error_rate": float(res["error_rate"]),
        "avg_decode_time_sec": float(res["avg_decode_time"]),
        "switch_rate": 0.0,
        "wall_time_sec": wall,
    }


def _run_bm(circuit: Any, *, shots: int, seed: int) -> Dict[str, float]:
    dec = BeliefMatchingDecoderWithSoftInfo(circuit, prefer_belief_propagation=True)
    dec.sampler = circuit.compile_detector_sampler(seed=int(seed))
    t0 = perf_counter()
    res = dec.benchmark(shots=int(shots), keep_soft_info_samples=0)
    wall = float(perf_counter() - t0)
    return {
        "error_rate": float(res["error_rate"]),
        "avg_decode_time_sec": float(res["avg_decode_time"]),
        "switch_rate": 0.0,
        "wall_time_sec": wall,
    }


def _run_adaptive(
    circuit: Any,
    *,
    shots: int,
    seed: int,
    g_threshold: float,
    fast_mode: bool,
) -> Dict[str, float]:
    mwpm = MWPMDecoderWithSoftInfo(circuit)
    uf = UnionFindDecoderWithSoftInfo(circuit, prefer_union_find=True)
    dec = AdaptiveDecoder(
        circuit=circuit,
        fast_decoder=uf,
        accurate_decoder=mwpm,
        config=AdaptiveConfig(
            g_threshold=float(g_threshold),
            compare_against_mwpm_in_benchmark=False,
        ),
    )
    dec.sampler = circuit.compile_detector_sampler(seed=int(seed))
    t0 = perf_counter()
    res = dec.benchmark_adaptive(
        shots=int(shots),
        g_threshold=float(g_threshold),
        keep_samples=0,
        compare_against_mwpm=False,
        fast_mode=bool(fast_mode),
    )
    wall = float(perf_counter() - t0)
    return {
        "error_rate": float(res["error_rate_adaptive"]),
        "avg_decode_time_sec": float(res["avg_decode_time_adaptive"]),
        "switch_rate": float(res["switch_rate"]),
        "wall_time_sec": wall,
    }


def _run_one(
    *,
    decoder: str,
    circuit: Any,
    shots: int,
    seed: int,
    g_threshold: float,
    adaptive_fast_mode: bool,
) -> Dict[str, float]:
    if decoder == "mwpm":
        return _run_mwpm(circuit, shots=shots, seed=seed)
    if decoder == "uf":
        return _run_uf(circuit, shots=shots, seed=seed)
    if decoder == "bm":
        return _run_bm(circuit, shots=shots, seed=seed)
    if decoder == "adaptive":
        return _run_adaptive(
            circuit,
            shots=shots,
            seed=seed,
            g_threshold=g_threshold,
            fast_mode=adaptive_fast_mode,
        )
    raise ValueError(f"unknown decoder: {decoder}")


def _run_one_with_memory(
    *,
    decoder: str,
    circuit: Any,
    shots: int,
    seed: int,
    g_threshold: float,
    adaptive_fast_mode: bool,
) -> Dict[str, float]:
    gc.collect()
    tracemalloc.start()
    try:
        out = _run_one(
            decoder=decoder,
            circuit=circuit,
            shots=shots,
            seed=seed,
            g_threshold=g_threshold,
            adaptive_fast_mode=adaptive_fast_mode,
        )
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    out["memory_peak_bytes"] = float(peak)
    out["memory_current_bytes"] = float(current)
    return out


def _fit_power_law(distances: Sequence[int], values: Sequence[float]) -> Dict[str, float]:
    if len(distances) != len(values) or not distances:
        raise ValueError("invalid fit inputs")
    xs = np.asarray(distances, dtype=float)
    ys = np.asarray(values, dtype=float)
    if np.any(xs <= 0.0):
        raise ValueError("distances must be > 0")
    if np.any(ys <= 0.0):
        raise ValueError("values must be > 0")
    if len(xs) == 1:
        a = float(ys[0])
        b = 0.0
        y_fit = np.asarray([a], dtype=float)
    else:
        log_x = np.log(xs)
        log_y = np.log(ys)
        b, log_a = np.polyfit(log_x, log_y, 1)
        a = float(math.exp(float(log_a)))
        y_fit = a * (xs ** float(b))
    ss_res = float(np.sum((ys - y_fit) ** 2))
    ss_tot = float(np.sum((ys - float(np.mean(ys))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return {"coefficient": float(a), "exponent": float(b), "r2": float(r2)}


def _build_scaling_models(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_decoder: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_decoder.setdefault(str(row["decoder"]), []).append(row)

    out: List[Dict[str, Any]] = []
    for decoder, arr in sorted(by_decoder.items()):
        arr_sorted = sorted(arr, key=lambda x: int(x["distance"]))
        ds = [int(x["distance"]) for x in arr_sorted]
        ts = [float(x["metrics"]["avg_decode_time_sec"]["mean"]) for x in arr_sorted]
        ms = [float(x["metrics"]["memory_peak_bytes"]["mean"]) for x in arr_sorted]
        t_fit = _fit_power_law(ds, ts)
        m_fit = _fit_power_law(ds, ms)
        out.append(
            {
                "decoder": decoder,
                "distances": ds,
                "time_fit": t_fit,
                "memory_fit": m_fit,
            }
        )
    return out


def _profile_hotspots(
    *,
    circuit: Any,
    decoder: str,
    shots: int,
    seed: int,
    g_threshold: float,
    adaptive_fast_mode: bool,
    top_n: int,
) -> Dict[str, Any]:
    prof = cProfile.Profile()
    prof.enable()
    t0 = perf_counter()
    bench = _run_one(
        decoder=decoder,
        circuit=circuit,
        shots=shots,
        seed=seed,
        g_threshold=g_threshold,
        adaptive_fast_mode=adaptive_fast_mode,
    )
    prof.disable()
    wall = float(perf_counter() - t0)

    st = pstats.Stats(prof)
    entries: List[Dict[str, Any]] = []
    for (filename, lineno, funcname), stat in st.stats.items():
        cc, nc, tt, ct, _ = stat
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
    top = entries[: int(top_n)]
    return {
        "decoder": decoder,
        "shots": int(shots),
        "seed": int(seed),
        "wall_time_sec": wall,
        "benchmark_summary": bench,
        "cprofile_summary": {
            "total_calls": int(st.total_calls),
            "prim_calls": int(st.prim_calls),
            "total_tt_sec": float(st.total_tt),
        },
        "top_cumulative": top,
    }


def run_case_decoder(
    *,
    case: Case,
    decoder: str,
    shots: int,
    repeats: int,
    base_seed: int,
    g_threshold: float,
    adaptive_fast_mode: bool,
) -> Dict[str, Any]:
    circuit = _build_circuit(case)
    run_rows: List[Dict[str, Any]] = []
    for i in range(repeats):
        seed_i = int(base_seed + i * 10007 + case.distance * 997 + case.rounds * 97)
        out = _run_one_with_memory(
            decoder=decoder,
            circuit=circuit,
            shots=shots,
            seed=seed_i,
            g_threshold=g_threshold,
            adaptive_fast_mode=adaptive_fast_mode,
        )
        run_rows.append({"repeat_index": int(i), "seed": int(seed_i), **out})

    metrics = {
        "error_rate": _summary([float(r["error_rate"]) for r in run_rows], clamp_01=True),
        "avg_decode_time_sec": _summary([float(r["avg_decode_time_sec"]) for r in run_rows]),
        "wall_time_sec": _summary([float(r["wall_time_sec"]) for r in run_rows]),
        "switch_rate": _summary([float(r["switch_rate"]) for r in run_rows], clamp_01=True),
        "memory_peak_bytes": _summary([float(r["memory_peak_bytes"]) for r in run_rows]),
        "memory_current_bytes": _summary([float(r["memory_current_bytes"]) for r in run_rows]),
    }

    return {
        "decoder": decoder,
        "distance": int(case.distance),
        "rounds": int(case.rounds),
        "p_phys": float(case.p),
        "noise_model": case.noise_model,
        "logical_basis": case.logical_basis,
        "shots": int(shots),
        "repeats": int(repeats),
        "base_seed": int(base_seed),
        "metrics": metrics,
        "repeat_runs": run_rows,
        "status": "ok",
    }


def save_json(data: Dict[str, Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return output


def print_summary(report: Dict[str, Any]) -> None:
    print("\n=== Week 5 Person 1 - Profiling + Scaling ===")
    print(
        f"distances={report['config']['distances']} | decoders={report['config']['decoders']} | "
        f"shots={report['config']['shots']} | repeats={report['config']['repeats']}"
    )

    print("\n--- Scaling fits (time and memory) ---")
    for m in report.get("scaling_models", []):
        tf = m["time_fit"]
        mf = m["memory_fit"]
        print(
            f"{m['decoder']:<8} | "
            f"time_exp={tf['exponent']:.3f} (r2={tf['r2']:.4f}) | "
            f"mem_exp={mf['exponent']:.3f} (r2={mf['r2']:.4f})"
        )

    hp = report.get("profile_hotspots", {})
    if hp:
        print("\n--- cProfile hotspot head ---")
        print(
            f"decoder={hp['decoder']} | distance={hp['distance']} | rounds={hp['rounds']} | "
            f"wall={hp['profile']['wall_time_sec']:.6f}s"
        )
        for row in hp["profile"]["top_cumulative"][:8]:
            print(
                f"{row['cumtime_sec']:.6f}s | {row['function']} | "
                f"{row['file']}:{row['line']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week 5 Block 3: profiling + memory + scaling benchmark."
    )
    parser.add_argument("--distances", type=str, default="5,7,9,11,13", help="CSV odd distances >= 3.")
    parser.add_argument("--rounds-mode", type=str, choices=["fixed", "distance"], default="distance", help="Use fixed rounds or rounds=distance.")
    parser.add_argument("--rounds", type=int, default=3, help="Rounds when --rounds-mode=fixed.")
    parser.add_argument("--decoders", type=str, default="mwpm,uf,bm,adaptive", help="CSV decoders to benchmark.")
    parser.add_argument("--noise-model", type=str, default="depolarizing", help="Noise model (week3 names).")
    parser.add_argument("--p-phys", type=float, default=0.01, help="Physical error rate.")
    parser.add_argument("--logical-basis", type=str, choices=["x", "z"], default="x", help="Logical basis.")
    parser.add_argument("--shots", type=int, default=200, help="Shots per run.")
    parser.add_argument("--repeats", type=int, default=2, help="Repeated runs per (decoder,distance).")
    parser.add_argument("--seed", type=int, default=2026, help="Base seed.")
    parser.add_argument("--g-threshold", type=float, default=0.35, help="Adaptive threshold.")
    parser.add_argument("--adaptive-fast-mode", action="store_true", help="Use fast_mode for adaptive benchmark.")
    parser.add_argument("--profile-decoder", type=str, default="adaptive", choices=["mwpm", "uf", "bm", "adaptive"], help="Decoder to cProfile.")
    parser.add_argument("--profile-distance", type=int, default=13, help="Distance used for cProfile section.")
    parser.add_argument("--profile-shots", type=int, default=120, help="Shots for cProfile section.")
    parser.add_argument("--profile-top-n", type=int, default=15, help="Top cProfile entries by cumulative time.")
    parser.add_argument(
        "--output",
        type=str,
        default="results/week5_person1_profile_scaling.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distances = parse_csv_ints(args.distances)
    decoders = parse_decoders_csv(args.decoders)

    if any(d < 3 or d % 2 == 0 for d in distances):
        raise ValueError("--distances must contain odd values >= 3")
    if args.rounds <= 0:
        raise ValueError("--rounds must be > 0")
    if args.noise_model not in NOISE_ALLOWED:
        raise ValueError(f"--noise-model must be one of {sorted(NOISE_ALLOWED)}")
    if not (0.0 <= float(args.p_phys) <= 1.0):
        raise ValueError("--p-phys must be in [0,1]")
    if args.shots <= 0:
        raise ValueError("--shots must be > 0")
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")
    if not (0.0 <= float(args.g_threshold) <= 1.0):
        raise ValueError("--g-threshold must be in [0,1]")
    if args.profile_distance < 3 or args.profile_distance % 2 == 0:
        raise ValueError("--profile-distance must be odd and >= 3")
    if args.profile_shots <= 0:
        raise ValueError("--profile-shots must be > 0")
    if args.profile_top_n <= 0:
        raise ValueError("--profile-top-n must be > 0")

    rows: List[Dict[str, Any]] = []
    total = len(decoders) * len(distances)
    idx = 0

    for dec in decoders:
        for d in distances:
            idx += 1
            rounds_i = int(d) if args.rounds_mode == "distance" else int(args.rounds)
            case = Case(
                distance=int(d),
                rounds=rounds_i,
                p=float(args.p_phys),
                noise_model=str(args.noise_model),
                logical_basis=args.logical_basis,
            )
            base_seed = int(args.seed + idx * 1000003 + d * 97)
            out = run_case_decoder(
                case=case,
                decoder=dec,
                shots=int(args.shots),
                repeats=int(args.repeats),
                base_seed=base_seed,
                g_threshold=float(args.g_threshold),
                adaptive_fast_mode=bool(args.adaptive_fast_mode),
            )
            rows.append(out)
            print(
                f"[{idx}/{total}] {dec:<8} | d={d:>2} | rounds={rounds_i:>2} | "
                f"ER={out['metrics']['error_rate']['mean']:.6f}+/-{out['metrics']['error_rate']['ci95_half_width']:.6f} | "
                f"t={out['metrics']['avg_decode_time_sec']['mean']:.6f}+/-{out['metrics']['avg_decode_time_sec']['ci95_half_width']:.6f}s | "
                f"mem_peak={out['metrics']['memory_peak_bytes']['mean']:.1f} B"
            )

    scaling_models = _build_scaling_models(rows)

    rounds_profile = int(args.profile_distance) if args.rounds_mode == "distance" else int(args.rounds)
    profile_case = Case(
        distance=int(args.profile_distance),
        rounds=rounds_profile,
        p=float(args.p_phys),
        noise_model=str(args.noise_model),
        logical_basis=args.logical_basis,
    )
    profile_circuit = _build_circuit(profile_case)
    profile_seed = int(args.seed + 7777777 + int(args.profile_distance) * 131)
    profile_data = _profile_hotspots(
        circuit=profile_circuit,
        decoder=args.profile_decoder,
        shots=int(args.profile_shots),
        seed=profile_seed,
        g_threshold=float(args.g_threshold),
        adaptive_fast_mode=bool(args.adaptive_fast_mode),
        top_n=int(args.profile_top_n),
    )

    report = {
        "metadata": {
            "report_name": "week5_person1_profile_scaling",
            "timestamp_utc": utc_now_iso(),
        },
        "config": {
            "distances": [int(x) for x in distances],
            "rounds_mode": args.rounds_mode,
            "rounds": int(args.rounds),
            "decoders": list(decoders),
            "noise_model": args.noise_model,
            "p_phys": float(args.p_phys),
            "logical_basis": args.logical_basis,
            "shots": int(args.shots),
            "repeats": int(args.repeats),
            "seed": int(args.seed),
            "g_threshold": float(args.g_threshold),
            "adaptive_fast_mode": bool(args.adaptive_fast_mode),
            "profile_decoder": args.profile_decoder,
            "profile_distance": int(args.profile_distance),
            "profile_shots": int(args.profile_shots),
            "profile_top_n": int(args.profile_top_n),
        },
        "rows": rows,
        "scaling_models": scaling_models,
        "profile_hotspots": {
            "decoder": args.profile_decoder,
            "distance": int(args.profile_distance),
            "rounds": int(rounds_profile),
            "profile": profile_data,
        },
    }

    saved = save_json(report, Path(args.output))
    print_summary(report)
    print(f"\nJSON saved at: {saved}")


if __name__ == "__main__":
    main()

