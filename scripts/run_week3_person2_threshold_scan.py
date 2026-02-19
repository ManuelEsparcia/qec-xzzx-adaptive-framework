from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
class ScanPoint:
    decoder: str
    noise_model: str
    distance: int
    rounds: int
    p_phys: float
    shots: int
    seed: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_csv_floats(raw: str) -> List[float]:
    vals: List[float] = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        vals.append(float(t))
    if not vals:
        raise ValueError("Empty float list.")
    return vals


def parse_csv_ints(raw: str) -> List[int]:
    vals: List[int] = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        vals.append(int(t))
    if not vals:
        raise ValueError("Empty int list.")
    return vals


def parse_decoders_csv(raw: str) -> List[str]:
    vals = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty decoder list.")
    bad = [x for x in vals if x not in DECODERS_ALLOWED]
    if bad:
        raise ValueError(f"Invalid decoders: {bad}. Allowed: {sorted(DECODERS_ALLOWED)}")
    out: List[str] = []
    seen = set()
    for x in vals:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def parse_noise_csv(raw: str) -> List[str]:
    vals = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty noise-model list.")
    bad = [x for x in vals if x not in NOISE_ALLOWED]
    if bad:
        raise ValueError(f"Invalid noise models: {bad}. Allowed: {sorted(NOISE_ALLOWED)}")
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
    raise ValueError(f"Unknown noise_name: {noise_name}")


def _safe_speedup(ref: float, cand: float) -> float:
    if cand <= 0:
        return float("nan")
    return float(ref / cand)


def _benchmark_decoder(
    *,
    circuit: Any,
    decoder: str,
    shots: int,
    seed: int,
    g_threshold: float,
    adaptive_fast_mode: bool,
) -> Dict[str, Any]:
    if decoder == "mwpm":
        d = MWPMDecoderWithSoftInfo(circuit)
        d.sampler = circuit.compile_detector_sampler(seed=int(seed))
        res = d.benchmark(shots=shots, keep_soft_info_samples=0)
        return {
            "error_rate": float(res["error_rate"]),
            "avg_decode_time_sec": float(res["avg_decode_time"]),
            "switch_rate": 0.0,
        }

    if decoder == "uf":
        d = UnionFindDecoderWithSoftInfo(circuit, prefer_union_find=True)
        d.sampler = circuit.compile_detector_sampler(seed=int(seed))
        res = d.benchmark(shots=shots, keep_soft_info_samples=0)
        return {
            "error_rate": float(res["error_rate"]),
            "avg_decode_time_sec": float(res["avg_decode_time"]),
            "switch_rate": 0.0,
            "backend_info": res.get("backend_info", {}),
        }

    if decoder == "bm":
        d = BeliefMatchingDecoderWithSoftInfo(circuit, prefer_belief_propagation=True)
        d.sampler = circuit.compile_detector_sampler(seed=int(seed))
        res = d.benchmark(shots=shots, keep_soft_info_samples=0)
        return {
            "error_rate": float(res["error_rate"]),
            "avg_decode_time_sec": float(res["avg_decode_time"]),
            "switch_rate": 0.0,
            "backend_info": res.get("backend_info", {}),
        }

    if decoder == "adaptive":
        mwpm = MWPMDecoderWithSoftInfo(circuit)
        uf = UnionFindDecoderWithSoftInfo(circuit, prefer_union_find=True)
        adp = AdaptiveDecoder(
            circuit=circuit,
            fast_decoder=uf,
            accurate_decoder=mwpm,
            config=AdaptiveConfig(
                g_threshold=g_threshold,
                compare_against_mwpm_in_benchmark=False,
            ),
        )
        adp.sampler = circuit.compile_detector_sampler(seed=int(seed))
        res = adp.benchmark_adaptive(
            shots=shots,
            g_threshold=g_threshold,
            keep_samples=0,
            compare_against_mwpm=False,
            fast_mode=adaptive_fast_mode,
        )
        return {
            "error_rate": float(res["error_rate_adaptive"]),
            "avg_decode_time_sec": float(res["avg_decode_time_adaptive"]),
            "switch_rate": float(res["switch_rate"]),
            "fast_mode": bool(res.get("fast_mode", adaptive_fast_mode)),
        }

    raise ValueError(f"Unknown decoder: {decoder}")


def run_point(
    point: ScanPoint,
    *,
    logical_basis: str,
    g_threshold: float,
    adaptive_fast_mode: bool,
) -> Dict[str, Any]:
    base = generate_xzzx_circuit(
        distance=point.distance,
        rounds=point.rounds,
        noise_model="none",
        p=0.0,
        logical_basis=logical_basis,
    )
    noisy = apply_noise_model(base, model=noise_spec_from_name(point.noise_model, point.p_phys))
    metrics = _benchmark_decoder(
        circuit=noisy,
        decoder=point.decoder,
        shots=point.shots,
        seed=point.seed,
        g_threshold=g_threshold,
        adaptive_fast_mode=adaptive_fast_mode,
    )

    return {
        "decoder": point.decoder,
        "noise_model": point.noise_model,
        "distance": int(point.distance),
        "rounds": int(point.rounds),
        "p_phys": float(point.p_phys),
        "shots": int(point.shots),
        "seed": int(point.seed),
        "error_rate": float(metrics["error_rate"]),
        "avg_decode_time_sec": float(metrics["avg_decode_time_sec"]),
        "switch_rate": float(metrics.get("switch_rate", 0.0)),
        "status": "ok",
        **({"backend_info": metrics["backend_info"]} if "backend_info" in metrics else {}),
        **({"fast_mode": metrics["fast_mode"]} if "fast_mode" in metrics else {}),
    }


def _estimate_threshold_for_pair(
    p_values: List[float],
    er_small: List[float],
    er_large: List[float],
) -> Dict[str, Any]:
    diffs = [b - a for a, b in zip(er_small, er_large)]
    for i in range(len(p_values) - 1):
        d0 = diffs[i]
        d1 = diffs[i + 1]
        if d0 == 0.0:
            return {
                "method": "exact_point",
                "p_threshold_estimate": float(p_values[i]),
                "index_pair": [i, i],
            }
        if d0 * d1 < 0:
            p0, p1 = p_values[i], p_values[i + 1]
            # linear interpolation in diff-space
            frac = abs(d0) / (abs(d0) + abs(d1))
            p_cross = p0 + (p1 - p0) * frac
            return {
                "method": "linear_crossing",
                "p_threshold_estimate": float(p_cross),
                "index_pair": [i, i + 1],
            }

    # fallback: nearest absolute difference
    j = min(range(len(p_values)), key=lambda k: abs(diffs[k]))
    return {
        "method": "nearest_abs_diff",
        "p_threshold_estimate": float(p_values[j]),
        "index_pair": [j, j],
    }


def estimate_thresholds(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (str(r["decoder"]), str(r["noise_model"]))
        grouped.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    for (decoder, noise_model), items in grouped.items():
        distances = sorted({int(r["distance"]) for r in items})
        if len(distances) < 2:
            continue

        # Use first and last distance as a stable coarse threshold estimate.
        d_small = distances[0]
        d_large = distances[-1]

        small = [r for r in items if int(r["distance"]) == d_small]
        large = [r for r in items if int(r["distance"]) == d_large]

        p_common = sorted({float(r["p_phys"]) for r in small}.intersection(float(r["p_phys"]) for r in large))
        if len(p_common) < 2:
            continue

        def _er_at(arr: Sequence[Dict[str, Any]], p: float) -> float:
            vals = [float(x["error_rate"]) for x in arr if math.isclose(float(x["p_phys"]), p, rel_tol=0.0, abs_tol=1e-12)]
            return float(mean(vals)) if vals else float("nan")

        er_small = [_er_at(small, p) for p in p_common]
        er_large = [_er_at(large, p) for p in p_common]
        if any(not math.isfinite(x) for x in er_small + er_large):
            continue

        est = _estimate_threshold_for_pair(p_common, er_small, er_large)
        out.append(
            {
                "decoder": decoder,
                "noise_model": noise_model,
                "distance_pair": [d_small, d_large],
                "p_values": p_common,
                "error_rate_small_distance": er_small,
                "error_rate_large_distance": er_large,
                **est,
            }
        )
    return out


def aggregate_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}

    by_pair: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        by_pair.setdefault((str(r["decoder"]), str(r["noise_model"])), []).append(r)

    pair_summary: List[Dict[str, Any]] = []
    for (decoder, noise_model), arr in sorted(by_pair.items()):
        pair_summary.append(
            {
                "decoder": decoder,
                "noise_model": noise_model,
                "num_points": len(arr),
                "mean_error_rate": float(mean(float(x["error_rate"]) for x in arr)),
                "mean_avg_decode_time_sec": float(mean(float(x["avg_decode_time_sec"]) for x in arr)),
                "mean_switch_rate": float(mean(float(x.get("switch_rate", 0.0)) for x in arr)),
            }
        )

    # Pareto-like summary at same (decoder, noise, distance): min error and min time points.
    pareto_refs: List[Dict[str, Any]] = []
    by_triplet: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for r in rows:
        by_triplet.setdefault((str(r["decoder"]), str(r["noise_model"]), int(r["distance"])), []).append(r)
    for (decoder, noise_model, distance), arr in sorted(by_triplet.items()):
        best_er = min(arr, key=lambda x: (float(x["error_rate"]), float(x["avg_decode_time_sec"])))
        best_t = min(arr, key=lambda x: (float(x["avg_decode_time_sec"]), float(x["error_rate"])))
        pareto_refs.append(
            {
                "decoder": decoder,
                "noise_model": noise_model,
                "distance": distance,
                "best_error_rate_point": {
                    "p_phys": float(best_er["p_phys"]),
                    "error_rate": float(best_er["error_rate"]),
                    "avg_decode_time_sec": float(best_er["avg_decode_time_sec"]),
                },
                "best_time_point": {
                    "p_phys": float(best_t["p_phys"]),
                    "error_rate": float(best_t["error_rate"]),
                    "avg_decode_time_sec": float(best_t["avg_decode_time_sec"]),
                },
            }
        )

    return {
        "num_points": len(rows),
        "pair_summary": pair_summary,
        "pareto_reference_points": pareto_refs,
    }


def save_json(data: Dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return output_path


def build_report(
    *,
    rows: Sequence[Dict[str, Any]],
    distances: Sequence[int],
    p_values: Sequence[float],
    decoders: Sequence[str],
    noise_models: Sequence[str],
    shots: int,
    rounds: int,
    seed: int,
    g_threshold: float,
    adaptive_fast_mode: bool,
    logical_basis: str,
    partial: bool,
) -> Dict[str, Any]:
    return {
        "metadata": {
            "report_name": "week3_person2_threshold_scan",
            "timestamp_utc": utc_now_iso(),
            "partial": bool(partial),
        },
        "config": {
            "distances": [int(x) for x in distances],
            "p_values": [float(x) for x in p_values],
            "decoders": list(decoders),
            "noise_models": list(noise_models),
            "shots": int(shots),
            "rounds": int(rounds),
            "seed": int(seed),
            "g_threshold": float(g_threshold),
            "adaptive_fast_mode": bool(adaptive_fast_mode),
            "logical_basis": logical_basis,
        },
        "points": list(rows),
        "aggregates": aggregate_rows(rows),
        "threshold_estimates": estimate_thresholds(rows),
    }


def print_summary(report: Dict[str, Any]) -> None:
    print("\n=== Week 3 Person 2 - Threshold Scan ===")
    agg = report.get("aggregates", {})
    print(f"points: {agg.get('num_points', 0)}")
    for row in agg.get("pair_summary", [])[:20]:
        print(
            f"{row['decoder']:<8} | {row['noise_model']:<14} | "
            f"n={row['num_points']:<3} | ER={row['mean_error_rate']:.6f} | "
            f"t={row['mean_avg_decode_time_sec']:.6f}s | sw={100.0*row['mean_switch_rate']:.2f}%"
        )
    th = report.get("threshold_estimates", [])
    if th:
        print("\n--- Threshold estimates (distance crossing) ---")
        for t in th:
            print(
                f"{t['decoder']:<8} | {t['noise_model']:<14} | "
                f"d={t['distance_pair'][0]}->{t['distance_pair'][1]} | "
                f"p*={t['p_threshold_estimate']:.6f} ({t['method']})"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week 3 Person 2 massive simulation grid with threshold fitting."
    )
    parser.add_argument("--distances", type=str, default="3,5,7", help="CSV of odd distances.")
    parser.add_argument("--rounds", type=int, default=3, help="Rounds per circuit.")
    parser.add_argument(
        "--p-values",
        type=str,
        default="0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02",
        help="CSV of physical error rates.",
    )
    parser.add_argument("--decoders", type=str, default="mwpm,uf,bm,adaptive", help="CSV decoders.")
    parser.add_argument(
        "--noise-models",
        type=str,
        default="depolarizing,biased_eta10,biased_eta100,biased_eta500,circuit_level,correlated",
        help="CSV noise models.",
    )
    parser.add_argument("--shots", type=int, default=300, help="Shots per point.")
    parser.add_argument("--seed", type=int, default=2026, help="Base seed.")
    parser.add_argument("--g-threshold", type=float, default=0.35, help="Adaptive threshold.")
    parser.add_argument(
        "--adaptive-fast-mode",
        action="store_true",
        help="Use Adaptive fast_mode=True during adaptive benchmarks.",
    )
    parser.add_argument("--logical-basis", type=str, default="x", choices=["x", "z"], help="Logical basis.")
    parser.add_argument("--checkpoint-every", type=int, default=24, help="Save partial JSON every N points.")
    parser.add_argument(
        "--output",
        type=str,
        default="results/week3_person2_threshold_scan.json",
        help="Output JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distances = parse_csv_ints(args.distances)
    p_values = parse_csv_floats(args.p_values)
    decoders = parse_decoders_csv(args.decoders)
    noise_models = parse_noise_csv(args.noise_models)

    if any(d < 3 or d % 2 == 0 for d in distances):
        raise ValueError("--distances must contain odd values >= 3.")
    if args.rounds <= 0:
        raise ValueError("--rounds must be > 0")
    if args.shots <= 0:
        raise ValueError("--shots must be > 0")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be >= 0")
    if not (0.0 <= args.g_threshold <= 1.0):
        raise ValueError("--g-threshold must be in [0,1]")
    if any((p < 0.0 or p > 1.0) for p in p_values):
        raise ValueError("--p-values entries must be in [0,1]")

    rows: List[Dict[str, Any]] = []
    total = len(distances) * len(p_values) * len(decoders) * len(noise_models)
    idx = 0
    for d in distances:
        for noise in noise_models:
            for decoder in decoders:
                for p in p_values:
                    idx += 1
                    point_seed = int(args.seed + idx * 104729 + d * 1009)
                    point = ScanPoint(
                        decoder=decoder,
                        noise_model=noise,
                        distance=d,
                        rounds=int(args.rounds),
                        p_phys=float(p),
                        shots=int(args.shots),
                        seed=point_seed,
                    )
                    row = run_point(
                        point,
                        logical_basis=args.logical_basis,
                        g_threshold=float(args.g_threshold),
                        adaptive_fast_mode=bool(args.adaptive_fast_mode),
                    )
                    rows.append(row)
                    print(
                        f"[{idx}/{total}] {decoder} | {noise} | d={d} | p={p:.6f} | "
                        f"ER={row['error_rate']:.6f} | t={row['avg_decode_time_sec']:.6f}s"
                    )

                    if args.checkpoint_every > 0 and (idx % args.checkpoint_every == 0):
                        partial = build_report(
                            rows=rows,
                            distances=distances,
                            p_values=p_values,
                            decoders=decoders,
                            noise_models=noise_models,
                            shots=int(args.shots),
                            rounds=int(args.rounds),
                            seed=int(args.seed),
                            g_threshold=float(args.g_threshold),
                            adaptive_fast_mode=bool(args.adaptive_fast_mode),
                            logical_basis=args.logical_basis,
                            partial=True,
                        )
                        save_json(partial, Path(args.output))

    report = build_report(
        rows=rows,
        distances=distances,
        p_values=p_values,
        decoders=decoders,
        noise_models=noise_models,
        shots=int(args.shots),
        rounds=int(args.rounds),
        seed=int(args.seed),
        g_threshold=float(args.g_threshold),
        adaptive_fast_mode=bool(args.adaptive_fast_mode),
        logical_basis=args.logical_basis,
        partial=False,
    )
    saved = save_json(report, Path(args.output))
    print_summary(report)
    print(f"\nJSON saved at: {saved}")


if __name__ == "__main__":
    main()
