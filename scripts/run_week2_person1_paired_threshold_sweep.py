# scripts/run_week2_person1_paired_threshold_sweep.py
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import numpy as np

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


def parse_thresholds(raw: str) -> List[float]:
    vals = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        v = float(x)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Threshold out of [0,1]: {v}")
        vals.append(v)
    if not vals:
        raise ValueError("No valid threshold was provided.")
    # Remove duplicates while preserving order
    out = []
    seen = set()
    for v in vals:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def logical_fail(pred: np.ndarray, obs: np.ndarray) -> bool:
    pred = np.asarray(pred, dtype=np.uint8).ravel() & 1
    obs = np.asarray(obs, dtype=np.uint8).ravel() & 1
    n = min(pred.size, obs.size)
    if n == 0:
        return False
    return bool(np.any(pred[:n] != obs[:n]))


def safe_speedup(ref_t: float, cand_t: float) -> float:
    if cand_t <= 0:
        return float("nan")
    return float(ref_t / cand_t)


def run_case_paired(case: Case, shots: int, thresholds: List[float]) -> Dict[str, Any]:
    circuit = generate_xzzx_circuit(
        distance=case.distance,
        rounds=case.rounds,
        noise_model=case.noise_model,
        p=case.p,
        logical_basis=case.logical_basis,
    )

    # Single sample for all decoders (paired comparison)
    sampler = circuit.compile_detector_sampler()
    dets, obs = sampler.sample(shots=shots, separate_observables=True)

    dets = np.asarray(dets, dtype=np.uint8)
    obs = np.asarray(obs, dtype=np.uint8)
    if obs.ndim == 1:
        obs = obs[:, np.newaxis]

    mwpm = MWPMDecoderWithSoftInfo(circuit)
    uf = UnionFindDecoderWithSoftInfo(circuit, prefer_union_find=True)
    adaptive = AdaptiveDecoder(
        circuit=circuit,
        fast_decoder=uf,
        accurate_decoder=mwpm,
        config=AdaptiveConfig(g_threshold=0.5, compare_against_mwpm_in_benchmark=False),
    )

    # Base accumulators
    mwpm_fails: List[bool] = []
    uf_fails: List[bool] = []
    mwpm_times: List[float] = []
    uf_times: List[float] = []

    # Accumulators by threshold
    adp_fails: Dict[float, List[bool]] = {g: [] for g in thresholds}
    adp_times: Dict[float, List[float]] = {g: [] for g in thresholds}
    adp_switches: Dict[float, int] = {g: 0 for g in thresholds}

    for i in range(shots):
        s = dets[i]
        o = obs[i]

        # MWPM
        pm, _, tm = mwpm.decode_with_confidence(s)
        mwpm_fails.append(logical_fail(pm, o))
        mwpm_times.append(float(tm))

        # UF
        pu, _, tu = uf.decode_with_confidence(s)
        uf_fails.append(logical_fail(pu, o))
        uf_times.append(float(tu))

        # Adaptive sweep thresholds
        for g in thresholds:
            pa, info_a, ta = adaptive.decode_adaptive(s, g_threshold=g)
            adp_fails[g].append(logical_fail(pa, o))
            adp_times[g].append(float(ta))
            if bool(info_a.get("switched", False)):
                adp_switches[g] += 1

    mwpm_er = float(np.mean(mwpm_fails))
    uf_er = float(np.mean(uf_fails))
    mwpm_t = float(np.mean(mwpm_times))
    uf_t = float(np.mean(uf_times))

    adaptive_by_threshold: List[Dict[str, Any]] = []
    for g in thresholds:
        er = float(np.mean(adp_fails[g]))
        t = float(np.mean(adp_times[g]))
        sw = float(adp_switches[g] / max(1, shots))
        adaptive_by_threshold.append(
            {
                "g_threshold": g,
                "error_rate": er,
                "avg_decode_time_sec": t,
                "switch_rate": sw,
                "speedup_vs_mwpm": safe_speedup(mwpm_t, t),
            }
        )

    return {
        "case_name": case.name,
        "distance": case.distance,
        "rounds": case.rounds,
        "p": case.p,
        "shots": shots,
        "num_detectors": int(getattr(circuit, "num_detectors", 0)),
        "num_observables": int(getattr(circuit, "num_observables", 0)),
        "mwpm": {
            "error_rate": mwpm_er,
            "avg_decode_time_sec": mwpm_t,
        },
        "uf": {
            "error_rate": uf_er,
            "avg_decode_time_sec": uf_t,
            "speedup_vs_mwpm": safe_speedup(mwpm_t, uf_t),
            "backend_info": {
                "matcher_build_mode": uf.matcher_build_mode,
                "decode_mode": uf.decode_mode,
                "is_union_find_backend": bool("union_find" in f"{uf.matcher_build_mode}|{uf.decode_mode}".lower()),
            },
        },
        "adaptive_by_threshold": adaptive_by_threshold,
        "status": "ok",
    }


def aggregate_report(rows: List[Dict[str, Any]], thresholds: List[float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "mean_mwpm_error_rate": mean([r["mwpm"]["error_rate"] for r in rows]) if rows else float("nan"),
        "mean_uf_error_rate": mean([r["uf"]["error_rate"] for r in rows]) if rows else float("nan"),
        "mean_mwpm_avg_decode_time_sec": mean([r["mwpm"]["avg_decode_time_sec"] for r in rows]) if rows else float("nan"),
        "mean_uf_avg_decode_time_sec": mean([r["uf"]["avg_decode_time_sec"] for r in rows]) if rows else float("nan"),
        "mean_uf_speedup_vs_mwpm": mean([r["uf"]["speedup_vs_mwpm"] for r in rows]) if rows else float("nan"),
        "adaptive_means_by_threshold": [],
    }

    for g in thresholds:
        ers, ts, sws, spds = [], [], [], []
        for r in rows:
            item = next(x for x in r["adaptive_by_threshold"] if float(x["g_threshold"]) == float(g))
            ers.append(item["error_rate"])
            ts.append(item["avg_decode_time_sec"])
            sws.append(item["switch_rate"])
            spds.append(item["speedup_vs_mwpm"])

        out["adaptive_means_by_threshold"].append(
            {
                "g_threshold": g,
                "mean_error_rate": mean(ers),
                "mean_avg_decode_time_sec": mean(ts),
                "mean_switch_rate": mean(sws),
                "mean_speedup_vs_mwpm": mean(spds),
            }
        )
    return out


def save_json(data: Dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_summary(report: Dict[str, Any]) -> None:
    print("\n=== Week2 Person1 Paired Sweep (same samples for all decoders) ===")
    rows = report["cases_summary"]
    for r in rows:
        print(f"\nCase {r['case_name']} (d={r['distance']}, r={r['rounds']}, p={r['p']})")
        print(
            f"  MWPM: ER={r['mwpm']['error_rate']:.4f}  t={r['mwpm']['avg_decode_time_sec']:.6f}s"
        )
        print(
            f"  UF  : ER={r['uf']['error_rate']:.4f}  t={r['uf']['avg_decode_time_sec']:.6f}s  "
            f"spd_vs_mwpm={r['uf']['speedup_vs_mwpm']:.3f}"
        )
        for a in r["adaptive_by_threshold"]:
            print(
                f"  ADP g={a['g_threshold']:.2f}: ER={a['error_rate']:.4f}  "
                f"t={a['avg_decode_time_sec']:.6f}s  sw={100*a['switch_rate']:.2f}%  "
                f"spd_vs_mwpm={a['speedup_vs_mwpm']:.3f}"
            )

    print("\n--- Aggregates ---")
    agg = report["aggregates"]
    print(f"mean_mwpm_error_rate              : {agg['mean_mwpm_error_rate']:.6f}")
    print(f"mean_uf_error_rate                : {agg['mean_uf_error_rate']:.6f}")
    print(f"mean_mwpm_avg_decode_time_sec     : {agg['mean_mwpm_avg_decode_time_sec']:.6f}")
    print(f"mean_uf_avg_decode_time_sec       : {agg['mean_uf_avg_decode_time_sec']:.6f}")
    print(f"mean_uf_speedup_vs_mwpm           : {agg['mean_uf_speedup_vs_mwpm']:.6f}")
    for a in agg["adaptive_means_by_threshold"]:
        print(
            f"g={a['g_threshold']:.2f} | mean_ER={a['mean_error_rate']:.6f} | "
            f"mean_t={a['mean_avg_decode_time_sec']:.6f} | "
            f"mean_sw={100*a['mean_switch_rate']:.2f}% | "
            f"mean_spd_vs_mwpm={a['mean_speedup_vs_mwpm']:.6f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired threshold sweep for MWPM/UF/Adaptive (same sampled syndromes)."
    )
    parser.add_argument("--shots", type=int, default=2000, help="Shots per case")
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.20,0.35,0.40,0.60,0.80",
        help="Comma-separated threshold list",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/week2_person1_paired_threshold_sweep.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shots <= 0:
        raise ValueError("--shots must be > 0")
    thresholds = parse_thresholds(args.thresholds)

    cases = default_cases()
    rows = [run_case_paired(c, shots=args.shots, thresholds=thresholds) for c in cases]
    report = {
        "metadata": {
            "report_name": "week2_person1_paired_threshold_sweep",
            "timestamp_utc": utc_now_iso(),
            "shots_per_case": args.shots,
            "thresholds": thresholds,
            "num_cases": len(cases),
        },
        "cases_summary": rows,
        "aggregates": aggregate_report(rows, thresholds),
    }

    save_json(report, Path(args.output))
    print_summary(report)
    print(f"\nJSON saved at: {args.output}")


if __name__ == "__main__":
    main()
