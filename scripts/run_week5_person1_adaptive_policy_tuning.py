from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Ensure "src" import works when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.codes.xzzx_code import generate_xzzx_circuit
from src.decoders.belief_matching_decoder import BeliefMatchingDecoderWithSoftInfo
from src.decoders.mwpm_decoder import MWPMDecoderWithSoftInfo
from src.decoders.union_find_decoder import UnionFindDecoderWithSoftInfo
from src.noise.noise_models import apply_noise_model
from src.switching.adaptive_decoder import AdaptiveConfig, AdaptiveDecoder

NOISE_ALLOWED = {
    "depolarizing",
    "biased_eta10",
    "biased_eta100",
    "biased_eta500",
    "circuit_level",
    "correlated",
}
FAST_BACKEND_ALLOWED = {"uf", "bm"}


@dataclass(frozen=True)
class Case:
    distance: int
    rounds: int
    p: float
    noise_model: str
    logical_basis: str


@dataclass(frozen=True)
class Policy:
    g_threshold: float
    min_switch_weight: Optional[int]
    mode: str  # "standard" or "fast"
    fast_backend: str  # "uf" or "bm"

    @property
    def fast_mode(self) -> bool:
        return self.mode == "fast"

    @property
    def label(self) -> str:
        w = "none" if self.min_switch_weight is None else str(int(self.min_switch_weight))
        return f"{self.fast_backend}_g{self.g_threshold:.2f}_w{w}_{self.mode}"


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


def parse_csv_floats(raw: str) -> List[float]:
    vals: List[float] = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        vals.append(float(t))
    if not vals:
        raise ValueError("empty float list")
    return vals


def parse_fast_backends(raw: str) -> List[str]:
    vals = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("empty fast-backend list")
    bad = [x for x in vals if x not in FAST_BACKEND_ALLOWED]
    if bad:
        raise ValueError(
            f"invalid fast backend(s): {bad}; allowed: {sorted(FAST_BACKEND_ALLOWED)}"
        )
    out: List[str] = []
    seen = set()
    for x in vals:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def parse_min_switch_weights(raw: str) -> List[Optional[int]]:
    vals: List[Optional[int]] = []
    for token in raw.split(","):
        t = token.strip().lower()
        if not t:
            continue
        if t in {"none", "null", "na", "n/a"}:
            vals.append(None)
        else:
            vals.append(int(t))
    if not vals:
        raise ValueError("empty min-switch-weight list")
    out: List[Optional[int]] = []
    seen = set()
    for v in vals:
        key = ("none" if v is None else str(v))
        if key not in seen:
            seen.add(key)
            out.append(v)
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
        distance=int(case.distance),
        rounds=int(case.rounds),
        noise_model="none",
        p=0.0,
        logical_basis=case.logical_basis,
    )
    return apply_noise_model(base, model=noise_spec_from_name(case.noise_model, case.p))


def run_case_policy(
    *,
    case: Case,
    policy: Policy,
    shots: int,
    repeats: int,
    base_seed: int,
    max_delta_error: float,
    min_speedup: float,
) -> Dict[str, Any]:
    circuit = _build_circuit(case)
    mwpm = MWPMDecoderWithSoftInfo(circuit)
    if policy.fast_backend == "uf":
        fast_decoder = UnionFindDecoderWithSoftInfo(circuit, prefer_union_find=True)
    elif policy.fast_backend == "bm":
        fast_decoder = BeliefMatchingDecoderWithSoftInfo(
            circuit,
            prefer_belief_propagation=True,
        )
    else:
        raise ValueError(f"unsupported fast backend: {policy.fast_backend}")

    adaptive = AdaptiveDecoder(
        circuit=circuit,
        fast_decoder=fast_decoder,
        accurate_decoder=mwpm,
        config=AdaptiveConfig(
            g_threshold=float(policy.g_threshold),
            compare_against_mwpm_in_benchmark=True,
            min_syndrome_weight_for_switch=policy.min_switch_weight,
        ),
    )

    repeat_runs: List[Dict[str, Any]] = []
    for i in range(repeats):
        seed_i = int(base_seed + i * 10007 + case.distance * 997 + case.rounds * 97)
        adaptive.sampler = circuit.compile_detector_sampler(seed=seed_i)
        out = adaptive.benchmark_adaptive(
            shots=int(shots),
            g_threshold=float(policy.g_threshold),
            keep_samples=0,
            compare_against_mwpm=True,
            fast_mode=policy.fast_mode,
        )
        ref = out["reference_mwpm"]
        er_a = float(out["error_rate_adaptive"])
        er_m = float(ref["error_rate_mwpm"])
        delta_er = float(er_a - er_m)
        repeat_runs.append(
            {
                "repeat_index": int(i),
                "seed": int(seed_i),
                "error_rate_adaptive": er_a,
                "error_rate_mwpm": er_m,
                "delta_error_rate_adaptive_minus_mwpm": delta_er,
                "avg_decode_time_adaptive_sec": float(out["avg_decode_time_adaptive"]),
                "avg_decode_time_mwpm_sec": float(ref["avg_decode_time_mwpm"]),
                "speedup_vs_mwpm": float(out["speedup_vs_mwpm"]),
                "switch_rate": float(out["switch_rate"]),
            }
        )

    metrics = {
        "error_rate_adaptive": _summary(
            [float(r["error_rate_adaptive"]) for r in repeat_runs],
            clamp_01=True,
        ),
        "error_rate_mwpm": _summary(
            [float(r["error_rate_mwpm"]) for r in repeat_runs],
            clamp_01=True,
        ),
        "delta_error_rate_adaptive_minus_mwpm": _summary(
            [float(r["delta_error_rate_adaptive_minus_mwpm"]) for r in repeat_runs]
        ),
        "avg_decode_time_adaptive_sec": _summary(
            [float(r["avg_decode_time_adaptive_sec"]) for r in repeat_runs]
        ),
        "avg_decode_time_mwpm_sec": _summary(
            [float(r["avg_decode_time_mwpm_sec"]) for r in repeat_runs]
        ),
        "speedup_vs_mwpm": _summary([float(r["speedup_vs_mwpm"]) for r in repeat_runs]),
        "switch_rate": _summary([float(r["switch_rate"]) for r in repeat_runs], clamp_01=True),
    }

    delta_mean = float(metrics["delta_error_rate_adaptive_minus_mwpm"]["mean"])
    speedup_mean = float(metrics["speedup_vs_mwpm"]["mean"])
    within_error_budget = bool(delta_mean <= float(max_delta_error))
    meets_speedup_target = bool(speedup_mean >= float(min_speedup))

    return {
        "case_key": f"d{case.distance}_r{case.rounds}_p{case.p:.6f}_{case.noise_model}",
        "distance": int(case.distance),
        "rounds": int(case.rounds),
        "p_phys": float(case.p),
        "noise_model": str(case.noise_model),
        "logical_basis": str(case.logical_basis),
        "policy": {
            "label": policy.label,
            "fast_backend": policy.fast_backend,
            "g_threshold": float(policy.g_threshold),
            "min_switch_weight": None
            if policy.min_switch_weight is None
            else int(policy.min_switch_weight),
            "mode": policy.mode,
            "fast_mode": bool(policy.fast_mode),
        },
        "shots": int(shots),
        "repeats": int(repeats),
        "base_seed": int(base_seed),
        "metrics": metrics,
        "within_error_budget": within_error_budget,
        "meets_speedup_target": meets_speedup_target,
        "is_feasible": bool(within_error_budget and meets_speedup_target),
        "repeat_runs": repeat_runs,
        "status": "ok",
    }


def summarize_policies(rows: Sequence[Dict[str, Any]], *, max_delta_error: float, min_speedup: float) -> List[Dict[str, Any]]:
    by_policy: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        label = str(row["policy"]["label"])
        by_policy.setdefault(label, []).append(row)

    out: List[Dict[str, Any]] = []
    for label, group in sorted(by_policy.items()):
        speedups = [float(r["metrics"]["speedup_vs_mwpm"]["mean"]) for r in group]
        deltas = [float(r["metrics"]["delta_error_rate_adaptive_minus_mwpm"]["mean"]) for r in group]
        switches = [float(r["metrics"]["switch_rate"]["mean"]) for r in group]
        mean_speedup = float(mean(speedups))
        mean_delta = float(mean(deltas))
        max_delta = float(max(deltas))

        pass_error_all = bool(all(d <= float(max_delta_error) for d in deltas))
        pass_speedup_all = bool(all(s >= float(min_speedup) for s in speedups))
        out.append(
            {
                "policy_label": label,
                "policy": group[0]["policy"],
                "num_cases": int(len(group)),
                "mean_speedup_vs_mwpm": mean_speedup,
                "mean_delta_error_rate_adaptive_minus_mwpm": mean_delta,
                "max_delta_error_rate_adaptive_minus_mwpm": max_delta,
                "mean_switch_rate": float(mean(switches)),
                "pass_error_budget_all_cases": pass_error_all,
                "pass_speedup_target_all_cases": pass_speedup_all,
                "is_globally_feasible": bool(pass_error_all and pass_speedup_all),
            }
        )

    out.sort(
        key=lambda x: (
            not bool(x["is_globally_feasible"]),
            -float(x["mean_speedup_vs_mwpm"]),
            float(x["mean_delta_error_rate_adaptive_minus_mwpm"]),
            float(x["mean_switch_rate"]),
        )
    )
    return out


def select_best_global(policy_summary: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not policy_summary:
        return None
    feasible = [x for x in policy_summary if bool(x.get("is_globally_feasible", False))]
    if feasible:
        return feasible[0]
    return policy_summary[0]


def select_best_by_case(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_case: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_key"]), []).append(row)

    out: List[Dict[str, Any]] = []
    for case_key, group in sorted(by_case.items()):
        feasible = [g for g in group if bool(g.get("is_feasible", False))]
        ranked = feasible if feasible else list(group)
        ranked.sort(
            key=lambda x: (
                -float(x["metrics"]["speedup_vs_mwpm"]["mean"]),
                float(x["metrics"]["delta_error_rate_adaptive_minus_mwpm"]["mean"]),
                float(x["metrics"]["switch_rate"]["mean"]),
            )
        )
        best = ranked[0]
        out.append(
            {
                "case_key": case_key,
                "distance": int(best["distance"]),
                "rounds": int(best["rounds"]),
                "policy": best["policy"],
                "is_feasible": bool(best["is_feasible"]),
                "speedup_vs_mwpm_mean": float(best["metrics"]["speedup_vs_mwpm"]["mean"]),
                "delta_error_rate_mean": float(
                    best["metrics"]["delta_error_rate_adaptive_minus_mwpm"]["mean"]
                ),
                "switch_rate_mean": float(best["metrics"]["switch_rate"]["mean"]),
            }
        )
    return out


def save_json(data: Dict[str, Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return output


def print_summary(report: Dict[str, Any]) -> None:
    cfg = report["config"]
    print("\n=== Week 5 Person 1 - Adaptive Policy Tuning ===")
    print(
        f"distances={cfg['distances']} | fast_backends={cfg['fast_backends']} | "
        f"g_thresholds={cfg['g_thresholds']} | "
        f"min_switch_weights={cfg['min_switch_weights']} | mode={cfg['mode']} | "
        f"shots={cfg['shots']} | repeats={cfg['repeats']}"
    )
    print(
        f"constraints: max_delta_error={cfg['max_delta_error']:.6f}, "
        f"min_speedup={cfg['min_speedup']:.3f}"
    )

    best = report.get("best_global_policy")
    if best:
        print("\n--- Best global policy ---")
        print(
            f"{best['policy_label']} | mean_speedup={best['mean_speedup_vs_mwpm']:.3f} | "
            f"mean_delta_error={best['mean_delta_error_rate_adaptive_minus_mwpm']:.6f} | "
            f"globally_feasible={best['is_globally_feasible']}"
        )

    print("\n--- Top policy summary (first 8) ---")
    for row in report.get("policy_summary", [])[:8]:
        print(
            f"{row['policy_label']:<24} | spd={row['mean_speedup_vs_mwpm']:.3f} | "
            f"dER={row['mean_delta_error_rate_adaptive_minus_mwpm']:.6f} | "
            f"sw={100.0 * row['mean_switch_rate']:.2f}% | "
            f"feasible={row['is_globally_feasible']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Week 5 Block 4: adaptive policy tuning for speedup vs MWPM "
            "with controlled error-rate degradation."
        )
    )
    parser.add_argument("--distances", type=str, default="5,7,9,11,13", help="CSV odd distances >= 3.")
    parser.add_argument(
        "--rounds-mode",
        type=str,
        choices=["fixed", "distance"],
        default="distance",
        help="Use fixed rounds or rounds=distance.",
    )
    parser.add_argument("--rounds", type=int, default=3, help="Rounds when --rounds-mode=fixed.")
    parser.add_argument("--noise-model", type=str, default="depolarizing", help="Noise model alias.")
    parser.add_argument("--p-phys", type=float, default=0.01, help="Physical error rate.")
    parser.add_argument("--logical-basis", type=str, choices=["x", "z"], default="x", help="Logical basis.")
    parser.add_argument("--shots", type=int, default=150, help="Shots per run.")
    parser.add_argument("--repeats", type=int, default=2, help="Repeated runs per case/policy.")
    parser.add_argument("--seed", type=int, default=2026, help="Base seed.")
    parser.add_argument("--g-thresholds", type=str, default="0.20,0.35,0.50,0.65,0.80", help="CSV policy thresholds.")
    parser.add_argument(
        "--fast-backends",
        type=str,
        default="uf,bm",
        help="CSV of adaptive fast backends.",
    )
    parser.add_argument(
        "--min-switch-weights",
        type=str,
        default="none,1,2,3",
        help="CSV of switch-weight gates (int or 'none').",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "fast"],
        default="fast",
        help="Adaptive benchmark mode.",
    )
    parser.add_argument(
        "--max-delta-error",
        type=float,
        default=0.01,
        help="Maximum allowed (adaptive_er - mwpm_er).",
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=1.0,
        help="Minimum required speedup_vs_mwpm for feasibility.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/week5_person1_adaptive_policy_tuning.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distances = parse_csv_ints(args.distances)
    g_thresholds = parse_csv_floats(args.g_thresholds)
    fast_backends = parse_fast_backends(args.fast_backends)
    min_switch_weights = parse_min_switch_weights(args.min_switch_weights)

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
    if any((g < 0.0 or g > 1.0) for g in g_thresholds):
        raise ValueError("--g-thresholds entries must be in [0,1]")
    if any((w is not None and int(w) < 0) for w in min_switch_weights):
        raise ValueError("--min-switch-weights int entries must be >= 0")
    if float(args.max_delta_error) < 0.0:
        raise ValueError("--max-delta-error must be >= 0")
    if float(args.min_speedup) <= 0.0:
        raise ValueError("--min-speedup must be > 0")

    cases: List[Case] = []
    for d in distances:
        rounds_i = int(d) if args.rounds_mode == "distance" else int(args.rounds)
        cases.append(
            Case(
                distance=int(d),
                rounds=rounds_i,
                p=float(args.p_phys),
                noise_model=str(args.noise_model),
                logical_basis=str(args.logical_basis),
            )
        )

    policies = [
        Policy(
            g_threshold=float(g),
            min_switch_weight=w,
            mode=str(args.mode),
            fast_backend=str(fb),
        )
        for fb in fast_backends
        for g in g_thresholds
        for w in min_switch_weights
    ]

    total = len(cases) * len(policies)
    idx = 0
    rows: List[Dict[str, Any]] = []
    for c_i, case in enumerate(cases):
        for p_i, policy in enumerate(policies):
            idx += 1
            base_seed = int(args.seed + c_i * 1000003 + p_i * 50021 + case.distance * 97)
            row = run_case_policy(
                case=case,
                policy=policy,
                shots=int(args.shots),
                repeats=int(args.repeats),
                base_seed=base_seed,
                max_delta_error=float(args.max_delta_error),
                min_speedup=float(args.min_speedup),
            )
            rows.append(row)
            print(
                f"[{idx}/{total}] d={case.distance:>2} r={case.rounds:>2} | {policy.label:<24} | "
                f"spd={row['metrics']['speedup_vs_mwpm']['mean']:.3f} | "
                f"dER={row['metrics']['delta_error_rate_adaptive_minus_mwpm']['mean']:.6f} | "
                f"sw={100.0 * row['metrics']['switch_rate']['mean']:.2f}% | "
                f"feasible={row['is_feasible']}"
            )

    policy_summary = summarize_policies(
        rows,
        max_delta_error=float(args.max_delta_error),
        min_speedup=float(args.min_speedup),
    )
    best_global = select_best_global(policy_summary)
    best_by_case = select_best_by_case(rows)

    report = {
        "metadata": {
            "report_name": "week5_person1_adaptive_policy_tuning",
            "timestamp_utc": utc_now_iso(),
        },
        "config": {
            "distances": [int(x) for x in distances],
            "rounds_mode": args.rounds_mode,
            "rounds": int(args.rounds),
            "noise_model": str(args.noise_model),
            "p_phys": float(args.p_phys),
            "logical_basis": str(args.logical_basis),
            "shots": int(args.shots),
            "repeats": int(args.repeats),
            "seed": int(args.seed),
            "fast_backends": list(fast_backends),
            "g_thresholds": [float(x) for x in g_thresholds],
            "min_switch_weights": [
                None if w is None else int(w) for w in min_switch_weights
            ],
            "mode": str(args.mode),
            "max_delta_error": float(args.max_delta_error),
            "min_speedup": float(args.min_speedup),
        },
        "rows": rows,
        "policy_summary": policy_summary,
        "best_global_policy": best_global,
        "best_policy_by_case": best_by_case,
    }

    saved = save_json(report, Path(args.output))
    print_summary(report)
    print(f"\nJSON saved at: {saved}")


if __name__ == "__main__":
    main()
