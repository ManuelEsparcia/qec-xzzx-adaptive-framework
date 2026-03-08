from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence


DECODERS_ALLOWED = {"mwpm", "uf", "bm", "adaptive"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_csv_ints(raw: str) -> List[int]:
    out: List[int] = []
    for tok in str(raw).split(","):
        s = tok.strip()
        if not s:
            continue
        out.append(int(s))
    if not out:
        raise ValueError("empty integer CSV")
    return out


def parse_decoders_csv(raw: str) -> List[str]:
    out = [x.strip().lower() for x in str(raw).split(",") if x.strip()]
    if not out:
        raise ValueError("empty decoder CSV")
    bad = [x for x in out if x not in DECODERS_ALLOWED]
    if bad:
        raise ValueError(f"invalid decoders: {bad}; allowed={sorted(DECODERS_ALLOWED)}")
    return out


def _is_finite_number(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _stress_row_from_profile_row(row: Dict[str, Any], *, source_mode: str) -> Dict[str, Any]:
    metrics = dict(row.get("metrics", {}))
    error_rate = float(metrics.get("error_rate", {}).get("mean", float("nan")))
    avg_decode_time_sec = float(metrics.get("avg_decode_time_sec", {}).get("mean", float("nan")))
    memory_peak_bytes = float(metrics.get("memory_peak_bytes", {}).get("mean", float("nan")))
    switch_rate = float(metrics.get("switch_rate", {}).get("mean", float("nan")))
    status = str(row.get("status", "unknown"))
    health = {
        "status_ok": bool(status == "ok"),
        "finite_error_rate": _is_finite_number(error_rate),
        "finite_avg_decode_time_sec": _is_finite_number(avg_decode_time_sec),
        "finite_memory_peak_bytes": _is_finite_number(memory_peak_bytes),
        "error_rate_in_01": bool(_is_finite_number(error_rate) and 0.0 <= error_rate <= 1.0),
    }
    pass_flag = all(bool(v) for v in health.values())
    return {
        "decoder": str(row["decoder"]),
        "distance": int(row["distance"]),
        "status": status,
        "pass": bool(pass_flag),
        "source_mode": str(source_mode),
        "metrics": {
            "error_rate": error_rate,
            "avg_decode_time_sec": avg_decode_time_sec,
            "memory_peak_bytes": memory_peak_bytes,
            "switch_rate": switch_rate,
        },
        "health": health,
    }


def _rows_from_profile_artifact(
    *,
    profile_artifact: Path,
    distances: Sequence[int],
    decoders: Sequence[str],
) -> List[Dict[str, Any]]:
    payload = json.loads(profile_artifact.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"invalid profile artifact rows in {profile_artifact}")
    by_key: Dict[tuple[str, int], Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            key = (str(row["decoder"]), int(row["distance"]))
        except (KeyError, TypeError, ValueError):
            continue
        by_key[key] = row

    out: List[Dict[str, Any]] = []
    for d in distances:
        for dec in decoders:
            key = (str(dec), int(d))
            if key not in by_key:
                raise ValueError(
                    f"profile artifact missing row for decoder={dec}, distance={d}: {profile_artifact}"
                )
            out.append(_stress_row_from_profile_row(by_key[key], source_mode="profile_artifact"))
    return out


def _rows_from_direct_benchmark(
    *,
    distances: Sequence[int],
    decoders: Sequence[str],
    rounds_mode: str,
    rounds_fixed: int,
    noise_model: str,
    p_phys: float,
    logical_basis: str,
    shots: int,
    repeats: int,
    seed: int,
    g_threshold: float,
    adaptive_fast_mode: bool,
) -> List[Dict[str, Any]]:
    from scripts.run_week5_person1_profile_scaling import Case, run_case_decoder

    out: List[Dict[str, Any]] = []
    idx = 0
    total = len(distances) * len(decoders)
    for d in distances:
        rounds_i = int(d) if rounds_mode == "distance" else int(rounds_fixed)
        case = Case(
            distance=int(d),
            rounds=rounds_i,
            p=float(p_phys),
            noise_model=str(noise_model),
            logical_basis=str(logical_basis),
        )
        for dec in decoders:
            idx += 1
            base_seed = int(seed + idx * 1000003 + int(d) * 97)
            row = run_case_decoder(
                case=case,
                decoder=str(dec),
                shots=int(shots),
                repeats=int(repeats),
                base_seed=base_seed,
                g_threshold=float(g_threshold),
                adaptive_fast_mode=bool(adaptive_fast_mode),
            )
            stress_row = _stress_row_from_profile_row(row, source_mode="direct_benchmark")
            out.append(stress_row)
            print(
                f"[{idx}/{total}] {dec:<8} | d={d:>2} | "
                f"status={stress_row['status']:<2} | pass={stress_row['pass']} | "
                f"t={stress_row['metrics']['avg_decode_time_sec']:.6f}s | "
                f"mem={stress_row['metrics']['memory_peak_bytes']:.1f} B"
            )
    return out


def save_json(data: Dict[str, Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Week 5 dedicated stress test for high distances (d=11,13) across core decoders."
        )
    )
    parser.add_argument("--distances", type=str, default="11,13", help="CSV odd distances >= 3.")
    parser.add_argument("--decoders", type=str, default="mwpm,uf,bm,adaptive", help="CSV decoders.")
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
    parser.add_argument("--shots", type=int, default=120, help="Shots per run.")
    parser.add_argument("--repeats", type=int, default=1, help="Repeated runs per point.")
    parser.add_argument("--seed", type=int, default=2026, help="Base seed.")
    parser.add_argument("--g-threshold", type=float, default=0.35, help="Adaptive threshold.")
    parser.add_argument("--adaptive-fast-mode", action="store_true", help="Use adaptive fast mode.")
    parser.add_argument(
        "--from-profile-artifact",
        type=str,
        default=None,
        help=(
            "Optional existing profile artifact path to materialize stress rows without rerunning decoding."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/week5_stress_test.json",
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
    if args.shots <= 0:
        raise ValueError("--shots must be > 0")
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")
    if not (0.0 <= float(args.p_phys) <= 1.0):
        raise ValueError("--p-phys must be in [0,1]")
    if not (0.0 <= float(args.g_threshold) <= 1.0):
        raise ValueError("--g-threshold must be in [0,1]")

    profile_artifact_path = None
    if args.from_profile_artifact is not None:
        profile_artifact_path = Path(args.from_profile_artifact)
        if not profile_artifact_path.exists():
            raise FileNotFoundError(profile_artifact_path)

    if profile_artifact_path is not None:
        rows = _rows_from_profile_artifact(
            profile_artifact=profile_artifact_path,
            distances=distances,
            decoders=decoders,
        )
        source_mode = "profile_artifact"
    else:
        rows = _rows_from_direct_benchmark(
            distances=distances,
            decoders=decoders,
            rounds_mode=str(args.rounds_mode),
            rounds_fixed=int(args.rounds),
            noise_model=str(args.noise_model),
            p_phys=float(args.p_phys),
            logical_basis=str(args.logical_basis),
            shots=int(args.shots),
            repeats=int(args.repeats),
            seed=int(args.seed),
            g_threshold=float(args.g_threshold),
            adaptive_fast_mode=bool(args.adaptive_fast_mode),
        )
        source_mode = "direct_benchmark"

    n_pass = int(sum(1 for r in rows if bool(r.get("pass", False))))
    report = {
        "schema_version": "week5_stress_test.v1",
        "metadata": {
            "report_name": "week5_stress_test",
            "timestamp_utc": utc_now_iso(),
        },
        "provenance": {
            "generator_script": "scripts/run_week5_stress_test.py",
            "source_mode": source_mode,
            "profile_artifact_input": None
            if profile_artifact_path is None
            else str(profile_artifact_path),
            "memory_measurement_mode": "tracemalloc_python_allocations",
            "timing_measurement_mode": "core_decode_time",
            "memory_limitations": (
                "tracemalloc tracks Python allocator behavior; it is not full process RSS."
            ),
            "distances_covered": [int(x) for x in distances],
            "decoders_covered": list(decoders),
        },
        "config": {
            "distances": [int(x) for x in distances],
            "decoders": list(decoders),
            "rounds_mode": str(args.rounds_mode),
            "rounds": int(args.rounds),
            "noise_model": str(args.noise_model),
            "p_phys": float(args.p_phys),
            "logical_basis": str(args.logical_basis),
            "shots": int(args.shots),
            "repeats": int(args.repeats),
            "seed": int(args.seed),
            "g_threshold": float(args.g_threshold),
            "adaptive_fast_mode": bool(args.adaptive_fast_mode),
        },
        "rows": rows,
        "summary": {
            "total_rows": int(len(rows)),
            "pass_rows": int(n_pass),
            "all_pass": bool(n_pass == len(rows)),
        },
    }

    saved = save_json(report, Path(args.output))
    print("\n=== Week 5 Stress Test ===")
    print(
        f"distances={report['config']['distances']} | decoders={report['config']['decoders']} | "
        f"source_mode={source_mode}"
    )
    print(
        f"rows={report['summary']['total_rows']} | pass={report['summary']['pass_rows']} | "
        f"all_pass={report['summary']['all_pass']}"
    )
    print(f"JSON saved at: {saved}")


if __name__ == "__main__":
    main()
