from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


DECODERS_ALLOWED = {"mwpm", "uf", "bm", "adaptive"}
NOISE_ALLOWED = {
    "depolarizing",
    "biased_eta10",
    "biased_eta100",
    "biased_eta500",
    "circuit_level",
    "correlated",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_csv_ints(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty int list.")
    return vals


def parse_csv_floats(raw: str) -> List[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty float list.")
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


def save_json_atomic(payload: Dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Week 3 local/cluster campaign manifest.")
    p.add_argument("--distances", type=str, default="3,5,7,9,11,13")
    p.add_argument("--p-values", type=str, default="0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.03")
    p.add_argument("--decoders", type=str, default="mwpm,uf,bm,adaptive")
    p.add_argument(
        "--noise-models",
        type=str,
        default="depolarizing,biased_eta10,biased_eta100,biased_eta500,circuit_level,correlated",
    )
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--shots", type=int, default=300)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--g-threshold", type=float, default=0.35)
    p.add_argument(
        "--threshold-method",
        type=str,
        choices=["crossing", "crossing_then_fit", "fit_only"],
        default="crossing_then_fit",
    )
    p.add_argument("--checkpoint-every", type=int, default=24)
    p.add_argument("--logical-basis", type=str, default="x", choices=["x", "z"])
    p.add_argument("--adaptive-fast-mode", action="store_true")
    p.add_argument("--output-dir", type=str, default="results/week3_campaign_chunks")
    p.add_argument("--output", type=str, default="manifests/week3_campaign_manifest.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    distances = parse_csv_ints(args.distances)
    p_values = parse_csv_floats(args.p_values)
    decoders = parse_decoders_csv(args.decoders)
    noise_models = parse_noise_csv(args.noise_models)

    if any(d < 3 or d % 2 == 0 for d in distances):
        raise ValueError("--distances must contain odd values >= 3.")
    if args.rounds <= 0:
        raise ValueError("--rounds must be > 0.")
    if args.shots <= 0:
        raise ValueError("--shots must be > 0.")
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0.")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be >= 0.")
    if not (0.0 <= float(args.g_threshold) <= 1.0):
        raise ValueError("--g-threshold must be in [0,1].")
    if any((p < 0.0 or p > 1.0) for p in p_values):
        raise ValueError("--p-values entries must be in [0,1].")

    p_values_csv = ",".join(f"{x:.12g}" for x in p_values)
    jobs: List[Dict[str, Any]] = []
    out_dir = Path(args.output_dir)

    job_idx = 0
    for d in distances:
        for noise in noise_models:
            for decoder in decoders:
                job_idx += 1
                seed = int(args.seed + job_idx * 1000003 + d * 97)
                chunk_output = out_dir / f"week3_scan_job{job_idx:04d}_{decoder}_{noise}_d{d}.json"
                cmd: List[str] = [
                    sys.executable,
                    "-m",
                    "scripts.run_week3_person2_threshold_scan",
                    "--distances",
                    str(int(d)),
                    "--rounds",
                    str(int(args.rounds)),
                    "--p-values",
                    p_values_csv,
                    "--decoders",
                    str(decoder),
                    "--noise-models",
                    str(noise),
                    "--shots",
                    str(int(args.shots)),
                    "--repeats",
                    str(int(args.repeats)),
                    "--seed",
                    str(seed),
                    "--g-threshold",
                    str(float(args.g_threshold)),
                    "--threshold-method",
                    str(args.threshold_method),
                    "--checkpoint-every",
                    str(int(args.checkpoint_every)),
                    "--logical-basis",
                    str(args.logical_basis),
                    "--output",
                    str(chunk_output),
                ]
                if bool(args.adaptive_fast_mode):
                    cmd.append("--adaptive-fast-mode")

                jobs.append(
                    {
                        "job_id": f"week3_job_{job_idx:04d}",
                        "job_key": f"{decoder}|{noise}|d={int(d)}",
                        "decoder": str(decoder),
                        "noise_model": str(noise),
                        "distance": int(d),
                        "rounds": int(args.rounds),
                        "p_values": [float(x) for x in p_values],
                        "shots": int(args.shots),
                        "repeats": int(args.repeats),
                        "seed": int(seed),
                        "g_threshold": float(args.g_threshold),
                        "threshold_method": str(args.threshold_method),
                        "logical_basis": str(args.logical_basis),
                        "adaptive_fast_mode": bool(args.adaptive_fast_mode),
                        "checkpoint_every": int(args.checkpoint_every),
                        "output": str(chunk_output),
                        "command": cmd,
                    }
                )

    payload = {
        "metadata": {
            "report_name": "week3_campaign_manifest",
            "schema_version": "week3_manifest_v1",
            "timestamp_utc": utc_now_iso(),
            "num_jobs": int(len(jobs)),
            "expected_total_points": int(len(jobs) * len(p_values)),
            "covered_distances": [int(x) for x in distances],
            "covered_decoders": list(decoders),
            "covered_noise_models": list(noise_models),
        },
        "campaign_config": {
            "distances": [int(x) for x in distances],
            "p_values": [float(x) for x in p_values],
            "decoders": list(decoders),
            "noise_models": list(noise_models),
            "shots": int(args.shots),
            "repeats": int(args.repeats),
            "rounds": int(args.rounds),
            "seed": int(args.seed),
            "g_threshold": float(args.g_threshold),
            "threshold_method": str(args.threshold_method),
            "logical_basis": str(args.logical_basis),
            "adaptive_fast_mode": bool(args.adaptive_fast_mode),
            "checkpoint_every": int(args.checkpoint_every),
            "output_dir": str(out_dir),
        },
        "jobs": jobs,
    }

    saved = save_json_atomic(payload, Path(args.output))
    print(f"Manifest saved at: {saved}")
    print(f"Jobs: {len(jobs)} | expected points: {len(jobs) * len(p_values)}")


if __name__ == "__main__":
    main()
