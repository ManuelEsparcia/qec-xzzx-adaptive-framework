from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
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

Q1_FULL_DISTANCES = [3, 5, 7, 9, 11, 13]
Q1_FULL_DECODERS = ["mwpm", "uf", "bm", "adaptive"]
Q1_FULL_NOISE_MODELS = [
    "depolarizing",
    "biased_eta10",
    "biased_eta100",
    "biased_eta500",
    "circuit_level",
    "correlated",
]

POINT_KEY_P_DIGITS = 12


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


def linspace_closed(start: float, stop: float, n: int) -> List[float]:
    if n <= 1:
        return [float(start)]
    step = (float(stop) - float(start)) / float(n - 1)
    return [float(start + i * step) for i in range(n)]


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


def _round_p(p_phys: float) -> float:
    return float(round(float(p_phys), POINT_KEY_P_DIGITS))


def _point_base_key(
    *,
    decoder: str,
    noise_model: str,
    distance: int,
    rounds: int,
    p_phys: float,
    shots: int,
) -> Tuple[str, str, int, int, float, int]:
    return (
        str(decoder),
        str(noise_model),
        int(distance),
        int(rounds),
        _round_p(float(p_phys)),
        int(shots),
    )


def _base_key_from_point(point: ScanPoint) -> Tuple[str, str, int, int, float, int]:
    return _point_base_key(
        decoder=point.decoder,
        noise_model=point.noise_model,
        distance=point.distance,
        rounds=point.rounds,
        p_phys=point.p_phys,
        shots=point.shots,
    )


def _base_key_from_row(row: Dict[str, Any]) -> Optional[Tuple[str, str, int, int, float, int]]:
    try:
        return _point_base_key(
            decoder=str(row["decoder"]),
            noise_model=str(row["noise_model"]),
            distance=int(row["distance"]),
            rounds=int(row["rounds"]),
            p_phys=float(row["p_phys"]),
            shots=int(row["shots"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _repeat_key(base_key: Tuple[str, str, int, int, float, int], repeat_index: int) -> Tuple[Any, ...]:
    return (*base_key, int(repeat_index))


def _canonical_repeat_entry(
    *,
    repeat_index: int,
    seed: int,
    error_rate: float,
    avg_decode_time_sec: float,
    switch_rate: float,
) -> Dict[str, Any]:
    return {
        "repeat_index": int(repeat_index),
        "seed": int(seed),
        "error_rate": float(error_rate),
        "avg_decode_time_sec": float(avg_decode_time_sec),
        "switch_rate": float(switch_rate),
    }


def _extract_repeat_map_from_row(row: Dict[str, Any], *, requested_repeats: int) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    raw_runs = row.get("repeat_runs", None)
    if isinstance(raw_runs, list):
        for rr in raw_runs:
            if not isinstance(rr, dict):
                continue
            try:
                idx = int(rr["repeat_index"])
                if idx < 0:
                    continue
                out[idx] = _canonical_repeat_entry(
                    repeat_index=idx,
                    seed=int(rr["seed"]),
                    error_rate=float(rr["error_rate"]),
                    avg_decode_time_sec=float(rr["avg_decode_time_sec"]),
                    switch_rate=float(rr.get("switch_rate", 0.0)),
                )
            except (KeyError, TypeError, ValueError):
                continue

    # Backward compatibility for old schema rows without repeat_runs.
    if (not out) and int(requested_repeats) == 1 and str(row.get("status", "")).lower() == "ok":
        try:
            out[0] = _canonical_repeat_entry(
                repeat_index=0,
                seed=int(row.get("seed", 0)),
                error_rate=float(row["error_rate"]),
                avg_decode_time_sec=float(row["avg_decode_time_sec"]),
                switch_rate=float(row.get("switch_rate", 0.0)),
            )
        except (KeyError, TypeError, ValueError):
            pass

    return out


def _has_all_repeats(row: Dict[str, Any], *, requested_repeats: int) -> bool:
    rr_map = _extract_repeat_map_from_row(row, requested_repeats=requested_repeats)
    return all(i in rr_map for i in range(int(requested_repeats)))


def _row_sort_key(row: Dict[str, Any]) -> Tuple[int, str, str, float]:
    return (
        int(row.get("distance", 0)),
        str(row.get("noise_model", "")),
        str(row.get("decoder", "")),
        _round_p(float(row.get("p_phys", 0.0))),
    )


def _safe_speedup(ref: float, cand: float) -> float:
    if cand <= 0:
        return float("nan")
    return float(ref / cand)


def _summary_stats(values: Sequence[float], *, clamp_01: bool = False) -> Dict[str, float]:
    if not values:
        raise ValueError("values must be non-empty")
    vals = [float(v) for v in values]
    n = len(vals)
    m = float(mean(vals))
    sd = float(stdev(vals)) if n > 1 else 0.0
    ci_hw = float(1.96 * sd / math.sqrt(n)) if n > 1 else 0.0
    lo = float(m - ci_hw)
    hi = float(m + ci_hw)
    if clamp_01:
        lo = float(max(0.0, lo))
        hi = float(min(1.0, hi))
    return {
        "n": float(n),
        "mean": m,
        "std": sd,
        "ci95_half_width": ci_hw,
        "ci95_low": lo,
        "ci95_high": hi,
    }


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
            time_metric="core",
        )
        return {
            "error_rate": float(res["error_rate_adaptive"]),
            "avg_decode_time_sec": float(res["avg_decode_time_adaptive"]),
            "switch_rate": float(res["switch_rate"]),
            "fast_mode": bool(res.get("fast_mode", adaptive_fast_mode)),
        }

    raise ValueError(f"Unknown decoder: {decoder}")


def run_point_once(
    point: ScanPoint,
    *,
    logical_basis: str,
    g_threshold: float,
    adaptive_fast_mode: bool,
    seed_override: Optional[int] = None,
) -> Dict[str, Any]:
    run_seed = int(point.seed if seed_override is None else seed_override)
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
        seed=run_seed,
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
        "seed": run_seed,
        "error_rate": float(metrics["error_rate"]),
        "avg_decode_time_sec": float(metrics["avg_decode_time_sec"]),
        "switch_rate": float(metrics.get("switch_rate", 0.0)),
        "status": "ok",
        **({"backend_info": metrics["backend_info"]} if "backend_info" in metrics else {}),
        **({"fast_mode": metrics["fast_mode"]} if "fast_mode" in metrics else {}),
    }


def _build_point_row_from_repeat_runs(
    *,
    point: ScanPoint,
    repeats: int,
    repeat_runs: Sequence[Dict[str, Any]],
    backend_info: Optional[Dict[str, Any]] = None,
    fast_mode: Optional[bool] = None,
) -> Dict[str, Any]:
    repeat_runs_sorted = sorted(
        (
            _canonical_repeat_entry(
                repeat_index=int(rr["repeat_index"]),
                seed=int(rr["seed"]),
                error_rate=float(rr["error_rate"]),
                avg_decode_time_sec=float(rr["avg_decode_time_sec"]),
                switch_rate=float(rr.get("switch_rate", 0.0)),
            )
            for rr in repeat_runs
        ),
        key=lambda x: int(x["repeat_index"]),
    )
    if len(repeat_runs_sorted) != int(repeats):
        raise ValueError(
            f"Repeat-run count mismatch. Expected {repeats}, got {len(repeat_runs_sorted)}."
        )
    if [int(x["repeat_index"]) for x in repeat_runs_sorted] != list(range(int(repeats))):
        raise ValueError("repeat_runs must contain exactly repeat_index=0..repeats-1.")

    er_stats = _summary_stats([float(x["error_rate"]) for x in repeat_runs_sorted], clamp_01=True)
    t_stats = _summary_stats(
        [float(x["avg_decode_time_sec"]) for x in repeat_runs_sorted],
        clamp_01=False,
    )
    sw_stats = _summary_stats(
        [float(x.get("switch_rate", 0.0)) for x in repeat_runs_sorted],
        clamp_01=True,
    )

    out = {
        "decoder": point.decoder,
        "noise_model": point.noise_model,
        "distance": int(point.distance),
        "rounds": int(point.rounds),
        "p_phys": float(point.p_phys),
        "shots": int(point.shots),
        "seed": int(point.seed),
        "repeats": int(repeats),
        "repeat_seeds": [int(x["seed"]) for x in repeat_runs_sorted],
        # Keep legacy keys as means for backward compatibility.
        "error_rate": float(er_stats["mean"]),
        "avg_decode_time_sec": float(t_stats["mean"]),
        "switch_rate": float(sw_stats["mean"]),
        # Uncertainty fields for error bars / CIs.
        "error_rate_std": float(er_stats["std"]),
        "error_rate_ci95_half_width": float(er_stats["ci95_half_width"]),
        "error_rate_ci95_low": float(er_stats["ci95_low"]),
        "error_rate_ci95_high": float(er_stats["ci95_high"]),
        "avg_decode_time_sec_std": float(t_stats["std"]),
        "avg_decode_time_sec_ci95_half_width": float(t_stats["ci95_half_width"]),
        "avg_decode_time_sec_ci95_low": float(t_stats["ci95_low"]),
        "avg_decode_time_sec_ci95_high": float(t_stats["ci95_high"]),
        "switch_rate_std": float(sw_stats["std"]),
        "switch_rate_ci95_half_width": float(sw_stats["ci95_half_width"]),
        "switch_rate_ci95_low": float(sw_stats["ci95_low"]),
        "switch_rate_ci95_high": float(sw_stats["ci95_high"]),
        "repeat_runs": repeat_runs_sorted,
        "status": "ok",
    }
    if backend_info is not None:
        out["backend_info"] = backend_info
    if fast_mode is not None:
        out["fast_mode"] = bool(fast_mode)
    return out


def run_point(
    point: ScanPoint,
    *,
    logical_basis: str,
    g_threshold: float,
    adaptive_fast_mode: bool,
    repeats: int,
    existing_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeats must be int > 0")

    repeat_map: Dict[int, Dict[str, Any]] = {}
    backend_info: Optional[Dict[str, Any]] = None
    fast_mode: Optional[bool] = None

    if isinstance(existing_row, dict):
        repeat_map.update(_extract_repeat_map_from_row(existing_row, requested_repeats=int(repeats)))
        if "backend_info" in existing_row and isinstance(existing_row["backend_info"], dict):
            backend_info = existing_row["backend_info"]
        if "fast_mode" in existing_row:
            fast_mode = bool(existing_row["fast_mode"])

    for r_idx in range(int(repeats)):
        if r_idx in repeat_map:
            continue
        run_seed = int(point.seed + r_idx * 10007)
        out = run_point_once(
            point,
            logical_basis=logical_basis,
            g_threshold=g_threshold,
            adaptive_fast_mode=adaptive_fast_mode,
            seed_override=run_seed,
        )
        repeat_map[r_idx] = _canonical_repeat_entry(
            repeat_index=r_idx,
            seed=int(out["seed"]),
            error_rate=float(out["error_rate"]),
            avg_decode_time_sec=float(out["avg_decode_time_sec"]),
            switch_rate=float(out.get("switch_rate", 0.0)),
        )
        if backend_info is None and "backend_info" in out and isinstance(out["backend_info"], dict):
            backend_info = out["backend_info"]
        if fast_mode is None and "fast_mode" in out:
            fast_mode = bool(out["fast_mode"])

    repeat_runs = [repeat_map[i] for i in sorted(repeat_map.keys()) if i < int(repeats)]
    return _build_point_row_from_repeat_runs(
        point=point,
        repeats=int(repeats),
        repeat_runs=repeat_runs,
        backend_info=backend_info,
        fast_mode=fast_mode,
    )


def _preferred_repeat_entry(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    # Deterministic tie-breaker: prefer lower seed, then lower decode time.
    sa = int(a.get("seed", 0))
    sb = int(b.get("seed", 0))
    if sa != sb:
        return a if sa < sb else b
    ta = float(a.get("avg_decode_time_sec", float("inf")))
    tb = float(b.get("avg_decode_time_sec", float("inf")))
    return a if ta <= tb else b


def _merge_rows_same_key(
    row_a: Dict[str, Any],
    row_b: Dict[str, Any],
    *,
    requested_repeats: int,
) -> Dict[str, Any]:
    base_a = _base_key_from_row(row_a)
    base_b = _base_key_from_row(row_b)
    if base_a is None or base_b is None or base_a != base_b:
        return row_a

    repeat_map: Dict[int, Dict[str, Any]] = {}
    for src in (row_a, row_b):
        src_map = _extract_repeat_map_from_row(src, requested_repeats=requested_repeats)
        for idx, rr in src_map.items():
            if idx not in repeat_map:
                repeat_map[idx] = rr
            else:
                repeat_map[idx] = _preferred_repeat_entry(repeat_map[idx], rr)

    point = ScanPoint(
        decoder=str(row_a["decoder"]),
        noise_model=str(row_a["noise_model"]),
        distance=int(row_a["distance"]),
        rounds=int(row_a["rounds"]),
        p_phys=float(row_a["p_phys"]),
        shots=int(row_a["shots"]),
        seed=int(min(int(row_a.get("seed", 0)), int(row_b.get("seed", 0)))),
    )

    backend_info = None
    for src in (row_a, row_b):
        if isinstance(src.get("backend_info", None), dict):
            backend_info = src["backend_info"]
            break
    fast_mode = row_a.get("fast_mode", row_b.get("fast_mode", None))

    repeat_runs = [repeat_map[i] for i in sorted(repeat_map.keys()) if i < int(requested_repeats)]
    if len(repeat_runs) < int(requested_repeats):
        # Keep the richest row if not enough repeat data.
        ra = len(_extract_repeat_map_from_row(row_a, requested_repeats=requested_repeats))
        rb = len(_extract_repeat_map_from_row(row_b, requested_repeats=requested_repeats))
        return row_a if ra >= rb else row_b

    return _build_point_row_from_repeat_runs(
        point=point,
        repeats=int(requested_repeats),
        repeat_runs=repeat_runs,
        backend_info=backend_info,
        fast_mode=(None if fast_mode is None else bool(fast_mode)),
    )


def load_rows_for_resume(
    *,
    resume_path: Path,
    requested_repeats: int,
) -> Dict[Tuple[str, str, int, int, float, int], Dict[str, Any]]:
    if not resume_path.exists():
        raise FileNotFoundError(f"--resume-from file does not exist: {resume_path}")
    with resume_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid resume payload in {resume_path}. Expected JSON object.")
    rows = payload.get("points", None)
    if not isinstance(rows, list):
        raise ValueError(f"Invalid resume payload in {resume_path}. Missing list at key 'points'.")

    merged: Dict[Tuple[str, str, int, int, float, int], Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = _base_key_from_row(row)
        if key is None:
            continue
        if key not in merged:
            merged[key] = row
        else:
            merged[key] = _merge_rows_same_key(
                merged[key],
                row,
                requested_repeats=int(requested_repeats),
            )
    return merged


def _estimate_threshold_for_pair(
    p_values: List[float],
    er_small: List[float],
    er_large: List[float],
    *,
    method_preference: str,
) -> Dict[str, Any]:
    diffs = [b - a for a, b in zip(er_small, er_large)]
    if len(diffs) != len(p_values):
        raise ValueError("p_values and error-rate arrays must have the same length.")
    if not diffs:
        raise ValueError("Need at least one point to estimate thresholds.")

    def _linear_fit_root() -> Dict[str, Any]:
        n = len(p_values)
        if n < 2:
            return {
                "valid": False,
                "slope": None,
                "intercept": None,
                "root": None,
                "rmse": None,
                "in_scan_range": False,
            }
        mx = float(mean(p_values))
        my = float(mean(diffs))
        var_x = float(sum((x - mx) ** 2 for x in p_values))
        if var_x <= 0.0:
            return {
                "valid": False,
                "slope": None,
                "intercept": None,
                "root": None,
                "rmse": None,
                "in_scan_range": False,
            }
        cov_xy = float(sum((x - mx) * (y - my) for x, y in zip(p_values, diffs)))
        slope = float(cov_xy / var_x)
        intercept = float(my - slope * mx)
        if abs(slope) <= 1e-15:
            return {
                "valid": False,
                "slope": slope,
                "intercept": intercept,
                "root": None,
                "rmse": None,
                "in_scan_range": False,
            }
        root = float(-intercept / slope)
        residuals = [float(intercept + slope * x - y) for x, y in zip(p_values, diffs)]
        rmse = float(math.sqrt(sum(r * r for r in residuals) / float(n)))
        return {
            "valid": True,
            "slope": slope,
            "intercept": intercept,
            "root": root,
            "rmse": rmse,
            "in_scan_range": bool(min(p_values) <= root <= max(p_values)),
        }

    if method_preference not in {"crossing", "crossing_then_fit", "fit_only"}:
        raise ValueError(
            "method_preference must be one of: crossing, crossing_then_fit, fit_only"
        )

    fit = _linear_fit_root()
    min_abs_diff = float(min(abs(d) for d in diffs))
    j_min_abs = int(min(range(len(p_values)), key=lambda k: abs(diffs[k])))

    for i in range(len(p_values) - 1):
        d0 = diffs[i]
        d1 = diffs[i + 1]
        if d0 == 0.0:
            return {
                "method": "exact_point",
                "fit_method": "piecewise_linear_diff_crossing",
                "p_threshold_estimate": float(p_values[i]),
                "index_pair": [i, i],
                "fallback_used": False,
                "fallback_reason": "",
                "crossing_interval": [float(p_values[i]), float(p_values[i])],
                "quality_level": "high",
                "threshold_quality": {
                    "num_p_points": int(len(p_values)),
                    "min_abs_diff": min_abs_diff,
                    "selected_abs_diff": float(abs(d0)),
                    "diff_slope_estimate": fit.get("slope"),
                    "fit_rmse": fit.get("rmse"),
                    "fit_in_scan_range": bool(fit.get("in_scan_range", False)),
                },
            }
        if d0 * d1 < 0:
            p0, p1 = p_values[i], p_values[i + 1]
            # linear interpolation in diff-space
            frac = abs(d0) / (abs(d0) + abs(d1))
            p_cross = p0 + (p1 - p0) * frac
            return {
                "method": "linear_crossing",
                "fit_method": "piecewise_linear_diff_crossing",
                "p_threshold_estimate": float(p_cross),
                "index_pair": [i, i + 1],
                "fallback_used": False,
                "fallback_reason": "",
                "crossing_interval": [float(p0), float(p1)],
                "quality_level": "high",
                "threshold_quality": {
                    "num_p_points": int(len(p_values)),
                    "min_abs_diff": min_abs_diff,
                    "selected_abs_diff": float(min(abs(d0), abs(d1))),
                    "diff_slope_estimate": fit.get("slope"),
                    "fit_rmse": fit.get("rmse"),
                    "fit_in_scan_range": bool(fit.get("in_scan_range", False)),
                },
            }

    can_use_fit = bool(fit.get("valid", False) and fit.get("in_scan_range", False))
    use_fit = method_preference in {"crossing_then_fit", "fit_only"} and can_use_fit
    if use_fit:
        assert fit["root"] is not None
        root = float(fit["root"])
        j = int(min(range(len(p_values)), key=lambda k: abs(float(p_values[k]) - root)))
        return {
            "method": "linear_fit_root",
            "fit_method": "linear_least_squares",
            "p_threshold_estimate": root,
            "index_pair": [j, j],
            "fallback_used": bool(method_preference == "crossing_then_fit"),
            "fallback_reason": (
                "no_crossing_detected" if method_preference == "crossing_then_fit" else ""
            ),
            "crossing_interval": None,
            "quality_level": "medium",
            "threshold_quality": {
                "num_p_points": int(len(p_values)),
                "min_abs_diff": min_abs_diff,
                "selected_abs_diff": float(abs(diffs[j])),
                "diff_slope_estimate": fit.get("slope"),
                "fit_rmse": fit.get("rmse"),
                "fit_in_scan_range": bool(fit.get("in_scan_range", False)),
            },
        }

    # Explicit fallback: nearest absolute difference at scanned p.
    j = int(j_min_abs)
    return {
        "method": "nearest_abs_diff",
        "fit_method": "none",
        "p_threshold_estimate": float(p_values[j]),
        "index_pair": [j, j],
        "fallback_used": True,
        "fallback_reason": (
            "fit_unavailable_or_disabled"
            if method_preference in {"crossing_then_fit", "fit_only"}
            else "no_crossing_detected"
        ),
        "crossing_interval": None,
        "quality_level": "low",
        "threshold_quality": {
            "num_p_points": int(len(p_values)),
            "min_abs_diff": min_abs_diff,
            "selected_abs_diff": float(abs(diffs[j])),
            "diff_slope_estimate": fit.get("slope"),
            "fit_rmse": fit.get("rmse"),
            "fit_in_scan_range": bool(fit.get("in_scan_range", False)),
        },
    }


def _build_threshold_estimate_row(
    *,
    decoder: str,
    noise_model: str,
    d_small: int,
    d_large: int,
    p_common: List[float],
    er_small: List[float],
    er_large: List[float],
    pair_scope: str,
    method_preference: str,
) -> Dict[str, Any]:
    est = _estimate_threshold_for_pair(
        p_common,
        er_small,
        er_large,
        method_preference=method_preference,
    )
    method = str(est["method"])
    return {
        "decoder": decoder,
        "noise_model": noise_model,
        "distance_pair": [int(d_small), int(d_large)],
        "distance_gap": int(d_large - d_small),
        "pair_scope": pair_scope,
        "crossing_detected": bool(method in {"exact_point", "linear_crossing"}),
        "threshold_method_preference": str(method_preference),
        "p_values": p_common,
        "error_rate_small_distance": er_small,
        "error_rate_large_distance": er_large,
        **est,
    }


def estimate_thresholds(
    rows: Sequence[Dict[str, Any]],
    *,
    method_preference: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (str(r["decoder"]), str(r["noise_model"]))
        grouped.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    for (decoder, noise_model), items in grouped.items():
        distances = sorted({int(r["distance"]) for r in items})
        if len(distances) < 2:
            continue

        def _er_at(arr: Sequence[Dict[str, Any]], p: float) -> float:
            vals = [float(x["error_rate"]) for x in arr if math.isclose(float(x["p_phys"]), p, rel_tol=0.0, abs_tol=1e-12)]
            return float(mean(vals)) if vals else float("nan")

        by_distance: Dict[int, List[Dict[str, Any]]] = {
            d: [r for r in items if int(r["distance"]) == d] for d in distances
        }

        pair_candidates: List[Tuple[int, int, str]] = []
        for i in range(len(distances) - 1):
            pair_candidates.append((int(distances[i]), int(distances[i + 1]), "adjacent"))
        if len(distances) >= 2:
            pair_candidates.append((int(distances[0]), int(distances[-1]), "extreme"))

        seen_pairs: set[Tuple[int, int, str]] = set()
        for d_small, d_large, pair_scope in pair_candidates:
            pair_key = (int(d_small), int(d_large), str(pair_scope))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            small = by_distance[int(d_small)]
            large = by_distance[int(d_large)]
            p_common = sorted(
                {float(r["p_phys"]) for r in small}.intersection(float(r["p_phys"]) for r in large)
            )
            if len(p_common) < 2:
                continue

            er_small = [_er_at(small, p) for p in p_common]
            er_large = [_er_at(large, p) for p in p_common]
            if any(not math.isfinite(x) for x in er_small + er_large):
                continue

            out.append(
                _build_threshold_estimate_row(
                    decoder=decoder,
                    noise_model=noise_model,
                    d_small=int(d_small),
                    d_large=int(d_large),
                    p_common=p_common,
                    er_small=er_small,
                    er_large=er_large,
                    pair_scope=str(pair_scope),
                    method_preference=method_preference,
                )
            )
    out.sort(
        key=lambda x: (
            str(x["decoder"]),
            str(x["noise_model"]),
            0 if str(x.get("pair_scope", "")) == "adjacent" else 1,
            int(x["distance_pair"][0]),
            int(x["distance_pair"][1]),
        )
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
    pair_distance_summary: List[Dict[str, Any]] = []
    for (decoder, noise_model, distance), arr in sorted(by_triplet.items()):
        pair_distance_summary.append(
            {
                "decoder": decoder,
                "noise_model": noise_model,
                "distance": int(distance),
                "num_points": len(arr),
                "mean_error_rate": float(mean(float(x["error_rate"]) for x in arr)),
                "mean_avg_decode_time_sec": float(mean(float(x["avg_decode_time_sec"]) for x in arr)),
                "mean_switch_rate": float(mean(float(x.get("switch_rate", 0.0)) for x in arr)),
            }
        )
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
        "pair_summary_note": (
            "Averages over all distances and p-values for each (decoder, noise_model); "
            "use pair_distance_summary or threshold_estimates for scientific comparisons."
        ),
        "pair_summary": pair_summary,
        "pair_distance_summary": pair_distance_summary,
        "pareto_reference_points": pareto_refs,
    }


def save_json(data: Dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_path)
    return output_path


def build_report(
    *,
    rows: Sequence[Dict[str, Any]],
    distances: Sequence[int],
    p_values: Sequence[float],
    decoders: Sequence[str],
    noise_models: Sequence[str],
    shots: int,
    repeats: int,
    rounds: int,
    seed: int,
    g_threshold: float,
    adaptive_fast_mode: bool,
    logical_basis: str,
    threshold_method: str,
    resume_from: Optional[str],
    partial: bool,
) -> Dict[str, Any]:
    point_rows = list(rows)
    covered_distances = sorted({int(x["distance"]) for x in point_rows}) if point_rows else []
    covered_decoders = sorted({str(x["decoder"]) for x in point_rows}) if point_rows else []
    covered_noise_models = (
        sorted({str(x["noise_model"]) for x in point_rows}) if point_rows else []
    )
    expected_points = int(len(distances) * len(p_values) * len(decoders) * len(noise_models))
    return {
        "metadata": {
            "report_name": "week3_person2_threshold_scan",
            "schema_version": "week3_scan_v2",
            "timestamp_utc": utc_now_iso(),
            "partial": bool(partial),
            "resumed_from": (str(resume_from) if resume_from else None),
            "grid_expected_points": expected_points,
            "grid_completed_points": int(len(point_rows)),
            "grid_completion_ratio": (
                float(len(point_rows) / expected_points) if expected_points > 0 else 0.0
            ),
            "covered_distances": covered_distances,
            "covered_decoders": covered_decoders,
            "covered_noise_models": covered_noise_models,
        },
        "config": {
            "distances": [int(x) for x in distances],
            "p_values": [float(x) for x in p_values],
            "decoders": list(decoders),
            "noise_models": list(noise_models),
            "shots": int(shots),
            "repeats": int(repeats),
            "rounds": int(rounds),
            "seed": int(seed),
            "g_threshold": float(g_threshold),
            "adaptive_fast_mode": bool(adaptive_fast_mode),
            "adaptive_benchmark_time_metric": "core",
            "logical_basis": logical_basis,
            "threshold_method": str(threshold_method),
        },
        "points": point_rows,
        "aggregates": aggregate_rows(point_rows),
        "threshold_estimates": estimate_thresholds(
            point_rows,
            method_preference=str(threshold_method),
        ),
    }


def print_summary(report: Dict[str, Any]) -> None:
    print("\n=== Week 3 Person 2 - Threshold Scan ===")
    agg = report.get("aggregates", {})
    print(f"points: {agg.get('num_points', 0)}")
    note = agg.get("pair_summary_note", None)
    if note:
        print(f"note: {note}")
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
    parser.add_argument(
        "--distances",
        type=str,
        default="3,5,7,9,11,13",
        help="CSV of odd distances.",
    )
    parser.add_argument("--rounds", type=int, default=3, help="Rounds per circuit.")
    parser.add_argument(
        "--p-values",
        type=str,
        default="0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.03",
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
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeated runs per point for CI/error bars.")
    parser.add_argument("--seed", type=int, default=2026, help="Base seed.")
    parser.add_argument("--g-threshold", type=float, default=0.35, help="Adaptive threshold.")
    parser.add_argument(
        "--threshold-method",
        type=str,
        choices=["crossing", "crossing_then_fit", "fit_only"],
        default="crossing_then_fit",
        help="Threshold estimator policy for distance-curve crossings.",
    )
    parser.add_argument(
        "--adaptive-fast-mode",
        action="store_true",
        help="Use Adaptive fast_mode=True during adaptive benchmarks.",
    )
    parser.add_argument("--logical-basis", type=str, default="x", choices=["x", "z"], help="Logical basis.")
    parser.add_argument("--checkpoint-every", type=int, default=24, help="Save partial JSON every N points.")
    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Optional JSON report to resume from. Completed points are skipped idempotently.",
    )
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
    threshold_method = str(args.threshold_method)
    resume_from = str(args.resume_from).strip()
    resume_path = Path(resume_from) if resume_from else None

    if any(d < 3 or d % 2 == 0 for d in distances):
        raise ValueError("--distances must contain odd values >= 3.")
    if args.rounds <= 0:
        raise ValueError("--rounds must be > 0")
    if args.shots <= 0:
        raise ValueError("--shots must be > 0")
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be >= 0")
    if not (0.0 <= args.g_threshold <= 1.0):
        raise ValueError("--g-threshold must be in [0,1]")
    if any((p < 0.0 or p > 1.0) for p in p_values):
        raise ValueError("--p-values entries must be in [0,1]")

    points: List[ScanPoint] = []
    expected_keys: set[Tuple[str, str, int, int, float, int]] = set()
    idx = 0
    for d in distances:
        for noise in noise_models:
            for decoder in decoders:
                for p in p_values:
                    idx += 1
                    point = ScanPoint(
                        decoder=str(decoder),
                        noise_model=str(noise),
                        distance=int(d),
                        rounds=int(args.rounds),
                        p_phys=float(p),
                        shots=int(args.shots),
                        seed=int(args.seed + idx * 104729 + int(d) * 1009),
                    )
                    points.append(point)
                    expected_keys.add(_base_key_from_point(point))

    rows_by_key: Dict[Tuple[str, str, int, int, float, int], Dict[str, Any]] = {}
    if resume_path is not None:
        loaded = load_rows_for_resume(
            resume_path=resume_path,
            requested_repeats=int(args.repeats),
        )
        # Keep only points that belong to current campaign grid.
        rows_by_key = {k: v for k, v in loaded.items() if k in expected_keys}
        print(
            f"Loaded resume rows: {len(loaded)} (usable in current grid: {len(rows_by_key)}) "
            f"from {resume_path}"
        )

    total = len(points)
    run_count = 0
    skip_count = 0
    for idx, point in enumerate(points, start=1):
        base_key = _base_key_from_point(point)
        existing = rows_by_key.get(base_key)
        if existing is not None and _has_all_repeats(existing, requested_repeats=int(args.repeats)):
            skip_count += 1
            if idx == 1 or idx % 25 == 0 or idx == total:
                print(
                    f"[{idx}/{total}] skip-complete | {point.decoder} | {point.noise_model} | "
                    f"d={point.distance} | p={point.p_phys:.6f}"
                )
        else:
            row = run_point(
                point,
                logical_basis=args.logical_basis,
                g_threshold=float(args.g_threshold),
                adaptive_fast_mode=bool(args.adaptive_fast_mode),
                repeats=int(args.repeats),
                existing_row=existing,
            )
            rows_by_key[base_key] = row
            run_count += 1
            print(
                f"[{idx}/{total}] run | {point.decoder} | {point.noise_model} | d={point.distance} | "
                f"p={point.p_phys:.6f} | ER={row['error_rate']:.6f}+/-{row['error_rate_ci95_half_width']:.6f} | "
                f"t={row['avg_decode_time_sec']:.6f}+/-{row['avg_decode_time_sec_ci95_half_width']:.6f}s"
            )

        if args.checkpoint_every > 0 and (idx % args.checkpoint_every == 0):
            rows_sorted = sorted(rows_by_key.values(), key=_row_sort_key)
            partial = build_report(
                rows=rows_sorted,
                distances=distances,
                p_values=p_values,
                decoders=decoders,
                noise_models=noise_models,
                shots=int(args.shots),
                repeats=int(args.repeats),
                rounds=int(args.rounds),
                seed=int(args.seed),
                g_threshold=float(args.g_threshold),
                adaptive_fast_mode=bool(args.adaptive_fast_mode),
                logical_basis=args.logical_basis,
                threshold_method=threshold_method,
                resume_from=(str(resume_path) if resume_path is not None else None),
                partial=True,
            )
            save_json(partial, Path(args.output))

    rows_sorted = sorted(rows_by_key.values(), key=_row_sort_key)
    report = build_report(
        rows=rows_sorted,
        distances=distances,
        p_values=p_values,
        decoders=decoders,
        noise_models=noise_models,
        shots=int(args.shots),
        repeats=int(args.repeats),
        rounds=int(args.rounds),
        seed=int(args.seed),
        g_threshold=float(args.g_threshold),
        adaptive_fast_mode=bool(args.adaptive_fast_mode),
        logical_basis=args.logical_basis,
        threshold_method=threshold_method,
        resume_from=(str(resume_path) if resume_path is not None else None),
        partial=False,
    )
    saved = save_json(report, Path(args.output))
    print_summary(report)
    print(f"execution summary: ran={run_count}, skipped={skip_count}, total={total}")
    print(f"\nJSON saved at: {saved}")


if __name__ == "__main__":
    main()
