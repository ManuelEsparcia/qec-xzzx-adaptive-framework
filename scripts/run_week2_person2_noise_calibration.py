# scripts/run_week2_person2_noise_calibration.py
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _ensure_project_root_on_path() -> None:
    """
    Allows running:
      - python scripts/run_week2_person2_noise_calibration.py
      - python -m scripts.run_week2_person2_noise_calibration
    without `src` import errors.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


_ensure_project_root_on_path()

# ---------------------------------------------------------------------------
# Config / constantes
# ---------------------------------------------------------------------------

ALLOWED_MODELS = {
    "depolarizing",
    "biased",
    "circuit_level",
    "phenomenological",
    "correlated",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pick(d: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _fmt_float(value: Any, ndigits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return str(value)


def _call_with_supported_kwargs(func: Any, kwargs: Dict[str, Any]) -> Any:
    """
    Call `func` while filtering kwargs based on its signature.
    """
    sig = inspect.signature(func)
    accepted = {}
    for name, param in sig.parameters.items():
        if name in kwargs:
            accepted[name] = kwargs[name]
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            # If **kwargs is accepted, pass everything.
            return func(**kwargs)
    return func(**accepted)


def _build_obj_if_class(cls: Any, values: Dict[str, Any]) -> Any:
    """
    Build a dataclass/object with keys accepted by __init__.
    If it fails, return the original dict.
    """
    if cls is None:
        return values
    try:
        sig = inspect.signature(cls)
        kwargs = {k: v for k, v in values.items() if k in sig.parameters}
        return cls(**kwargs)
    except Exception:
        return values


# ---------------------------------------------------------------------------
# Default cases and sweeps
# ---------------------------------------------------------------------------

def default_cases() -> List[Dict[str, Any]]:
    """
    Base cases used in weeks 1-2 of the roadmap.
    """
    return [
        {"case_name": "d3_r2_p0.005", "distance": 3, "rounds": 2, "p": 0.005},
        {"case_name": "d3_r3_p0.010", "distance": 3, "rounds": 3, "p": 0.010},
        {"case_name": "d5_r3_p0.010", "distance": 5, "rounds": 3, "p": 0.010},
    ]


def default_sweep_templates(fast: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Sweep templates by model.
    """
    if fast:
        return {
            "depolarizing": {
                "model_type": "depolarizing",
                "param_name": "p",
                "values": [0.001, 0.0025, 0.005],
                "base_params": {},
            },
            "biased": {
                "model_type": "biased",
                "param_name": "eta",
                "values": [2.0, 4.0, 8.0],
                "base_params": {"p": 0.01},
            },
            "circuit_level": {
                "model_type": "circuit_level",
                "param_name": "p",
                "values": [0.005, 0.01, 0.02],
                "base_params": {},
            },
            "phenomenological": {
                "model_type": "phenomenological",
                "param_name": "p",
                "values": [0.001, 0.0025, 0.005],
                "base_params": {},
            },
            "correlated": {
                "model_type": "correlated",
                "param_name": "p",
                "values": [0.002, 0.005, 0.01],
                "base_params": {"correlation_strength": 0.35},
            },
        }

    # Full mode
    return {
        "depolarizing": {
            "model_type": "depolarizing",
            "param_name": "p",
            "values": [0.001, 0.0025, 0.005, 0.01, 0.02],
            "base_params": {},
        },
        "biased": {
            "model_type": "biased",
            "param_name": "eta",
            "values": [1.5, 2.0, 4.0, 8.0, 12.0],
            "base_params": {"p": 0.01},
        },
        "circuit_level": {
            "model_type": "circuit_level",
            "param_name": "p",
            "values": [0.0025, 0.005, 0.01, 0.015, 0.02],
            "base_params": {},
        },
        "phenomenological": {
            "model_type": "phenomenological",
            "param_name": "p",
            "values": [0.0005, 0.001, 0.0025, 0.005, 0.01],
            "base_params": {},
        },
        "correlated": {
            "model_type": "correlated",
            "param_name": "p",
            "values": [0.001, 0.0025, 0.005, 0.01, 0.015],
            "base_params": {"correlation_strength": 0.35},
        },
    }


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def _parse_models_csv(models_csv: str) -> List[str]:
    models = [m.strip().lower() for m in models_csv.split(",") if m.strip()]
    if not models:
        raise ValueError("Model list is empty.")
    bad = [m for m in models if m not in ALLOWED_MODELS]
    if bad:
        raise ValueError(
            f"Invalid models: {bad}. Allowed: {sorted(ALLOWED_MODELS)}"
        )
    # Keep order without duplicates
    unique: List[str] = []
    seen = set()
    for m in models:
        if m not in seen:
            unique.append(m)
            seen.add(m)
    return unique


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Week 2 Person 2 - Noise calibration sweep runner."
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=300,
        help="Number of shots per calibration point (default: 300).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Base seed for reproducibility.",
    )
    parser.add_argument(
        "--logical-basis",
        type=str,
        default="x",
        choices=["x", "z"],
        help="Logical basis for the XZZX code.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="depolarizing,biased,circuit_level,phenomenological,correlated",
        help="CSV of models to calibrate.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="min_ler",
        choices=["min_ler", "min_time"],
        help="Sweep optimization objective.",
    )
    parser.add_argument(
        "--keep-soft",
        type=int,
        default=0,
        help="Number of soft samples to keep per point.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use shorter sweeps for faster execution.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/week2_person2_noise_calibration.json",
        help="Output JSON path.",
    )
    return parser


# ---------------------------------------------------------------------------
# Integration with src.noise.noise_calibration
# ---------------------------------------------------------------------------

def _import_noise_calibration_module():
    try:
        return importlib.import_module("src.noise.noise_calibration")
    except Exception as exc:
        raise RuntimeError(
            "Could not import src.noise.noise_calibration. "
            "Check repository structure and PYTHONPATH."
        ) from exc


def _prepare_inputs(
    nc: Any,
    cases_dicts: List[Dict[str, Any]],
    sweeps_dicts: List[Dict[str, Any]],
    shots: int,
    seed: int,
    keep_soft_info_samples: int,
) -> Dict[str, Any]:
    """
    Build objects using module classes when available;
    otherwise keep dicts (compatibility).
    """
    CalibrationCase = getattr(nc, "CalibrationCase", None)
    SweepSpec = getattr(nc, "SweepSpec", None)
    CalibrationConfig = getattr(nc, "CalibrationConfig", None)

    cases = [_build_obj_if_class(CalibrationCase, c) for c in cases_dicts]
    sweeps = [_build_obj_if_class(SweepSpec, s) for s in sweeps_dicts]

    cfg_dict = {
        "shots": shots,
        "seed": seed,
        "keep_soft_info_samples": keep_soft_info_samples,
    }
    config = _build_obj_if_class(CalibrationConfig, cfg_dict)

    return {"cases": cases, "sweeps": sweeps, "config": config}


def _run_multi_model_calibration(
    nc: Any,
    *,
    cases: Any,
    sweeps: Any,
    config: Any,
    logical_basis: str,
    objective: str,
) -> Dict[str, Any]:
    """
    Call run_multi_model_calibration robustly against signature changes.
    """
    fn = getattr(nc, "run_multi_model_calibration", None)
    if fn is None:
        raise RuntimeError(
            "No existe run_multi_model_calibration en src.noise.noise_calibration."
        )

    attempts = [
        {
            "cases": cases,
            "sweeps": sweeps,
            "config": config,
            "logical_basis": logical_basis,
            "objective": objective,
        },
        {
            "cases": cases,
            "sweeps": sweeps,
            "config": config,
            "logical_basis": logical_basis,
        },
        {
            "cases": cases,
            "sweeps": sweeps,
            "calibration_config": config,
            "logical_basis": logical_basis,
            "objective": objective,
        },
        {
            "cases": cases,
            "sweeps": sweeps,
            "calibration_config": config,
            "logical_basis": logical_basis,
        },
    ]

    last_exc: Optional[Exception] = None
    for kw in attempts:
        try:
            out = _call_with_supported_kwargs(fn, kw)
            if isinstance(out, dict):
                return out
            return {"raw_report": out}
        except Exception as exc:
            last_exc = exc

    # Last positional attempt
    try:
        out = fn(cases, sweeps, config, logical_basis=logical_basis)
        if isinstance(out, dict):
            return out
        return {"raw_report": out}
    except Exception as exc:
        if last_exc is not None:
            raise RuntimeError(
                f"Failed to run run_multi_model_calibration. Last error: {last_exc}"
            ) from exc
        raise


# ---------------------------------------------------------------------------
# Report normalization / printing
# ---------------------------------------------------------------------------

def _infer_sweeps_summary_from_reports(
    sweeps_reports: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []

    for rep in sweeps_reports:
        sweep = rep.get("sweep", {}) if isinstance(rep, Mapping) else {}
        model_type = _pick(sweep, "model_type", "model", default="n/a")
        param_name = _pick(sweep, "param_name", "sweep_param_name", default="n/a")

        best_value = _pick(rep, "best_value", "best_param_value")
        cases_summary = rep.get("cases_summary", []) if isinstance(rep, Mapping) else []
        n_cases = len(cases_summary) if isinstance(cases_summary, list) else None

        mean_ler = _pick(rep, "mean_ler_at_best", "mean_best_ler", "mean_ler")
        mean_t = _pick(
            rep,
            "mean_avg_decode_time_sec_at_best",
            "mean_best_time_sec",
            "mean_avg_decode_time_sec",
            "mean_time",
        )

        # Fallback: compute means from best_point by case
        if (mean_ler is None or mean_t is None) and isinstance(cases_summary, list) and cases_summary:
            lers: List[float] = []
            times: List[float] = []
            best_vals: List[float] = []
            for cs in cases_summary:
                if not isinstance(cs, Mapping):
                    continue
                bp = _pick(cs, "best_point", "best", default={})
                if isinstance(bp, Mapping):
                    ler_bp = _pick(bp, "ler", "benchmark_error_rate")
                    t_bp = _pick(bp, "avg_decode_time_sec", "decode_time_sec")
                    v_bp = _pick(bp, "sweep_value", "best_value")
                    if isinstance(ler_bp, (int, float)):
                        lers.append(float(ler_bp))
                    if isinstance(t_bp, (int, float)):
                        times.append(float(t_bp))
                    if isinstance(v_bp, (int, float)):
                        best_vals.append(float(v_bp))

            if mean_ler is None and lers:
                mean_ler = sum(lers) / len(lers)
            if mean_t is None and times:
                mean_t = sum(times) / len(times)
            if best_value is None and best_vals:
                best_value = sum(best_vals) / len(best_vals)

        summary.append(
            {
                "model_type": model_type,
                "param_name": param_name,
                "best_value": best_value,
                "mean_ler_at_best": mean_ler,
                "mean_avg_decode_time_sec_at_best": mean_t,
                "num_cases": n_cases,
            }
        )

    return summary


def normalize_report(
    raw_report: Mapping[str, Any],
    *,
    shots: int,
    seed: int,
    logical_basis: str,
    objective: str,
    keep_soft_info_samples: int,
    selected_templates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "metadata": {
            "report_name": "noise_calibration_multi_model",
            "timestamp_utc": _utc_now_iso(),
            "script": "run_week2_person2_noise_calibration.py",
            "shots": int(shots),
            "seed": int(seed),
            "logical_basis": logical_basis,
            "objective": objective,
        },
        "config": {
            "shots": int(shots),
            "seed": int(seed),
            "keep_soft_info_samples": int(keep_soft_info_samples),
        },
        "selected_sweep_templates": list(selected_templates),
    }

    # Standard keys
    for k in ("num_sweeps", "sweeps_summary", "sweeps_reports"):
        if k in raw_report:
            report[k] = raw_report[k]

    # Fallback for alternate key names
    if "sweeps_reports" not in report:
        if "reports" in raw_report and isinstance(raw_report["reports"], list):
            report["sweeps_reports"] = raw_report["reports"]
        elif "sweeps" in raw_report and isinstance(raw_report["sweeps"], list):
            report["sweeps_reports"] = raw_report["sweeps"]

    # If summary is missing, infer it
    if "sweeps_summary" not in report and isinstance(report.get("sweeps_reports"), list):
        report["sweeps_summary"] = _infer_sweeps_summary_from_reports(report["sweeps_reports"])

    if "num_sweeps" not in report:
        if isinstance(report.get("sweeps_reports"), list):
            report["num_sweeps"] = len(report["sweeps_reports"])
        elif isinstance(report.get("sweeps_summary"), list):
            report["num_sweeps"] = len(report["sweeps_summary"])
        else:
            report["num_sweeps"] = len(selected_templates)

    # Keep raw report if no standard structure exists
    if "sweeps_summary" not in report and "sweeps_reports" not in report:
        report["raw_report"] = raw_report

    return report


def print_summary_table(report: Mapping[str, Any]) -> None:
    rows = report.get("sweeps_summary", [])
    if not isinstance(rows, list):
        rows = []

    print("\n=== Week 2 Person 2 - Noise Calibration ===")
    print(
        f"{'MODEL':<18} {'PARAM':<12} {'BEST_VALUE':>12} "
        f"{'MEAN_BEST_LER':>14} {'MEAN_BEST_t(s)':>16} {'N_CASES':>8}"
    )
    print("-" * 87)

    for entry in rows:
        if not isinstance(entry, Mapping):
            continue
        model = _pick(entry, "model_type", "model", default="n/a")
        param = _pick(entry, "param_name", "sweep_param_name", "parameter", default="n/a")
        best = _pick(entry, "best_value", "best_param_value", "best_param")
        mean_ler = _pick(entry, "mean_ler_at_best", "mean_best_ler", "mean_ler")
        mean_t = _pick(
            entry,
            "mean_avg_decode_time_sec_at_best",
            "mean_best_time_sec",
            "mean_best_t",
            "mean_avg_decode_time_sec",
            "mean_time",
        )
        n_cases = _pick(entry, "num_cases", "n_cases", default="n/a")

        print(
            f"{str(model):<18} {str(param):<12} {_fmt_float(best, 6):>12} "
            f"{_fmt_float(mean_ler, 6):>14} {_fmt_float(mean_t, 6):>16} {str(n_cases):>8}"
        )


def save_json(data: Mapping[str, Any], output_path: str) -> None:
    out = os.path.abspath(output_path)
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.shots <= 0:
        parser.error("--shots must be > 0")
    if args.keep_soft < 0:
        parser.error("--keep-soft must be >= 0")

    try:
        selected_models = _parse_models_csv(args.models)
    except ValueError as exc:
        parser.error(str(exc))

    templates = default_sweep_templates(fast=args.fast)
    selected_templates = [templates[m] for m in selected_models]

    nc = _import_noise_calibration_module()
    cases_dicts = default_cases()

    prepared = _prepare_inputs(
        nc,
        cases_dicts=cases_dicts,
        sweeps_dicts=selected_templates,
        shots=args.shots,
        seed=args.seed,
        keep_soft_info_samples=args.keep_soft,
    )

    raw_report = _run_multi_model_calibration(
        nc,
        cases=prepared["cases"],
        sweeps=prepared["sweeps"],
        config=prepared["config"],
        logical_basis=args.logical_basis,
        objective=args.objective,
    )

    report = normalize_report(
        raw_report,
        shots=args.shots,
        seed=args.seed,
        logical_basis=args.logical_basis,
        objective=args.objective,
        keep_soft_info_samples=args.keep_soft,
        selected_templates=selected_templates,
    )

    print_summary_table(report)
    save_json(report, args.output)
    print(f"\nJSON saved to: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
