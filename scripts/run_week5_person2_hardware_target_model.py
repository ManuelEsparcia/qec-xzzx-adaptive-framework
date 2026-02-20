from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

try:
    import seaborn as sns
except Exception:
    sns = None  # type: ignore[assignment]


# Ensure "src" import works when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.codes.xzzx_code import generate_xzzx_circuit  # noqa: E402
from src.decoders.belief_matching_decoder import BeliefMatchingDecoderWithSoftInfo  # noqa: E402
from src.decoders.mwpm_decoder import MWPMDecoderWithSoftInfo  # noqa: E402
from src.decoders.union_find_decoder import UnionFindDecoderWithSoftInfo  # noqa: E402
from src.hardware.hardware_models import (  # noqa: E402
    build_latency_table,
    compatibility_matrix,
    default_architectures,
)
from src.noise.noise_models import apply_noise_model  # noqa: E402
from src.switching.adaptive_decoder import AdaptiveConfig, AdaptiveDecoder  # noqa: E402


NOISE_ALLOWED = {
    "none",
    "depolarizing",
    "biased_eta10",
    "biased_eta100",
    "biased_eta500",
    "circuit_level",
    "correlated",
}
FAST_BACKEND_ALLOWED = {"uf", "bm"}


@dataclass(frozen=True)
class DecoderRowSpec:
    decoder_label: str
    backend: str
    g_threshold: Optional[float] = None


@dataclass(frozen=True)
class WorkloadFeatures:
    distance: int
    rounds: int
    num_detectors: int
    dem_terms: int


@dataclass(frozen=True)
class TargetArchModel:
    name: str
    ops_per_sec: float
    fixed_overhead_sec: float
    detector_penalty_sec: float

    def __post_init__(self) -> None:
        if float(self.ops_per_sec) <= 0.0:
            raise ValueError("ops_per_sec must be > 0")
        if float(self.fixed_overhead_sec) < 0.0:
            raise ValueError("fixed_overhead_sec must be >= 0")
        if float(self.detector_penalty_sec) < 0.0:
            raise ValueError("detector_penalty_sec must be >= 0")


@dataclass(frozen=True)
class SwitchRateModel:
    slope: float
    intercept: float


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


def parse_min_switch_weight(raw: str) -> Optional[int]:
    token = raw.strip().lower()
    if token in {"none", "null", "na", "n/a"}:
        return None
    val = int(token)
    if val < 0:
        raise ValueError("min switch weight must be >= 0 or none")
    return val


def format_seconds(x: float) -> str:
    if x < 1e-9:
        return f"{x * 1e12:.3f} ps"
    if x < 1e-6:
        return f"{x * 1e9:.2f} ns"
    if x < 1e-3:
        return f"{x * 1e6:.2f} us"
    if x < 1.0:
        return f"{x * 1e3:.2f} ms"
    return f"{x:.3f} s"


def make_decoder_rows(
    adaptive_thresholds: Sequence[float],
    *,
    include_bm: bool,
) -> List[DecoderRowSpec]:
    rows: List[DecoderRowSpec] = [
        DecoderRowSpec(decoder_label="mwpm", backend="mwpm"),
        DecoderRowSpec(decoder_label="uf", backend="uf"),
    ]
    if include_bm:
        rows.append(DecoderRowSpec(decoder_label="bm", backend="bm"))
    for g in adaptive_thresholds:
        rows.append(
            DecoderRowSpec(
                decoder_label=f"adaptive_g{g:.2f}",
                backend="adaptive",
                g_threshold=float(g),
            )
        )
    return rows


def default_target_arch_models() -> List[TargetArchModel]:
    return [
        TargetArchModel(
            name="GoogleSuperconducting",
            ops_per_sec=8.0e10,
            fixed_overhead_sec=35e-9,
            detector_penalty_sec=1.5e-11,
        ),
        TargetArchModel(
            name="IBMEagle",
            ops_per_sec=6.0e10,
            fixed_overhead_sec=45e-9,
            detector_penalty_sec=2.0e-11,
        ),
        TargetArchModel(
            name="IonQForte",
            ops_per_sec=2.0e10,
            fixed_overhead_sec=80e-9,
            detector_penalty_sec=3.0e-11,
        ),
        TargetArchModel(
            name="PsiQuantumPhotonic",
            ops_per_sec=5.0e11,
            fixed_overhead_sec=1.5e-9,
            detector_penalty_sec=5.0e-12,
        ),
    ]


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


def _benchmark_single_run(
    *,
    circuit: Any,
    row: DecoderRowSpec,
    shots: int,
    seed: int,
    adaptive_fast_mode: bool,
    adaptive_fast_backend: str,
    adaptive_min_switch_weight: Optional[int],
) -> Dict[str, float]:
    if row.backend == "mwpm":
        d = MWPMDecoderWithSoftInfo(circuit)
        d.sampler = circuit.compile_detector_sampler(seed=int(seed))
        res = d.benchmark(shots=int(shots), keep_soft_info_samples=0)
        return {
            "error_rate": float(res["error_rate"]),
            "avg_decode_time_sec": float(res["avg_decode_time"]),
            "switch_rate": 0.0,
        }

    if row.backend == "uf":
        d = UnionFindDecoderWithSoftInfo(circuit, prefer_union_find=True)
        d.sampler = circuit.compile_detector_sampler(seed=int(seed))
        res = d.benchmark(shots=int(shots), keep_soft_info_samples=0)
        return {
            "error_rate": float(res["error_rate"]),
            "avg_decode_time_sec": float(res["avg_decode_time"]),
            "switch_rate": 0.0,
        }

    if row.backend == "bm":
        d = BeliefMatchingDecoderWithSoftInfo(circuit, prefer_belief_propagation=True)
        d.sampler = circuit.compile_detector_sampler(seed=int(seed))
        res = d.benchmark(shots=int(shots), keep_soft_info_samples=0)
        return {
            "error_rate": float(res["error_rate"]),
            "avg_decode_time_sec": float(res["avg_decode_time"]),
            "switch_rate": 0.0,
        }

    if row.backend == "adaptive":
        if row.g_threshold is None:
            raise ValueError("adaptive row requires g_threshold")
        mwpm = MWPMDecoderWithSoftInfo(circuit)
        if adaptive_fast_backend == "uf":
            fast = UnionFindDecoderWithSoftInfo(circuit, prefer_union_find=True)
        elif adaptive_fast_backend == "bm":
            fast = BeliefMatchingDecoderWithSoftInfo(circuit, prefer_belief_propagation=True)
        else:
            raise ValueError(f"unsupported adaptive_fast_backend: {adaptive_fast_backend}")

        adaptive = AdaptiveDecoder(
            circuit=circuit,
            fast_decoder=fast,
            accurate_decoder=mwpm,
            config=AdaptiveConfig(
                g_threshold=float(row.g_threshold),
                compare_against_mwpm_in_benchmark=False,
                min_syndrome_weight_for_switch=adaptive_min_switch_weight,
            ),
        )
        adaptive.sampler = circuit.compile_detector_sampler(seed=int(seed))
        res = adaptive.benchmark_adaptive(
            shots=int(shots),
            g_threshold=float(row.g_threshold),
            keep_samples=0,
            compare_against_mwpm=False,
            fast_mode=bool(adaptive_fast_mode),
        )
        return {
            "error_rate": float(res["error_rate_adaptive"]),
            "avg_decode_time_sec": float(res["avg_decode_time_adaptive"]),
            "switch_rate": float(res["switch_rate"]),
        }

    raise ValueError(f"unknown decoder backend: {row.backend}")


def benchmark_row_distance(
    *,
    row: DecoderRowSpec,
    distance: int,
    rounds: int,
    noise_model: str,
    p_phys: float,
    logical_basis: str,
    shots: int,
    repeats: int,
    base_seed: int,
    adaptive_fast_mode: bool,
    adaptive_fast_backend: str,
    adaptive_min_switch_weight: Optional[int],
) -> Dict[str, Any]:
    base_circuit = generate_xzzx_circuit(
        distance=int(distance),
        rounds=int(rounds),
        noise_model="none",
        p=0.0,
        logical_basis=logical_basis,
    )
    if noise_model == "none":
        circuit = base_circuit
    else:
        circuit = apply_noise_model(base_circuit, model=noise_spec_from_name(noise_model, float(p_phys)))

    run_rows: List[Dict[str, float]] = []
    for i in range(int(repeats)):
        seed_i = int(base_seed + i * 10007 + distance * 977 + rounds * 53)
        metrics = _benchmark_single_run(
            circuit=circuit,
            row=row,
            shots=int(shots),
            seed=seed_i,
            adaptive_fast_mode=adaptive_fast_mode,
            adaptive_fast_backend=adaptive_fast_backend,
            adaptive_min_switch_weight=adaptive_min_switch_weight,
        )
        run_rows.append(
            {
                "repeat_index": float(i),
                "seed": float(seed_i),
                "error_rate": float(metrics["error_rate"]),
                "avg_decode_time_sec": float(metrics["avg_decode_time_sec"]),
                "switch_rate": float(metrics["switch_rate"]),
            }
        )

    return {
        "decoder_label": row.decoder_label,
        "backend": row.backend,
        "g_threshold": None if row.g_threshold is None else float(row.g_threshold),
        "distance": int(distance),
        "rounds": int(rounds),
        "noise_model": noise_model,
        "p_phys": float(p_phys),
        "shots": int(shots),
        "repeats": int(repeats),
        "mean_error_rate": float(mean(r["error_rate"] for r in run_rows)),
        "mean_avg_decode_time_sec": float(mean(r["avg_decode_time_sec"] for r in run_rows)),
        "mean_switch_rate": float(mean(r["switch_rate"] for r in run_rows)),
        "repeat_runs": run_rows,
        "status": "ok",
    }


def fit_power_law(distances: Sequence[int], values: Sequence[float]) -> Dict[str, float]:
    if len(distances) != len(values) or not distances:
        raise ValueError("distances and values must have same non-zero length")

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


def predict_power_law(distance: int, *, coefficient: float, exponent: float) -> float:
    if distance <= 0:
        raise ValueError("distance must be > 0")
    return float(float(coefficient) * (float(distance) ** float(exponent)))


def build_python_scaling_models(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_decoder: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_decoder.setdefault(str(row["decoder_label"]), []).append(row)

    out: List[Dict[str, Any]] = []
    for label, arr in sorted(by_decoder.items()):
        arr_sorted = sorted(arr, key=lambda x: int(x["distance"]))
        ds = [int(x["distance"]) for x in arr_sorted]
        ts = [float(x["mean_avg_decode_time_sec"]) for x in arr_sorted]
        model = fit_power_law(ds, ts)
        out.append(
            {
                "decoder_label": label,
                "backend": str(arr_sorted[0]["backend"]),
                "g_threshold": arr_sorted[0].get("g_threshold"),
                "fit_distances": ds,
                "fit_times_sec": ts,
                **model,
            }
        )
    return out


def build_python_predictions(
    *,
    scaling_models: Sequence[Dict[str, Any]],
    target_distances: Sequence[int],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for model in scaling_models:
        label = str(model["decoder_label"])
        for d in target_distances:
            t_pred = predict_power_law(
                int(d),
                coefficient=float(model["coefficient"]),
                exponent=float(model["exponent"]),
            )
            out.append(
                {
                    "decoder_label": label,
                    "distance": int(d),
                    "predicted_decode_time_sec": float(t_pred),
                }
            )
    return out


def fit_switch_rate_models(rows: Sequence[Dict[str, Any]]) -> Dict[str, SwitchRateModel]:
    by_label: Dict[str, List[Tuple[int, float]]] = {}
    for row in rows:
        if str(row["backend"]) != "adaptive":
            continue
        label = str(row["decoder_label"])
        by_label.setdefault(label, []).append(
            (int(row["distance"]), float(row["mean_switch_rate"]))
        )

    out: Dict[str, SwitchRateModel] = {}
    for label, arr in sorted(by_label.items()):
        arr_sorted = sorted(arr, key=lambda x: x[0])
        xs = np.asarray([x[0] for x in arr_sorted], dtype=float)
        ys = np.asarray([x[1] for x in arr_sorted], dtype=float)
        ys = np.clip(ys, 0.0, 1.0)
        if len(xs) == 1:
            slope = 0.0
            intercept = float(ys[0])
        else:
            slope, intercept = np.polyfit(xs, ys, 1)
        out[label] = SwitchRateModel(slope=float(slope), intercept=float(intercept))
    return out


def predict_switch_rate(
    *,
    model: Optional[SwitchRateModel],
    distance: int,
) -> float:
    if model is None:
        return 0.0
    raw = float(model.slope * float(distance) + model.intercept)
    return float(min(1.0, max(0.0, raw)))


def compute_workload_features(
    *,
    distances: Sequence[int],
    rounds_mode: str,
    rounds_fixed: int,
    logical_basis: str,
) -> List[WorkloadFeatures]:
    out: List[WorkloadFeatures] = []
    for d in distances:
        r = int(d) if rounds_mode == "distance" else int(rounds_fixed)
        c = generate_xzzx_circuit(
            distance=int(d),
            rounds=int(r),
            noise_model="none",
            p=0.0,
            logical_basis=logical_basis,
        )
        dem = c.detector_error_model(decompose_errors=True)
        dem_terms = len(str(dem).splitlines())
        out.append(
            WorkloadFeatures(
                distance=int(d),
                rounds=int(r),
                num_detectors=int(getattr(c, "num_detectors", 0)),
                dem_terms=int(dem_terms),
            )
        )
    return out


def proxy_ops_for_backend(
    *,
    backend: str,
    features: WorkloadFeatures,
    mwpm_factor: float,
    uf_factor: float,
    bm_factor: float,
) -> float:
    m = max(2.0, float(features.dem_terms))
    if backend == "mwpm":
        return float(mwpm_factor * m * math.log2(m))
    if backend == "uf":
        return float(uf_factor * m)
    if backend == "bm":
        return float(bm_factor * m * math.log2(m))
    raise ValueError(f"unsupported backend for proxy ops: {backend}")


def build_target_predictions(
    *,
    row_specs: Sequence[DecoderRowSpec],
    workload: Sequence[WorkloadFeatures],
    arch_models: Sequence[TargetArchModel],
    switch_models: Mapping[str, SwitchRateModel],
    adaptive_fast_backend: str,
    ops_factor_mwpm: float,
    ops_factor_uf: float,
    ops_factor_bm: float,
    ops_adaptive_dispatch: float,
    ops_rate_scale: float,
    fixed_overhead_scale: float,
    detector_penalty_scale: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    wl_by_distance = {int(w.distance): w for w in workload}
    for arch in arch_models:
        eff_ops_per_sec = float(arch.ops_per_sec) * float(ops_rate_scale)
        fixed_sec = float(arch.fixed_overhead_sec) * float(fixed_overhead_scale)
        det_penalty_sec = float(arch.detector_penalty_sec) * float(detector_penalty_scale)

        for row in row_specs:
            for d, w in sorted(wl_by_distance.items()):
                if row.backend == "adaptive":
                    sw = predict_switch_rate(
                        model=switch_models.get(str(row.decoder_label)),
                        distance=int(d),
                    )
                    ops_fast = proxy_ops_for_backend(
                        backend=adaptive_fast_backend,
                        features=w,
                        mwpm_factor=float(ops_factor_mwpm),
                        uf_factor=float(ops_factor_uf),
                        bm_factor=float(ops_factor_bm),
                    )
                    ops_mwpm = proxy_ops_for_backend(
                        backend="mwpm",
                        features=w,
                        mwpm_factor=float(ops_factor_mwpm),
                        uf_factor=float(ops_factor_uf),
                        bm_factor=float(ops_factor_bm),
                    )
                    ops = float(ops_adaptive_dispatch + (1.0 - sw) * ops_fast + sw * ops_mwpm)
                    switch_rate_pred = float(sw)
                else:
                    ops = proxy_ops_for_backend(
                        backend=str(row.backend),
                        features=w,
                        mwpm_factor=float(ops_factor_mwpm),
                        uf_factor=float(ops_factor_uf),
                        bm_factor=float(ops_factor_bm),
                    )
                    switch_rate_pred = 0.0

                latency = float(fixed_sec + det_penalty_sec * float(w.num_detectors) + ops / eff_ops_per_sec)
                out.append(
                    {
                        "architecture": arch.name,
                        "decoder_label": str(row.decoder_label),
                        "backend": str(row.backend),
                        "distance": int(d),
                        "rounds": int(w.rounds),
                        "num_detectors": int(w.num_detectors),
                        "dem_terms": int(w.dem_terms),
                        "switch_rate_pred": float(switch_rate_pred),
                        "proxy_ops": float(ops),
                        "predicted_decode_time_sec": float(latency),
                    }
                )
    return out


def build_dual_compatibility_report(
    *,
    python_predicted_rows: Sequence[Dict[str, Any]],
    target_predicted_rows: Sequence[Dict[str, Any]],
    decoder_labels: Sequence[str],
    distances: Sequence[int],
    safety_factor: float,
) -> Dict[str, Any]:
    py_table = build_latency_table(python_predicted_rows)
    arch_reports: List[Dict[str, Any]] = []

    for arch in default_architectures():
        py_matrix = compatibility_matrix(
            architecture=arch,
            decoder_labels=decoder_labels,
            distances=distances,
            latency_table=py_table,
            safety_factor=float(safety_factor),
        )
        py_total = len(decoder_labels) * len(distances)
        py_ok = int(sum(sum(r) for r in py_matrix))

        target_rows_arch = [
            r for r in target_predicted_rows if str(r["architecture"]) == str(arch.name)
        ]
        target_table = build_latency_table(target_rows_arch)
        target_matrix = compatibility_matrix(
            architecture=arch,
            decoder_labels=decoder_labels,
            distances=distances,
            latency_table=target_table,
            safety_factor=float(safety_factor),
        )
        target_total = len(decoder_labels) * len(distances)
        target_ok = int(sum(sum(r) for r in target_matrix))

        arch_reports.append(
            {
                "architecture": arch.name,
                "cycle_time_sec": float(arch.cycle_time_sec),
                "classical_budget_sec": float(arch.classical_budget_sec),
                "python_runtime_matrix": py_matrix,
                "target_model_matrix": target_matrix,
                "python_runtime_compatible_cells": py_ok,
                "python_runtime_total_cells": py_total,
                "python_runtime_ratio": float(py_ok / py_total) if py_total > 0 else float("nan"),
                "target_model_compatible_cells": target_ok,
                "target_model_total_cells": target_total,
                "target_model_ratio": float(target_ok / target_total) if target_total > 0 else float("nan"),
                "ratio_delta_target_minus_python": float(target_ok / target_total - py_ok / py_total)
                if (py_total > 0 and target_total > 0)
                else float("nan"),
            }
        )

    return {
        "decoder_labels": list(decoder_labels),
        "distances": [int(d) for d in distances],
        "architectures": arch_reports,
        "safety_factor": float(safety_factor),
    }


def _plot_heatmap_matrix(
    *,
    ax: Any,
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[int],
    title: str,
) -> None:
    cmap = ListedColormap(["#d73027", "#1a9850"])
    annot = np.where(matrix > 0, "OK", "NO")
    if sns is not None:
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            vmin=0,
            vmax=1,
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            annot=annot,
            fmt="",
        )
    else:
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, annot[i, j], ha="center", va="center", fontsize=8, color="black")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Decoder")
    ax.set_xticks(np.arange(len(col_labels)) + 0.5 if sns is not None else np.arange(len(col_labels)))
    ax.set_xticklabels([str(x) for x in col_labels], rotation=0)
    ax.set_yticks(np.arange(len(row_labels)) + 0.5 if sns is not None else np.arange(len(row_labels)))
    ax.set_yticklabels(list(row_labels), rotation=0)


def save_dual_heatmaps(
    *,
    compatibility: Dict[str, Any],
    output_path: Path,
    title: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    archs = compatibility["architectures"]
    row_labels = compatibility["decoder_labels"]
    distances = compatibility["distances"]

    n_rows = len(archs)
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, max(3.8 * n_rows, 6.0)))
    if n_rows == 1:
        axes_grid = np.asarray([axes])
    else:
        axes_grid = np.asarray(axes)

    for i, arch in enumerate(archs):
        budget = format_seconds(float(arch["classical_budget_sec"]))
        mat_py = np.asarray(arch["python_runtime_matrix"], dtype=float)
        mat_target = np.asarray(arch["target_model_matrix"], dtype=float)

        ax0 = axes_grid[i, 0]
        ax1 = axes_grid[i, 1]
        _plot_heatmap_matrix(
            ax=ax0,
            matrix=mat_py,
            row_labels=row_labels,
            col_labels=distances,
            title=f"{arch['architecture']} | Python runtime | budget={budget}",
        )
        _plot_heatmap_matrix(
            ax=ax1,
            matrix=mat_target,
            row_labels=row_labels,
            col_labels=distances,
            title=f"{arch['architecture']} | Target model | budget={budget}",
        )

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def save_json(data: Dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return output_path


def build_report(
    *,
    benchmark_rows: Sequence[Dict[str, Any]],
    python_scaling_models: Sequence[Dict[str, Any]],
    python_predicted_rows: Sequence[Dict[str, Any]],
    switch_models: Mapping[str, SwitchRateModel],
    workload_features: Sequence[WorkloadFeatures],
    target_arch_models: Sequence[TargetArchModel],
    target_predicted_rows: Sequence[Dict[str, Any]],
    compatibility: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "metadata": {
            "report_name": "week5_person2_hardware_target_model",
            "timestamp_utc": utc_now_iso(),
        },
        "config": {
            "benchmark_distances": [int(x) for x in parse_csv_ints(args.benchmark_distances)],
            "distances": [int(x) for x in parse_csv_ints(args.distances)],
            "rounds_mode": args.rounds_mode,
            "rounds_fixed": int(args.rounds),
            "noise_model": args.noise_model,
            "p_phys": float(args.p_phys),
            "logical_basis": args.logical_basis,
            "adaptive_thresholds": [float(x) for x in parse_csv_floats(args.adaptive_thresholds)],
            "include_bm": bool(args.include_bm),
            "shots": int(args.shots),
            "repeats": int(args.repeats),
            "seed": int(args.seed),
            "adaptive_fast_mode": bool(args.adaptive_fast_mode),
            "adaptive_fast_backend": args.adaptive_fast_backend,
            "adaptive_min_switch_weight": parse_min_switch_weight(args.adaptive_min_switch_weight),
            "safety_factor": float(args.safety_factor),
            "ops_factor_mwpm": float(args.ops_factor_mwpm),
            "ops_factor_uf": float(args.ops_factor_uf),
            "ops_factor_bm": float(args.ops_factor_bm),
            "ops_adaptive_dispatch": float(args.ops_adaptive_dispatch),
            "ops_rate_scale": float(args.ops_rate_scale),
            "fixed_overhead_scale": float(args.fixed_overhead_scale),
            "detector_penalty_scale": float(args.detector_penalty_scale),
            "figure_output": args.figure_output,
        },
        "benchmark_rows": list(benchmark_rows),
        "python_scaling_models": list(python_scaling_models),
        "python_predicted_rows": list(python_predicted_rows),
        "switch_models": {
            k: {"slope": float(v.slope), "intercept": float(v.intercept)}
            for k, v in sorted(switch_models.items())
        },
        "workload_features": [
            {
                "distance": int(w.distance),
                "rounds": int(w.rounds),
                "num_detectors": int(w.num_detectors),
                "dem_terms": int(w.dem_terms),
            }
            for w in workload_features
        ],
        "target_arch_models": [
            {
                "name": m.name,
                "ops_per_sec": float(m.ops_per_sec),
                "fixed_overhead_sec": float(m.fixed_overhead_sec),
                "detector_penalty_sec": float(m.detector_penalty_sec),
            }
            for m in target_arch_models
        ],
        "target_predicted_rows": list(target_predicted_rows),
        "compatibility": compatibility,
    }


def print_summary(report: Dict[str, Any]) -> None:
    print("\n=== Week 5 Block 5 - Hardware Target Model ===")
    cfg = report["config"]
    print(
        f"bench_distances={cfg['benchmark_distances']} | target_distances={cfg['distances']} | "
        f"shots={cfg['shots']} | repeats={cfg['repeats']} | "
        f"adaptive_fast_backend={cfg['adaptive_fast_backend']}"
    )
    print(
        f"target model: ops_factors(mwpm={cfg['ops_factor_mwpm']:.3f}, "
        f"uf={cfg['ops_factor_uf']:.3f}, bm={cfg['ops_factor_bm']:.3f}), "
        f"dispatch_ops={cfg['ops_adaptive_dispatch']:.3f}"
    )

    print("\n--- Compatibility delta (target vs python) ---")
    for a in report["compatibility"]["architectures"]:
        py_ratio = 100.0 * float(a["python_runtime_ratio"])
        tg_ratio = 100.0 * float(a["target_model_ratio"])
        d_ratio = 100.0 * float(a["ratio_delta_target_minus_python"])
        print(
            f"{a['architecture']:<22} | python={py_ratio:>6.2f}% | "
            f"target={tg_ratio:>6.2f}% | delta={d_ratio:+6.2f}%"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Week 5 Block 5: hardware compatibility with target-aware classical latency model "
            "(beyond raw Python runtime)."
        )
    )
    parser.add_argument("--benchmark-distances", type=str, default="5,7,9", help="CSV distances for timing fits.")
    parser.add_argument("--distances", type=str, default="5,7,9,11,13", help="CSV target distances.")
    parser.add_argument("--rounds-mode", type=str, choices=["fixed", "distance"], default="distance", help="Use fixed rounds or rounds=distance.")
    parser.add_argument("--rounds", type=int, default=3, help="Rounds when --rounds-mode=fixed.")
    parser.add_argument(
        "--noise-model",
        type=str,
        default="depolarizing",
        help="Noise model for benchmark timings.",
    )
    parser.add_argument("--p-phys", type=float, default=0.01, help="Physical error rate.")
    parser.add_argument("--logical-basis", type=str, choices=["x", "z"], default="x", help="Logical basis.")
    parser.add_argument("--adaptive-thresholds", type=str, default="0.20,0.35,0.50,0.65,0.80", help="CSV thresholds for adaptive rows.")
    parser.add_argument("--include-bm", action="store_true", help="Include BM row in analysis.")
    parser.add_argument("--shots", type=int, default=120, help="Shots per benchmark run.")
    parser.add_argument("--repeats", type=int, default=2, help="Repeats per benchmark point.")
    parser.add_argument("--seed", type=int, default=2026, help="Base seed.")
    parser.add_argument("--adaptive-fast-mode", action="store_true", help="Use fast_mode=True for adaptive benchmark.")
    parser.add_argument("--adaptive-fast-backend", type=str, choices=["uf", "bm"], default="bm", help="Fast backend used by adaptive in benchmark and target model.")
    parser.add_argument("--adaptive-min-switch-weight", type=str, default="none", help="Adaptive min switch weight (int or 'none').")
    parser.add_argument("--safety-factor", type=float, default=1.0, help="Latency safety factor before budget check.")
    parser.add_argument("--ops-factor-mwpm", type=float, default=4.0, help="Proxy ops factor for MWPM.")
    parser.add_argument("--ops-factor-uf", type=float, default=1.2, help="Proxy ops factor for UF.")
    parser.add_argument("--ops-factor-bm", type=float, default=2.2, help="Proxy ops factor for BM.")
    parser.add_argument("--ops-adaptive-dispatch", type=float, default=600.0, help="Extra proxy ops for adaptive dispatch.")
    parser.add_argument("--ops-rate-scale", type=float, default=1.0, help="Global multiplier on target ops/s.")
    parser.add_argument("--fixed-overhead-scale", type=float, default=1.0, help="Global multiplier on fixed overhead.")
    parser.add_argument("--detector-penalty-scale", type=float, default=1.0, help="Global multiplier on detector penalty.")
    parser.add_argument(
        "--output",
        type=str,
        default="results/week5_person2_hardware_target_model.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--figure-output",
        type=str,
        default="figures/week5_person2_hardware_target_model_heatmaps.png",
        help="Output comparison heatmap figure (PNG).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_distances = parse_csv_ints(args.benchmark_distances)
    target_distances = parse_csv_ints(args.distances)
    adaptive_thresholds = parse_csv_floats(args.adaptive_thresholds)
    adaptive_min_switch_weight = parse_min_switch_weight(args.adaptive_min_switch_weight)

    if any(d < 3 or d % 2 == 0 for d in benchmark_distances):
        raise ValueError("--benchmark-distances must contain odd values >= 3")
    if any(d < 3 or d % 2 == 0 for d in target_distances):
        raise ValueError("--distances must contain odd values >= 3")
    if args.rounds <= 0:
        raise ValueError("--rounds must be > 0")
    if args.shots <= 0:
        raise ValueError("--shots must be > 0")
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")
    if args.noise_model not in NOISE_ALLOWED:
        raise ValueError(f"--noise-model must be one of {sorted(NOISE_ALLOWED)}")
    if not (0.0 <= float(args.p_phys) <= 1.0):
        raise ValueError("--p-phys must be in [0,1]")
    if any((g < 0.0 or g > 1.0) for g in adaptive_thresholds):
        raise ValueError("--adaptive-thresholds entries must be in [0,1]")
    if float(args.safety_factor) <= 0.0:
        raise ValueError("--safety-factor must be > 0")
    if float(args.ops_factor_mwpm) <= 0.0:
        raise ValueError("--ops-factor-mwpm must be > 0")
    if float(args.ops_factor_uf) <= 0.0:
        raise ValueError("--ops-factor-uf must be > 0")
    if float(args.ops_factor_bm) <= 0.0:
        raise ValueError("--ops-factor-bm must be > 0")
    if float(args.ops_adaptive_dispatch) < 0.0:
        raise ValueError("--ops-adaptive-dispatch must be >= 0")
    if float(args.ops_rate_scale) <= 0.0:
        raise ValueError("--ops-rate-scale must be > 0")
    if float(args.fixed_overhead_scale) <= 0.0:
        raise ValueError("--fixed-overhead-scale must be > 0")
    if float(args.detector_penalty_scale) <= 0.0:
        raise ValueError("--detector-penalty-scale must be > 0")
    if args.adaptive_fast_backend not in FAST_BACKEND_ALLOWED:
        raise ValueError(f"--adaptive-fast-backend must be one of {sorted(FAST_BACKEND_ALLOWED)}")

    row_specs = make_decoder_rows(adaptive_thresholds, include_bm=bool(args.include_bm))
    benchmark_rows: List[Dict[str, Any]] = []
    total = len(row_specs) * len(benchmark_distances)
    idx = 0

    for row in row_specs:
        for d in benchmark_distances:
            idx += 1
            rounds_i = int(d) if args.rounds_mode == "distance" else int(args.rounds)
            seed_i = int(args.seed + idx * 100003 + d * 103)
            out = benchmark_row_distance(
                row=row,
                distance=int(d),
                rounds=rounds_i,
                noise_model=str(args.noise_model),
                p_phys=float(args.p_phys),
                logical_basis=args.logical_basis,
                shots=int(args.shots),
                repeats=int(args.repeats),
                base_seed=seed_i,
                adaptive_fast_mode=bool(args.adaptive_fast_mode),
                adaptive_fast_backend=str(args.adaptive_fast_backend),
                adaptive_min_switch_weight=adaptive_min_switch_weight,
            )
            benchmark_rows.append(out)
            print(
                f"[{idx}/{total}] {row.decoder_label:<15} | d={d:>2} | rounds={rounds_i:>2} | "
                f"t={out['mean_avg_decode_time_sec']:.6f}s | ER={out['mean_error_rate']:.6f} | "
                f"sw={100.0 * out['mean_switch_rate']:.2f}%"
            )

    python_scaling_models = build_python_scaling_models(benchmark_rows)
    python_predicted_rows = build_python_predictions(
        scaling_models=python_scaling_models,
        target_distances=target_distances,
    )

    switch_models = fit_switch_rate_models(benchmark_rows)
    workload_features = compute_workload_features(
        distances=target_distances,
        rounds_mode=args.rounds_mode,
        rounds_fixed=int(args.rounds),
        logical_basis=args.logical_basis,
    )
    target_arch_models = default_target_arch_models()
    target_predicted_rows = build_target_predictions(
        row_specs=row_specs,
        workload=workload_features,
        arch_models=target_arch_models,
        switch_models=switch_models,
        adaptive_fast_backend=str(args.adaptive_fast_backend),
        ops_factor_mwpm=float(args.ops_factor_mwpm),
        ops_factor_uf=float(args.ops_factor_uf),
        ops_factor_bm=float(args.ops_factor_bm),
        ops_adaptive_dispatch=float(args.ops_adaptive_dispatch),
        ops_rate_scale=float(args.ops_rate_scale),
        fixed_overhead_scale=float(args.fixed_overhead_scale),
        detector_penalty_scale=float(args.detector_penalty_scale),
    )

    compatibility = build_dual_compatibility_report(
        python_predicted_rows=python_predicted_rows,
        target_predicted_rows=target_predicted_rows,
        decoder_labels=[r.decoder_label for r in row_specs],
        distances=target_distances,
        safety_factor=float(args.safety_factor),
    )

    fig_path = save_dual_heatmaps(
        compatibility=compatibility,
        output_path=Path(args.figure_output),
        title="Week 5 Block 5 - Hardware Compatibility: Python vs Target Model",
    )

    report = build_report(
        benchmark_rows=benchmark_rows,
        python_scaling_models=python_scaling_models,
        python_predicted_rows=python_predicted_rows,
        switch_models=switch_models,
        workload_features=workload_features,
        target_arch_models=target_arch_models,
        target_predicted_rows=target_predicted_rows,
        compatibility=compatibility,
        args=args,
    )
    json_path = save_json(report, Path(args.output))
    print_summary(report)
    print(f"\nJSON saved at: {json_path}")
    print(f"Figure saved at: {fig_path}")


if __name__ == "__main__":
    main()
