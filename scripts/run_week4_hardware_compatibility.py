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
from src.noise.noise_models import apply_noise_model  # noqa: E402
from src.hardware.hardware_models import (  # noqa: E402
    build_latency_table,
    compatibility_matrix,
    default_architectures,
)
from src.switching.adaptive_decoder import AdaptiveConfig, AdaptiveDecoder  # noqa: E402


DECODER_BACKENDS = {"mwpm", "uf", "bm", "adaptive"}
NOISE_ALLOWED = {
    "none",
    "depolarizing",
    "biased_eta10",
    "biased_eta100",
    "biased_eta500",
    "circuit_level",
    "correlated",
}


@dataclass(frozen=True)
class DecoderRowSpec:
    decoder_label: str
    backend: str
    g_threshold: Optional[float] = None


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


def parse_csv_strings(raw: str) -> List[str]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("empty string list")
    return vals


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
        uf = UnionFindDecoderWithSoftInfo(circuit, prefer_union_find=True)
        adaptive = AdaptiveDecoder(
            circuit=circuit,
            fast_decoder=uf,
            accurate_decoder=mwpm,
            config=AdaptiveConfig(
                g_threshold=float(row.g_threshold),
                compare_against_mwpm_in_benchmark=False,
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


def fit_power_law(distances: Sequence[int], times: Sequence[float]) -> Dict[str, float]:
    if len(distances) != len(times) or not distances:
        raise ValueError("distances and times must have the same non-zero length")

    xs = np.asarray(distances, dtype=float)
    ys = np.asarray(times, dtype=float)
    if np.any(xs <= 0.0):
        raise ValueError("distances must be > 0")
    if np.any(ys <= 0.0):
        raise ValueError("times must be > 0")

    if len(xs) == 1:
        coefficient = float(ys[0])
        exponent = 0.0
        y_fit = np.asarray([coefficient], dtype=float)
    else:
        log_x = np.log(xs)
        log_y = np.log(ys)
        exponent, log_a = np.polyfit(log_x, log_y, 1)
        coefficient = float(math.exp(float(log_a)))
        y_fit = coefficient * (xs ** float(exponent))

    ss_res = float(np.sum((ys - y_fit) ** 2))
    ss_tot = float(np.sum((ys - float(np.mean(ys))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0

    return {
        "coefficient": float(coefficient),
        "exponent": float(exponent),
        "r2": float(r2),
    }


def predict_power_law_time(distance: int, *, coefficient: float, exponent: float) -> float:
    if distance <= 0:
        raise ValueError("distance must be > 0")
    return float(float(coefficient) * (float(distance) ** float(exponent)))


def build_scaling_models(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_decoder: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_decoder.setdefault(str(row["decoder_label"]), []).append(row)

    out: List[Dict[str, Any]] = []
    for decoder_label, arr in sorted(by_decoder.items()):
        arr_sorted = sorted(arr, key=lambda x: int(x["distance"]))
        distances = [int(x["distance"]) for x in arr_sorted]
        times = [float(x["mean_avg_decode_time_sec"]) for x in arr_sorted]
        model = fit_power_law(distances, times)
        out.append(
            {
                "decoder_label": decoder_label,
                "backend": str(arr_sorted[0]["backend"]),
                "g_threshold": arr_sorted[0].get("g_threshold"),
                "fit_distances": distances,
                "fit_times_sec": times,
                **model,
            }
        )
    return out


def build_predictions(
    *,
    scaling_models: Sequence[Dict[str, Any]],
    target_distances: Sequence[int],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for model in scaling_models:
        label = str(model["decoder_label"])
        a = float(model["coefficient"])
        b = float(model["exponent"])
        for d in target_distances:
            t_pred = predict_power_law_time(int(d), coefficient=a, exponent=b)
            out.append(
                {
                    "decoder_label": label,
                    "distance": int(d),
                    "predicted_decode_time_sec": float(t_pred),
                }
            )
    return out


def build_compatibility_report(
    *,
    predicted_rows: Sequence[Dict[str, Any]],
    decoder_labels: Sequence[str],
    distances: Sequence[int],
    safety_factor: float,
) -> Dict[str, Any]:
    latency_table = build_latency_table(predicted_rows)
    arch_reports: List[Dict[str, Any]] = []
    for arch in default_architectures():
        matrix = compatibility_matrix(
            architecture=arch,
            decoder_labels=decoder_labels,
            distances=distances,
            latency_table=latency_table,
            safety_factor=float(safety_factor),
        )
        total_cells = len(decoder_labels) * len(distances)
        compat_cells = int(sum(sum(row) for row in matrix))
        arch_reports.append(
            {
                "architecture": arch.name,
                "cycle_time_sec": float(arch.cycle_time_sec),
                "classical_budget_sec": float(arch.classical_budget_sec),
                "matrix": matrix,
                "compatible_cells": compat_cells,
                "total_cells": total_cells,
                "compatibility_ratio": float(compat_cells / total_cells) if total_cells > 0 else float("nan"),
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
                ax.text(
                    j,
                    i,
                    annot[i, j],
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Decoder")
    ax.set_xticks(np.arange(len(col_labels)) + 0.5 if sns is not None else np.arange(len(col_labels)))
    ax.set_xticklabels([str(x) for x in col_labels], rotation=0)
    ax.set_yticks(np.arange(len(row_labels)) + 0.5 if sns is not None else np.arange(len(row_labels)))
    ax.set_yticklabels(list(row_labels), rotation=0)


def save_heatmaps(
    *,
    compatibility: Dict[str, Any],
    output_path: Path,
    title: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    archs = compatibility["architectures"]
    row_labels = compatibility["decoder_labels"]
    distances = compatibility["distances"]

    n = len(archs)
    cols = 2
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.0 * rows))
    axes_arr = np.atleast_1d(axes).ravel()

    for i, arch in enumerate(archs):
        ax = axes_arr[i]
        mat = np.asarray(arch["matrix"], dtype=float)
        budget = format_seconds(float(arch["classical_budget_sec"]))
        ttl = f"{arch['architecture']} | budget={budget}"
        _plot_heatmap_matrix(
            ax=ax,
            matrix=mat,
            row_labels=row_labels,
            col_labels=distances,
            title=ttl,
        )

    for j in range(len(archs), len(axes_arr)):
        axes_arr[j].axis("off")

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
    scaling_models: Sequence[Dict[str, Any]],
    predicted_rows: Sequence[Dict[str, Any]],
    compatibility: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "metadata": {
            "report_name": "week4_hardware_compatibility",
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
            "safety_factor": float(args.safety_factor),
            "figure_output": args.figure_output,
        },
        "benchmark_rows": list(benchmark_rows),
        "scaling_models": list(scaling_models),
        "predicted_rows": list(predicted_rows),
        "compatibility": compatibility,
    }


def print_summary(report: Dict[str, Any]) -> None:
    print("\n=== Week 4 - Hardware Compatibility ===")
    cfg = report["config"]
    print(
        "bench_distances="
        f"{cfg['benchmark_distances']} | target_distances={cfg['distances']} | "
        f"shots={cfg['shots']} | repeats={cfg['repeats']} | rounds_mode={cfg['rounds_mode']}"
    )
    print("\n--- Fitted latency models ---")
    for m in report["scaling_models"]:
        print(
            f"{m['decoder_label']:<15} | t(d)=a*d^b | "
            f"a={m['coefficient']:.3e}, b={m['exponent']:.3f}, r2={m['r2']:.4f}"
        )
    print("\n--- Architecture compatibility ---")
    for a in report["compatibility"]["architectures"]:
        ratio = 100.0 * float(a["compatibility_ratio"])
        print(
            f"{a['architecture']:<22} | budget={format_seconds(float(a['classical_budget_sec'])):<10} | "
            f"compatible={a['compatible_cells']}/{a['total_cells']} ({ratio:.2f}%)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week 4 hardware-aware compatibility analysis for decoder latency vs architecture budgets."
    )
    parser.add_argument("--benchmark-distances", type=str, default="3,5,7", help="CSV distances used for empirical timing fit.")
    parser.add_argument("--distances", type=str, default="3,5,7,9,11,13", help="CSV target distances for compatibility matrix.")
    parser.add_argument("--rounds-mode", type=str, choices=["fixed", "distance"], default="distance", help="Use fixed rounds or rounds=distance.")
    parser.add_argument("--rounds", type=int, default=3, help="Rounds value when --rounds-mode=fixed.")
    parser.add_argument(
        "--noise-model",
        type=str,
        default="depolarizing",
        help="Noise model for benchmark timings (none,depolarizing,biased_eta10,biased_eta100,biased_eta500,circuit_level,correlated).",
    )
    parser.add_argument("--p-phys", type=float, default=0.01, help="Physical error rate for benchmark timings.")
    parser.add_argument("--logical-basis", type=str, choices=["x", "z"], default="x", help="Logical basis for generated circuits.")
    parser.add_argument("--adaptive-thresholds", type=str, default="0.20,0.35,0.50,0.65,0.80", help="CSV thresholds for adaptive rows.")
    parser.add_argument("--include-bm", action="store_true", help="Include BM row in compatibility analysis.")
    parser.add_argument("--shots", type=int, default=160, help="Shots per benchmark run.")
    parser.add_argument("--repeats", type=int, default=2, help="Repeats per benchmark point.")
    parser.add_argument("--seed", type=int, default=2026, help="Base seed.")
    parser.add_argument("--adaptive-fast-mode", action="store_true", help="Use fast_mode=True for adaptive benchmark.")
    parser.add_argument("--safety-factor", type=float, default=1.0, help="Latency safety factor before budget check.")
    parser.add_argument(
        "--output",
        type=str,
        default="results/week4_hardware_compatibility.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--figure-output",
        type=str,
        default="figures/week4_hardware_compatibility_heatmaps.png",
        help="Output figure path (PNG).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_distances = parse_csv_ints(args.benchmark_distances)
    target_distances = parse_csv_ints(args.distances)
    adaptive_thresholds = parse_csv_floats(args.adaptive_thresholds)

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
            )
            benchmark_rows.append(out)
            print(
                f"[{idx}/{total}] {row.decoder_label:<15} | d={d:>2} | rounds={rounds_i:>2} | "
                f"t={out['mean_avg_decode_time_sec']:.6f}s | ER={out['mean_error_rate']:.6f} | "
                f"sw={100.0 * out['mean_switch_rate']:.2f}%"
            )

    scaling_models = build_scaling_models(benchmark_rows)
    predicted_rows = build_predictions(
        scaling_models=scaling_models,
        target_distances=target_distances,
    )

    decoder_labels = [r.decoder_label for r in row_specs]
    compatibility = build_compatibility_report(
        predicted_rows=predicted_rows,
        decoder_labels=decoder_labels,
        distances=target_distances,
        safety_factor=float(args.safety_factor),
    )

    fig_path = save_heatmaps(
        compatibility=compatibility,
        output_path=Path(args.figure_output),
        title="Week 4 Hardware-Aware Decoder Compatibility",
    )

    report = build_report(
        benchmark_rows=benchmark_rows,
        scaling_models=scaling_models,
        predicted_rows=predicted_rows,
        compatibility=compatibility,
        args=args,
    )
    json_path = save_json(report, Path(args.output))
    print_summary(report)
    print(f"\nJSON saved at: {json_path}")
    print(f"Figure saved at: {fig_path}")


if __name__ == "__main__":
    main()
