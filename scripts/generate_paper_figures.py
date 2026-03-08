"""Generate publication-ready figures from canonical JSON artifacts.

This script builds a consistent Figure1..Figure6 set (PNG+PDF by default)
using the current paper-grade and hardware artifacts in `results/`.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


DECODER_COLORS = {
    "mwpm": "#1f77b4",
    "uf": "#ff7f0e",
    "bm": "#2ca02c",
    "adaptive": "#d62728",
}

NOISE_COLORS = {
    "depolarizing": "#4C72B0",
    "biased_eta10": "#55A868",
    "biased_eta100": "#C44E52",
    "biased_eta500": "#8172B2",
    "circuit_level": "#CCB974",
    "correlated": "#64B5CD",
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "savefig.bbox": "tight",
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_figure(
    fig: plt.Figure,
    stem: str,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[Path]:
    paths: list[Path] = []
    for fmt in formats:
        fmt = fmt.strip().lower()
        if not fmt:
            continue
        p = output_dir / f"{stem}.{fmt}"
        if fmt == "png":
            fig.savefig(p, dpi=dpi)
        else:
            fig.savefig(p)
        paths.append(p)
    return paths


def _decoder_label(label: str) -> str:
    return {
        "mwpm": "MWPM",
        "uf": "Union-Find",
        "bm": "Belief-Matching",
        "adaptive": "Adaptive",
    }.get(label, label)


def _noise_label(label: str) -> str:
    return label.replace("_", " ")


def _build_point_index(points: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    idx: dict[tuple[Any, ...], dict[str, Any]] = {}
    for p in points:
        key = (p["decoder"], p["noise_model"], p["distance"], p["p_phys"])
        idx[key] = p
    return idx


def figure1_threshold_curves(scan: dict[str, Any], noise_model: str = "depolarizing") -> plt.Figure:
    points = scan["points"]
    decoders = list(scan["config"]["decoders"])
    distances = sorted(scan["config"]["distances"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes_list = axes.flatten()

    distance_palette = plt.cm.viridis(np.linspace(0.15, 0.9, len(distances)))

    for ax, dec in zip(axes_list, decoders):
        for dist, c in zip(distances, distance_palette):
            rows = [
                p
                for p in points
                if p["decoder"] == dec and p["noise_model"] == noise_model and p["distance"] == dist
            ]
            rows.sort(key=lambda r: float(r["p_phys"]))
            x = np.array([r["p_phys"] for r in rows], dtype=float)
            y = np.array([r["error_rate"] for r in rows], dtype=float)
            ci = np.array([r.get("error_rate_ci95_half_width", 0.0) for r in rows], dtype=float)

            ax.plot(x, y, marker="o", markersize=3.5, linewidth=1.7, color=c, label=f"d={dist}")
            ax.fill_between(x, np.clip(y - ci, 0.0, 1.0), np.clip(y + ci, 0.0, 1.0), color=c, alpha=0.16)

        ax.set_xscale("log")
        ax.set_title(_decoder_label(dec))
        ax.set_xlabel("Physical Error Rate p")
        ax.set_ylabel("Logical Error Rate (LER)")
        ax.set_ylim(-0.01, 0.72)

    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), frameon=False)
    fig.suptitle(
        f"Figure 1: Threshold Curves with CI95 Bands ({_noise_label(noise_model)})\n"
        "Data: week3_person2_threshold_scan_paper_grade_r2.json",
        y=1.03,
    )
    fig.tight_layout()
    return fig


def _adaptive_pair_summary(scan: dict[str, Any]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for row in scan["aggregates"]["pair_summary"]:
        noise = row["noise_model"]
        summary[noise] = {
            "mean_error_rate": float(row["mean_error_rate"]),
            "mean_avg_decode_time_sec": float(row["mean_avg_decode_time_sec"]),
            "mean_switch_rate": float(row.get("mean_switch_rate", 0.0)),
        }
    return summary


def figure2_g_sensitivity(scans: dict[float, dict[str, Any]]) -> plt.Figure:
    g_values = sorted(scans.keys())
    noise_models = list(scans[g_values[0]]["config"]["noise_models"])

    sw = np.zeros((len(noise_models), len(g_values)))
    tm = np.zeros((len(noise_models), len(g_values)))
    er = np.zeros((len(noise_models), len(g_values)))

    for j, g in enumerate(g_values):
        s = _adaptive_pair_summary(scans[g])
        for i, n in enumerate(noise_models):
            sw[i, j] = s[n]["mean_switch_rate"]
            tm[i, j] = s[n]["mean_avg_decode_time_sec"]
            er[i, j] = s[n]["mean_error_rate"]

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6))
    panels = [
        (sw, "Mean Switch Rate", "magma"),
        (tm * 1e6, "Mean Decode Time (µs)", "viridis"),
        (er, "Mean LER", "plasma"),
    ]

    for ax, (arr, title, cmap) in zip(axes, panels):
        im = ax.imshow(arr, aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(range(len(g_values)))
        ax.set_xticklabels([f"{g:.2f}" for g in g_values])
        ax.set_yticks(range(len(noise_models)))
        ax.set_yticklabels([_noise_label(n) for n in noise_models])
        ax.set_xlabel("g-threshold")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                text = f"{arr[i, j]:.3f}" if title != "Mean Decode Time (µs)" else f"{arr[i, j]:.1f}"
                ax.text(j, i, text, ha="center", va="center", color="white", fontsize=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=8)

    fig.suptitle("Figure 2: Adaptive g-threshold Sensitivity (switch/time/ER)", y=1.04)
    fig.tight_layout()
    return fig


def figure3_tradeoff(scan: dict[str, Any]) -> plt.Figure:
    points = scan["points"]
    idx = _build_point_index(points)
    decoders = ["uf", "bm", "adaptive"]

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4), sharey=True, sharex=True)
    noise_order = list(scan["config"]["noise_models"])

    for ax, dec in zip(axes, decoders):
        xs: list[float] = []
        ys: list[float] = []

        for n in noise_order:
            x_noise: list[float] = []
            y_noise: list[float] = []
            for d in scan["config"]["distances"]:
                for p in scan["config"]["p_values"]:
                    mw = idx.get(("mwpm", n, d, p))
                    dv = idx.get((dec, n, d, p))
                    if mw is None or dv is None:
                        continue
                    speedup = mw["avg_decode_time_sec"] / dv["avg_decode_time_sec"]
                    delta_ler = dv["error_rate"] - mw["error_rate"]
                    x_noise.append(speedup)
                    y_noise.append(delta_ler)

            if x_noise:
                xs.extend(x_noise)
                ys.extend(y_noise)
                ax.scatter(
                    x_noise,
                    y_noise,
                    s=18,
                    alpha=0.50,
                    color=NOISE_COLORS.get(n, "#555555"),
                    label=_noise_label(n),
                    edgecolors="none",
                )

        ax.axvline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
        if xs:
            ax.scatter(np.mean(xs), np.mean(ys), marker="X", s=120, color=DECODER_COLORS.get(dec, "#222222"))
        ax.set_title(_decoder_label(dec))
        ax.set_xlabel("Speedup vs MWPM ( >1 faster )")

    axes[0].set_ylabel("ΔLER vs MWPM ( <0 better )")
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Figure 3: Accuracy-Speed Trade-off Relative to MWPM", y=1.05)
    fig.tight_layout()
    return fig


def _extract_profile_rows(profile: dict[str, Any], metric_name: str) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    grouped: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for row in profile["rows"]:
        d = row["decoder"]
        m = row["metrics"][metric_name]
        grouped[d].append((int(row["distance"]), float(m["mean"]), float(m.get("ci95_half_width", 0.0))))

    for dec, triples in grouped.items():
        triples.sort(key=lambda x: x[0])
        dist = np.array([t[0] for t in triples], dtype=float)
        val = np.array([t[1] for t in triples], dtype=float)
        ci = np.array([t[2] for t in triples], dtype=float)
        out[dec] = {"distance": dist, "value": val, "ci": ci}
    return out


def figure4_scaling(profile: dict[str, Any]) -> plt.Figure:
    time_rows = _extract_profile_rows(profile, "avg_decode_time_sec")
    mem_rows = _extract_profile_rows(profile, "memory_peak_bytes")
    fits = {m["decoder"]: m for m in profile["scaling_models"]}

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    panels = [
        (axes[0], time_rows, "Time Scaling", "Mean Decode Time (s)", "time_fit"),
        (axes[1], mem_rows, "Memory Scaling", "Peak Memory (bytes)", "memory_fit"),
    ]

    d_smooth = np.linspace(3, 13, 200)
    for ax, rows, title, ylabel, fit_key in panels:
        for dec in ["mwpm", "uf", "bm", "adaptive"]:
            if dec not in rows:
                continue
            x = rows[dec]["distance"]
            y = rows[dec]["value"]
            ci = rows[dec]["ci"]
            c = DECODER_COLORS.get(dec, "#444444")
            ax.errorbar(x, y, yerr=ci, marker="o", linewidth=1.8, markersize=4, capsize=2, color=c, label=_decoder_label(dec))
            fit = fits.get(dec, {}).get(fit_key, {})
            a = fit.get("coefficient")
            b = fit.get("exponent")
            if a is not None and b is not None:
                y_fit = a * np.power(d_smooth, b)
                ax.plot(d_smooth, y_fit, linestyle="--", linewidth=1.2, color=c, alpha=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Code Distance d")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Figure 4: Profiling/Scaling Fits (Week 5)", y=1.05)
    fig.tight_layout()
    return fig


def figure5_hardware(target_model: dict[str, Any]) -> plt.Figure:
    comp = target_model["compatibility"]
    decoders = comp["decoder_labels"]
    distances = comp["distances"]
    archs = comp["architectures"]

    fig, axes = plt.subplots(2, len(archs), figsize=(4.0 * len(archs), 8), sharex=True, sharey=True)
    cmap = ListedColormap(["#B22222", "#2E8B57"])

    if len(archs) == 1:
        axes = np.array([[axes[0]], [axes[1]]])  # pragma: no cover

    for j, arch in enumerate(archs):
        py_mat = np.array(arch["python_runtime_matrix"], dtype=float)
        tg_mat = np.array(arch["target_model_matrix"], dtype=float)
        for i, (mat, row_name) in enumerate([(py_mat, "Python Runtime"), (tg_mat, "Target Model")]):
            ax = axes[i, j]
            ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
            ax.set_title(f"{arch['architecture']}\n{row_name}", fontsize=10)
            ax.set_xticks(range(len(distances)))
            ax.set_xticklabels([str(d) for d in distances], rotation=0)
            if j == 0:
                ax.set_yticks(range(len(decoders)))
                ax.set_yticklabels([d.replace("adaptive_", "adp_") for d in decoders])
                ax.set_ylabel("Decoder")
            else:
                ax.set_yticks(range(len(decoders)))
                ax.set_yticklabels([])
            if i == 1:
                ax.set_xlabel("Distance")
            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    ax.text(c, r, str(int(mat[r, c])), ha="center", va="center", color="white", fontsize=7)

    fig.suptitle("Figure 5: Hardware Compatibility (Python Runtime vs Target Model)", y=0.99)
    fig.tight_layout()
    return fig


def figure6_uncertainty_thresholds(scan: dict[str, Any]) -> plt.Figure:
    points = scan["points"]
    thresholds = scan["threshold_estimates"]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))

    # Panel A: CI distribution by decoder
    for dec in scan["config"]["decoders"]:
        vals = [p.get("error_rate_ci95_half_width", 0.0) for p in points if p["decoder"] == dec]
        if not vals:
            continue
        axes[0].hist(
            vals,
            bins=28,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=DECODER_COLORS.get(dec, "#555555"),
            label=_decoder_label(dec),
        )
    axes[0].axvline(0.02, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[0].axvline(0.03, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[0].set_xlabel("ER CI95 Half-Width")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Uncertainty Distribution")
    axes[0].legend(frameon=False)

    # Panel B: threshold estimates by decoder/noise
    decoders = scan["config"]["decoders"]
    x_pos = {d: i for i, d in enumerate(decoders)}
    rng = np.random.default_rng(20260309)
    for row in thresholds:
        dec = row["decoder"]
        n = row["noise_model"]
        x = x_pos[dec] + rng.uniform(-0.16, 0.16)
        y = float(row["p_threshold_estimate"])
        axes[1].scatter(x, y, s=30, alpha=0.75, color=NOISE_COLORS.get(n, "#666666"), edgecolors="white", linewidths=0.3)
    axes[1].set_xticks(range(len(decoders)))
    axes[1].set_xticklabels([_decoder_label(d) for d in decoders], rotation=0)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Estimated Threshold p*")
    axes[1].set_title("Threshold Estimates (distance crossings)")
    p_vals = sorted(scan["config"]["p_values"])
    axes[1].axhline(p_vals[0], color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[1].axhline(p_vals[-1], color="black", linestyle="--", linewidth=1.0, alpha=0.6)

    # Legend (unique noise labels)
    handles = []
    labels = []
    for n in scan["config"]["noise_models"]:
        handles.append(plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=NOISE_COLORS.get(n, "#666666"), markersize=6))
        labels.append(_noise_label(n))
    axes[1].legend(handles, labels, frameon=False, fontsize=8, loc="upper left")

    fig.suptitle("Figure 6: CI Quality and Threshold-Estimate Landscape", y=1.03)
    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Figure1..Figure6 publication plots.")
    parser.add_argument(
        "--paper-scan",
        type=Path,
        default=Path("results/week3_person2_threshold_scan_paper_grade_r2.json"),
        help="Path to canonical paper-grade week3 scan JSON.",
    )
    parser.add_argument(
        "--adaptive-g035",
        type=Path,
        default=Path("results/week3_person2_threshold_scan_adaptive_g035_r2.json"),
        help="Path to adaptive-only week3 scan with g=0.35.",
    )
    parser.add_argument(
        "--adaptive-g065",
        type=Path,
        default=Path("results/week3_person2_threshold_scan_adaptive_g065_r2.json"),
        help="Path to adaptive-only week3 scan with g=0.65.",
    )
    parser.add_argument(
        "--adaptive-g080",
        type=Path,
        default=Path("results/week3_person2_threshold_scan_adaptive_g080_r2.json"),
        help="Path to adaptive-only week3 scan with g=0.80.",
    )
    parser.add_argument(
        "--profile-scaling",
        type=Path,
        default=Path("results/week5_person1_profile_scaling.json"),
        help="Path to week5 profile/scaling JSON.",
    )
    parser.add_argument(
        "--hardware-target",
        type=Path,
        default=Path("results/week5_person2_hardware_target_model.json"),
        help="Path to week5 hardware target-model JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/paper"),
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="png,pdf",
        help="Comma-separated output formats.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
        help="DPI for raster formats (PNG).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _configure_style()

    paper = _load_json(args.paper_scan)
    g035 = _load_json(args.adaptive_g035)
    g065 = _load_json(args.adaptive_g065)
    g080 = _load_json(args.adaptive_g080)
    profile = _load_json(args.profile_scaling)
    hardware = _load_json(args.hardware_target)

    formats = [f.strip() for f in args.formats.split(",")]
    outputs: list[Path] = []

    fig1 = figure1_threshold_curves(paper, noise_model="depolarizing")
    outputs.extend(_save_figure(fig1, "Figure1_threshold_curves", args.output_dir, formats, args.dpi))
    plt.close(fig1)

    fig2 = figure2_g_sensitivity({0.35: g035, 0.65: g065, 0.80: g080})
    outputs.extend(_save_figure(fig2, "Figure2_adaptive_g_sensitivity", args.output_dir, formats, args.dpi))
    plt.close(fig2)

    fig3 = figure3_tradeoff(paper)
    outputs.extend(_save_figure(fig3, "Figure3_tradeoff_vs_mwpm", args.output_dir, formats, args.dpi))
    plt.close(fig3)

    fig4 = figure4_scaling(profile)
    outputs.extend(_save_figure(fig4, "Figure4_scaling_time_memory", args.output_dir, formats, args.dpi))
    plt.close(fig4)

    fig5 = figure5_hardware(hardware)
    outputs.extend(_save_figure(fig5, "Figure5_hardware_runtime_vs_target", args.output_dir, formats, args.dpi))
    plt.close(fig5)

    fig6 = figure6_uncertainty_thresholds(paper)
    outputs.extend(_save_figure(fig6, "Figure6_uncertainty_thresholds", args.output_dir, formats, args.dpi))
    plt.close(fig6)

    print("\nGenerated figure files:")
    for p in outputs:
        print(f"- {p}")


if __name__ == "__main__":
    main()
