# Paper Figure Set (Figure1..Figure6)

This document maps the generated figure artifacts to their scientific intent.

## Generation command

```bash
python -m scripts.generate_paper_figures --output-dir figures/paper --formats png,pdf
```

## Inputs

- `results/week3_person2_threshold_scan_paper_grade_r2.json`
- `results/week3_person2_threshold_scan_adaptive_g035_r2.json`
- `results/week3_person2_threshold_scan_adaptive_g065_r2.json`
- `results/week3_person2_threshold_scan_adaptive_g080_r2.json`
- `results/week5_person1_profile_scaling.json`
- `results/week5_person2_hardware_target_model.json`

## Output mapping

- `Figure1_threshold_curves`
  - LER vs `p` (log-scale), CI95 bands, one panel per decoder, depolarizing slice.

- `Figure2_adaptive_g_sensitivity`
  - Heatmaps for adaptive policy sensitivity across `g` and noise families:
    - switch rate,
    - decode time,
    - LER.

- `Figure3_tradeoff_vs_mwpm`
  - Pointwise trade-off against MWPM:
    - x-axis: speedup vs MWPM,
    - y-axis: `ΔLER = LER_decoder - LER_mwpm`,
    - one panel per decoder (`uf`, `bm`, `adaptive`).

- `Figure4_scaling_time_memory`
  - Log-log scaling curves from Week 5 profiling:
    - time vs distance,
    - peak memory vs distance,
    - with fitted power-law overlays.

- `Figure5_hardware_runtime_vs_target`
  - Architecture-wise compatibility matrices:
    - row 1: Python-runtime inferred compatibility,
    - row 2: target-model compatibility.

- `Figure6_uncertainty_thresholds`
  - CI95 distribution and threshold-estimate landscape:
    - ER CI95 half-width density by decoder,
    - crossing-based threshold estimates by decoder/noise.
