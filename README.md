# qec-xzzx-adaptive-framework

A research framework for evaluating QEC decoders on XZZX-like memory circuits using `stim` and `pymatching`.

## Goal

Build a reproducible pipeline for:
- XZZX memory-circuit generation (Stim rotated-memory templates)
- noise model injection and calibration workflows
- decoding with MWPM, Union-Find, Belief-Matching/BP, and adaptive switching
- benchmark execution, profiling, scaling fits, and hardware-oriented reporting

## Current Status

- XZZX circuit generation: implemented
- Noise model workflows: implemented (depolarizing / biased / circuit-level / correlated families used in scripts)
- Decoders:
  - MWPM with soft-info: implemented
  - Union-Find with soft-info: implemented
  - Belief-Matching/BP with soft-info: implemented (fallbacks included)
  - Adaptive switching: implemented (threshold-based, configurable fast backend `uf`/`bm`, `fast_mode`)
- Week 1-5 benchmark/report scripts: implemented
- Tests: unit + integration + smoke tests (`156 passed` in local conda environment `qec-xzzx`)

## Repository Structure

```text
src/
  codes/        # circuit generation and core helpers
  noise/        # noise models and calibration utilities
  decoders/     # MWPM / UF / BM decoders
  switching/    # adaptive decoder logic
  hardware/     # hardware latency/compatibility helpers
  pipelines/    # end-to-end pipelines (reserved/experimental)
scripts/        # reproducible benchmark/report scripts (weeks 1-5)
tests/          # unit + integration + smoke tests
results/        # benchmark JSON outputs
docs/           # reports and notes
figures/        # generated figures/plots
```

## Requirements

- Python 3.11 recommended
- Base dependencies listed in `requirements.txt`

Base dependencies:
- `numpy`
- `stim`
- `pymatching`
- `pytest`

Optional plotting dependencies (needed for some Week 4/5 scripts):
- `matplotlib` (required for hardware compatibility/target-model figures)
- `seaborn` (optional; scripts fall back if unavailable)

## Quick Setup (Conda)

```bash
conda create -n qec-xzzx python=3.11 -y
conda activate qec-xzzx
python -m pip install -r requirements.txt
```

Optional extras for plotting scripts:

```bash
python -m pip install matplotlib seaborn
```

## Run Tests

```bash
python -m pytest -q
```

## Main Scripts

### Week 1 - baseline benchmark

```bash
python -m scripts.run_week1_person1_benchmark
```

### Week 1 - noise benchmark

```bash
python -m scripts.run_week1_person2_noise_benchmark --shots 400
```

### Week 2 - decoder comparison

```bash
python -m scripts.run_week2_person1_decoder_comparison --shots 300 --g-threshold 0.65
```

### Week 2 - paired threshold sweep

```bash
python -m scripts.run_week2_person1_paired_threshold_sweep --shots 2000 --thresholds 0.20,0.35,0.40,0.60,0.80
```

### Week 2 - noise calibration

```bash
python -m scripts.run_week2_person2_noise_calibration --quick
```

### Week 3 - adaptive fast_mode benchmark

```bash
python -m scripts.run_week3_person1_fastmode_benchmark --shots 400 --repeats 3 --g-threshold 0.35
```

### Week 3 - adaptive profile (cProfile)

```bash
python -m scripts.run_week3_person1_profile_adaptive --distance 7 --rounds 5 --p 0.01 --shots 500 --g-threshold 0.35
```

### Week 3 - scaling benchmark (larger d/rounds)

```bash
python -m scripts.run_week3_person1_scaling_benchmark --shots 300 --repeats 2 --g-threshold 0.35
```

### Week 3 - threshold scan (person 2)

```bash
python -m scripts.run_week3_person2_threshold_scan --distances 3,5,7 --p-values 0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02 --decoders mwpm,uf,bm,adaptive --noise-models depolarizing,biased_eta10,biased_eta100,biased_eta500,circuit_level,correlated --shots 300
```

### Week 4 - hardware compatibility report

```bash
python -m scripts.run_week4_hardware_compatibility --shots 120 --repeats 2
```

### Week 5 - adaptive policy tuning

```bash
python -m scripts.run_week5_person1_adaptive_policy_tuning --shots 150 --repeats 3 --time-metric core
```

### Week 5 - profiling + scaling (time/memory)

```bash
python -m scripts.run_week5_person1_profile_scaling --shots 200 --repeats 2 --profile-shots 120
```

### Week 5 - hardware target model / trace-calibrated report

```bash
python -m scripts.run_week5_person2_hardware_target_model --shots 120 --repeats 2
```

## Output Conventions

- JSON outputs are written under `results/`.
- Several adaptive benchmark reports explicitly store `adaptive_benchmark_time_metric` (currently `core`) in report metadata/config for reproducibility.
- Some scripts also generate figures under `figures/` (or custom output paths).

## Notes

- Run commands from the repository root (`from src...` imports are used).
- In VS Code, pin the interpreter in `.vscode/settings.json` if needed.
- For reproducible comparisons, keep seeds/time metrics consistent across runs and preserve generated JSON reports with their config blocks.
