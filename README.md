# qec-xzzx-adaptive-framework

A research framework for evaluating QEC decoders on XZZX-like memory circuits using `stim` and `pymatching`.

## Goal

Build a reproducible pipeline for:
- XZZX circuit generation (Stim rotated memory templates)
- noise model injection
- decoding with MWPM, Union-Find, Belief-Matching/BP, and adaptive switching
- benchmark execution and result reporting

## Current Status

- XZZX circuit generation: implemented
- Noise models (5): implemented
- Decoders:
  - MWPM with soft-info: implemented
  - Union-Find with soft-info: implemented
  - Belief-Matching/BP with soft-info: implemented (robust fallback included)
  - Adaptive (UF -> MWPM switch by threshold): implemented
- Test suite: passing in local conda environment `qec-xzzx`

## Repository Structure

```text
src/
  codes/        # circuit generation and core helpers
  noise/        # noise models and calibration utilities
  decoders/     # MWPM / UF / BM decoders
  switching/    # adaptive decoder logic
  pipelines/    # end-to-end pipelines
scripts/        # reproducible benchmark/report scripts
tests/          # unit + integration + smoke tests
results/        # benchmark JSON outputs
docs/           # reports and notes
```

## Requirements

- Python 3.11 recommended
- Dependencies listed in `requirements.txt`

Main dependencies:
- `numpy`
- `stim`
- `pymatching`
- `pytest`

## Quick Setup (Conda)

```bash
conda create -n qec-xzzx python=3.11 -y
conda activate qec-xzzx
python -m pip install -r requirements.txt
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

## Results

JSON outputs are written under `results/`.

## Notes

- Execute commands from the repository root (`from src...` imports are used).
- In VS Code, pin the interpreter in `.vscode/settings.json` if needed.
