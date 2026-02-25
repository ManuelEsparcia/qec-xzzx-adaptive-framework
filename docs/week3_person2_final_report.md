# Week 3 - Person 2 Final Report (Massive Simulations + Threshold Scan)

## 1) Scope completed
- Implemented a Week 3 threshold-scan runner aligned with roadmap "Simulaciones Masivas":
  - expanded grid over decoder/noise/distance/physical-error-rate,
  - checkpointed JSON saves for long runs,
  - automatic threshold estimation from distance-curve crossings.
- Added smoke tests for script reliability and argument validation.

## 2) Main artifacts
### Source
- `scripts/run_week3_person2_threshold_scan.py`

### Tests
- `tests/test_week3_person2_threshold_scan_smoke.py`

### Results
- `results/week3_person2_threshold_scan_quick.json`
- `results/week3_person2_threshold_scan_verify.json`
- `results/week3_person2_threshold_scan_full.json`

## 3) Implemented roadmap behaviors
- Grid scan over:
  - decoders: MWPM, UF, BM/BP, Adaptive (configurable)
  - distances: CSV input (default `3,5,7`)
  - noise models: depolarizing, biased (`eta=10/100/500`), circuit-level, correlated
  - physical error rates `p`: CSV input (default roadmap-style range)
- Per-point metrics:
  - `error_rate`
  - `avg_decode_time_sec`
  - `switch_rate` (adaptive)
- Checkpoints:
  - partial write every `N` points via `--checkpoint-every`
- Threshold fitting:
  - crossing estimate between smallest/largest distance curves per `(decoder, noise_model)`
  - methods: `exact_point`, `linear_crossing`, `nearest_abs_diff` fallback

## 4) Validation status
- `python -m pytest -q -p no:cacheprovider tests/test_week3_person2_threshold_scan_smoke.py`
  - **3 passed**
- Quick execution evidence:
  - `python -m scripts.run_week3_person2_threshold_scan --distances 3,5 --rounds 2 --p-values 0.005,0.01 --decoders mwpm,uf,adaptive --noise-models depolarizing,biased_eta10,correlated --shots 40 --adaptive-fast-mode --checkpoint-every 4 --output results/week3_person2_threshold_scan_quick.json`
  - output generated successfully.

## 5) Quick evidence snapshot
From `results/week3_person2_threshold_scan_quick.json`:
- total points: `36`
- threshold estimates generated for all `(decoder, noise_model)` combinations in the run
- contract includes:
  - `points`
  - `aggregates.pair_summary`
  - `aggregates.pareto_reference_points`
  - `threshold_estimates`

## 6) Recommended next run (full Week 3 grid)
Use:
- `--distances 3,5,7`
- `--p-values 0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02`
- `--decoders mwpm,uf,bm,adaptive`
- `--noise-models depolarizing,biased_eta10,biased_eta100,biased_eta500,circuit_level,correlated`
- `--shots 300`
- `--checkpoint-every 24`

## 7) Full run closure (executed)
Execution date:
- **2026-02-19**

Command used:
- `python -m scripts.run_week3_person2_threshold_scan --distances 3,5,7 --p-values 0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02 --decoders mwpm,uf,bm,adaptive --noise-models depolarizing,biased_eta10,biased_eta100,biased_eta500,circuit_level,correlated --shots 300 --checkpoint-every 24 --adaptive-fast-mode --output results/week3_person2_threshold_scan_full.json`

Full-run output:
- `results/week3_person2_threshold_scan_full.json`

Closure metrics:
- total points: `576`
- threshold estimates: `24`
- point status: `576/576 ok`
- metadata partial flag: `False`

Validated config in output:
- distances: `3,5,7`
- p-values: `0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02`
- decoders: `mwpm,uf,bm,adaptive`
- noise models: `depolarizing,biased_eta10,biased_eta100,biased_eta500,circuit_level,correlated`
- shots: `300`
- rounds: `3`

Post-run test check:
- `python -m pytest -q -p no:cacheprovider tests/test_week3_person2_threshold_scan_smoke.py`
  - **3 passed**

## 8) Status
Week 3 Person 2 is now fully completed (implementation + validation + full-scale execution evidence).
