# Week 5 - Person 2 Block 5 Report (Hardware Target-Latency Model)

## 1) Scope completed
- Implemented a Week 5 hardware analysis that goes beyond raw Python runtime.
- Added a target-aware latency model per architecture:
  - fixed overhead
  - detector-size penalty
  - proxy operation throughput (`ops/s`)
- Added side-by-side compatibility comparison:
  - `python_runtime` baseline (Week 4-style)
  - `target_model` (implementation-oriented)

## 2) Main artifacts
### Source
- `scripts/run_week5_person2_hardware_target_model.py`

### Tests
- `tests/test_week5_person2_hardware_target_model_smoke.py`

### Expected results
- `results/week5_person2_hardware_target_model.json`
- `figures/week5_person2_hardware_target_model_heatmaps.png`

## 3) Implemented Block 5 contract
- Benchmark collection on selected distances for:
  - `mwpm`, `uf`, optional `bm`, and `adaptive_g*`
- Python-runtime scaling model (`t(d)=a*d^b`) for baseline compatibility.
- Target-latency model from structural workload proxies:
  - circuit features: `num_detectors`, `dem_terms`
  - backend proxy ops:
    - `mwpm`: `k * dem_terms * log2(dem_terms)`
    - `uf`: `k * dem_terms`
    - `bm`: `k * dem_terms * log2(dem_terms)`
    - `adaptive`: weighted mix by predicted switch rate + dispatch ops
- Architecture-wise compatibility matrices for both models and delta summary.

## 4) Validation commands
- Smoke test:
  - `python -m pytest -q -p no:cacheprovider tests/test_week5_person2_hardware_target_model_smoke.py`
- Full run:
  - `python -m scripts.run_week5_person2_hardware_target_model --include-bm --adaptive-fast-mode --adaptive-fast-backend bm --output results/week5_person2_hardware_target_model.json --figure-output figures/week5_person2_hardware_target_model_heatmaps.png`

## 5) Validation status (executed)
- Date (UTC): `2026-02-20`
- Smoke test result:
  - `tests/test_week5_person2_hardware_target_model_smoke.py`: **3 passed**
- Full-run artifacts generated:
  - `results/week5_person2_hardware_target_model.json`
  - `figures/week5_person2_hardware_target_model_heatmaps.png`

## 6) Quantitative snapshot from full run
- Configuration:
  - benchmark distances: `5,7,9`
  - target distances: `5,7,9,11,13`
  - rows: `mwpm, uf, bm, adaptive_g{0.20,0.35,0.50,0.65,0.80}`
  - adaptive fast backend: `bm`
- Compatibility ratio by architecture:
  - `GoogleSuperconducting`: python `0.00%` -> target `62.50%` (delta `+62.50%`)
  - `IBMEagle`: python `0.00%` -> target `77.50%` (delta `+77.50%`)
  - `IonQForte`: python `0.00%` -> target `100.00%` (delta `+100.00%`)
  - `PsiQuantumPhotonic`: python `0.00%` -> target `0.00%` (delta `+0.00%`)

Interpretation:
- The target-aware model changes the hardware conclusion materially vs pure Python runtime.
- Ultra-strict photonic budgets remain incompatible under both models.

## 7) Status
Block 5 is implemented and validated with executable evidence (script + tests + full JSON/figure + report).
