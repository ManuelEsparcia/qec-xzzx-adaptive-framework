# Week 5 - Person 1 Block 4 Report (Adaptive Strategy Tuning)

## 1) Scope completed
- Extended adaptive policy with a syndrome-weight gate:
  - `min_syndrome_weight_for_switch`
  - switching now requires both: low confidence (`confidence < g_threshold`) and gate pass (`syndrome_weight >= min_switch_weight`, if set).
- Implemented a dedicated Block 4 tuning pipeline to optimize policy settings for:
  - speedup vs MWPM
  - bounded error-rate degradation
- Added tests for both:
  - adaptive core behavior
  - tuning script smoke/failure path

## 2) Main artifacts
### Source
- `src/switching/adaptive_decoder.py`
- `scripts/run_week5_person1_adaptive_policy_tuning.py`

### Tests
- `tests/test_adaptive_decoder.py`
- `tests/test_week5_person1_adaptive_policy_tuning_smoke.py`

### Expected results output
- `results/week5_person1_adaptive_policy_tuning.json`

## 3) Implemented Block 4 contract
- Policy grid dimensions:
  - threshold grid: `--g-thresholds`
  - syndrome-weight gate grid: `--min-switch-weights` (`none` or integer)
  - mode: `standard` or `fast`
- Per `(case, policy)` metrics with repeats + CI:
  - `error_rate_adaptive`
  - `error_rate_mwpm`
  - `delta_error_rate_adaptive_minus_mwpm`
  - `avg_decode_time_adaptive_sec`
  - `avg_decode_time_mwpm_sec`
  - `speedup_vs_mwpm`
  - `switch_rate`
- Feasibility constraints:
  - `delta_error <= max_delta_error`
  - `speedup >= min_speedup`
- Ranking outputs:
  - best global policy (`best_global_policy`)
  - best policy per case (`best_policy_by_case`)
  - aggregated policy table (`policy_summary`)

## 4) Validation commands
- Core adaptive tests:
  - `python -m pytest -q -p no:cacheprovider tests/test_adaptive_decoder.py`
- Script smoke test:
  - `python -m pytest -q -p no:cacheprovider tests/test_week5_person1_adaptive_policy_tuning_smoke.py`
- Full run:
  - `python -m scripts.run_week5_person1_adaptive_policy_tuning --distances 5,7,9,11,13 --fast-backends uf,bm --g-thresholds 0.20,0.35,0.50,0.65,0.80 --min-switch-weights none,1,2,3 --mode fast --shots 120 --repeats 2 --max-delta-error 0.01 --min-speedup 1.0 --output results/week5_person1_adaptive_policy_tuning.json`

## 5) Validation status (executed)
- Date (UTC): `2026-02-20`
- Test results:
  - `tests/test_adaptive_decoder.py` + `tests/test_week5_person1_adaptive_policy_tuning_smoke.py`: **19 passed**
- Evidence outputs:
  - `results/_verify_week5_person1_adaptive_policy_tuning.json` (quick verification grid)
  - `results/week5_person1_adaptive_policy_tuning.json` (full Block 4 grid)

## 6) Quantitative snapshot from full run
- Full grid:
  - distances: `5,7,9,11,13`
  - fast backends: `uf,bm`
  - thresholds: `0.20,0.35,0.50,0.65,0.80`
  - min switch weights: `none,1,2,3`
  - total candidates: `200`
- Under strict feasibility (`max_delta_error=0.01`, `min_speedup=1.0`):
  - feasible policies found: **0**
  - best global policy by ranking fallback: `bm_g0.20_w1_fast`
  - `mean_speedup_vs_mwpm`: `0.630`
  - `mean_delta_error_rate_adaptive_minus_mwpm`: `0.000000`
- Interpretation:
  - The new policy gate controls switching while preserving error parity.
  - In current Python/runtime regime, adaptive remains slower than MWPM in decode-time metric (speedup < 1 across grid).

## 7) Status
Block 4 is implemented and validated with executable evidence (core policy extension + tuner + tests + quick/full outputs + report).
