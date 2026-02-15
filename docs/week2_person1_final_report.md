# Week 2 — Person 1 Final Report (UF + Adaptive + Paired Sweep)

## Scope completed
- Implemented Union-Find decoder wrapper with soft-info interface.
- Implemented Adaptive decoder (UF fast path + MWPM fallback by confidence threshold).
- Added decoder-comparison benchmark script.
- Added paired threshold sweep script (same sampled syndromes for all decoders).
- Added smoke tests for script-level reliability.

## Main artifacts
### Source
- `src/decoders/union_find_decoder.py`
- `src/switching/adaptive_decoder.py`
- `scripts/run_week2_person1_decoder_comparison.py`
- `scripts/run_week2_person1_paired_threshold_sweep.py`

### Tests
- `tests/test_union_find_decoder.py`
- `tests/test_adaptive_decoder.py`
- `tests/test_week2_person1_paired_sweep_smoke.py`

### Results
- `results/week2_person1_decoder_comparison.json`
- `results/week2_person1_decoder_comparison_2k_g035.json`
- `results/week2_cmp_g020.json`
- `results/week2_cmp_g040.json`
- `results/week2_cmp_g060.json`
- `results/week2_cmp_g080.json`
- `results/week2_person1_paired_threshold_sweep.json`

## Validation status
- Full selected suite passed: **50/50 tests**.

## Canonical evidence (paired comparison)
Run:
- `scripts/run_week2_person1_paired_threshold_sweep.py`
- shots = 2000
- thresholds = [0.20, 0.35, 0.40, 0.60, 0.80]
- cases:
  - d=3, r=2, p=0.005
  - d=3, r=3, p=0.010
  - d=5, r=3, p=0.010

### Aggregate metrics
- mean_mwpm_error_rate: **0.014000**
- mean_uf_error_rate: **0.014000**
- mean_mwpm_avg_decode_time_sec: **0.000018**
- mean_uf_avg_decode_time_sec: **0.000025**
- mean_uf_speedup_vs_mwpm: **0.698228**

Adaptive (means by threshold):
- g=0.20 → mean_ER=0.014000, mean_t=0.000088, mean_sw=0.00%, mean_spd_vs_mwpm=0.199074
- g=0.35 → mean_ER=0.014000, mean_t=0.000088, mean_sw=0.02%, mean_spd_vs_mwpm=0.200943
- g=0.40 → mean_ER=0.014000, mean_t=0.000088, mean_sw=0.27%, mean_spd_vs_mwpm=0.200202
- g=0.60 → mean_ER=0.014000, mean_t=0.000090, mean_sw=3.77%, mean_spd_vs_mwpm=0.195808
- g=0.80 → mean_ER=0.014000, mean_t=0.000105, mean_sw=25.95%, mean_spd_vs_mwpm=0.169431

## Interpretation
- UF backend is active and correctly used.
- Adaptive switching behaves as expected with threshold changes.
- For current implementation + instance sizes, MWPM is still faster in wall-clock time.
- Functional objective is complete; remaining objective is performance optimization.

## Recommended default for current branch
- `g_threshold = 0.35` (very low switch rate, best adaptive time among tested thresholds, stable ER).

## Next technical sprint (performance-focused)
1. Add fast-path benchmark mode in AdaptiveDecoder (minimal dict creation / optional soft-info off).
2. Reuse preallocated buffers in loop-level benchmark code.
3. Profile decode path (`cProfile` / `py-spy`) to isolate Python overhead.
4. Re-run paired sweep on larger d/rounds to evaluate scaling crossover.
