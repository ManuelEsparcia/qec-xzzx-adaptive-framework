# Week 3 - Person 1 Final Report (Adaptive Performance Sprint)

## 1) Scope completed
- Added low-overhead benchmark path in adaptive decoder (`fast_mode=True`).
- Added paired benchmark script to compare `fast_mode=False` vs `fast_mode=True`.
- Added dedicated cProfile script to identify runtime hotspots.
- Added larger-case scaling benchmark script for `d/r` growth analysis.
- Added smoke tests for all Week 3 scripts.

## 2) Main artifacts
### Source
- `src/switching/adaptive_decoder.py`
- `scripts/run_week3_person1_fastmode_benchmark.py`
- `scripts/run_week3_person1_profile_adaptive.py`
- `scripts/run_week3_person1_scaling_benchmark.py`

### Tests
- `tests/test_adaptive_decoder.py`
- `tests/test_week3_person1_fastmode_benchmark_smoke.py`
- `tests/test_week3_person1_profile_adaptive_smoke.py`
- `tests/test_week3_person1_scaling_benchmark_smoke.py`

### Results
- `results/week3_person1_fastmode_benchmark.json`
- `results/week3_person1_adaptive_profile.json`
- `results/week3_person1_scaling_benchmark.json`

## 3) Validation status
- Week 3 smoke suite:
  - `tests/test_week3_person1_fastmode_benchmark_smoke.py`
  - `tests/test_week3_person1_profile_adaptive_smoke.py`
  - `tests/test_week3_person1_scaling_benchmark_smoke.py`
  - Result: **all passing**
- Full project suite executed after Week 3 changes:
  - `python -m pytest -q -p no:cacheprovider`
  - Result: **passing**

## 4) Quantitative evidence
## 4.1 cProfile evidence (d=7, r=5, p=0.01, shots=200, g=0.35)
From `results/week3_person1_adaptive_profile.json`:
- standard wall time: `0.081895 s`
- fast wall time: `0.075015 s`
- wall speedup fast/standard: `1.091715`
- standard avg decode time: `0.000328 s`
- fast avg decode time: `0.000308 s`
- avg decode speedup fast/standard: `1.065149`

Top cumulative hotspots were consistently:
- `AdaptiveDecoder.benchmark_adaptive(...)`
- `AdaptiveDecoder.decode_adaptive(...)` or `_decode_adaptive_fastpath(...)`
- `UnionFindDecoderWithSoftInfo.decode_with_confidence(...)`
- `pymatching.Matching.decode(...)`

Interpretation:
- Python loop + adaptive orchestration overhead is significant.
- Decoder backend (`pymatching.decode`) is still a dominant hotspot.

## 4.2 Scaling benchmark evidence (d=5/7/9, shots=200, repeats=2, g=0.35)
From `results/week3_person1_scaling_benchmark.json`:
- mean_mwpm_avg_decode_time_sec: `0.000097`
- mean_uf_avg_decode_time_sec: `0.000221`
- mean_adaptive_standard_avg_decode_time_sec: `0.000360`
- mean_adaptive_fast_avg_decode_time_sec: `0.000383`
- mean_speedup_fast_vs_standard_decode_time: `0.947685`
- mean_speedup_fast_vs_standard_wall_time: `0.993937`
- mean_adaptive_standard_error_rate: `0.020833`
- mean_adaptive_fast_error_rate: `0.020833`

Interpretation:
- Functional parity is preserved (`delta error rate ~ 0`).
- Fast mode advantage is workload-dependent; it improves some cases but not all.
- At tested sizes and threshold, adaptive remains slower than MWPM in decode-time metric.

## 4.3 Fast-mode benchmark evidence (baseline cases, shots=200, repeats=3, g=0.35)
From `results/week3_person1_fastmode_benchmark.json`:
- mean_standard_avg_decode_time_sec: `0.000111`
- mean_fast_avg_decode_time_sec: `0.000118`
- mean_time_speedup_fast_vs_standard: `0.937611`
- mean_standard_error_rate: `0.012778`
- mean_fast_error_rate: `0.012778`

Interpretation:
- Error-rate equivalence is maintained across modes.
- For these smaller baseline cases, `fast_mode` did not provide net speedup.
- This is consistent with a regime where backend decode dominates and Python-object savings are limited.

## 5) Recommended defaults (current branch)
- `g_threshold = 0.35` (stable switching behavior in current evidence).
- `fast_mode = True` for benchmark/profiling experiments.
- Keep paired repeated runs (`repeats >= 2`) for reliable timing comparisons.

## 6) Week 3 completion status
- Week 3 Person 1 objectives are completed for this sprint:
  - fast-path implementation
  - profiling workflow
  - larger-case scaling benchmark
  - reproducible scripts + tests + evidence files

## 7) Next technical step (Week 4 handoff)
1. Optimize UF soft-info path (`_cluster_stats_from_syndrome`, repeated numpy reductions).
2. Add optional "minimal metrics" mode in UF/MWPM decoders to reduce per-shot Python overhead.
3. Re-run scaling with larger shots and one threshold sweep (`g in {0.35, 0.5, 0.65}`).
