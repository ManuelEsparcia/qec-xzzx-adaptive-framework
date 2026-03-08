# Week 2 Validation and Closure Guide

## Scope
Week 2 closure targets:
- Union-Find decoder with soft-info
- Belief-Matching/BP fallback with BP diagnostics contract
- Adaptive confidence-threshold switching (UF fast, MWPM accurate)
- g-threshold sweep on d=3,5 with depolarizing and biased noise
- Updated Week 2 artifacts and Pareto summary

## Environment
- Python: 3.10+ (tested with 3.10.15)
- OS: Windows/Linux/macOS
- Required packages pinned in `requirements.txt`

Install:
```bash
python -m pip install -r requirements.txt
```

## Week 2 tests
Core decoder/switching tests:
```bash
python -m pytest tests/test_union_find_decoder.py tests/test_belief_matching_decoder.py tests/test_adaptive_decoder.py -q
```

Week 2 script smoke tests:
```bash
python -m pytest tests/test_week2_person1_decoder_comparison_smoke.py tests/test_week2_person1_paired_sweep_smoke.py -q
```

If your environment has restricted default temp directories, run:
```bash
set TMP=./tmp_test_runtime
set TEMP=./tmp_test_runtime
python -m pytest tests/test_week2_person1_decoder_comparison_smoke.py tests/test_week2_person1_paired_sweep_smoke.py -q
```

## Week 2 benchmark commands
Canonical comparison artifact:
```bash
python -m scripts.run_week2_person1_decoder_comparison --output results/week2_person1_decoder_comparison.json
```

Canonical threshold sweep artifact (roadmap grid + noise coverage):
```bash
python -m scripts.run_week2_person1_paired_threshold_sweep --output results/week2_person1_paired_threshold_sweep.json
```

Optional comparison artifacts used by reports:
```bash
python -m scripts.run_week2_person1_decoder_comparison --shots 500 --keep-soft 120 --g-threshold 0.20 --output results/week2_cmp_g020.json
python -m scripts.run_week2_person1_decoder_comparison --shots 500 --keep-soft 120 --g-threshold 0.40 --output results/week2_cmp_g040.json
python -m scripts.run_week2_person1_decoder_comparison --shots 500 --keep-soft 120 --g-threshold 0.60 --output results/week2_cmp_g060.json
python -m scripts.run_week2_person1_decoder_comparison --shots 500 --keep-soft 120 --g-threshold 0.80 --output results/week2_cmp_g080.json
python -m scripts.run_week2_person1_decoder_comparison --shots 2000 --keep-soft 120 --g-threshold 0.35 --output results/week2_person1_decoder_comparison_2k_g035.json
```

## Expected Week 2 artifacts
- `results/week2_person1_decoder_comparison.json`
- `results/week2_person1_paired_threshold_sweep.json`
- `results/week2_cmp_g020.json`
- `results/week2_cmp_g040.json`
- `results/week2_cmp_g060.json`
- `results/week2_cmp_g080.json`
- `results/week2_person1_decoder_comparison_2k_g035.json`
- `docs/week2_pareto.md`

## BP diagnostics contract
`src/decoders/belief_matching_decoder.py` now guarantees these soft-info keys:
- `convergence_flag`
- `num_iterations`
- `residual_error`

When backend-native diagnostics are unavailable, explicit heuristic fallback values are emitted and tagged with `bp_diagnostics_source`.

## Week 2 binary acceptance checklist
- [ ] UF decoder runs and returns UF soft-info keys
- [ ] BM/BP decoder runs and returns `convergence_flag`, `num_iterations`, `residual_error`
- [ ] AdaptiveDecoder switches by confidence threshold
- [ ] `benchmark_adaptive` metrics are present (`error_rate_adaptive`, `avg_decode_time_adaptive`, `switch_rate`, `speedup_vs_mwpm` when reference enabled)
- [ ] g-grid in sweep is `0.1,0.2,...,0.9`
- [ ] Sweep covers both noise families (`depolarizing`, `biased`)
- [ ] Sweep covers d=3 and d=5
- [ ] Canonical Week 2 JSON artifacts are regenerated and schema-current
- [ ] Pareto artifact exists and states recommended preliminary threshold policy
