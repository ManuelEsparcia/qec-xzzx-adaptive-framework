# Week 1 - Person 2 Final Report (Noise Modeling)

## 1) Goal

Implement and validate the Week 1 noise-modeling block for QEC-XZZX:

- `src/noise/noise_models.py`
- `tests/test_noise_models.py`
- integration with circuit + decoder flow
- baseline benchmark artifact for Week 1 evidence

## 2) Implemented models

Implemented in `src/noise/noise_models.py`:

- `DepolarizingNoise`
- `BiasedNoise`
- `CircuitLevelNoise`
- `PhenomenologicalNoise`
- `CorrelatedNoise`

Factory and helper entry points:

- `build_noise_model(...)`
- `apply_noise_model(...)`

## 3) Validation tests

Week 1-relevant tests:

- `tests/test_noise_models.py`
- `tests/test_week1_person2_integration.py`
- `tests/test_week1_person2_noise_benchmark_smoke.py`
- `tests/test_week1_closure.py`

These tests check model construction, circuit injection contract, correlated-neighbor logic, and smoke-level end-to-end execution.

## 4) Week 1 benchmark command

```bash
python -m scripts.run_week1_person2_noise_benchmark --shots 400 --distances 3,5,7,9 --output results/week1_person2_noise_benchmark.json
```

Generated artifact:

- `results/week1_person2_noise_benchmark.json`

The JSON contains:

- per-case summary (`cases_summary`)
- per-model metrics (`ler`, decode times, detector/observable counts)
- aggregate metrics (`mean_ler_by_model`, decode-time means, correlated-vs-depolarizing delta)

## 5) Related Week 1 artifacts

- `results/week1_person1_baseline.json`
- `results/week1_person2_noise_benchmark.json`
- `docs/week1_validation.md`

## 6) Week 1 closure contribution

From the noise-modeling side, Week 1 closure is satisfied when:

1. all 5 models instantiate and apply to Stim circuits,
2. integration tests do not break the basic pipeline contract,
3. benchmark script runs and writes the Week 1 JSON artifact.
