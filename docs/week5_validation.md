# Week 5 Validation (Profiling and Scalability)

## Scope
Week 5 closure for:
- profiling/scaling workflow
- explicit time + memory artifacts
- canonical profiling coverage `d=3,5,7,9,11,13`
- explicit stress workflow at `d=11,13`
- schema-current artifacts

## Environment
- Recommended Python: `3.10` or `3.11` (Stim/PyMatching compatibility).
- If multiple Pythons are installed, use an environment where both `stim` and `pymatching` import successfully.

## Install
```bash
python -m pip install -r requirements.txt
```

## Week 5 tests
```bash
python -m pytest -q -p no:cacheprovider tests/test_week5_person1_profile_scaling_smoke.py
python -m pytest -q -p no:cacheprovider tests/test_week5_person1_adaptive_policy_tuning_smoke.py
python -m pytest -q -p no:cacheprovider tests/test_week5_person2_hardware_target_model_smoke.py
python -m pytest -q -p no:cacheprovider tests/test_week5_profile_schema_regression.py
python -m pytest -q -p no:cacheprovider tests/test_week5_stress_smoke.py
```

## Canonical profiling run
```bash
python -m scripts.run_week5_person1_profile_scaling --distances 3,5,7,9,11,13 --decoders mwpm,uf,bm,adaptive --rounds-mode distance --shots 120 --repeats 2 --profile-decoder adaptive --profile-distance 13 --profile-shots 80 --output results/week5_person1_profile_scaling.json
```

## Stress run
```bash
python -m scripts.run_week5_stress_test --distances 11,13 --decoders mwpm,uf,bm,adaptive --rounds-mode distance --shots 120 --repeats 1 --output results/week5_stress_test.json
```

## Expected artifacts
- `results/week5_person1_profile_scaling.json`
- `results/week5_stress_test.json`

## Interpretation note (memory)
- Week 5 memory measurement uses `tracemalloc`.
- This measures Python allocation behavior, not full process RSS/native allocator usage.
- This is acceptable for Week 5 closure, but memory conclusions should be interpreted accordingly.

## Week 5 closure checklist
- [ ] Profiling artifact has `schema_version` and provenance fields.
- [ ] Profiling artifact covers `d=3,5,7,9,11,13`.
- [ ] Profiling artifact includes all core decoders: MWPM, UF, BM/BP, Adaptive.
- [ ] Profiling artifact stores explicit time and memory metrics.
- [ ] Stress script runs explicitly for `d=11,13`.
- [ ] Stress artifact exists with status/pass, timing, memory, distance, decoder fields.
- [ ] Week-5 smoke tests pass.
- [ ] Week-5 schema regression test passes.
