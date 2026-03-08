# Week 4 Validation Guide (Hardware-Aware Compatibility)

## Scope
This validation closes Week 4 as a hardware-aware milestone:
- hardware models and compatibility logic,
- compatibility JSON artifact with schema/provenance metadata,
- heatmap artifact tied to computed compatibility matrices,
- robust Week-4 test path.

## Environment
- Python 3.11 recommended (or another compatible environment with `stim` and `pymatching` installed).
- OS: Windows/Linux/macOS.

## Install
```bash
python -m pip install -r requirements.txt
```

## Tests
```bash
python -m pytest -q -p no:cacheprovider tests/test_hardware_models.py
python -m pytest -q -p no:cacheprovider tests/test_week4_hardware_compatibility_smoke.py
python -m pytest -q -p no:cacheprovider tests/test_week4_hardware_schema_regression.py
```

## Canonical Week-4 Run
```bash
python -m scripts.run_week4_hardware_compatibility \
  --adaptive-fast-mode \
  --output results/week4_hardware_compatibility.json \
  --figure-output figures/week4_hardware_compatibility_heatmaps.png
```

## Optional latency-input mode (for integration/handoff)
Use previously computed predicted latency rows instead of recomputing benchmark timings:
```bash
python -m scripts.run_week4_hardware_compatibility \
  --latency-input results/week4_hardware_compatibility.json \
  --adaptive-fast-mode \
  --output results/week4_hardware_compatibility.json \
  --figure-output figures/week4_hardware_compatibility_heatmaps.png
```

## Expected Artifacts
- `results/week4_hardware_compatibility.json`
- `figures/week4_hardware_compatibility_heatmaps.png`

## Interpretation note (important)
- Current Week-4 compatibility uses benchmarked Python decoder latencies plus fitted extrapolation to larger distances unless `--latency-input` is provided.
- This is acceptable for Week-4 closure as a hardware-aware mapping backbone.
- It is not hardware-trace-calibrated.
- A fully degenerate all-incompatible map is still a valid computed outcome under strict budgets and should be interpreted cautiously.

## Week-4 Closure Checklist
- [ ] Hardware model module exists and includes Google/IBM/IonQ/PsiQuantum architectures.
- [ ] Compatibility logic is exercised from decoder-latency data.
- [ ] Week-4 JSON includes schema and provenance metadata.
- [ ] Compatibility covers roadmap distances (`3,5,7,9,11,13`).
- [ ] Required decoder rows exist (`mwpm`, `uf`, adaptive thresholds).
- [ ] Canonical JSON artifact exists and is schema-current.
- [ ] Canonical heatmap artifact exists and matches the computed data.
- [ ] Week-4 tests pass.
