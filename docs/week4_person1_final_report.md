# Week 4 - Person 1 Final Report (Hardware Models)

## 1) Scope completed
- Implemented hardware-aware architecture models for Week 4.
- Added explicit per-architecture cycle and classical-decoding budgets.
- Added compatibility API to evaluate whether a decoder+distance latency fits a hardware budget.

## 2) Main artifacts
### Source
- `src/hardware/hardware_models.py`

### Tests
- `tests/test_hardware_models.py`

## 3) Implemented Week 4 hardware model contract
- Architectures implemented:
  - `GoogleSuperconducting`: cycle `1e-6 s`, budget `200e-9 s`
  - `IBMEagle`: cycle `1.5e-6 s`, budget `300e-9 s`
  - `IonQForte`: cycle `10e-6 s`, budget `2e-6 s`
  - `PsiQuantumPhotonic`: cycle `10e-9 s`, budget `2e-9 s`
- Public API:
  - `is_compatible(decoder, distance, latency_table, safety_factor=1.0)`
  - `budget_utilization(decoder, distance, latency_table)`
  - `build_latency_table(...)`
  - `compatibility_matrix(...)`

## 4) Validation status
- Executed:
  - `python -m pytest -q -p no:cacheprovider tests/test_hardware_models.py`
- Result:
  - **4 passed**

## 5) Status
Week 4 Person 1 deliverable is implemented, tested, and integrated with the Week 4 compatibility pipeline.

