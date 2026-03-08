# Week 1 Validation and Closure Guide

This document defines the reviewer-facing procedure to validate that Week 1 is closed.

## 1) Environment requirements

- Python 3.11 required for the pinned Week 1 dependency set
- OS: Linux/macOS/Windows supported
- Dependencies pinned in `requirements.txt`

## 2) Install

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3) Week 1 test commands

Run the Week 1-focused tests:

```bash
python -m pytest -q tests/test_xzzx_code.py
python -m pytest -q tests/test_mwpm_decoder.py
python -m pytest -q tests/test_noise_models.py
python -m pytest -q tests/test_week1_person1_integration.py
python -m pytest -q tests/test_week1_person2_integration.py
python -m pytest -q tests/test_week1_person2_noise_benchmark_smoke.py
python -m pytest -q tests/test_week1_closure.py
```

## 4) Week 1 benchmark commands

Person 1 baseline benchmark:

```bash
python -m scripts.run_week1_person1_benchmark --distances 3,5,7,9 --shots 300 --ref-shots 300 --output results/week1_person1_baseline.json
```

Person 2 noise benchmark:

```bash
python -m scripts.run_week1_person2_noise_benchmark --distances 3,5,7,9 --shots 400 --output results/week1_person2_noise_benchmark.json
```

## 5) Expected artifacts

- `results/week1_person1_baseline.json`
- `results/week1_person2_noise_benchmark.json`
- `docs/week1_person1_summary.md`
- `docs/week1_person2_report.md`

## 6) Binary closure criteria (all must pass)

- XZZX generation validated for `d=3,5,7,9`
- MWPM decoder with soft-info runs and returns required soft keys
- All 5 noise models instantiate and apply to circuits
- Week 1 integration flow executes
- Week 1 closure smoke test passes (`tests/test_week1_closure.py`)
- Week 1 benchmark scripts generate JSON outputs for `d=3,5,7,9`

## 7) XZZX implementation scope statement

Current Week 1 XZZX generation is implemented through Stim rotated-memory templates via wrapper logic (`src/codes/xzzx_code.py`, `generate_xzzx_circuit`).

This is acceptable for Week 1 closure as a practical implementation route for end-to-end integration and benchmarking.

It should not be presented as a custom circuit-construction research contribution.
