# Week 5 - Person 1 Block 3 Report (Profiling + Memory + Scaling)

## 1) Scope completed
- Implemented an end-to-end benchmark script for Block 3:
  - runtime profiling (`cProfile`)
  - memory peak/current tracking (`tracemalloc`)
  - scaling-law fit over distance (`time` and `memory`)
- Added reproducibility controls with explicit `seed`, `repeats`, and parameterized grids.
- Added smoke and failure-path tests for the new script.

## 2) Main artifacts
### Source
- `scripts/run_week5_person1_profile_scaling.py`

### Tests
- `tests/test_week5_person1_profile_scaling_smoke.py`

### Expected results output
- `results/week5_person1_profile_scaling.json`

## 3) Implemented contract (Block 3)
- Decoders: `mwpm`, `uf`, `bm`, `adaptive`
- Distances: configurable CSV (default `5,7,9,11,13`)
- Noise model aliases aligned with Week 3 names:
  - `depolarizing`, `biased_eta10`, `biased_eta100`, `biased_eta500`, `circuit_level`, `correlated`
- Per point (`decoder`, `distance`) metrics with repeats:
  - `error_rate`, `avg_decode_time_sec`, `wall_time_sec`, `switch_rate`
  - `memory_peak_bytes`, `memory_current_bytes`
  - each reported as `mean/std/ci95`
- Scaling fit output:
  - power law `a * d^b` for time and memory
  - with `coefficient`, `exponent`, `r2`
- Hotspot output:
  - top cumulative functions from `cProfile`

## 4) Validation commands
- Smoke test:
  - `python -m pytest -q -p no:cacheprovider tests/test_week5_person1_profile_scaling_smoke.py`
- Full run:
  - `python -m scripts.run_week5_person1_profile_scaling --distances 5,7,9,11,13 --decoders mwpm,uf,bm,adaptive --shots 120 --repeats 2 --profile-decoder adaptive --profile-distance 13 --profile-shots 80 --output results/week5_person1_profile_scaling.json`

## 5) Validation status (executed)
- Date (UTC): `2026-02-20`
- Smoke test result:
  - `tests/test_week5_person1_profile_scaling_smoke.py`: **3 passed**
- Evidence runs generated:
  - `results/_verify_week5_person1_profile_scaling.json` (quick verification)
  - `results/week5_person1_profile_scaling.json` (full Block 3 run)

## 6) Quantitative snapshot from full run
- Time-scaling exponent (`avg_decode_time_sec ~ d^b`):
  - `mwpm`: `b=2.456`, `r2=0.9626`
  - `uf`: `b=1.834`, `r2=0.9246`
  - `bm`: `b=1.804`, `r2=0.9155`
  - `adaptive`: `b=0.589`, `r2=0.6270`
- Memory-scaling exponent (`memory_peak_bytes ~ d^b`):
  - `mwpm`: `b=2.294`, `r2=0.9984`
  - `uf`: `b=2.275`, `r2=0.9986`
  - `bm`: `b=2.267`, `r2=0.9987`
  - `adaptive`: `b=2.280`, `r2=0.9984`
- cProfile (adaptive, `d=13`, `shots=80`) top hotspot families:
  - matcher construction/loading (`pymatching.matching.*`)
  - decoder initialization (`union_find_decoder.__init__`, `mwpm_decoder.__init__`)
  - graph conversion (`detector_error_model_to_matching_graph`)

## 7) Status
Block 3 is implemented and validated with executable evidence (script + tests + quick/full JSON outputs + report).
