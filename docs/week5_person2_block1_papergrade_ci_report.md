# Week 5 - Person 2 Block 1 Report (Paper-Grade Replication + CIs)

## 1) Scope completed
- Re-executed the massive threshold scan with seed replication enabled to obtain paper-grade confidence intervals.
- Used the full Q1 distance grid (`d=3,5,7,9,11,13`) with all decoders and all Week 3/5 noise models.
- Preserved the same physical error-rate grid used in prior runs for comparability.

## 2) Execution details
- Execution date (UTC): `2026-02-20`
- Runner:
  - `scripts/run_week3_person2_threshold_scan.py`
- Command:
  - `python -m scripts.run_week3_person2_threshold_scan --distances 3,5,7,9,11,13 --rounds 3 --p-values 0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02 --decoders mwpm,uf,bm,adaptive --noise-models depolarizing,biased_eta10,biased_eta100,biased_eta500,circuit_level,correlated --shots 300 --repeats 2 --seed 20260220 --g-threshold 0.35 --adaptive-fast-mode --checkpoint-every 24 --output results/week5_block1_threshold_scan_paper_grade_r2.json`
- Wall-clock runtime:
  - ~`804 s` (about `13.4 min`)

## 3) Output artifact
- `results/week5_block1_threshold_scan_paper_grade_r2.json`

## 4) Output contract snapshot
- Total points: `1152`
- Distances: `[3,5,7,9,11,13]`
- Decoders: `mwpm, uf, bm, adaptive`
- Noise models:
  - `depolarizing`
  - `biased_eta10`
  - `biased_eta100`
  - `biased_eta500`
  - `circuit_level`
  - `correlated`
- Replication:
  - `repeats=2` with per-repeat seed tracking and per-point CI95 metrics.

## 5) CI quality summary
- Error-rate CI95 half-width:
  - mean: `0.012800`
  - median: `0.006533`
  - p90: `0.035933`
- Decode-time CI95 half-width (seconds):
  - mean: `8.37e-06`
  - median: `3.56e-06`
  - p90: `1.82e-05`
- Coverage by CI tightness:
  - `ER CI <= 0.02`: `911/1152` (`79.08%`)
  - `ER CI <= 0.03`: `997/1152` (`86.55%`)
  - `time CI <= 1e-5 s`: `928/1152` (`80.56%`)
  - `time CI <= 2e-5 s`: `1049/1152` (`91.06%`)

## 6) High-variance hotspots (largest ER CI95 half-width)
- `adaptive | correlated | d=5 | p=0.01`: `ER=0.293333 +/- 0.117600`
- `bm | biased_eta10 | d=13 | p=0.015`: `ER=0.331667 +/- 0.101267`
- `adaptive | biased_eta100 | d=11 | p=0.02`: `ER=0.490000 +/- 0.098000`
- `mwpm | biased_eta10 | d=3 | p=0.02`: `ER=0.223333 +/- 0.098000`

Interpretation:
- The replicated scan is stable for most of the grid.
- Remaining wide CIs cluster near high-noise/high-error regimes, which is expected and suitable for targeted extra repeats if needed.

## 7) Status
Block 1 paper-grade replication + CI scan is completed with a full artifact and quantitative stability checks.
