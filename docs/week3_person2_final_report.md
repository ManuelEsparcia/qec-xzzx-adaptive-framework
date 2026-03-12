# Week 3 - Person 2 Final Report (Massive Simulations + Threshold Scan)

## 1) Scope Closed
- Week 3 threshold-scan runner implemented with:
  - multi-axis grid over decoder/noise/distance/physical error rate,
  - checkpointed atomic writes,
  - resumable execution,
  - threshold estimates from distance-curve crossings.
- Validation path in place (`smoke`, `resume`, `manifest`, `schema` tests).

## 2) Canonical Scientific Artifact (Current)
Primary Week 3 reference artifact:
- `results/week3_person2_threshold_scan_paper_grade_r2.json`

Canonical run characteristics:
- distances: `3,5,7,9,11,13`
- p-values: `0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02`
- decoders: `mwpm,uf,bm,adaptive`
- noise models: `depolarizing,biased_eta10,biased_eta100,biased_eta500,circuit_level,correlated`
- shots: `300`
- repeats: `2`
- adaptive config: `g_threshold=0.35`, `adaptive_fast_mode=true`

Coverage summary from artifact:
- points: `1152`
- threshold estimates: `24`
- CI coverage:
  - `ER CI95 half-width <= 0.02`: `911/1152` (`79.08%`)
  - `ER CI95 half-width <= 0.03`: `997/1152` (`86.55%`)
  - `time CI95 half-width <= 1e-5 s`: `928/1152` (`80.56%`)
  - `time CI95 half-width <= 2e-5 s`: `1049/1152` (`91.06%`)

## 3) Adaptive g-Threshold Sensitivity Addendum (2026-03-08)
New artifacts:
- `results/week3_person2_threshold_scan_adaptive_g035_r2.json`
- `results/week3_person2_threshold_scan_adaptive_g065_r2.json`
- `results/week3_person2_threshold_scan_adaptive_g080_r2.json`

Grid for all three:
- distances `3,5,7,9,11,13`, same p-grid/noise-grid as canonical,
- decoder fixed to `adaptive`,
- `shots=300`, `repeats=2`.

Aggregate behavior (all 288 points per run):

| g | mean_switch_rate | mean_avg_decode_time_sec | mean_error_rate |
|---|---:|---:|---:|
| 0.35 | 0.000110 | 3.686e-05 | 0.101453 |
| 0.65 | 0.047251 | 3.976e-05 | 0.101453 |
| 0.80 | 0.447054 | 5.931e-05 | 0.101453 |

Interpretation:
- switching becomes clearly observable and monotonic with `g`,
- higher `g` substantially increases switching and average decode time on this grid,
- no aggregate ER gain is observed from increasing `g` in this setup.

## 4) Historical Artifacts (Non-Canonical)
The following remain useful for runtime smoke and debugging, but are not the primary scientific reference:
- `results/week3_person2_threshold_scan_full.json` (`shots=40`, `repeats=1`)
- `results/week3_person2_threshold_scan_quick.json`
- `results/week3_person2_threshold_scan_verify.json`

## 5) Closure Status
Week 3 Person 2 is closed with:
- implementation complete,
- test coverage active,
- paper-grade canonical artifact designated,
- explicit adaptive threshold-sensitivity evidence for switching observability.
