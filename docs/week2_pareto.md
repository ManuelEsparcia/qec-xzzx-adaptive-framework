# Week 2 Pareto Summary

## Source artifact
- Input: `results/week2_person1_paired_threshold_sweep.json`
- Data used: `aggregates.adaptive_means_by_threshold`

## Metrics
For each threshold `g`, we track:
- `mean_error_rate` (minimize)
- `mean_avg_decode_time_sec` (minimize)
- `mean_switch_rate` (minimize)
- `mean_speedup_vs_mwpm` (maximize)

## Aggregate points (from current canonical sweep)

| g | mean_error_rate | mean_avg_decode_time_sec | mean_switch_rate | mean_speedup_vs_mwpm |
|---|---:|---:|---:|---:|
| 0.1 | 0.039750 | 0.000031784 | 0.000000 | 0.218456 |
| 0.2 | 0.039750 | 0.000031640 | 0.000000 | 0.219444 |
| 0.3 | 0.039750 | 0.000031244 | 0.000000 | 0.222211 |
| 0.4 | 0.039750 | 0.000031069 | 0.000875 | 0.223461 |
| 0.5 | 0.039750 | 0.000031064 | 0.004750 | 0.223489 |
| 0.6 | 0.039750 | 0.000031435 | 0.022875 | 0.220931 |
| 0.7 | 0.039750 | 0.000033071 | 0.090125 | 0.210710 |
| 0.8 | 0.039750 | 0.000037025 | 0.260750 | 0.189571 |
| 0.9 | 0.039750 | 0.000045866 | 0.645750 | 0.154546 |

## Pareto-optimal points
Using the dominance rule:
- minimize (`error_rate`, `avg_time`, `switch_rate`)
- maximize (`speedup`)

Pareto set:
- `g = 0.3`
- `g = 0.4`
- `g = 0.5`

Interpretation:
- `g=0.3` is the conservative point: zero switching on aggregate, strong time/speed profile.
- `g=0.4` is a faster point with still tiny switching (0.0875%).
- `g=0.5` is the fastest point in this sweep, but with a higher switch rate (0.475%).

## Recommended preliminary policy
Use `g_threshold = 0.3` as Week 2 default policy for handoff to Week 3:
- keeps switching effectively at zero in this sweep
- remains on the Pareto frontier
- avoids increasing switching overhead while preserving the best practical speedup regime among low-switch points

`g=0.4` is a valid alternative when prioritizing minimal decode time over strict switch suppression.
