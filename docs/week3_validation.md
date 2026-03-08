# Week 3 Validation Guide (Massive-Simulation Backbone)

## Scope
This document defines the runtime checks for Week 3 closure:
- threshold-scan backbone,
- roadmap-aligned campaign defaults,
- checkpoint + resume behavior,
- auditable campaign orchestration (manifest + local parallel + SLURM template),
- schema-consistent canonical artifacts.

## Environment
- Python 3.11 recommended.
- OS: Linux/macOS/Windows (for `.sh` launchers, use a POSIX shell).

## Install
```bash
python -m pip install -r requirements.txt
```

## Core Week 3 Tests
```bash
python -m pytest -q -p no:cacheprovider tests/test_week3_person2_threshold_scan_smoke.py
python -m pytest -q -p no:cacheprovider tests/test_week3_person2_threshold_scan_resume.py
python -m pytest -q -p no:cacheprovider tests/test_week3_campaign_manifest.py
python -m pytest -q -p no:cacheprovider tests/test_week3_threshold_schema_regression.py
```

## Threshold Scan (roadmap-aligned defaults)
Default config now targets:
- distances: `3,5,7,9,11,13`
- p-grid: `0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.03`
- decoders: `mwpm,uf,bm,adaptive`
- noise: `depolarizing,biased_eta10,biased_eta100,biased_eta500,circuit_level,correlated`

Canonical artifact command (runtime-friendly):
```bash
python -m scripts.run_week3_person2_threshold_scan \
  --shots 40 \
  --checkpoint-every 48 \
  --adaptive-fast-mode \
  --output results/week3_person2_threshold_scan_full.json
```

Optional deeper run (slower):
```bash
python -m scripts.run_week3_person2_threshold_scan --output results/week3_person2_threshold_scan_full.json
```

## Partial Run + Resume Verification
1) Produce a partial campaign:
```bash
python -m scripts.run_week3_person2_threshold_scan \
  --distances 3,5,7,9,11,13 \
  --p-values 0.001,0.002,0.003,0.005 \
  --decoders mwpm,uf,bm,adaptive \
  --noise-models depolarizing,biased_eta10,biased_eta100,biased_eta500,circuit_level,correlated \
  --shots 40 \
  --repeats 1 \
  --checkpoint-every 12 \
  --output results/week3_person2_threshold_scan_partial.json
```

2) Resume into full p-grid:
```bash
python -m scripts.run_week3_person2_threshold_scan \
  --resume-from results/week3_person2_threshold_scan_partial.json \
  --shots 40 \
  --repeats 1 \
  --checkpoint-every 24 \
  --output results/week3_person2_threshold_scan_full.json
```

Resume semantics:
- Point identity is deduplicated with `(decoder, noise_model, distance, rounds, p_phys, shots, repeat_index)`.
- Completed points are skipped.
- Re-running with the same `--resume-from` is idempotent.

## Campaign Manifest + Launchers
Generate manifest:
```bash
python -m scripts.generate_week3_campaign_manifest \
  --output manifests/week3_campaign_manifest.json \
  --output-dir results/week3_campaign_chunks
```

Local parallel launcher example:
```bash
bash scripts/run_week3_campaign_local_parallel.sh manifests/week3_campaign_manifest.json 4
```

SLURM template usage example:
```bash
# Count jobs and submit as array [0..N-1]
python - <<'PY'
import json
from pathlib import Path
p = Path("manifests/week3_campaign_manifest.json")
jobs = json.loads(p.read_text(encoding="utf-8"))["jobs"]
print(len(jobs))
PY

# Example (replace N_MINUS_1)
sbatch --array=0-N_MINUS_1 scripts/submit_week3_slurm.sh
```

## Expected Artifact Paths
- Main canonical scan:
  - `results/week3_person2_threshold_scan_full.json`
- Optional partial for resume check:
  - `results/week3_person2_threshold_scan_partial.json`
- Manifest:
  - `manifests/week3_campaign_manifest.json`
- Chunk outputs (manifest jobs):
  - `results/week3_campaign_chunks/*.json`

## Week 3 Closure Acceptance Checklist
- [ ] `run_week3_person2_threshold_scan.py` supports `--resume-from` and skips completed points.
- [ ] Checkpoints/final JSON writes are atomic (tmp + rename).
- [ ] Default p-grid reaches `~0.03` with 8-10 points.
- [ ] Default distances include `3,5,7,9,11,13`.
- [ ] Week 3 canonical artifact contains all decoders and all required noise families.
- [ ] Biased `eta=10/100/500` are present in canonical outputs.
- [ ] Manifest generator exists and produces unique jobs.
- [ ] Local parallel launcher exists and consumes manifest jobs.
- [ ] SLURM launcher/template exists and runs manifest job by array index.
- [ ] Resume and manifest/schema tests pass.
