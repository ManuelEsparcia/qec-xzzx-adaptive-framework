#!/usr/bin/env bash
#SBATCH --job-name=week3-threshold-scan
#SBATCH --output=logs/week3_%A_%a.out
#SBATCH --error=logs/week3_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-0

set -euo pipefail

MANIFEST_PATH="${MANIFEST_PATH:-manifests/week3_campaign_manifest.json}"
JOB_INDEX="${SLURM_ARRAY_TASK_ID:-${1:-0}}"

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Manifest not found: ${MANIFEST_PATH}" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python is required to execute manifest jobs." >&2
  exit 1
fi

mkdir -p logs

python - "${MANIFEST_PATH}" "${JOB_INDEX}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
job_index = int(sys.argv[2])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
jobs = payload.get("jobs", [])
if not isinstance(jobs, list) or not jobs:
    raise SystemExit("Manifest has no jobs.")
if job_index < 0 or job_index >= len(jobs):
    raise SystemExit(f"job index out of range: {job_index} (n_jobs={len(jobs)})")

job = jobs[job_index]
cmd = job.get("command", [])
if not isinstance(cmd, list) or not cmd:
    raise SystemExit(f"Invalid command for job index {job_index}")

print(f"Running job {job_index}: {job.get('job_id', '<unknown>')}")
print(" ".join(str(x) for x in cmd))
res = subprocess.run([str(x) for x in cmd], check=False)
raise SystemExit(res.returncode)
PY
