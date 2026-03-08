#!/usr/bin/env bash
set -euo pipefail

MANIFEST_PATH="${1:-manifests/week3_campaign_manifest.json}"
MAX_PARALLEL="${2:-4}"

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Manifest not found: ${MANIFEST_PATH}" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python is required to read the manifest." >&2
  exit 1
fi

if ! [[ "${MAX_PARALLEL}" =~ ^[0-9]+$ ]] || [[ "${MAX_PARALLEL}" -le 0 ]]; then
  echo "MAX_PARALLEL must be a positive integer. Got: ${MAX_PARALLEL}" >&2
  exit 1
fi

echo "Running week3 campaign from ${MANIFEST_PATH} with max_parallel=${MAX_PARALLEL}"

if command -v parallel >/dev/null 2>&1; then
  python - "${MANIFEST_PATH}" <<'PY' | parallel -j "${MAX_PARALLEL}"
import json
import shlex
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
for job in payload.get("jobs", []):
    cmd = job.get("command", [])
    if isinstance(cmd, list) and cmd:
        print(shlex.join(str(x) for x in cmd))
PY
else
  python - "${MANIFEST_PATH}" <<'PY' | xargs -I{} -P "${MAX_PARALLEL}" bash -lc "{}"
import json
import shlex
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
for job in payload.get("jobs", []):
    cmd = job.get("command", [])
    if isinstance(cmd, list) and cmd:
        print(shlex.join(str(x) for x in cmd))
PY
fi
