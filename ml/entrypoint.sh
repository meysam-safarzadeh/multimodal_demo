#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] start $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "[entrypoint] workdir=$(pwd)  python=$(python --version)  pip=$(pip --version)"

# -----------------------------
# Model selection & requirements
# -----------------------------
MODEL="${MODEL:-multimodal}"   # default model folder name under models/
MODEL_REQ="/app/models/${MODEL}/requirements.txt"

# -----------------------------
# Install requirements dynamically
# -----------------------------
install_reqs() {
  local req_file="$1"
  if [[ -n "$req_file" && -f "$req_file" ]]; then
    echo "[entrypoint] installing requirements: ${req_file}"
    pip install --no-cache-dir -r "$req_file"
  else
    echo "[entrypoint] no requirements file found at: ${req_file} (skipping)"
  fi
}

install_reqs "${MODEL_REQ}"

echo "[entrypoint] starting training using model: ${MODEL}"
exec python /app/main.py
