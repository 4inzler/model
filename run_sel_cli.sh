#!/usr/bin/env bash
set -euo pipefail

# Helper to run SEL CLI inside Docker on this machine.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$ROOT_DIR/.env"
MODELS_SRC="${1:-/home/rinz/models}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "[err] .env not found at $ENV_FILE" >&2
  exit 1
fi

if [[ ! -d "$MODELS_SRC" ]]; then
  echo "[err] models directory not found: $MODELS_SRC" >&2
  exit 2
fi

CONTAINER_NAME="sel-cli"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

exec docker run --rm -it --gpus all --name "$CONTAINER_NAME" \
  --env-file "$ENV_FILE" \
  -v "$MODELS_SRC":/models:ro \
  sel-bot:latest sel-cli
