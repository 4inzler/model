#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/download_piper_voice.sh en_US-amy-medium
# Downloads the Piper ONNX + JSON from Hugging Face into ~/models/piper

VOICE="${1:-en_US-amy-medium}"
BASE_DIR="${PIPER_BASE:-$HOME/models/piper}"
URL_BASE="https://huggingface.co/datasets/rhasspy/piper-voices/resolve/main"

mkdir -p "$BASE_DIR"

LANG="${VOICE%%-*}"
if [[ "$VOICE" == en_* ]]; then
  LANG_DIR="en"
else
  LANG_DIR="${LANG}"
fi

echo "[info] downloading $VOICE into $BASE_DIR"
for ext in onnx "onnx.json"; do
  FILE="$VOICE.$ext"
  TARGET="$BASE_DIR/$VOICE.${ext}"
  echo "  -> $FILE"
  curl -Lf "$URL_BASE/$LANG_DIR/$VOICE.$ext" -o "$TARGET"
done

echo "[info] done. Set PIPER_MODEL=$BASE_DIR/$VOICE.onnx"
