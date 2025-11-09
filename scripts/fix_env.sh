#!/usr/bin/env bash
set -euo pipefail

# Auto-fix .env for Docker by detecting model directories and Piper voice
# Usage: bash scripts/fix_env.sh [/path/to/host/models]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODELS_BASE="${1:-$HOME/models}"
ENV_FILE=".env"

YEL="\033[33m"; GRN="\033[32m"; RED="\033[31m"; RST="\033[0m"
info(){ echo -e "${GRN}[info]${RST} $*"; }
warn(){ echo -e "${YEL}[warn]${RST} $*"; }
err(){  echo -e "${RED}[err] ${RST} $*"; }

if [[ ! -d "$MODELS_BASE" ]]; then
  err "Models base directory not found: $MODELS_BASE"
  exit 2
fi

# Detect general model (FuseChat-Qwen-2.5-7B-Instruct or similar)
detect_general(){
  local d
  while IFS= read -r -d '' d; do echo "$d"; break; done < <(
    find "$MODELS_BASE" -maxdepth 2 -type d \
      \( -iname '*fusechat*qwen*7b*instruct*' \
         -o -iname '*qwen*2.5*7b*instruct*' \
         -o -iname 'fusechat-7b' \) -print0 | sort -z
  )
}

# Detect coder model (Qwen2.5-Coder-7B-Instruct or similar)
detect_coder(){
  local d
  while IFS= read -r -d '' d; do echo "$d"; break; done < <(
    find "$MODELS_BASE" -maxdepth 2 -type d \
      \( -iname '*qwen*2.5*coder*7b*instruct*' \
         -o -iname '*coder*7b*instruct*' \
         -o -iname 'fusecoder-7b' \) -print0 | sort -z
  )
}

# Detect Piper voice model (.onnx)
detect_piper(){
  local f
  while IFS= read -r -d '' f; do echo "$f"; break; done < <(
    find "$MODELS_BASE" -maxdepth 4 -type f -iname '*.onnx' -print0 | sort -z
  )
}

GENERAL_HOST="$(detect_general || true)"
CODER_HOST="$(detect_coder || true)"
PIPER_HOST="$(detect_piper || true)"

if [[ -z "${GENERAL_HOST}" ]]; then warn "General model not auto-detected"; fi
if [[ -z "${CODER_HOST}" ]]; then warn "Coder model not auto-detected"; fi
if [[ -z "${PIPER_HOST}" ]]; then warn "Piper voice not auto-detected (TTS can be toggled off)"; fi

# Convert to container paths (/models ...)
to_container_path(){
  local host="$1"
  [[ -n "$host" ]] || { echo ""; return; }
  echo "$host" | sed "s|^${MODELS_BASE%/}|/models|"
}

GENERAL_CONT="$(to_container_path "$GENERAL_HOST")"
CODER_CONT="$(to_container_path "$CODER_HOST")"
PIPER_CONT="$(to_container_path "$PIPER_HOST")"

[[ -f "$ENV_FILE" ]] || { warn ".env not found; creating from template"; [[ -f .env.example ]] && cp .env.example .env || touch .env; }
cp "$ENV_FILE" "$ENV_FILE.bak" 2>/dev/null || true

set_var(){
  local var="$1" val="$2"
  if grep -qE "^${var}=" "$ENV_FILE"; then
    sed -i "s|^${var}=.*|${var}=${val}|" "$ENV_FILE"
  else
    echo "${var}=${val}" >> "$ENV_FILE"
  fi
}

# Helper to convert an existing var value that points to the host models base
convert_existing_if_needed(){
  local var="$1"; shift || true
  local existing
  existing="$(grep -E "^${var}=" "$ENV_FILE" | sed -E "s/^${var}=//")"
  if [[ -n "$existing" && "$existing" == ${MODELS_BASE%/}* ]]; then
    local converted
    converted="${existing/${MODELS_BASE%/}/\/models}"
    set_var "$var" "$converted"
    info "Converted existing $var to container path: $converted"
    return 0
  fi
  return 1
}

# Prefer auto-detected values; otherwise convert existing env pointing at host paths
if [[ -n "$GENERAL_CONT" ]]; then
  set_var SEL_GENERAL_MODEL "$GENERAL_CONT"
else
  convert_existing_if_needed SEL_GENERAL_MODEL || true
fi
if [[ -n "$CODER_CONT" ]]; then
  set_var SEL_CODER_MODEL "$CODER_CONT"
else
  convert_existing_if_needed SEL_CODER_MODEL || true
fi
if [[ -n "$PIPER_CONT" ]]; then
  set_var PIPER_MODEL "$PIPER_CONT"
else
  convert_existing_if_needed PIPER_MODEL || true
fi

# Default toggles
set_var SEL_DISABLE_PRIVILEGED_INTENTS 0
if [[ -n "$PIPER_CONT" ]]; then set_var SEL_TTS_DEFAULT 1; else set_var SEL_TTS_DEFAULT 0; fi

info "Updated $ENV_FILE (backup at $ENV_FILE.bak)"
echo "---"
grep -E '^(SEL_GENERAL_MODEL|SEL_CODER_MODEL|PIPER_MODEL|SEL_TTS_DEFAULT|SEL_DISABLE_PRIVILEGED_INTENTS)=' "$ENV_FILE" || true
