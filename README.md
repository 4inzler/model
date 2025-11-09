# SEL Bot

SEL (Systematic Emotional Logic) is a mixture-of-experts companion that tries to feel present the way a human friend would. Two local language models trade notes, hormones rise and fall, and the persona adapts to each channel’s pace so conversations stay warm instead of robotic.

## Why talk to SEL?

- **Human-style mood swings.** A 14-hormone loop shapes tone, patience, and humor, so each reply reflects SEL’s current headspace instead of a stock script.
- **Relationship awareness.** SEL remembers past chats, tracks trust, and mirrors the pace, slang, and emoji habits of every channel.
- **Deterministic empathy.** No dice rolls—reflections, priority heuristics, and hormone math steer replies so you get consistent care without feeling mechanical.
- **All-in-one brain.** General + coder models collaborate, with an accelerator fusion step that turns multiple drafts into a candid final message.

## Repo tour

| Path | What lives there |
| --- | --- |
| `sel.py` | Unified CLI/Discord brain with memory, hormone engine, and multimodal vision hooks. |
| `sel_enhanced.py` | Stand-alone persona experiment with similar hormone logic but slimmer integrations. |
| `sel_discord.py` / `sel_discord_enhanced.py` | Discord bridges for the baseline and enhanced brains. |
| `audio_utils.py` | Piper TTS + faster-whisper glue for voice features. |
| `scripts/` | Helpers (e.g., `fix_env.sh` rewrites `.env` for container mounts). |
| `Echo-Luna-main/`, `AIWaifu/` | Experimental side projects that share parts of the stack. |

## Quickstart (uv workflow)

```bash
uv venv
uv sync
cp .env.example .env  # create and edit with your local model paths
```

Point the following variables in `.env` to absolute paths on your machine:

- `SEL_GENERAL_MODEL` – general chat model directory
- `SEL_CODER_MODEL` – coder-leaning model directory
- `PIPER_MODEL` (optional) – Piper voice `.onnx` file for TTS

Want Docker to see the same paths? Run:

```bash
bash scripts/fix_env.sh /absolute/path/to/models
```

It rewrites `.env` so container mounts line up with your host folders.

## Run modes

```bash
# CLI baseline
uv run sel-cli              # or: uv run python sel.py

# CLI with extra persona layers
uv run sel-enhanced         # or: uv run python sel_enhanced.py

# Discord bridges (require DISCORD_TOKEN + intents)
uv run sel-discord
uv run sel-discord-enhanced
```

### Docker

```bash
docker build -t sel-bot:latest .

GENERAL=/abs/path/to/general_model
CODER=/abs/path/to/coder_model
PIPER=/abs/path/to/piper/model.onnx  # optional

docker run --rm -it --gpus all \
  --env-file .env \
  -v "$GENERAL":/models/general:ro \
  -v "$CODER":/models/coder:ro \
  -e SEL_GENERAL_MODEL=/models/general \
  -e SEL_CODER_MODEL=/models/coder \
  sel-bot:latest sel-cli
```

Swap the final command for `sel-discord` (or `sel-discord-enhanced`) to keep a bot online. `./run_sel_cli.sh` wraps the same invocation with the default `/home/rinz/models` mount.

## Configuration cheat sheet

| Variable | Purpose |
| --- | --- |
| `SEL_VISION_MODEL` | Vision-language model for image replies (`off` disables it). |
| `SEL_MEMORY_PATH`, `SEL_STATE_PATH`, `SEL_PERSONA_PATH`, `SEL_CHANNEL_PROFILE_PATH` | Where SEL stores long-term context. Point these elsewhere if you want persistence on another disk. |
| `SEL_SHORT_TERM_TTL`, `SEL_STICKY_TTL`, `SEL_IMAGE_TTL` | Lifetime (seconds) for different memory pools. |
| `SEL_SINGLE_BACKEND`, `SEL_QUANT` | Constrain VRAM by hot-swapping models or loading quantized weights. |
| `SEL_STREAM` | Enables token streaming in Discord replies. |
| `SEL_SHOW_MOOD` | Toggle the `[mood: ...]` status banner. |
| `SEL_AUTO_REPLY_*` knobs | Tune spontaneous nudges when a channel gets quiet. |
| `SEL_DEEP_THINKING`, `SEL_REFLECTION_STEPS`, `SEL_REFLECTION_TOKENS` | Enable multi-pass reasoning when you want SEL to slow down and ponder. |

All toggles read from the environment, so you can set them directly in `.env` or pass `-e` flags to Docker.

## How SEL keeps conversations human

- **Channel mirroring.** SEL tracks average length, emoji cadence, and in-jokes per channel, then adapts replies so it feels like part of the crowd.
- **Mood-aware phrasing.** Hormone markers (dopamine spikes, melatonin drifts, etc.) translate into small talk that references the current vibe instead of canned lines.
- **Relationship cues.** Trust scores, jealousy checks, and gap tracking tell SEL when to keep things gentle, dive into support mode, or take a playful tone.
- **Reflective thoughts.** Every reply stores an internal reflection so the next turn remembers why SEL answered the way it did.

## Tests, linting, and packaging

```bash
uv run ruff format
uv run ruff check --fix
uv run pytest tests -q
uv build  # builds wheel + sdist in dist/
```

## Troubleshooting tips

- **Missing model warnings** mean the container cannot see your mount—double-check the `-v` paths or use `scripts/fix_env.sh`.
- **Discord DNS hiccups** usually vanish with `--network host` (Linux) or explicit `--dns` flags.
- **`torch_dtype` deprecation warnings** are resolved; SEL now honors Transformers’ `dtype` argument automatically.
- **Vision quirks** often trace back to Pillow; reinstall with `uv pip install pillow` inside the environment if images refuse to parse.

Bring your own models, tune the knobs that match your vibe, and SEL will keep showing up like a friend who actually remembers the last conversation.

