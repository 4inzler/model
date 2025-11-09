# SEL Bot

Mixture-of-experts assistants (general + coder models) with optional enhanced personas, CLI mode, and Discord bots. This repo also contains experimental side projects (`AIWaifu`, `Echo-Luna`), but the SEL tooling lives at the repo root.

## Components
- `sel.py` – baseline CLI brain (`uv run sel-cli`)
- `sel_enhanced.py` – richer hormones/memory system (`uv run sel-enhanced`)
- `sel_discord.py` – lightweight Discord client
- `sel_discord_enhanced.py` – Discord + enhanced brain, web crawl, memes, TTS/STT
- `audio_utils.py` – Piper TTS + faster-whisper glue
- `scripts/fix_env.sh` – converts host model paths in `.env` to container paths

## Hormone-driven deterministic brain

- 14-hormone system (dopamine, serotonin, cortisol, adrenaline, melatonin, oxytocin, norepinephrine, estrogen, testosterone,
  endorphin, acetylcholine, prolactin, vasopressin, ACTH) with circadian + multi-day cycles and deterministic participation.
- Persona traits now live in `sel_persona.json` and evolve only from hormone trends (“hug dealer”, “restless tinkerer”, etc.).
- Hierarchical text + image memory reuses `sel_memory.json`: short-term buffer, sticky long-term layer with TTL, and image summaries.
- Vision integration (set `SEL_VISION_MODEL` or `off`) reacts to attachments with human-like comments and logs summaries.
- Every turn produces an internal reflective thought (captured in the response payload) before a 1–3 sentence public reply.
- Channel pacing memory tracks average length/emojis/signature slang per server so SEL mirrors each crowd, and when hormones are “bored” it can drop a casual nudge even without a mention.
- Hormone spike logs feed the reflective mind, and reinforcement loops learn which channels/users miss SEL when it stays quiet, slightly biasing future participation without using randomness.

### Key environment knobs

```
SEL_VISION_MODEL=Qwen/Qwen2-VL-7B-Instruct  # set to "off" to disable image analysis
SEL_MEMORY_PATH=sel_memory.json
SEL_STATE_PATH=sel_state.json
SEL_PERSONA_PATH=sel_persona.json
SEL_CHANNEL_PROFILE_PATH=sel_channels.json

# Optional decay overrides (seconds)
SEL_SHORT_TERM_TTL=21600
SEL_STICKY_TTL=604800
SEL_IMAGE_TTL=172800
```

Files are created automatically; point the paths elsewhere if you want to keep state on another volume.

## Requirements
- Python 3.11 and [uv](https://docs.astral.sh/uv/)
- Local GGUF/HF models on disk:
  - `SEL_GENERAL_MODEL` – general chat model directory
  - `SEL_CODER_MODEL` – coder model directory
  - Optional `PIPER_MODEL` – Piper `.onnx` voice for TTS
- `ffmpeg` (already installed in Dockerfile) for Discord voice features

## Setup
```bash
uv venv
uv sync          # installs deps declared in pyproject.toml
cp .env.example .env
```

Edit `.env` and point the model variables at absolute host paths. When targeting Docker, run:
```bash
bash scripts/fix_env.sh /absolute/path/to/models
```
This rewrites `.env` so that container paths line up with your host volume mounts.

## Running Locally (uv)
```bash
# CLI baseline
uv run sel-cli              # or: uv run python sel.py

# CLI enhanced persona
uv run sel-enhanced         # or: uv run python sel_enhanced.py

# Discord (requires DISCORD_TOKEN + intents)
uv run sel-discord          # or: uv run python sel_discord.py
uv run sel-discord-enhanced # or: uv run python sel_discord_enhanced.py
```

Tips
- Set `SEL_GENERAL_MODEL`, `SEL_CODER_MODEL`, and `PIPER_MODEL` in `.env`.
- `SEL_SINGLE_BACKEND=1` swaps models (lower VRAM). `SEL_QUANT=4bit` for 4-bit quantization.
- `SEL_STREAM=1` enables token streaming in Discord replies.
- `SEL_SHOW_MOOD=0` hides the `[mood: ...]` line for a cleaner, more human CLI/Discord tone (defaults to 1).
- `SEL_LOG_LEVEL=DEBUG` turns on the detailed runtime telemetry in `sel_discord.py` (INFO by default).
- `SEL_RESPOND_TO_ALL=1` makes SEL answer even when you don’t @mention it (more human-in-channel).
- `SEL_AUTO_REPLY_CHANCE` / `SEL_AUTO_REPLY_COOLDOWN` + `SEL_FOLLOWUP_WINDOW` tune spontaneous replies and keep back-and-forth chats flowing with the same person even without tags.
- `SEL_DEEP_THINKING`, `SEL_REFLECTION_STEPS`, and `SEL_REFLECTION_TOKENS` enable multi-pass private reasoning so answers feel smarter (higher values = slower but more thoughtful).
- `SEL_DM_CHECKIN_MINUTES` tunes when the bot nudges folks directly if it hasn’t heard from them and its hormones say it’s bored.

## Docker Workflow
Build once:
```bash
docker build -t sel-bot:latest .
```

### CLI inside Docker
```bash
GENERAL=/abs/path/to/general_model
CODER=/abs/path/to/coder_model
PIPER=/abs/path/to/piper/model.onnx   # optional

 docker run --rm -it --gpus all \
   --env-file .env \
   -v "$GENERAL":/models/general:ro \
   -v "$CODER":/models/coder:ro \
   -e SEL_GENERAL_MODEL=/models/general \
  -e SEL_CODER_MODEL=/models/coder \
 sel-bot:latest sel-cli
```
Add the Piper mount/env if you want TTS.
On this workstation you can just run `./run_sel_cli.sh` (it mounts `/home/rinz/models` by default).

### Discord inside Docker
Discord bots run headless; keep DNS working (`--network host` on Linux is simplest):
```bash
docker run -d --name sel-discord --gpus all \
  --network host \
  --env-file .env \
  -v "$GENERAL":/models/general:ro \
  -v "$CODER":/models/coder:ro \
  -e SEL_GENERAL_MODEL=/models/general \
  -e SEL_CODER_MODEL=/models/coder \
  sel-bot:latest sel-discord
```
Check logs with `docker logs -f sel-discord`. Enhanced flavor: replace `sel-discord` with `sel-discord-enhanced`.

If your `.env` already stores host paths, mount them at the same absolute path inside the container using `--mount type=bind,src="$SEL_GENERAL_MODEL",dst="$SEL_GENERAL_MODEL"`.

## Tests, Format, Lint
```bash
uv run ruff format
uv run ruff check --fix
uv run pytest tests -q
```

## Building Artifacts
```bash
uv build                 # wheel + sdist in dist/
uv pip install dist/*.whl
```
Container creation (`docker build -t sel-bot:latest .`) also gives you a runnable image with the CLI entrypoint.

## Utilities
- `scripts/fix_env.sh` – auto-maps host model directories to container paths.
- `cleanup_memory.py` – trims noisy memory entries (`uv run python cleanup_memory.py`).
- `scripts/download_piper_voice.sh en_US-amy-medium` – fetches Piper voices into `~/models/piper`.
- `sel_memory.json` / `sel_state.json` – runtime data files; keep them alongside the scripts.

## Troubleshooting
- Missing model warnings mean the container cannot see your mounted directories. Ensure the env var path matches the `-v` target.
- Discord DNS errors usually vanish with `--network host` or explicit `--dns 1.1.1.1 --dns 8.8.8.8`.
- Piper/ffmpeg must be on `PATH` for TTS playback; install them natively or inside the container.
