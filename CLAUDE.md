# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SEL (Systematic Emotional Logic)** - A Discord bot with Mixture-of-Experts (MoE) architecture featuring local LLM integration, emotional simulation, context-aware memory systems, and multi-modal capabilities (text, voice, images).

## Development Environment

- **Python**: 3.11+ managed with `uv` package manager
- **Formatting**: `uv run ruff format <paths>`
- **Linting**: `uv run ruff check --fix <paths>`
- **Testing**: `uv run pytest path/to/tests` (run targeted tests on specific paths)
- **Dependencies**: Declared in `pyproject.toml`; keep additions minimal and justified

## Running the Bot

```bash
# Interactive CLI mode (for testing/development)
python3 sel.py

# Discord mode (production)
python3 sel_discord.py

# Docker deployment
docker build -t sel-bot .
docker run -v /path/to/models:/models sel-bot
```

## Required Environment Variables

Must be set in `.env` file (see `.env.example`):

- `DISCORD_TOKEN` - Discord bot token
- `SEL_GENERAL_MODEL` - Path to general conversation model (HuggingFace format)
- `SEL_CODER_MODEL` - Path to coder model (HuggingFace format)
- `PIPER_MODEL` - Path to Piper TTS model file (.onnx)
- `STT_MODEL` - Whisper model name (default: base.en)
- `SEL_TTS_DEFAULT` - Enable TTS by default (0/1)
- `SEL_QUANT` - Quantization mode: "4bit", "8bit", or empty for fp16
- `SEL_SINGLE_BACKEND` - Swap models to save VRAM: 1 (default) or 0

Optional configuration:
- `SEL_MODE` - Personality mode: casual, deep_focus, playful, quiet_companion
- `SEL_MAX_NEW_TOKENS` - Token generation limit (default: 128)
- `SEL_STREAM` - Enable streaming responses (default: 1)
- `SEL_CHECKINS` - Enable periodic check-ins (default: 0)
- `SEL_DROP_OLD_SECONDS` - Ignore messages older than X seconds (default: 30)
- `SEL_COMPILE` - Enable torch.compile optimization for 20-40% speedup (default: 1)

## Core Architecture

### MixtureBrain (sel.py:913 lines)

The intelligence core with:

**Dual Model System**: Maintains two specialized models (general and coder) with smart swapping when `SEL_SINGLE_BACKEND=1` to manage VRAM. Models are loaded with quantization support (4-bit/8-bit via bitsandbytes) and optimized attention (Flash Attention 2 or SDPA).

**Intent Detection**: Classifies user requests as `code`, `vision`, `memory`, or `general` to route to appropriate model/handler.

**Priority Classification**: Filters messages by urgency to avoid responding to low-priority chatter (returns `None` to signal silence).

**Hormone-Based Emotion Simulation**: Seven hormones (dopamine, serotonin, adrenaline, cortisol, melatonin, estrogen, testosterone) with decay rates, circadian rhythms, and stimulus-based updates. Influences response tone and emoji usage.

**Memory System**:
- Consent-based storage (only remembers users who opt-in)
- TTL-based expiration (soft: 14 days since last seen, hard: permanent retention)
- Vector similarity search using sentence-transformers embeddings
- Tag-based categorization
- SVG memory map visualization export (`!memsvg` command)
- Persists to `sel_memory.json`

**Style Mirroring**: Adapts to user's writing patterns (lowercase preference, emoji frequency, exclamation mark usage) for natural conversation flow.

**Streaming**: Token-by-token generation with real-time Discord message editing (0.4s minimum gap between edits for rate limiting).

### Discord Integration (sel_discord.py:383 lines)

**Message Handling**:
- Per-channel message history (50-message sliding window using deque)
- Ignores messages older than `SEL_DROP_OLD_SECONDS` to avoid processing backlogs
- Rate-limited streaming responses with edit batching

**Voice Support**:
- Join/leave voice channels
- TTS playback using Piper subprocess
- STT transcription of audio/video attachments via faster-whisper

**Commands**:
- `!memsvg` - Export user's memory visualization as SVG attachment
- Standard Discord events: on_ready, on_message, on_voice_state_update

**Check-ins**: Optional periodic engagement when `SEL_CHECKINS=1` (default disabled).

### Audio Processing (audio_utils.py:93 lines)

**STT**: faster-whisper with CUDA support, async execution, saves to temp WAV files.

**TTS**: Piper subprocess execution, generates WAV output for Discord voice playback.

### Alternative Implementation: connor.py (3000+ lines)

Separate Discord bot using OpenAI API with extended features:
- Thought tree generation and expansion
- Belief system updates with confidence levels
- Dream narrative generation
- Web crawling and meme generation
- YouTube audio processing
- Birthday/rebirth mechanics
- Hostility classification

Both bots can coexist but serve different use cases (SEL = local LLM, Connor = OpenAI API).

## Key Design Patterns

1. **Mixture of Experts**: Route requests to specialized models based on detected intent
2. **Consent-Based Memory**: Only store data for users who explicitly opt-in via conversation
3. **Hormonal State Machine**: Simulate emotional responses through hormone level dynamics
4. **Priority-Based Processing**: Filter low-priority messages to reduce computational load
5. **Graceful Degradation**: Fallback to simpler methods when dependencies (TTS/STT) unavailable
6. **Lazy Loading**: Import heavy dependencies (transformers, torch) only when needed
7. **Single Responsibility**: Clear separation (core logic, Discord client, audio, memory)

## Memory Management

**ConversationMemory** in sel.py:
- **Storage**: `sel_memory.json` (auto-created on first run)
- **Consent**: User must grant permission in conversation before storage begins
- **TTL Types**:
  - Soft: Expires 14 days after user's last interaction
  - Hard: Permanent retention (e.g., important personal facts)
- **Retrieval**: Cosine similarity search via sentence-transformers embeddings
- **Visualization**: SVG export shows items colored by tag, sized by recency

When working with memory:
- Never store data without consent flag set
- Use `remember()` with appropriate TTL and tags
- Call `prune_old_entries()` periodically for maintenance
- Test retrieval with `recall()` using semantic queries

## Model Loading and Generation

**Key considerations**:
- Models must be HuggingFace AutoModel-compatible (typically Qwen, Llama, Mistral architectures)
- Quantization requires compatible bitsandbytes config (4-bit: nf4, 8-bit: int8)
- Generation params: temperature=0.7, top_p=0.9, repetition_penalty=1.1, no beam search
- Flash Attention 2 auto-enabled if available, otherwise falls back to SDPA
- Device mapping: auto-distributed across available CUDA GPUs or CPU fallback

When swapping models (`SEL_SINGLE_BACKEND=1`):
- Explicitly delete model/tokenizer objects
- Call `torch.cuda.empty_cache()` to free VRAM
- Reload with same quantization settings

## Testing and Validation

Since this is an interactive bot:
- Test CLI mode with `python3 sel.py` for quick iteration
- Validate intent detection with varied prompts (code requests, vision, memory, general chat)
- Check hormone levels with diagnostic output after emotional stimuli
- Verify memory consent flow and retrieval accuracy
- Test streaming with long responses (>128 tokens)
- Validate audio transcription with sample files if modifying audio_utils.py

## Codex Agent Conventions (AGENTS.md)

- **@codex**: Fast fixes, surgical edits via apply_patch, ship immediately
- **@codex plan**: Multi-step tasks requiring written plan before editing
- **@codex build**: New features with multiple files, full validation cycle
- **@codex spec**: Early design when requirements unclear, propose before coding
- **@codex review**: Code review with severity-ordered findings and line references

All modes expect concise responses with explicit risk/follow-up callouts.

## Deployment Notes

**Docker**:
- Base image: `python:3.11-slim`
- Includes PyTorch with CUDA 12.1 support
- System deps: git, ffmpeg (for audio processing)
- Volume mount models directory: `/models`
- Default command: `python3 sel.py`

**Production considerations**:
- Ensure `.env` file present with all required variables
- Pre-download models before deployment (avoid runtime downloads)
- Monitor VRAM usage if running dual models without single backend mode
- Set `SEL_DROP_OLD_SECONDS` appropriately to avoid backlog processing on bot restart

## Performance Optimizations

**The codebase includes several performance optimizations**:

1. **Singleton Embedder with Caching (sel.py:118-175)**:
   - Embedder is now a singleton to avoid reloading the sentence-transformer model
   - LRU cache for embeddings (1000 most recent queries) reduces redundant computation
   - Shared across all ConversationMemory instances

2. **Periodic Memory Cleanup (sel.py:217-234)**:
   - Memory cleanup now runs every 5 minutes instead of on every search
   - Reduces I/O and improves query latency by ~50ms per message

3. **Torch.compile Optimization (sel.py:543-549)**:
   - PyTorch 2.0+ models auto-compile with "reduce-overhead" mode
   - Provides 20-40% inference speedup on CUDA
   - Disable with `SEL_COMPILE=0` if encountering compatibility issues

4. **Selective Memory Search (sel.py:839-844)**:
   - Memory search only triggered for messages with 4+ words or explicit memory intent
   - Reduces embedding computation by ~60% for typical chat patterns

5. **Optimized Discord Streaming (sel_discord.py:249-252)**:
   - Message edits throttled to 2Hz (was 4Hz) reducing API calls by 50%
   - Still feels responsive while avoiding rate limits

6. **Model Loading Bug Fix**:
   - Fixed critical bug where quantized models were discarded after loading
   - Ensures 4-bit/8-bit quantization actually works, saving significant VRAM

**Performance tuning tips**:
- Use `SEL_QUANT=4bit` for 3-4x VRAM savings with minimal quality loss
- Enable `SEL_SINGLE_BACKEND=1` to swap models and halve VRAM usage
- Set `SEL_MAX_NEW_TOKENS=128` or lower for faster responses
- Use `SEL_DROP_OLD_SECONDS=30` to avoid processing message backlogs

## Common Pitfalls

- **Missing models**: Ensure model paths in `.env` are absolute and accessible
- **VRAM exhaustion**: Enable `SEL_SINGLE_BACKEND=1` or use 4-bit quantization
- **Rate limiting**: Discord has strict message edit limits; streaming handles this automatically
- **Consent forgetting**: Memory consent is stored in `sel_state.json`; deleting this resets consent
- **Audio failures**: ffmpeg must be installed for TTS/STT; check subprocess error output
