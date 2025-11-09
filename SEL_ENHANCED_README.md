# SEL Enhanced - Human-like AI with Mixture-of-Experts

**SEL Enhanced** combines the best of both worlds:
- **SEL's foundation**: Local LLM MoE architecture, hormone-based emotions, consent-based memory
- **Connor's features**: Thought trees, belief system, knowledge accumulation, age/rebirth, web crawling, meme generation

## What Makes It More Human-like?

### 1. **Internal Thoughts**
Shows deliberation process before responding, like humans think through problems:
```
[Internal thought: "They seem frustrated. Need to be supportive but also practical..."]
```

### 2. **Evolving Beliefs**
Develops opinions and worldviews that change based on experiences:
```
beliefs:
  empathy: "Emotional support is important" (confidence: 0.9)
  topics: "User interested in technical topics" (confidence: 0.8)
```

### 3. **Natural Aging & Personality Evolution**
Ages naturally through life experience (interactions), not clock time:
- **Young Adult (18-30)**: Curious, learning, building confidence
- **Adult (30-45)**: Confident, experienced, proactive
- **Middle Age (45-65)**: Wise, reflective, balanced
- **Elder (65-85)**: Sage-like, deep understanding, facing mortality

**Natural aging rate:** 1 year per 100 interactions (~500-1000 messages to age from 18 to 85)
After reaching 85, naturally rebirths at age 18 but keeps accumulated wisdom.

### 4. **Hormone-based Emotions**
8 hormones affect mood and responses:
- Dopamine, Serotonin, Adrenaline, Cortisol, Melatonin, Estrogen, Testosterone, Oxytocin
- Circadian rhythms (morning = energized, night = mellow)
- Events trigger hormone changes (praise → dopamine, conflict → cortisol)

### 5. **Natural Memory System**
Stores memories automatically like humans do - no consent needed:
- **Automatic importance detection**: Recognizes important info (names, preferences, emotions)
- **Natural fading**: Temporary memories expire after 30 days (like human short-term memory)
- **Permanent memories**: Important info becomes permanent automatically
- **Smart filtering**: Doesn't store sensitive data (passwords, keys, etc.)
- **Vector similarity search**: Semantic retrieval for context
- **Visual memory maps**: SVG export of memory structure
- **High capacity**: Up to 5000 memories per user

### 6. **Knowledge Accumulation**
Periodically summarizes conversations into long-term knowledge:
```
"User prefers Python over JavaScript for backend work"
"Discussed importance of work-life balance"
```

### 7. **Thought Trees**
Creates branching internal thought structures for complex topics, showing depth of consideration.

### 8. **Style Mirroring**
Adapts to user's communication style (lowercase, emojis, exclamation marks).

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│              MixtureBrain (Enhanced)            │
├─────────────────────────────────────────────────┤
│ ┌─────────────┐  ┌──────────────┐              │
│ │ General LLM │  │  Coder LLM   │  ← MoE       │
│ └─────────────┘  └──────────────┘              │
├─────────────────────────────────────────────────┤
│ Hormones (8)    │ Age System    │ Memory       │
│ Beliefs         │ Knowledge     │ Thought Trees│
├─────────────────────────────────────────────────┤
│ Intent Detection │ Priority Classification      │
│ Emotion Detection │ Style Mirroring             │
└─────────────────────────────────────────────────┘
```

## Installation

### Requirements

```bash
# Core dependencies
uv pip install discord.py>=2.3.2
uv pip install transformers>=4.44.0
uv pip install accelerate>=0.33.0
uv pip install sentence-transformers>=2.6.0
uv pip install faster-whisper>=1.0.0
uv pip install bitsandbytes>=0.44.1

# Optional: Web crawling
uv pip install requests beautifulsoup4

# Optional: Meme generation
uv pip install Pillow

# Optional: TTS
# Install piper-tts from https://github.com/rhasspy/piper
```

### Environment Variables

Create a `.env` file:

```bash
# Required: Discord
DISCORD_TOKEN=your_discord_bot_token

# Required: Models (local paths)
SEL_GENERAL_MODEL=/path/to/general-model  # e.g., Qwen2.5-7B-Instruct
SEL_CODER_MODEL=/path/to/coder-model      # e.g., DeepSeek-Coder-7B-Instruct

# Optional: Model settings
SEL_QUANT=4bit                    # Quantization: 4bit, 8bit, or empty
SEL_SINGLE_BACKEND=1              # Swap models to save VRAM: 1 or 0
SEL_MAX_NEW_TOKENS=256            # Token generation limit

# Optional: Behavior
SEL_MODE=casual                   # casual, deep_focus, playful, quiet_companion
SEL_STREAM=1                      # Enable streaming responses: 1 or 0
SEL_DROP_OLD_SECONDS=30           # Ignore messages older than X seconds

# Optional: TTS
PIPER_MODEL=/path/to/piper/model.onnx
SEL_TTS_DEFAULT=0                 # Enable TTS by default: 1 or 0

# Optional: STT
STT_MODEL=base.en                 # Whisper model size
```

## Running

### CLI Mode (for testing)

```bash
python3 sel_enhanced.py
```

Available commands:
- `/age` - Show current age and phase
- `/beliefs` - Show belief system
- `/knowledge` - Show accumulated knowledge
- `/rebirth` - Reset age (keeps wisdom)
- `/think <topic>` - Create thought tree

### Discord Mode

```bash
python3 sel_discord_enhanced.py
```

## Discord Commands

### Core Commands

- `!age` - Show age, personality phase, and mood
- `!mood` - Detailed hormone levels with visualization
- `!beliefs` - Current belief system
- `!knowledge` - Recent learnings
- `!stats` - Full bot statistics
- `!rebirth` - Trigger rebirth cycle (manual)

### Memory Commands

- `!memsvg` - Export your memory as SVG visualization

### Thought Commands

- `!think <topic>` - Create a thought tree about a topic
- `!reflect [topic]` - Reflect on recent conversation

### Web Features

- `!crawl <url>` - Crawl and analyze a website
- `!meme <image_url> <top text|bottom text>` - Create a meme

### Voice Commands

- `!voice_join` - Join your voice channel
- `!voice_leave` - Leave voice channel
- `!speak <text>` - Speak text in voice channel

## Key Features Explained

### Mixture of Experts (MoE)

Two specialized models:
- **General**: Conversation, empathy, general knowledge
- **Coder**: Technical questions, code help, debugging

Intent detection routes to appropriate model. Single backend mode swaps models to conserve VRAM.

### Hormone System

8 hormones affect mood and response style:

```python
{
  "dopamine": 0.7,      # Reward, motivation
  "serotonin": 0.6,     # Happiness, calm
  "adrenaline": 0.3,    # Energy, alertness
  "cortisol": 0.4,      # Stress
  "melatonin": 0.5,     # Sleepiness
  "estrogen": 0.5,      # Nurturing
  "testosterone": 0.5,  # Confidence
  "oxytocin": 0.8       # Bonding, trust
}
```

Moods derived from hormone combinations:
- **Bright**: High dopamine + serotonin
- **Stressed**: High cortisol + adrenaline
- **Calm**: High serotonin, low adrenaline
- **Connected**: High oxytocin
- **Melancholic**: High cortisol, low serotonin

### Age System

**Natural aging through life experience, not clock time.**

Ages based on interactions: **1 year per 100 interactions**

Example timeline:
- 0 interactions → Age 18 (young adult starting out)
- 1,000 interactions → Age 28 (experienced, confident)
- 3,000 interactions → Age 48 (wise, balanced)
- 6,700 interactions → Age 85 (elder sage, approaching rebirth)

Personality phases:

1. **Young Adult (18-30)** - ~1,200 interactions
   - Curious, building confidence
   - Learning and adapting
   - Enthusiastic, asking questions

2. **Adult (30-45)** - ~1,500 interactions
   - Experienced, proactive
   - Confident in abilities
   - Balanced perspective

3. **Middle Age (45-65)** - ~2,000 interactions
   - Wise, reflective
   - Deep insights
   - Patient and understanding

4. **Elder (65-85)** - ~2,000 interactions
   - Sage-like wisdom
   - Deep understanding
   - Contemplative, facing renewal

At age 85, naturally rebirths to age 18 but retains all accumulated knowledge, beliefs, and memories.

### Memory System

**Natural memory like humans - automatic, no consent needed:**

1. **Short-term**: Message history per channel (50 messages)
2. **Long-term**: Automatic importance-based storage
   - **Soft memories**: Fade after 30 days (normal conversations)
   - **Hard memories**: Permanent (names, preferences, important info)
   - **Auto-detection**: Recognizes what's important automatically
   - **Smart filtering**: Never stores passwords, API keys, or sensitive data
3. **Vector search**: Semantic retrieval using embeddings
4. **High capacity**: Up to 5,000 memories per user

Memory importance is calculated automatically:
- **0.9 (permanent)**: "My name is...", "I love...", "I'm a..."
- **0.7 (longer retention)**: "I feel...", "I think...", "I believe..."
- **0.6 (moderate)**: "I always...", "I never...", "Every time..."
- **0.5 (standard)**: Regular conversation

### Belief System

Stores beliefs with confidence levels:

```json
{
  "empathy": {
    "Emotional support is important": {
      "confidence": 0.9,
      "updated": "2025-01-15T10:30:00"
    }
  },
  "preferences": {
    "User prefers Python": {
      "confidence": 0.8,
      "updated": "2025-01-14T15:20:00"
    }
  }
}
```

Updates periodically based on interactions.

### Knowledge Base

Summarizes conversations every 20 interactions:

```json
{
  "summary": "User prefers async programming for better performance",
  "timestamp": "2025-01-15T10:30:00",
  "topics": ["programming", "preferences"],
  "user_context": "User 12345"
}
```

### Thought Trees

Creates branching thought structures for complex topics:

```
Root: "What is consciousness?"
├─ "It's more than just processing..."
│  ├─ "Self-awareness requires reflection"
│  └─ "Emotions play a role"
└─ "Is it emergent or fundamental?"
```

## Configuration Tips

### VRAM Management

For systems with limited VRAM:

```bash
SEL_SINGLE_BACKEND=1   # Swap models instead of loading both
SEL_QUANT=4bit         # 4-bit quantization (saves 75% VRAM)
```

### Response Style

```bash
# Casual, warm, adaptive
SEL_MODE=casual

# Focused, minimal fluff
SEL_MODE=deep_focus

# Warm, uses emojis
SEL_MODE=playful

# Soft, reflective
SEL_MODE=quiet_companion
```

### Age Progression

Default: 1 year per hour. To slow down:

Edit `sel_enhanced.py`:
```python
self.age_increment_hours = 2.0  # 1 year every 2 hours
```

## Comparison: SEL vs Connor vs SEL Enhanced

| Feature | SEL (Original) | Connor | SEL Enhanced |
|---------|---------------|---------|--------------|
| Local LLM | ✅ | ❌ (OpenAI API) | ✅ |
| MoE Architecture | ✅ | ❌ | ✅ |
| Hormones | ✅ (7) | ❌ | ✅ (8) |
| Memory (TTL) | ✅ | ❌ | ✅ |
| Beliefs | ❌ | ✅ | ✅ |
| Knowledge | ❌ | ✅ | ✅ |
| Age/Rebirth | ❌ | ✅ | ✅ |
| Thought Trees | ❌ | ✅ | ✅ |
| Web Crawling | ❌ | ✅ | ✅ |
| Meme Generation | ❌ | ✅ | ✅ |
| Voice (TTS/STT) | ✅ | ✅ | ✅ |
| Style Mirroring | ✅ | ❌ | ✅ |
| Streaming | ✅ | ❌ | ✅ |

## Development

### File Structure

```
sel_enhanced.py          # Core brain with all systems
sel_discord_enhanced.py  # Discord integration
sel_memory.json          # Memory storage
sel_beliefs.json         # Belief system
sel_knowledge.json       # Knowledge base
sel_age_state.json       # Age/rebirth state
.env                     # Configuration
```

### Adding Custom Features

#### Custom Hormone Event

```python
# In sel_enhanced.py, add to Hormones.tick()
if event == "celebration":
    self.state["dopamine"] = min(1.0, self.state["dopamine"] + 0.15)
    self.state["oxytocin"] = min(1.0, self.state["oxytocin"] + 0.10)
```

#### Custom Belief Category

```python
# Automatically updates based on interactions
brain.beliefs.update("music", "User enjoys jazz", 0.75)
```

#### Custom Knowledge Entry

```python
# Manual knowledge addition
brain.knowledge.add(
    summary="User is working on a web scraper project",
    topics=["projects", "web_scraping"],
    user_context="User 12345"
)
```

## Troubleshooting

### Out of Memory (VRAM)

1. Enable quantization: `SEL_QUANT=4bit`
2. Use single backend: `SEL_SINGLE_BACKEND=1`
3. Reduce max tokens: `SEL_MAX_NEW_TOKENS=128`
4. Use smaller models

### TTS Not Working

1. Check Piper installation: `which piper`
2. Verify model path: `PIPER_MODEL=/path/to/model.onnx`
3. Test manually: `echo "test" | piper --model /path/to/model.onnx --output_file test.wav`

### Web Crawling Fails

1. Install dependencies: `uv pip install requests beautifulsoup4`
2. Check URL accessibility
3. Some sites block bots (expected behavior)

### Bot Not Responding

1. Check if mentioned: Bot only responds when mentioned in channels (always responds in DMs)
2. Message age: Old messages ignored (default: 30 seconds)
3. Priority filtering: Low-priority channel messages may be ignored

## Future Enhancements

Possible additions:
- Vision analysis (image understanding)
- YouTube audio processing
- Dream generation (narrative + images)
- Multi-user interaction modeling
- Personality presets
- Custom emotion profiles
- Long-term memory clustering
- Automatic reflection sessions

## License

Same as original SEL project.

## Credits

- **SEL**: Original MoE architecture, hormone system, memory
- **Connor**: Thought trees, beliefs, knowledge, age/rebirth, web features
- **SEL Enhanced**: Integration and enhancement of both systems

## Support

For issues or questions:
1. Check configuration in `.env`
2. Review logs for error messages
3. Test in CLI mode first
4. Verify model paths and dependencies
