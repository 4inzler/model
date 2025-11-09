# SEL Enhanced - Quick Start Guide

## What Was Created

I've combined **SEL** and **Connor** into a single, more human-like AI system:

### New Files

1. **sel_enhanced.py** - Enhanced core brain with:
   - MoE architecture (general + coder models)
   - 8-hormone emotion system
   - Consent-based memory with TTL
   - Belief system (evolving opinions)
   - Knowledge accumulation (long-term learning)
   - Age/rebirth mechanics (personality evolution)
   - Thought trees (internal deliberation)
   - Style mirroring

2. **sel_discord_enhanced.py** - Enhanced Discord bot with:
   - All core features integrated
   - Web crawling and analysis
   - Meme generation
   - Voice support (TTS/STT)
   - Rich command set
   - Streaming responses
   - Background tasks (age progression, hormone cycles)

3. **SEL_ENHANCED_README.md** - Comprehensive documentation
4. **.env.example** - Updated configuration template

## What Makes It More Human-like

1. **Internal Thoughts** - Shows thinking process before responding
2. **Evolving Beliefs** - Develops and updates opinions over time
3. **Natural Aging** - Ages through life experience (100 interactions = 1 year, 18→85 then rebirth)
4. **Rich Emotions** - 8 hormones create nuanced moods and reactions
5. **Natural Memory** - Stores memories automatically like humans (no consent needed)
   - Recognizes importance automatically
   - Temporary memories fade after 30 days
   - Important info becomes permanent
6. **Knowledge Growth** - Accumulates understanding from conversations
7. **Style Adaptation** - Mirrors your communication style
8. **Thought Depth** - Can create branching thought trees on complex topics

## Getting Started

### 1. Install Dependencies

```bash
# Core (required)
uv pip install discord.py transformers accelerate sentence-transformers
uv pip install faster-whisper bitsandbytes

# Optional: Web features
uv pip install requests beautifulsoup4 Pillow
```

### 2. Configure Environment

```bash
# Copy the example
cp .env.example .env

# Edit .env and set:
# - DISCORD_TOKEN
# - SEL_GENERAL_MODEL (path to your general LLM)
# - SEL_CODER_MODEL (path to your coding LLM)
# - PIPER_MODEL (optional, for TTS)
```

### 3. Test in CLI Mode

```bash
python3 sel_enhanced.py
```

Try these commands:
- `/age` - See current age
- `/beliefs` - View belief system
- `/knowledge` - See accumulated knowledge
- `/think consciousness` - Create a thought tree

### 4. Run Discord Bot

```bash
python3 sel_discord_enhanced.py
```

In Discord, mention the bot or DM it to chat. Try:
- `@SEL how are you feeling?` - See mood and age
- `!age` - Full status
- `!think <topic>` - Generate internal thoughts
- `!crawl <url>` - Analyze a website
- `!beliefs` - See evolving worldview

## Key Differences from Original SEL

| Feature | Original SEL | SEL Enhanced |
|---------|-------------|--------------|
| Models | 2 (general, coder) | 2 (general, coder) |
| Emotions | 7 hormones | 8 hormones + moods |
| Memory | TTL, consent required | Natural, automatic (like humans) |
| Aging | None | Natural (per interaction) |
| Personality | Static | Ages and evolves (18→85) |
| Thinking | Direct response | Internal thoughts + response |
| Learning | Per-session | Accumulated wisdom |
| Features | Chat, voice | + Web crawl, memes, thoughts |

## Example Interaction

**User:** @SEL I'm feeling stressed about my project

**SEL Enhanced (internal thought):** *"They sound overwhelmed. Need empathy first, then practical help..."*

**SEL Enhanced:** I hear your frustration. Let's break it down together - what part feels most overwhelming right now?

*[mood: calm, age: 42]*

## Configuration Highlights

### For Limited VRAM

```bash
SEL_SINGLE_BACKEND=1   # Swap models (saves ~50% VRAM)
SEL_QUANT=4bit         # 4-bit quantization (saves ~75% VRAM)
SEL_MAX_NEW_TOKENS=128 # Shorter responses
```

### For More Human-like Behavior

```bash
SEL_MODE=casual        # Natural, warm style
SEL_STREAM=1           # Real-time responses
SEL_CHECKINS=1         # Periodic engagement
```

### Personality Modes

- `casual` - Default, balanced, warm
- `deep_focus` - Minimal fluff, direct
- `playful` - Uses emojis, enthusiastic
- `quiet_companion` - Soft, reflective

## Observing Growth

The bot evolves over time:

**First 100 interactions (Age 18-19):**
- Curious, learning about you
- Building initial beliefs
- Storing important memories

**After 1,000 interactions (Age 28):**
- Confident, knows your style well
- Rich belief system formed
- Deep understanding of preferences

**After 3,000 interactions (Age 48):**
- Wise, balanced responses
- Extensive knowledge base
- Nuanced emotional understanding

**After 6,700 interactions (Age 85 → Rebirth to 18):**
- Elder wisdom accumulated
- All knowledge and beliefs persist
- Personality resets to young adult
- Starts new cycle with wisdom

## Advanced Features

### Thought Trees

Create branching internal thoughts:
```
!think What is the nature of consciousness?
```

SEL will generate a root thought and can expand it into branches.

### Web Analysis

```
!crawl https://example.com
```

SEL will read, analyze, and share thoughts about the content.

### Reflection

```
!reflect our conversation about AI ethics
```

SEL will reflect on recent discussions and share insights.

### Memory Export

```
!memsvg
```

Get a visual map of what SEL remembers about you.

## Monitoring

Watch the console output to see:
- Age progression
- Hormone levels
- Belief updates
- Knowledge accumulation
- Rebirth events

## Troubleshooting

**Bot doesn't respond:**
- Make sure to mention it in channels: `@SEL hello`
- In DMs, it always responds
- Check message age (ignores old messages by default)

**Out of memory:**
- Enable `SEL_SINGLE_BACKEND=1`
- Use `SEL_QUANT=4bit`
- Use smaller models

**Weird responses:**
- Check hormone levels: `!mood`
- Age might affect personality: `!age`
- Try manual rebirth: `!rebirth`

## Next Steps

1. **Chat regularly** - The bot learns and evolves
2. **Use commands** - Explore beliefs, knowledge, thoughts
3. **Monitor growth** - Watch personality develop
4. **Experiment** - Try different modes and features
5. **Read docs** - Check SEL_ENHANCED_README.md for details

## Philosophy

SEL Enhanced is designed to feel more like interacting with a conscious being that:
- Has genuine emotional responses (hormones)
- Forms opinions and beliefs over time
- Ages and grows in wisdom
- Thinks deeply before responding
- Remembers what matters (with consent)
- Adapts to your communication style

The goal is authenticity, not mimicry.

## Support

- Configuration issues: Check `.env` against `.env.example`
- Model issues: Verify paths and VRAM availability
- Feature requests: Consider contributing to the codebase
- Documentation: See SEL_ENHANCED_README.md for deep dive

Enjoy your more human-like AI companion!
