#!/usr/bin/env python3
"""
SEL Enhanced Discord Bot - Combines SEL's MoE architecture with Connor's human-like features
"""

import os
import sys
import asyncio
import discord
from discord.ext import commands, tasks
from typing import Optional, Dict, Any
from pathlib import Path
from collections import deque
import tempfile
import subprocess
import re
from datetime import datetime

# Web crawling
try:
    import requests
    from bs4 import BeautifulSoup
    WEB_CRAWL_AVAILABLE = True
except ImportError:
    WEB_CRAWL_AVAILABLE = False
    print("[Warning] Web crawling disabled: install requests and beautifulsoup4")

# Image/meme generation
try:
    from PIL import Image, ImageDraw, ImageFont
    import io
    MEME_GENERATION_AVAILABLE = True
except ImportError:
    MEME_GENERATION_AVAILABLE = False
    print("[Warning] Meme generation disabled: install Pillow")

# Import enhanced brain
sys.path.insert(0, str(Path(__file__).parent))
from sel_enhanced import MixtureBrain, _load_dotenv

_load_dotenv()


# ------------------------------
# Configuration
# ------------------------------

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    print("[Error] DISCORD_TOKEN not set in .env", file=sys.stderr)
    sys.exit(1)

STREAM = os.getenv("SEL_STREAM", "1").strip() in {"1", "true", "yes"}
TTS_DEFAULT = os.getenv("SEL_TTS_DEFAULT", "0").strip() in {"1", "true", "yes"}
DROP_OLD_SECONDS = int(os.getenv("SEL_DROP_OLD_SECONDS", "30"))


# ------------------------------
# Web Crawling Functions
# ------------------------------


def crawl_website(url: str) -> Dict[str, str]:
    """Crawl a website and extract content"""
    if not WEB_CRAWL_AVAILABLE:
        return {"error": "Web crawling not available"}

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove scripts and styles
        for script in soup(["script", "style"]):
            script.decompose()

        # Get title
        title = soup.find('title')
        title_text = title.get_text() if title else "No title"

        # Extract main content
        main_content = ""
        for selector in ['main', 'article', '.content', '.main']:
            main_elem = soup.select_one(selector)
            if main_elem:
                main_content = main_elem.get_text()
                break

        if not main_content:
            paragraphs = soup.find_all('p')
            main_content = ' '.join([p.get_text() for p in paragraphs[:5]])

        # Limit length
        if len(main_content) > 1500:
            main_content = main_content[:1500] + "..."

        return {
            'title': title_text.strip(),
            'content': main_content.strip(),
            'url': url
        }
    except Exception as e:
        return {
            'error': str(e),
            'url': url
        }


def analyze_webpage(brain: MixtureBrain, webpage_data: Dict, user_id: str) -> str:
    """Analyze webpage content using the brain"""
    if 'error' in webpage_data:
        return f"Couldn't access that page: {webpage_data['error']}"

    title = webpage_data.get('title', 'Unknown')
    content = webpage_data.get('content', 'No content')
    url = webpage_data.get('url', '')

    prompt = f"""I just read this webpage:
Title: {title}
URL: {url}
Content: {content[:800]}

Share your thoughts about it. What's interesting? What did you learn?"""

    return brain.respond(prompt, user_id=user_id, context={"is_dm": False, "mentioned_me": True})


# ------------------------------
# Meme Generation
# ------------------------------


def create_meme(image_url: str, top_text: str = "", bottom_text: str = "") -> Optional[bytes]:
    """Create a meme from image URL with text"""
    if not MEME_GENERATION_AVAILABLE:
        return None

    try:
        # Download image
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))

        # Resize if too large
        max_size = 800
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size))

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        draw = ImageDraw.Draw(img)

        # Try to use Impact font, fallback to default
        try:
            font_size = int(img.height / 10)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        # Draw top text
        if top_text:
            # Get text size
            bbox = draw.textbbox((0, 0), top_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (img.width - text_width) / 2
            y = 10
            # Outline
            for offset_x in [-2, 0, 2]:
                for offset_y in [-2, 0, 2]:
                    draw.text((x + offset_x, y + offset_y), top_text, font=font, fill='black')
            draw.text((x, y), top_text, font=font, fill='white')

        # Draw bottom text
        if bottom_text:
            bbox = draw.textbbox((0, 0), bottom_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (img.width - text_width) / 2
            y = img.height - text_height - 20
            # Outline
            for offset_x in [-2, 0, 2]:
                for offset_y in [-2, 0, 2]:
                    draw.text((x + offset_x, y + offset_y), bottom_text, font=font, fill='black')
            draw.text((x, y), bottom_text, font=font, fill='white')

        # Save to bytes
        output = io.BytesIO()
        img.save(output, format='PNG')
        output.seek(0)
        return output.read()

    except Exception as e:
        print(f"[Meme Error] {e}")
        return None


# ------------------------------
# Audio Utilities (STT/TTS)
# ------------------------------


def transcribe_audio(file_path: str) -> Optional[str]:
    """Transcribe audio file using faster-whisper"""
    try:
        from faster_whisper import WhisperModel

        model_size = os.getenv("STT_MODEL", "base.en")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, info = model.transcribe(file_path, beam_size=5)
        text = " ".join([seg.text for seg in segments])
        return text.strip()
    except Exception as e:
        print(f"[STT Error] {e}")
        return None


async def tts_to_file(text: str, output_path: str) -> bool:
    """Generate TTS audio file using Piper"""
    try:
        piper_model = os.getenv("PIPER_MODEL")
        if not piper_model or not Path(piper_model).exists():
            return False

        cmd = ["piper", "--model", piper_model, "--output_file", output_path]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate(input=text.encode())
        return proc.returncode == 0
    except Exception as e:
        print(f"[TTS Error] {e}")
        return False


# ------------------------------
# Discord Bot Setup
# ------------------------------


intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

bot = commands.Bot(command_prefix="!", intents=intents)
brain = MixtureBrain()

# Per-channel message history
channel_history: Dict[int, deque] = {}
MAX_HISTORY = 50


# ------------------------------
# Helper Functions
# ------------------------------


def split_message(text: str, max_length: int = 2000) -> list:
    """Split long messages"""
    if len(text) <= max_length:
        return [text]

    chunks = []
    current = ""
    for word in text.split():
        if len(current) + len(word) + 1 <= max_length:
            current += word + " "
        else:
            chunks.append(current.strip())
            current = word + " "
    if current:
        chunks.append(current.strip())
    return chunks


def get_history(channel_id: int) -> list:
    """Get conversation history for channel"""
    if channel_id not in channel_history:
        channel_history[channel_id] = deque(maxlen=MAX_HISTORY)
    # Convert to list of tuples
    history = []
    for item in channel_history[channel_id]:
        history.append((item["role"], item["content"]))
    return history


def add_to_history(channel_id: int, role: str, content: str):
    """Add message to history"""
    if channel_id not in channel_history:
        channel_history[channel_id] = deque(maxlen=MAX_HISTORY)
    channel_history[channel_id].append({"role": role, "content": content})


# ------------------------------
# Background Tasks
# ------------------------------


@tasks.loop(hours=1)
async def age_check():
    """Periodic age check and rebirth"""
    if brain.age_system.should_rebirth():
        print("[Age System] Rebirth triggered")
        brain.age_system.rebirth()
        # Could post to a channel here


@tasks.loop(hours=2)
async def hormone_circadian():
    """Update circadian rhythms"""
    brain.hormones.circadian_tick()


# ------------------------------
# Event Handlers
# ------------------------------


@bot.event
async def on_ready():
    print(f"[SEL Enhanced] Connected as {bot.user}")
    print(f"[SEL Enhanced] Age: {brain.age_system.current_age()}")
    print(f"[SEL Enhanced] Mood: {brain.hormones.mood()}")
    print(f"[SEL Enhanced] Streaming: {STREAM}")

    # Start background tasks
    if not age_check.is_running():
        age_check.start()
    if not hormone_circadian.is_running():
        hormone_circadian.start()


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # Process commands first
    await bot.process_commands(message)

    # Check message age
    if DROP_OLD_SECONDS > 0:
        age = (discord.utils.utcnow() - message.created_at).total_seconds()
        if age > DROP_OLD_SECONDS:
            return

    # Determine context
    is_dm = isinstance(message.channel, discord.DMChannel)
    mentioned_me = bot.user in message.mentions

    # Skip if not mentioned in channels (unless DM)
    if not is_dm and not mentioned_me:
        return

    # Get user ID
    user_id = str(message.author.id)

    # Get history
    history = get_history(message.channel.id)

    # Context
    context = {
        "is_dm": is_dm,
        "mentioned_me": mentioned_me,
        "channel_id": message.channel.id,
        "author": message.author.name,
    }

    # Clean content
    content = message.content
    if mentioned_me:
        content = content.replace(f"<@{bot.user.id}>", "").strip()

    # Check for attachments (audio/video transcription)
    if message.attachments:
        for att in message.attachments:
            if any(att.filename.lower().endswith(ext) for ext in ['.mp3', '.wav', '.ogg', '.m4a', '.webm', '.mp4']):
                async with message.channel.typing():
                    with tempfile.NamedTemporaryFile(suffix=Path(att.filename).suffix, delete=False) as tmp:
                        await att.save(tmp.name)
                        transcription = transcribe_audio(tmp.name)
                        Path(tmp.name).unlink()
                        if transcription:
                            content += f"\n[Audio transcription: {transcription}]"

    if not content.strip():
        return

    # Add to history
    add_to_history(message.channel.id, "user", content)

    # Generate response
    if STREAM:
        # Streaming response
        async with message.channel.typing():
            reply_msg = None
            accumulated = ""
            last_edit = 0

            for piece in brain.respond_stream(content, history, user_id=user_id, context=context):
                accumulated += piece
                now = asyncio.get_event_loop().time()

                # Edit every 0.5 seconds minimum
                if (now - last_edit) >= 0.5:
                    if reply_msg is None:
                        reply_msg = await message.reply(accumulated)
                    else:
                        try:
                            await reply_msg.edit(content=accumulated[:2000])
                        except discord.errors.HTTPException:
                            pass
                    last_edit = now

            # Final edit
            if reply_msg and accumulated:
                try:
                    await reply_msg.edit(content=accumulated[:2000])
                except discord.errors.HTTPException:
                    pass
            elif not reply_msg and accumulated:
                await message.reply(accumulated[:2000])
    else:
        # Non-streaming response
        async with message.channel.typing():
            reply = brain.respond(content, history, user_id=user_id, context=context)
            if reply and reply.strip():
                # Split if needed
                chunks = split_message(reply)
                for chunk in chunks:
                    await message.reply(chunk)
                add_to_history(message.channel.id, "assistant", reply)


# ------------------------------
# Commands
# ------------------------------


@bot.command(name="age")
async def age_command(ctx):
    """Show current age and personality phase"""
    age = brain.age_system.current_age()
    phase = brain.age_system.get_personality_phase()
    mood = brain.hormones.mood()

    embed = discord.Embed(title="SEL Status", color=discord.Color.blue())
    embed.add_field(name="Age", value=f"{age} years", inline=True)
    embed.add_field(name="Phase", value=phase.replace("_", " ").title(), inline=True)
    embed.add_field(name="Mood", value=mood.title(), inline=True)
    embed.add_field(name="Description", value=brain.age_system.get_age_description(), inline=False)

    await ctx.send(embed=embed)


@bot.command(name="beliefs")
async def beliefs_command(ctx):
    """Show current beliefs"""
    summary = brain.beliefs.get_summary(limit=10)
    await ctx.send(f"**Current Beliefs:**\n{summary}")


@bot.command(name="knowledge")
async def knowledge_command(ctx):
    """Show recent knowledge"""
    recent = brain.knowledge.recent(limit=5)
    if not recent:
        await ctx.send("No knowledge accumulated yet.")
        return

    text = "**Recent Learnings:**\n"
    for k in recent:
        text += f"• {k.summary}\n"
    await ctx.send(text)


@bot.command(name="rebirth")
async def rebirth_command(ctx):
    """Trigger manual rebirth"""
    old_age = brain.age_system.current_age()
    brain.age_system.rebirth()
    new_age = brain.age_system.current_age()
    await ctx.send(f"Reborn! Age reset from {old_age} to {new_age}. Starting a new cycle with accumulated wisdom.")


@bot.command(name="mood")
async def mood_command(ctx):
    """Show detailed hormone levels"""
    hormones = brain.hormones.state
    mood = brain.hormones.mood()

    text = f"**Current Mood: {mood.title()}**\n\nHormone Levels:\n"
    for hormone, level in hormones.items():
        bar_len = int(level * 10)
        bar = "█" * bar_len + "░" * (10 - bar_len)
        text += f"{hormone.title()}: {bar} {level:.2f}\n"

    await ctx.send(text)


@bot.command(name="memsvg")
async def memsvg_command(ctx):
    """Export memory as SVG"""
    user_id = str(ctx.author.id)
    with tempfile.TemporaryDirectory() as tmpdir:
        svg_path = Path(tmpdir) / f"memory_{user_id}.svg"
        brain.memory.export_svg(user_id, str(svg_path))
        if svg_path.exists():
            await ctx.send(file=discord.File(svg_path, filename="memory_map.svg"))
        else:
            await ctx.send("No memories to export yet.")


@bot.command(name="think")
async def think_command(ctx, *, topic: str):
    """Create a thought tree about a topic"""
    async with ctx.typing():
        tree_id = brain.create_thought_tree(topic)
        tree = brain.thought_trees.get(tree_id)
        if tree and tree.nodes:
            root_node = list(tree.nodes.values())[0]
            await ctx.send(f"**Thought Tree Created:** {tree_id}\n\n**Initial thought:**\n{root_node.content}")
        else:
            await ctx.send("Failed to create thought tree.")


@bot.command(name="crawl")
async def crawl_command(ctx, url: str):
    """Crawl and analyze a website"""
    if not WEB_CRAWL_AVAILABLE:
        await ctx.send("Web crawling not available. Install requests and beautifulsoup4.")
        return

    async with ctx.typing():
        webpage_data = crawl_website(url)
        user_id = str(ctx.author.id)
        analysis = analyze_webpage(brain, webpage_data, user_id)

        # Send title and analysis
        if 'title' in webpage_data:
            await ctx.send(f"**{webpage_data['title']}**\n{url}\n\n{analysis}")
        else:
            await ctx.send(analysis)


@bot.command(name="meme")
async def meme_command(ctx, image_url: str, *, text: str = ""):
    """Create a meme from image URL"""
    if not MEME_GENERATION_AVAILABLE:
        await ctx.send("Meme generation not available. Install Pillow.")
        return

    # Parse text (top/bottom separated by |)
    parts = text.split("|")
    top_text = parts[0].strip() if len(parts) > 0 else ""
    bottom_text = parts[1].strip() if len(parts) > 1 else ""

    async with ctx.typing():
        meme_bytes = create_meme(image_url, top_text, bottom_text)
        if meme_bytes:
            await ctx.send(file=discord.File(io.BytesIO(meme_bytes), filename="meme.png"))
        else:
            await ctx.send("Failed to create meme. Check the image URL and try again.")


@bot.command(name="voice_join")
async def voice_join_command(ctx):
    """Join voice channel"""
    if not ctx.author.voice:
        await ctx.send("You're not in a voice channel!")
        return

    channel = ctx.author.voice.channel
    if ctx.voice_client:
        await ctx.voice_client.move_to(channel)
    else:
        await channel.connect()
    await ctx.send(f"Joined {channel.name}")


@bot.command(name="voice_leave")
async def voice_leave_command(ctx):
    """Leave voice channel"""
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("Left voice channel")
    else:
        await ctx.send("Not in a voice channel")


@bot.command(name="speak")
async def speak_command(ctx, *, text: str):
    """Speak text in voice channel"""
    if not ctx.voice_client:
        await ctx.send("I'm not in a voice channel. Use !voice_join first.")
        return

    async with ctx.typing():
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            success = await tts_to_file(text, tmp.name)
            if success and Path(tmp.name).exists():
                audio_source = discord.FFmpegPCMAudio(tmp.name)
                ctx.voice_client.play(audio_source)
                await ctx.send("Speaking...")
                # Clean up after playing
                while ctx.voice_client.is_playing():
                    await asyncio.sleep(0.5)
                Path(tmp.name).unlink()
            else:
                await ctx.send("TTS failed. Check PIPER_MODEL configuration.")


@bot.command(name="reflect")
async def reflect_command(ctx, *, topic: str = ""):
    """Reflect on recent conversations"""
    history = get_history(ctx.channel.id)
    if not history:
        await ctx.send("No conversation history to reflect on.")
        return

    # Build reflection prompt
    recent_conv = "\n".join([f"{role}: {content[:100]}" for role, content in history[-10:]])

    if topic:
        prompt = f"Reflect on our recent conversation about {topic}:\n\n{recent_conv}\n\nWhat are your thoughts?"
    else:
        prompt = f"Reflect on our recent conversation:\n\n{recent_conv}\n\nWhat stands out to you?"

    async with ctx.typing():
        user_id = str(ctx.author.id)
        reflection = brain.respond(prompt, history, user_id=user_id, context={"is_dm": False, "mentioned_me": True})
        await ctx.send(reflection)


@bot.command(name="stats")
async def stats_command(ctx):
    """Show bot statistics"""
    age = brain.age_system.current_age()
    phase = brain.age_system.get_personality_phase()
    mood = brain.hormones.mood()

    # Get memory stats
    mem_stats = brain.memory.get_memory_stats()
    memory_count = mem_stats.get("total_memories", 0)
    unique_users = mem_stats.get("unique_users", 0)
    avg_importance = mem_stats.get("avg_importance", 0)

    belief_count = sum(len(beliefs) for beliefs in brain.beliefs.beliefs.values())
    knowledge_count = len(brain.knowledge.entries)
    thought_trees = len(brain.thought_trees)

    embed = discord.Embed(title="SEL Enhanced Statistics", color=discord.Color.green())
    embed.add_field(name="Age", value=f"{age} years", inline=True)
    embed.add_field(name="Phase", value=phase.replace("_", " ").title(), inline=True)
    embed.add_field(name="Mood", value=mood.title(), inline=True)
    embed.add_field(name="Total Memories", value=memory_count, inline=True)
    embed.add_field(name="Users Remembered", value=unique_users, inline=True)
    embed.add_field(name="Avg Memory Importance", value=f"{avg_importance:.2f}", inline=True)
    embed.add_field(name="Beliefs", value=belief_count, inline=True)
    embed.add_field(name="Knowledge", value=knowledge_count, inline=True)
    embed.add_field(name="Thought Trees", value=thought_trees, inline=True)
    embed.add_field(name="Interactions", value=brain.age_system.interaction_count, inline=True)

    await ctx.send(embed=embed)


# ------------------------------
# Main
# ------------------------------


def main():
    print("=" * 60)
    print("SEL Enhanced Discord Bot")
    print("Combining MoE Architecture with Human-like Intelligence")
    print("=" * 60)
    print(f"Features: MoE, Hormones, Memory, Beliefs, Knowledge, Age/Rebirth")
    print(f"Streaming: {STREAM}")
    print(f"TTS Default: {TTS_DEFAULT}")
    print(f"Web Crawling: {WEB_CRAWL_AVAILABLE}")
    print(f"Meme Generation: {MEME_GENERATION_AVAILABLE}")
    print("=" * 60)

    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
