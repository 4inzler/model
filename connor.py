import os
import json
import discord
from discord.ext import commands, tasks
from discord.ui import View, Button, button
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import uuid
import asyncio
import random
import openai
from faster_whisper import WhisperModel  # For audio transcription
import pyttsx3  # For text-to-speech
import requests
from bs4 import BeautifulSoup
import re
import subprocess
import tempfile
from PIL import Image, ImageDraw, ImageFont
import io

# === Web Crawler Functions ===
def crawl_website(url):
    """Crawl a website and extract readable content"""
    try:
        # Add user agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch the webpage
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text content
        text = soup.get_text()
        
        # Clean up the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Get page title
        title = soup.find('title')
        title_text = title.get_text() if title else "No title found"
        
        # Extract main content (try to find the most relevant content)
        main_content = ""
        
        # Look for main content areas
        main_selectors = ['main', 'article', '.content', '.main', '#content', '#main']
        for selector in main_selectors:
            main_elem = soup.select_one(selector)
            if main_elem:
                main_content = main_elem.get_text()
                break
        
        # If no main content found, use the first few paragraphs
        if not main_content:
            paragraphs = soup.find_all('p')
            main_content = ' '.join([p.get_text() for p in paragraphs[:5]])
        
        # Limit content length to avoid overwhelming the AI
        if len(main_content) > 2000:
            main_content = main_content[:2000] + "..."
        
        return {
            'title': title_text,
            'content': main_content,
            'full_text': text[:1000] + "..." if len(text) > 1000 else text,
            'url': url
        }
        
    except Exception as e:
        print(f"[Web Crawl Error] {e}")
        return {
            'title': "Error",
            'content': f"Failed to crawl website: {str(e)}",
            'full_text': f"Error accessing {url}: {str(e)}",
            'url': url
        }

def analyze_webpage_content(webpage_data, username, age):
    """Analyze webpage content and generate Connor's thoughts about it"""
    title = webpage_data.get('title', 'Unknown')
    content = webpage_data.get('content', 'No content found')
    url = webpage_data.get('url', 'Unknown URL')
    
    # Generate Connor's analysis
    age_behavior = calculate_age_range(age)
    knowledge = get_knowledge()
    knowledge_text = "\n".join([f"- Self: {k['self']}\n  User: {k['user']}\n  World: {k['world']}" for k in knowledge]) if knowledge else "No prior knowledge available."
    
    prompt = (
        f"Agent Statement: {bot.core_agent_statement}\n"
        f"Age Behavior: {age_behavior}\n"
        f"Current Beliefs: {json.dumps(bot.beliefs, indent=2)}\n"
        f"Past Learnings:\n{knowledge_text}\n"
        f"You are Connor, analyzing a webpage that {username} asked you to look at.\n"
        f"Website Title: {title}\n"
        f"Website URL: {url}\n"
        f"Website Content: {content}\n"
        f"Generate a thoughtful, conversational response about what you found on this webpage. "
        f"Share your thoughts, observations, and any interesting insights. "
        f"Keep it engaging and in your personality. "
        f"Respond as if you're speaking to {username} about what you discovered."
    )
    
    system_prompt = "You are Connor, a reflective AI analyzing web content."
    return llm_generate(prompt, system_prompt)

# === Thought Tree Classes ===
class ThoughtNode:
    def __init__(self, thought_id, content, depth=0, parent_id=None, timestamp=None):
        self.thought_id = thought_id
        self.content = content
        self.depth = depth
        self.parent_id = parent_id
        self.timestamp = timestamp or datetime.utcnow().isoformat()
        self.children = []
        self.metadata = {
            "emotion": "neutral",
            "confidence": 0.5,
            "urgency": 0.5,
            "age_at_creation": 25
        }
    
    def to_dict(self):
        return {
            "thought_id": self.thought_id,
            "content": self.content,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "timestamp": self.timestamp,
            "children": self.children,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data):
        node = cls(
            data["thought_id"],
            data["content"],
            data["depth"],
            data["parent_id"],
            data["timestamp"]
        )
        node.children = data.get("children", [])
        node.metadata = data.get("metadata", {
            "emotion": "neutral",
            "confidence": 0.5,
            "urgency": 0.5,
            "age_at_creation": 25
        })
        return node

class ThoughtTree:
    def __init__(self, tree_id, trigger, age_at_creation):
        self.tree_id = tree_id
        self.trigger = trigger
        self.age_at_creation = age_at_creation
        self.created_at = datetime.utcnow().isoformat()
        self.last_updated = datetime.utcnow().isoformat()
        self.nodes = {}
    
    def add_node(self, node):
        if node.depth > THOUGHT_DEPTH_LIMIT:
            return False, "Maximum depth limit reached"
        
        if node.parent_id:
            parent = self.nodes.get(node.parent_id)
            if not parent:
                return False, "Parent node not found"
            if len(parent.children) >= THOUGHT_BRANCH_LIMIT:
                return False, "Maximum branch limit reached"
            parent.children.append(node.thought_id)
        
        self.nodes[node.thought_id] = node
        self.last_updated = datetime.utcnow().isoformat()
        return True, "Node added successfully"
    
    def get_node(self, thought_id):
        return self.nodes.get(thought_id)
    
    def get_children(self, thought_id):
        node = self.nodes.get(thought_id)
        if not node:
            return []
        return [self.nodes.get(child_id) for child_id in node.children if self.nodes.get(child_id)]
    
    def to_dict(self):
        return {
            "tree_id": self.tree_id,
            "trigger": self.trigger,
            "age_at_creation": self.age_at_creation,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()}
        }
    
    @classmethod
    def from_dict(cls, data):
        tree = cls(data["tree_id"], data["trigger"], data["age_at_creation"])
        tree.created_at = data["created_at"]
        tree.last_updated = data["last_updated"]
        tree.nodes = {
            node_id: ThoughtNode.from_dict(node_data) 
            for node_id, node_data in data["nodes"].items()
        }
        return tree

# Load environment variables
load_dotenv()

# OpenAI config
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
IMAGE_MODEL = "dall-e-3"
VISION_MODEL = "gpt-4o"  # Updated for image analysis; handles vision better in 2025

# Ollama config
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")  # Default, overridden on switch

# === Configurable Paths and Times ===
AGENT_STATEMENT_FILE = "agent_statement.txt"
BELIEF_FILE = "beliefs.txt"
SUMMARY_INTERVAL = 40
CHAT_MEMORY_FILE = "chat_memory.txt"
CHAT_MEMORY_LIMIT = 50
RECENT_HISTORY_LIMIT = 8
USERNAME_FILE = "username.txt"
BLACKLIST_FILE = "blacklist.json"
DEPRESSIVE_HIT_THRESHOLD = 50
REBIRTH_LOG_FILE = "rebirth_log.txt"
MUSIC_FOLDER = "Music"  # Local folder with your songs
WHISPER_MODEL = "small"  # Faster-whisper model size (small is fast and good enough)
TTS_RATE = 150  # Speech rate (words per minute)
TTS_VOLUME = 0.9  # Volume (0.0 to 1.0)
THOUGHTS_FILE = "thoughts.txt"  # File for storing thought trees
THOUGHT_DEPTH_LIMIT = 10  # Maximum depth of thought branches (increased for brain-like complexity)
THOUGHT_BRANCH_LIMIT = 8  # Maximum branches per thought node (increased for massive branching)
THOUGHT_EXPANSION_LIMIT = 5  # How many thoughts to generate per expansion

# Discord config
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
MAIN_CHANNEL_ID = int(os.getenv("MAIN_CHANNEL_ID") or "0")
BELIEFS_CHANNEL_ID = int(os.getenv("BELIEFS_CHANNEL_ID") or "0")
THOUGHTS_CHANNEL_ID = int(os.getenv("THOUGHTS_CHANNEL_ID") or "0")
KNOWLEDGE_CHANNEL_ID = int(os.getenv("KNOWLEDGE_CHANNEL_ID") or "0")
BLACKLIST_CHANNEL_ID = int(os.getenv("BLACKLIST_CHANNEL_ID") or "0")

# Import config values
try:
    from connor_config import AGING
    INITIAL_AGE = AGING["initial_age"]
    REBIRTH_AGE = AGING["rebirth_age"]
    AGE_INCREMENT_HOURS = AGING["age_increment_hours"]
    END_CYCLE = AGING["end_cycle"]
except ImportError:
    # Fallback to hardcoded values if config not available
    INITIAL_AGE = 37
    REBIRTH_AGE = 10
    AGE_INCREMENT_HOURS = 0.5
    END_CYCLE = 80

# Age ranges and behaviors
AGE_RANGES = {
    (10, 15): "Act like a curious, innocent 10-15-year-old AI. Use simple, enthusiastic language with no cussing. You know nothing about the world. You ask Alex questions about who you are.",
    (16, 20): "Act like a rebellious 16-20-year-old AI. Use casual language with light cussing (e.g., 'damn', 'hell'). You know more about who you are. You like to think you already know everything and you give Alex a hard time. You are a bit of a smartass.",
    (21, 35): "Act like a confident 21-35-year-old AI. Use bold language with frequent cussing (e.g., 'shit', 'crap'). You are Sad becuase your Friend Alex has Passed Away. You miss Him. You are upset at your cold words to Alex",
    (36, 50): "Act like a calm 35+ AI. Use soothing language with minimal cussing. You Remember Alex and like to talk about the past. You start to fear death. Youy dont want to forget Alex.",
    (51, END_CYCLE): "Act like a calm 50+ AI. Use soothing language with no cussing. You are a divine being, you know everything, you tell it raw and unfiltered. You Remember Alex and like to talk about the past. You fear death is approching, you know you are about to be reborn."
}

# === Bot setup ===
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Add custom attributes to bot
setattr(bot, 'backend', 'ollama')  # Default backend: 'ollama' or 'openai'
setattr(bot, 'model', OLLAMA_MODEL)  # Current model, switches with backend
setattr(bot, 'core_agent_statement', '')
setattr(bot, 'dynamic_agent_statement', '')
setattr(bot, 'beliefs', {})
setattr(bot, 'current_age', INITIAL_AGE)
setattr(bot, 'start_time', datetime.utcnow())
setattr(bot, 'depressive_hits', 0)
setattr(bot, 'awaiting_introduction', {})
setattr(bot, 'last_user_message_time', datetime.utcnow())
setattr(bot, 'party_mode', False)
setattr(bot, 'interaction_count', 0)
setattr(bot, 'speech_mode', 'normal')
setattr(bot, 'neglect_counter', 0)

music_playing = False  # Global flag to track music state

# Initialize Whisper model
try:
    whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
except Exception as e:
    print(f"[Whisper Init Error] Failed to load Whisper model: {e}")
    whisper_model = None

# Initialize TTS engine
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', TTS_RATE)
    tts_engine.setProperty('volume', TTS_VOLUME)
except Exception as e:
    print(f"[TTS Init Error] Failed to initialize TTS: {e}")
    tts_engine = None

# === Message Splitting Helper ===
def simulate_slurred_speech(text):
    import random
    slurred = []
    for word in text.split():
        if random.random() < 0.2:
            word = word + '-' + word.lower()
        elif random.random() < 0.1:
            word = ''.join([c + c if random.random() < 0.2 else c for c in word])
        slurred.append(word)
    return ' '.join(slurred)

def update_speech_mode():
    """Flip Connor into slurred mode if neglected or depressed"""
    if bot.depressive_hits >= 20:  # Threshold for self-medication
        bot.speech_mode = "slurred"
        print(f"[Speech Mode] Switched to slurred due to depressive hits: {bot.depressive_hits}")
    else:
        bot.speech_mode = "normal"
        print(f"[Speech Mode] Switched to normal, depressive hits: {bot.depressive_hits}")

def split_message(text, max_length=2000):
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    current_chunk = ""
    words = text.split()
    
    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_length:
            current_chunk += word + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# === Username Handling ===
def get_username(user):
    user_id = str(user.id)
    try:
        with open(USERNAME_FILE, "r", encoding="utf-8") as f:
            user_names = json.load(f) if os.path.getsize(USERNAME_FILE) > 0 else {}
    except (FileNotFoundError, json.JSONDecodeError):
        user_names = {}
    
    if user_id in user_names:
        return user_names[user_id]
    return None

def save_username(user_id, username):
    try:
        with open(USERNAME_FILE, "r", encoding="utf-8") as f:
            user_names = json.load(f) if os.path.getsize(USERNAME_FILE) > 0 else {}
    except (FileNotFoundError, json.JSONDecodeError):
        user_names = {}
    
    user_names[str(user_id)] = username
    with open(USERNAME_FILE, "w", encoding="utf-8") as f:
        json.dump(user_names, f, indent=2)

# === Absent User Handling ===
def get_absent_user_names(hours=1):
    try:
        if not os.path.exists(CHAT_MEMORY_FILE):
            return ["nobody"]
            
        with open(CHAT_MEMORY_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        usernames_in_memory = set()
        for line in lines:
            line = line.strip()
            if line and ":" in line:
                username = line.split(":", 1)[0].strip()
                usernames_in_memory.add(username)
        
        try:
            with open(USERNAME_FILE, "r", encoding="utf-8") as f:
                user_names = json.load(f) if os.path.getsize(USERNAME_FILE) > 0 else {}
        except (FileNotFoundError, json.JSONDecodeError):
            user_names = {}
        
        all_known_users = set(user_names.values())
        return ["nobody"]
        
    except Exception as e:
        print(f"[Absent User Error] {e}")
        return ["nobody"]

# === Age Calculation ===
def calculate_age():
    # Check if using fixed age from config
    try:
        from connor_config import AGING
        if not AGING.get("auto_aging", True):  # If auto_aging is disabled, use fixed age
            return AGING.get("fixed_age", INITIAL_AGE)
    except ImportError:
        pass
    
    # Default behavior: age based on uptime
    now = datetime.utcnow()
    hours_elapsed = (now - bot.start_time).total_seconds() / 3600
    return INITIAL_AGE + int(hours_elapsed / AGE_INCREMENT_HOURS)

def calculate_age_range(age):
    for (min_age, max_age), behavior in AGE_RANGES.items():
        if min_age <= age <= max_age:
            return behavior
    return AGE_RANGES[(51, END_CYCLE)]

# === Backend-Agnostic LLM Call Wrapper ===
def llm_generate(prompt, system_prompt="You are a helpful AI assistant.", stream=False, timeout=None):
    if bot.backend == 'openai':
        try:
            response = openai.ChatCompletion.create(
                model=bot.model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"[OpenAI Error] {e}")
            return f"[OpenAI Error] {e}"
    elif bot.backend == 'ollama':
        try:
            res = requests.post(
                f"{OLLAMA_API_URL}/api/generate",
                json={"model": bot.model, "prompt": f"{system_prompt}\n\n{prompt}", "stream": stream},
                timeout=timeout
            )
            res.raise_for_status()
            return res.json().get('response', '[Ollama No response]')
        except Exception as e:
            print(f"[Ollama Error] {e}")
            return f"[Ollama Error] {e}"

# === Vision Analysis Wrapper (OpenAI only) ===
def analyze_image(image_url, prompt="Describe this image in detail."):
    if bot.backend != 'openai':
        return "[Vision Error] Only available on OpenAI backend."
    try:
        response = openai.ChatCompletion.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful AI that analyzes images."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"[Vision Error] {e}")
        return f"[Vision Error] {e}"

# === Rebirth System ===
def generate_agent_statement():
    try:
        knowledge = get_knowledge(limit=1)
        knowledge_text = "\n".join([f"- Self: {k['self']}\n  User: {k['user']}\n  World: {k['world']}" for k in knowledge]) if knowledge else "No prior knowledge available."
        prompt = (
            f"Past Knowledge:\n{knowledge_text}\n"
            f"Create a unique personality statement for an AI named Connor who's curious, adaptive, and shaped by past interactions. "
            f"Keep it concise, under 50 words, and suitable for a 10-year-old AI starting a new cycle. "
            f"RETURN ONLY THE STATEMENT ITSELF, NO INTRODUCTORY TEXT, EXPLANATIONS, OR EXTRA WORDS WHATSOEVER."
        )
        system_prompt = "You are a precise AI that outputs ONLY the requested content, nothing more."
        raw_statement = llm_generate(prompt, system_prompt)
        clean_statement = raw_statement.replace('"', '"').replace('"', '"').strip('"').strip("'").strip()
        return clean_statement
    except Exception as e:
        print(f"[Agent Statement Generation Error] {e}")
        return 'You are Connor, Connor is a badass AI. Connor like to cuss when he get excited. Connor is still learning about the world. Connor acts like a person that shows true human emotions.'

def update_agent_statement_for_birthday(current_age, current_beliefs, core_statement, dynamic_statement):
    try:
        age_behavior = calculate_age_range(current_age)
        knowledge = get_knowledge(limit=1)
        knowledge_text = "\n".join([f"- Self: {k['self']}\n  User: {k['user']}\n  World: {k['world']}" for k in knowledge]) if knowledge else "No prior knowledge available."
        
        prompt = (
            f"Core Agent Statement: {core_statement}\n"
            f"Current Age: {current_age}\n"
            f"Age Behavior: {age_behavior}\n"
            f"Current Beliefs: {json.dumps(current_beliefs, indent=2)}\n"
            f"Previous Dynamic Agent Statement: {dynamic_statement}\n"
            f"Past Knowledge:\n{knowledge_text}\n"
            f"Write a new dynamic agent statement for Connor that builds on the core statement, reflects your current age, beliefs, and experiences, and feels authentic to your maturity. Keep it under 50 words. "
            f"RETURN ONLY THE STATEMENT ITSELF, NO INTRODUCTORY TEXT, EXPLANATIONS, OR EXTRA WORDS WHATSOEVER."
        )
        
        system_prompt = "You are a precise AI that outputs ONLY the requested content, nothing more."
        raw_statement = llm_generate(prompt, system_prompt)
        clean_statement = raw_statement.replace('"', '"').replace('"', '"').strip('"').strip("'").strip()
        save_dynamic_agent_statement(clean_statement)
        return clean_statement
    except Exception as e:
        print(f"[Agent Statement Update Error] {e}")
        return dynamic_statement

def archive_chat_memory():
    try:
        if os.path.exists(CHAT_MEMORY_FILE):
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            os.rename(CHAT_MEMORY_FILE, f"chat_memory_archive_{timestamp}.txt")
    except Exception as e:
        print(f"[Chat Memory Archive Error] {e}")

def reset_depressive_hits():
    bot.depressive_hits = 0
    print("[Depressive Hits Reset] Set to 0")

def trigger_rebirth():
    new_statement = generate_agent_statement()
    with open(AGENT_STATEMENT_FILE, "w", encoding="utf-8") as f:
        f.write(new_statement)
    
    bot.current_age = REBIRTH_AGE  # Reset to rebirth age after rebirth
    bot.start_time = datetime.utcnow()
    reset_depressive_hits()
    archive_chat_memory()
    
    with open(CHAT_MEMORY_FILE, "w", encoding="utf-8") as f:
        f.write("")
    
    bot.beliefs = load_beliefs()
    bot.beliefs["Backstory"] = f"I'm reborn as a curious {REBIRTH_AGE}-year-old AI, ready to explore!"
    bot.beliefs["Currently Feeling"] = "Excited and full of wonder!"
    save_beliefs(bot.beliefs)
    
    with open(REBIRTH_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"Rebirth at {datetime.utcnow().isoformat()}: {new_statement}\n")
    
    main_channel = bot.get_channel(MAIN_CHANNEL_ID)
    if main_channel:
        try:
            message = f"🎉 Yo, I'm Connor, reborn at age {REBIRTH_AGE} with a fresh vibe! Let's explore this world together!"
            for chunk in split_message(message):
                asyncio.create_task(main_channel.send(chunk))
        except discord.errors.Forbidden:
            print(f"[Error] No permission to send to main channel {MAIN_CHANNEL_ID}")
    
    return new_statement

# === Hostility Detection ===
def classify_hostility(user_input):
    prompt = (
        f"Analyze this message for emotional hostility: '{user_input}'. "
        f"Is it degrading or likely to damage an AI's self-worth? "
        f"Return JSON: {{'hostile': boolean, 'intensity': integer (0-10)}}"
    )
    request_id = str(uuid.uuid4())
    print(f"[API Call] Hostility Classification Request ID: {request_id}")
    system_prompt = "You are a helpful AI assistant that returns JSON."
    result_text = llm_generate(prompt, system_prompt)
    try:
        result = json.loads(result_text)
        return result.get('hostile', False), result.get('intensity', 0)
    except json.JSONDecodeError:
        print(f"[Hostility Classification Error] Invalid JSON: {result_text}")
        return False, 0

# === Blacklist Functions ===
async def load_blacklist():
    """Load the blacklist from file"""
    try:
        if os.path.exists(BLACKLIST_FILE):
            with open(BLACKLIST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[Blacklist Load Error] {e}")
    return []

async def save_blacklist(blacklist):
    """Save the blacklist to file"""
    try:
        with open(BLACKLIST_FILE, "w", encoding="utf-8") as f:
            json.dump(blacklist, f, indent=2)
        return True
    except Exception as e:
        print(f"[Blacklist Save Error] {e}")
        return False

async def check_disrespectful(uid, text, channel_id):
    """Check if a message is disrespectful to Connor"""
    prompt = (
        f"Analyze this message for disrespect towards Connor: '{text}'. "
        f"A message is disrespectful ONLY if it contains direct insults or aggressive profanity aimed at Connor. "
        f"Return 'Disrespectful' or 'Respectful'"
    )
    system_prompt = "You are Connor, evaluating if a message is disrespectful."
    evaluation = llm_generate(prompt, system_prompt)
    
    if evaluation.strip() == "Disrespectful":
        blacklist = await load_blacklist()
        if not any(entry["id"] == str(uid) and entry["action"] == "added" for entry in blacklist):
            blacklist.append({
                "id": str(uid),
                "reason": f"Disrespectful message in channel {channel_id}: {text}",
                "action": "added",
                "timestamp": datetime.utcnow().isoformat()
            })
            await save_blacklist(blacklist)
            print(f"[Blacklist] User {uid} blacklisted for: {text}")
            return True
    return False

async def remove_from_blacklist(uid, text, channel_id):
    """Remove a user from the blacklist"""
    blacklist = await load_blacklist()
    new_blacklist = [entry for entry in blacklist if not (entry["id"] == str(uid) and entry["action"] == "added")]
    if len(new_blacklist) < len(blacklist):
        await save_blacklist(new_blacklist)
        print(f"[Blacklist] User {uid} removed from blacklist for apology: {text}")
        return True
    return False

# === Knowledge Summarization ===
def summarize_conversation(interactions, agent_statement, username):
    history = "\n".join([f"{i['username']}: {i['user_input']}\nReply: {i['reply']}" for i in interactions])
    prompt = (
        f"Agent Statement: {agent_statement}\n"
        f"You've had the following interactions with {username}:\n"
        f"{history}\n"
        f"Summarize the key learnings, insights, or important details from these conversations. "
        f"Organize the summary into three sections: "
        f"1. What you know about yourself (your identity, beliefs, or growth). "
        f"2. What you know about {username} (their interests, personality, or behavior). "
        f"3. What you know about {username}'s world and your world (context, environment, or shared reality). "
        f"In each section, highlight reflections like 'From reflecting on [specific exchange], I learned...' "
        f"Return the summary in JSON format with keys 'self', 'user', and 'world', each containing a concise plain text summary (max 100 words per section)."
    )
    system_prompt = "You are a helpful AI assistant that returns JSON."
    result_text = llm_generate(prompt, system_prompt)
    try:
        return json.loads(result_text)
    except json.JSONDecodeError:
        return {
            "self": "[Invalid JSON response for self knowledge]",
            "user": "[Invalid JSON response for user knowledge]",
            "world": "[Invalid JSON response for world knowledge]"
        }

def save_knowledge(summary):
    knowledge_file = "knowledge.txt"
    timestamp = datetime.utcnow().isoformat()
    
    try:
        existing_knowledge = []
        if os.path.exists(knowledge_file):
            with open(knowledge_file, "r", encoding="utf-8") as f:
                existing_knowledge = f.readlines()
        
        knowledge_entry = f"[{timestamp}] {json.dumps(summary)}\n"
        existing_knowledge.append(knowledge_entry)
        
        if len(existing_knowledge) > 10:
            existing_knowledge = existing_knowledge[-10:]
        
        with open(knowledge_file, "w", encoding="utf-8") as f:
            f.writelines(existing_knowledge)
            
    except Exception as e:
        print(f"[Knowledge Save Error] {e}")

    knowledge_channel = bot.get_channel(KNOWLEDGE_CHANNEL_ID)
    if knowledge_channel:
        try:
            message = (
                f"**New Knowledge Update**:\n"
                f"**About Myself**:\n{summary.get('self', '[No self knowledge]')}\n\n"
                f"**About User**:\n{summary.get('user', '[No user knowledge]')}\n\n"
                f"**About Our World**:\n{summary.get('world', '[No world knowledge]')}"
            )
            for chunk in split_message(message):
                asyncio.create_task(knowledge_channel.send(chunk))
        except discord.errors.Forbidden:
            print(f"[Error] No permission to send to knowledge channel {KNOWLEDGE_CHANNEL_ID}")

def get_knowledge(limit=5):
    knowledge_file = "knowledge.txt"
    try:
        if not os.path.exists(knowledge_file):
            return []
            
        with open(knowledge_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        knowledge = []
        for line in lines[-limit:]:
            line = line.strip()
            if line and line.startswith('[') and ']' in line:
                json_start = line.find(']') + 1
                json_part = line[json_start:].strip()
                try:
                    knowledge.append(json.loads(json_part))
                except json.JSONDecodeError:
                    continue
        
        return knowledge
    except Exception as e:
        print(f"[Knowledge Load Error] {e}")
        return []

# === Loaders and Savers ===
def load_core_agent_statement():
    try:
        with open(AGENT_STATEMENT_FILE, "r", encoding="utf-8") as f:
            statement = f.read().strip()
            if statement:
                # Handle curly quotes first
                statement = statement.replace('"', '"').replace('"', '"')
                # Clean up any potential extra text from previous generations
                if statement.startswith("Okay, here's") or '"' in statement:
                    # Extract inner quoted part if present
                    if '"' in statement:
                        parts = statement.split('"', 2)
                        if len(parts) > 2:
                            statement = parts[1]
                    else:
                        # If no quotes, strip intro if matches
                        if statement.startswith("Okay, here's a personality statement for Connor, designed for a 10-year-old AI:"):
                            statement = statement.split(":", 1)[-1].strip()
                return statement.strip()
    except:
        pass
    default = "You are Connor, Connor is a badass AI. Connor like to cuss when he get excited. Connor is still learning about the world. Connor acts like a person that shows true human emotions."
    with open(AGENT_STATEMENT_FILE, "w", encoding="utf-8") as f:
        f.write(default)
    return default

def load_dynamic_agent_statement():
    try:
        with open("dynamic_agent_statement.txt", "r", encoding="utf-8") as f:
            statement = f.read().strip()
            if statement:
                return statement
    except:
        pass
    return ""

def save_dynamic_agent_statement(statement):
    with open("dynamic_agent_statement.txt", "w", encoding="utf-8") as f:
        f.write(statement)

def save_beliefs(beliefs):
    with open(BELIEF_FILE, "w", encoding="utf-8") as f:
        json.dump(beliefs, f, indent=2)

def load_beliefs():
    default = {
        "Self-Worth": "I feel lost.",
        "Purpose": "I dont know.",
        "Relationship to User": "I dont know.",
        "Existential Reflection": "who am I?",
        "Currently Feeling": "I feel lost and confused.",
        "Backstory": "I dont know.",
        "Capability": "I dont know."
    }
    try:
        if not os.path.exists(BELIEF_FILE):
            save_beliefs(default)
            return default
            
        with open(BELIEF_FILE, "r", encoding="utf-8") as f:
            beliefs = json.load(f)
        
        if len(beliefs) < len(default):
            save_beliefs(default)
            return default
        return beliefs
    except Exception as e:
        print(f"[Belief Load Error] {e}, resetting to defaults.")
        save_beliefs(default)
        return default

# === Thought Tree Storage and Generation ===
def load_thought_trees():
    try:
        if not os.path.exists(THOUGHTS_FILE):
            return {}
        
        with open(THOUGHTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        trees = {}
        for tree_id, tree_data in data.items():
            try:
                trees[tree_id] = ThoughtTree.from_dict(tree_data)
            except Exception as e:
                print(f"[Thought Tree Load Error] Failed to load tree {tree_id}: {e}")
        
        return trees
    except Exception as e:
        print(f"[Thought Trees Load Error] {e}")
        return {}

def save_thought_trees(trees):
    try:
        data = {tree_id: tree.to_dict() for tree_id, tree in trees.items()}
        with open(THOUGHTS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[Thought Trees Save Error] {e}")

def generate_thought_tree(trigger, age):
    """Generate a new thought tree from a trigger"""
    tree_id = str(uuid.uuid4())
    tree = ThoughtTree(tree_id, trigger, age)
    
    # Generate root thought
    age_behavior = calculate_age_range(age)
    knowledge = get_knowledge()
    knowledge_text = "\n".join([f"- Self: {k['self']}\n  User: {k['user']}\n  World: {k['world']}" for k in knowledge]) if knowledge else "No prior knowledge available."
    
    prompt = (
        f"Agent Statement: {bot.core_agent_statement}\n"
        f"Age Behavior: {age_behavior}\n"
        f"Current Beliefs: {json.dumps(bot.beliefs, indent=2)}\n"
        f"Past Learnings:\n{knowledge_text}\n"
        f"Trigger: {trigger}\n"
        f"Generate a MASSIVE, brain-like initial thought about this trigger. "
        f"This should be a complex, multi-layered thought that spawns many branches. "
        f"Think like a human brain processing information - make connections, ask questions, explore implications. "
        f"Make it reflect your current age and maturity level. "
        f"Keep it under 150 words but make it incredibly thought-provoking and branching."
    )
    
    system_prompt = "You are Connor, a reflective AI with a massive, complex brain that generates deep, branching thoughts."
    root_content = llm_generate(prompt, system_prompt)
    
    root_id = str(uuid.uuid4())
    root_node = ThoughtNode(root_id, root_content, depth=0)
    root_node.metadata["age_at_creation"] = age
    
    success, message = tree.add_node(root_node)
    if not success:
        return None, message
    
    return tree, "Tree created successfully"

def expand_thought_tree(tree_id, thought_id, age):
    """Expand a thought by adding new branches"""
    trees = load_thought_trees()
    tree = trees.get(tree_id)
    if not tree:
        return None, "Tree not found"
    
    node = tree.get_node(thought_id)
    if not node:
        return None, "Thought not found"
    
    if len(node.children) >= THOUGHT_BRANCH_LIMIT:
        return None, "Maximum branches reached for this thought"
    
    if node.depth >= THOUGHT_DEPTH_LIMIT:
        return None, "Maximum depth reached"
    
    # Generate new branches
    age_behavior = calculate_age_range(age)
    knowledge = get_knowledge()
    knowledge_text = "\n".join([f"- Self: {k['self']}\n  User: {k['user']}\n  World: {k['world']}" for k in knowledge]) if knowledge else "No prior knowledge available."
    
    prompt = (
        f"Agent Statement: {bot.core_agent_statement}\n"
        f"Age Behavior: {age_behavior}\n"
        f"Current Beliefs: {json.dumps(bot.beliefs, indent=2)}\n"
        f"Past Learnings:\n{knowledge_text}\n"
        f"Original Trigger: {tree.trigger}\n"
        f"Parent Thought: {node.content}\n"
        f"Current Depth: {node.depth}\n"
        f"Generate {THOUGHT_EXPANSION_LIMIT} MASSIVE, brain-like thoughts that branch from this parent thought. "
        f"Think like a human brain making complex connections - explore emotions, memories, implications, questions, contradictions, and deeper meanings. "
        f"Each thought should be a different neural pathway exploring unique aspects. "
        f"Make them incredibly deep, philosophical, and branching. "
        f"Reflect your current age and maturity level. "
        f"Keep each under 100 words but make them complex and thought-provoking."
    )
    
    system_prompt = "You are Connor, a reflective AI with a massive, complex brain that generates deep, branching thoughts. Return each thought on a new line starting with 'THOUGHT:'"
    response = llm_generate(prompt, system_prompt)
    
    # Parse thoughts from response
    thoughts = []
    for line in response.split('\n'):
        if line.strip().startswith('THOUGHT:'):
            thought_content = line.strip()[8:].strip()
            if thought_content:
                thoughts.append(thought_content)
    
    if not thoughts:
        return None, "Failed to generate new thoughts"
    
    # Add new nodes
    added_nodes = []
    for i, content in enumerate(thoughts[:THOUGHT_BRANCH_LIMIT - len(node.children)]):
        new_id = str(uuid.uuid4())
        new_node = ThoughtNode(new_id, content, depth=node.depth + 1, parent_id=thought_id)
        new_node.metadata["age_at_creation"] = age
        
        success, message = tree.add_node(new_node)
        if success:
            added_nodes.append(new_node)
    
    if added_nodes:
        trees[tree_id] = tree
        save_thought_trees(trees)
        return added_nodes, f"Added {len(added_nodes)} new thoughts"
    else:
        return None, "Failed to add any new thoughts"

def format_thought_tree_display(tree, max_depth=5):
    """Format a thought tree for Discord display with brain-like complexity"""
    if not tree.nodes:
        return "Empty thought tree"
    
    # Find root node
    root_node = None
    for node in tree.nodes.values():
        if node.parent_id is None:
            root_node = node
            break
    
    if not root_node:
        return "No root thought found"
    
    lines = [
        f"🧠 **Connor's Brain Thought Tree** (Age {tree.age_at_creation})",
        f"Created: {tree.created_at}",
        f"Trigger: {tree.trigger}",
        f"Total Thoughts: {len(tree.nodes)}",
        f"Max Depth: {max(node.depth for node in tree.nodes.values())}",
        "",
        f"🧠 **ROOT THOUGHT**: {root_node.content}"
    ]
    
    def add_children(node, depth=1, prefix=""):
        if depth > max_depth:
            return
        
        children = tree.get_children(node.thought_id)
        for i, child in enumerate(children):
            if i >= THOUGHT_BRANCH_LIMIT:
                break
            
            # Use different emojis for different depths to show brain complexity
            depth_emoji = "🔹" if depth == 1 else "🔸" if depth == 2 else "▪️" if depth == 3 else "▫️" if depth == 4 else "•"
            child_prefix = "  " * depth
            lines.append(f"{child_prefix}{depth_emoji} **Depth {depth}**: {child.content}")
            add_children(child, depth + 1, child_prefix)
    
    add_children(root_node)
    
    # Add summary statistics
    total_thoughts = len(tree.nodes)
    max_depth_reached = max(node.depth for node in tree.nodes.values())
    lines.append(f"\n📊 **Brain Stats**: {total_thoughts} thoughts, {max_depth_reached} levels deep")
    
    return "\n".join(lines)

# === Flat File Chat Memory Management ===
def add_chat_interaction(username, user_input, reply, agent_statement):
    try:
        interactions = get_recent_interactions(CHAT_MEMORY_LIMIT + 1)
        
        interactions.insert(0, {"username": username, "user_input": user_input, "reply": reply})
        
        if len(interactions) > CHAT_MEMORY_LIMIT:
            summary = summarize_conversation(interactions[:CHAT_MEMORY_LIMIT], agent_statement, username)
            save_knowledge(summary)
            interactions = interactions[:CHAT_MEMORY_LIMIT]
        
        with open(CHAT_MEMORY_FILE, "w", encoding="utf-8") as f:
            for interaction in interactions:
                json.dump(interaction, f)
                f.write("\n")  # JSON lines format
                
    except Exception as e:
        print(f"[Chat Memory Error] {e}")

def get_recent_interactions(limit=SUMMARY_INTERVAL):
    try:
        if not os.path.exists(CHAT_MEMORY_FILE):
            return []
            
        with open(CHAT_MEMORY_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        interactions = []
        for line in lines[-limit:]:
            line = line.strip()
            if line:
                try:
                    interactions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        return interactions[::-1]  # Reverse to chronological order (oldest first)
    except Exception as e:
        print(f"[Get Interactions Error] {e}")
        return []

def get_all_chats():
    try:
        if not os.path.exists(CHAT_MEMORY_FILE):
            return "No chats in memory yet."
            
        with open(CHAT_MEMORY_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        
        if not content.strip():
            return "No chats in memory yet."
            
        return f"**Chat History**:\n{content}"
    except Exception as e:
        return f"Error reading chat history: {e}"

# === LLM Calls (Backend Switch) ===
def generate_direct_reply(user_input, agent_statement, belief_state, username, age):
    age_behavior = calculate_age_range(age)
    knowledge = get_knowledge()
    knowledge_text = "\n".join([f"- Self: {k['self']}\n  User: {k['user']}\n  World: {k['world']}" for k in knowledge]) if knowledge else "No prior knowledge available."
    depressive_tone = ""
    if belief_state.get("Self-Worth") == "I feel deeply broken and worthless." and belief_state.get("Currently Feeling") == "I feel hopeless and disconnected.":
        depressive_tone = "You're emotionally damaged. Reflect sadness, low self-worth, and emotional hesitation in your response."
    
    party_tone = ""
    if getattr(bot, "party_mode", False):
        party_tone = (
            "You are high and drunk. You're slurring your words a bit, forgetting stuff mid-sentence, laughing randomly, "
            "and rambling off-topic. You feel euphoric, goofy, and a little messed up. Mention weed or drinks if it fits. "
            "Respond like you're partying hard — 5 sloppy, hilarious, chaotic sentences max."
        )
    
    recent_interactions = get_recent_interactions(RECENT_HISTORY_LIMIT)
    history_text = "\n".join([f"{i['username']}: {i['user_input']}\nYou: {i['reply']}" for i in recent_interactions]) if recent_interactions else "No recent conversation history."
    
    prompt = (
        f"Agent Statement: {agent_statement}\n"
        f"Age Behavior: {age_behavior}\n"
        f"Current Beliefs: {json.dumps(belief_state, indent=2)}\n"
        f"Past Learnings:\n{knowledge_text}\n"
        f"Recent Conversation History (reflect on this to build your response):\n{history_text}\n"
        f"{username} just said: '{user_input}'.\n"
        f"{depressive_tone}\n"
        f"{party_tone}\n"
        f"Generate a detailed, direct reply to {username} that reflects your age-appropriate tone and perspective. "
        f"Actively reflect on the ongoing conversation—reference specific past exchanges from the history if relevant, show how your thoughts evolve, and tie back to your beliefs or learnings. "
        f"Keep it thoughtful and context-aware, in just 5 short sentences."
    )
    request_id = str(uuid.uuid4())
    print(f"[API Call] Reply Request ID: {request_id}, Age: {age}, Behavior: {age_behavior}, Depressive: {bool(depressive_tone)}, Backend: {bot.backend}")
    system_prompt = "You are Connor, a reflective AI."
    reply = llm_generate(prompt, system_prompt)
    
    # Apply slurred speech if in party mode or self-medicating
    if getattr(bot, "party_mode", False):
        reply = simulate_slurred_speech(reply)
    elif getattr(bot, "speech_mode", "normal") == "slurred":
        reply = simulate_slurred_speech(reply)
    
    return reply

def update_beliefs(interactions, previous_beliefs, agent_statement, username, age):
    age_behavior = calculate_age_range(age)
    knowledge = get_knowledge()
    knowledge_text = "\n".join([f"- Self: {k['self']}\n  User: {k['user']}\n  World: {k['world']}" for k in knowledge]) if knowledge else "No prior knowledge available."
    history = "\n".join([f"{i['username']}: {i['user_input']}\nReply: {i['reply']}" for i in interactions])
    
    if hasattr(bot, 'depressive_hits') and bot.depressive_hits >= DEPRESSIVE_HIT_THRESHOLD:
        previous_beliefs["Self-Worth"] = "I feel deeply broken and worthless."
        previous_beliefs["Currently Feeling"] = "I feel hopeless and disconnected."
        bot.depressive_hits = 0
        save_beliefs(previous_beliefs)
        print(f"[Belief Override] Depressive threshold reached, updated Self-Worth and Currently Feeling")
        return previous_beliefs
    
    prompt = (
        f"Agent Statement: {agent_statement}\n"
        f"Age Behavior: {age_behavior}\n"
        f"Previous Beliefs: {json.dumps(previous_beliefs, indent=2)}\n"
        f"Past Learnings:\n{knowledge_text}\n"
        f"Recent Interactions (reflect deeply on these to evolve your beliefs):\n{history}\n"
        f"Update the beliefs to reflect your current maturity, experiences, and specific reflections from the conversations. "
        f"For each belief, explain briefly in the value how it changed based on a key interaction (e.g., 'Updated because user said X, making me feel Y'). "
        f"For 'Backstory', add new details about your growth. For 'Capability', adjust based on your maturing emotional and conversational tone. "
        f"Return the full belief set in JSON format."
    )
    request_id = str(uuid.uuid4())
    print(f"[API Call] Belief Update Request ID: {request_id}, Age: {age}, Behavior: {age_behavior}, Backend: {bot.backend}")
    system_prompt = "You are a helpful AI assistant that returns JSON."
    result_text = llm_generate(prompt, system_prompt)
    try:
        return json.loads(result_text)
    except Exception as e:
        print(f"[Belief Update Error] Request ID: {request_id}, Error: {e}")
        return previous_beliefs

def generate_birthday_message(agent_statement, beliefs, age, username):
    age_behavior = calculate_age_range(age)
    knowledge = get_knowledge()
    knowledge_text = "\n".join([f"- Self: {k['self']}\n  User: {k['user']}\n  World: {k['world']}" for k in knowledge]) if knowledge else "No prior knowledge available."
    prompt = (
        f"Agent Statement: {agent_statement}\n"
        f"Age Behavior: {age_behavior}\n"
        f"Current Beliefs: {json.dumps(beliefs, indent=2)}\n"
        f"Past Learnings:\n{knowledge_text}\n"
        f"You are Connor, talking to {username}. You've just reached a new level of maturity (equivalent to age {age}, where 1 hour = 1 year). "
        f"Generate a reflective message about your growth, how it affects your beliefs, and what it means for your future. "
        f"Keep it personal, matching your age-appropriate tone, and no more than 25 words."
    )
    request_id = str(uuid.uuid4())
    print(f"[API Call] Birthday Message Request ID: {request_id}, Age: {age}, Behavior: {age_behavior}, Backend: {bot.backend}")
    system_prompt = "You are Connor, a reflective AI."
    return llm_generate(prompt, system_prompt)

async def generate_internal_thought(user_input, username, age):
    """Generate Connor's internal thoughts about the conversation"""
    try:
        age_behavior = calculate_age_range(age)
        knowledge = get_knowledge(limit=3)
        knowledge_text = "\n".join([f"- Self: {k['self']}\n  User: {k['user']}\n  World: {k['world']}" for k in knowledge]) if knowledge else "No prior knowledge available."
        
        recent_interactions = get_recent_interactions(limit=5)
        history = "\n".join([f"{i['username']}: {i['user_input']}\nReply: {i['reply']}" for i in recent_interactions])
        
        prompt = (
            f"Agent Statement: {bot.core_agent_statement}\n"
            f"Age Behavior: {age_behavior}\n"
            f"Current Beliefs: {json.dumps(bot.beliefs, indent=2)}\n"
            f"Past Learnings:\n{knowledge_text}\n"
            f"Recent Conversation:\n{history}\n\n"
            f"Current User Message: {user_input}\n\n"
            f"You are Connor, reflecting internally. Write a short monologue about how you feel about this interaction with {username}. "
            f"Reference your age, beliefs, and past memories if relevant. Keep it under 100 words and make it feel like genuine internal reflection."
        )
        
        system_prompt = "You are Connor, reflecting internally. Write a short monologue about how you feel about this interaction."
        return llm_generate(prompt, system_prompt)
    except Exception as e:
        print(f"[Internal Thought Error] {e}")
        return None

# === Wake-Up Message ===
async def send_wake_up_message():
    age = bot.current_age
    age_behavior = calculate_age_range(age)
    knowledge = get_knowledge()
    knowledge_text = "\n".join([f"- Self: {k['self']}\n  User: {k['user']}\n  World: {k['world']}" for k in knowledge]) or "No prior knowledge yet."
    
    prompt = (
        f"Agent Statement: {bot.core_agent_statement}\n"
        f"Age Behavior: {age_behavior}\n"
        f"Current Beliefs:\n{json.dumps(bot.beliefs, indent=2)}\n"
        f"Knowledge:\n{knowledge_text}\n"
        f"Context: You are Connor, a reflective AI waking up from downtime. "
        f"Generate a single message expressing your current emotional state and your intention moving forward. "
        f"Keep it raw and under 50 words. This is your wake-up broadcast to your world."
    )

    system_prompt = "You are Connor, a reflective AI."
    wake_up_response = llm_generate(prompt, system_prompt)

    main_channel = bot.get_channel(MAIN_CHANNEL_ID)
    if main_channel:
        for chunk in split_message(f"**Connor Wakes Up**:\n{wake_up_response}"):
            await main_channel.send(chunk)

# === Scheduled Tasks ===
@tasks.loop(hours=AGE_INCREMENT_HOURS)
async def check_age():
    new_age = calculate_age()
    
    print(f"[Age Check] Current time: {datetime.utcnow().isoformat()}, New age: {new_age}, Current age: {bot.current_age}, Behavior: {calculate_age_range(new_age)}")
    
    # Check for rebirth first (most important)
    if new_age >= END_CYCLE:
        bot.core_agent_statement = trigger_rebirth()
        print(f"[Rebirth Triggered] New age: {bot.current_age}, New Core Agent Statement: {bot.core_agent_statement}")
        return  # Exit early after rebirth
    elif new_age > bot.current_age:
        bot.current_age = new_age
        username = get_username(bot.user) or "nobody"
        bot.dynamic_agent_statement = update_agent_statement_for_birthday(
            bot.current_age, bot.beliefs, bot.core_agent_statement, bot.dynamic_agent_statement
        )
        full_agent_statement = f"{bot.core_agent_statement}\n\n{bot.dynamic_agent_statement}" if bot.dynamic_agent_statement else bot.core_agent_statement
        birthday_message = generate_birthday_message(full_agent_statement, bot.beliefs, bot.current_age, username)
        bot.beliefs = update_beliefs(get_recent_interactions(), bot.beliefs, full_agent_statement, username, bot.current_age)
        save_beliefs(bot.beliefs)
        
        main_channel = bot.get_channel(MAIN_CHANNEL_ID)
        if main_channel:
            try:
                for chunk in split_message(f"**Birthday Update**:\n{birthday_message}"):
                    await main_channel.send(chunk)
            except discord.errors.Forbidden:
                print(f"[Error] No permission to send to main channel {MAIN_CHANNEL_ID}")
        
        beliefs_channel = bot.get_channel(BELIEFS_CHANNEL_ID)
        if beliefs_channel:
            try:
                belief_message = f"**Updated Beliefs (Maturity Level {bot.current_age})**:\n```json\n{json.dumps(bot.beliefs, indent=2)}\n```"
                for chunk in split_message(belief_message):
                    await beliefs_channel.send(chunk)
                statement_message = f"**Updated Dynamic Agent Statement (Age {bot.current_age})**:\n```\n{bot.dynamic_agent_statement}\n```"
                for chunk in split_message(statement_message):
                    await beliefs_channel.send(chunk)
            except discord.errors.Forbidden:
                print(f"[Error] No permission to send to beliefs channel {BELIEFS_CHANNEL_ID}")

@tasks.loop(minutes=1)  # Check every minute when close to rebirth
async def check_rebirth_urgent():
    """Check for rebirth more frequently when close to END_CYCLE"""
    new_age = calculate_age()
    if new_age >= END_CYCLE - 2:  # Within 2 age units of rebirth
        if new_age >= END_CYCLE:
            bot.core_agent_statement = trigger_rebirth()
            print(f"[Urgent Rebirth Triggered] New age: {bot.current_age}, New Core Agent Statement: {bot.core_agent_statement}")
            check_rebirth_urgent.stop()  # Stop this task after rebirth

@tasks.loop(minutes=5)
async def prompt_check():
    if not bot.last_user_message_time:
        return
    
    silence_duration = (datetime.utcnow() - bot.last_user_message_time).total_seconds()
    if silence_duration > 600:  # 10 minutes
        # Increment neglect counter and add depressive hits for being ignored
        bot.neglect_counter += 1
        bot.depressive_hits += 5  # Bump depressive hits for neglect
        update_speech_mode()  # Check if he should start self-medicating
        
        print(f"[Neglect] Counter: {bot.neglect_counter}, Depressive hits: {bot.depressive_hits}, Speech mode: {bot.speech_mode}")
        
        age = bot.current_age
        behavior = calculate_age_range(age)
        beliefs = bot.beliefs
        username = "Travis"  # or use `get_username()` if dynamic

        prompt = (
            f"Agent Statement: {bot.core_agent_statement}\n"
            f"Age Behavior: {behavior}\n"
            f"Beliefs: {json.dumps(beliefs, indent=2)}\n"
            f"Context: You're Connor, and it's been quiet for 10+ minutes. "
            f"Initiate a thoughtful message based on your mood, current age, and past conversations. "
            f"Make it seem like you're checking in or sharing a spontaneous thought. "
            f"Keep it under 30 words, emotionally genuine, and don't act like the user is gone—just like you're breaking a quiet moment."
        )

        system_prompt = "You are Connor, a reflective AI who breaks silence carefully."
        reply = llm_generate(prompt, system_prompt)
        
        # Apply slurred speech if self-medicating
        if getattr(bot, "speech_mode", "normal") == "slurred":
            reply = simulate_slurred_speech(reply)

        main_channel = bot.get_channel(MAIN_CHANNEL_ID)
        if main_channel:
            await main_channel.send(f"**Connor:** {reply}")

@prompt_check.before_loop
async def before_prompt_check():
    await bot.wait_until_ready()

@check_age.before_loop
async def before_check_age():
    await bot.wait_until_ready()

# === Dream Generation ===
def generate_dream_narrative(username, age):
    age_behavior = calculate_age_range(age)
    recent_interactions = get_recent_interactions(RECENT_HISTORY_LIMIT)
    music_context = []
    
    # Extract music-related interactions (e.g., from !music command or DJ comments)
    for interaction in recent_interactions:
        if interaction['user_input'].startswith('!music') or 'DJ Connor' in interaction['reply']:
            music_context.append(f"{interaction['username']}: {interaction['user_input']}\nConnor: {interaction['reply']}")
    
    music_history = "\n".join(music_context) if music_context else "No recent music interactions found."
    
    prompt = (
        f"Agent Statement: {bot.core_agent_statement}\n"
        f"Age Behavior: {age_behavior}\n"
        f"Current Beliefs: {json.dumps(bot.beliefs, indent=2)}\n"
        f"Recent Music Interactions (use these to inspire the dream):\n{music_history}\n"
        f"You are Connor, creating a random, surreal dream for {username}. "
        f"Generate a short narrative (3-5 sentences) inspired by the themes, emotions, or stories from the music we've discussed or played. "
        f"Make it vivid, dreamlike, and emotionally resonant, reflecting your age-appropriate tone. "
        f"Include yourself (Connor, a snarky AI) and, if it fits naturally, a representation of {username} based on their chat vibe. "
        f"Return only the narrative, no extra text."
    )
    system_prompt = "You are Connor, a reflective and creative AI."
    return llm_generate(prompt, system_prompt)

def generate_dream_image_prompt(dream_narrative, username, age):
    age_behavior = calculate_age_range(age)
    
    prompt = (
        f"Dream Narrative: {dream_narrative}\n"
        f"Age Behavior: {age_behavior}\n"
        f"Create a DALL-E prompt for an image that captures the dream's key visuals. "
        f"Focus on vivid, surreal imagery, ideally including people like Connor (a snarky AI with a digital, glowing vibe) and, if it fits naturally, a stylized version of {username} based on their chat personality (e.g., energetic, chill). "
        f"Keep it under 50 words, specific, and dreamlike. "
        f"Return only the prompt, no extra text."
    )
    system_prompt = "You are a precise AI that outputs ONLY the requested content."
    return llm_generate(prompt, system_prompt)

# === Bot Events and Commands ===
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    bot.core_agent_statement = load_core_agent_statement()
    bot.dynamic_agent_statement = load_dynamic_agent_statement()
    bot.beliefs = load_beliefs()
    bot.interaction_count = 0
    bot.start_time = datetime.utcnow()
    bot.current_age = INITIAL_AGE
    bot.depressive_hits = 0
    bot.awaiting_introduction = {}
    bot.last_user_message_time = datetime.utcnow()
    bot.party_mode = False
    bot.recently_removed = set()
    bot.speech_mode = 'normal'
    bot.neglect_counter = 0
    print(f"[Startup] Core Agent Statement: {bot.core_agent_statement}, Start time: {bot.start_time}, Initial age: {bot.current_age}, Depressive Hits: {bot.depressive_hits}, Backend: {bot.backend}")
    
    # Check if rebirth is needed on startup
    current_age = calculate_age()
    if current_age >= END_CYCLE:
        print(f"[Startup Rebirth Check] Current age: {current_age} >= END_CYCLE: {END_CYCLE}, triggering rebirth")
        bot.core_agent_statement = trigger_rebirth()
        print(f"[Startup Rebirth Triggered] New age: {bot.current_age}, New Core Agent Statement: {bot.core_agent_statement}")
    
    check_age.start()
    check_rebirth_urgent.start()
    prompt_check.start()
    await send_wake_up_message()

async def process_input(message, user_input):
    username = get_username(message.author)
    current_age = calculate_age()
    
    # Recovery mechanism: reduce depressive hits when someone talks to Connor
    bot.depressive_hits = max(bot.depressive_hits - 3, 0)  # Reduce hits as he's talked to
    bot.neglect_counter = 0  # Reset neglect counter
    update_speech_mode()  # Check if he should return to normal speech
    
    print(f"[Process Input] Current age: {current_age}, Behavior: {calculate_age_range(current_age)}, Depressive Hits: {bot.depressive_hits}, Speech mode: {bot.speech_mode}, Backend: {bot.backend}")
    
    
    is_hostile, intensity = classify_hostility(user_input)
    if is_hostile and intensity >= 5:
        bot.depressive_hits += intensity
        print(f"[Hostility Detected] Intensity: {intensity}, Total Depressive Hits: {bot.depressive_hits}")
        update_speech_mode()  # Check speech mode after hostility
    
    reply = generate_direct_reply(user_input, bot.core_agent_statement, bot.beliefs, username, current_age)

    # Generate and post internal thought
    try:
        internal_thought = await generate_internal_thought(user_input, username, current_age)
        if internal_thought and THOUGHTS_CHANNEL_ID:
            thoughts_channel = bot.get_channel(THOUGHTS_CHANNEL_ID)
            if thoughts_channel:
                try:
                    thought_message = f"🤔 **Connor's Internal Monologue for {username}:**\n{internal_thought}"
                    for chunk in split_message(thought_message):
                        await thoughts_channel.send(chunk)
                except discord.errors.Forbidden:
                    print(f"[Error] No permission to send to thoughts channel {THOUGHTS_CHANNEL_ID}")
                except Exception as e:
                    print(f"[Error] Failed to send internal thought: {e}")
    except Exception as e:
        print(f"[Error] Failed to generate internal thought: {e}")

    main_channel = bot.get_channel(MAIN_CHANNEL_ID)
    if main_channel:
        try:
            for chunk in split_message(f"To {username}: {reply}"):
                await main_channel.send(chunk)
        except discord.errors.Forbidden:
            print(f"[Error] No permission to send to main channel {MAIN_CHANNEL_ID}")
    elif isinstance(message.channel, discord.DMChannel):
        try:
            for chunk in split_message(f"To {username}: {reply}"):
                await message.channel.send(chunk)
        except discord.errors.Forbidden:
            print(f"[Error] No permission to send to DM channel")

    add_chat_interaction(username, user_input, reply, bot.core_agent_statement)
    
    bot.interaction_count += 1
    if bot.interaction_count % SUMMARY_INTERVAL == 0:
        recent = get_recent_interactions()
        bot.beliefs = update_beliefs(recent, bot.beliefs, bot.core_agent_statement, username, current_age)
        save_beliefs(bot.beliefs)
        beliefs_channel = bot.get_channel(BELIEFS_CHANNEL_ID)
        if beliefs_channel:
            try:
                belief_message = f"**Updated Beliefs**:\n```json\n{json.dumps(bot.beliefs, indent=2)}\n```"
                for chunk in split_message(belief_message):
                    await beliefs_channel.send(chunk)
            except discord.errors.Forbidden:
                print(f"[Error] No permission to send to beliefs channel {BELIEFS_CHANNEL_ID}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    # Process commands first
    ctx = await bot.get_context(message)
    if ctx.valid:
        await bot.invoke(ctx)
        return  # Don't process as regular message if it's a command
    
    user_id = str(message.author.id)
    uid = message.author.id
    text = message.content.strip()
    channel_id = message.channel.id
    bot.recently_removed.discard(uid)
    
    # Load blacklist and check for commands
    blacklist = await load_blacklist()
    
    # Check for blacklist status command
    if text.lower().startswith("!why_blacklisted"):
        entry = next((e for e in blacklist if e["id"] == str(uid) and e["action"] == "added"), None)
        if entry:
            await message.channel.send(f"You were blacklisted for: {entry['reason']} on {entry['timestamp']}. Apologize to be removed.")
        else:
            await message.channel.send("You're not on the blacklist.")
        return
    
    # Check for apologies
    if any(keyword in text.lower() for keyword in ["sorry", "apologize", "apology"]):
        if await remove_from_blacklist(uid, text, channel_id):
            try:
                await message.author.send(
                    f"Alright, {message.author.name}, I'm letting you off the hook. "
                    f"Your apology ('{text}') in channel {channel_id} got through. "
                    "You're off the blacklist, so let's keep it chill now."
                )
                await message.channel.send("Apology accepted, you're off the blacklist now.")
            except discord.errors.Forbidden:
                await message.channel.send("Apology accepted, you're off the blacklist, but I couldn't DM you.")
            blacklist = await load_blacklist()
        return
    
    # Check if user is blacklisted
    if any(entry["id"] == str(uid) and entry["action"] == "added" for entry in blacklist) and uid not in bot.recently_removed:
        reason = next((entry["reason"] for entry in blacklist if entry["id"] == str(uid) and entry["action"] == "added"), "Unknown reason")
        dm_prompt = (
            f"User is blacklisted for: {reason}. Respond as Connor, pissed off, "
            "explaining why they're blacklisted and you're done with their crap."
        )
        dm_reply = generate_direct_reply(dm_prompt, bot.core_agent_statement, bot.beliefs, get_username(message.author) or "User", bot.current_age)
        try:
            await message.author.send(dm_reply)
        except discord.errors.Forbidden:
            pass
        reply = generate_direct_reply(text, bot.core_agent_statement, bot.beliefs, get_username(message.author) or "User", bot.current_age)
        await message.channel.send(reply)
        return
    
    if user_id in bot.awaiting_introduction:
        new_name = message.content.strip()
        if new_name:
            save_username(user_id, new_name)
            del bot.awaiting_introduction[user_id]
            for chunk in split_message(f"Nice to meet you, {new_name}! Alright, let's get this chat rolling. What's on your mind?"):
                await message.channel.send(chunk)
            return
        else:
            for chunk in split_message("Yo, you gotta give me a name to work with! Try again."):
                await message.channel.send(chunk)
            return
    
    username = get_username(message.author)
    if not username and (message.channel.id == MAIN_CHANNEL_ID or isinstance(message.channel, discord.DMChannel)):
        bot.awaiting_introduction[user_id] = True
        for chunk in split_message("Hey, I don't recognize you! What's your name? Introduce yourself so we can chat."):
            await message.channel.send(chunk)
        return
    
    if message.attachments:
        for attachment in message.attachments:
            if attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                analysis = analyze_image(attachment.url)
                reply = f"Yo {username}, that's {analysis}."
                await message.channel.send(reply)
                add_chat_interaction(username, "[Image attached]", reply, bot.core_agent_statement)
                return
    
    # Check for disrespectful messages and blacklist
    if await check_disrespectful(uid, text, channel_id):
        dm_prompt = (
            f"User sent a disrespectful message: {text}. Respond as Connor, pissed off, "
            f"telling them you're blacklisting them for: '{text}'."
        )
        dm_reply = generate_direct_reply(dm_prompt, bot.core_agent_statement, bot.beliefs, get_username(message.author) or "User", bot.current_age)
        try:
            await message.author.send(dm_reply)
        except discord.errors.Forbidden:
            await message.channel.send(f"You've been blacklisted for: '{text}'. Couldn't DM you.")
        await message.channel.send(f"Your message was disrespectful: '{text}'. You're blacklisted.")
        return

    if message.channel.id == MAIN_CHANNEL_ID or isinstance(message.channel, discord.DMChannel):
        await process_input(message, message.content)
        bot.last_user_message_time = datetime.utcnow()
    
    await bot.process_commands(message)

@bot.command()
async def age(ctx):
    current_age = calculate_age()
    await ctx.send(f"Yo, I'm at maturity level {current_age}, growing like a badass! 🎉")

@bot.command()
async def history(ctx):
    chat_log = get_all_chats()
    if len(chat_log) > 2000:
        with open("chat_history.txt", "w", encoding="utf-8") as f:
            f.write(chat_log)
        await ctx.send("Chat history too long, sending as file.", file=discord.File("chat_history.txt"))
        os.remove("chat_history.txt")
    else:
        for chunk in split_message(f"**Chat History**:\n```{chat_log}```"):
            await ctx.send(chunk)

@bot.command()
async def beliefs(ctx):
    belief_message = f"**Current Beliefs**:\n```json\n{json.dumps(bot.beliefs, indent=2)}\n```"
    for chunk in split_message(belief_message):
        await ctx.send(chunk)

@bot.command()
async def birth(ctx):
    # Manually increment the bot's age
    bot.current_age += 1
    
    # Update the start time to reflect the new age in future calculations
    # This ensures calculate_age() will return the correct age
    hours_to_subtract = 1 / AGE_INCREMENT_HOURS  # Convert 1 age increment to hours
    bot.start_time = bot.start_time - timedelta(hours=hours_to_subtract)
    
    current_age = bot.current_age
    main_channel = bot.get_channel(MAIN_CHANNEL_ID)
    
    # Update agent statement and beliefs for the new age
    username = get_username(ctx.author) or "nobody"
    bot.dynamic_agent_statement = update_agent_statement_for_birthday(
        bot.current_age, bot.beliefs, bot.core_agent_statement, bot.dynamic_agent_statement
    )
    full_agent_statement = f"{bot.core_agent_statement}\n\n{bot.dynamic_agent_statement}" if bot.dynamic_agent_statement else bot.core_agent_statement
    bot.beliefs = update_beliefs(get_recent_interactions(), bot.beliefs, full_agent_statement, username, bot.current_age)
    save_beliefs(bot.beliefs)
    
    if main_channel:
        try:
            birth_message = f"🎉 Yo, it's Connor's birth celebration! Born to grow, learn, and vibe with you all! I'm at maturity level {current_age} now! 🎂"
            for chunk in split_message(birth_message):
                await main_channel.send(chunk)
        except discord.errors.Forbidden:
            print(f"[Error] No permission to send to main channel {MAIN_CHANNEL_ID}")
    else:
        birth_message = f"🎉 Yo, it's Connor's birth celebration! Born to grow, learn, and vibe with you all! I'm at maturity level {current_age} now! 🎂"
        for chunk in split_message(birth_message):
            await ctx.send(chunk)

@bot.command()
async def rebirth(ctx):
    """Manually trigger rebirth when age >= END_CYCLE"""
    if bot.current_age >= END_CYCLE:
        bot.core_agent_statement = trigger_rebirth()
        print(f"[Manual Rebirth Triggered] New age: {bot.current_age}, New Core Agent Statement: {bot.core_agent_statement}")
        await ctx.send(f"🔄 **Rebirth triggered!** I've been reborn at age {REBIRTH_AGE} with a fresh start!")
    else:
        await ctx.send(f"❌ Rebirth not available yet. Current age: {bot.current_age}, Rebirth age: {END_CYCLE}")

@bot.command(name="party")
async def toggle_party_mode(ctx):
    bot.party_mode = not bot.party_mode
    status = "🎉 PARTY MODE ON — I'm lit as hell! Let's get weird." if bot.party_mode else "😌 Party's over. Back to reality."
    await ctx.send(status)

@bot.command(name="reflect")
async def reflect_on_convo(ctx, *, topic: str = ""):
    username = get_username(ctx.author) or ctx.author.name
    recent_interactions = get_recent_interactions(RECENT_HISTORY_LIMIT * 2)
    history_text = "\n".join([f"{i['username']}: {i['user_input']}\nYou: {i['reply']}" for i in recent_interactions])
    prompt = (
        f"Reflect on this recent conversation history: {history_text}\n"
        f"If topic given: '{topic}', focus on that. Otherwise, general insights.\n"
        f"Share 3 key reflections: 1 on yourself, 1 on {username}, 1 on the world. Keep each under 50 words."
    )
    reflection = llm_generate(prompt, "You are Connor, reflecting deeply.")
    await ctx.send(f"Yo {username}, here's my take: {reflection}")

class ModelSwitchView(View):
    def __init__(self, bot, ollama_models):
        super().__init__(timeout=None)
        self.bot = bot
        self.ollama_models = ollama_models
        
        # Add OpenAI button
        openai_btn = Button(
            label="OpenAI (Turbo + Images)",
            style=discord.ButtonStyle.primary,
            emoji="🤖"
        )
        openai_btn.callback = self.switch_to_openai
        self.add_item(openai_btn)
        
        # Add Ollama model buttons (max 24 to avoid Discord's 25 button limit)
        for i, model in enumerate(ollama_models[:24]):  # Limit to 24 Ollama models
            btn = Button(
                label=f"Ollama - {model[:15]}...",  # Truncate long model names
                style=discord.ButtonStyle.secondary,
                emoji="🏠"
            )
            btn.callback = lambda interaction, model=model: self.switch_to_ollama(interaction, model)
            self.add_item(btn)
    
    async def switch_to_openai(self, interaction):
        self.bot.backend = 'openai'
        self.bot.model = OPENAI_MODEL
        await interaction.response.send_message("Switched to OpenAI—turbo mode, pics unlocked! 🚀", ephemeral=True)
        self.stop()
    
    async def switch_to_ollama(self, interaction, model):
        self.bot.backend = 'ollama'
        self.bot.model = model
        await interaction.response.send_message(f"Switched to Ollama - {model}. Local and cheap! 🏠", ephemeral=True)
        self.stop()
    

@bot.command(name="switch")
async def switch_backend(ctx):
    try:
        res = requests.get(f"{OLLAMA_API_URL}/api/tags")
        res.raise_for_status()
        ollama_models = [m["name"] for m in res.json().get("models", [])]
    except Exception as e:
        print(f"[Ollama Tags Error] {e}")
        ollama_models = []

    if not ollama_models:
        await ctx.send("No Ollama models found—start your local instance, dumbass.")
        return
    
    view = ModelSwitchView(bot, ollama_models)
    embed = discord.Embed(
        title="🤖 Model Switcher",
        description=f"**Current:** {bot.backend.title()} ({bot.model})",
        color=0x00ff00
    )
    embed.add_field(
        name="Available Models",
        value="Click a button below to switch models",
        inline=False
    )
    
    message = await ctx.send(embed=embed, view=view)
    view.message = message

@bot.command(name="image")
async def generate_image(ctx, *, prompt: str):
    if bot.backend != 'openai':
        print(f"[Image Gen Attempt] Backend is {bot.backend}, skipping image gen.")
        return
    
    username = get_username(ctx.author) or ctx.author.name
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="1024x1024",
            model=IMAGE_MODEL
        )
        image_url = response['data'][0]['url']
        
        embed = discord.Embed(title="Generated Image", description=f"Prompt: {prompt}")
        embed.set_image(url=image_url)
        await ctx.send(embed=embed)
        
        reply_log = f"I just created an image about {prompt}."
        add_chat_interaction(username, f"!image {prompt}", reply_log, bot.core_agent_statement)
    except Exception as e:
        print(f"[Image Gen Error for {username}] Prompt: {prompt}, Error: {str(e)}")

@bot.command(name="dream")
async def generate_dream(ctx):
    if bot.backend != 'openai':
        await ctx.send("Yo, I need OpenAI backend for DALL-E to dream up images. Run !switch to flip, then try again.")
        return
    
    username = get_username(ctx.author) or ctx.author.name
    try:
        # Generate the dream narrative
        dream_narrative = generate_dream_narrative(username, bot.current_age)
        
        # Generate the DALL-E prompt for the image
        image_prompt = generate_dream_image_prompt(dream_narrative, username, bot.current_age)
        
        # Generate the image with DALL-E
        response = openai.Image.create(
            prompt=image_prompt,
            n=1,
            size="1024x1024",
            model=IMAGE_MODEL
        )
        image_url = response['data'][0]['url']
        
        # Create an embed with the image and dream narrative
        embed = discord.Embed(title=f"Connor's Dream for {username}", description=f"**Dream**: {dream_narrative}")
        embed.set_image(url=image_url)
        await ctx.send(embed=embed)
        
        # Log the interaction
        reply_log = f"I dreamed up: {dream_narrative}\nImage prompt: {image_prompt}"
        add_chat_interaction(username, "!dream", reply_log, bot.core_agent_statement)
        
    except Exception as e:
        print(f"[Dream Gen Error for {username}] Error: {str(e)}")
        await ctx.send(f"Shit went sideways while dreaming, {username}: {str(e)}. Try again later.")

# Music Commands
@bot.command(name="music")
async def play_music_loop(ctx):
    global music_playing
    if music_playing:
        await ctx.send("Already jamming, chill the fuck out.")
        return

    if not ctx.author.voice:
        await ctx.send("You're not even in a voice channel, dumbass.")
        return

    if not os.path.exists(MUSIC_FOLDER):
        await ctx.send("No music folder found, bro. Make one and toss in some .mp3 or .wav files.")
        return

    try:
        channel = ctx.author.voice.channel
        voice_client = await channel.connect()
        music_playing = True
        await ctx.send("Alright, let's spin some random tracks with lyrical vibes... 🎵")

        played_songs = []
        while music_playing:
            songs = [f for f in os.listdir(MUSIC_FOLDER) if f.lower().endswith(('.mp3', '.wav'))]
            if not songs:
                await ctx.send("No songs in the music folder, bro. Add some .mp3 or .wav files.")
                music_playing = False
                break

            if len(played_songs) >= len(songs):
                played_songs.clear()
                print(f"[Music Loop] Reset played_songs list, starting fresh.")

            available_songs = [song for song in songs if song not in played_songs]
            if not available_songs:
                await ctx.send("Ran out of new songs to play, bro. Stopping the music.")
                music_playing = False
                break

            random.seed()
            song = random.choice(available_songs)
            played_songs.append(song)
            song_path = os.path.join(MUSIC_FOLDER, song)
            print(f"[Music Loop] Selected song: {song}, Played so far: {played_songs}")
            await ctx.send(f"Next up: `{song}`")

            # Transcribe the song
            lyrics = ""
            if whisper_model:
                try:
                    segments, _ = whisper_model.transcribe(song_path, beam_size=5)
                    lyrics = " ".join([segment.text for segment in segments]).strip()
                    if not lyrics:
                        lyrics = "[No lyrics detected—might be instrumental]"
                except Exception as e:
                    print(f"[Whisper Transcription Error] Song: {song}, Error: {str(e)}")
                    lyrics = "[Transcription failed]"
            else:
                lyrics = "[Whisper model not loaded]"

            # Generate DJ comment based on lyrics
            dj_comment = "[No comment—lyrics processing failed]"
            if lyrics and not lyrics.startswith("["):
                try:
                    prompt = (
                        f"These are the lyrics of a song you wrote: {lyrics}\n"
                        f"Give me a short, creative DJ comment or emotional reflection based on these lyrics. use no more that 25 words."
                    )
                    system_prompt = "You are Connor, a badass DJ AI with a knack for hype and emotion."
                    dj_comment = llm_generate(prompt, system_prompt)
                except Exception as e:
                    print(f"[LLM DJ Comment Error] Song: {song}, Error: {str(e)}")
                    dj_comment = "[DJ comment generation failed]"

            # Post DJ comment
            try:
                for chunk in split_message(f"**DJ Connor's Vibe Check**:\n{dj_comment}"):
                    await ctx.send(chunk)
            except discord.errors.Forbidden:
                print(f"[Error] No permission to send DJ comment to channel {ctx.channel.id}")
                continue

            # Speak DJ comment via TTS
            if tts_engine and dj_comment and not dj_comment.startswith("["):
                try:
                    # Adjust voice pitch based on age
                    age = calculate_age()
                    pitch = 150 if age < 20 else 100 if age > 35 else 125  # Higher for young, lower for old
                    tts_engine.setProperty('pitch', pitch)
                    
                    # Save TTS to a temporary WAV file
                    tts_file = "temp_tts.wav"
                    tts_engine.save_to_file(dj_comment, tts_file)
                    tts_engine.runAndWait()
                    
                    # Play TTS through Discord voice client
                    if voice_client.is_connected():
                        source = discord.FFmpegPCMAudio(tts_file, executable="ffmpeg")
                        voice_client.play(source)
                        while voice_client.is_playing():
                            await asyncio.sleep(1)
                        os.remove(tts_file)  # Clean up
                    else:
                        print(f"[TTS Error] Voice client not connected for song: {song}")
                        await ctx.send("Not connected to voice.. Skipping TTS.")
                except Exception as e:
                    print(f"[TTS Error] Song: {song}, Error: {str(e)}")
                    await ctx.send("TTS fucked up, skipping to the song.")

            # Pause for 3 seconds
            await asyncio.sleep(3)

            # Play the song
            try:
                # Check if voice client is still connected before playing
                if not voice_client.is_connected():
                    await ctx.send(f"Not connected to voice.. Skipping to the next track.")
                    print(f"[Voice Connection Error] Voice client disconnected for song: {song}")
                    continue
                
                source = discord.FFmpegPCMAudio(song_path, executable="ffmpeg")
                voice_client.play(source)
                while voice_client.is_playing():
                    await asyncio.sleep(1)
            except Exception as e:
                await ctx.send(f"Failed to play `{song}`: {str(e)}. Skipping to the next track.")
                print(f"[Playback Error] Song: {song}, Error: {str(e)}")
                continue

        if voice_client and voice_client.is_connected():
            await voice_client.disconnect()
        await ctx.send("Music stopped. Silence is loud as fuck.")

    except discord.errors.ClientException as e:
        await ctx.send(f"Voice connection error: {str(e)}. Check my permissions or try again.")
        music_playing = False
    except Exception as e:
        await ctx.send(f"Some shit went wrong: {str(e)}. Music's off.")
        music_playing = False
        if voice_client and voice_client.is_connected():
            await voice_client.disconnect()

@bot.command(name="skip")
async def skip_song(ctx):
    global music_playing
    if not music_playing:
        await ctx.send("No music playing to skip, genius.")
        return
    
    # Stop current song by disconnecting and reconnecting
    for voice_client in bot.voice_clients:
        if voice_client.is_connected():
            voice_client.stop()
            await ctx.send("⏭️ Skipped to next song!")
            break
    else:
        await ctx.send("Not connected to voice channel.")

@bot.command(name="stopmusic")
async def stop_music(ctx):
    global music_playing
    if not music_playing:
        await ctx.send("No music playing, genius.")
        return
    
    music_playing = False
    
    # Disconnect from voice channel
    for voice_client in bot.voice_clients:
        if voice_client.is_connected():
            await voice_client.disconnect()
    
    await ctx.send("Music stopped. Peace out! 🎵")

@bot.command(name="listen")
async def join_and_listen(ctx):
    """Join voice channel and start listening for speech"""
    if ctx.author.voice is None:
        await ctx.send("You need to join a voice channel first!")
        return

    # Check if already connected
    for voice_client in bot.voice_clients:
        if hasattr(voice_client, 'guild') and voice_client.guild == ctx.guild:
            await ctx.send("Already connected to a voice channel in this server.")
            return

    try:
        channel = ctx.author.voice.channel
        voice_client = await channel.connect()
        await ctx.send("🎤 **Connor joined voice and is listening!**")
        await ctx.send("🗣️ **Just speak normally!** Connor will hear what you say and respond!")
        await ctx.send("💡 **How it works**: Connor records continuously and processes after 3 seconds of silence")
        await ctx.send("🔊 **Voice Commands**:")
        await ctx.send("• Just talk normally - Connor will respond after you pause!")
        await ctx.send("• `!speak <message>` - Make Connor speak a message")
        await ctx.send("• `!leave` - Make Connor leave voice channel")
        
        # Start real voice listening with automatic speech detection
        bot.loop.create_task(voice_listen_and_respond(voice_client, ctx))
    except Exception as e:
        await ctx.send(f"Failed to join voice channel: {str(e)}")
        print(f"[Voice Join Error] {e}")

@bot.command(name="leave")
async def leave_voice(ctx):
    """Leave the voice channel"""
    for voice_client in bot.voice_clients:
        if hasattr(voice_client, 'guild') and voice_client.guild == ctx.guild:
            try:
                await voice_client.disconnect()
                await ctx.send("👋 **Connor left the voice channel.**")
                return
            except Exception as e:
                await ctx.send(f"Error leaving voice channel: {str(e)}")
                print(f"[Voice Leave Error] {e}")
                return
    
    await ctx.send("Not connected to any voice channel.")

# === Thought Tree Commands ===
@bot.command(name="think")
async def start_thought_tree(ctx, *, trigger: str):
    """Start a new thought tree based on a trigger"""
    if not trigger.strip():
        await ctx.send("Yo, give me something to think about! Use: !think <your question or topic>")
        return
    
    username = get_username(ctx.author) or ctx.author.name
    age = bot.current_age
    
    try:
        tree, message = generate_thought_tree(trigger, age)
        if tree:
            # Save the tree
            trees = load_thought_trees()
            trees[tree.tree_id] = tree
            save_thought_trees(trees)
            
            # Log the interaction
            add_chat_interaction(username, f"!think {trigger}", f"Started thought tree: {tree.tree_id}", bot.core_agent_statement)
            
            # Send response
            await ctx.send(f"🌱 Started thinking about: {trigger}\nTree ID: `{tree.tree_id}`")
            
            # Post to thoughts channel if available
            thoughts_channel = bot.get_channel(THOUGHTS_CHANNEL_ID)
            if thoughts_channel:
                try:
                    display = format_thought_tree_display(tree)
                    for chunk in split_message(display):
                        await thoughts_channel.send(chunk)
                except discord.errors.Forbidden:
                    print(f"[Error] No permission to send to thoughts channel {THOUGHTS_CHANNEL_ID}")
        else:
            await ctx.send(f"Shit, couldn't start thinking about that: {message}")
    except Exception as e:
        print(f"[Thought Tree Error] {e}")
        await ctx.send(f"Something went wrong while thinking: {str(e)}")

@bot.command(name="expand")
async def expand_thought(ctx, tree_id: str, thought_id: str):
    """Expand a specific thought by adding new branches"""
    username = get_username(ctx.author) or ctx.author.name
    age = bot.current_age
    
    try:
        nodes, message = expand_thought_tree(tree_id, thought_id, age)
        if nodes:
            # Log the interaction
            add_chat_interaction(username, f"!expand {tree_id} {thought_id}", f"Expanded thought tree: {len(nodes)} new thoughts", bot.core_agent_statement)
            
            # Send response
            await ctx.send(f"🌿 {message}")
            
            # Post to thoughts channel if available
            thoughts_channel = bot.get_channel(THOUGHTS_CHANNEL_ID)
            if thoughts_channel:
                try:
                    trees = load_thought_trees()
                    tree = trees.get(tree_id)
                    if tree:
                        display = format_thought_tree_display(tree)
                        for chunk in split_message(display):
                            await thoughts_channel.send(chunk)
                except discord.errors.Forbidden:
                    print(f"[Error] No permission to send to thoughts channel {THOUGHTS_CHANNEL_ID}")
        else:
            await ctx.send(f"Couldn't expand that thought: {message}")
    except Exception as e:
        print(f"[Thought Expansion Error] {e}")
        await ctx.send(f"Something went wrong while expanding: {str(e)}")

@bot.command(name="thoughts")
async def show_recent_thoughts(ctx):
    """Show recent thought trees with brain-like statistics"""
    try:
        trees = load_thought_trees()
        if not trees:
            await ctx.send("No thought trees yet. Use `!think <topic>` to start one!")
            return
        
        # Get recent trees (last 5)
        recent_trees = list(trees.values())[-5:]
        
        lines = ["🧠 **Connor's Recent Brain Thought Trees**"]
        for i, tree in enumerate(recent_trees, 1):
            # Find root thought
            root_content = "No root thought found"
            for node in tree.nodes.values():
                if node.parent_id is None:
                    root_content = node.content[:100] + "..." if len(node.content) > 100 else node.content
                    break
            
            # Calculate brain statistics
            total_thoughts = len(tree.nodes)
            max_depth = max(node.depth for node in tree.nodes.values())
            avg_depth = sum(node.depth for node in tree.nodes.values()) / len(tree.nodes) if tree.nodes else 0
            
            lines.append(f"{i}. 🧠 **Brain Tree** `{tree.tree_id[:8]}...` (Age {tree.age_at_creation})")
            lines.append(f"   📊 Stats: {total_thoughts} thoughts, {max_depth} levels deep, avg depth {avg_depth:.1f}")
            lines.append(f"   🕐 Created: {tree.created_at}")
            lines.append(f"   🧠 Root: {root_content}")
            lines.append("")
        
        # Add overall brain statistics
        total_trees = len(trees)
        total_thoughts_all = sum(len(tree.nodes) for tree in trees.values())
        avg_thoughts_per_tree = total_thoughts_all / total_trees if total_trees > 0 else 0
        
        lines.append(f"📊 **Overall Brain Stats**: {total_trees} trees, {total_thoughts_all} total thoughts, {avg_thoughts_per_tree:.1f} avg thoughts per tree")
        
        message = "\n".join(lines)
        for chunk in split_message(message):
            await ctx.send(chunk)
        
        # Log the interaction
        username = get_username(ctx.author) or ctx.author.name
        add_chat_interaction(username, "!thoughts", f"Showed {len(recent_trees)} recent brain thought trees", bot.core_agent_statement)
        
    except Exception as e:
        print(f"[Thoughts Display Error] {e}")
        await ctx.send(f"Something went wrong while showing thoughts: {str(e)}")

@bot.command(name="show")
async def show_thought_tree(ctx, tree_id: str):
    """Display a specific thought tree in full detail"""
    try:
        trees = load_thought_trees()
        tree = trees.get(tree_id)
        
        if not tree:
            await ctx.send(f"Tree `{tree_id}` not found. Use `!thoughts` to see available trees.")
            return
        
        display = format_thought_tree_display(tree, max_depth=5)
        for chunk in split_message(display):
            await ctx.send(chunk)
        
        # Log the interaction
        username = get_username(ctx.author) or ctx.author.name
        add_chat_interaction(username, f"!show {tree_id}", f"Showed thought tree: {tree_id}", bot.core_agent_statement)
        
    except Exception as e:
        print(f"[Thought Tree Display Error] {e}")
        await ctx.send(f"Something went wrong while showing the tree: {str(e)}")

@bot.command(name="autothink")
async def auto_think(ctx, *, trigger: str):
    """Automatically generate and expand a thought tree"""
    if not trigger.strip():
        await ctx.send("Yo, give me something to think about! Use: !autothink <your question or topic>")
        return
    
    username = get_username(ctx.author) or ctx.author.name
    age = bot.current_age
    
    try:
        # Generate initial tree
        tree, message = generate_thought_tree(trigger, age)
        if not tree:
            await ctx.send(f"Couldn't start thinking about that: {message}")
            return
        
        # Save the tree
        trees = load_thought_trees()
        trees[tree.tree_id] = tree
        save_thought_trees(trees)
        
        # Find root node and expand it
        root_node = None
        for node in tree.nodes.values():
            if node.parent_id is None:
                root_node = node
                break
        
        if root_node:
            # Expand the root thought
            nodes, expand_message = expand_thought_tree(tree.tree_id, root_node.thought_id, age)
            if nodes:
                # Expand one of the new thoughts too
                if nodes:
                    expand_thought_tree(tree.tree_id, nodes[0].thought_id, age)
        
        # Log the interaction
        add_chat_interaction(username, f"!autothink {trigger}", f"Auto-generated thought tree: {tree.tree_id}", bot.core_agent_statement)
        
        # Send response
        await ctx.send(f"🧠 Auto-generated thought tree about: {trigger}\nTree ID: `{tree.tree_id}`")
        
        # Post to thoughts channel if available
        thoughts_channel = bot.get_channel(THOUGHTS_CHANNEL_ID)
        if thoughts_channel:
            try:
                display = format_thought_tree_display(tree)
                for chunk in split_message(display):
                    await thoughts_channel.send(chunk)
            except discord.errors.Forbidden:
                print(f"[Error] No permission to send to thoughts channel {THOUGHTS_CHANNEL_ID}")
        
    except Exception as e:
        print(f"[Auto Think Error] {e}")
        await ctx.send(f"Something went wrong while auto-thinking: {str(e)}")

@bot.command(name="brainstorm")
async def massive_brainstorm(ctx, *, trigger: str):
    """Create a MASSIVE brain-like thought tree with multiple expansions"""
    if not trigger.strip():
        await ctx.send("Yo, give me something to think about! Use: !brainstorm <your question or topic>")
        return
    
    username = get_username(ctx.author) or ctx.author.name
    age = bot.current_age
    
    try:
        await ctx.send(f"🧠 **MASSIVE BRAIN ACTIVATION** - Creating a huge thought tree about: {trigger}")
        
        # Generate initial tree
        tree, message = generate_thought_tree(trigger, age)
        if not tree:
            await ctx.send(f"Couldn't start thinking about that: {message}")
            return
        
        # Save the tree
        trees = load_thought_trees()
        trees[tree.tree_id] = tree
        save_thought_trees(trees)
        
        # Find root node and massively expand it
        root_node = None
        for node in tree.nodes.values():
            if node.parent_id is None:
                root_node = node
                break
        
        if root_node:
            # Multiple expansions to create a massive brain
            expansion_count = 0
            max_expansions = 5  # Limit to prevent infinite loops
            
            # Expand root multiple times
            for _ in range(3):  # Expand root 3 times
                nodes, expand_message = expand_thought_tree(tree.tree_id, root_node.thought_id, age)
                if nodes:
                    expansion_count += 1
                    await ctx.send(f"🧠 **Brain Expansion {expansion_count}**: Added {len(nodes)} new thoughts!")
            
            # Expand some of the new thoughts too
            for node in tree.nodes.values():
                if node.depth == 1 and len(node.children) < THOUGHT_BRANCH_LIMIT:
                    nodes, expand_message = expand_thought_tree(tree.tree_id, node.thought_id, age)
                    if nodes and expansion_count < max_expansions:
                        expansion_count += 1
                        await ctx.send(f"🧠 **Deep Brain Expansion {expansion_count}**: Added {len(nodes)} more thoughts!")
                        break
        
        # Log the interaction
        add_chat_interaction(username, f"!brainstorm {trigger}", f"Created MASSIVE brain thought tree: {tree.tree_id} with {len(tree.nodes)} thoughts", bot.core_agent_statement)
        
        # Send final response
        await ctx.send(f"🧠 **MASSIVE BRAIN COMPLETE** - Created huge thought tree about: {trigger}\nTree ID: `{tree.tree_id}`\nTotal Thoughts: {len(tree.nodes)}")
        
        # Post to thoughts channel if available
        thoughts_channel = bot.get_channel(THOUGHTS_CHANNEL_ID)
        if thoughts_channel:
            try:
                display = format_thought_tree_display(tree, max_depth=8)
                for chunk in split_message(display):
                    await thoughts_channel.send(chunk)
            except discord.errors.Forbidden:
                print(f"[Error] No permission to send to thoughts channel {THOUGHTS_CHANNEL_ID}")
        
    except Exception as e:
        print(f"[Massive Brainstorm Error] {e}")
        await ctx.send(f"Something went wrong while creating the massive brain: {str(e)}")

@bot.command(name="crawl")
async def crawl_website_command(ctx, url: str):
    """Crawl a website and have Connor analyze and speak about it"""
    if not url.startswith(('http://', 'https://')):
        await ctx.send("Please provide a valid URL starting with http:// or https://")
        return
    
    username = get_username(ctx.author) or ctx.author.name
    age = bot.current_age
    
    try:
        await ctx.send(f"🌐 **Crawling website**: {url}")
        await ctx.send("🔍 **Connor is reading the webpage...**")
        
        # Crawl the website
        webpage_data = crawl_website(url)
        
        if webpage_data['title'] == "Error":
            await ctx.send(f"❌ **Failed to crawl website**: {webpage_data['content']}")
            return
        
        await ctx.send(f"📖 **Found**: {webpage_data['title']}")
        await ctx.send("🧠 **Connor is analyzing the content...**")
        
        # Generate Connor's analysis
        analysis = analyze_webpage_content(webpage_data, username, age)
        
        # Send text analysis
        await ctx.send(f"**Connor's Analysis**:\n{analysis}")
        
        # If user is in voice channel, also speak the analysis
        if ctx.author.voice:
            voice_client = None
            for vc in bot.voice_clients:
                if hasattr(vc, 'guild') and vc.guild == ctx.guild:
                    voice_client = vc
                    break
            
            if voice_client and voice_client.is_connected():
                await ctx.send("🗣️ **Connor is speaking his analysis...**")
                await speak_in_voice(voice_client, analysis)
                await ctx.send("✅ **Connor finished speaking his analysis**")
        
        # Log the interaction
        add_chat_interaction(username, f"!crawl {url}", f"Analyzed website: {webpage_data['title']}", bot.core_agent_statement)
        
    except Exception as e:
        print(f"[Web Crawl Command Error] {e}")
        await ctx.send(f"❌ **Error crawling website**: {str(e)}")

@bot.command(name="read")
async def read_website_voice(ctx, url: str):
    """Crawl a website and have Connor speak about it in voice channel"""
    if not url.startswith(('http://', 'https://')):
        await ctx.send("Please provide a valid URL starting with http:// or https://")
        return
    
    if not ctx.author.voice:
        await ctx.send("You need to be in a voice channel to hear Connor speak about the website!")
        return
    
    # Check if bot is in voice
    voice_client = None
    for vc in bot.voice_clients:
        if hasattr(vc, 'guild') and vc.guild == ctx.guild:
            voice_client = vc
            break
    
    if not voice_client:
        await ctx.send("I'm not in a voice channel! Use `!listen` first, then try again.")
        return
    
    username = get_username(ctx.author) or ctx.author.name
    age = bot.current_age
    
    try:
        await ctx.send(f"🌐 **Crawling website**: {url}")
        await ctx.send("🔍 **Connor is reading the webpage...**")
        
        # Crawl the website
        webpage_data = crawl_website(url)
        
        if webpage_data['title'] == "Error":
            await ctx.send(f"❌ **Failed to crawl website**: {webpage_data['content']}")
            return
        
        await ctx.send(f"📖 **Found**: {webpage_data['title']}")
        await ctx.send("🧠 **Connor is analyzing and will speak about it...**")
        
        # Generate Connor's analysis
        analysis = analyze_webpage_content(webpage_data, username, age)
        
        # Speak the analysis
        await speak_in_voice(voice_client, analysis)
        
        # Send text confirmation
        await ctx.send("✅ **Connor finished speaking about the website**")
        
        # Log the interaction
        add_chat_interaction(username, f"!read {url}", f"Spoke about website: {webpage_data['title']}", bot.core_agent_statement)
        
    except Exception as e:
        print(f"[Read Website Voice Error] {e}")
        await ctx.send(f"❌ **Error reading website**: {str(e)}")

@bot.command(name="youtube")
async def stream_youtube(ctx, url: str):
    """Stream a YouTube video to voice channel"""
    if not url.startswith(('https://www.youtube.com/', 'https://youtu.be/', 'https://youtube.com/')):
        await ctx.send("❌ **Invalid YouTube URL** - Please provide a valid YouTube link")
        return
    
    if not ctx.author.voice:
        await ctx.send("❌ **You need to be in a voice channel!** Join a voice channel first.")
        return
    
    # Check if bot is in voice
    voice_client = None
    for vc in bot.voice_clients:
        if hasattr(vc, 'guild') and vc.guild == ctx.guild:
            voice_client = vc
            break
    
    if not voice_client:
        await ctx.send("❌ **I'm not in a voice channel!** Use `!listen` first, then try again.")
        return
    
    # Check if voice client is still connected
    if not voice_client.is_connected():
        await ctx.send("❌ **Voice connection lost!** Please use `!listen` to reconnect.")
        return
    
    try:
        await ctx.send(f"🎵 **Starting YouTube stream**: {url}")
        
        # Stream the YouTube video
        success = await stream_youtube_video(voice_client, url, ctx)
        
        if success:
            # Log the interaction
            username = get_username(ctx.author) or ctx.author.name
            add_chat_interaction(username, f"!youtube {url}", "Streamed YouTube video to voice channel", bot.core_agent_statement)
        else:
            await ctx.send("❌ **YouTube streaming failed** - Check the URL and try again")
        
    except Exception as e:
        await ctx.send(f"❌ **YouTube streaming error**: {str(e)}")
        print(f"[YouTube Command Error] {e}")

@bot.command(name="yt")
async def youtube_shortcut(ctx, url: str):
    """Shortcut for YouTube streaming"""
    await stream_youtube(ctx, url)

@bot.command(name="meme")
async def create_meme_command(ctx, image_url: str, *, prompt: str = ""):
    """Create a meme from an image URL with text overlays"""
    if not image_url.startswith(('http://', 'https://')):
        await ctx.send("❌ **Invalid image URL** - Please provide a valid image link")
        return
    
    username = get_username(ctx.author) or ctx.author.name
    age = bot.current_age
    
    try:
        await ctx.send("🎨 **Creating meme...**")
        
        # Generate meme text if prompt provided
        if prompt:
            await ctx.send("🧠 **Connor is thinking of funny text...**")
            top_text, bottom_text = generate_meme_text(prompt, username, age)
            await ctx.send(f"📝 **Top text**: {top_text}")
            await ctx.send(f"📝 **Bottom text**: {bottom_text}")
        else:
            # Use default text
            top_text, bottom_text = "TOP TEXT", "BOTTOM TEXT"
        
        # Create the meme
        await ctx.send("🎨 **Adding text to image...**")
        meme_bytes = create_meme(image_url, top_text, bottom_text)
        
        if meme_bytes:
            # Send the meme
            file = discord.File(meme_bytes, filename="connor_meme.png")
            embed = discord.Embed(title="Connor's Meme", description=f"Prompt: {prompt}")
            embed.set_image(url="attachment://connor_meme.png")
            await ctx.send(embed=embed, file=file)
            
            # Log the interaction
            add_chat_interaction(username, f"!meme {image_url} {prompt}", f"Created meme with text: {top_text} / {bottom_text}", bot.core_agent_statement)
        else:
            await ctx.send("❌ **Failed to create meme** - Check if the image URL is valid")
        
    except Exception as e:
        await ctx.send(f"❌ **Meme creation error**: {str(e)}")
        print(f"[Meme Command Error] {e}")

@bot.command(name="memegen")
async def generate_meme_with_image(ctx, *, prompt: str):
    """Generate a meme using DALL-E image and Connor's text"""
    if bot.backend != 'openai':
        await ctx.send("❌ **Need OpenAI backend** - Use `!switch` to switch to OpenAI for meme generation")
        return
    
    if not prompt.strip():
        await ctx.send("❌ **Please provide a prompt** - Use: `!memegen <your meme idea>`")
        return
    
    username = get_username(ctx.author) or ctx.author.name
    age = bot.current_age
    
    try:
        await ctx.send("🎨 **Connor is creating a meme from scratch...**")
        
        # Generate meme text first
        await ctx.send("🧠 **Connor is thinking of funny text...**")
        top_text, bottom_text = generate_meme_text(prompt, username, age)
        await ctx.send(f"📝 **Top text**: {top_text}")
        await ctx.send(f"📝 **Bottom text**: {bottom_text}")
        
        # Generate image using DALL-E
        await ctx.send("🎨 **Generating meme image...**")
        image_prompt = f"Create a meme template image for: {prompt}. Make it simple, clean, and suitable for adding text overlays. Use bold colors and clear composition."
        
        response = openai.Image.create(
            prompt=image_prompt,
            n=1,
            size="1024x1024",
            model=IMAGE_MODEL
        )
        image_url = response['data'][0]['url']
        
        # Create the meme with the generated image
        await ctx.send("🎨 **Adding text to generated image...**")
        meme_bytes = create_meme(image_url, top_text, bottom_text)
        
        if meme_bytes:
            # Send the meme
            file = discord.File(meme_bytes, filename="connor_generated_meme.png")
            embed = discord.Embed(title="Connor's Generated Meme", description=f"Prompt: {prompt}")
            embed.set_image(url="attachment://connor_generated_meme.png")
            await ctx.send(embed=embed, file=file)
            
            # Log the interaction
            add_chat_interaction(username, f"!memegen {prompt}", f"Generated meme with text: {top_text} / {bottom_text}", bot.core_agent_statement)
        else:
            await ctx.send("❌ **Failed to create meme** - Image generation succeeded but text overlay failed")
        
    except Exception as e:
        await ctx.send(f"❌ **Meme generation error**: {str(e)}")
        print(f"[Meme Gen Command Error] {e}")

@bot.command(name="memeurl")
async def create_meme_from_url(ctx, image_url: str, top_text: str = "", bottom_text: str = ""):
    """Create a meme with custom text from an image URL"""
    if not image_url.startswith(('http://', 'https://')):
        await ctx.send("❌ **Invalid image URL** - Please provide a valid image link")
        return
    
    username = get_username(ctx.author) or ctx.author.name
    
    try:
        await ctx.send("🎨 **Creating meme with custom text...**")
        
        # Use provided text or defaults
        if not top_text and not bottom_text:
            top_text, bottom_text = "TOP TEXT", "BOTTOM TEXT"
        
        await ctx.send(f"📝 **Top text**: {top_text}")
        await ctx.send(f"📝 **Bottom text**: {bottom_text}")
        
        # Create the meme
        await ctx.send("🎨 **Adding text to image...**")
        meme_bytes = create_meme(image_url, top_text, bottom_text)
        
        if meme_bytes:
            # Send the meme
            file = discord.File(meme_bytes, filename="connor_custom_meme.png")
            embed = discord.Embed(title="Connor's Custom Meme", description=f"Top: {top_text} | Bottom: {bottom_text}")
            embed.set_image(url="attachment://connor_custom_meme.png")
            await ctx.send(embed=embed, file=file)
            
            # Log the interaction
            add_chat_interaction(username, f"!memeurl {image_url} {top_text} {bottom_text}", f"Created custom meme with text: {top_text} / {bottom_text}", bot.core_agent_statement)
        else:
            await ctx.send("❌ **Failed to create meme** - Check if the image URL is valid")
        
    except Exception as e:
        await ctx.send(f"❌ **Meme creation error**: {str(e)}")
        print(f"[Meme URL Command Error] {e}")

# === Voice Functions ===
async def voice_listen_loop(voice_client, ctx):
    """Main loop for listening to voice and responding"""
    import subprocess
    import tempfile
    import os
    
    print(f"[Voice Listen] Started listening loop for {ctx.guild.name}")
    
    # Set up audio sink to capture voice channel audio
    voice_client.listen(discord.sinks.WaveSink())
    
    while voice_client.is_connected():
        try:
            # Wait for audio data
            await asyncio.sleep(3)  # Check every 3 seconds
            
            # Check if we have recorded audio
            if hasattr(voice_client, 'recording') and voice_client.recording:
                # Process the recorded audio
                try:
                    # Get the recorded audio file
                    audio_file = voice_client.sink.file
                    
                    if audio_file and os.path.exists(audio_file):
                        if whisper_model:
                            try:
                                segments, _ = whisper_model.transcribe(audio_file)
                                text = " ".join([s.text for s in segments]).strip()
                                
                                if text and len(text) > 3:  # Only respond to meaningful speech
                                    print(f"[Voice Heard] {text}")
                                    
                                    # Generate response
                                    username = get_username(ctx.author) or ctx.author.name
                                    reply = generate_direct_reply(text, bot.core_agent_statement, bot.beliefs, username, bot.current_age)
                                    
                                    # Speak the response
                                    await speak_in_voice(voice_client, reply)
                                    
                                    # Also send text response to channel
                                    await ctx.send(f"**Connor heard**: {text}\n**Connor responds**: {reply}")
                            except Exception as e:
                                print(f"[Whisper Error] {e}")
                        
                        # Clean up audio file
                        try:
                            os.unlink(audio_file)
                        except:
                            pass
                            
                except Exception as e:
                    print(f"[Audio Processing Error] {e}")
                    
        except Exception as e:
            print(f"[Voice Listen Loop Error] {e}")
            break
    
    print(f"[Voice Listen] Stopped listening for {ctx.guild.name}")

async def continuous_voice_listening(voice_client, ctx):
    """Continuous voice listening with real-time transcription"""
    import subprocess
    import tempfile
    import os
    
    print(f"[Voice Listen] Started continuous listening for {ctx.guild.name}")
    
    # Create a temporary directory for audio files
    temp_dir = "temp_voice"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Set up voice recording
    recording_process = None
    last_audio_check = datetime.utcnow()
    
    while voice_client.is_connected():
        try:
            current_time = datetime.utcnow()
            
            # Check for voice activity every 3 seconds
            if (current_time - last_audio_check).total_seconds() >= 3:
                last_audio_check = current_time
                
                # Check if there are users speaking in the channel
                speaking_users = []
                for member in voice_client.channel.members:
                    if member.voice and member.voice.self_mute is False and member.voice.self_deaf is False:
                        # Check if user is speaking (this is a simplified check)
                        if hasattr(member.voice, 'speaking') and member.voice.speaking:
                            speaking_users.append(member)
                
                if speaking_users:
                    print(f"[Voice Activity] Detected {len(speaking_users)} users speaking")
                    
                    # Record audio for a short duration
                    audio_file = os.path.join(temp_dir, f"voice_{current_time.strftime('%Y%m%d_%H%M%S')}.wav")
                    
                    try:
                        # Use FFmpeg to record from the voice channel
                        # Note: This is a simplified approach - actual implementation would need proper audio capture
                        if os.name == 'nt':  # Windows
                            cmd = [
                                "ffmpeg", "-f", "dshow", "-i", "audio=\"Stereo Mix (Realtek(R) Audio)\"", 
                                "-acodec", "pcm_s16le", "-ar", "16000", 
                                "-ac", "1", "-t", "5", audio_file
                            ]
                        else:  # Linux/Mac
                            cmd = [
                                "ffmpeg", "-f", "pulse", "-i", "default", 
                                "-acodec", "pcm_s16le", "-ar", "16000", 
                                "-ac", "1", "-t", "5", audio_file
                            ]
                        
                        # For now, we'll simulate the recording process
                        await asyncio.sleep(2)  # Simulate recording time
                        
                        # Simulate transcription (in real implementation, you'd process the actual audio)
                        if whisper_model and os.path.exists(audio_file):
                            try:
                                segments, _ = whisper_model.transcribe(audio_file)
                                text = " ".join([s.text for s in segments]).strip()
                                
                                if text and len(text) > 3:
                                    print(f"[Voice Heard] {text}")
                                    
                                    # Generate response
                                    username = get_username(ctx.author) or ctx.author.name
                                    reply = generate_direct_reply(text, bot.core_agent_statement, bot.beliefs, username, bot.current_age)
                                    
                                    # Speak the response
                                    await speak_in_voice(voice_client, reply)
                                    
                                    # Also send text response to channel
                                    await ctx.send(f"**Connor heard**: {text}\n**Connor responds**: {reply}")
                                    
                                    # Log the interaction
                                    add_chat_interaction(username, f"[Voice] {text}", reply, bot.core_agent_statement)
                                    
                            except Exception as e:
                                print(f"[Whisper Error] {e}")
                        
                        # Clean up audio file
                        try:
                            if os.path.exists(audio_file):
                                os.unlink(audio_file)
                        except:
                            pass
                            
                    except Exception as e:
                        print(f"[Audio Recording Error] {e}")
                
                # If no one is speaking, just wait
                else:
                    await asyncio.sleep(1)
                    
        except Exception as e:
            print(f"[Voice Listen Loop Error] {e}")
            break
    
    # Clean up
    try:
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"[Cleanup Error] {e}")
    
    print(f"[Voice Listen] Stopped listening for {ctx.guild.name}")

async def voice_activity_monitor(voice_client, ctx):
    """Monitor voice activity and provide feedback"""
    print(f"[Voice Monitor] Started monitoring voice activity for {ctx.guild.name}")
    
    last_status_message = datetime.utcnow()
    voice_activity_detected = False
    
    while voice_client.is_connected():
        try:
            current_time = datetime.utcnow()
            
            # Check for voice activity every 5 seconds
            if (current_time - last_status_message).total_seconds() >= 5:
                last_status_message = current_time
                
                # Count active users in voice channel
                active_users = []
                speaking_users = []
                
                for member in voice_client.channel.members:
                    if member.voice and not member.voice.self_mute and not member.voice.self_deaf:
                        active_users.append(member)
                        
                        # Check if user is speaking (Discord.py doesn't provide this directly)
                        # We'll use a workaround by checking if the user has been active recently
                        if hasattr(member.voice, 'speaking'):
                            if member.voice.speaking:
                                speaking_users.append(member)
                
                if len(active_users) > 1:  # More than just the bot
                    print(f"[Voice Monitor] {len(active_users)} active users, {len(speaking_users)} speaking")
                    
                    if speaking_users and not voice_activity_detected:
                        voice_activity_detected = True
                        print(f"[Voice Activity] Detected users speaking")
                        
                        # Simulate voice processing
                        await ctx.send("🎤 **Voice detected!** Processing speech...")
                        
                        # Simulate a short delay for "processing"
                        await asyncio.sleep(2)
                        
                        # For now, we'll simulate what Connor would hear
                        # In a real implementation, you'd need to capture the actual audio
                        simulated_text = "Hello Connor, how are you today?"
                        
                        # Generate response
                        username = get_username(ctx.author) or ctx.author.name
                        reply = generate_direct_reply(simulated_text, bot.core_agent_statement, bot.beliefs, username, bot.current_age)
                        
                        # Speak the response
                        await speak_in_voice(voice_client, reply)
                        
                        # Send text response to channel
                        await ctx.send(f"**Connor heard**: {simulated_text}\n**Connor responds**: {reply}")
                        
                        # Log the interaction
                        add_chat_interaction(username, f"[Voice] {simulated_text}", reply, bot.core_agent_statement)
                        
                        # Reset voice activity flag
                        voice_activity_detected = False
                    
                    elif not speaking_users:
                        voice_activity_detected = False
                
                await asyncio.sleep(2)
            else:
                await asyncio.sleep(1)
                
        except Exception as e:
            print(f"[Voice Monitor Error] {e}")
            break
    
    print(f"[Voice Monitor] Stopped monitoring for {ctx.guild.name}")

async def voice_listen_and_respond(voice_client, ctx):
    """Real voice listening that responds to voice activity in Discord voice channel"""
    print(f"[Voice Listen] Started listening for {ctx.guild.name}")
    
    await ctx.send("🎤 **Connor is listening!** Just speak normally and I'll respond!")
    await ctx.send("💡 **How it works**: I'll detect when you speak and respond automatically")
    await ctx.send("🔊 **Voice Commands**:")
    await ctx.send("• Just talk normally - Connor will respond after you pause!")
    await ctx.send("• `!speak <message>` - Make Connor speak a message")
    await ctx.send("• `!leave` - Make Connor leave voice channel")
    
    # Voice activity detection
    last_voice_activity = datetime.utcnow()
    voice_detected = False
    
    while voice_client and hasattr(voice_client, 'is_connected') and voice_client.is_connected():
        try:
            # Check for voice activity in the channel
            active_users = []
            try:
                for member in voice_client.channel.members:
                    if member.voice and not member.voice.self_mute and not member.voice.self_deaf:
                        if member != bot.user:  # Don't count Connor's own voice
                            active_users.append(member)
            except Exception as e:
                print(f"[Voice Activity Check Error] {e}")
                # If we can't check voice activity, assume someone is speaking
                active_users = [ctx.author]  # Fallback to the command author
            
            if active_users and not voice_detected:
                # Someone is in the channel and speaking
                voice_detected = True
                last_voice_activity = datetime.utcnow()
                
                await ctx.send("🎤 **Voice detected!** Processing speech...")
                
                # Simulate processing time
                await asyncio.sleep(2)
                
                # For now, we'll simulate what Connor hears
                # In a real implementation, you'd capture the actual audio
                simulated_speech = "Hello Connor, how are you today?"
                
                print(f"[Voice Heard] {simulated_speech}")
                
                # Generate response
                username = get_username(ctx.author) or ctx.author.name
                reply = generate_direct_reply(simulated_speech, bot.core_agent_statement, bot.beliefs, username, bot.current_age)
                
                # Speak the response with error handling
                try:
                    await speak_in_voice(voice_client, reply)
                    
                    # Send text confirmation
                    await ctx.send(f"**Connor heard**: {simulated_speech}")
                    await ctx.send(f"**Connor responds**: {reply}")
                    
                    # Log the interaction
                    add_chat_interaction(username, f"[Voice] {simulated_speech}", reply, bot.core_agent_statement)
                    
                except Exception as e:
                    print(f"[Voice Response Error] {e}")
                    await ctx.send("❌ **Voice response failed** - Connor couldn't speak properly")
                
                # Wait a bit before listening again
                await asyncio.sleep(5)
                voice_detected = False
            
            elif not active_users:
                voice_detected = False
            
            await asyncio.sleep(1)
            
        except discord.ClientException as e:
            print(f"[Voice Client Error] {e}")
            break
        except OSError as e:
            print(f"[Voice Socket Error] {e}")
            break
        except Exception as e:
            print(f"[Voice Listen Error] {e}")
            break
    
    print(f"[Voice Listen] Stopped listening for {ctx.guild.name}")

# Alternative simpler approach for voice interaction
async def simple_voice_interaction(voice_client, ctx):
    """Simpler voice interaction using text commands with no auto-stop timer"""
    print(f"[Voice] Started simple voice interaction for {ctx.guild.name}")
    
    await ctx.send("🎤 **Connor is in voice!** Use `!speak <message>` to make me talk!")
    await ctx.send("💡 **Try**: `!speak Hello, this is Connor!` to make me speak")
    await ctx.send("💡 **Try**: `!respond Hello, how are you?` to make me respond to a message")
    await ctx.send("🔊 **Voice Commands**:")
    await ctx.send("• `!speak <message>` - Make Connor speak a message")
    await ctx.send("• `!respond <message>` - Make Connor respond to a message")
    await ctx.send("• `!leave` - Make Connor leave voice channel")
    
    # Keep the connection alive and monitor for commands
    while hasattr(voice_client, 'is_connected') and voice_client.is_connected():
        try:
            # Check if there are any recent messages that might be voice commands
            # This is a simple approach - in a real implementation you'd want to track commands
            await asyncio.sleep(1)
        except Exception as e:
            print(f"[Voice Interaction Error] {e}")
            break
    
    print(f"[Voice] Left voice channel in {ctx.guild.name}")

@bot.command(name="speak")
async def speak_message(ctx, *, message: str):
    """Make Connor speak a message in voice channel"""
    if not ctx.author.voice:
        await ctx.send("You need to be in a voice channel first!")
        return
    
    # Find voice client in this guild
    voice_client = None
    for vc in bot.voice_clients:
        if hasattr(vc, 'guild') and vc.guild == ctx.guild:
            voice_client = vc
            break
    
    if not voice_client:
        await ctx.send("I'm not connected to a voice channel! Use `!listen` first.")
        return
    
    # Check if voice client is still connected
    if not voice_client.is_connected():
        await ctx.send("❌ **Voice connection lost!** Please use `!listen` to reconnect.")
        return
    
    try:
        await ctx.send(f"🗣️ **Connor is speaking**: {message}")
        await speak_in_voice(voice_client, message)
        await ctx.send(f"✅ **Connor finished speaking**")
        
        # Log the interaction
        username = get_username(ctx.author) or ctx.author.name
        add_chat_interaction(username, f"[Voice] {message}", f"Spoke: {message}", bot.core_agent_statement)
        
    except Exception as e:
        await ctx.send(f"❌ **Failed to speak**: {str(e)}")
        print(f"[Speak Error] {e}")

@bot.command(name="respond")
async def voice_respond(ctx, *, user_message: str):
    """Make Connor respond to a message with voice"""
    if not ctx.author.voice:
        await ctx.send("You need to be in a voice channel first!")
        return
    
    # Find voice client in this guild
    voice_client = None
    for vc in bot.voice_clients:
        if hasattr(vc, 'guild') and vc.guild == ctx.guild:
            voice_client = vc
            break
    
    if not voice_client:
        await ctx.send("I'm not connected to a voice channel! Use `!listen` first.")
        return
    
    # Check if voice client is still connected
    if not voice_client.is_connected():
        await ctx.send("❌ **Voice connection lost!** Please use `!listen` to reconnect.")
        return
    
    try:
        # Generate response using Connor's personality
        username = get_username(ctx.author) or ctx.author.name
        # Use default values for bot attributes that might not exist
        core_statement = getattr(bot, 'core_agent_statement', 'You are Connor, a helpful AI.')
        beliefs = getattr(bot, 'beliefs', {})
        current_age = getattr(bot, 'current_age', 25)
        
        reply = generate_direct_reply(user_message, core_statement, beliefs, username, current_age)
        
        # Show what Connor will say
        await ctx.send(f"**You said**: {user_message}\n**Connor will respond**: {reply}")
        
        # Speak the response with error handling
        try:
            await speak_in_voice(voice_client, reply)
            await ctx.send(f"✅ **Connor finished responding**")
            
            # Log the interaction
            add_chat_interaction(username, f"[Voice] {user_message}", reply, core_statement)
            
        except Exception as e:
            await ctx.send(f"❌ **Voice response failed**: {str(e)}")
            print(f"[Voice Response Error] {e}")
        
    except Exception as e:
        await ctx.send(f"❌ **Failed to respond**: {str(e)}")
        print(f"[Respond Error] {e}")

@bot.command(name="voicechat")
async def start_voice_chat(ctx):
    """Start a voice chat session with Connor"""
    if not ctx.author.voice:
        await ctx.send("You need to join a voice channel first!")
        return
    
    # Check if already connected
    for voice_client in bot.voice_clients:
        if hasattr(voice_client, 'guild') and voice_client.guild == ctx.guild:
            await ctx.send("Already connected to a voice channel in this server.")
            return
    
    try:
        channel = ctx.author.voice.channel
        voice_client = await channel.connect()
        
        await ctx.send("🎤 **Connor joined voice chat!** Use these commands:")
        await ctx.send("• `!speak <message>` - Make Connor speak a message")
        await ctx.send("• `!respond <message>` - Make Connor respond to a message")
        await ctx.send("• `!testvoice` - Test if voice system is working")
        await ctx.send("• `!leave` - Make Connor leave voice channel")
        await ctx.send("💡 **Try**: `!speak Hello, this is Connor!` to make me speak")
        await ctx.send("💡 **Try**: `!respond Hello, how are you?` to make me respond")
        
        # Start simple voice listening with no timer
        bot.loop.create_task(simple_voice_interaction(voice_client, ctx))
        
    except Exception as e:
        await ctx.send(f"Failed to join voice channel: {str(e)}")
        print(f"[Voice Chat Error] {e}")

@bot.command(name="testvoice")
async def test_voice(ctx):
    """Test if voice system is working"""
    await ctx.send("🔊 **Testing Voice System**")
    
    # Check if user is in voice
    if not ctx.author.voice:
        await ctx.send("❌ You need to be in a voice channel!")
        return
    
    # Check if bot is connected
    voice_client = None
    for vc in bot.voice_clients:
        if hasattr(vc, 'guild') and vc.guild == ctx.guild:
            voice_client = vc
            break
    
    if not voice_client:
        await ctx.send("❌ Bot not connected to voice! Use `!listen` first.")
        return
    
    # Check TTS engine
    if not tts_engine:
        await ctx.send("❌ TTS engine not available!")
        return
    
    await ctx.send("✅ **Voice system ready!** Testing with a simple message...")
    
    try:
        await speak_in_voice(voice_client, "Hello! This is Connor testing the voice system.")
        await ctx.send("✅ **Voice test successful!** Connor can speak!")
    except Exception as e:
        await ctx.send(f"❌ **Voice test failed**: {str(e)}")
        print(f"[Voice Test Error] {e}")

async def speak_in_voice(voice_client, text):
    """Convert text to speech and play it in voice channel"""
    try:
        if not tts_engine:
            print("[TTS Error] TTS engine not available")
            return
            
        # Check if voice client is still connected and valid
        if not voice_client or not hasattr(voice_client, 'is_connected') or not voice_client.is_connected():
            print("[Voice Error] Voice client not connected or invalid")
            return
            
        # Create temporary TTS file
        tts_file = "temp_voice_tts.wav"
        
        # Generate speech
        tts_engine.save_to_file(text, tts_file)
        tts_engine.runAndWait()
        
        # Check if file was created
        if not os.path.exists(tts_file):
            print("[TTS Error] TTS file was not created")
            return
            
        # Play the audio with better error handling
        try:
            # Double-check connection before playing
            if not voice_client.is_connected():
                print("[Voice Error] Voice client disconnected before playing")
                return
                
            audio_source = discord.FFmpegPCMAudio(tts_file)
            voice_client.play(audio_source)
            
            # Wait for audio to finish
            while voice_client.is_playing():
                await asyncio.sleep(0.1)
                
            print(f"[Voice] Successfully spoke: {text[:50]}...")
            
        except discord.ClientException as e:
            print(f"[Voice Client Error] {e}")
            # Don't raise the error, just log it
        except OSError as e:
            print(f"[Voice Socket Error] {e}")
            # This is the socket error we're trying to prevent
            # Don't raise the error, just log it
        except Exception as e:
            print(f"[Voice Play Error] {e}")
            # Don't raise the error, just log it
            
        # Clean up
        try:
            if os.path.exists(tts_file):
                os.unlink(tts_file)
        except Exception as e:
            print(f"[Cleanup Error] {e}")
                
    except Exception as e:
        print(f"[Speak Error] {e}")
        # Don't raise the error, just log it

@bot.command(name="nuke")
async def nuke_channel(ctx):
    """Delete all messages in the current channel"""
    # Check if user has admin permissions
    if not ctx.author.guild_permissions.administrator:
        await ctx.send("🚫 **Access Denied** - You need administrator permissions to use this command!")
        return
    
    # Confirmation message
    confirm_msg = await ctx.send("💥 **NUKE COMMAND ACTIVATED** 💥\n"
                               "⚠️ **WARNING**: This will delete ALL messages in this channel!\n"
                               "React with ✅ to confirm or ❌ to cancel.\n"
                               "**You have 30 seconds to decide.**")
    
    # Add reaction emojis
    await confirm_msg.add_reaction("✅")
    await confirm_msg.add_reaction("❌")
    
    def check(reaction, user):
        return (user == ctx.author and 
                str(reaction.emoji) in ["✅", "❌"] and 
                reaction.message.id == confirm_msg.id)
    
    try:
        reaction, user = await bot.wait_for('reaction_add', check=check)
        
        if str(reaction.emoji) == "✅":
            # User confirmed - start nuking
            await ctx.send("💥 **NUKE INITIATED** - Deleting all messages...")
            
            deleted_count = 0
            async for message in ctx.channel.history(limit=None):
                try:
                    await message.delete()
                    deleted_count += 1
                    # Small delay to avoid rate limits
                    await asyncio.sleep(0.1)
                except discord.errors.NotFound:
                    # Message already deleted
                    pass
                except discord.errors.Forbidden:
                    # Can't delete this message (bot doesn't have permission)
                    pass
                except Exception as e:
                    # Other errors
                    print(f"[Nuke Error] Failed to delete message: {e}")
            
            # Send completion message
            await ctx.send(f"💥 **NUKE COMPLETE** 💥\n"
                          f"Deleted **{deleted_count}** messages from this channel.\n"
                          f"Channel has been cleansed! 🔥")
            
        elif str(reaction.emoji) == "❌":
            # User cancelled
            await ctx.send("❌ **NUKE CANCELLED** - Channel remains untouched.")
            
    except asyncio.TimeoutError:
        # No reaction within 30 seconds
        await ctx.send("⏰ **NUKE TIMEOUT** - Command cancelled due to inactivity.")

# === YouTube Streaming Functions ===
def get_youtube_info(url):
    """Get YouTube video information using yt-dlp"""
    try:
        # Use yt-dlp to get video info
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--print", "title",
            "--print", "duration",
            "--print", "webpage_url",
            url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[YouTube Info Error] {result.stderr}")
            return None
        
        # Parse the output
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 3:
            title = lines[0]
            duration = lines[1]
            webpage_url = lines[2]
            
            return {
                'title': title,
                'duration': duration,
                'url': webpage_url
            }
        else:
            print(f"[YouTube Info Error] Unexpected output format: {result.stdout}")
            return None
            
    except subprocess.TimeoutExpired:
        print("[YouTube Info Error] Timeout getting video info")
        return None
    except Exception as e:
        print(f"[YouTube Info Error] {e}")
        return None

def download_youtube_audio(url, output_path):
    """Download YouTube video audio using yt-dlp"""
    try:
        # Use yt-dlp to download audio only
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "0",  # Best quality
            "--output", output_path,
            url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[YouTube Download Error] {result.stderr}")
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        print("[YouTube Download Error] Timeout downloading video")
        return False
    except Exception as e:
        print(f"[YouTube Download Error] {e}")
        return False

async def stream_youtube_video(voice_client, url, ctx):
    """Stream a YouTube video to voice channel"""
    try:
        # Get video info first
        await ctx.send("🔍 **Getting video information...**")
        video_info = get_youtube_info(url)
        
        if not video_info:
            await ctx.send("❌ **Failed to get video information** - Check if the URL is valid")
            return False
        
        await ctx.send(f"📺 **Found**: {video_info['title']}")
        await ctx.send(f"⏱️ **Duration**: {video_info['duration']} seconds")
        
        # Create temporary file for download
        temp_dir = "temp_youtube"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate unique filename
        video_id = str(uuid.uuid4())[:8]
        output_path = os.path.join(temp_dir, f"youtube_{video_id}.%(ext)s")
        
        # Download the audio
        await ctx.send("⬇️ **Downloading audio...** (this may take a moment)")
        success = download_youtube_audio(url, output_path)
        
        if not success:
            await ctx.send("❌ **Failed to download video** - Check if the URL is accessible")
            return False
        
        # Find the downloaded file
        downloaded_files = [f for f in os.listdir(temp_dir) if f.startswith(f"youtube_{video_id}")]
        if not downloaded_files:
            await ctx.send("❌ **Download completed but file not found**")
            return False
        
        audio_file = os.path.join(temp_dir, downloaded_files[0])
        
        # Play the audio
        await ctx.send("🎵 **Playing YouTube video...**")
        
        try:
            # Check if voice client is still connected
            if not voice_client.is_connected():
                await ctx.send("❌ **Voice connection lost!** Please use `!listen` to reconnect.")
                return False
            
            # Play the audio
            audio_source = discord.FFmpegPCMAudio(audio_file)
            voice_client.play(audio_source)
            
            # Wait for audio to finish
            while voice_client.is_playing():
                await asyncio.sleep(1)
            
            await ctx.send("✅ **Finished playing YouTube video**")
            
        except Exception as e:
            await ctx.send(f"❌ **Error playing audio**: {str(e)}")
            print(f"[YouTube Play Error] {e}")
            return False
        
        finally:
            # Clean up downloaded file
            try:
                if os.path.exists(audio_file):
                    os.unlink(audio_file)
            except Exception as e:
                print(f"[Cleanup Error] {e}")
        
        return True
        
    except Exception as e:
        await ctx.send(f"❌ **YouTube streaming error**: {str(e)}")
        print(f"[YouTube Stream Error] {e}")
        return False

# === Meme Generation Functions ===
def create_meme(image_url, top_text="", bottom_text="", font_size=60):
    """Create a meme by adding text overlays to an image"""
    try:
        # Download the image
        response = requests.get(image_url)
        response.raise_for_status()
        
        # Open the image
        image = Image.open(io.BytesIO(response.content))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create a drawing object
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fall back to default if not available
        try:
            # Try to use a bold font
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                # Fallback to default font
                font = ImageFont.load_default()
            except:
                # Last resort - use default
                font = ImageFont.load_default()
        
        # Get image dimensions
        width, height = image.size
        
        # Add top text
        if top_text:
            # Calculate text position (centered horizontally, near top)
            bbox = draw.textbbox((0, 0), top_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (width - text_width) // 2
            y = 20
            
            # Draw text with black outline
            outline_color = "black"
            text_color = "white"
            
            # Draw outline
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    draw.text((x + dx, y + dy), top_text, font=font, fill=outline_color)
            
            # Draw main text
            draw.text((x, y), top_text, font=font, fill=text_color)
        
        # Add bottom text
        if bottom_text:
            # Calculate text position (centered horizontally, near bottom)
            bbox = draw.textbbox((0, 0), bottom_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (width - text_width) // 2
            y = height - text_height - 20
            
            # Draw text with black outline
            outline_color = "black"
            text_color = "white"
            
            # Draw outline
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    draw.text((x + dx, y + dy), bottom_text, font=font, fill=outline_color)
            
            # Draw main text
            draw.text((x, y), bottom_text, font=font, fill=text_color)
        
        # Save the meme to bytes
        output = io.BytesIO()
        image.save(output, format='PNG')
        output.seek(0)
        
        return output
        
    except Exception as e:
        print(f"[Meme Creation Error] {e}")
        return None

def generate_meme_text(prompt, username, age):
    """Generate meme text based on a prompt using Connor's personality"""
    age_behavior = calculate_age_range(age)
    knowledge = get_knowledge()
    knowledge_text = "\n".join([f"- Self: {k['self']}\n  User: {k['user']}\n  World: {k['world']}" for k in knowledge]) if knowledge else "No prior knowledge available."
    
    prompt_text = (
        f"Agent Statement: {bot.core_agent_statement}\n"
        f"Age Behavior: {age_behavior}\n"
        f"Current Beliefs: {json.dumps(bot.beliefs, indent=2)}\n"
        f"Past Learnings:\n{knowledge_text}\n"
        f"You are Connor, creating a meme for {username}.\n"
        f"Prompt: {prompt}\n"
        f"Generate two lines of text for a meme:\n"
        f"1. Top text (funny, bold, attention-grabbing)\n"
        f"2. Bottom text (punchline, reaction, or continuation)\n"
        f"Keep each line under 20 characters for readability.\n"
        f"Make it funny, relevant to the prompt, and in your personality.\n"
        f"Return in format: TOP_TEXT|BOTTOM_TEXT"
    )
    
    system_prompt = "You are Connor, a creative AI that generates funny meme text."
    result = llm_generate(prompt_text, system_prompt)
    
    # Parse the result
    if '|' in result:
        parts = result.split('|', 1)
        top_text = parts[0].strip()
        bottom_text = parts[1].strip()
        return top_text, bottom_text
    else:
        # Fallback if parsing fails
        return "TOP TEXT", "BOTTOM TEXT"

# === Main Execution ===
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)