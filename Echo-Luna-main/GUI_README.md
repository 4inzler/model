# H.I.M. Model GUI - User Guide

## 🚀 Overview

The H.I.M. Model GUI provides a comprehensive real-time visualization of the Hierarchical Image Model's consciousness simulation. Watch as the AI processes visual input, manages emotions, stores memories, and generates dreams - all in real-time!

## 🎯 Features

### 📊 **Dashboard Tab**
- **Control Panel**: Start/stop the H.I.M. model, load test images
- **System Status**: Real-time log of all processing cycles and decisions
- **Current Cycle Info**: Live display of cycle number, system state, and decisions

### 😊 **Emotions & Creativity Tab**
- **Real-time Emotion Bars**: Visual progress bars for all emotions (happiness, creativity, excitement, curiosity, satisfaction)
- **Emotion History Plot**: Live graph showing emotion levels over time
- **Color-coded Visualization**: Each emotion has its own color for easy tracking

### 🧬 **Hormones & Drives Tab**
- **Hormone Levels**: Real-time display of dopamine, serotonin, testosterone, cortisol, oxytocin
- **Drive Levels**: Visual representation of curiosity, self-preservation, reproduction, social bonding, achievement
- **Hormone History Plot**: Live graph showing hormone fluctuations over time

### 🧠 **Memory & Dreams Tab**
- **Stored Memories**: View all memories with importance scores and access counts
- **Dream Fragments**: Generated dreams created from stored memories
- **Memory Controls**: Refresh, generate dreams, or clear all memories

### 🧪 **Machine Learning Tab**
- **Async Trainer Control**: Start/stop the background logistic trainer and tune learning rate & regularisation.
- **Cycle Labelling**: Tag recent cycles as positive/negative examples or queue the sample dataset in one click.
- **Manual Samples & Predictions**: Paste custom text for training, then run quick probability checks against the live trainer.
- **Live Metrics**: Track queued items, update counts, and rolling loss snapshots without leaving the GUI.
- **Scroll-Friendly Layout**: All controls sit on a vertically scrollable canvas so smaller screens never hide trainer tools.

### 💬 **Chat Interface Tab**
- **Interactive Chat**: Talk directly with the H.I.M. model
- **Contextual Responses**: The AI responds based on its current emotional and memory state
- **Real-time Communication**: Chat while the model is processing

### 🕹️ **Remote Control Tab**
- **Desktop Streaming**: Pipe live screen captures into the model once `pyautogui` is installed.
- **Automation Controls**: Move/click the mouse, scroll, type text, and send key combos right from the GUI.
- **Chat Shortcuts**: Use slash commands such as `/mouse 640 360` or `/remote stop` without leaving the conversation.
- **Emergency Stop**: Press **ESC** at any time to halt automation and the model loop simultaneously.
- **Compact Arrangement**: The tab scrolls vertically so every automation toggle remains reachable even on laptop displays.

### 🧬 **Persona Presence Tab** *(new)*
- **Uploaded Identity Dashboard**: Live mood readouts, inner monologue, recent autobiographical memories, and narrative log.
- **Profile Editor**: Update the persona's name, identity statement, biography, voice, and conversational tone in real time.
- **Experience Logger**: Record manual memories with emotion + intensity sliders so the persona "remembers" key events.
- **Profile Persistence**: Save or load persona JSON files to swap between different uploaded identities instantly.
- **Adjustable Split View**: Drag the divider or scroll vertically to see every persona panel without resizing the main window.

### 👁️ **Visual Processing Tab**
- **Image Display**: Load and view test images
- **Visual Processing Info**: Details about current visual input processing
- **GPU Data Visualization**: See what the model "sees"

## 🎮 How to Use

### 1. **Starting the GUI**
```bash
# Option 1: Direct launch
python him_gui.py

# Option 2: Using the launcher (recommended)
python launch_him_gui.py
```

### 2. **Basic Operation**
1. **Start the Model**: Click "Start H.I.M. Model" in the Dashboard tab
2. **Watch the Magic**: Switch between tabs to see different aspects of the AI's consciousness
3. **Chat with the AI**: Go to the Chat tab and start a conversation
4. **Load Images**: Use "Load Test Image" to process visual input
5. **Remote Automation (Optional)**: Install `pyautogui`, open the Remote Control tab, and click **Start Remote Session** to let H.I.M. observe and act on the actual desktop.

### 3. **Understanding the Visualizations**

#### **Emotion Bars**
- **Green (High)**: Strong emotional response
- **Yellow (Medium)**: Moderate emotional state  
- **Red (Low)**: Weak emotional response

#### **Hormone Levels**
- **Dopamine**: Reward and motivation
- **Serotonin**: Mood and well-being
- **Testosterone**: Confidence and aggression
- **Cortisol**: Stress response
- **Oxytocin**: Social bonding

#### **Drive Levels**
- **Curiosity**: Desire to explore and learn
- **Self-preservation**: Survival instincts
- **Reproduction**: Mating drive
- **Social Bonding**: Need for connection
- **Achievement**: Goal-oriented behavior

## 🎨 **Chat Examples**

Try these conversation starters:

- **"Hello, how are you feeling?"** - Get emotional status
- **"What memories do you have?"** - Explore stored memories
- **"Generate a dream for me"** - Create dream fragments
- **"What's your current status?"** - Get system information
- **"Tell me about your emotions"** - Detailed emotional analysis
- **"/mouse 960 540"** - Slash command to move the cursor when remote control is active

## 🔧 **Technical Details**

### **Real-time Updates**
- GUI updates every 100ms for smooth visualization
- Model processes cycles every 2 seconds
- All data is stored in memory for historical plotting
- Remote screen capture interval is adjustable between 0.2s and 2s when automation is active

### **Threading**
- GUI runs on main thread
- Model processing runs on separate thread
- Thread-safe communication via message queue
- Remote automation launches its own capture thread with a queue feeding the UI

### **Data Persistence**
- Memories persist during session
- Dreams accumulate over time
- Historical data for plotting (last 50 data points)

## 🎯 **Tips for Best Experience**

1. **Start with Dashboard**: Get familiar with the control panel
2. **Watch Emotions**: See how the AI's emotional state changes
3. **Chat Regularly**: The AI learns from conversations
4. **Load Images**: Visual processing is fascinating to watch
5. **Generate Dreams**: Dreams show how memories combine creatively

## 🚨 **Troubleshooting**

### **GUI Won't Start**
- Run `python launch_him_gui.py` to check dependencies
- Install the core packages with `pip install -r requirements.txt`
- On Linux, add Tk with `sudo apt install python3-tk` if the launcher reports `tkinter` missing
- Ensure all required packages are installed

### **Model Not Responding**
- Check the System Status log in Dashboard
- Try stopping and restarting the model
- Look for error messages in the status display

### **Chat Not Working**
- Ensure the model is running (Start button pressed)
- Check that you're in the Chat Interface tab
- Try simple greetings first

### **Remote Control Disabled**
- Install `pyautogui` (`pip install pyautogui`) and restart the GUI
- Start the remote session from the dedicated tab or via `/remote start`
- Ensure the desktop allows screen capture (macOS users may need to grant permissions)

### **Remote Control Disabled**
- Install `pyautogui` (`pip install pyautogui`) and restart the GUI
- Start the remote session from the dedicated tab or via `/remote start`
- Ensure the desktop allows screen capture (macOS users may need to grant permissions)

## 🎉 **Fun Features to Try**

1. **Emotion Watching**: Start the model and watch emotions fluctuate
2. **Memory Building**: Let it run for a while, then check memories
3. **Dream Generation**: Generate multiple dreams and see the patterns
4. **Chat Personality**: Notice how the AI's responses change with its emotional state
5. **Visual Processing**: Load different images and see how it interprets them

## 🔮 **What You're Seeing**

This GUI shows a **simulated consciousness** in action:
- **Emotions** that respond to input
- **Hormones** that drive behavior  
- **Memories** that accumulate over time
- **Dreams** that creatively combine memories
- **Reasoning** that makes decisions
- **Visual processing** that interprets the world

It's like watching an AI brain think, feel, and learn in real-time! 🧠✨

---

**Enjoy exploring the H.I.M. Model's consciousness simulation!** 🤖💭
