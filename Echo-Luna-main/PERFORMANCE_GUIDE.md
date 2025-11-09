# H.I.M. Model Performance & Persistence Guide

## 🚀 New Features Added

### 💾 **Memory Persistence System**
- **Auto-Save**: Automatically saves your H.I.M. model state every 30 seconds (configurable)
- **Manual Save/Load**: Save and load states manually with timestamps
- **Crash Recovery**: Automatically loads the most recent state on startup
- **State Preservation**: Saves emotions, hormones, memories, dreams, and cycle count

### ⚡ **Performance Controls**
- **Performance Modes**: Low, Balanced, High performance settings
- **Cycle Delay Control**: Adjustable delay between processing cycles (0.5-10 seconds)
- **System Monitoring**: Real-time CPU and RAM usage display
- **Auto-Adjustment**: Automatically reduces performance when system load is high

## 🎯 **Performance Modes Explained**

### **Low Performance Mode**
- **Cycle Delay**: 3.0 seconds
- **Auto-Save**: Every 60 seconds
- **Use When**: High CPU/RAM usage, older hardware, or when running other intensive programs

### **Balanced Performance Mode** (Default)
- **Cycle Delay**: 2.0 seconds  
- **Auto-Save**: Every 30 seconds
- **Use When**: Normal usage, good hardware, moderate system load

### **High Performance Mode**
- **Cycle Delay**: 1.0 seconds
- **Auto-Save**: Every 15 seconds
- **Use When**: Powerful hardware, dedicated system, want maximum responsiveness

## 🔧 **How to Use Performance Controls**

### **1. Performance Mode Selection**
- Go to Dashboard tab → Performance Controls
- Select from dropdown: Low, Balanced, or High
- Mode automatically adjusts cycle delay and save frequency

### **2. Manual Cycle Delay Adjustment**
- Use the slider to set custom cycle delay (0.5-10 seconds)
- Lower = faster processing, higher = less CPU usage
- Real-time display shows current delay

### **3. System Monitoring**
- Watch CPU and RAM usage in real-time
- GUI automatically switches to Low mode if:
  - CPU usage > 80%
  - RAM usage > 85%
- Switches back to Balanced when system load decreases

## 💾 **Memory Persistence Features**

### **Auto-Save System**
- **Automatic**: Saves state every 30 seconds (configurable by mode)
- **File Location**: `him_saves/him_autosave.pkl`
- **Silent Operation**: No user interaction required
- **Crash Protection**: Last state automatically restored on restart

### **Manual Save/Load**
- **Save State**: Creates timestamped save file
- **Load State**: Restores from most recent save
- **File Format**: `him_state_YYYYMMDD_HHMMSS.pkl`
- **Complete State**: Saves everything including memories and dreams

### **What Gets Saved**
- ✅ H.I.M. model state (all sectors)
- ✅ Current cycle count
- ✅ Emotion history (last 50 data points)
- ✅ Hormone history (last 50 data points)
- ✅ All stored memories
- ✅ All dream fragments
- ✅ Performance settings
- ✅ Cycle delay settings

## 🛡️ **Crash Recovery**

### **Automatic Recovery**
1. **On Startup**: GUI automatically looks for saved states
2. **Load Latest**: Restores most recent save file
3. **Continue Where Left Off**: Resumes from exact cycle and state
4. **Memory Preservation**: All memories and dreams restored

### **Manual Recovery**
1. **Load State Button**: Manually load any saved state
2. **File Selection**: Choose specific save file to restore
3. **State Verification**: Check system status to confirm restoration

## 📊 **Performance Optimization Tips**

### **For Low-End Hardware**
- Use **Low Performance Mode**
- Set cycle delay to 4-5 seconds
- Close other applications
- Monitor CPU/RAM usage

### **For High-End Hardware**
- Use **High Performance Mode**
- Set cycle delay to 1 second or less
- Enable more frequent auto-saves
- Run multiple instances if desired

### **For Balanced Usage**
- Use **Balanced Performance Mode** (default)
- Let auto-adjustment handle system load
- Monitor performance indicators
- Adjust manually if needed

## 🔍 **Troubleshooting**

### **High CPU Usage**
- Switch to Low Performance Mode
- Increase cycle delay to 3+ seconds
- Check for other running applications
- Restart the application

### **Memory Issues**
- Clear old memories if needed
- Reduce auto-save frequency
- Use Low Performance Mode
- Monitor RAM usage indicator

### **Save/Load Issues**
- Check `him_saves` folder exists
- Verify file permissions
- Try manual save/load
- Check disk space

## 🎮 **Best Practices**

1. **Start with Balanced Mode**: Let the system auto-adjust
2. **Monitor Performance**: Watch CPU/RAM indicators
3. **Save Regularly**: Use manual save for important states
4. **Adjust as Needed**: Change settings based on system performance
5. **Clean Up**: Periodically clear old save files if needed

## 📁 **File Structure**

```
him_saves/
├── him_autosave.pkl          # Latest auto-save
├── him_state_20250105_143022.pkl  # Manual save with timestamp
├── him_state_20250105_150145.pkl  # Another manual save
└── ...
```

## 🎉 **Benefits**

- **Never Lose Progress**: Auto-save prevents data loss
- **Optimized Performance**: Adapts to your hardware
- **Crash Recovery**: Resume exactly where you left off
- **Flexible Control**: Adjust settings for your needs
- **System Monitoring**: Real-time performance feedback

Your H.I.M. model will now remember everything and adapt to your PC's performance automatically! 🚀✨
