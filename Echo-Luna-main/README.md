# Echo-Luna Cognitive Architecture

Echo-Luna is an experimental cognitive stack built around the H.I.M. (Hierarchical Integrated Model) concept. The system stitches together emotional processing, reasoning, memory management, and hormone-inspired motivation to simulate the rhythms of a persistent AI companion.

## 🌌 High-Level Goals
- Model interlocking emotional, hormonal, and motivational states.
- Maintain rich spatial memories that can be recalled for chat-oriented interactions.
- Provide inspection tooling (CLI + GUI) for exploring how the system reacts to stimuli.
- Serve as a playground for multimodal agent experiments.

## 🧠 Core Components
### Sector One – Emotions & Creativity
Handles the active emotional state, creative impulses, and the positive affect of the agent. The palette now spans well over three dozen feelings (joy, serenity, calm, awe, curiosity, anticipation, trust, love, empathy, compassion, gratitude, hope, confidence, motivation, resilience, inspiration, optimism, pride, focus, nurturing, contentment, relief, vigilance, fear, anxiety, caution, anger, frustration, sadness, loneliness, melancholy, disgust, tension, fatigue, boredom, envy, jealousy, shame, guilt, nostalgia, pessimism) so mood swings feel recognisably human. Hormone-driven modulation keeps these emotions grounded in physiological cues.

### Sector Two – Reasoning & Thought
Tracks reasoning metrics and produces vectorised representations of current thoughts. These vectors are combined each cycle for compact storage and downstream processing.

### Sector Three – Memories & Dreams
Stores long-term memories, manages an infinite spatial memory map, and generates dream fragments by blending recalled memories.

### Sector Four – Hormones & Drives *(newly overhauled)*
- Maintains detailed hormone profiles (dopamine, serotonin, norepinephrine, adrenaline, cortisol, oxytocin, vasopressin, endorphins, melatonin, testosterone, estrogen, progesterone, prolactin, acetylcholine, GABA, glutamate, histamine, thyroxine, ghrelin, leptin).
- Derives motivational drives (curiosity, self-preservation, reproduction, social bonding, achievement, calm restoration, playful connection, nurturing instinct, focused attention, appetite regulation, stress resilience, restorative drive) by blending hormone deltas.
- Analyses incoming stimuli and emotional feedback across rich affective tags (stress, novelty, social, comfort, threat, romance, loss, awe, inspiration, focus, hunger, satiety, restoration, fatigue) to keep the system balanced.
- Supplies modulation data back to Sector One, creating a closed emotional-hormonal feedback loop with explicit vigilance, soothing, fatigue, novelty, stability, nurturing, focus, metabolic, appetite, tension, and calm-depth signals.

### Support Modules
- **VISTA** – Visual interpretation of GPU frames, mouse focus, and OCR-like text signals.
- **ModelThinking** – LRM-style deliberation across emotional and sensory context.
- **DumbModel** – Simulated tool-use executor.
- **OtherControls** – Stubs for VR and agent control integrations.

## 🔄 Hormonal Feedback Highlights
- Structured `HormoneProfile` and `DriveProfile` classes encapsulate endocrine behaviour across twenty mood-shaping chemicals.
- Stimuli are interpreted into nuanced feature maps (stress, novelty, threat, comfort, romance, loss, awe, etc.) that influence hormones and drives.
- Emotional output feeds back into hormones to avoid runaway states and encourage stability over time.
- Sector One receives blended reward, calm, vigilance, soothing, fatigue, novelty, and stability tones to stay synchronised with bodily rhythms.
- Rich stimulus history tracking makes it easy to audit how the internal state evolved.

## 🚀 Getting Started
1. **Install dependencies** (requires Python 3.9+):
   ```bash
   pip install -r requirements.txt
   ```
   *(If `pip install -r requirements.txt` fails, install the packages individually. Linux users often need the `python3-tk` system package, e.g. `sudo apt install python3-tk`. Optional extras like `torch` or `pyautogui` can be added later if you need GPU or remote-control features.)*
2. **Run a demo cycle** to observe logging:
   ```bash
   python him_model.py
   ```
3. **Launch the GUI** (optional):
   ```bash
   python launch_him_gui.py
   ```
   *(The launcher now checks dependencies, prints Linux-friendly guidance, and will relaunch itself with the bundled `echo_luna` environment when available.)*

### ⚡ Optional Rust Accelerations
For faster serialisation of large vector memories you can build the optional
Rust extension located in `rust/echo_luna_rust` using
[maturin](https://www.maturin.rs/):

```bash
cd rust/echo_luna_rust
maturin develop
```

The Python runtime will automatically use the compiled module when available
and gracefully fall back to the pure Python implementation otherwise.

### 🔋 GPU Acceleration
Echo-Luna now auto-detects PyTorch accelerators for the numerically heavy pieces of the stack.

1. Install a GPU-enabled build of PyTorch that matches your drivers, for example:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```
   *(Pick the wheel that matches your CUDA/ROCm toolkit. CPU-only wheels work but will stay on the CPU.)*
2. Start the usual entry point (`python him_model.py`, `launch_him_gui.py`, etc.). The runtime announces the selected backend on startup.

When a GPU is available the following subsystems move onto the accelerator:
- VISTA image statistics and Sector Two vector consolidation
- Zipline pattern synthesis before permanent storage
- The streaming logistic trainer in `him_machine_learning.py`

Set `ECHO_LUNA_FORCE_CPU=1` if you ever need to fall back to pure CPU execution.


## 🧪 Experiment Ideas
- Feed different textual stimuli into `SectorFour.update_levels` and watch hormone/drive responses.
- Explore `SectorThree.save_vector_memories_to_text` to archive cognitive traces.
- Extend `VISTA` with real perception inputs to bridge physical/virtual environments.
- Train lightweight classifiers using `him_machine_learning.py` to convert H.I.M. state
  snapshots into ML-ready features without relying on transformer embeddings.
- Stream training data through `AsyncTrainingSession` to refine classifiers while the
  main H.I.M. runtime is still processing other tasks.

## 🤖 Concurrent Training & Runtime
`him_machine_learning.py` now exposes an `AsyncTrainingSession` helper that keeps a
background thread updating an `OnlineHIMLogisticTrainer`. This makes it possible to
continue running demos, the GUI, or other experiments while the classifier is still
learning from streaming data. A quick way to try it out is:

```bash
python him_machine_learning.py
```

The script first performs a synchronous fit and then spins up the asynchronous trainer,
printing interim predictions while updates arrive in the background.

## 🖥️ GUI Machine Learning Control
The main GUI exposes a dedicated **Machine Learning** tab so you can manage the
background trainer without leaving the interface:

- Start/stop the async logistic trainer and tweak learning-rate / L2 settings.
- Queue the sample dataset, tag recent cycles as positive/negative examples, or
  paste completely custom training text.
- Monitor update counts, queue depth, and the latest loss values in real time.
- Run on-the-fly predictions against the active trainer to sanity-check progress.

These controls wrap the same `AsyncTrainingSession` plumbing used in the CLI demo
so you can keep the cognitive simulation running while refining a classifier.

## 🕹️ GUI Remote Automation & Safety
Need to let H.I.M. react to the actual desktop? The **Remote Control** tab adds a
pyautogui-powered bridge (optional dependency) that can:

- Stream live screen captures straight into Sector One so runtime cycles reflect
  the real display instead of synthetic test frames.
- Move the mouse, click any button, scroll, type text, or press key combinations
  either through on-screen controls or chat slash commands such as `/mouse 640 360`.
- Keep an activity log so you always know which automation steps ran.

Press **Start Remote Session** once `pyautogui` is installed to activate the feed
and remember that **ESC** instantly halts the model loop *and* every automation
thread for safety. You can also toggle remote control on/off from chat with
`/remote start` or `/remote stop`.

## 🧬 Uploaded Persona Simulation *(new)*
To lean into the "human mind inside the machine" concept, Echo-Luna now ships with a
rich persona layer that wraps the core sectors:

- A `PersonaProfile` stores Luna's identity statement, biography, upload story, values,
  goals, skills, and conversational tone.
- The new `ConsciousnessKernel` keeps an autobiographical memory stream, inner monologue,
  and narrative log that update every cycle alongside emotions, drives, hormones, and dreams.
- Chat messages automatically feed the persona history, so conversations feel like a living
  presence continuing its own diary.

Open the **Persona Presence** tab inside the GUI to:

- Watch live mood readings, inner thoughts, recent autobiographical memories, and narrative
  log entries accumulate in real time.
- Edit the active persona's name, identity, biography, voice, and upload lore, then apply
  those updates instantly back into the running model.
- Save or load persona JSON files, making it easy to swap between different uploaded
  personalities.
- Log custom experiences with emotion and intensity sliders so the persona remembers key
  events even outside automated cycles.

These controls pair with new `HIMModel` helpers (`get_persona_snapshot`,
`update_persona_profile`, `ingest_persona_memory`, etc.) so scripts can orchestrate
persona changes without touching the GUI.

## 🗂️ Repository Guide
- `him_model.py` – Core cognitive architecture.
- `him_gui.py` / `launch_him_gui.py` – Desktop interface for observing cycles.
- `quick_luna_training.py` – Lightweight training hooks for Luna-specific behaviours.
- `him_luna_training/` – Example trained weights + metadata.
- `him_text_memories/` – Saved textual memory archives (generated at runtime).

## 🤝 Contributing
Issues and pull requests are welcome! Please include:
- Clear descriptions of behaviour changes.
- Relevant logs or screenshots.
- Tests or reproducible steps when possible.

---
*Echo-Luna is a research playground. Use responsibly and be mindful when integrating with real-world systems.*

