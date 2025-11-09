# Unlimited Natural Memory System

## Overview

The memory system now works like a real human brain:
- **Unlimited capacity** - No arbitrary limits per user
- **Natural forgetting** - Memories fade based on importance, usage, and time
- **Global memory pool** - All memories in one place, like a human brain
- **Automatic importance** - Recognizes what matters without being told
- **Dynamic retrieval** - Finds relevant memories across all users when needed

## How It Works

### 1. Memory Storage (Unlimited)

**No per-user limits:**
```python
# Before: Limited to 5,000 memories per user
# After: Unlimited - grows naturally like human memory
```

**Automatic importance detection:**
```python
"My name is Sarah" → importance: 0.9 (very high)
"I feel anxious" → importance: 0.7 (high)
"I usually wake up early" → importance: 0.6 (medium)
"Nice weather today" → importance: 0.5 (standard)
```

### 2. Natural Forgetting

Memories aren't deleted on a timer. They fade naturally based on:

**Memory Strength Formula:**
```
Strength = Importance + Access Boost - Time Decay

Components:
- Importance (0.0-1.0): How meaningful the memory is
- Access Boost (0.0-0.3): How often it's been recalled
- Time Decay: How long since last use
```

**Decay Rates (Human-like forgetting curve):**
- **Very important (0.8+)**: 90 days to 50% strength
- **Moderate (0.6-0.8)**: 45 days to 50% strength
- **Normal (<0.6)**: 20 days to 50% strength

**Reinforcement:**
- Every time a memory is recalled, it's strengthened
- `access_count` increases
- `last_accessed` updates
- Frequently used memories stick around longer

### 3. Forgetting Threshold

Memories are kept if `strength > 0.2-0.3` (random factor)

This mimics human memory:
- Sometimes you remember
- Sometimes you forget
- Depends on many factors

### 4. Practical Example

```
Day 0: User says "My favorite color is blue"
       → Stored with importance 0.7

Day 10: Bot recalls it in conversation
       → access_count += 1, last_accessed updated
       → Strength boosted

Day 30: Not accessed recently
       → Strength = 0.7 + 0.05 - (30/45) = 0.08
       → Fading but still above threshold

Day 60: Still not used
       → Strength = 0.7 + 0.05 - (60/45) = -0.58
       → Below threshold → forgotten

BUT if accessed on day 40:
       → Reinforced, stays strong
       → Like practicing a skill
```

## Memory Importance Auto-Detection

### Very High Importance (0.9) - Permanent
```
"My name is..."
"I am a..."
"I'm a..."
"I love..."
"I hate..."
"My birthday is..."
"My family..."
```

### High Importance (0.7) - Long retention
```
"I feel..."
"I think..."
"I believe..."
"I wish..."
"I hope..."
"I prefer..."
"favorite"
```

### Medium Importance (0.6) - Moderate retention
```
"I always..."
"I never..."
"I usually..."
"Every time..."
"I often..."
```

### Standard (0.5) - Normal conversation
Everything else that doesn't match above patterns

### Emotional Boost (+0.1)
Words like: happy, sad, angry, excited, worried, scared, love, hate

### Question Boost (+0.05)
Messages ending with "?"

## Global Memory Pool

**How it works:**

```
Before (Per-User):
{
  "user123": [memory1, memory2, ...],
  "user456": [memory3, memory4, ...],
}

After (Global Pool):
[
  {user_id: "user123", text: "...", importance: 0.9, ...},
  {user_id: "user456", text: "...", importance: 0.7, ...},
  {user_id: "user123", text: "...", importance: 0.5, ...},
  ...
]
```

**Benefits:**
- No artificial limits
- More human-like structure
- Can recall cross-user patterns
- Natural memory associations

## Memory Retrieval (Smart)

When searching for memories:

**Scoring system:**
```python
score = semantic_similarity  # Vector similarity to query
      + user_boost           # +0.3 if same user
      + importance_boost     # +0.2 * importance
      + recency_boost        # +0.2 decaying over time
```

**Example:**
```
Query: "What music do I like?"

Memories found:
1. "Sarah loves jazz music" (user: sarah, importance: 0.9)
   Score: 0.95 (semantic) + 0.3 (same user) + 0.18 (importance) + 0.15 (recent)
   = 1.58 ✓ Top result

2. "Someone mentioned jazz yesterday" (user: other, importance: 0.5)
   Score: 0.80 (semantic) + 0.0 (different user) + 0.10 (importance) + 0.18 (very recent)
   = 1.08 ✓ Also returned

3. "Classical music concert" (user: sarah, importance: 0.6)
   Score: 0.60 (semantic) + 0.3 (same user) + 0.12 (importance) + 0.05 (old)
   = 1.07 ✓ Might be returned
```

## Memory Statistics

Track memory health with `get_memory_stats()`:

```python
{
  "total_memories": 15420,
  "unique_users": 47,
  "avg_importance": 0.63,
  "oldest_days": 89.3,
  "newest_days": 0.1,
  "avg_age_days": 23.7
}
```

## Periodic Cleanup

**Automatic forgetting runs:**
- Every 100 new memories
- On every `search()` call
- Maintains memory health naturally

**Not aggressive:**
- Doesn't delete everything at once
- Gradual forgetting like humans
- Important memories persist indefinitely

## Comparison: Old vs New

| Feature | Old System | New System |
|---------|-----------|------------|
| Capacity | 5,000 per user | Unlimited (natural forgetting) |
| Storage | Per-user buckets | Global memory pool |
| Forgetting | Fixed 30-day TTL | Dynamic based on use |
| Importance | Manual | Auto-detected |
| Reinforcement | None | Access count strengthens |
| Cross-user | Separated | Can recall any memory |
| Structure | Artificial limits | Brain-like |

## Configuration

### Adjust Forgetting Speed

In `sel_enhanced.py`, modify decay rates:

```python
# Slower forgetting (keep memories longer)
if m.importance >= 0.8:
    time_penalty = recency_days / 180  # 180 days instead of 90

# Faster forgetting (more aggressive cleanup)
if m.importance >= 0.8:
    time_penalty = recency_days / 45  # 45 days instead of 90
```

### Adjust Forget Threshold

```python
# Keep more memories (remember more)
forget_threshold = 0.1 + (random.random() * 0.1)  # 0.1-0.2

# Keep fewer memories (forget more)
forget_threshold = 0.3 + (random.random() * 0.1)  # 0.3-0.4
```

### Disable Forgetting (If Desired)

```python
def _natural_forgetting(self):
    """Disable forgetting - keep everything"""
    return  # Do nothing
```

## Migration

Old memories automatically migrate to new format:

```python
def _migrate_entry(self, old_entry: dict) -> MemoryEntry:
    return MemoryEntry(
        user_id=old_entry.get("user_id"),
        text=old_entry.get("text"),
        ts=old_entry.get("ts"),
        importance=0.9 if old_entry.get("kind") == "hard" else 0.5,
        access_count=0,
        last_accessed=old_entry.get("ts"),
        vector=old_entry.get("vector"),
        tags=old_entry.get("tags"),
        context="",
    )
```

## Privacy & Security

**Still protected:**
- Never stores passwords, API keys, tokens
- Sensitive data filtering active
- Each user's memories tagged with user_id
- Retrieval prioritizes same-user memories

**Pattern matching for sensitive data:**
```python
if re.search(r"\b(password|token|secret|api[_\s]?key|credit\s*card)\b", text):
    return  # Don't store
```

## Commands

### CLI Mode
```bash
/memory  # Show memory statistics
```

### Discord Bot
```bash
!stats   # Includes memory stats
!memsvg  # Your personal memory map
```

## Examples

### Natural Memory Formation

**Conversation 1:**
```
User: My name is Alex
Bot: Nice to meet you, Alex!
Memory: [importance: 0.9, access_count: 0]
```

**Conversation 50:**
```
User: What's my name again?
Bot: You're Alex!
Memory: [importance: 0.9, access_count: 1, reinforced]
```

**Conversation 200:**
```
Bot internally recalls: "Alex likes Python"
Memory: [importance: 0.7, access_count: 3, well-established]
```

### Natural Forgetting

**Unimportant memory:**
```
Day 1: "Nice weather today" (importance: 0.5)
Day 20: Strength = 0.5 - (20/20) = -0.5 → Forgotten
```

**Important memory:**
```
Day 1: "My birthday is March 15" (importance: 0.9)
Day 90: Strength = 0.9 - (90/90) = 0.0 → Still above threshold
Day 180: Never used → Eventually forgotten
Day 180: BUT if accessed even once → Strength restored → Persists
```

### Cross-User Memory

```
User A: "I love rock music"
User B: "What kind of music does Alex like?"
Bot: *Searches globally, finds User A's preference*
Bot: "Alex mentioned loving rock music!"
```

## Philosophy

This system mimics human memory:

1. **No arbitrary limits** - Your brain doesn't have a "5,000 memory limit"
2. **Use it or lose it** - Memories fade without rehearsal
3. **Importance matters** - Meaningful events stick
4. **Reinforcement** - Recalling strengthens memories
5. **Gradual forgetting** - Not instant deletion
6. **Imperfect** - Sometimes remember, sometimes forget
7. **Associations** - Can recall related memories across contexts

The goal is authentic human-like memory, not a database.

## Performance

**Efficient even with large memory:**
- Vector search is O(n) but optimized
- Periodic cleanup prevents bloat
- Embeddings cached in memory
- Natural forgetting keeps size reasonable

**Expected memory size:**
- Active bot: 10,000-50,000 memories
- Natural forgetting maintains equilibrium
- Very important memories accumulate over time

## Future Enhancements

Possible additions:
- **Memory consolidation**: Merge similar memories
- **Sleep cycles**: Periodic "sleep" to consolidate
- **Emotional tagging**: Remember emotional moments better
- **Spatial memory**: Remember where/when things happened
- **Associative chains**: Follow memory connections
- **Forgetting curves**: Per-user forgetting rates

The foundation is now in place for truly brain-like memory.
