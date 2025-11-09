#!/usr/bin/env python3
"""
Remove noisy '(generation unavailable...)' lines from sel_memory.json
and keep only the latest 6000 entries/vectors.
"""
import json
from pathlib import Path

mem_path = Path("sel_memory.json")
if not mem_path.exists():
    print("sel_memory.json not found here; place this script next to the file and run again.")
    raise SystemExit(1)

data = json.loads(mem_path.read_text(encoding="utf-8"))
entries = data.get("entries", [])
vectors = data.get("vectors", [])

def ok(entry):
    t = (entry.get("text") or "").strip().lower()
    if t.startswith("(generation unavailable") or t.startswith("(generation failed"):
        return False
    return True

filtered = [e for e in entries if ok(e)]
# Trim to last 6000 consistently with code
filtered = filtered[-6000:]
if isinstance(vectors, list) and len(vectors) >= len(entries):
    vectors = vectors[-len(filtered):]

out = {"entries": filtered, "vectors": vectors}
mem_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"Cleaned memory: {len(entries)} -> {len(filtered)} entries.")
