import json
import re

MALAYALAM_RE = re.compile(r"[ഀ-ൿ]")
REFUSAL_MARKER = "ക്ഷമിക്കണം"  # opens every Malayalam refusal response we wrote

with open("data/train_data.jsonl", "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f if line.strip()]

malayalam_facts = [
    p for p in lines
    if MALAYALAM_RE.search(p["instruction"]) and not p["response"].startswith(REFUSAL_MARKER)
]

print(f"Found {len(malayalam_facts)} Malayalam fact pairs to oversample.")

# Add 2 more copies of each (3x total exposure), matching the ~2-3x average repetition
# the English facts already get from prepare_data.py's simple+paraphrase passes.
with open("data/train_data.jsonl", "a", encoding="utf-8") as f:
    for _ in range(2):
        for p in malayalam_facts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"Dataset size now: {len(lines) + 2 * len(malayalam_facts)}")
