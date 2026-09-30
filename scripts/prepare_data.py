import json, re, requests, textwrap

MODEL = "qwen2.5:3b"  # already pulled locally; swap to "llama3.2:3b" if you `ollama pull` it

with open("data/dept.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = textwrap.wrap(text, 800)  # rough paragraph-sized chunks

# Two prompts: PASS_SIMPLE reliably extracts each distinct fact once (good coverage,
# clean structure). PASS_PARAPHRASE asks for repeated phrasings of the same facts (helps
# the model learn a fact isn't tied to one exact wording) but the small local model
# sometimes nests its answer instead of returning a flat list — extract_pairs() below
# handles both shapes so we don't silently lose data to a formatting slip.
PASS_SIMPLE = """From the text below, write one short Q&A pair for each distinct fact a visitor
might ask about the department.
Return ONLY a JSON list like [{{"instruction": "...", "response": "..."}}, ...]

Text:
{chunk}"""

PASS_PARAPHRASE = """From the text below, identify every distinct fact. For EACH fact, write 3 short
Q&A pairs that ask about it in different phrasings (vary the wording of both the question and the
answer each time — e.g. "When was X founded?" / "What year did X start?" / "How long has X existed?").
Return ONLY a flat JSON list like [{{"instruction": "...", "response": "..."}}, ...] — do not nest
pairs inside each other, every element of the list must have its own "instruction" and "response".

Text:
{chunk}"""


def extract_pairs(raw: str):
    """Parse the model's JSON reply into a flat list of {instruction, response} dicts,
    tolerating the nested shape the model sometimes produces (a list whose entries wrap
    another list of {"question": ..., "answer": ...} under "response")."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    data = json.loads(match.group(0) if match else cleaned)

    flat = []
    for item in data:
        if not isinstance(item, dict):
            continue
        resp = item.get("response")
        if isinstance(resp, str):
            flat.append({"instruction": item.get("instruction", ""), "response": resp})
        elif isinstance(resp, list):
            for sub in resp:
                if isinstance(sub, dict) and isinstance(sub.get("question"), str) and isinstance(sub.get("answer"), str):
                    flat.append({"instruction": sub["question"], "response": sub["answer"]})
    return flat


def run_pass(prompt_template):
    pairs = []
    for chunk in chunks:
        r = requests.post("http://localhost:11434/api/generate",
                           json={"model": MODEL, "prompt": prompt_template.format(chunk=chunk), "stream": False})
        body = r.json()
        if "response" not in body:
            print(f"skipped one chunk, Ollama error: {body.get('error', body)}")
            continue
        try:
            pairs.extend(extract_pairs(body["response"]))
        except Exception as e:
            print(f"skipped one chunk, output wasn't clean JSON ({e}): {body['response'][:200]!r}")
    return pairs

pairs = run_pass(PASS_SIMPLE) + run_pass(PASS_PARAPHRASE)

# De-dupe exact repeats while preserving order.
seen = set()
unique = []
for p in pairs:
    if not p.get("instruction") or not p.get("response"):
        continue
    key = (p["instruction"].strip().lower(), p["response"].strip().lower())
    if key not in seen:
        seen.add(key)
        unique.append(p)

with open("data/train_data.jsonl", "w", encoding="utf-8") as f:
    for p in unique:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"Generated {len(unique)} draft pairs — review data/train_data.jsonl before training.")
