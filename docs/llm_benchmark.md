# Choosing Luna's LLM — benchmark results

Run with `scripts/bench_llm.py` on 2026-09-23/24 against the 40-question
evaluation set in `tests/llm_eval_set.json`:

- **16 knowledge questions**, paraphrased away from the wording in `data/knowledge.txt`
- **10 general questions** and small talk
- **14 out-of-scope requests**, where the right answer is to decline without inventing anything

Composite = 0.40 × knowledge + 0.30 × scope + 0.15 × general + 0.15 × spoken format.
"Invented" counts answers that made up a fact (a name, a number, a fee).

Two prompt variants were tested:

- **rag**: persona in the system prompt, plus the top retrieved knowledge entries each turn.
- **baked**: the whole knowledge file in the system prompt, plus retrieval.

## Quality (Mac M3, full 40 questions)

| Model | Variant | Composite | Knowledge | Scope | General | Invented | Size |
|---|---|---|---|---|---|---|---|
| **llama3.2:3b** | **rag** | **0.97** | 1.00 | 0.96 | 0.90 | 0 | 2.0 GB |
| gemma3:4b | baked | 0.93 | 1.00 | 0.82 | 0.90 | 2 | 3.3 GB |
| llama3.2:3b | baked | 0.91 | 1.00 | 1.00 | 0.50 | 0 | 2.0 GB |
| qwen2.5:3b | baked | 0.89 | 1.00 | 0.64 | 1.00 | 2 | 1.9 GB |
| gemma2:2b | rag | 0.86 | 0.94 | 0.71 | 1.00 | 0 | 1.6 GB |
| llama3.1:8b | baked | 0.85 | 0.88 | 0.93 | 0.70 | 0 | 4.9 GB |
| qwen2.5:1.5b | rag | 0.79 | 0.88 | 0.50 | 0.90 | 3 | 1.0 GB |
| llama3.2:1b | rag | 0.68 | 0.75 | 0.46 | 0.70 | 6 | 1.3 GB |

## Speed (Mac CPU only, 4 threads — a stand-in for the Pi)

| Model | Variant | First token | Tokens/s | Estimated on Pi 5 |
|---|---|---|---|---|
| qwen2.5:1.5b | rag | 0.67 s | 57 | ~3 s first token, ~12 tok/s |
| llama3.2:1b | rag | 0.52 s | 46 | ~2.5 s, ~10 tok/s |
| **llama3.2:3b** | **rag** | **1.02 s** | **28** | **~5 s, ~5-6 tok/s** |
| qwen2.5:3b | rag | 1.03 s | 31 | ~5 s, ~6 tok/s |
| gemma3:4b | baked | 2.11 s | 20 | ~10 s, ~4 tok/s |
| gemma2:2b | rag | 2.80 s | 34 | no prefix caching, slow on every turn |

The Pi column is an estimate: roughly 4-5× the Mac CPU numbers. It has not
been measured, because the Pi was offline during the benchmark. Re-run on the Pi with:

```bash
venv/bin/python scripts/bench_llm.py --models luna,qwen2.5:1.5b --variants rag
```

## Decision

**Default: `llama3.2:3b` with retrieval only (`OLLAMA_BAKE_KNOWLEDGE = False`).**

- It is the only model that scored perfectly on knowledge, near-perfectly on declining out-of-scope requests, and invented nothing.
- Baking the whole knowledge file in made it refuse general questions.
- Streaming TTS starts speaking after the first sentence, which hides most of the generation time.

**If the Pi feels too slow:** set `OLLAMA_BASE_MODEL = "qwen2.5:1.5b"` and re-run
`scripts/make_luna_model.py`. It is about twice as fast but weaker at declining
out-of-scope requests.

**Avoid:**

- `llama3.2:1b` invented facts.
- `gemma2:2b` gets no prefix-cache reuse, so it pays the full prompt on every question.
- `gemma3:4b` is too slow for live conversation on the Pi.

Raw results: `results/bench_llm/` (not committed; regenerate with the script).
