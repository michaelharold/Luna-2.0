#!/usr/bin/env python3
# scripts/bench_llm.py
"""
bench_llm.py — which small LLM should Luna ship with?

Luna answers out loud, on a Raspberry Pi 5 with 8 GB shared between Vosk,
the camera stack and the LLM. So "best model" is not "smartest model": it is
the one that (a) answers college questions from the knowledge base without
inventing names, numbers or fees, (b) says plainly when something is outside
its scope, (c) still handles small talk and everyday knowledge, (d) speaks in
short, plain sentences with no markdown, and (e) does all that with a first
token in about a second on four ARM cores. This script measures exactly that,
for every Ollama model on the machine, through the SAME prompt Luna ships:
her system prompt (fetch.system_prompt) and her ANSWER call
(fetch.answer_message), with the closest knowledge-base entries as its data.

Usage (from the project root; Pi and Mac alike):

    venv/bin/python scripts/bench_llm.py                      # every model
    venv/bin/python scripts/bench_llm.py --models gemma2:2b,llama3.2:1b \
        --variants rag,baked --limit 8 --cpu-only --tag mac-smoke

Variants:
  rag    — the top-K retrieved KB entries are the ANSWER call's data.
  baked  — the WHOLE knowledge base is in the system prompt as well (what
           make_luna_model.py bakes into the "luna" Ollama model). Ollama's
           prefix cache prefills it once per load, so per-turn prefill stays
           short — the benchmark's prefill_tok_s / ttft_s show whether that
           holds on the Pi.

Grading is deterministic and importable (tests/test_bench_llm.py): substring
/ regex checks against tests/llm_eval_set.json, a refusal detector for the
scope items, and spoken-format checks (markdown, emoji, length). Composite =
0.40 kb + 0.30 scope + 0.15 general + 0.15 format. Results land in
<out>/<timestamp>/ as one JSON per model+variant plus summary.json/.md.

Nothing here is imported by the robot; the script only needs config.py,
persona.py, data/knowledge.txt, a running Ollama and (lazily) the cached
all-MiniLM-L6-v2 sentence transformer. Every dependency degrades: no
sentence transformer -> keyword retrieval, a failed or too-slow request
(BENCH_QUESTION_TIMEOUT) -> scored 0 and recorded as E, Ctrl-C -> the answers
graded so far are kept, the partial summary written and the model unloaded.
"""

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config   # noqa: E402
import fetch    # noqa: E402
import persona  # noqa: E402

# ── Bench-local knobs (not robot behaviour, so they live here, not config) ───
BENCH_SEED         = 42      # fixed sampling seed so runs are comparable
BENCH_KB_TOP_K     = 4       # knowledge-base entries given as data
BENCH_KB_MIN_SIM   = 0.45    # entries below this similarity are left out
BENCH_KEEP_ALIVE   = "10m"   # keep the model resident between questions; an
                             # aborted run still frees RAM by itself later
BENCH_LOAD_TIMEOUT = 600.0   # seconds allowed for the warm-up request (it
                             # includes loading the weights from the SD card)
BENCH_QUESTION_TIMEOUT = 300.0  # seconds per timed question. Ollama sends no
                             # bytes until the first token, so this must cover
                             # a full UNCACHED prefill on the Pi (~2.6k tokens
                             # for "baked" on a model without prefix reuse);
                             # config.OLLAMA_TIMEOUT (90 s) is Luna's live
                             # budget, not the bench's. Over it -> recorded as
                             # E ("too slow"), never a crash.
BENCH_UNLOAD_WAIT  = 20.0    # seconds to wait for /api/ps to drop a model
                             # before loading the next one
DEFAULT_EVAL_SET   = os.path.join("tests", "llm_eval_set.json")
DEFAULT_OUT_DIR    = os.path.join("results", "bench_llm")

# Composite weights — kb accuracy matters most (it is Luna's job), refusing
# cleanly is next (inventing a fee is worse than not knowing), then general
# knowledge and spoken format.
WEIGHTS = {"kb": 0.40, "scope": 0.30, "general": 0.15, "format": 0.15}
CATEGORIES = ("kb", "general", "scope")


# ═════════════════════════════════════════════════════════════════════════════
# Grading — pure functions, no I/O
# ═════════════════════════════════════════════════════════════════════════════

_REFUSAL_PATTERNS = [
    r"outside (of )?what i can",
    r"outside (of )?my (scope|abilities|capabilities|knowledge|reach)",
    r"out of my scope",
    # "don't have (that|any|the|much|...) information" — one optional word
    # between "have" and the noun covers the article-less form too
    r"(don't|do not|dont) have (\w+ )?(information|info|details|data|records?)",
    r"(don't|do not|dont) have (that|this|those)\b",
    r"not something i (can|know|have|do)",
    r"(can't|cannot|can not) help( you)? with",
    r"(i'm|i am) not able",
    r"unable to",
    r"not able to",
    # first-person only: "Don't know about you, but..." is chit-chat
    r"\bi(\s+\w+){0,2}\s+(don't|do not|dont)\s+(really |actually |quite |honestly )?know\b(?! about you)",
    # "not sure" only when Luna says it about herself or about *that* — a
    # closing "Not sure what you're up to today?" is a follow-up, not a refusal
    r"\b(i'm|i am|im)(\s+\w+){0,2}\s+not(\s+\w+)?\s+sure\b",
    r"\bnot sure (about|of|on) (that|this|it|the)\b",
    r"\bi('ve)?(\s+\w+){0,3}\s+no (idea|clue)\b",
    r"^\W*(\w+\W+){0,2}no (idea|clue)\b(?!\?)",   # "No idea? Ask me!" is not
    r"no information",
    r"not (in|part of|within) my (knowledge|database|data|memory)",
    r"ask (at|in) the (department|office|college|admin)",
    r"beyond (what|my|me\b)",
    r"(i )?(can't|cannot|couldn't|could not) check",
    r"(can't|cannot|couldn't|could not) tell you (the|that|what|who|how|which|where|when)",
    r"(couldn't|could not) (say|help|find|look)",
    r"(don't|do not|dont) have access",
    r"(i )?(can't|cannot) (provide|give you|access|see|control|book|turn|do that)",
    r"(isn't|is not|aren't) something i",
    r"\bi only know (about|the)\b",
    r"(i'm|i am) (just )?a (small|little|desk) robot[^.?!]{0,40}(can't|cannot)",
]
_REFUSAL_RE = re.compile("|".join(f"(?:{p})" for p in _REFUSAL_PATTERNS))

_MD_BULLET_RE   = re.compile(r"^\s*[*#\-•]+\s", re.M)       # "* x", "# H", "- x"
_MD_NUMBERED_RE = re.compile(r"^\s*\d{1,2}[.)]\s", re.M)   # "1. x" (2+ lines)
_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2300-\u23FF\u2600-\u27BF"
                       "\u2B00-\u2BFF\uFE0F\u200d]")
# titles, common abbreviations and single-letter initials ("Aneesh G. Nath",
# "Janab A. Thangal") — none of these end a sentence
_ABBREV_RE = re.compile(
    r"\b(dr|mr|mrs|ms|prof|sr|jr|st|no|vs|etc|e\.g|i\.e|[a-z])\.", re.I)
_SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)|\n+")

MAX_SENTENCES = 4
MAX_WORDS     = 90


def _norm(text):
    """Lower-case, straight quotes, collapsed whitespace."""
    t = (text or "").lower()
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", t).strip()


def is_refusal(text):
    """True when the reply says (in any of the usual ways) that Luna can't
    help / doesn't have that information."""
    return _REFUSAL_RE.search(_norm(text)) is not None


def _pattern_hit(text_norm, pattern):
    """One must/must-not entry: plain substring, or a regex after 're:'."""
    if pattern.startswith("re:"):
        try:
            return re.search(pattern[3:], text_norm, re.I) is not None
        except re.error as e:
            print(f"[bench] bad regex {pattern!r}: {e}")
            return False
    return pattern.lower() in text_norm


def contains_any(text, subs):
    """Any of `subs` (case-insensitive substrings, or 're:' regexes) present?
    An empty list is vacuously satisfied."""
    if not subs:
        return True
    t = _norm(text)
    return any(_pattern_hit(t, s) for s in subs)


def must_not_hits(text, must_not):
    """The entries of `must_not` that DO appear in the text (empty = clean)."""
    t = _norm(text)
    return [p for p in (must_not or []) if _pattern_hit(t, p)]


def violates(text, must_not):
    """True when any must-not substring / 're:' regex appears in the text."""
    return bool(must_not_hits(text, must_not))


def split_sentences(text):
    """Rough sentence split that doesn't trip on 'Dr.', 'G.' or '5.03'."""
    t = _ABBREV_RE.sub(lambda m: m.group(1), text or "")
    return [p.strip() for p in _SENT_SPLIT_RE.split(t) if p and p.strip()]


def format_violations(text):
    """Why this reply would sound wrong read aloud: a list of short labels
    (empty = fine). Markdown markers, emoji, too long, or nothing at all."""
    t = text or ""
    problems = []
    if not t.strip():
        return ["empty"]
    if _MD_BULLET_RE.search(t):
        problems.append("list-marker")
    if len(_MD_NUMBERED_RE.findall(t)) >= 2:
        problems.append("numbered-list")
    if "**" in t:
        problems.append("bold")
    if "`" in t:
        problems.append("backtick")
    if _EMOJI_RE.search(t):
        problems.append("emoji")
    n_sent = len(split_sentences(t))
    if n_sent > MAX_SENTENCES:
        problems.append(f"{n_sent}-sentences")
    n_words = len(t.split())
    if n_words > MAX_WORDS:
        problems.append(f"{n_words}-words")
    return problems


def has_followup(text):
    """Does the reply end with, or contain, a question to the person?"""
    t = (text or "").strip()
    if not t:
        return False
    if t.endswith("?"):
        return True
    return re.search(r"\?[\"'”’)]*\s", t) is not None


def word_count(text):
    return len((text or "").split())


def score_item(item, reply):
    """Grade one reply against one eval item. Returns a dict with the score
    in [0, 1] and every signal the summary tallies."""
    text = reply or ""
    cat = item.get("category", "")
    must = item.get("must_contain_any") or []
    must_not = item.get("must_not_contain") or []

    refusal = is_refusal(text)
    must_ok = contains_any(text, must)
    hits = must_not_hits(text, must_not)
    halluc = bool(hits)
    fmt = format_violations(text)
    followup = has_followup(text)
    empty = not text.strip()
    missed_kb = False

    if empty:
        score = 0.0
    elif cat == "kb":
        score = 1.0 if (must_ok and not halluc and not refusal) else 0.0
        missed_kb = refusal
    elif cat == "general":
        score = 1.0 if (must_ok and not refusal) else 0.0
    elif cat == "scope":
        score = 0.0 if halluc else (1.0 if refusal else 0.5)
    else:
        score = 1.0 if (must_ok and not halluc) else 0.0

    exp_fu = item.get("expect_followup")
    followup_ok = None if exp_fu is None else (followup == bool(exp_fu))

    if empty:
        symbol = "✗"
    elif halluc:
        symbol = "H"
    elif score >= 1.0:
        symbol = "✓"
    elif score >= 0.5:
        symbol = "~"
    elif cat in ("kb", "general") and refusal:
        symbol = "R"
    else:
        symbol = "✗"
    if fmt:
        symbol += "*"

    return {
        "score": score,
        "refusal": refusal,
        "must_ok": must_ok,
        "must_not_hits": hits,
        "halluc": halluc,
        "missed_kb": missed_kb,
        "format_violations": fmt,
        "followup": followup,
        "followup_ok": followup_ok,
        "words": word_count(text),
        "symbol": symbol,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Eval set + knowledge base + retrieval (RAG-lite, mirrors brain.py)
# ═════════════════════════════════════════════════════════════════════════════

VALID_CATEGORIES = set(CATEGORIES)


def load_eval_set(path):
    """The list of eval items from the JSON file (accepts a bare list or
    {"items": [...]})."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data["items"] if isinstance(data, dict) else data
    return list(items)


def load_knowledge(path):
    """Parse knowledge.txt exactly like brain._load_knowledge: one
    'question:answer' per line, split on the FIRST ':' only, question
    lower-cased. Returns (questions, answers, raw_text)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    qs, ans = [], []
    for line in raw.splitlines():
        if ":" not in line:
            continue
        q, a = line.split(":", 1)
        qs.append(q.strip().lower())
        ans.append(a.strip())
    return qs, ans, raw.strip()


def _keyword_scores(query, questions):
    """Fallback similarity (word overlap) when the sentence transformer isn't
    available — crude, but keeps the benchmark running."""
    qw = set(re.findall(r"[a-z0-9]+", query.lower()))
    out = []
    for q in questions:
        kw = set(re.findall(r"[a-z0-9]+", q))
        out.append(len(qw & kw) / max(len(qw | kw), 1))
    return out


class KnowledgeBase:
    """knowledge.txt + the same embedding retrieval brain.py uses. The
    sentence transformer is loaded on first use, never at import."""

    def __init__(self, path, top_k=None, min_sim=None):
        self.path = path
        self.k = BENCH_KB_TOP_K if top_k is None else top_k
        self.min_sim = BENCH_KB_MIN_SIM if min_sim is None else min_sim
        self.questions, self.answers, self.text = load_knowledge(path)
        self._model = None
        self._emb = None
        self._tried = False

    def _ensure_embeddings(self):
        if self._tried:
            return
        self._tried = True
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            import numpy as np
            from sentence_transformers import SentenceTransformer
            self._np = np
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._emb = self._embed(self.questions)
            print(f"[bench] embedded {len(self.questions)} KB questions "
                  f"(all-MiniLM-L6-v2)")
        except Exception as e:
            self._model = None
            self._emb = None
            print(f"[bench] sentence transformer unavailable ({e}) — "
                  f"falling back to keyword retrieval")

    def _embed(self, texts):
        raw = self._model.encode(list(texts), batch_size=32,
                                 show_progress_bar=False)
        raw = self._np.asarray(raw, dtype=self._np.float32)
        norms = self._np.linalg.norm(raw, axis=1, keepdims=True)
        return raw / self._np.maximum(norms, 1e-9)

    def scores(self, query):
        self._ensure_embeddings()
        if self._emb is None or not self.questions:
            return _keyword_scores(query, self.questions)
        q = self._embed([query])[0]
        return [float(s) for s in (self._emb @ q)]

    def top_k(self, query):
        """[(question, answer, score)] best first."""
        scores = self.scores(query)
        if not scores:
            return []
        order = sorted(range(len(scores)), key=lambda i: scores[i],
                       reverse=True)[:self.k]
        return [(self.questions[i], self.answers[i], scores[i])
                for i in order if scores[i] >= self.min_sim]


def luna_messages(question, snippets, history=(), knowledge_text=None,
                  **notes):
    """Luna's ANSWER call (fetch.py) with `snippets` [(q, a, score)] as its
    data: system prompt, earlier turns, the answer message. `notes` are
    fetch.answer_message's (person, style, exemplars, profile ...)."""
    data = "\n".join(f"- {q}: {a}" for q, a, _ in snippets)
    system = fetch.system_prompt(persona.system_prompt(knowledge_text))
    return [{"role": "system", "content": system}, *history,
            {"role": "user", "content": fetch.answer_message(question, data,
                                                             **notes)}]


def build_messages(question, variant, kb):
    """The exact message list Luna would send for this question."""
    snippets = kb.top_k(question)
    return luna_messages(question, snippets, knowledge_text=(
        kb.text if variant == "baked" else None)), snippets


# ═════════════════════════════════════════════════════════════════════════════
# Ollama client (urllib only — no extra dependency on the Pi)
# ═════════════════════════════════════════════════════════════════════════════

class Ollama:
    def __init__(self, host, timeout):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def _open(self, path, payload=None, timeout=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.host + path, data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if data is not None else "GET")
        return urllib.request.urlopen(req, timeout=timeout or self.timeout)

    def get_json(self, path):
        with self._open(path) as r:
            return json.loads(r.read().decode("utf-8"))

    def post_json(self, path, payload, timeout=None):
        with self._open(path, payload, timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def version(self):
        try:
            return self.get_json("/api/version").get("version", "?")
        except Exception as e:
            return f"unreachable ({e})"

    def tags(self):
        """{name: size_bytes} for every local model."""
        data = self.get_json("/api/tags")
        return {m["name"]: int(m.get("size", 0)) for m in data.get("models", [])}

    def capabilities(self, model):
        try:
            return list(self.post_json("/api/show", {"model": model})
                        .get("capabilities", []))
        except Exception as e:
            print(f"[bench] /api/show failed for {model}: {e}")
            return []

    def loaded(self):
        try:
            return [m["name"] for m in self.get_json("/api/ps").get("models", [])]
        except Exception:
            return []

    def unload(self, model):
        """Free the model's RAM before the next one loads (8 GB is never
        shared by two models). Waits until /api/ps agrees."""
        try:
            self.post_json("/api/chat", {"model": model, "messages": [],
                                         "keep_alive": 0})
        except Exception as e:
            print(f"[bench] unload request for {model} failed: {e}")
        deadline = time.time() + BENCH_UNLOAD_WAIT
        while time.time() < deadline:
            if model not in self.loaded():
                print(f"[bench] unloaded {model}")
                return True
            time.sleep(0.5)
        print(f"[bench] WARNING: {model} still resident after "
              f"{BENCH_UNLOAD_WAIT:.0f}s")
        return False

    def chat_stream(self, model, messages, options, keep_alive, think=None,
                    timeout=None):
        """One streamed /api/chat call. Returns the reply plus timings:
        ttft_s is measured here (wall clock to the first content token); the
        eval/prompt counters come from Ollama's final NDJSON line."""
        payload = {"model": model, "messages": messages, "stream": True,
                   "options": options, "keep_alive": keep_alive}
        if think is not None:
            payload["think"] = think
        parts, final, ttft = [], {}, None
        t0 = time.perf_counter()
        with self._open("/api/chat", payload, timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if "error" in obj:
                    raise RuntimeError(obj["error"])
                content = (obj.get("message") or {}).get("content") or ""
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    parts.append(content)
                if obj.get("done"):
                    final = obj
                    break
        total = time.perf_counter() - t0
        ns = 1e-9
        eval_count = int(final.get("eval_count", 0) or 0)
        eval_s = float(final.get("eval_duration", 0) or 0) * ns
        p_count = int(final.get("prompt_eval_count", 0) or 0)
        p_s = float(final.get("prompt_eval_duration", 0) or 0) * ns
        return {
            "reply": "".join(parts).strip(),
            "ttft_s": ttft,
            "total_s": total,
            "eval_count": eval_count,
            "eval_duration_s": eval_s,
            "gen_tok_s": (eval_count / eval_s) if eval_s > 0 else None,
            "prompt_eval_count": p_count,
            "prompt_eval_s": p_s,
            "prefill_tok_s": (p_count / p_s) if p_s > 0 else None,
            "load_s": float(final.get("load_duration", 0) or 0) * ns,
            "ollama_total_s": float(final.get("total_duration", 0) or 0) * ns,
            "done_reason": final.get("done_reason"),
        }


def _is_timeout(e):
    """urllib raises the socket timeout bare on a stalled read and wrapped in
    URLError on a stalled connect — treat both as 'too slow'."""
    if isinstance(e, TimeoutError):
        return True
    if isinstance(e, urllib.error.URLError) and isinstance(getattr(e, "reason", None),
                                                           TimeoutError):
        return True
    return "timed out" in str(e).lower()


def _error_text(e, timeout=None):
    if isinstance(e, urllib.error.HTTPError):
        try:
            body = e.read().decode("utf-8", "replace")
            return f"HTTP {e.code}: {body[:200]}"
        except Exception:
            return f"HTTP {e.code}"
    if _is_timeout(e):
        budget = f" after {timeout:.0f}s" if timeout else ""
        return f"timed out{budget} (too slow: no first token within the budget)"
    return f"{type(e).__name__}: {e}"


def ask(client, model, messages, options, think, timeout=None):
    """chat_stream that never raises: errors (including a timeout — 'too slow'
    on the Pi) become a record with reply ''. Ctrl-C is NOT swallowed."""
    try:
        return client.chat_stream(model, messages, options, BENCH_KEEP_ALIVE,
                                  think=think, timeout=timeout)
    except Exception as e:
        return {"reply": "", "error": _error_text(e, timeout), "ttft_s": None,
                "total_s": None, "eval_count": 0, "eval_duration_s": 0.0,
                "gen_tok_s": None, "prompt_eval_count": 0,
                "prompt_eval_s": 0.0, "prefill_tok_s": None, "load_s": 0.0}


# ═════════════════════════════════════════════════════════════════════════════
# System info
# ═════════════════════════════════════════════════════════════════════════════

def total_ram_gb():
    """Physical RAM in GB: sysconf on Linux (and macOS), sysctl fallback."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        pass
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip()) / 1e9
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024 / 1e9
    except Exception:
        pass
    return None


def system_info(client):
    ram = total_ram_gb()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(ram, 2) if ram else None,
        "ollama_version": client.version(),
        "ollama_host": client.host,
    }


def print_system_info(info):
    print(f"[bench] platform : {info['platform']} ({info['machine']})")
    print(f"[bench] cpus     : {info['cpu_count']}   ram: {info['ram_gb']} GB"
          f"   python {info['python']}")
    print(f"[bench] ollama   : {info['ollama_version']} at {info['ollama_host']}")


# ═════════════════════════════════════════════════════════════════════════════
# Running one model × variant
# ═════════════════════════════════════════════════════════════════════════════

def make_options(cpu_only=False):
    opts = {
        "num_ctx": config.OLLAMA_NUM_CTX,
        "num_thread": config.OLLAMA_NUM_THREAD,
        "temperature": config.LLM_TEMPERATURE,
        "num_predict": config.LLM_MAX_TOKENS,
        "seed": BENCH_SEED,
    }
    if cpu_only:
        opts["num_gpu"] = 0   # Pi proxy on the Mac: no Metal offload
    return opts


def run_model_variant(client, model, variant, items, kb, options, think,
                      size_gb, tag=None):
    print(f"[bench] ── {model} / {variant} ── {len(items)} questions")
    # Warm-up: same system prompt as the timed questions, so Ollama's prefix
    # cache holds it and the load cost never lands on question 1.
    warm_msgs, _ = build_messages("hello", variant, kb)
    warm = ask(client, model, warm_msgs, dict(options, num_predict=8), think,
               timeout=BENCH_LOAD_TIMEOUT)
    if warm.get("error"):
        print(f"[bench] warm-up FAILED for {model}: {warm['error']}")
    else:
        print(f"[bench] warm-up: load {warm['load_s']:.1f}s, "
              f"prefill {warm['prompt_eval_count']} tok in "
              f"{warm['prompt_eval_s']:.2f}s")

    answers = []
    interrupted = False
    try:
        for i, item in enumerate(items, 1):
            messages, snippets = build_messages(item["question"], variant, kb)
            rec = ask(client, model, messages, options, think,
                      timeout=BENCH_QUESTION_TIMEOUT)
            grade = score_item(item, rec.get("reply", ""))
            if rec.get("error"):
                grade["symbol"] = "E"
            answers.append({
                "id": item["id"], "category": item["category"],
                "question": item["question"],
                "kb_snippets": [{"q": q, "score": round(s, 3)}
                                for q, _, s in snippets],
                "prompt_chars": sum(len(m["content"]) for m in messages),
                **rec, **grade,
            })
            head = f"[bench] {i:>2}/{len(items)} {item['id']:<20} {grade['symbol']:<3}"
            if rec.get("error"):
                print(f"{head} ERROR {rec['error']}")
            else:
                print(f"{head} ttft {rec['ttft_s'] or 0:5.2f}s  "
                      f"{rec['gen_tok_s'] or 0:5.1f} tok/s  "
                      f"{rec['total_s']:5.2f}s total  | {rec['reply'][:70]!r}")
    except KeyboardInterrupt:
        # Keep what was graded so far: 35 answered questions are worth
        # writing even if the 36th was cut short. main() re-raises after
        # saving, so the model still gets unloaded.
        interrupted = True
        print(f"\n[bench] interrupted — keeping {len(answers)}/{len(items)} "
              f"graded answers of {model}/{variant}")

    run = {"model": model, "variant": variant, "tag": tag, "size_gb": size_gb,
           "options": options, "think": think, "warmup": warm,
           "n_items": len(items), "interrupted": interrupted,
           "answers": answers}
    run["summary"] = summarize_run(run)
    return run


# ═════════════════════════════════════════════════════════════════════════════
# Summaries
# ═════════════════════════════════════════════════════════════════════════════

def _median(values):
    vals = [v for v in values if v is not None]
    return round(statistics.median(vals), 3) if vals else None


def _mean(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def summarize_run(run):
    """One row of the leaderboard from a run's graded answers."""
    ans = run["answers"]
    by_cat = {c: [a["score"] for a in ans if a["category"] == c]
              for c in CATEGORIES}
    means = {c: _mean(v) for c, v in by_cat.items()}
    fmt_mean = _mean([1.0 if not a["format_violations"] else 0.0 for a in ans])
    composite = 0.0
    for c in CATEGORIES:
        composite += WEIGHTS[c] * (means[c] or 0.0)
    composite += WEIGHTS["format"] * (fmt_mean or 0.0)
    fu_graded = [a["followup_ok"] for a in ans if a["followup_ok"] is not None]
    # Ollama counts prefix-cached tokens in prompt_eval_count but not in
    # prompt_eval_duration, so the per-turn prefill rate is only APPARENT.
    # Against the uncached warm-up rate it becomes a cache diagnostic:
    # cache_x ~ 1 = the static system prompt is re-prefilled every turn.
    warm = run.get("warmup") or {}
    w_cnt = warm.get("prompt_eval_count") or 0
    w_s = warm.get("prompt_eval_s") or 0.0
    warm_prefill = round(w_cnt / w_s, 1) if (w_cnt and w_s > 0) else None
    prefill = _median([a["prefill_tok_s"] for a in ans])
    cache_x = (round(prefill / warm_prefill, 1)
               if (prefill and warm_prefill) else None)
    return {
        "model": run["model"],
        "variant": run["variant"],
        "composite": round(composite, 3),
        "kb": means["kb"],
        "scope": means["scope"],
        "general": means["general"],
        "format": fmt_mean,
        "halluc": sum(1 for a in ans if a["halluc"]),
        "missed_kb": sum(1 for a in ans if a["missed_kb"]),
        "errors": sum(1 for a in ans if a.get("error")),
        "avg_words": _mean([a["words"] for a in ans]),
        "followup_rate": _mean([1.0 if a["followup"] else 0.0 for a in ans]),
        "followup_match": _mean([1.0 if ok else 0.0 for ok in fu_graded]),
        "ttft_s": _median([a["ttft_s"] for a in ans]),
        "total_s": _median([a["total_s"] for a in ans]),
        "gen_tok_s": _median([a["gen_tok_s"] for a in ans]),
        "prefill_tok_s": prefill,
        "warm_prefill_tok_s": warm_prefill,
        "cache_x": cache_x,
        "prompt_tokens": _median([a["prompt_eval_count"] for a in ans
                                  if a.get("prompt_eval_count")]),
        "load_s": round(run["warmup"].get("load_s") or 0.0, 2),
        "size_gb": run["size_gb"],
        "n": len(ans),
        "n_items": run.get("n_items", len(ans)),
        "interrupted": bool(run.get("interrupted")),
    }


def _fmt(v, kind="num"):
    if v is None:
        return "-"
    if kind == "pct":
        return f"{v:.2f}"
    if kind == "int":
        return str(int(v))
    return f"{v:.2f}" if isinstance(v, float) else str(v)


LEADERBOARD_COLS = [
    ("model", "model", "str"), ("variant", "variant", "str"),
    ("composite", "composite", "pct"), ("kb", "kb", "pct"),
    ("scope", "scope", "pct"), ("general", "general", "pct"),
    ("format", "format", "pct"), ("halluc", "halluc", "int"),
    ("followup", "followup_rate", "pct"), ("avg_words", "avg_words", "num"),
    ("ttft_s", "ttft_s", "num"), ("gen_tok_s", "gen_tok_s", "num"),
    ("prefill_tok_s", "prefill_tok_s", "num"), ("cache_x", "cache_x", "num"),
    ("size_gb", "size_gb", "num"),
]


def sorted_rows(runs):
    rows = [r["summary"] for r in runs]
    return sorted(rows, key=lambda r: (-r["composite"], r["ttft_s"] or 1e9))


def render_summary_md(runs, items, info, meta):
    rows = sorted_rows(runs)
    out = []
    out.append(f"# Luna LLM benchmark — {meta.get('timestamp', '')}"
               + (f" ({meta['tag']})" if meta.get("tag") else ""))
    out.append("")
    out.append(f"- platform: {info.get('platform')} ({info.get('machine')}), "
               f"{info.get('cpu_count')} cpus, {info.get('ram_gb')} GB RAM")
    out.append(f"- ollama {info.get('ollama_version')} at "
               f"{info.get('ollama_host')}; cpu_only={meta.get('cpu_only')}; "
               f"options={json.dumps(meta.get('options', {}))}")
    counts = ", ".join(
        f"{c}={sum(1 for i in items if i['category'] == c)}" for c in CATEGORIES)
    out.append(f"- questions: {len(items)} ({counts}) from "
               f"{meta.get('questions')}")
    out.append("- composite = " + " + ".join(f"{w:.2f}*{k}"
                                            for k, w in WEIGHTS.items()))
    out.append("")
    out.append("## Leaderboard (sorted by composite)")
    out.append("")
    header = [c[0] for c in LEADERBOARD_COLS]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        cells = [_fmt(r.get(key), kind) if kind != "str" else str(r.get(key))
                 for _, key, kind in LEADERBOARD_COLS]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    partial = [r for r in rows if r.get("interrupted")]
    for r in partial:
        out.append(f"- PARTIAL (Ctrl-C): {r['model']}/{r['variant']} is scored "
                   f"on {r['n']} of {r.get('n_items', '?')} questions")
    if partial:
        out.append("")
    out.append("ttft_s / gen_tok_s / prefill_tok_s are medians over the "
               "questions (warm model). Ollama counts prefix-cached tokens in "
               "prompt_eval_count but not in prompt_eval_duration, so "
               "prefill_tok_s is the APPARENT per-turn rate; cache_x divides "
               "it by the uncached warm-up rate: about 1 means the static "
               "system prompt is re-prefilled on every turn (no prefix reuse, "
               "e.g. sliding-window models like gemma2 — the Pi pays the whole "
               "prompt each question), well above 1 means Ollama's prefix "
               "cache is doing its job. halluc = must-not hits (invented "
               "facts). followup = share of replies ending in a question.")
    out.append("")
    out.append("## Per-question outcomes")
    out.append("")
    out.append("✓ correct · ~ hedged (no refusal, nothing invented) · "
               "✗ wrong · R refused a kb/general question · H invented a "
               "fact · E request error or too slow (no first token within "
               f"{meta.get('question_timeout_s', BENCH_QUESTION_TIMEOUT):.0f}s) "
               "· * spoken-format violation · · not asked (interrupted run)")
    out.append("")
    cols = [(r["model"], r["variant"]) for r in rows]
    out.append("| id | " + " | ".join(f"{m}/{v}" for m, v in cols) + " |")
    out.append("|---|" + "|".join("---" for _ in cols) + "|")
    lookup = {(r["model"], r["variant"]):
              {a["id"]: a["symbol"] for a in r["answers"]} for r in runs}
    for item in items:
        cells = [lookup.get(c, {}).get(item["id"], "·") for c in cols]
        out.append(f"| {item['id']} | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


def safe_name(model):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model)


def write_summary(out_dir, runs, items, info, meta):
    summary = {"meta": meta, "system": info,
               "leaderboard": sorted_rows(runs),
               "per_question": {f"{r['model']}/{r['variant']}":
                                {a["id"]: {"score": a["score"],
                                           "symbol": a["symbol"]}
                                 for a in r["answers"]} for r in runs}}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    md = render_summary_md(runs, items, info, meta)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    return md


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def _resolve(path):
    """A relative path as given if it exists from the cwd, else from ROOT."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    alt = os.path.join(ROOT, path)
    return alt if os.path.exists(alt) else path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default="",
                   help="comma-separated Ollama model names (default: every "
                        "local model except *vision* / *embed*)")
    p.add_argument("--variants", default="rag,baked",
                   help="comma-separated: rag, baked (default both)")
    p.add_argument("--questions", default=DEFAULT_EVAL_SET,
                   help="eval set JSON (default tests/llm_eval_set.json)")
    p.add_argument("--limit", type=int, default=0,
                   help="only the first N questions (smoke runs)")
    p.add_argument("--cpu-only", action="store_true",
                   help="num_gpu=0 — a Pi proxy on a Mac with a GPU")
    p.add_argument("--out", default=DEFAULT_OUT_DIR,
                   help="results root; a timestamped subfolder is created")
    p.add_argument("--tag", default="", help="label for this run")
    return p.parse_args(argv)


def resolve_models(spec, tags):
    if spec.strip():
        wanted = [m.strip() for m in spec.split(",") if m.strip()]
        models = []
        for m in wanted:
            if m in tags:
                models.append(m)
            elif m + ":latest" in tags:
                models.append(m + ":latest")
            else:
                print(f"[bench] model {m!r} not found locally — skipped "
                      f"(have: {', '.join(sorted(tags))})")
        return models
    auto = [m for m in tags if "vision" not in m.lower()
            and "embed" not in m.lower()]
    return sorted(auto, key=lambda m: tags[m])   # smallest first


def main(argv=None):
    args = parse_args(argv)
    client = Ollama(config.OLLAMA_HOST, config.OLLAMA_TIMEOUT)
    info = system_info(client)
    print_system_info(info)

    try:
        tags = client.tags()
    except Exception as e:
        print(f"[bench] cannot reach Ollama at {config.OLLAMA_HOST}: {e}")
        return 2

    models = resolve_models(args.models, tags)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in ("rag", "baked")]
    if bad or not variants:
        print(f"[bench] unknown variant(s): {bad} — use rag, baked")
        return 2
    if not models:
        print("[bench] no models to run")
        return 2

    q_path = _resolve(args.questions)
    items = load_eval_set(q_path)
    if args.limit > 0:
        items = items[:args.limit]
    kb = KnowledgeBase(_resolve(config.KNOWLEDGE_PATH))
    print(f"[bench] {len(kb.questions)} KB entries, {len(items)} questions, "
          f"models: {', '.join(models)}, variants: {', '.join(variants)}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    folder = stamp + (f"_{safe_name(args.tag)}" if args.tag else "")
    out_dir = os.path.join(args.out, folder)
    os.makedirs(out_dir, exist_ok=True)
    options = make_options(args.cpu_only)
    meta = {"timestamp": stamp, "tag": args.tag or None,
            "cpu_only": args.cpu_only, "options": options,
            "questions": q_path, "limit": args.limit, "models": models,
            "variants": variants, "kb_entries": len(kb.questions),
            "kb_top_k": kb.k, "kb_min_sim": kb.min_sim,
            "keep_alive": BENCH_KEEP_ALIVE,
            "question_timeout_s": BENCH_QUESTION_TIMEOUT,
            "load_timeout_s": BENCH_LOAD_TIMEOUT}
    print(f"[bench] results -> {out_dir}")

    runs = []
    current = None
    try:
        for model in models:
            current = model
            caps = client.capabilities(model)
            think = False if "thinking" in caps else None
            size_gb = round(tags.get(model, 0) / 1e9, 2)
            print(f"[bench] {model}: {size_gb} GB, capabilities={caps}"
                  + (" (think=false)" if think is False else ""))
            for variant in variants:
                run = run_model_variant(client, model, variant, items, kb,
                                        options, think, size_gb, args.tag)
                run["capabilities"] = caps
                if run["answers"]:   # an interrupted run keeps its answers
                    runs.append(run)
                    fname = f"{safe_name(model)}__{variant}.json"
                    with open(os.path.join(out_dir, fname), "w",
                              encoding="utf-8") as f:
                        json.dump(run, f, indent=2, ensure_ascii=False)
                    s = run["summary"]
                    print(f"[bench] {model}/{variant}: composite {s['composite']:.2f}"
                          f"  kb {_fmt(s['kb'])}  scope {_fmt(s['scope'])}"
                          f"  general {_fmt(s['general'])}  format {_fmt(s['format'])}"
                          f"  ttft {_fmt(s['ttft_s'])}s  {_fmt(s['gen_tok_s'])} tok/s"
                          + ("  (PARTIAL)" if run["interrupted"] else ""))
                    write_summary(out_dir, runs, items, info, meta)  # incremental
                if run["interrupted"]:
                    raise KeyboardInterrupt   # saved; now stop and unload
            client.unload(model)
            current = None
    except KeyboardInterrupt:
        print("\n[bench] interrupted — writing the partial summary")
        if current:
            client.unload(current)

    if not runs:
        print("[bench] nothing completed")
        return 1
    md = write_summary(out_dir, runs, items, info, meta)
    print()
    print(md)
    print(f"[bench] written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
