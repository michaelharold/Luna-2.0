# llm_ollama.py
"""
llm_ollama.py — Luna's generative brain, served by a local Ollama server.

WHY a server instead of the in-process llama.cpp (llm_local.py)?
  • Swapping models is `ollama pull <tag>` + one config line — no gguf hunting,
    no rebuilding llama-cpp-python for the Pi's CPU.
  • Ollama keeps the model resident across Luna restarts (keep_alive=-1), so a
    code change + restart never pays the multi-second model load again.
  • Its prefix KV-cache means the (long, static) system prompt — persona plus
    the whole knowledge base — is prefilled once per load, not per question.
  • Still 100% offline: the server runs ON the Pi, bound to 127.0.0.1.

Same five-function contract as llm_local.py (warmup / available / stream /
ask / shutdown) so brain.py and main.py never know which backend answered,
plus last_stats() for the benchmark and model_name()/set_model() so the
bench and tests can point this module at any pulled model.

Boot ordering: on the Pi, Luna and ollama.service come up together and nothing
guarantees the server is listening when main.py's warmup thread runs. So
warmup() keeps trying with backoff for a while, and available() — the ONE
hook brain.py calls before every question — kicks off a throttled background
re-warm whenever the backend is down. A slow service, a service restart or a
model pulled after boot is therefore picked up without restarting Luna.

Everything here is the standard library (urllib + json): the Pi's venv gains
no new dependency, and every failure prints "[llm] ..." and degrades to the
knowledge base instead of ever raising into the voice loop.
"""

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from config import (LLM_ENABLED, LLM_MAX_TOKENS, LLM_TEMPERATURE,
                    OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_BASE_MODEL,
                    OLLAMA_NUM_CTX, OLLAMA_NUM_THREAD, OLLAMA_KEEP_ALIVE,
                    OLLAMA_TIMEOUT, OLLAMA_AUTOSTART, OLLAMA_LOG)
import console

# Boot: while the server is merely unreachable (ollama.service still starting
# — the realistic systemd ordering on the Pi) warmup() retries with backoff
# (WARMUP_BOOT_BACKOFF seconds, doubling, capped at 30 s) for up to
# WARMUP_BOOT_WAIT seconds. It runs in main.py's daemon thread, so boot never
# waits on it, and the lock is released between attempts.
WARMUP_BOOT_WAIT = 120.0
WARMUP_BOOT_BACKOFF = 2.0

# After that, a down backend is re-tried from available() (in the background)
# or stream() (direct callers) — but no more often than this, so a dead server
# costs one cheap GET per WARMUP_RETRY_SECS at most rather than one per
# question.
WARMUP_RETRY_SECS = 30.0

# One opener with an EMPTY proxy map. A plain urlopen() honours http_proxy /
# HTTP_PROXY from the environment (and on Linux does not bypass 127.0.0.1
# unless no_proxy says so), so a campus proxy in the Pi user's shell would
# swallow the loopback call and Luna would silently run knowledge-base-only.
# Tests swap this object for a fake server.
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_sleep = time.sleep             # indirection so tests can skip the backoff

_lock = threading.Lock()        # serialises warmup + generation, like llm_local
_rewarm_guard = threading.Lock()  # protects the "spawn one re-warm" check
_rewarm_thread = None           # the background re-warm, if one is running
_model = None                   # resolved model name, once warmup succeeds
_model_override = None          # set_model(): bench/tests pick the model
_available = False
_supports_thinking = False      # send "think": false only when True
_last_attempt = 0.0             # time.monotonic() of the last warmup attempt
_server_proc = None             # the `ollama serve` Luna started, if any
_last_stats = {}                # final-chunk stats of the latest generation


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _request(path, payload=None, timeout=None):
    """Open an HTTP request to the Ollama server. GET when payload is None,
    JSON POST otherwise. Returns the (streaming) response object; the caller
    owns it. Raises on any network/HTTP failure — callers catch."""
    url = OLLAMA_HOST.rstrip("/") + path
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    return _opener.open(req, timeout=timeout or OLLAMA_TIMEOUT)


def _json(path, payload=None, timeout=None):
    """One non-streaming round trip, parsed."""
    with _request(path, payload, timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ── Model resolution ──────────────────────────────────────────────────────────

def _listed(name, names):
    """Ollama lists models as "name:tag"; a bare "luna" means "luna:latest"."""
    if name in names:
        return True
    return ":" not in name and f"{name}:latest" in names


def _resolve(names):
    """Pick the model to use from the server's list, or None with a printed
    reason. Order: an explicit set_model() override, else OLLAMA_MODEL (the
    custom "luna" build), else OLLAMA_BASE_MODEL with a hint to build it."""
    if _model_override:
        if _listed(_model_override, names):
            return _model_override
        print(f"[llm] Model '{_model_override}' is not on the Ollama server "
              f"(have: {', '.join(names) or 'nothing'})")
        return None
    if _listed(OLLAMA_MODEL, names):
        return OLLAMA_MODEL
    if _listed(OLLAMA_BASE_MODEL, names):
        print(f"[llm] Custom model '{OLLAMA_MODEL}' not found — using the "
              f"base '{OLLAMA_BASE_MODEL}'. Run scripts/make_luna_model.py "
              f"to bake the persona + knowledge in.")
        return OLLAMA_BASE_MODEL
    print(f"[llm] Neither '{OLLAMA_MODEL}' nor '{OLLAMA_BASE_MODEL}' is on "
          f"the Ollama server (have: {', '.join(names) or 'nothing'}). "
          f"Pull the base (`ollama pull {OLLAMA_BASE_MODEL}`) — "
          f"knowledge-base-only until then.")
    return None


def _capabilities(name):
    """Ask /api/show what the model can do. Only models that list "thinking"
    accept "think": false; every other model rejects the field outright."""
    try:
        info = _json("/api/show", {"model": name})
        return list(info.get("capabilities") or [])
    except Exception as e:
        print(f"[llm] /api/show failed for {name} ({e}) — assuming no "
              f"thinking support")
        return []


def _options():
    """Generation options — sent on EVERY /api/chat call, the warmup load
    included. Ollama's scheduler reloads the model whenever a runner option
    (num_ctx, num_thread, ...) differs from the resident instance's, so the
    load call must carry exactly what the real questions carry; otherwise the
    first answer after every boot pays the multi-second load a second time."""
    return {
        "num_ctx": OLLAMA_NUM_CTX,
        "num_thread": OLLAMA_NUM_THREAD,
        "temperature": LLM_TEMPERATURE,
        "num_predict": LLM_MAX_TOKENS,
    }


def _start_server():
    """Start `ollama serve` in the background, its output going to
    OLLAMA_LOG. Its runner logs every request at high verbosity, which
    buried Luna's own output when it shared her terminal (`ollama serve &`).
    Only for a local server, only once, only if the binary is installed. The
    server outlives Luna, so the next start finds the model still loaded.
    Returns True if a server was launched."""
    global _server_proc
    if not OLLAMA_AUTOSTART or _server_proc is not None:
        return False
    host = urllib.parse.urlparse(OLLAMA_HOST).hostname
    exe = shutil.which("ollama")
    if host not in ("127.0.0.1", "localhost") or not exe:
        return False
    path = OLLAMA_LOG if os.path.isabs(OLLAMA_LOG) else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), OLLAMA_LOG)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "ab") as log:
            _server_proc = subprocess.Popen(
                [exe, "serve"], stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
    except OSError as e:
        print(f"[llm] Could not start the Ollama server ({e})")
        return False
    print(f"[llm] Started the Ollama server (its logs: {path})")
    return True


def _warmup_locked():
    """The real warmup; caller holds _lock. Never raises. Returns None on
    success, otherwise a short reason: "disabled", "unreachable", "no-model"
    or "load" — warmup() retries only the first of those."""
    global _model, _available, _supports_thinking, _last_attempt
    if _available:
        return None
    _last_attempt = time.monotonic()
    if not LLM_ENABLED:
        print("[llm] LLM_ENABLED is False — knowledge-base-only.")
        return "disabled"
    try:
        tags = _json("/api/tags", timeout=min(OLLAMA_TIMEOUT, 10.0))
    except (urllib.error.URLError, OSError, ValueError) as e:
        if _start_server() or (_server_proc is not None
                               and _server_proc.poll() is None):
            console.detail(f"[llm] Waiting for the Ollama server to start ({e})")
            return "unreachable"         # warmup() retries while it comes up
        print(f"[llm] Ollama server not reachable at {OLLAMA_HOST} ({e}) — "
              f"knowledge-base-only. Is the ollama service running?")
        return "unreachable"
    names = [m.get("name", "") for m in tags.get("models", [])]
    name = _resolve(names)
    if not name:
        return "no-model"

    _supports_thinking = "thinking" in _capabilities(name)

    # Load the weights into RAM now (an empty chat is Ollama's documented
    # "just load it" call) so the first real question only pays generation.
    # It MUST carry the same options as a real question — see _options().
    try:
        t0 = time.monotonic()
        _json("/api/chat", {"model": name, "messages": [],
                            "keep_alive": OLLAMA_KEEP_ALIVE,
                            "options": _options()})
        console.detail(f"[llm] Ollama model {name} ready "
              f"({time.monotonic() - t0:.1f}s, thinking="
              f"{'yes' if _supports_thinking else 'no'})")
    except Exception as e:
        print(f"[llm] Could not load {name} into Ollama ({e}) — "
              f"knowledge-base-only.")
        return "load"
    _model = name
    _available = True
    return None


def _warmup_once():
    """One warmup attempt under the lock; returns _warmup_locked's reason."""
    with _lock:
        return _warmup_locked()


def _maybe_rewarm():
    """Start ONE background warmup attempt if none is running and the
    cooldown has elapsed; return immediately either way. Never when
    LLM_ENABLED is False — that is a decision, not an outage."""
    global _rewarm_thread
    if not LLM_ENABLED:
        return
    with _rewarm_guard:
        if _rewarm_thread is not None and _rewarm_thread.is_alive():
            return
        if time.monotonic() - _last_attempt < WARMUP_RETRY_SECS:
            return
        _rewarm_thread = threading.Thread(target=_warmup_once, daemon=True,
                                          name="llm-rewarm")
        _rewarm_thread.start()


# ── Public contract (mirrors llm_local) ───────────────────────────────────────

def set_model(name):
    """Point this module at a specific Ollama model (benchmark / tests).
    Resets availability, so call warmup() afterwards. None restores the
    OLLAMA_MODEL → OLLAMA_BASE_MODEL resolution."""
    global _model_override, _model, _available, _supports_thinking
    global _last_attempt, _last_stats
    with _lock:
        _model_override = name or None
        _model = None
        _available = False
        _supports_thinking = False
        _last_attempt = 0.0
        _last_stats = {}


def model_name():
    """The resolved model (None until warmup succeeds)."""
    return _model


def warmup():
    """Check the server, resolve the model, load it into RAM. Run from a
    background daemon thread at boot (main.py) exactly like the llama.cpp
    warmup. While the server is merely unreachable this keeps trying with
    backoff for WARMUP_BOOT_WAIT seconds; any other outcome — loaded,
    disabled, no model, load error — returns at once."""
    deadline = time.monotonic() + WARMUP_BOOT_WAIT
    delay = WARMUP_BOOT_BACKOFF
    while _warmup_once() == "unreachable":
        if time.monotonic() + delay > deadline:
            print("[llm] Still no Ollama server — giving up for now; will "
                  "retry in the background when a question comes in.")
            return
        console.detail(f"[llm] Retrying the Ollama server in {delay:.0f}s")
        _sleep(delay)
        delay = min(delay * 2, 30.0)


def available():
    """True only after a successful warmup. Never blocks on a load.

    brain.py calls this before every question and calls nothing else while
    the backend is down, so this is where a failed warmup gets another
    chance: one throttled attempt in a background thread (_maybe_rewarm).
    The question that triggered it still takes the knowledge-base path; the
    next one gets the LLM."""
    if not _available:
        _maybe_rewarm()
    return _available


def last_stats():
    """Stats from the final chunk of the latest generation (eval_count,
    eval_duration, prompt_eval_count, prompt_eval_duration, load_duration,
    total_duration — durations in ns), or {} if nothing has run yet."""
    return dict(_last_stats)


def _chat_payload(messages):
    payload = {
        "model": _model,
        "messages": list(messages),
        "stream": True,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": _options(),
    }
    if _supports_thinking:
        payload["think"] = False    # Luna speaks; she never shows her working
    return payload


_STAT_KEYS = ("eval_count", "eval_duration", "prompt_eval_count",
              "prompt_eval_duration", "load_duration", "total_duration")


def _record_stats(final):
    """Keep the timing fields off the final NDJSON line and print one
    readable line — the number the benchmark and the Pi tuning care about."""
    global _last_stats
    _last_stats = {k: final.get(k, 0) for k in _STAT_KEYS}
    n = _last_stats["eval_count"] or 0
    secs = (_last_stats["eval_duration"] or 0) / 1e9
    rate = n / secs if secs > 0 else 0.0
    console.detail(f"[llm] {n} tok in {secs:.1f}s ({rate:.1f} tok/s, "
          f"prefill {_last_stats['prompt_eval_count'] or 0} tok)")


def _iter_deltas(resp):
    """Yield content deltas from an NDJSON /api/chat stream; stop at done."""
    for raw in resp:
        raw = raw.strip()
        if not raw:
            continue
        try:
            chunk = json.loads(raw)
        except ValueError:
            continue
        if chunk.get("error"):
            print(f"[llm] Ollama error: {chunk['error']}")
            return
        delta = (chunk.get("message") or {}).get("content")
        if delta:
            yield delta
        if chunk.get("done"):
            _record_stats(chunk)
            return


def stream(messages):
    """Yield reply-text deltas for a chat `messages` list.

    Same contract as llm_local.stream: the first next() does real work, so a
    failure surfaces before anything is spoken and brain.py can still fall
    back to the knowledge base. The lock is held for the WHOLE generation —
    the consumer (chunked TTS) drains it promptly.
    """
    with _lock:
        if not _available:
            # Direct callers (bench, tests) that skip available(): one
            # synchronous, throttled retry — never hammer a dead server.
            if time.monotonic() - _last_attempt >= WARMUP_RETRY_SECS:
                _warmup_locked()
            if not _available:
                return
        try:
            with _request("/api/chat", _chat_payload(messages)) as resp:
                yield from _iter_deltas(resp)
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", "replace")[:200]
            except Exception:
                detail = ""
            print(f"[llm] Generation error: HTTP {e.code} {detail}")
        except Exception as e:
            print(f"[llm] Generation error: {e}")


def ask(messages):
    """Whole reply as one string, or None on failure."""
    parts = list(stream(messages))
    reply = "".join(parts).strip()
    return reply or None


def shutdown():
    """Deliberately a no-op. The model lives in the Ollama server, not in
    this process, and keep_alive=-1 keeps it resident — so a code change and
    `restart Luna` (or a crash) never pays the multi-second load again. To
    really evict it: `ollama stop <model>` or restart the ollama service."""
    return None
