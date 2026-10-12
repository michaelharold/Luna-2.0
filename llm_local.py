# llm_local.py
"""
llm_local.py — Luna's generative brain, running entirely ON the device.

This is the ONE module brain.py and main.py talk to (warmup / available /
stream / ask / shutdown). Which engine actually answers is config.LLM_BACKEND:

  "ollama"   — a local Ollama server (llm_ollama.py). Models are pulled once
               at install and stay resident across Luna restarts; swapping
               models is one config line; the static system prompt is
               prefix-cached. At import time this module simply re-exports
               llm_ollama's functions, so callers never change.
  "llamacpp" — the in-process llama.cpp path below: one .gguf file via
               llama-cpp-python (LLM_MODEL_PATH). No server, no daemon —
               the smallest possible moving part.

Either way nothing touches the network, and if the chosen engine is missing
`available()` stays False and brain.py degrades to knowledge-base-only
answers — the same graceful fallback the old code had for a missing API key.

llama.cpp path notes: the default model is Qwen2.5-1.5B-Instruct Q4_K_M
(~1 GB, ~8-12 tok/s on the Pi 5 CPU), but LLM_MODEL_PATH accepts any GGUF.
The model is loaded ONCE, in a background thread at boot (LLM_WARMUP), so
the several-second load never lands on the first question. Generation is
serialized with a lock — llama.cpp contexts are not thread-safe, and Luna
only ever answers one thing at a time anyway.
"""

import atexit
import os
import threading

from config import (LLM_BACKEND, LLM_ENABLED, LLM_MODEL_PATH, LLM_CTX,
                    LLM_THREADS, LLM_MAX_TOKENS, LLM_TEMPERATURE)
import console

_lock = threading.Lock()        # serializes load + generation
_llm = None
_failed = False


def _load():
    """Load the model (idempotent). Called under _lock."""
    global _llm, _failed
    if _llm is not None or _failed or not LLM_ENABLED:
        return
    if not os.path.exists(LLM_MODEL_PATH):
        _failed = True
        print(f"[llm] Model not found at {LLM_MODEL_PATH} — "
              f"running knowledge-base-only. Run scripts/download_models.sh.")
        return
    try:
        from llama_cpp import Llama
        console.detail(f"[llm] Loading {os.path.basename(LLM_MODEL_PATH)} "
              f"(ctx={LLM_CTX}, threads={LLM_THREADS})...")
        _llm = Llama(model_path=LLM_MODEL_PATH,
                     n_ctx=LLM_CTX,
                     n_threads=LLM_THREADS,
                     verbose=False)
        console.detail("[llm] Model ready")
    except Exception as e:
        _failed = True
        print(f"[llm] Could not load model ({e}) — knowledge-base-only.")


def warmup():
    """Load the model in the background at boot (started from main.py, like
    the TTS warmup) so the first real question only pays for generation."""
    with _lock:
        _load()


def available():
    """True once the model is loaded. Does NOT trigger a load — brain checks
    this to pick a path without blocking the voice loop behind a cold load."""
    return _llm is not None


def stream(messages):
    """Yield reply-text deltas for a chat `messages` list.

    Mirrors the old Groq streaming contract: the first next() performs real
    work, so a failure surfaces before anything has been spoken and the
    caller can still fall back to the knowledge base. The lock is held for
    the WHOLE generation — the consumer (chunked TTS) drains it promptly.
    """
    with _lock:
        _load()
        if _llm is None:
            return
        try:
            out = _llm.create_chat_completion(
                messages=messages,
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                stream=True,
            )
            for event in out:
                delta = event["choices"][0]["delta"].get("content")
                if delta:
                    yield delta
        except Exception as e:
            print(f"[llm] Generation error: {e}")


def ask(messages):
    """Whole reply as one string, or None on failure. For callers that don't
    want to deal with a stream (commands, tests)."""
    parts = list(stream(messages))
    reply = "".join(parts).strip()
    return reply or None


def shutdown():
    """Free the model explicitly. Left to the garbage collector at interpreter
    teardown, llama-cpp-python's __del__ runs after its own module globals are
    gone and spews a harmless-but-ugly TypeError — so close it while the world
    still exists (also registered via atexit below)."""
    global _llm
    with _lock:
        if _llm is not None:
            try:
                _llm.close()
            except Exception:
                pass
            _llm = None


def last_stats():
    """llama-cpp-python's stream carries no timing fields; the benchmark
    (scripts/bench_llm.py) times this path itself. Same name as the Ollama
    backend so callers are backend-agnostic."""
    return {}


def model_name():
    """The gguf file name, once loaded (None before / on failure)."""
    return os.path.basename(LLM_MODEL_PATH) if _llm is not None else None


def set_model(name):
    """The llama.cpp path has no runtime model switch — LLM_MODEL_PATH is
    fixed at import. Present so the bench/tests can call it on either
    backend without checking which one is live."""
    print(f"[llm] set_model({name!r}) ignored — the llama.cpp backend loads "
          f"LLM_MODEL_PATH; edit config.py to change it.")


# ── Backend dispatch ──────────────────────────────────────────────────────────
# On the Ollama path the names above are shadowed by llm_ollama's — brain.py
# and main.py keep calling llm_local.* and never learn the difference. The
# llama.cpp atexit is registered ONLY on the llama.cpp path: Ollama's model
# lives in the server and must NOT be torn down when Luna restarts.
if LLM_BACKEND == "ollama":
    try:
        from llm_ollama import (warmup, available, stream, ask,   # noqa: F811
                                shutdown, last_stats, model_name, set_model)
    except Exception as _e:   # stdlib-only module; this is belt and braces
        print(f"[llm] Ollama backend failed to import ({_e}) — falling back "
              f"to llama.cpp")
        atexit.register(shutdown)
else:
    atexit.register(shutdown)
