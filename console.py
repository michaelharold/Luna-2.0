# console.py
"""
console.py — what Luna prints to the terminal.

Two levels, chosen by config.LOG_VERBOSE:

  clean (False, the default) — one line per thing a person cares about:
        [heard]  what she heard (and whether she ignored it)
        [source] where the answer came from: website pages, knowledge-base
                 entries, the camera, or general knowledge
        [Luna]   what she said
        plus who she recognised, greetings, and anything that went wrong
  verbose (True) — everything above plus the internals: speech-recognition
        scores, LLM speed, retrieval routes, timings, servo moves, start-up
        chatter, and the libraries' own logs (Vosk, MediaPipe, ...)

Plain print() is still right for the clean lines and for warnings / errors;
call detail() for anything that is only useful when debugging.
"""

import os

from config import LOG_VERBOSE


def detail(*args, **kwargs):
    """print(), but only in verbose mode."""
    if LOG_VERBOSE:
        print(*args, **kwargs)


def heard(text, ignored=None):
    """What Luna heard. `ignored` is a short reason when she let it pass."""
    if ignored:
        print(f"[heard] {text}   (ignored — {ignored})")
    else:
        print(f"[heard] {text}")


def _short_url(url):
    url = str(url).split("#", 1)[0]
    for prefix in ("https://", "http://"):
        if url.startswith(prefix):
            url = url[len(prefix):]
    if url.startswith("www."):
        url = url[4:]
    return url.rstrip("/")


def source(web=(), kb=(), kind=None):
    """Where an answer came from — the pages and knowledge-base entries that
    were put in front of the model (or answered directly)."""
    parts = []
    pages = []
    for url in web or ():
        short = _short_url(url)
        if short and short not in pages:
            pages.append(short)
    if pages:
        parts.append("website — " + ", ".join(pages[:4]))
    entries = [str(q) for q in (kb or ()) if q]
    if entries:
        parts.append("knowledge base — "
                     + ", ".join(f'"{q}"' for q in entries[:3]))
    if kind:
        parts.insert(0, kind)
    if not parts:
        parts.append("general knowledge (nothing matched in the knowledge "
                     "base or website)")
    print("[source] " + " · ".join(parts))


def quiet_libraries():
    """Turn down the libraries that write to the terminal on their own.
    Must run before they are imported (main.py does it first thing)."""
    if LOG_VERBOSE:
        return
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")     # TensorFlow Lite
    os.environ.setdefault("GLOG_minloglevel", "2")         # MediaPipe / glog
    try:
        import vosk
        vosk.SetLogLevel(-1)
    except Exception:                                       # noqa: BLE001
        pass
