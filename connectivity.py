# connectivity.py
"""
connectivity.py — a cached "are we online?" answer that never blocks a turn.

Everything core to Luna (knowledge base, Ollama, Vosk, Piper) works with no
network at all. Two optional features need the internet at the moment they
run: the weather command (commands.py) and the Wikipedia fallback
(wiki_rag.py). Letting them "just try" is fine on a good connection, but on
the Pi's flaky Wi-Fi a half-up network makes every attempt stall — a stuck DNS
lookup alone can take several seconds, and urllib's timeout does not cover
it — and that stall lands in the middle of a conversation.

So the probe never runs on the voice path. state() answers instantly from the
last probe and, when that answer is older than PROBE_TTL, starts ONE
background re-probe. Callers decide what an unknown (None) means for them:
the Wikipedia fallback skips (it is optional context), the weather command
still tries (the person asked for it). A request that fails at the network
level calls mark_offline(), so an outage costs one failed attempt, not one per
question until the next probe.
"""

import socket
import threading
import time

# Probing by hostname (not an IP) on purpose: DNS failing while raw IP works
# is the most common half-up network, and a hostname probe catches it.
PROBE_HOST    = "en.wikipedia.org"
PROBE_PORT    = 443
PROBE_TIMEOUT = 2.0      # seconds for the TCP connect (background thread only)
PROBE_TTL     = 30.0     # seconds an answer is trusted before a re-probe

_lock       = threading.Lock()
_online     = None       # None = never probed
_checked_at = None       # time.monotonic() of the last answer
_thread     = None


def _probe():
    global _online, _checked_at
    try:
        with socket.create_connection((PROBE_HOST, PROBE_PORT), timeout=PROBE_TIMEOUT):
            ok = True
    except OSError:
        ok = False
    with _lock:
        _online, _checked_at = ok, time.monotonic()


def state():
    """True (online), False (offline) or None (no answer yet). Never blocks."""
    global _thread
    with _lock:
        stale = _checked_at is None or time.monotonic() - _checked_at >= PROBE_TTL
        if stale and (_thread is None or not _thread.is_alive()):
            _thread = threading.Thread(target=_probe, daemon=True, name="connectivity")
            _thread.start()
        return _online


def mark_offline():
    """A live request just failed at the network level — trust that until the
    next probe instead of making the next question pay the same timeout."""
    global _online, _checked_at
    with _lock:
        _online, _checked_at = False, time.monotonic()


def wait(timeout=None):
    """Block until a running probe finishes (tests only)."""
    t = _thread
    if t is not None:
        t.join(timeout)
