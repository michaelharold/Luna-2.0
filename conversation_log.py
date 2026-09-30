# conversation_log.py
"""
conversation_log.py — everything Luna heard and said, one JSON line per turn.

WHY: tuning a robot that lives on a desk is guesswork without a record. The
addressing gate ignored a real question? The KB answered with the wrong
entry? The LLM took nine seconds? Each of those is a line in today's log,
with the numbers that explain it (Vosk confidence, loudness, the addressing
breakdown, the KB matches, the timings). The log is ALSO the training data
for learning.py: rewards that arrive after a turn are appended to it.

Format: data/logs/conversation_YYYY-MM-DD.jsonl (local day), append-only.
  • a turn:    {"turn_id", "ts", "mode", "heard_raw", "heard_clean", "conf",
                "loudness", "addressed", "addr_reason", "person", "path",
                "kb_matches", "web_matches", "wiki_matches", "reply", "timings",
                "style_arm", "context", "reward", "reward_signals"}
  • an update: {"update": turn_id, "ts", ...changed fields} — never rewrite a
                line; iter_day() merges updates into their turn.
  • ignored:   a turn record with path "ignored", addressed False and the
                gate's reason — an ignored question is as fixable as a wrong
                answer.

Rules: append-only, one lock, and NOTHING here may raise into a voice turn —
the first failure of each kind is printed once and swallowed. Old files are
pruned at import (CONVO_LOG_RETENTION_DAYS).

Public API
  configure(**kw)          enabled / log_dir / retention_days / mode
  new_turn(**fields)       -> turn_id
  update_turn(turn_id, **fields)
  log_ignored(text, conf, loudness, reason)
  note_heard(text, conf, loudness, reason)   (optional: STT's accept path)
  last_turn()              -> dict | None (latest new_turn, merged updates)
  iter_day(date)           -> [turn dicts] with updates merged
  prune(days)              -> number of files removed
"""

import datetime as _dt
import itertools
import json
import os
import threading
import time

try:
    from config import (CONVO_LOG_ENABLED, CONVO_LOG_DIR,
                        CONVO_LOG_RETENTION_DAYS, IS_MAC)
except Exception:                                   # config must never be fatal
    CONVO_LOG_ENABLED, CONVO_LOG_DIR = True, "data/logs"
    CONVO_LOG_RETENTION_DAYS, IS_MAC = 45, False

_ROOT = os.path.dirname(os.path.abspath(__file__))

# Fields every turn record carries (defaults), in the order they are written.
_DEFAULTS = (
    ("heard_raw", None), ("heard_clean", None), ("conf", None),
    ("loudness", None), ("addressed", True), ("addr_reason", None),
    ("person", None), ("path", None), ("kb_matches", []),
    ("web_matches", []), ("wiki_matches", []), ("reply", None), ("timings", {}),
    ("style_arm", None), ("context", None), ("reward", None),
    ("reward_signals", []),
)
PATHS = ("command", "vision", "kb", "llm", "fallback", "ignored", "wake")

# A note_heard() entry only describes the utterance brain is about to log if
# it is this fresh and the text matches.
HEARD_FRESH_SECS = 30.0

_lock = threading.Lock()
_cfg = {
    "enabled": CONVO_LOG_ENABLED,
    "log_dir": CONVO_LOG_DIR,
    "retention_days": CONVO_LOG_RETENTION_DAYS,
    "mode": "mac" if IS_MAC else "pi",
}
_counter = itertools.count(1)
_last = None                  # latest new_turn record (a merged copy)
_heard = None                 # latest note_heard() entry
_errors_printed = set()


def _print_once(where, exc):
    key = (where, type(exc).__name__)
    if key in _errors_printed:
        return
    _errors_printed.add(key)
    print(f"[convlog] {where} failed ({type(exc).__name__}: {exc}) — "
          f"ignoring further errors of this kind")


def _dir():
    d = _cfg["log_dir"]
    return d if os.path.isabs(d) else os.path.join(_ROOT, d)


def _day(ts):
    return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def path_for(date):
    """Log file for a date ('YYYY-MM-DD', a date/datetime, or a timestamp)."""
    if isinstance(date, (int, float)):
        date = _day(date)
    elif isinstance(date, (_dt.date, _dt.datetime)):
        date = date.strftime("%Y-%m-%d")
    return os.path.join(_dir(), f"conversation_{date}.jsonl")


def _append(record, ts):
    """One JSON line to the day file of `ts`. Caller holds _lock."""
    if not _cfg["enabled"]:
        return
    try:
        os.makedirs(_dir(), exist_ok=True)
        with open(path_for(ts), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as e:                                   # noqa: BLE001
        _print_once("write", e)


def configure(**kw):
    """Override settings (tests, simulations): enabled, log_dir,
    retention_days, mode. Also forgets the in-memory last turn."""
    global _last, _heard
    with _lock:
        for k, v in kw.items():
            if k in _cfg:
                _cfg[k] = v
        _last = None
        _heard = None


def note_heard(text, conf=None, loudness=None, reason=None, now=None):
    """STT's accept path may call this so the next new_turn() for the same
    text gets the Vosk confidence and the addressing breakdown. Optional —
    without it those fields are simply null."""
    global _heard
    try:
        with _lock:
            _heard = {"text": (text or "").strip(), "conf": conf,
                      "loudness": loudness, "reason": reason,
                      "ts": time.time() if now is None else float(now)}
    except Exception as e:                                   # noqa: BLE001
        _print_once("note_heard", e)


def _heard_for(text, ts):
    """Pop the note_heard() entry if it describes `text`. Caller holds _lock."""
    global _heard
    h = _heard
    if not h or not text:
        return None
    t = text.strip().lower()
    if ts - h["ts"] > HEARD_FRESH_SECS or t not in h["text"].lower():
        return None
    _heard = None
    return h


def new_turn(**fields):
    """Log a turn; returns its turn_id ("" only if something went badly
    wrong). Unknown fields are kept as-is."""
    global _last
    try:
        ts = float(fields.pop("ts", None) or time.time())
        with _lock:
            turn_id = (f"{_dt.datetime.fromtimestamp(ts):%Y%m%d-%H%M%S}"
                       f"-{next(_counter)}")
            rec = {"turn_id": turn_id, "ts": round(ts, 3),
                   "mode": _cfg["mode"]}
            for k, default in _DEFAULTS:
                fresh = (type(default)() if isinstance(default, (list, dict))
                         else default)
                rec[k] = fields.pop(k, fresh)
            rec.update(fields)
            h = _heard_for(rec.get("heard_clean") or rec.get("heard_raw"), ts)
            if h:
                for src, dst in (("conf", "conf"), ("loudness", "loudness"),
                                 ("reason", "addr_reason")):
                    if rec.get(dst) is None:
                        rec[dst] = h[src]
                if rec.get("heard_raw") is None:
                    rec["heard_raw"] = h["text"]
            _append(rec, ts)
            if rec.get("path") != "ignored":
                _last = json.loads(json.dumps(rec, default=str))
        return turn_id
    except Exception as e:                                   # noqa: BLE001
        _print_once("new_turn", e)
        return ""


def update_turn(turn_id, **fields):
    """Append {"update": turn_id, ...fields}; also patches last_turn()."""
    if not turn_id:
        return
    try:
        now = time.time()
        with _lock:
            ts = now
            if _last and _last.get("turn_id") == turn_id:
                _last.update(json.loads(json.dumps(fields, default=str)))
                ts = _last.get("ts", now)      # same day file as the turn
            rec = {"update": turn_id, "ts": round(now, 3)}
            rec.update(fields)
            _append(rec, ts)
    except Exception as e:                                   # noqa: BLE001
        _print_once("update_turn", e)


def log_ignored(text, conf=None, loudness=None, reason=None, person=None):
    """An in-window utterance the addressing gate decided wasn't for Luna."""
    return new_turn(heard_raw=text, heard_clean=text, conf=conf,
                    loudness=loudness, addressed=False, addr_reason=reason,
                    person=person, path="ignored")


def last_turn():
    """The latest non-ignored turn (with its updates), or None."""
    with _lock:
        return dict(_last) if _last else None


def iter_day(date=None):
    """Every record of a day ('YYYY-MM-DD' / date / timestamp; default
    today), updates merged into their turn, in order. Corrupt lines are
    skipped."""
    try:
        path = path_for(date if date is not None else time.time())
        if not os.path.exists(path):
            return []
        turns, order = {}, []
        with _lock, open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if "update" in rec:
                    t = turns.get(rec["update"])
                    if t is not None:
                        upd = {k: v for k, v in rec.items()
                               if k not in ("update", "ts")}
                        t.update(upd)
                    continue
                tid = rec.get("turn_id") or f"_anon{len(order)}"
                turns[tid] = rec
                order.append(tid)
        return [turns[t] for t in order]
    except Exception as e:                                   # noqa: BLE001
        _print_once("iter_day", e)
        return []


def days():
    """Dates ('YYYY-MM-DD') that have a log file, oldest first."""
    try:
        out = []
        for name in os.listdir(_dir()):
            if name.startswith("conversation_") and name.endswith(".jsonl"):
                out.append(name[len("conversation_"):-len(".jsonl")])
        return sorted(out)
    except Exception:                                        # noqa: BLE001
        return []


def prune(days_to_keep=None, now=None):
    """Delete day files older than `days_to_keep` (default: retention)."""
    try:
        keep = _cfg["retention_days"] if days_to_keep is None else days_to_keep
        if keep is None or keep <= 0:
            return 0
        today = _dt.date.fromtimestamp(time.time() if now is None else now)
        removed = 0
        for d in days():
            try:
                age = (today - _dt.date.fromisoformat(d)).days
            except ValueError:
                continue
            if age > keep:
                os.remove(path_for(d))
                removed += 1
        if removed:
            print(f"[convlog] pruned {removed} log file(s) older than "
                  f"{keep} days")
        return removed
    except Exception as e:                                   # noqa: BLE001
        _print_once("prune", e)
        return 0


prune()
