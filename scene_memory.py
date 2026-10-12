# scene_memory.py
"""
scene_memory.py — what Luna remembers about the last half hour in front of
her camera, in a form a small LLM can use.

The vision thread already knows *who is here right now*; what it lacks is
*what happened*: that Bob walked off five minutes ago, that Alice looked sad a
moment ago and then cheered up, that a laptop came into view. Without that,
Luna can only ever talk about the current frame. With it, she can say "you
looked tired earlier" or "Bob was here a few minutes ago" without being asked
"what do you see" — the difference between a camera with a voice and a robot
who has been paying attention.

Design rules (all deliberate):

  • EVENTS, not frames. observe() is called every vision tick, but it only
    records state CHANGES: an arrival, a departure, a mood that lasted, a new
    object. Twelve events cover a busy half hour; twelve thousand frames tell
    the LLM nothing.
  • Debounced. A missed detection is not a departure (LEAVE_DEBOUNCE_SECS),
    one smiling frame is not a mood (MOOD_HOLD_SECS). Recognition flicker
    (name → None → name while the face is still there) is ignored: a named
    presence persists as long as *a* face is visible. A DIFFERENT name only
    replaces the current visitor once it has held for SWITCH_DEBOUNCE_SECS —
    with two people in view the recogniser can alternate between them, and
    that must not read as a stream of arrivals and departures.
  • Pure and light. No numpy, no models, no other Luna modules — just config
    for the tunables — so it imports in tests and can never slow the vision
    loop. One lock guards everything; observe() and summary() never raise
    (an exception is printed once and swallowed), because nothing about
    remembering may ever crash the vision thread or a voice turn.
  • Spoken-style output. summary() is one or two short sentences with human
    durations ("just now", "2 minutes ago"), ready to drop into a prompt.

Public API
  observe(face_count, person_name, emotion, objects=None, now=None)
  summary(now=None) -> str                 seconds_since_left(name, now=None)
  present_since(now=None)                  presence(now=None) -> dict
  last_events(n=None) -> [dict]            note(kind, who, detail, now=None)
  reset()
Pure helpers (tested directly): compose_summary, human_duration, human_ago.
"""

import threading
import time
from collections import deque

from config import SCENE_MEMORY_EVENTS, SCENE_MEMORY_MAX_AGE, PERSON_AWAY_SECS

# ── Module tunables (not in config.py — see new_tunables in the report) ───────
# Face/name must be gone this long to count as "left". It used to be 1.5 s, so
# glancing at a phone was a departure and the return a "came back" — which
# the LLM then turned into "you're back again". Shared with vision, which
# holds on to the recognised name for the same time.
LEAVE_DEBOUNCE_SECS = PERSON_AWAY_SECS
SWITCH_DEBOUNCE_SECS = 1.5   # a different name must hold this long to replace
                             # the visitor (vision has already debounced it)
MOOD_HOLD_SECS      = 2.0    # a non-neutral emotion must last this long to be a mood
OBJECT_FORGET_SECS  = 300.0  # an object seen again after this long is "new" again
ASKED_RELEVANT_SECS = 120.0  # Luna's own open question stays the top history item this long

# Emotion label → the word Luna would actually say.
_MOOD_WORDS = {
    "happy": "happy", "sad": "sad", "angry": "angry", "surprise": "surprised",
    "surprised": "surprised", "fear": "a bit scared", "disgust": "unimpressed",
    "neutral": "calm",
}

_lock = threading.Lock()

# Presence (all guarded by _lock)
_present       = False      # is anyone in front of the camera (debounced)?
_present_name  = None       # recognised name of the current visitor, or None
_present_since = 0.0        # when the current visit began
_face_count    = 0          # faces in the last frame
_absent_since  = None       # first tick with nobody, while the departure is pending
_switch_name   = None       # a different recognised name, pending (see debounce)
_switch_since  = 0.0

# Mood tracking for the current visitor
_mood_candidate = None      # emotion currently being held
_mood_since     = 0.0
_mood_noted     = False     # has this held emotion been recorded yet?

_last_seen = {}             # name -> ts of the most recent frame with that name
_last_left = {}             # name -> ts they were last seen before a confirmed departure
_objects   = {}             # label -> ts last seen in `objects`

_events = deque(maxlen=max(1, int(SCENE_MEMORY_EVENTS)))
_errors_printed = set()


# ══════════════════════════════════════════════════════════════════════════════
# Small pure helpers
# ══════════════════════════════════════════════════════════════════════════════
def _now(now):
    return time.time() if now is None else float(now)


def _print_once(where, exc):
    key = (where, type(exc).__name__, str(exc))
    if key in _errors_printed:
        return
    _errors_printed.add(key)
    print(f"[scene] {where} failed ({type(exc).__name__}: {exc}) — ignoring "
          f"further errors of this kind")


def human_duration(secs):
    """A span of time as Luna would say it: 'a moment', '40 seconds',
    'a minute', '3 minutes', 'an hour', '2 hours'."""
    secs = max(0.0, float(secs))
    if secs < 5:
        return "a moment"
    if secs < 60:
        return f"{int(secs)} seconds"
    minutes = int(secs // 60)
    if minutes < 2:
        return "a minute"
    if minutes < 60:
        return f"{minutes} minutes"
    hours = int(minutes // 60)
    return "an hour" if hours < 2 else f"{hours} hours"


def human_ago(secs):
    """How long ago something happened: 'just now', 'a moment ago',
    'a minute ago', '5 minutes ago', 'an hour ago'."""
    secs = max(0.0, float(secs))
    if secs < 10:
        return "just now"
    if secs < 60:
        return "a moment ago"
    return human_duration(secs) + " ago"


def _mood_word(emotion):
    if not emotion:
        return ""
    return _MOOD_WORDS.get(str(emotion).strip().lower(), str(emotion).lower())


def _is_neutral(emotion):
    return not emotion or str(emotion).strip().lower() in ("neutral", "none", "")


def _who_text(who, capital=True):
    if who:
        return str(who)
    return "Someone" if capital else "someone"


def _join_labels(labels):
    labels = [str(x) for x in labels if x]
    if not labels:
        return ""
    if len(labels) == 1:
        return f"a {labels[0]}"
    return "a " + ", a ".join(labels[:-1]) + f" and a {labels[-1]}"


# ══════════════════════════════════════════════════════════════════════════════
# Summary composition (pure — takes a snapshot, returns text)
# ══════════════════════════════════════════════════════════════════════════════
def _presence_sentence(presence, now, moods=True):
    """'Alice has been in front of you for 3 minutes and looks happy.' etc."""
    if not presence or not presence.get("present"):
        return "Nobody is in front of you right now."

    name    = presence.get("name")
    count   = int(presence.get("count") or 1)
    held    = max(0.0, now - float(presence.get("since") or now))
    mood    = presence.get("mood") if moods else None
    looks   = f" and looks {_mood_word(mood)}" if mood and not _is_neutral(mood) else ""
    others  = max(0, count - 1)

    if name:
        if held < 5:
            s = f"{name} just arrived{looks}."
        else:
            s = f"{name} has been in front of you for {human_duration(held)}{looks}."
        if others:
            s = s[:-1] + (", with one other person." if others == 1
                          else f", with {others} other people.")
        return s

    if count <= 1:
        if held < 60:
            return f"One person you have not met is here{looks}."
        return (f"One person you have not met has been here for "
                f"{human_duration(held)}{looks}.")
    return (f"{count} people you have not met are here" + looks + ".")


def _event_sentence(ev, now, presence):
    """One event as a sentence, or None when it adds nothing useful."""
    kind   = ev.get("kind")
    who    = ev.get("who")
    detail = ev.get("detail")
    ago    = human_ago(now - float(ev.get("ts", now)))
    present_name = presence.get("name") if presence and presence.get("present") else None

    if kind == "left":
        # The person came back — "Alice stopped by" would contradict sentence 1.
        if who and who == present_name:
            return f"{who} stepped away for a bit and came back."
        return f"{_who_text(who)} stopped by {ago}."
    if kind == "mood":
        # Already said in the presence sentence when it is the current mood.
        if (presence and presence.get("present") and who == present_name
                and presence.get("mood") == detail):
            return None
        return f"{_who_text(who)} looked {_mood_word(detail)} {ago}."
    if kind == "objects":
        if not detail:
            return None
        labels = [x.strip() for x in str(detail).split(",") if x.strip()]
        return f"You noticed {_join_labels(labels)} {ago}."
    if kind == "arrived":
        # Current visitor's arrival is sentence 1; a past arrival always has
        # a matching "left" event that tells the story better.
        return None
    if kind == "greeted":
        if who and who == present_name:
            return f"You already said hi to {who} {ago}."
        return None
    if kind == "asked":
        # Luna opened the conversation herself — the next thing the person
        # says is probably the answer, and the LLM must know the question.
        if detail and (not who or who == present_name):
            return f'You asked {_who_text(who, False)} {ago}: "{detail}"'
        return None
    if detail:
        return f"{_who_text(who)}: {detail} ({ago})." if who else f"{detail} ({ago})."
    return None


def compose_summary(presence, events, now, moods=True, objects=True):
    """Pure: presence snapshot + event list → the one-or-two-sentence summary.

    presence — {"present": bool, "name": str|None, "since": ts, "count": int,
                "mood": str|None}   (as returned by presence())
    events   — [{ts, kind, who, detail}] oldest → newest
    moods / objects — False leaves out what the camera GUESSED (expressions,
               things on the desk): on an ordinary question those are noise
               a small model will happily start talking about.
    """
    skip = set()
    if not moods:
        skip.add("mood")
    if not objects:
        skip.add("objects")
    events = [e for e in (events or ()) if e.get("kind") not in skip]
    anyone = bool(presence and presence.get("present"))
    if not anyone and not events:
        return ""

    first = _presence_sentence(presence, now, moods)

    # The most useful bit of history, in priority order: a question Luna has
    # just asked (the reply is coming), then the newest departure or mood
    # change, then the newest event that says anything at all.
    second = None
    for ev in reversed(events):
        if (ev.get("kind") == "asked"
                and now - float(ev.get("ts", now)) <= ASKED_RELEVANT_SECS):
            second = _event_sentence(ev, now, presence)
            if second:
                break
    if second is None:
        for ev in reversed(events):
            if ev.get("kind") in ("left", "mood"):
                second = _event_sentence(ev, now, presence)
                if second:
                    break
    if second is None:
        for ev in reversed(events):
            second = _event_sentence(ev, now, presence)
            if second:
                break
    return first if not second else f"{first} {second}"


# ══════════════════════════════════════════════════════════════════════════════
# State machine (all callers hold _lock)
# ══════════════════════════════════════════════════════════════════════════════
def _prune(now):
    """Forget events older than SCENE_MEMORY_MAX_AGE (deque keeps the cap)."""
    if not _events:
        return
    keep = [e for e in _events if now - e["ts"] <= SCENE_MEMORY_MAX_AGE]
    if len(keep) != len(_events):
        _events.clear()
        _events.extend(keep)


def _push(ts, kind, who, detail):
    _events.append({"ts": float(ts), "kind": kind, "who": who, "detail": detail})


def _reset_mood():
    global _mood_candidate, _mood_since, _mood_noted
    _mood_candidate, _mood_since, _mood_noted = None, 0.0, False


def _begin_visit(now, name):
    global _present, _present_name, _present_since, _absent_since
    global _switch_name, _switch_since
    _present, _present_name, _present_since, _absent_since = True, name, now, None
    _switch_name, _switch_since = None, 0.0
    _reset_mood()
    _push(now, "arrived", name, None)


def _end_visit(last_seen_ts):
    global _present, _present_name, _absent_since, _switch_name, _switch_since
    if _present_name:
        _last_left[_present_name] = last_seen_ts
    _push(last_seen_ts, "left", _present_name, None)
    _present, _present_name, _absent_since = False, None, None
    _switch_name, _switch_since = None, 0.0
    _reset_mood()


def _upgrade_to_name(now, name):
    """The visitor was anonymous and has just been recognised: keep the visit
    (and its start time) and put the name on this visit's anonymous events
    instead of adding a second arrival — one person, one arrival."""
    global _present_name
    _present_name = name
    renamed_arrival = False
    for ev in _events:
        if ev["who"] is None and ev["ts"] >= _present_since and ev["kind"] in (
                "arrived", "mood", "objects"):
            ev["who"] = name        # "Someone looked sad" → "Alice looked sad"
            renamed_arrival = renamed_arrival or ev["kind"] == "arrived"
    if not renamed_arrival:
        _push(now, "arrived", name, None)


def _observe_locked(face_count, person_name, emotion, objects, now):
    global _face_count, _absent_since, _mood_candidate, _mood_since, _mood_noted
    global _switch_name, _switch_since

    name    = str(person_name).strip() if person_name else None
    someone = bool(name) or (face_count or 0) > 0
    _face_count = int(face_count or 0) if someone else 0
    if someone and name:
        _last_seen[name] = now

    # ── presence transitions ──────────────────────────────────────────────
    if someone:
        if not _present:
            _begin_visit(now, name)
        else:
            _absent_since = None                     # they are still here
            if name and _present_name is None:
                _upgrade_to_name(now, name)
            elif name and name != _present_name:
                # Someone else took their place — but only once the new name
                # has held (two people in view can make the recogniser flip).
                if name != _switch_name:
                    _switch_name, _switch_since = name, now
                elif now - _switch_since >= SWITCH_DEBOUNCE_SECS:
                    since = _switch_since
                    old = _present_name
                    _end_visit(_last_seen.get(old, since) if old else since)
                    _begin_visit(since, name)
            elif name:
                _switch_name, _switch_since = None, 0.0
            # name → None while a face is still there: recognition flicker, ignore
    elif _present:
        if _absent_since is None:
            _absent_since = now
        elif now - _absent_since > LEAVE_DEBOUNCE_SECS:
            _end_visit(_absent_since)

    # ── mood: a non-neutral emotion that lasts ────────────────────────────
    if _present and someone:
        em = None if _is_neutral(emotion) else str(emotion).strip()
        if em != _mood_candidate:
            _mood_candidate, _mood_since, _mood_noted = em, now, False
        elif em and not _mood_noted and now - _mood_since >= MOOD_HOLD_SECS:
            _mood_noted = True
            _push(now, "mood", _present_name, em)

    # ── objects: only labels we have not seen lately ──────────────────────
    if objects is not None:
        new = []
        for label in objects:
            label = str(label).strip().lower()
            if not label:
                continue
            last = _objects.get(label)
            if last is None or now - last > OBJECT_FORGET_SECS:
                if label not in new:
                    new.append(label)
            _objects[label] = now
        if new:
            _push(now, "objects", _present_name if _present else None, ", ".join(new))

    _prune(now)


def _presence_locked(now):
    mood = _mood_candidate if (_present and _mood_noted) else None
    return {
        "present": _present,
        "name":    _present_name,
        "since":   _present_since if _present else None,
        "count":   _face_count if _present else 0,
        "mood":    mood,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════
def observe(face_count, person_name, emotion, objects=None, now=None):
    """Feed one vision tick. Records only state changes (see module doc).
    Never raises."""
    try:
        now = _now(now)
        with _lock:
            _observe_locked(face_count, person_name, emotion, objects, now)
    except Exception as e:                                  # noqa: BLE001
        _print_once("observe", e)


def summary(now=None, moods=True, objects=True):
    """One or two spoken-style sentences for the LLM, or "" when there is
    nothing to say. moods / objects=False: people only (see compose_summary).
    Never raises."""
    try:
        now = _now(now)
        with _lock:
            _prune(now)
            presence = _presence_locked(now)
            events   = [dict(e) for e in _events]
        return compose_summary(presence, events, now, moods, objects)
    except Exception as e:                                  # noqa: BLE001
        _print_once("summary", e)
        return ""


def asked_recently(now=None, within=ASKED_RELEVANT_SECS):
    """Did Luna ask someone a question in the last `within` seconds? Then the
    next thing said is probably the answer, and the scene notes (which hold
    her question) belong in the prompt. Never raises."""
    try:
        now = _now(now)
        with _lock:
            return any(e["kind"] == "asked" and 0 <= now - e["ts"] <= within
                       for e in _events)
    except Exception as e:                                  # noqa: BLE001
        _print_once("asked_recently", e)
        return False


def seconds_since_left(name, now=None):
    """Seconds since `name` was last seen before a confirmed departure, or
    None if they have never left (or are unknown). Still answers while they
    are back in view — that is exactly when 'welcome back' needs it."""
    if not name:
        return None
    try:
        now = _now(now)
        with _lock:
            ts = _last_left.get(str(name).strip())
        return None if ts is None else max(0.0, now - ts)
    except Exception as e:                                  # noqa: BLE001
        _print_once("seconds_since_left", e)
        return None


def present_since(now=None):
    """Seconds the current visitor (named or not) has been in view, or None."""
    try:
        now = _now(now)
        with _lock:
            if not _present:
                return None
            return max(0.0, now - _present_since)
    except Exception as e:                                  # noqa: BLE001
        _print_once("present_since", e)
        return None


def presence(now=None):
    """Snapshot of the current visitor: {present, name, since, count, mood}."""
    now = _now(now)
    with _lock:
        return _presence_locked(now)


def last_events(n=None):
    """The newest `n` events (all when n is None), newest last, as copies."""
    with _lock:
        evs = [dict(e) for e in _events]
    if n is None:
        return evs
    n = int(n)
    return evs[-n:] if n > 0 else []


def note(kind, who, detail, now=None):
    """Let another module add an event. Known kinds with their own wording
    in summary(): "greeted" (who) and "asked" (who, detail = the question
    Luna asked). Anything else is narrated generically. Never raises."""
    try:
        now = _now(now)
        with _lock:
            _push(now, str(kind), who, detail)
            _prune(now)
    except Exception as e:                                  # noqa: BLE001
        _print_once("note", e)


def reset():
    """Forget everything (tests, and "forget everyone")."""
    global _present, _present_name, _present_since, _face_count, _absent_since
    global _switch_name, _switch_since
    with _lock:
        _present, _present_name, _present_since = False, None, 0.0
        _face_count, _absent_since = 0, None
        _switch_name, _switch_since = None, 0.0
        _reset_mood()
        _last_seen.clear()
        _last_left.clear()
        _objects.clear()
        _events.clear()
