# face.py
"""
face.py — the one place anything outside the renderer touches Luna's face.

Every expression is a plain function call. Import it and say what you mean:

    import face

    face.happy()            # look happy for a few seconds, then go back
    face.shy(2)             # shy for exactly 2 seconds
    face.angry(hold=True)   # angry until something clears it
    face.sleep()            # go to sleep
    face.wake()             # wake up
    face.clear()            # drop back to whatever vision says
    face.recognition_moment("Alice")   # the "I know you!" beat

These are safe to call from ANY thread — the brain, the gesture handler, a
REPL — and they never touch pygame. They set a face override on the shared
state, which the renderer picks up on its next frame.

Why an override and not `RobotFace.set_state()` directly: the render loop
rewrites the face from vision several times a second, so a direct write is
erased almost immediately. The override is the only channel that survives it.

To change how an expression LOOKS, edit expressions.py, not this file. This
file only decides *when* a face is worn; expressions.py decides what it is.

Run `python face.py` to preview every expression in turn.
"""

import time

from shared_state import state
from config import (FACE_OVERRIDE_SECS, GREET_BANNER_SECS,
                    RECOG_ANIMATION, RECOG_EXPRESSION_SECS)
from expressions import EXPRESSIONS, NAMES

__all__ = [
    "happy", "sad", "angry", "surprised", "excited", "love", "shy",
    "curious", "annoyed", "amazed", "recognised",
    "sleep", "wake", "neutral", "clear", "show", "current", "available",
    "banner", "recognition_moment",
]

#: Passed as `seconds` to keep an expression up until something clears it.
FOREVER = float("inf")


# ══════════════════════════════════════════════════════════════════════════════
# Core
# ══════════════════════════════════════════════════════════════════════════════
def show(name, seconds=None, hold=False):
    """Wear expression `name`. The general form behind every function below.

    seconds — how long to hold it; defaults to config.FACE_OVERRIDE_SECS
    hold    — True keeps it up indefinitely (same as seconds=FOREVER)

    An unknown name is ignored rather than raising: a bad emotion string from
    the LLM or from vision should leave Luna's face alone, not kill the caller.
    """
    if name not in EXPRESSIONS:
        print(f"[face] Unknown expression {name!r} — ignored. "
              f"Known: {', '.join(NAMES)}")
        return False

    with state.lock:
        _set_override(name, _duration(seconds, hold, FACE_OVERRIDE_SECS),
                      time.time())
    return True


def _duration(seconds, hold, default):
    if hold:
        return FOREVER
    return default if seconds is None else seconds


def _set_override(name, seconds, now):
    """Caller holds state.lock."""
    state.face_override = name
    state.face_override_until = now + seconds


def _set_banner(text, seconds, now):
    """Caller holds state.lock. banner_since drives the renderer's slide-in."""
    state.banner_text  = text
    state.banner_since = now
    state.banner_until = now + (seconds or GREET_BANNER_SECS)


def clear():
    """Drop the override and hand the face back to whatever vision reports."""
    with state.lock:
        state.face_override = None
        state.face_override_until = 0.0


def banner(text, seconds=None):
    """Show a short text pill under the face ("Hi Alice!"). Rendered by
    robot_face._draw_banner; safe from any thread, like everything here."""
    with state.lock:
        _set_banner(text, seconds, time.time())


def recognition_moment(name):
    """The "I know you!" beat, when Luna recognises someone she has met.

    Wears "recognised" for RECOG_EXPRESSION_SECS, posts "Hi {name}!" and
    stamps recog_event_time, which the renderer turns into a quick eye
    double-take. All three are written under ONE lock acquisition, so the
    render thread can never see the banner without the face (or vice versa).
    With RECOG_ANIMATION off it degrades to the plain V2 look: a happy face
    and the banner, no double-take."""
    now = time.time()
    with state.lock:
        if RECOG_ANIMATION:
            _set_override("recognised", RECOG_EXPRESSION_SECS, now)
            state.recog_event_time = now
        else:
            _set_override("happy", RECOG_EXPRESSION_SECS, now)
        _set_banner(f"Hi {name}!", None, now)
    return True


def current():
    """The expression Luna is wearing right now, as a name."""
    with state.lock:
        override = state.face_override
        until = state.face_override_until
    if override and time.time() <= until:
        return override
    return None


def available():
    """Every expression name that can be shown."""
    return NAMES


# ══════════════════════════════════════════════════════════════════════════════
# Expressions — one function each
# ══════════════════════════════════════════════════════════════════════════════
def neutral(seconds=None, hold=False):
    """Resting face — eyes only, no mouth."""
    return show("neutral", seconds, hold)


def happy(seconds=None, hold=False):
    """Arched eyes, a smile, a small hop and a light blush."""
    return show("happy", seconds, hold)


def sad(seconds=None, hold=False):
    """Narrow drooping eyes, a downturned mouth, the occasional tear."""
    return show("sad", seconds, hold)


def angry(seconds=None, hold=False):
    """Hard squint, angled brows, a red throb and venting steam."""
    return show("angry", seconds, hold)


def surprised(seconds=None, hold=False):
    """Wide eyes, blown pupils, an O of a mouth and a one-shot starburst."""
    return show("surprised", seconds, hold)


def excited(seconds=None, hold=False):
    """Surprise plus delight — wide, arched, bouncing, trailing stars.
    Also throws both arms up if the servos are connected."""
    return show("excited", seconds, hold)


def love(seconds=None, hold=False):
    """Deep blush and floating hearts. What compliments get."""
    return show("love", seconds, hold)


def shy(seconds=None, hold=False):
    """Bashful — heavy blush, head tipped away, eyes averted."""
    return show("shy", seconds, hold)


def curious(seconds=None, hold=False):
    """Head cocked, one eye raised — 'ooh, what's that?'"""
    return show("curious", seconds, hold)


def annoyed(seconds=None, hold=False):
    """Half-lidded deadpan stare. The quiet sibling of angry."""
    return show("annoyed", seconds, hold)


def amazed(seconds=None, hold=False):
    """Everything maxed — wide eyes, starburst, bounce."""
    return show("amazed", seconds, hold)


def recognised(seconds=None, hold=False):
    """"I know you!" — bright arc eyes, big smile, stars then hearts. Held for
    RECOG_EXPRESSION_SECS by default. Usually worn via recognition_moment(),
    which also posts the name banner and triggers the eye double-take."""
    if seconds is None and not hold:
        seconds = RECOG_EXPRESSION_SECS
    return show("recognised", seconds, hold)


# ── Sleep is a state, not a mood, so it gets a matched pair ───────────────────
def sleep(seconds=None, hold=True):
    """Go to sleep: eyes shut, z's drifting up. Held indefinitely by default,
    since sleep should last until something wakes her."""
    return show("sleeping", seconds, hold)


def wake(seconds=None):
    """Wake up: eyes pop open with a yawn-and-stretch, a ripple, and both arms
    reaching up if servos are attached.

    Held briefly at neutral rather than cleared outright — clearing would let
    the idle timer put her straight back to sleep on the very next frame."""
    return show("neutral", seconds if seconds is not None else 5.0)


#: Alias — `wakeup()` reads more naturally next to `sleep()` for some callers.
wakeup = wake

#: Alias — vision reports this emotion as "Surprise".
surprise = surprised


# ══════════════════════════════════════════════════════════════════════════════
# Preview:  python face.py
# ══════════════════════════════════════════════════════════════════════════════
def _demo(seconds_each=3.0):
    """Cycle through every expression so you can eyeball your changes.

    Drives RobotFace directly rather than going through the override, because
    in a preview there is no vision thread to override in the first place.
    """
    from robot_face import RobotFace

    face = RobotFace()
    order = list(NAMES)
    index = -1
    next_switch = 0.0

    print(f"[face] Preview — {seconds_each:g}s each. "
          f"1/2 switch style, ESC or Q quits.")
    while True:
        now = time.time()
        if now >= next_switch:
            index = (index + 1) % len(order)
            next_switch = now + seconds_each
            face.set_state(order[index])
            print(f"[face] {order[index]}")
        face.draw()


if __name__ == "__main__":
    try:
        _demo()
    except (SystemExit, KeyboardInterrupt):
        print("\n[face] Preview closed")
