# proactive.py
"""
proactive.py — Luna acts first. This is most of what makes EMO/Vector-class
robots feel alive rather than voice-activated: they notice you.

A small poll loop (PROACTIVE_POLL_SECS) that watches shared state and, when
Luna is otherwise idle:

  greet   — a recognised face appears → the recognition moment (face.py),
            a wave, and "Hi Alice!" — or "Welcome back, Alice!" when scene
            memory says she left less than WELCOME_BACK_SECS ago. Once per
            person per GREET_COOLDOWN_SECS. If the greeting is a question the
            conversation window opens, so she can answer without the wake word.
  starter — a known person who has lingered SCENE_LINGER_SECS, was greeted a
            while ago and still hasn't said anything gets ONE friendly opener
            per visit ("So Alice, what brings you here today?").
  smile   — someone has been smiling at her for PROACTIVE_STABLE_SECS →
            she smiles back and says something warm.
  sad     — someone has looked sad for a while → a gentle check-in.
  touch   — the renderer saw the face held (state.touch_enroll_request) →
            start the "what's your name?" enrollment dialog.

The loop also feeds scene_memory.observe() on every poll, so Luna's memory of
who came and went works regardless of which thread owns the camera (the calls
are cheap and idempotent: only changes become events).

Hard rules, all enforced in _may_act():
  • never while she's speaking, listening, or mid-conversation — proactive
    charm must not interrupt actual interaction;
  • never before state.ready — camera warm-up noise can fake an emotion;
  • every reaction is cooldown-limited, because a robot that comments on
    every smile is needy, not charming.
The touch-enroll request is the one exception to "not mid-conversation": it is
the person asking, not Luna volunteering — but it still waits for her to stop
speaking.

Pure policy helpers (unit-tested): choose_greeting, should_start_conversation.
"""

import random
import threading
import time

import face
from shared_state import state
from servo_module import servo
from config import (ENABLE_PROACTIVE, PROACTIVE_GREET_RECOGNIZED,
                    GREET_COOLDOWN_SECS, PROACTIVE_SMILE_REACT,
                    SMILE_REACT_COOLDOWN, PROACTIVE_SAD_REACT,
                    SAD_REACT_COOLDOWN, PROACTIVE_POLL_SECS,
                    PROACTIVE_STABLE_SECS, GREET_PHRASES, SMILE_PHRASES,
                    SAD_PHRASES, RECOG_ANIMATION,
                    RECOG_GREET_OPENS_CONVERSATION, WELCOME_BACK_SECS,
                    WELCOME_BACK_PHRASES, ENABLE_SCENE_MEMORY,
                    SCENE_LINGER_SECS, PROACTIVE_CONVERSATION_STARTERS,
                    CONVERSATION_STARTERS, TOUCH_ENROLL)

# A conversation starter waits at least this long after the greeting, so the
# two never land back to back.
STARTER_AFTER_GREET_SECS = GREET_COOLDOWN_SECS / 5.0

_last_greet = {}          # name -> time.time() of last greeting
_started_visits = set()   # visit ids (person_since) that already got a starter


# ══════════════════════════════════════════════════════════════════════════════
# Pure policy (tested in tests/test_scene_memory.py)
# ══════════════════════════════════════════════════════════════════════════════
def choose_greeting(name, seconds_since_left, rng=random):
    """The line Luna says on recognising `name`. A person who left less than
    WELCOME_BACK_SECS ago gets a "welcome back"; everyone else the full
    greeting. seconds_since_left is None when they have never left."""
    back = (seconds_since_left is not None
            and seconds_since_left < WELCOME_BACK_SECS)
    phrases = WELCOME_BACK_PHRASES if back else GREET_PHRASES
    return rng.choice(phrases).format(name=name)


def should_start_conversation(name, now, person_since, greeted_at,
                              started_visits=(), conversation_active=False,
                              speaking=False, linger_secs=SCENE_LINGER_SECS,
                              after_greet_secs=STARTER_AFTER_GREET_SECS):
    """Should Luna open a conversation with `name` now?

    Yes only when a KNOWN person has been in view for linger_secs, was
    already greeted at least after_greet_secs ago, nobody is talking, and this
    visit (identified by person_since) has not had a starter yet."""
    if not name or not person_since:
        return False
    if conversation_active or speaking:
        return False
    if person_since in started_visits:
        return False
    if now - person_since < linger_secs:
        return False
    if not greeted_at or now - greeted_at < after_greet_secs:
        return False
    return True


def is_question(text):
    return bool(text) and text.rstrip().endswith("?")


# ══════════════════════════════════════════════════════════════════════════════
# Side effects
# ══════════════════════════════════════════════════════════════════════════════
def _speak(text):
    from text_to_speech import speak   # deferred, same pattern as commands.py
    speak(text)


def _scene_memory():
    """The scene_memory module, or None when disabled / broken. Deferred so a
    problem there can never stop Luna greeting people."""
    if not ENABLE_SCENE_MEMORY:
        return None
    try:
        import scene_memory
        return scene_memory
    except Exception as e:                                  # noqa: BLE001
        print(f"[proactive] scene memory unavailable: {e}")
        return None


def _note(kind, who, detail=None):
    sm = _scene_memory()
    if sm is not None:
        sm.note(kind, who, detail)


def _may_act():
    with state.lock:
        return (state.ready
                and not state.speaking
                and not state.tts_active
                and not state.listening
                and not state.conversation_active
                and state.enroll_request is None)


def _may_respond():
    """Looser than _may_act, for things the PERSON asked for (touch enroll):
    ready, silent, and no enrollment already running."""
    with state.lock:
        return (state.ready
                and not state.speaking
                and not state.tts_active
                and state.enroll_request is None)


def _open_conversation():
    """Hold the conversation window open for an answer to what Luna just
    asked. Brain's history is left alone — the scene memory "asked" note is
    what tells the LLM what the answer is answering."""
    with state.lock:
        state.conversation_active = True
        state.awaiting_answer     = True
        state.last_activity_time  = time.time()
    print("[proactive] Asked a question — conversation window open")


def _recognition_moment(name):
    """face.recognition_moment when face.py has it; otherwise the V2 look
    (happy face + name banner)."""
    moment = getattr(face, "recognition_moment", None) if RECOG_ANIMATION else None
    if moment is not None:
        try:
            moment(name)
            return
        except Exception as e:                              # noqa: BLE001
            print(f"[proactive] recognition_moment failed: {e}")
    face.happy(5)
    face.banner(f"Hi {name}!")


def _greet(name):
    now = time.time()
    _last_greet[name] = now
    sm = _scene_memory()
    since_left = sm.seconds_since_left(name) if sm is not None else None
    line = choose_greeting(name, since_left)
    print(f"[proactive] Greeting {name}"
          + (f" (back after {since_left:.0f}s)" if since_left is not None else ""))

    _recognition_moment(name)
    try:
        servo.wave()
    except Exception:
        pass
    _note("greeted", name, line)
    asks = RECOG_GREET_OPENS_CONVERSATION and is_question(line)
    if asks:
        _note("asked", name, line)
    _speak(line)
    if asks:
        _open_conversation()


def _start_conversation(name, visit):
    _started_visits.add(visit)
    line = random.choice(CONVERSATION_STARTERS).format(name=name)
    print(f"[proactive] Conversation starter for {name}")
    face.curious(4)
    _note("asked", name, line)
    _speak(line)
    _open_conversation()


def _handle_touch_enroll():
    """The face was held on the touchscreen → ask for the name."""
    with state.lock:
        state.touch_enroll_request = False
    if not TOUCH_ENROLL:
        return
    print("[proactive] Touch enroll requested")
    try:
        import commands
        start = getattr(commands, "start_enrollment_dialog", None)
        if start is not None:
            start()
            return
    except Exception as e:                                  # noqa: BLE001
        print(f"[proactive] start_enrollment_dialog failed: {e}")
    face.curious(4)
    _speak("Say: remember me, my name is — and your name.")
    _open_conversation()


def _visit_id(person_since):
    """person_since from the vision thread, or scene memory's visit start
    when vision doesn't set it."""
    if person_since:
        return person_since
    sm = _scene_memory()
    if sm is None:
        return 0.0
    try:
        return sm.presence().get("since") or 0.0
    except Exception:                                       # noqa: BLE001
        return 0.0


def proactive_loop():
    emotion_since = {"value": None, "ts": time.time()}
    last_smile = 0.0
    last_sad   = 0.0

    while True:
        time.sleep(PROACTIVE_POLL_SECS)

        with state.lock:
            person       = state.person_name
            emotion      = state.emotion
            seen         = state.face_detected
            face_count   = state.face_count
            person_since = state.person_since
            touch_req    = state.touch_enroll_request

        sm = _scene_memory()
        if sm is not None:
            sm.observe(face_count if seen else 0, person, emotion)

        # Track how long the CURRENT emotion has been continuously held —
        # a single smiling frame is noise, two seconds of it is a mood.
        if emotion != emotion_since["value"]:
            emotion_since = {"value": emotion, "ts": time.time()}
        stable_for = time.time() - emotion_since["ts"]

        if touch_req and _may_respond():
            _handle_touch_enroll()
            continue

        if not _may_act():
            continue

        now = time.time()

        if (PROACTIVE_GREET_RECOGNIZED and person
                and now - _last_greet.get(person, 0.0) > GREET_COOLDOWN_SECS):
            _greet(person)
            continue

        if PROACTIVE_CONVERSATION_STARTERS and person:
            visit = _visit_id(person_since)
            if should_start_conversation(person, now, visit,
                                         _last_greet.get(person),
                                         _started_visits):
                _start_conversation(person, visit)
                continue

        if (PROACTIVE_SMILE_REACT and seen and emotion == "Happy"
                and stable_for >= PROACTIVE_STABLE_SECS
                and now - last_smile > SMILE_REACT_COOLDOWN):
            last_smile = now
            print("[proactive] Smiling back")
            face.happy(4)
            _speak(random.choice(SMILE_PHRASES))
            continue

        if (PROACTIVE_SAD_REACT and seen and emotion == "Sad"
                and stable_for >= PROACTIVE_STABLE_SECS
                and now - last_sad > SAD_REACT_COOLDOWN):
            last_sad = now
            print("[proactive] Sad check-in")
            face.curious(4)
            _speak(random.choice(SAD_PHRASES))
            continue


def start_proactive():
    if not ENABLE_PROACTIVE:
        print("[proactive] Disabled in config")
        return
    t = threading.Thread(target=proactive_loop, daemon=True, name="proactive")
    t.start()
