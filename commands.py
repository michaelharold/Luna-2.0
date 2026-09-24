# commands.py
"""
commands.py — spoken commands, handled before any model gets involved.

These are the things that must work instantly and deterministically — no LLM
creativity wanted when someone says "volume down". brain.process() calls
`handle(text)` first; a True return means the utterance was consumed here.

What lives here:
  • volume        — "volume up / down / fifty percent" (pactl on the Pi,
                    osascript on a Mac — chosen by RUN_MODE)
  • sleep / wake  — "go to sleep", "wake up"
  • face style    — "change your face" cycles styles 1→2→3
  • enrollment    — "remember me, my name is Alice" / "my name is Alice";
                    a bare "remember me" asks for the name and waits.
                    With ENROLL_CONFIRM_NAME Luna first checks "Did you say
                    Alice?" (Vosk mishears names more than any other word),
                    and with ENROLL_GUIDED_POSES she asks for straight / left /
                    right head poses so the stored embedding is the average of
                    several angles — recognition then survives a turned head.
                    start_enrollment_dialog() opens the same flow from outside
                    (touch-and-hold on the face, proactive offers).
  • recognition   — "who am I?", "who do you know?"
  • forgetting    — "forget me" / "forget Alice"
  • feedback      — "good answer" / "that's wrong" about Luna's last reply
                    (learning.py rewards it); after "that's wrong" Luna asks
                    for the right answer and queues it for human approval.
  • teaching      — "remember that …" / "note that …" → pending facts
                    (scripts/approve_facts.py puts them into knowledge.txt).

The password shutdown ("exit" → password → "exit") stays in speech_to_text,
where it must run BEFORE any gating.

Multi-turn state (asked-for-name, confirming-the-name) follows the same
module-level pattern as speech_to_text's _awaiting_terminate.
"""

import random
import re
import subprocess
import time

import face
import face_store
from shared_state import state
from config import (ENABLE_VOICE_COMMANDS, ENABLE_FACE_RECOGNITION,
                    VOLUME_STEP, VOLUME_MIN, VOLUME_MAX,
                    ENROLL_TIMEOUT, IS_MAC,
                    ENROLL_GUIDED_POSES, ENROLL_POSES,
                    ENROLL_FRAMES_PER_POSE, ENROLL_CONFIRM_NAME)

# Guided registration timing. The pose prompt itself (speak blocks) is most of
# each capture window; after it ends Luna waits at least ENROLL_POSE_SETTLE so
# the head has finished turning, then until vision holds
# ENROLL_FRAMES_PER_POSE samples for the pose, but never past ENROLL_POSE_WAIT.
ENROLL_POSE_SETTLE = 0.4
ENROLL_POSE_WAIT   = 2.5
# "Did you say Alice?" stays answerable this long; after that the next
# utterance is treated normally (the person walked off mid-dialog).
ENROLL_CONFIRM_TIMEOUT = 30.0


def _speak(text):
    """Deferred import: text_to_speech pulls in the TTS engine, and brain
    already imports this module at load time — importing speak lazily keeps
    module import order flat and cheap."""
    from text_to_speech import speak
    speak(text)


# ── Volume ────────────────────────────────────────────────────────────────────

_WORD_NUMBERS = {
    "ten": 10, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "full": 100, "max": 100, "maximum": 100, "half": 50,
}


def _get_volume():
    try:
        if IS_MAC:
            out = subprocess.run(
                ["osascript", "-e", "output volume of (get volume settings)"],
                capture_output=True, text=True, timeout=5).stdout.strip()
            return int(out)
        out = subprocess.run(
            ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
            capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"(\d+)%", out)
        return int(m.group(1)) if m else None
    except Exception as e:
        print(f"[cmd] Could not read volume: {e}")
        return None


def _set_volume(percent):
    percent = max(VOLUME_MIN, min(VOLUME_MAX, int(percent)))
    try:
        if IS_MAC:
            subprocess.run(["osascript", "-e",
                            f"set volume output volume {percent}"],
                           timeout=5)
        else:
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@",
                            f"{percent}%"], timeout=5)
        print(f"[cmd] Volume -> {percent}%")
        return percent
    except Exception as e:
        print(f"[cmd] Could not set volume: {e}")
        return None


def _volume_command(lower, words):
    """Returns True if this was a volume command (handled)."""
    is_volume = "volume" in words
    louder  = is_volume and ("up" in words or "increase" in words) \
        or "louder" in words
    quieter = is_volume and ("down" in words or "decrease" in words or
                             "reduce" in words) \
        or "quieter" in words or "softer" in words

    # an explicit level: "volume fifty percent", "set volume to 80"
    level = None
    if is_volume:
        m = re.search(r"\b(\d{1,3})\b", lower)
        if m:
            level = int(m.group(1))
        else:
            for word, value in _WORD_NUMBERS.items():
                if word in words:
                    level = value
                    break

    if level is not None:
        new = _set_volume(level)
        _speak(f"Volume set to {new} percent." if new is not None
               else "Sorry, I couldn't change the volume.")
        return True

    if louder or quieter:
        current = _get_volume()
        if current is None:
            _speak("Sorry, I couldn't change the volume.")
            return True
        new = _set_volume(current + (VOLUME_STEP if louder else -VOLUME_STEP))
        if new is None:
            _speak("Sorry, I couldn't change the volume.")
        elif louder and new >= VOLUME_MAX:
            _speak("That's as loud as I go!")
        elif quieter and new <= VOLUME_MIN:
            _speak("Any quieter and you won't hear me at all!")
        else:
            _speak("Okay!")
        return True

    return False


# ── Enrollment / recognition ──────────────────────────────────────────────────

_NAME_PATTERNS = (
    r"my name is (\w+)",
    r"remember me as (\w+)",
    r"call me (\w+)",
    r"i am called (\w+)",
)

_ENROLL_PHRASES = ("remember me", "remember my face", "register me",
                   "register my face", "scan my face", "learn my face",
                   "scan me")

_awaiting_name = False
_pending_name = None         # heard a name, waiting for "yes" / "no"
_pending_since = 0.0

_YES_WORDS = {"yes", "yeah", "yep", "yup", "correct", "right", "exactly",
              "sure", "affirmative"}
_NO_WORDS  = {"no", "nope", "wrong", "nah", "incorrect", "not"}


def _extract_name(lower):
    for pat in _NAME_PATTERNS:
        m = re.search(pat, lower)
        if m:
            return m.group(1).title()
    return None


def _hold_window():
    """Keep the conversation window open — Luna just asked something."""
    with state.lock:
        state.awaiting_answer     = True
        state.conversation_active = True


def _ask_confirm(name):
    """Ask "Did you say {name}?" and wait for yes / no in handle()."""
    global _pending_name, _pending_since
    _pending_name, _pending_since = name, time.time()
    _hold_window()
    face.curious()
    _speak(f"Did you say {name}?")


def _enroll_or_confirm(name):
    """Every heard name goes through here: confirm first when configured."""
    if ENROLL_CONFIRM_NAME and ENABLE_FACE_RECOGNITION:
        _ask_confirm(name)
    else:
        _do_enroll(name)


def _ask_name(prompt):
    """Ask for a name; the next utterance is taken as the answer."""
    global _awaiting_name
    _awaiting_name = True           # set BEFORE speaking: never miss the reply
    _hold_window()
    face.curious()
    _speak(prompt)


def start_enrollment_dialog():
    """Offer registration from outside a voice command — the touch-and-hold
    gesture or a proactive "want me to remember you?". Safe from any thread:
    it only flips module flags / shared state and speaks. Returns True when
    the dialog was started."""
    global _pending_name
    if not ENABLE_FACE_RECOGNITION:
        return False
    try:
        _pending_name = None
        _ask_name("Want me to remember you? Tell me your name.")
        return True
    except Exception as e:
        print(f"[cmd] Could not start enrollment dialog: {e}")
        return False


def _wait_result(timeout):
    """Poll state.enroll_result until vision reports or `timeout` passes."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with state.lock:
            result = state.enroll_result
        if result is not None:
            return result
        time.sleep(0.2)
    return None


def _sample_count(pose):
    """Samples vision holds for `pose`, or None when vision isn't loaded
    (deferred import: vision_module pulls in torch + OpenCV models)."""
    try:
        from vision_module import enroll_sample_count
        return enroll_sample_count(pose)
    except Exception:
        return None


def _wait_pose_samples(pose):
    """After a pose prompt: let the head settle, then wait for enough samples
    of this pose (capped at ENROLL_POSE_WAIT)."""
    start = time.time()
    if ENROLL_POSE_SETTLE > 0:
        time.sleep(ENROLL_POSE_SETTLE)
    while time.time() - start < ENROLL_POSE_WAIT:
        count = _sample_count(pose)
        if count is None or count >= ENROLL_FRAMES_PER_POSE:
            return count
        time.sleep(0.1)
    return _sample_count(pose)


def _guided_capture(name, seen):
    """Straight / left / right: vision files samples under state.enroll_pose
    while each prompt is spoken, then stores their average on enroll_finish."""
    poses = tuple(ENROLL_POSES)
    with state.lock:
        state.enroll_result  = None
        state.enroll_finish  = False
        state.enroll_pose    = 0
        state.enroll_request = name

    face.curious(ENROLL_TIMEOUT + len(poses) * (ENROLL_POSE_WAIT + 3.0))
    if seen:
        _speak(f"Okay {name}, let me learn your face.")
    else:
        _speak(f"Come a little closer, {name}, so I can see your face!")

    for i, prompt in enumerate(poses):
        with state.lock:
            state.enroll_pose = i
        _speak(prompt)
        count = _wait_pose_samples(i)
        print(f"[cmd] Enrollment pose {i}: {count} samples")

    with state.lock:
        state.enroll_finish = True
    return _wait_result(ENROLL_TIMEOUT)


def _burst_capture(name, seen):
    """The original one-shot capture: vision gathers ENROLL_FRAMES samples
    while Luna talks, then reports back through state.enroll_result."""
    with state.lock:
        state.enroll_result  = None
        state.enroll_finish  = False
        state.enroll_request = name

    face.curious(ENROLL_TIMEOUT)
    if seen:
        _speak(f"Hold still, {name} — I'm taking a good look at you!")
    else:
        _speak(f"Come a little closer, {name}, so I can see your face!")
    return _wait_result(ENROLL_TIMEOUT)


def _celebrate(name):
    """The "I know you now" beat: the renderer's recognition moment when it
    exists (another module owns it), else a happy face + name banner."""
    moment = getattr(face, "recognition_moment", None)
    if callable(moment):
        try:
            moment(name)
            return
        except Exception as e:
            print(f"[cmd] recognition_moment failed: {e}")
    face.happy(6)
    face.banner(f"Hi {name}!")


def _do_enroll(name):
    """Run the live registration (guided poses or the one-shot burst) and
    announce the outcome. Always leaves the enroll_* state cleared."""
    if not ENABLE_FACE_RECOGNITION:
        _speak("Face memory is turned off in my settings right now.")
        return

    with state.lock:
        seen = state.face_detected

    result = None
    try:
        if ENROLL_GUIDED_POSES and ENROLL_POSES:
            result = _guided_capture(name, seen)
        else:
            result = _burst_capture(name, seen)
    finally:
        with state.lock:
            state.enroll_request = None
            state.enroll_result  = None
            state.enroll_finish  = False
            state.enroll_pose    = 0

    if result is not None and result[1]:
        _celebrate(name)
        _speak(f"Nice to meet you, {name}! I'll remember your face.")
    else:
        face.sad(3)
        _speak("I couldn't get a clear look at your face. "
               "Let's try again with a bit more light!")


def _handle_confirmation(lower, words):
    """The utterance after "Did you say {name}?". Returns True if consumed."""
    global _pending_name
    name = _pending_name
    _pending_name = None

    new = _extract_name(lower)
    if new and new != name:                 # "no, my name is Bob" / "call me Bob"
        _ask_confirm(new)
        return True
    negative = bool(words & _NO_WORDS)
    if new == name or (words & _YES_WORDS and not negative):
        _do_enroll(name)
        return True
    if negative:
        _ask_name("Okay, tell me your name again.")
        return True
    if 0 < len(words) <= 2:                 # they just repeated / corrected it
        _ask_confirm(lower.split()[0].strip(".,!?").title())
        return True
    return False                            # moved on — let brain handle it


def _who_am_i():
    with state.lock:
        name = state.person_name
        seen = state.face_detected
    if name:
        face.happy()
        face.banner(f"Hi {name}!")
        _speak(f"You're {name}, of course! Good to see you.")
    elif seen:
        face.curious()
        _speak("I can see you, but I don't think we've met. "
               "Say: remember me, my name is — and then your name!")
    else:
        _speak("I can't see anyone right now — come stand in front of me!")


def _forget(lower, words):
    if not ENABLE_FACE_RECOGNITION:
        _speak("Face memory is turned off in my settings right now.")
        return
    if "me" in words or "my face" in lower:
        with state.lock:
            name = state.person_name
        if not name:
            _speak("I don't actually know who you are, so there's nothing "
                   "to forget!")
            return
    else:
        m = re.search(r"forget (\w+)", lower)
        name = m.group(1).title() if m else None
    if name and face_store.forget(name):
        _speak(f"Okay. I've forgotten {name}.")
    else:
        _speak("I don't know anyone by that name.")


def _who_do_you_know():
    names = face_store.known_names()
    if not names:
        _speak("I haven't met anyone yet! Say: remember me, my name is — "
               "and then your name.")
    elif len(names) == 1:
        _speak(f"So far I know {names[0]}.")
    else:
        _speak(f"I know {', '.join(names[:-1])} and {names[-1]}.")


# ── Face style ────────────────────────────────────────────────────────────────

def _cycle_style():
    from face_style import theme, apply_style
    nxt = theme.style % 3 + 1
    apply_style(nxt)
    _speak("How do I look?")
    face.happy()


# ── Feedback + teaching (learning.py) ─────────────────────────────────────────
# Only SHORT utterances that are clearly about the last reply count as
# feedback (learning.feedback_command); anything longer is a question and goes
# to brain, where learning scores it implicitly. The correction reply is
# queued for a human to approve — Luna never edits her own knowledge base.

CORRECTION_TIMEOUT = 45.0     # s the "tell me the right answer" stays open
_awaiting_correction = None   # (since, question Luna got wrong)

_TEACH_RE = re.compile(
    r"^(?:(?:hey|ok|okay)\s+)?(?:luna[\s,]+)?(?:please\s+)?"
    r"(?:remember|note|keep in mind)\s+that\s+(.+)$")
_CANCEL_PHRASES = ("never mind", "nevermind", "forget it", "cancel",
                   "nothing", "no", "leave it")
_QUESTION_START = re.compile(
    r"^(what|who|where|when|why|how|which|can|could|do|does|will|would|"
    r"tell me|are|should)\b")
_THANKS_REPLIES = ("Thank you!", "Yay, thank you!", "That makes me happy!")


def _learning():
    """Deferred import — learning is optional and must never break commands."""
    try:
        import learning
        return learning
    except Exception as e:                        # noqa: BLE001
        print(f"[cmd] learning unavailable: {e}")
        return None


def _current_person():
    with state.lock:
        return state.person_name


def _feedback(sign):
    """Explicit feedback on the last reply. False if there was no recent
    answered turn to rate (then brain handles the utterance normally)."""
    global _awaiting_correction
    learning = _learning()
    if learning is None:
        return False
    question = learning.last_answered_question()
    if not learning.explicit_feedback(sign):
        return False
    if sign > 0:
        face.love()
        _speak(random.choice(_THANKS_REPLIES))
        return True
    _awaiting_correction = (time.time(), question)
    _hold_window()
    face.sad(2)
    _speak("Sorry about that. Tell me the right answer and I'll remember it "
           "for approval.")
    return True


def _take_correction(text, lower, question):
    """The utterance after "tell me the right answer". Returns True if it was
    consumed (a correction or a cancel), False for a fresh question."""
    bare = lower.strip(" .,!?")
    if bare in _CANCEL_PHRASES:
        _speak("Okay, never mind.")
        return True
    if _QUESTION_START.match(bare):
        return False                    # they moved on to a new question
    learning = _learning()
    if learning is not None and learning.teach(text, _current_person(),
                                               question=question):
        face.happy(2)
        _speak("Got it, thank you. I'll ask my humans to approve that.")
    else:
        _speak("Sorry, I couldn't write that down.")
    return True


def _teach_command(text, lower):
    """"Remember that …" / "note that …" → pending facts. True if handled."""
    m = _TEACH_RE.match(lower.strip())
    if not m:
        return False
    fact = m.group(1).strip(" .,!?")
    if not fact:
        return False
    learning = _learning()
    if learning is not None and learning.teach(fact, _current_person()):
        face.happy(2)
        _speak("Noted — I'll ask my humans to approve that.")
    else:
        _speak("Sorry, I couldn't write that down.")
    return True


# ── Main entry ────────────────────────────────────────────────────────────────

def handle(text):
    """True if the utterance was a command and has been fully dealt with."""
    global _awaiting_name, _pending_name, _awaiting_correction

    if not ENABLE_VOICE_COMMANDS:
        return False

    lower = text.lower().strip()
    words = set(re.findall(r"[a-z']+", lower))

    # Luna asked "Did you say Alice?" — this utterance is yes / no / a new name.
    if _pending_name is not None:
        if time.time() - _pending_since <= ENROLL_CONFIRM_TIMEOUT:
            if _handle_confirmation(lower, words):
                return True
        else:
            _pending_name = None

    # Luna asked "what's your name?" — this utterance IS the answer.
    if _awaiting_name:
        _awaiting_name = False
        name = _extract_name(lower)
        if name is None and 0 < len(words) <= 2:
            # A one-or-two word reply is almost certainly just the name.
            name = lower.split()[0].strip(".,!?").title()
        if name:
            _enroll_or_confirm(name)
        else:
            _speak("Sorry, I didn't catch a name. Say: remember me, "
                   "my name is — and then your name.")
        return True

    # Luna asked for the right answer after "that's wrong" — this is it.
    if _awaiting_correction is not None:
        since, question = _awaiting_correction
        _awaiting_correction = None
        if time.time() - since <= CORRECTION_TIMEOUT:
            if _take_correction(text, lower, question):
                return True

    # enrollment — with the name attached, or ask for it
    if any(p in lower for p in _ENROLL_PHRASES):
        name = _extract_name(lower)
        if name:
            _enroll_or_confirm(name)
        else:
            _ask_name("Of course! What's your name?")
        return True

    # a bare "my name is X" also enrolls (live registration, per the sketch)
    name = _extract_name(lower)
    if name and ENABLE_FACE_RECOGNITION:
        with state.lock:
            known = state.person_name
        if known == name:
            _speak(f"I know! Good to see you, {name}.")
        else:
            _enroll_or_confirm(name)
        return True

    # "good answer" / "that's wrong" — only when there IS a recent reply
    learning = _learning()
    sign = learning.feedback_command(text) if learning is not None else 0
    if sign and _feedback(sign):
        return True

    if _teach_command(text, lower):
        return True

    if "who am i" in lower or "do you know me" in lower \
            or "do you recognize me" in lower or "do you recognise me" in lower:
        _who_am_i()
        return True

    if "who do you know" in lower or "who have you met" in lower:
        _who_do_you_know()
        return True

    if "forget" in words and ("me" in words or re.search(r"forget \w+", lower)):
        _forget(lower, words)
        return True

    if _volume_command(lower, words):
        return True

    if "go to sleep" in lower or "sleep now" in lower or "good night" in lower \
            or "goodnight" in lower:
        _speak("Okay, good night! Wake me any time.")
        face.sleep()
        return True

    if "wake up" in lower:
        face.wake()
        _speak("I'm awake! What's up?")
        return True

    if "change your face" in lower or "switch your face" in lower \
            or "change your style" in lower or "next face" in lower:
        _cycle_style()
        return True

    return False
