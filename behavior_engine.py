"""
behavior_engine.py — reacts to gestures with face state + voice + servos.

Wave       → happy face + "Hi there!" + servo wave
Thumbs up  → excited face (starburst) + "Thank you!" + both arms up
Peace      → excited face (starburst) + "Peace!" + right arm up
Open palm  → happy face (silent friendly acknowledgement)
Point      → eyes follow direction (handled in robot_face via look_dir)

Interrupt rules (order matters — answering ALWAYS wins):
  • Luna speaking or processing an answer  → gestures ignored completely
  • Conversation window active (attentive) → face + servo only, no speech,
    so a wave never talks over the user's next question
  • Fully idle                             → full reaction with speech
"""

import threading
import time
from shared_state import state
from servo_module import servo
from config import ROBOT_NAME, GESTURE_REACT_COOLDOWN
import face

_last_react   = {}     # gesture name → last reaction time
_prev_gesture = None   # what state.gesture was on the previous poll


def _cooled(gesture):
    now = time.time()
    if now - _last_react.get(gesture, 0.0) < GESTURE_REACT_COOLDOWN:
        return False
    _last_react[gesture] = now
    return True


def _fires(gesture, current, previous):
    """True only on the *transition* into `gesture`, and only if cooled.

    The cooldown alone used to be the whole guard, which made reactions
    level-triggered: someone leaving a hand in frame got thanked again every
    time the cooldown lapsed, forever. Requiring an edge means one hold = one
    reaction; you have to lower the hand and raise it again to get another.
    The cooldown stays as a second guard against a gesture flickering off and
    straight back on.
    """
    return current == gesture and previous != gesture and _cooled(gesture)




def behavior_loop():
    from text_to_speech import speak   # deferred — avoids circular import

    global _prev_gesture

    while True:
        with state.lock:
            ready   = state.ready
            gesture = state.gesture
            # tts_active covers the synthesis window too, so a gesture can't
            # queue an arm move into the middle of an answer that is about to
            # start playing
            busy    = (state.speaking or state.tts_active
                       or state.luna_mode in ("processing", "speaking"))
            in_convo = state.conversation_active

        if ready and not busy:
            # in an active conversation react silently (face + servo only)
            quiet = in_convo

            if _fires("WAVE", gesture, _prev_gesture):
                face.happy()
                servo.wave()
                if not quiet:
                    speak("Hi there! I am " + ROBOT_NAME, can_drop=True)

            elif _fires("THUMBS_UP", gesture, _prev_gesture):
                face.excited()
                servo.arms_up()
                if not quiet:
                    speak("Thank you! You are awesome!", can_drop=True)

            elif _fires("PEACE", gesture, _prev_gesture):
                face.excited()
                servo.arm_up("right")
                if not quiet:
                    speak("Peace!", can_drop=True)

            elif _fires("OPEN_PALM", gesture, _prev_gesture):
                # silent friendly acknowledgement — just a happy face
                face.happy()

            # Only advance the edge detector while Luna is actually watching.
            # Updating it during a busy/not-ready spell would consume the
            # transition silently and swallow the reaction entirely.
            _prev_gesture = gesture

        time.sleep(0.2)


def start_behavior():
    t = threading.Thread(target=behavior_loop, daemon=True)
    t.name = "behavior"
    t.start()
