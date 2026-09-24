# face_renderer.py
"""
face_renderer.py — decides which expression Luna should be wearing, frame by
frame, and hands it to RobotFace.

This is only the *policy*: the priority order below is the whole file. Drawing
lives in robot_face.py, expression design lives in expressions.py, and anything
that wants to force a face calls face.py.

Priority, highest first:
  1. Override      — an explicit face.happy() / gesture reaction / compliment
  2. Sleep         — nobody around for long enough
  3. Speaking      — hold the current face; the mouth is driven by audio
  4. Vision        — whatever emotion the camera reports
"""

import time

from shared_state import state
from robot_face import RobotFace
from config import SLEEP_AFTER_FRAMES

#: DeepFace's labels -> our expression names. Anything unlisted falls back to
#: neutral, so a new label from the model can never crash the render thread.
EMOTION_MAP = {
    "Happy":    "happy",
    "Surprise": "surprised",
    "Angry":    "angry",
    "Disgust":  "angry",
    "Fear":     "surprised",
    "Sad":      "sad",
    "Neutral":  "neutral",
}


def _read_state():
    """Take one consistent snapshot, expiring a stale override while we hold
    the lock. Reading these one at a time outside the lock let the face act on
    a mix of two different frames."""
    now = time.time()
    with state.lock:
        if state.face_override and now > state.face_override_until:
            state.face_override = None
        return (state.face_override, state.emotion, state.speaking)


def choose_expression(face, override, emotion, speaking):
    """The priority ladder. Returns an expression name, or None to hold the
    current one. Pure apart from reading `face.idle_frames`, so it is easy to
    reason about and to test."""
    if override:
        return override
    if face.idle_frames > SLEEP_AFTER_FRAMES:
        return "sleeping"
    if speaking:
        return None                      # keep the face she started talking with
    return EMOTION_MAP.get(emotion, "neutral")


def renderer_loop():
    face = RobotFace()
    while True:
        wanted = choose_expression(face, *_read_state())
        if wanted:
            face.set_state(wanted)       # no-op when already in that state
        face.draw()
