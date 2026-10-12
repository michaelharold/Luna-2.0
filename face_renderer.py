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
  4. Vision        — the camera's emotion, once it has settled (EmotionFilter)
"""

import time

from shared_state import state
from robot_face import RobotFace
from config import SLEEP_AFTER_FRAMES, EMOTION_ONSET_SECS, EMOTION_MIN_HOLD_SECS

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


class EmotionFilter:
    """Turns the camera's flickering emotion into a mood worth wearing.

    Vision guesses 8 times a second and its majority vote still flips on a
    blink or a missed detection. Copied straight onto the face that gave a
    smile for one second, then a snap to something else. Two rules fix it:

      onset — a new mood must be reported continuously for `onset` seconds
              before it is adopted, so a blip never reaches the face
      hold  — an adopted mood is worn for at least `hold` seconds, so a
              smile, once it appears, stays long enough to read as one

    Pure apart from the `now` it is handed, so it is unit-testable."""

    def __init__(self, onset=EMOTION_ONSET_SECS, hold=EMOTION_MIN_HOLD_SECS):
        self.onset = onset
        self.hold = hold
        self.current = "neutral"
        self.since = float("-inf")       # when `current` was adopted
        self._candidate = None
        self._candidate_since = 0.0

    def update(self, name, now):
        if name == self.current:
            self._candidate = None
            return self.current
        if name != self._candidate:
            self._candidate, self._candidate_since = name, now
        if (now - self._candidate_since >= self.onset
                and now - self.since >= self.hold):
            self.current, self.since = name, now
            self._candidate = None
        return self.current


def _read_state():
    """Take one consistent snapshot, expiring a stale override while we hold
    the lock. Reading these one at a time outside the lock let the face act on
    a mix of two different frames."""
    now = time.time()
    with state.lock:
        if state.face_override and now > state.face_override_until:
            state.face_override = None
        return (state.face_override, state.emotion, state.speaking)


def choose_expression(face, override, mood, speaking):
    """The priority ladder. `mood` is the settled vision mood, already an
    expression name. Returns an expression name, or None to hold the current
    one. Pure apart from reading `face.idle_frames`, so it is easy to reason
    about and to test."""
    if override:
        return override
    if face.idle_frames > SLEEP_AFTER_FRAMES:
        return "sleeping"
    if speaking:
        return None                      # keep the face she started talking with
    return mood


def renderer_loop():
    face = RobotFace()
    moods = EmotionFilter()
    while True:
        override, emotion, speaking = _read_state()
        mood = moods.update(EMOTION_MAP.get(emotion, "neutral"), time.time())
        wanted = choose_expression(face, override, mood, speaking)
        if wanted:
            face.set_state(wanted)       # no-op when already in that state
        face.draw()
