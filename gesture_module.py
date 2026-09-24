"""
gesture_module.py — detects hand gestures and pointing direction.

Gestures detected:
  WAVE       — hand present + wrist moving horizontally (smile + "hi" + servo wave)
  THUMBS_UP  — thumb up, other fingers folded (excited face + thank-you)
  PEACE      — index + middle up, others folded (excited face + starburst)
  OPEN_PALM  — all five fingers extended and still (friendly acknowledgement)
  POINT_LEFT / POINT_RIGHT / POINT_UP / POINT_DOWN — index finger direction
  HAND       — generic hand visible

All gestures come from the same MediaPipe hand landmarks — no extra models,
no extra CPU cost on the Pi.
"""

import mediapipe as mp
import threading
import time
from config import (ENABLE_GESTURES, GESTURE_FPS, GESTURE_STABLE_FRAMES,
                    THUMB_RISE_RATIO, THUMB_ABOVE_INDEX_RATIO,
                    HAND_MODEL_COMPLEXITY, HAND_DETECT_CONF,
                    HAND_TRACK_CONF, HAND_MIN_SPAN, HAND_MIN_HANDEDNESS,
                    GESTURE_DEBUG, GESTURE_REACT_STABLE_FRAMES)
from shared_state import state

mp_hands = mp.solutions.hands

# MediaPipe's own confidence gates. These were previously left unset, which
# means they defaulted to 0.5 — permissive enough that a face, an ear or a
# background object gets reported as a "hand", complete with 21 nonsense
# landmarks. Tightening the THUMBS_UP pose test did nothing for that case,
# because the pose maths was being fed garbage in the first place. This is the
# upstream fix: reject weak detections before they are ever classified.
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=HAND_MODEL_COMPLEXITY,
    min_detection_confidence=HAND_DETECT_CONF,
    min_tracking_confidence=HAND_TRACK_CONF,
)

# Wave detection state
_prev_wrist_x  = None
_wave_dx_hist  = []
_WAVE_THRESH   = 0.04    # normalised wrist displacement per frame to count as wave
_WAVE_HIST_LEN = 8       # frames

# Gesture debounce state. model_complexity=0 keeps MediaPipe cheap enough for
# the Pi, but its landmarks jitter, so a single bad frame used to be enough to
# make Luna speak. A gesture must now hold for GESTURE_STABLE_FRAMES frames
# before it is published at all.
_gesture_hist = []
_last_logged  = "__init__"   # sentinel: never equals a real gesture

# Gestures that make Luna DO something (speak / move servos). These must be
# held deliberately; the POINT_* gestures only steer her eyes and stay quick.
REACTIVE_GESTURES = {"WAVE", "THUMBS_UP", "PEACE", "OPEN_PALM"}


_thumbs_noted = False   # the current held thumbs-up was already reported


def _note_thumbs(gesture):
    """A held thumbs-up right after a reply is a reward for learning.py —
    reported once per gesture, not once per frame. Deferred + guarded:
    learning must never stall or crash the gesture thread."""
    global _thumbs_noted
    held = gesture == "THUMBS_UP"
    if held and not _thumbs_noted:
        try:
            import learning
            learning.note_gesture("thumbs_up")
        except Exception as e:                   # noqa: BLE001
            print(f"[gesture] learning.note_gesture failed: {e}")
    _thumbs_noted = held


def _reset_wave():
    """Drop the wave motion history. Called whenever the tracked hand is lost
    or rejected, so displacement across a detection gap can't read as a wave."""
    global _prev_wrist_x, _wave_dx_hist
    _prev_wrist_x = None
    _wave_dx_hist = []


def _get_landmark(lm_list, idx):
    lm = lm_list[idx]
    return lm.x, lm.y


def _hand_span(lm):
    """Largest edge of the hand's bounding box, in normalised frame units.

    A deliberate gesture is made towards the camera and occupies a real
    fraction of the frame. Spurious detections (a face, a patterned shirt,
    something in the background) are typically small, so this is a cheap way
    to throw them out before classifying a pose."""
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def _is_plausible_hand(landmarks, handedness):
    """Reject detections that are too small or too uncertain to act on."""
    if _hand_span(landmarks) < HAND_MIN_SPAN:
        return False
    if handedness is not None:
        try:
            if handedness.classification[0].score < HAND_MIN_HANDEDNESS:
                return False
        except (AttributeError, IndexError):
            pass
    return True


def _finger_states(lm):
    """Return dict of which fingers are extended.
    A finger counts as extended when its tip is farther from the wrist
    than its PIP joint (scale-invariant, works at any distance)."""
    wrist = lm[0]

    def dist(a):
        return ((a.x - wrist.x) ** 2 + (a.y - wrist.y) ** 2) ** 0.5

    return {
        # thumb: tip vs IP joint
        "thumb":  dist(lm[4])  > dist(lm[3])  * 1.1,
        "index":  dist(lm[8])  > dist(lm[6])  * 1.1,
        "middle": dist(lm[12]) > dist(lm[10]) * 1.1,
        "ring":   dist(lm[16]) > dist(lm[14]) * 1.1,
        "pinky":  dist(lm[20]) > dist(lm[18]) * 1.1,
    }


def _classify(landmarks):
    """Return gesture string or None."""
    global _prev_wrist_x, _wave_dx_hist

    wrist     = landmarks[0]
    index_tip = landmarks[8]
    index_mcp = landmarks[5]

    wrist_x = wrist.x

    # ── Wave detection ──────────────────────────────────────────────────────
    if _prev_wrist_x is not None:
        dx = abs(wrist_x - _prev_wrist_x)
        _wave_dx_hist.append(dx)
        if len(_wave_dx_hist) > _WAVE_HIST_LEN:
            _wave_dx_hist.pop(0)
        if len(_wave_dx_hist) == _WAVE_HIST_LEN:
            avg_dx = sum(_wave_dx_hist) / _WAVE_HIST_LEN
            if avg_dx > _WAVE_THRESH:
                _prev_wrist_x = wrist_x
                return "WAVE"

    _prev_wrist_x = wrist_x

    # ── Static hand poses ───────────────────────────────────────────────────
    fingers  = _finger_states(landmarks)
    n_up     = sum(fingers.values())

    # THUMBS_UP — thumb up, all four fingers folded.
    #
    # This used to be `fingers["thumb"] and landmarks[4].y < landmarks[2].y
    # and n_up == 1`, which fired on a resting fist: a thumb tip one pixel
    # above its own base cleared the test, and a curled thumb is still farther
    # from the wrist than its IP joint, so `fingers["thumb"]` was true too.
    # Luna would thank people for closing their hand. Both checks now need a
    # real margin, and the thumb must clear the index knuckle — a fist can't
    # fake that without actually sticking the thumb up.
    # Both margins are measured RELATIVE TO HAND SIZE. They used to be absolute
    # normalised-frame distances, which fails because the same pose yields
    # bigger numbers the closer the hand is to the camera. Captured from the
    # live camera, the false triggers sat at rise 0.087-0.105 against a 0.08
    # threshold — squeaking over it — while a real thumbs up is far clearer.
    # Divided by hand span the two separate cleanly: ~0.42 for a relaxed hand
    # whose thumb happens to sit high, ~0.67 for a deliberate thumbs up.
    span         = max(1e-6, _hand_span(landmarks))
    thumb_rise   = (landmarks[2].y - landmarks[4].y) / span   # + = above base
    thumb_vs_idx = (landmarks[5].y - landmarks[4].y) / span   # + = above knuckle
    if (n_up == 1 and fingers["thumb"]
            and thumb_rise   > THUMB_RISE_RATIO
            and thumb_vs_idx > THUMB_ABOVE_INDEX_RATIO):
        return "THUMBS_UP"

    # PEACE — index + middle extended, ring + pinky folded
    if (fingers["index"] and fingers["middle"]
            and not fingers["ring"] and not fingers["pinky"]):
        return "PEACE"

    # OPEN_PALM — all five fingers extended
    if n_up == 5:
        return "OPEN_PALM"

    # ── Pointing direction (index finger extended) ──────────────────────────
    # vector from MCP (knuckle) to tip
    vec_x = index_tip.x - index_mcp.x
    vec_y = index_tip.y - index_mcp.y   # y increases downward in image coords
    magnitude = (vec_x**2 + vec_y**2) ** 0.5

    if magnitude > 0.08:   # finger clearly extended
        if abs(vec_x) > abs(vec_y):
            return "POINT_LEFT" if vec_x < 0 else "POINT_RIGHT"
        else:
            return "POINT_UP" if vec_y < 0 else "POINT_DOWN"

    return "HAND"


def gesture_loop():
    sleep_time = 1.0 / GESTURE_FPS

    while True:
        t0 = time.monotonic()

        with state.lock:
            frame = state.frame

        if frame is None:
            time.sleep(0.1)
            continue

        rgb     = frame[:, :, ::-1]
        results = hands.process(rgb)

        raw      = None
        look_dir = None

        if results.multi_hand_landmarks:
            landmarks  = results.multi_hand_landmarks[0].landmark
            handedness = None
            if results.multi_handedness:
                handedness = results.multi_handedness[0]
            if _is_plausible_hand(landmarks, handedness):
                raw = _classify(landmarks)
            else:
                # Something was detected but it does not look like a hand
                # someone is deliberately holding up. Reset the wave tracker
                # so this noise can't contribute motion to a future wave.
                _reset_wave()
        else:
            # No hand at all. Without this the last known wrist x survives the
            # gap, so a hand reappearing elsewhere in frame produced one huge
            # displacement and registered as a wave.
            _reset_wave()

        # Publish only once a gesture has been HELD, not merely seen.
        #
        # Measured against the live camera with nobody gesturing on purpose: a
        # hand resting near the keyboard is detected by MediaPipe with
        # handedness 0.99-1.0 and a span of ~0.28 — it sails straight through
        # the plausibility gates above, because it genuinely IS a hand. It just
        # isn't doing anything. That idle hand still got classified POINT_DOWN,
        # OPEN_PALM and even WAVE as it moved about, which is what produced
        # spontaneous reactions. No confidence threshold can fix that: the
        # distinguishing feature of a deliberate gesture is that it is HELD.
        #
        # Reactive gestures (the ones that make Luna speak or move) therefore
        # need a longer hold than the pointing ones, which only steer her eyes
        # and should stay responsive.
        _gesture_hist.append(raw)
        if len(_gesture_hist) > GESTURE_REACT_STABLE_FRAMES:
            _gesture_hist.pop(0)

        need = (GESTURE_REACT_STABLE_FRAMES
                if raw in REACTIVE_GESTURES else GESTURE_STABLE_FRAMES)

        gesture = None
        if len(_gesture_hist) >= need and len(set(_gesture_hist[-need:])) == 1:
            gesture = raw

        # Set GESTURE_DEBUG=True in config.py to see exactly what Luna thinks
        # she is looking at. Prints only on change, so it stays readable.
        global _last_logged
        if GESTURE_DEBUG and gesture != _last_logged:
            _last_logged = gesture
            if gesture and results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark
                score = "?"
                if results.multi_handedness:
                    try:
                        score = f"{results.multi_handedness[0].classification[0].score:.2f}"
                    except (AttributeError, IndexError):
                        pass
                print(f"[gesture] {gesture}  span={_hand_span(lm):.2f} "
                      f"handedness={score}")
            elif gesture is None:
                print("[gesture] —")

        # Map pointing gestures to look directions for the eyes
        if gesture == "POINT_LEFT":
            look_dir = "left"
        elif gesture == "POINT_RIGHT":
            look_dir = "right"
        elif gesture == "POINT_UP":
            look_dir = "up"
        elif gesture == "POINT_DOWN":
            look_dir = "down"

        _note_thumbs(gesture)

        with state.lock:
            state.gesture  = gesture
            state.look_dir = look_dir

        elapsed = time.monotonic() - t0
        rem = sleep_time - elapsed
        if rem > 0:
            time.sleep(rem)


def start_gesture():
    if not ENABLE_GESTURES:
        # Gesture recognition disabled in config — don't start the MediaPipe
        # tracker at all (saves CPU) and leave state.gesture as None so
        # behavior_engine never fires a gesture reaction.
        state.gesture = None
        print("[gesture] Disabled (ENABLE_GESTURES=False) — hand gestures ignored")
        return
    t = threading.Thread(target=gesture_loop, daemon=True)
    t.name = "gesture"
    t.start()