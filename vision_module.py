# vision_module.py
"""
vision_module.py — everything Luna understands from a camera frame.

One thread, VISION_FPS times a second:

  detect   — YuNet (ONNX, fast, gives the 5 landmarks SFace needs). Falls
             back to the old Haar cascade if the model is missing or
             FACE_DETECTOR="haar" — Luna must keep seeing either way.
  emotion  — the fine-tuned PyTorch MobileNet, exactly as before. DeepFace
             (TensorFlow) is now opt-in via ENABLE_DEEPFACE: it only improves
             "Sad" and its ~1-2 GB of RAM is budgeted to the local LLM.
  recognise— SFace embeddings every RECOGNIZE_EVERY frames on the largest
             face, matched against the on-device face store. A name must win
             twice in a row before it's published — single-frame matches
             flicker, and greeting the wrong person is worse than greeting
             nobody.
  enroll   — when commands.py sets state.enroll_request, this thread collects
             embedding samples of the largest face. Guided mode
             (ENROLL_GUIDED_POSES) files them under state.enroll_pose —
             straight / left / right — keeping the latest few per pose (the
             ones taken after the head actually turned) until commands sets
             enroll_finish; the legacy burst stops at ENROLL_FRAMES. At least
             ENROLL_MIN_FRAMES samples are averaged into the face store,
             otherwise the enrollment fails rather than storing a bad face.
  log      — enrolled people are logged to data/face_log.jsonl when they
             (re)appear, at most once per FACE_LOG_COOLDOWN per person.
  scene    — every tick feeds scene_memory.observe() so the LLM prompt can
             say who came and went; person_since stamps when the current
             person was first recognised.

Performance: YuNet runs on a VISION_DETECT_SCALE-downscaled copy of the frame
(0.5 = a quarter of the pixels — the biggest single cost on the Pi); its boxes
and landmarks are scaled back up (scale_rows) so the emotion crop and SFace
alignment still use the FULL-resolution frame. scripts/bench_vision.py
measures the difference.
"""

import json
import os
import threading
import time

import cv2
import numpy as np
import torch

from config import (MODEL_PATH, VISION_FPS, VISION_FPS_THINKING, DEEPFACE_EVERY,
                    SAD_CONFIDENCE, EMOTION_HISTORY_LEN, ENABLE_DEEPFACE,
                    ENABLE_FACE_RECOGNITION, FACE_DETECTOR,
                    YUNET_MODEL_PATH, SFACE_MODEL_PATH, FACE_DETECT_CONF,
                    RECOGNIZE_EVERY, FACE_LOG_PATH, FACE_LOG_ENABLED,
                    FACE_LOG_COOLDOWN, ENROLL_FRAMES,
                    ENROLL_GUIDED_POSES, ENROLL_FRAMES_PER_POSE,
                    ENROLL_MIN_FRAMES, VISION_DETECT_SCALE,
                    PERSON_AWAY_SECS, RECOG_SWITCH_SECS,
                    FACE_AUTO_LEARN, FACE_LEARN_MIN_SIM,
                    FACE_LEARN_NOVEL_BELOW, FACE_LEARN_MARGIN,
                    FACE_LEARN_STABLE_SECS, FACE_LEARN_EVERY_SECS,
                    FACE_LEARN_SAMPLES)
import config as _config
from shared_state import state
from models.emotion_mobilenet import EmotionMobileNet
import face_store
import console

# ── PyTorch emotion model ─────────────────────────────────────────────────────
EMOTIONS_PT = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Surprise"]

console.detail("[vision] Loading PyTorch emotion model...")
pt_model = EmotionMobileNet(num_classes=6)
pt_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
pt_model.eval()
console.detail("[vision] PyTorch model ready")

# ── Face detector — YuNet with a Haar fallback ────────────────────────────────
_haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

_yunet = None
if FACE_DETECTOR == "yunet":
    try:
        if hasattr(cv2, "FaceDetectorYN") and os.path.exists(YUNET_MODEL_PATH):
            _yunet = cv2.FaceDetectorYN.create(
                YUNET_MODEL_PATH, "", (320, 240),
                score_threshold=FACE_DETECT_CONF,
                nms_threshold=0.3, top_k=500)
            console.detail("[vision] YuNet face detector ready")
        else:
            print("[vision] YuNet unavailable "
                  f"(model at {YUNET_MODEL_PATH}?) — using Haar cascade")
    except Exception as e:
        _yunet = None
        print(f"[vision] YuNet failed to load ({e}) — using Haar cascade")

_yunet_size = (0, 0)


def scale_rows(rows, factor):
    """Map YuNet rows found on a resized frame back to full-frame
    coordinates: x, y, w, h and the 5 landmark (x, y) pairs — columns 0..13 —
    are multiplied by `factor`; the score (column 14) is left alone. Pure,
    returns a new float32 array (None stays None)."""
    if rows is None:
        return None
    out = np.array(rows, dtype=np.float32, copy=True)
    if out.size == 0:
        return out
    if out.ndim == 1:
        out[:14] *= factor
    else:
        out[:, :14] *= factor
    return out


def _detect_scale(scale):
    s = VISION_DETECT_SCALE if scale is None else scale
    try:
        s = float(s)
    except (TypeError, ValueError):
        return 1.0
    return s if 0.05 <= s < 1.0 else 1.0


def _detect_faces(frame, scale=None):
    """Return a list of detections, largest face first. Each detection is
    (x, y, w, h, row) where `row` is the full YuNet output row (needed by
    SFace's alignCrop) or None when Haar produced it. Detection runs on a
    `scale`-sized copy (default VISION_DETECT_SCALE); all coordinates
    returned are in the full frame."""
    global _yunet_size
    s = _detect_scale(scale)
    if s < 1.0:
        small = cv2.resize(frame, None, fx=s, fy=s,
                           interpolation=cv2.INTER_AREA)
    else:
        small = frame
    h, w = small.shape[:2]

    if _yunet is not None:
        if (w, h) != _yunet_size:
            _yunet.setInputSize((w, h))
            _yunet_size = (w, h)
        try:
            _, rows = _yunet.detect(small)
        except Exception:
            rows = None
        if rows is None:
            return []
        if s < 1.0:
            rows = scale_rows(rows, 1.0 / s)
        dets = []
        for row in rows:
            x, y, fw, fh = (int(row[0]), int(row[1]),
                            int(row[2]), int(row[3]))
            if fw > 0 and fh > 0:
                dets.append((x, y, fw, fh, row))
        return sorted(dets, key=lambda d: -(d[2] * d[3]))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    min_side = max(20, int(round(60 * s)))
    faces = _haar.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,
                                   minSize=(min_side, min_side),
                                   flags=cv2.CASCADE_SCALE_IMAGE)
    inv = 1.0 / s
    return sorted([(int(x * inv), int(y * inv), int(fw * inv), int(fh * inv),
                    None) for (x, y, fw, fh) in faces],
                  key=lambda d: -(d[2] * d[3]))


# ── SFace recogniser ──────────────────────────────────────────────────────────
_sface = None
if ENABLE_FACE_RECOGNITION:
    try:
        if hasattr(cv2, "FaceRecognizerSF") and os.path.exists(SFACE_MODEL_PATH):
            _sface = cv2.FaceRecognizerSF.create(SFACE_MODEL_PATH, "")
            console.detail(f"[vision] SFace recogniser ready "
                  f"({len(face_store.known_names())} people enrolled)")
        else:
            print(f"[vision] SFace unavailable (model at {SFACE_MODEL_PATH}?) "
                  f"— face recognition off")
    except Exception as e:
        _sface = None
        print(f"[vision] SFace failed to load ({e}) — face recognition off")


def _embed(frame, det_row):
    """One 128-d SFace embedding for a YuNet detection, or None. Haar rows
    can't be aligned (no landmarks), so recognition requires YuNet."""
    if _sface is None or det_row is None:
        return None
    try:
        aligned = _sface.alignCrop(frame, det_row)
        return _sface.feature(aligned).flatten().copy()
    except Exception:
        return None


# ── DeepFace (optional — see ENABLE_DEEPFACE in config) ───────────────────────
DEEPFACE_OK = False
if ENABLE_DEEPFACE:
    console.detail("[vision] Loading DeepFace (ENABLE_DEEPFACE=True)...")
    try:
        from deepface import DeepFace
        _dummy = np.zeros((48, 48, 3), dtype=np.uint8)
        try:
            DeepFace.analyze(_dummy, actions=["emotion"],
                             enforce_detection=False, silent=True)
        except Exception:
            pass
        DEEPFACE_OK = True
        console.detail("[vision] DeepFace ready")
    except ImportError:
        print("[vision] DeepFace not installed — Sad detection reduced")

DEEPFACE_MAP = {
    "angry":   "Angry",   "disgust": "Disgust",
    "fear":    "Fear",    "happy":   "Happy",
    "sad":     "Sad",     "surprise":"Surprise",
    "neutral": "Neutral",
}

_history    = []
_df_counter = 0


def _majority(history):
    if not history:
        return "Neutral"
    return max(set(history), key=history.count)


def _run_pytorch(gray_48):
    face_arr = gray_48.astype("float32") / 255.0
    tensor = torch.from_numpy(face_arr).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        pred = int(torch.argmax(pt_model(tensor)))
    return EMOTIONS_PT[pred]


def _run_deepface(bgr_crop):
    try:
        r = DeepFace.analyze(bgr_crop, actions=["emotion"],
                             enforce_detection=False, silent=True)
        if isinstance(r, list):
            r = r[0]
        dominant = r["dominant_emotion"]
        conf     = r["emotion"].get(dominant, 0.0)
        return DEEPFACE_MAP.get(dominant.lower(), "Neutral"), conf
    except Exception:
        return None


# ── Recognition smoothing + sighting log ──────────────────────────────────────
_recog_last   = None    # last raw match (may be None = stranger)
_recog_conf   = 0.0
_recog_streak = 0
_recog_first  = 0.0     # when the current run of _recog_last began
_faceless_since = None  # first faceless frame, while no face is in view
_face_run_start = None  # when the current unbroken stretch of "a face is in
                        # view" began (a gap > FACE_GAP_SECS starts a new one)
_confirmed_at = None    # last time the published name was matched
FACE_GAP_SECS = 1.5
_last_logged  = {}      # name -> time.time() of last log entry


def _log_sighting(name, conf):
    if not FACE_LOG_ENABLED:
        return
    now = time.time()
    if now - _last_logged.get(name, 0.0) < FACE_LOG_COOLDOWN:
        return
    _last_logged[name] = now
    try:
        os.makedirs(os.path.dirname(FACE_LOG_PATH) or ".", exist_ok=True)
        with open(FACE_LOG_PATH, "a") as f:
            f.write(json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "name": name,
                "confidence": round(float(conf), 3),
            }) + "\n")
    except Exception as e:
        print(f"[vision] Could not write face log: {e}")


def _update_recognition(frame, det_row, now=None):
    """Match the largest face against the store, publish the (debounced)
    result, and log enrolled people as they appear."""
    global _recog_last, _recog_conf, _recog_streak, _recog_first, _confirmed_at

    now = time.time() if now is None else now
    emb = _embed(frame, det_row)
    if emb is None:
        return
    name, conf = face_store.match(emb)
    _maybe_learn(name, conf, emb, now)

    if name == _recog_last:
        _recog_streak += 1
    else:
        _recog_last, _recog_conf, _recog_streak = name, conf, 1
        _recog_first = now

    # Two agreeing samples before anything is published — recognition
    # flicker must never cause a mis-greeting.
    if _recog_streak < 2:
        return

    with state.lock:
        previous = state.person_name
        # Letting go of a name Luna knows takes longer than learning one: a
        # turned head or a blurry frame fails to match for a moment, and that
        # must not make her forget who she is talking to (and then
        # "recognise" them again a second later).
        if previous and name == previous:
            _confirmed_at = now
        # The face never left the frame since the name was last confirmed —
        # failed matches (looking down at the keyboard, a turned head) do not
        # make it someone else. Only another KNOWN face takes over.
        if (previous and name is None and _face_run_start is not None
                and _confirmed_at is not None and _face_run_start <= _confirmed_at):
            return
        if previous and name != previous and now - _recog_first < RECOG_SWITCH_SECS:
            return
        state.person_name = name
        state.person_conf = conf if name else 0.0
        if name != previous:
            state.person_since = now if name else 0.0
        if name:
            _confirmed_at = now
    if name and name != previous:
        print(f"[vision] Recognised {name} ({conf:.2f})")
        _log_sighting(name, conf)
        _sight_moment(name)


_sighted = {}      # name -> time.time() of the last on-sight "Hi {name}!"


def _sight_moment(name):
    """The moment a registered person is recognised: "Hi {name}!" on screen
    with the excited face — straight away, not only when proactive.py
    decides to greet out loud. Once per RECOG_SIGHT_COOLDOWN per person, and
    never in the middle of a registration."""
    if not getattr(_config, "RECOG_ON_SIGHT", False):
        return
    now = time.time()
    if now - _sighted.get(name, 0.0) < getattr(_config, "RECOG_SIGHT_COOLDOWN", 60.0):
        return
    with state.lock:
        if state.enroll_request:
            return
    _sighted[name] = now
    try:
        import face
        face.recognition_moment(name)
    except Exception as e:                       # noqa: BLE001
        print(f"[vision] recognition moment failed: {e}")


# ── Learning new looks (face_store keeps several per person) ─────────────────
_learn_buf = []        # consecutive qualifying embeddings of _learn_for
_learn_for = None
_learned_at = {}       # name -> time.time() of their last learned look


def should_learn(name, conf, published, since, faces, enrolling, runner_up,
                 last_learned, now):
    """May this sighting count toward a new look for `name`? Only when Luna
    is VERY sure: the person shown is `name` and has been for
    FACE_LEARN_STABLE_SECS, exactly one face is in view, no registration is
    running, the match is strong (>= FACE_LEARN_MIN_SIM, far above the 0.36
    cut-off) yet new (< FACE_LEARN_NOVEL_BELOW), every other enrolled person
    is at least FACE_LEARN_MARGIN further away, and no look was learned for
    them in the last FACE_LEARN_EVERY_SECS. Pure."""
    return bool(FACE_AUTO_LEARN and name is not None and name == published
                and since and now - since >= FACE_LEARN_STABLE_SECS
                and faces == 1 and not enrolling
                and FACE_LEARN_MIN_SIM <= conf < FACE_LEARN_NOVEL_BELOW
                and runner_up <= conf - FACE_LEARN_MARGIN
                and now - last_learned >= FACE_LEARN_EVERY_SECS)


def _maybe_learn(name, conf, emb, now):
    """Collect qualifying sightings; FACE_LEARN_SAMPLES in a row become one
    new look (their average) in face_store. A single sighting that does not
    qualify starts the count again. Returns True when a look was added."""
    global _learn_for
    if name != _learn_for:
        _learn_buf.clear()
        _learn_for = name
    with state.lock:
        published, since = state.person_name, state.person_since
        faces, enrolling = state.face_count, bool(state.enroll_request)
    other = face_store.runner_up(emb, name) if name else 1.0
    if not should_learn(name, conf, published, since, faces, enrolling, other,
                        _learned_at.get(name, float("-inf")), now):
        _learn_buf.clear()
        return False
    _learn_buf.append(np.asarray(emb, dtype=np.float32))
    if len(_learn_buf) < FACE_LEARN_SAMPLES:
        return False
    look = np.mean(_learn_buf, axis=0)
    _learn_buf.clear()
    if not face_store.add_template(name, look):
        return False
    _learned_at[name] = now
    print(f"[vision] Learned a new look for {name} (match {conf:.2f}, "
          f"{face_store.template_count(name)} looks)")
    return True


def _track_absence(face_visible, now):
    """Forget the recognised person once NO face has been in view for
    PERSON_AWAY_SECS. Shorter gaps — looking down, turning to a friend,
    missed detections — keep the name and the visit (person_since), so they
    are not greeted or "recognised" all over again when the face returns."""
    global _faceless_since, _face_run_start
    if face_visible:
        if _face_run_start is None or (_faceless_since is not None
                                       and now - _faceless_since > FACE_GAP_SECS):
            _face_run_start = now
        _faceless_since = None
    elif _faceless_since is None:
        _faceless_since = now
    elif now - _faceless_since > PERSON_AWAY_SECS:
        _clear_recognition()


def _clear_recognition():
    global _recog_last, _recog_streak, _confirmed_at
    _confirmed_at = None
    _recog_last, _recog_streak = None, 0
    with state.lock:
        state.person_name = None
        state.person_conf = 0.0
        state.person_since = 0.0


# ── Enrollment ────────────────────────────────────────────────────────────────
# pose index -> embeddings. Written by the vision thread, counted by the voice
# thread (commands waits on enroll_sample_count), hence the lock.
_enroll_lock    = threading.Lock()
_enroll_by_pose = {}
_enroll_done    = False     # reported; ignore frames until the request clears
# Guided mode keeps only the most recent samples per pose: frames captured
# while the prompt was still being spoken (head not yet turned) roll out.
ENROLL_POSE_KEEP = max(1, int(ENROLL_FRAMES_PER_POSE)) * 2


def enroll_sample_count(pose=None):
    """Samples collected so far for `pose`, or in total when pose is None."""
    with _enroll_lock:
        if pose is None:
            return sum(len(v) for v in _enroll_by_pose.values())
        return len(_enroll_by_pose.get(int(pose), ()))


def _reset_enrollment():
    with _enroll_lock:
        _enroll_by_pose.clear()


# While Luna asks for the name, the person is already looking at her: their
# "look straight" samples are taken then, so registration can skip that pose.
_precapture = []            # newest last, at most ENROLL_POSE_KEEP


def _update_precapture(frame, det_row):
    with state.lock:
        until = state.enroll_precapture_until
        faces = state.face_count
    if time.time() > until:
        if _precapture:
            with _enroll_lock:
                _precapture.clear()
        return
    if det_row is None or faces != 1:       # only one person in view: theirs
        return
    emb = _embed(frame, det_row)
    if emb is not None:
        with _enroll_lock:
            _precapture.append(emb)
            del _precapture[:-ENROLL_POSE_KEEP]


def take_precapture():
    """Move the samples taken during the name dialog into the "look
    straight" pose of the registration now starting. Returns how many."""
    with _enroll_lock:
        got = list(_precapture)
        _precapture.clear()
        if got:
            _enroll_by_pose.setdefault(0, [])[:0] = got
    return len(got)


def _finish_enrollment(request):
    """Store the collected samples (if there are enough) and report."""
    global _recog_last, _recog_streak, _enroll_done
    with _enroll_lock:
        samples = [e for k in sorted(_enroll_by_pose)
                   for e in _enroll_by_pose[k]]
        per_pose = {k: len(v) for k, v in sorted(_enroll_by_pose.items())}
        _enroll_by_pose.clear()

    need = ENROLL_MIN_FRAMES if ENROLL_GUIDED_POSES \
        else min(ENROLL_MIN_FRAMES, ENROLL_FRAMES)
    ok = False
    if len(samples) >= max(1, need):
        try:
            ok = bool(face_store.enroll(request, samples))
        except Exception as e:
            print(f"[vision] Enrollment store failed: {e}")
    else:
        print(f"[vision] Enrollment of {request} failed: only "
              f"{len(samples)} good samples (need {need})")
    print(f"[vision] Enrollment {request}: {per_pose} -> "
          f"{'stored' if ok else 'failed'}")

    _recog_last, _recog_streak = request if ok else None, 2
    with state.lock:
        state.enroll_result = (request, ok)
        if ok:
            if state.person_name != request:
                state.person_since = time.time()
            state.person_name = request
            state.person_conf = 1.0
    _enroll_done = True


def _update_enrollment(frame, det_row):
    """Collect embedding samples while an enrollment is pending; store the
    result and report back through state.enroll_result. Called every tick,
    face or not, so a finish with nobody in view still reports a failure."""
    global _enroll_done

    with state.lock:
        request = state.enroll_request
        finish  = state.enroll_finish
        pose    = state.enroll_pose
    if request is None:
        _enroll_done = False
        if _enroll_by_pose:
            _reset_enrollment()
        _update_precapture(frame, det_row)
        return
    if _enroll_done:
        return

    if not finish:
        emb = _embed(frame, det_row) if det_row is not None else None
        if emb is not None:
            key = int(pose or 0) if ENROLL_GUIDED_POSES else 0
            with _enroll_lock:
                bucket = _enroll_by_pose.setdefault(key, [])
                bucket.append(emb)
                if ENROLL_GUIDED_POSES and len(bucket) > ENROLL_POSE_KEEP:
                    del bucket[0]

    if finish or (not ENROLL_GUIDED_POSES
                  and enroll_sample_count() >= ENROLL_FRAMES):
        _finish_enrollment(request)


# ── Scene memory feed ─────────────────────────────────────────────────────────
_scene_memory = None
_scene_memory_failed = False


def _feed_scene_memory(face_count, emotion):
    """One observe() per tick. Deferred + guarded: scene memory is a nicety
    and must never take the vision thread down; one notice, then silence."""
    global _scene_memory, _scene_memory_failed
    if _scene_memory_failed or not getattr(_config, "ENABLE_SCENE_MEMORY",
                                           True):
        return
    try:
        if _scene_memory is None:
            import scene_memory
            _scene_memory = scene_memory
        with state.lock:
            # The name outlives brief dropouts (see _track_absence); scene
            # memory runs its own departure timer on what is actually seen.
            name = state.person_name if face_count else None
        _scene_memory.observe(face_count=face_count, person_name=name,
                              emotion=emotion)
    except Exception as e:
        _scene_memory_failed = True
        print(f"[vision] Scene memory unavailable ({e}) — not feeding it")


# ── Main loop ─────────────────────────────────────────────────────────────────

def vision_loop():
    global _df_counter
    sleep_time  = 1.0 / VISION_FPS
    target_size = (48, 48)
    recog_counter = 0

    while True:
        t0 = time.monotonic()

        with state.lock:
            frame = state.frame

        if frame is None:
            time.sleep(0.1)
            continue

        detections = _detect_faces(frame)
        face_detected = len(detections) > 0
        emotion = "Neutral"
        det_row = None

        _track_absence(face_detected, time.time())
        if face_detected:
            x, y, w, h, det_row = detections[0]
            x, y = max(0, x), max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_crop  = gray[y:y+h, x:x+w]
            if gray_crop.size:
                gray_48 = cv2.resize(gray_crop, target_size,
                                     interpolation=cv2.INTER_NEAREST)
                pt_emotion = _run_pytorch(gray_48)
            else:
                pt_emotion = "Neutral"

            df_emotion  = None
            _df_counter += 1
            if DEEPFACE_OK and _df_counter >= DEEPFACE_EVERY:
                _df_counter = 0
                result = _run_deepface(frame[y:y+h, x:x+w])
                if result is not None:
                    df_label, df_conf = result
                    if df_label == "Sad" and df_conf >= SAD_CONFIDENCE:
                        df_emotion = "Sad"
                    elif pt_emotion == "Neutral" and df_label != "Neutral":
                        df_emotion = df_label

            raw = df_emotion if df_emotion else pt_emotion
            _history.append(raw)
            if len(_history) > EMOTION_HISTORY_LEN:
                _history.pop(0)
            emotion = _majority(_history)

            face_cx = (x + w / 2) / frame.shape[1]
            face_cy = (y + h / 2) / frame.shape[0]

            with state.lock:
                state.emotion       = emotion
                state.face_detected = True
                state.face_count    = len(detections)
                state.face_x        = face_cx
                state.face_y        = face_cy

            # Routine recognition only every Nth frame to keep the loop light
            # (enrollment below samples every frame — it's a short burst).
            recog_counter += 1
            if _sface is not None and recog_counter >= RECOGNIZE_EVERY:
                recog_counter = 0
                _update_recognition(frame, det_row)
        else:
            _history.clear()
            with state.lock:
                state.emotion       = "Neutral"
                state.face_detected = False
                state.face_count    = 0

        _update_enrollment(frame, det_row)
        _feed_scene_memory(len(detections), emotion)

        elapsed   = time.monotonic() - t0
        with state.lock:                     # the LLM is reading a prompt:
            thinking = state.thinking        # give it the CPU cores
        remaining = ((1.0 / VISION_FPS_THINKING) if thinking
                     else sleep_time) - elapsed
        if remaining > 0:
            time.sleep(remaining)


def start_vision():
    t = threading.Thread(target=vision_loop, daemon=True)
    t.name = "vision"
    t.start()
