# scene_context.py
"""
scene_context.py — what Luna can say about what she sees.

When a question sounds like it's about the camera view ("what do you see?",
"who am I?", "how do I look?"), Luna assembles an English description of the
live scene and answers through the local LLM with that description as
context. Two layers feed it:

  1. What vision already knows, for free — faces, the recognised person's
     name, their emotion, where they're standing, and the frame brightness.
  2. Objects, on demand — YOLOv8n through OpenCV's dnn module. The network
     is loaded LAZILY on the first vision question and only ever run when
     one is asked, so it costs nothing while Luna idles.

Everything is offline: the ONNX ships in models/vision/. If it's missing,
scene descriptions still work from layer 1 alone.
"""

import os
import threading

import numpy as np

from config import (ENABLE_OBJECT_DETECTION, YOLO_MODEL_PATH, YOLO_INPUT_SIZE,
                    YOLO_CONF, YOLO_NMS, VISION_QUERY_PHRASES)

# The 80 COCO class names, index-aligned with YOLOv8's output rows.
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

_net = None
_net_failed = False
_net_lock = threading.Lock()


def is_vision_query(text):
    """Does this utterance ask about what the camera sees?"""
    lower = text.lower()
    return any(p in lower for p in VISION_QUERY_PHRASES)


# ── Object detection (YOLOv8n via cv2.dnn, on demand) ────────────────────────

def _load_net():
    global _net, _net_failed
    if _net is not None or _net_failed or not ENABLE_OBJECT_DETECTION:
        return
    if not os.path.exists(YOLO_MODEL_PATH):
        _net_failed = True
        print(f"[scene] {YOLO_MODEL_PATH} missing — object detection off "
              f"(scene answers still use faces/emotions).")
        return
    try:
        import cv2
        _net = cv2.dnn.readNetFromONNX(YOLO_MODEL_PATH)
        print("[scene] YOLOv8n loaded")
    except Exception as e:
        _net_failed = True
        print(f"[scene] Could not load YOLO ({e}) — object detection off.")


def detect_objects(frame_bgr):
    """Run YOLOv8n on one frame. Returns a sorted list of (label, count),
    biggest count first; [] when detection is unavailable.

    The frame is letterboxed to the square input the export expects — a plain
    resize would squash people into false aspect ratios and cost accuracy.
    """
    with _net_lock:
        _load_net()
        if _net is None or frame_bgr is None:
            return []
        import cv2

        size = YOLO_INPUT_SIZE
        h, w = frame_bgr.shape[:2]
        scale = size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(frame_bgr, (nw, nh))

        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (size, size),
                                     swapRB=True, crop=False)
        _net.setInput(blob)
        try:
            out = _net.forward()
        except Exception as e:
            print(f"[scene] YOLO inference failed: {e}")
            return []

    # Output is (1, 84, N): 4 box coords + 80 class scores per candidate.
    preds = out[0].T                       # (N, 84)
    class_scores = preds[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_ids)), class_ids]

    keep = confidences >= YOLO_CONF
    if not np.any(keep):
        return []
    boxes_cxcywh = preds[keep, :4]
    class_ids = class_ids[keep]
    confidences = confidences[keep]

    # cx,cy,w,h -> x,y,w,h for NMS
    boxes = np.empty_like(boxes_cxcywh)
    boxes[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    boxes[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    boxes[:, 2:] = boxes_cxcywh[:, 2:]

    import cv2
    idxs = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(),
                            YOLO_CONF, YOLO_NMS)
    if len(idxs) == 0:
        return []

    counts = {}
    for i in np.asarray(idxs).flatten():
        label = COCO_CLASSES[class_ids[i]]
        counts[label] = counts.get(label, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])


# ── Description ──────────────────────────────────────────────────────────────

def _brightness_word(frame_bgr):
    if frame_bgr is None:
        return None
    mean = float(np.mean(frame_bgr))
    if mean < 45:
        return "quite dark"
    if mean < 90:
        return "dimly lit"
    if mean > 190:
        return "very bright"
    return "well lit"


def _position_word(x):
    if x < 0.36:
        return "to my left"
    if x > 0.64:
        return "to my right"
    return "right in front of me"


def describe(face_count, person_name, emotion, face_x,
             objects=(), frame_bgr=None):
    """One compact English paragraph describing the scene — written as facts
    ('Camera: ...') for the LLM to phrase naturally, and readable enough to
    speak verbatim when the LLM is unavailable. Pure function of its inputs,
    so it's testable without a camera."""
    parts = []

    if face_count <= 0:
        parts.append("I don't see anyone in front of me right now.")
    else:
        if person_name:
            who = f"{person_name}, someone I know,"
        elif face_count == 1:
            who = "one person I haven't met yet"
        else:
            who = f"{face_count} people"
        sentence = f"I can see {who} {_position_word(face_x)}"
        if emotion and emotion.lower() not in ("neutral", ""):
            sentence += f", looking {emotion.lower()}"
        parts.append(sentence + ".")

    # People are already covered above (and the face pipeline is better at
    # them), so drop YOLO's own "person" entries from the object list.
    things = [(label, n) for label, n in objects if label != "person"]
    if things:
        listed = ", ".join(f"{n} {label}s" if n > 1 else f"a {label}"
                           for label, n in things[:6])
        parts.append(f"I can also spot: {listed}.")

    light = _brightness_word(frame_bgr)
    if light:
        parts.append(f"The room is {light}.")

    return " ".join(parts)
