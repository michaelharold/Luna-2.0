#!/usr/bin/env python3
"""
scripts/bench_vision.py — how much does VISION_DETECT_SCALE buy?

YuNet face detection is the single biggest cost in Luna's vision loop on the
Pi. This times it on a 640x480 frame at full scale and at
config.VISION_DETECT_SCALE (resize included, exactly as vision_module does),
plus one SFace alignCrop + embedding, so the numbers on the Pi can be compared
with the Mac and with the VISION_FPS budget (1000 / VISION_FPS ms per tick).

    python scripts/bench_vision.py              # synthetic frame
    python scripts/bench_vision.py --image me.jpg --iters 100

It deliberately does NOT import vision_module (that loads the PyTorch emotion
model); it only needs OpenCV and the two ONNX models.
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def synthetic_frame(w=640, h=480, seed=0):
    """Noise plus a crude bright 'face' blob — YuNet's cost is dominated by
    the network, not the content, so this is representative enough."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.ellipse(frame, (w // 2, h // 2), (70, 95), 0, 0, 360,
                (150, 170, 200), -1)
    return frame


def timeit(fn, iters):
    fn()                                    # warm-up (lazy allocations)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / iters


def bench_detect(detector, frame, scale, iters):
    def run():
        img = frame if scale >= 1.0 else cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        detector.setInputSize((w, h))
        detector.detect(img)
    return timeit(run, iters)


def fake_row(w, h):
    """A plausible YuNet row for a centred face: box, 5 landmarks, score."""
    cx, cy, fw, fh = w / 2, h / 2, 140.0, 190.0
    return np.array([cx - fw / 2, cy - fh / 2, fw, fh,
                     cx - 30, cy - 20, cx + 30, cy - 20,   # eyes
                     cx, cy + 10,                          # nose
                     cx - 25, cy + 45, cx + 25, cy + 45,   # mouth corners
                     0.99], dtype=np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--image", help="use this photo (resized to 640x480)")
    args = ap.parse_args(argv)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"[bench] cannot read {args.image}")
            return 2
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    else:
        frame = synthetic_frame()

    yunet_path = _abs(config.YUNET_MODEL_PATH)
    sface_path = _abs(config.SFACE_MODEL_PATH)
    if not os.path.exists(yunet_path):
        print(f"[bench] YuNet model missing: {yunet_path}")
        return 2
    detector = cv2.FaceDetectorYN.create(
        yunet_path, "", (640, 480), score_threshold=config.FACE_DETECT_CONF,
        nms_threshold=0.3, top_k=500)

    print(f"[bench] OpenCV {cv2.__version__}, {cv2.getNumThreads()} threads, "
          f"{args.iters} iterations, frame 640x480"
          f"{' (' + args.image + ')' if args.image else ' (synthetic)'}")

    full = bench_detect(detector, frame, 1.0, args.iters)
    print(f"  YuNet @ scale 1.00          : {full:7.2f} ms/call")
    scale = float(getattr(config, "VISION_DETECT_SCALE", 1.0))
    if 0 < scale < 1.0:
        small = bench_detect(detector, frame, scale, args.iters)
        print(f"  YuNet @ scale {scale:.2f} (+resize): {small:7.2f} ms/call "
              f"({full / max(small, 1e-6):.1f}x faster)")
    else:
        print(f"  VISION_DETECT_SCALE={scale} — downscaling is off")

    if os.path.exists(sface_path):
        recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
        row = fake_row(640, 480)
        # A real detection when there is one (photo), else the synthetic row.
        detector.setInputSize((640, 480))
        _, rows = detector.detect(frame)
        if rows is not None and len(rows):
            row = rows[0]

        def embed():
            recognizer.feature(recognizer.alignCrop(frame, row))
        print(f"  SFace alignCrop + embed     : "
              f"{timeit(embed, args.iters):7.2f} ms/call")
    else:
        print(f"  SFace model missing ({sface_path}) — embed not timed")

    budget = 1000.0 / max(1, int(getattr(config, "VISION_FPS", 8)))
    print(f"  tick budget at VISION_FPS   : {budget:7.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
