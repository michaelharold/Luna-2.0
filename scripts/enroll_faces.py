#!/usr/bin/env python3
"""
scripts/enroll_faces.py — register people from photos, no camera needed.

Why: voice registration needs the person standing in front of Luna. For a
department booth it is far easier to pre-load the staff from a few photos
each (ID-card shots, a phone snap from two angles) — they are greeted by
name from the first day. The result is the SAME face_store entry the live
enrollment writes: the average SFace embedding of all usable photos.

    # one folder per person — the folder name is the person's name
    #   data/enroll/Alice/1.jpg  data/enroll/Alice/2.jpg  data/enroll/Bob/a.png
    python scripts/enroll_faces.py --dir data/enroll

    # or one person from explicit files
    python scripts/enroll_faces.py --name Alice a.jpg b.jpg

For each photo the LARGEST face YuNet finds is used (group photos: crop them
first). Photos with no detectable face are reported and skipped. Re-enrolling
a name replaces the old embedding (face_store semantics).

Works identically on the Mac and the Pi: paths from config are resolved
against the project root, so it can be run from any directory. The detector
and embedder are plain callables so the tests can inject fakes.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# Detect on at most this many pixels on the long side — phone photos are
# 4000 px wide and YuNet's cost grows with area; SFace aligns to 112x112
# anyway, so a 960 px image loses nothing for recognition.
MAX_SIDE = 960


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


# ── Models (real) ─────────────────────────────────────────────────────────────

def load_models():
    """(detect, embed) callables backed by YuNet + SFace from config paths.
    detect(img) -> YuNet row of the largest face, or None.
    embed(img, row) -> 128-d embedding (np.ndarray), or None."""
    import cv2

    yunet_path = _abs(config.YUNET_MODEL_PATH)
    sface_path = _abs(config.SFACE_MODEL_PATH)
    for p in (yunet_path, sface_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"model missing: {p} (run scripts/download_models.sh)")

    detector = cv2.FaceDetectorYN.create(
        yunet_path, "", (320, 320),
        score_threshold=config.FACE_DETECT_CONF, nms_threshold=0.3, top_k=50)
    recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

    def detect(img):
        h, w = img.shape[:2]
        detector.setInputSize((w, h))
        _, rows = detector.detect(img)
        if rows is None or len(rows) == 0:
            return None
        return max(rows, key=lambda r: float(r[2]) * float(r[3]))

    def embed(img, row):
        aligned = recognizer.alignCrop(img, row)
        return recognizer.feature(aligned).flatten().copy()

    return detect, embed


def load_image(path):
    """BGR image, downscaled so the long side is at most MAX_SIDE; None if
    unreadable. cv2.imread applies EXIF rotation, so phone photos are upright."""
    import cv2
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = MAX_SIDE / float(max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    return img


# ── Job discovery ─────────────────────────────────────────────────────────────

def _images_in(folder):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(IMAGE_EXTS)
                  and not f.startswith("."))


def name_from_folder(folder):
    """data/enroll/alice_smith -> "Alice Smith"."""
    base = os.path.basename(os.path.normpath(folder))
    return base.replace("_", " ").replace("-", " ").strip().title()


def jobs_from_dir(root_dir):
    """[(name, [image paths])] — one job per sub-folder holding images."""
    jobs = []
    for entry in sorted(os.listdir(root_dir)):
        folder = os.path.join(root_dir, entry)
        if entry.startswith(".") or not os.path.isdir(folder):
            continue
        images = _images_in(folder)
        if images:
            jobs.append((name_from_folder(folder), images))
        else:
            print(f"[enroll] {folder}: no images — skipped")
    return jobs


# ── Enrollment ────────────────────────────────────────────────────────────────

def enroll_person(name, paths, detect, embed, store, loader=load_image):
    """Embed the largest face of every photo and store their average.
    Returns (ok, used, skipped_paths)."""
    samples, skipped = [], []
    for path in paths:
        img = loader(path)
        if img is None:
            print(f"[enroll] WARNING {path}: could not read image")
            skipped.append(path)
            continue
        row = detect(img)
        if row is None:
            print(f"[enroll] WARNING {path}: no face found")
            skipped.append(path)
            continue
        emb = embed(img, row)
        if emb is None:
            print(f"[enroll] WARNING {path}: could not embed the face")
            skipped.append(path)
            continue
        samples.append(emb)

    if not samples:
        print(f"[enroll] {name}: no usable photos — NOT enrolled")
        return False, 0, skipped
    ok = bool(store.enroll(name, samples))
    print(f"[enroll] {name}: {'enrolled' if ok else 'FAILED'} from "
          f"{len(samples)} photo(s)"
          + (f", {len(skipped)} skipped" if skipped else ""))
    return ok, len(samples), skipped


def run(jobs, detect, embed, store, loader=load_image):
    """Enroll every job; returns {name: ok}."""
    return {name: enroll_person(name, paths, detect, embed, store, loader)[0]
            for name, paths in jobs}


def _parse(argv):
    ap = argparse.ArgumentParser(
        description="Register faces from photos (no camera needed).")
    ap.add_argument("--dir", help="folder with one sub-folder per person "
                    f"(default {config.ENROLL_PHOTO_DIR})")
    ap.add_argument("--name", help="enroll one person from the given files")
    ap.add_argument("files", nargs="*", help="photos for --name")
    ap.add_argument("--db", help="face store folder "
                    f"(default {config.FACE_DB_DIR})")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse(argv)

    # face_store resolves FACE_DB_DIR at import, relative to the cwd — pin it
    # to the project (or --db) BEFORE the first import so Mac and Pi agree.
    config.FACE_DB_DIR = os.path.abspath(args.db) if args.db \
        else _abs(config.FACE_DB_DIR)
    import face_store

    if args.name:
        if not args.files:
            print("[enroll] --name needs at least one photo")
            return 2
        missing = [f for f in args.files if not os.path.isfile(f)]
        for f in missing:
            print(f"[enroll] WARNING {f}: no such file")
        jobs = [(args.name.strip().title(),
                 [f for f in args.files if os.path.isfile(f)])]
    else:
        root_dir = args.dir or _abs(config.ENROLL_PHOTO_DIR)
        if not os.path.isdir(root_dir):
            print(f"[enroll] {root_dir} does not exist — make one folder per "
                  f"person inside it (e.g. {root_dir}/Alice/1.jpg)")
            return 2
        jobs = jobs_from_dir(root_dir)
    if not jobs or not any(paths for _, paths in jobs):
        print("[enroll] nothing to enroll")
        return 2

    detect, embed = load_models()
    results = run(jobs, detect, embed, face_store)
    print(f"[enroll] done: {sum(results.values())}/{len(results)} enrolled "
          f"into {config.FACE_DB_DIR}. Known now: "
          f"{', '.join(face_store.known_names()) or 'nobody'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
