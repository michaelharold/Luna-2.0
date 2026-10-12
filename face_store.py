# face_store.py
"""
face_store.py — who Luna knows.

A tiny on-device face database of 128-d SFace embeddings, matched by cosine
similarity. Each person has up to FACE_MAX_TEMPLATES "looks" (templates): one
per registration (the average of its capture frames) plus any the vision
thread learns by itself when it is very sure (vision_module._maybe_learn).
Matching takes the best look, so someone registered in daylight is still
recognised at night once a night look exists. Stored as plain files in
data/faces/ —

    faces.npz    the embedding matrix, one row per look
    names.json   the names, row-aligned with the matrix (a name repeats once
                 per look)

so it is trivially inspectable, backupable, and wipeable (delete the folder =
Luna forgets everyone). Nothing here touches the camera or OpenCV: vision
computes embeddings, this module only remembers and compares them, which keeps
it importable (and unit-testable) anywhere numpy exists.

Thread-safety: enrollments come from the vision thread while matches are read
in the same thread, but "forget" arrives from the voice thread — hence the
lock around every mutation and read of the arrays.
"""

import json
import os
import threading

import numpy as np

from config import FACE_DB_DIR, FACE_MATCH_THRESH
try:
    from config import FACE_MAX_TEMPLATES
except ImportError:                     # an older config.py
    FACE_MAX_TEMPLATES = 6

_EMB_PATH   = os.path.join(FACE_DB_DIR, "faces.npz")
_NAMES_PATH = os.path.join(FACE_DB_DIR, "names.json")

_lock = threading.Lock()
_names = []          # list[str]
_embeddings = None   # np.ndarray (N, D), L2-normalised rows, or None when empty


def _normalise(vec):
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _save_locked():
    os.makedirs(FACE_DB_DIR, exist_ok=True)
    if _embeddings is None:
        for p in (_EMB_PATH, _NAMES_PATH):
            if os.path.exists(p):
                os.remove(p)
        return
    np.savez(_EMB_PATH, embeddings=_embeddings)
    with open(_NAMES_PATH, "w") as f:
        json.dump(_names, f, indent=2)


def load():
    """Read the store from disk. Called once at import; safe to call again."""
    global _names, _embeddings
    with _lock:
        try:
            with open(_NAMES_PATH) as f:
                names = json.load(f)
            embs = np.load(_EMB_PATH)["embeddings"]
            if len(names) == embs.shape[0]:
                _names, _embeddings = names, embs.astype(np.float32)
                return
            print("[faces] Store is inconsistent (names != rows) — starting empty")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[faces] Could not load face store ({e}) — starting empty")
        _names, _embeddings = [], None


def _add_locked(name, vec):
    """One more look for `name` (caller holds _lock). At FACE_MAX_TEMPLATES
    the new look replaces the one MOST LIKE it, so the store keeps the widest
    spread of looks (day, night, glasses) rather than just the newest."""
    global _names, _embeddings
    rows = [i for i, n in enumerate(_names) if n == name]
    if rows and len(rows) >= FACE_MAX_TEMPLATES:
        sims = _embeddings[rows] @ vec
        _embeddings[rows[int(np.argmax(sims))]] = vec
    elif _embeddings is None:
        _names = [name]
        _embeddings = vec[None, :]
    else:
        _names.append(name)
        _embeddings = np.vstack([_embeddings, vec[None, :]])


def enroll(name, embeddings):
    """Remember `name`: the average of the given samples becomes one more
    look for them. Registering again (at night, with glasses) ADDS a look
    instead of replacing the first, so recognition holds across lighting;
    "forget me" removes every look."""
    name = name.strip().title()
    if not name or not len(embeddings):
        return False

    mean = _normalise(np.mean([_normalise(e) for e in embeddings], axis=0))
    with _lock:
        _add_locked(name, mean)
        looks = _names.count(name)
        _save_locked()
    print(f"[faces] Enrolled {name} ({len(embeddings)} samples, "
          f"{looks} look{'s' * (looks > 1)})")
    return True


def add_template(name, embedding):
    """A look the vision thread learned by itself, for someone ALREADY
    enrolled (it never creates a person). False when `name` is unknown."""
    name = name.strip().title()
    vec = _normalise(embedding)
    with _lock:
        if name not in _names:
            return False
        _add_locked(name, vec)
        _save_locked()
    return True


def template_count(name):
    """How many looks `name` has (0 when unknown)."""
    name = name.strip().title()
    with _lock:
        return _names.count(name)


def runner_up(embedding, name):
    """Best similarity of anyone OTHER than `name` — learning a new look
    needs a clear margin over every other enrolled person."""
    name = (name or "").strip().title()
    with _lock:
        if _embeddings is None:
            return 0.0
        scores = _embeddings @ _normalise(embedding)
        others = [float(s) for s, n in zip(scores, _names) if n != name]
    return max(others) if others else 0.0


def match(embedding):
    """(name, similarity) of the closest enrolled person, or (None, 0.0) when
    nobody clears FACE_MATCH_THRESH — an un-enrolled visitor stays a stranger
    rather than being force-fitted to the nearest known face."""
    with _lock:
        if _embeddings is None:
            return None, 0.0
        scores = _embeddings @ _normalise(embedding)
        idx = int(np.argmax(scores))
        best = float(scores[idx])
        name = _names[idx]
    if best < FACE_MATCH_THRESH:
        return None, best
    return name, best


def forget(name):
    """Remove one person — every look of theirs. True if they existed."""
    global _names, _embeddings
    name = name.strip().title()
    with _lock:
        if name not in _names:
            return False
        keep = [i for i, n in enumerate(_names) if n != name]
        _names = [_names[i] for i in keep]
        _embeddings = _embeddings[keep] if keep else None
        _save_locked()
    print(f"[faces] Forgot {name}")
    return True


def known_names():
    """Everyone enrolled, once each, in enrolment order."""
    with _lock:
        return list(dict.fromkeys(_names))


load()
