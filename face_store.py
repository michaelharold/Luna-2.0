# face_store.py
"""
face_store.py — who Luna knows.

A tiny on-device face database: one 128-d SFace embedding per enrolled person
(the average of several capture frames), matched by cosine similarity. Stored
as plain files in data/faces/ —

    faces.npz    the embedding matrix
    names.json   the names, row-aligned with the matrix

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


def enroll(name, embeddings):
    """Remember `name` as the average of the given embedding samples.
    Re-enrolling an existing name REPLACES their embedding (people change:
    glasses, beards), rather than appending a duplicate row."""
    global _names, _embeddings
    name = name.strip().title()
    if not name or not len(embeddings):
        return False

    mean = _normalise(np.mean([_normalise(e) for e in embeddings], axis=0))
    with _lock:
        if name in _names:
            _embeddings[_names.index(name)] = mean
        elif _embeddings is None:
            _names = [name]
            _embeddings = mean[None, :]
        else:
            _names.append(name)
            _embeddings = np.vstack([_embeddings, mean[None, :]])
        _save_locked()
    print(f"[faces] Enrolled {name} ({len(embeddings)} samples)")
    return True


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
    """Remove one person. Returns True if they existed."""
    global _names, _embeddings
    name = name.strip().title()
    with _lock:
        if name not in _names:
            return False
        idx = _names.index(name)
        _names.pop(idx)
        _embeddings = np.delete(_embeddings, idx, axis=0)
        if _embeddings.shape[0] == 0:
            _embeddings = None
        _save_locked()
    print(f"[faces] Forgot {name}")
    return True


def known_names():
    with _lock:
        return list(_names)


load()
