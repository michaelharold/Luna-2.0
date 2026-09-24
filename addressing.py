# addressing.py
"""
addressing.py — "was that meant for Luna?"

Inside an open conversation window Luna hears EVERYTHING in the room, and the
hard problem is not transcription, it's attention: separating speech aimed at
her from chatter between people who happen to be standing near her. The old
gate used Vosk confidence alone, which conflates "clearly spoken" with
"spoken to Luna" — a crisp remark to a friend sailed through while a mumbled
real question was dropped.

This module scores the signals a *person* uses to know they're being spoken
to, each weighted in config:

    confidence   — Vosk's word confidence (clear speech, the base signal)
    loudness     — how close the talker sounds (the utterance's RMS level):
                   people talking TO Luna stand at her mic, chatter across
                   the room is quieter (speech_to_text measures it)
    name         — Luna's name or a wake word anywhere in the utterance
    question     — the utterance is question-shaped
    second person— "you / your": they are talking TO someone, probably her
    knowledge    — the topic matches Luna's knowledge base
    face         — someone is looking at her while speaking
    continuity   — Luna just asked a question; this is probably the answer

Two hard rules sit on top of the weighted sum, because a sum can be carried
by one strong base signal: with ADDR_REQUIRE_SIGNAL, confidence (plus a merely
average loudness) can never pass on its own — at least one real addressing
signal must be present. With ADDR_FACE_OR_NAME (strictest) nothing said
off-camera is answered unless Luna was named or is waiting for an answer.

Pure logic, no heavy imports — unit-testable on any machine. speech_to_text
calls `decide()` (or `is_addressed()`) and logs `explain()`; everything else
is implementation detail.
"""

import re

from config import (
    WAKE_WORDS, ROBOT_NAME,
    ADDR_THRESHOLD, ADDR_REQUIRE_SIGNAL, ADDR_FACE_OR_NAME,
    ADDR_W_CONF, ADDR_W_LOUDNESS, ADDR_W_NAME, ADDR_W_QUESTION,
    ADDR_W_SECOND_PERSON, ADDR_W_KB, ADDR_W_FACE, ADDR_W_CONTINUITY,
    LOUDNESS_NEAR_DB, LOUDNESS_FAR_DB,
    STT_MIN_WORDS,
)

# What counts as "an addressing signal" for the ADDR_REQUIRE_SIGNAL rule. The
# yes/no signals (name, question, "you", face, continuity) count whenever they
# are present; the two graded ones must be convincing on their own, not merely
# non-zero — a faint topic overlap or an ordinary room level proves nothing.
SIGNAL_KB_MIN       = 0.4   # KB similarity at/above this = "about Luna's topics"
SIGNAL_LOUDNESS_MIN = 0.6   # loudness (0..1) at/above this = "standing at her mic"

# Order of the contributions in explain() — and of the tokens on the log line.
SIGNALS = ("conf", "loudness", "name", "question", "second_person",
           "kb", "face", "continuity")

_QUESTION_STARTERS = (
    "what", "where", "when", "who", "whose", "why", "how", "which",
    "is", "are", "am", "was", "were", "do", "does", "did", "can", "could",
    "will", "would", "should", "shall", "may", "tell", "explain", "describe",
)

_SECOND_PERSON = frozenset(("you", "your", "yours", "yourself"))


def _words(text):
    return re.findall(r"[a-z']+", text.lower())


def _unit(x):
    """Clamp a graded signal to 0..1. None / NaN / junk count as absent (0)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    if x != x:                       # NaN
        return 0.0
    return max(0.0, min(1.0, x))


def mentions_name(text):
    """True if Luna's name or any wake word occurs anywhere in the text,
    matched on word boundaries so 'lunatic' doesn't count."""
    words = _words(text)
    joined = " ".join(words)
    if ROBOT_NAME.lower() in words:
        return True
    for wake in WAKE_WORDS:
        w = wake.lower().strip()
        if " " in w:
            if w in joined:
                return True
        elif w in words:
            return True
    return False


def _strip_leading_name(words):
    """Drop Luna's name / wake words from the FRONT of an utterance ("luna
    what courses are offered" is a question about courses), and a following
    filler ("luna hey what…"), so the question test sees the real sentence."""
    wake = {w for phrase in WAKE_WORDS for w in phrase.lower().split()}
    wake.add(ROBOT_NAME.lower())
    fillers = {"hey", "hi", "hello", "ok", "okay", "so", "um", "please"}
    i = 0
    while i < len(words) and (words[i] in wake or words[i] in fillers):
        i += 1
    return words[i:]


def is_question(text):
    """Question-shaped: starts with an interrogative/auxiliary (after any
    leading name or greeting), or Vosk's transcription kept a question mark
    (it usually doesn't, so the starter check does the real work)."""
    if text.rstrip().endswith("?"):
        return True
    words = _strip_leading_name(_words(text))
    return bool(words) and words[0] in _QUESTION_STARTERS


def has_second_person(text):
    return bool(_SECOND_PERSON & set(_words(text)))


def loudness_score(rms_db):
    """Map an utterance's RMS level (dBFS, as audio_preprocess measures it)
    onto 0..1: LOUDNESS_FAR_DB and quieter → 0, LOUDNESS_NEAR_DB and louder
    → 1, linear in between. None / NaN / -inf (digital silence) → 0."""
    try:
        db = float(rms_db)
    except (TypeError, ValueError):
        return 0.0
    if db != db:                     # NaN
        return 0.0
    span = LOUDNESS_NEAR_DB - LOUDNESS_FAR_DB
    if span <= 0:                    # degenerate config: a step at NEAR
        return 1.0 if db >= LOUDNESS_NEAR_DB else 0.0
    return max(0.0, min(1.0, (db - LOUDNESS_FAR_DB) / span))


def explain(text, conf, kb_score=0.0, face_visible=False,
            awaiting_answer=False, loudness=0.0):
    """Each signal's contribution to the score, keyed by SIGNALS, 0.0 when the
    signal is absent — so the log line can show WHY an utterance passed or
    failed. sum(explain(...).values()) is exactly address_score(...)."""
    return {
        "conf":          ADDR_W_CONF * _unit(conf),
        "loudness":      ADDR_W_LOUDNESS * _unit(loudness),
        "name":          ADDR_W_NAME if mentions_name(text) else 0.0,
        "question":      ADDR_W_QUESTION if is_question(text) else 0.0,
        "second_person": ADDR_W_SECOND_PERSON if has_second_person(text) else 0.0,
        "kb":            ADDR_W_KB * _unit(kb_score),
        "face":          ADDR_W_FACE if face_visible else 0.0,
        "continuity":    ADDR_W_CONTINUITY if awaiting_answer else 0.0,
    }


def format_explain(parts):
    """The explain() dict as one compact log token per NON-zero contribution,
    e.g. 'conf=0.96 +question=0.35 +loudness=0.05 = 1.36'."""
    items = [f"{k}={v:.2f}" for k, v in parts.items() if v]
    total = sum(parts.values())
    return (" +".join(items) if items else "nothing") + f" = {total:.2f}"


def address_score(text, conf, kb_score=0.0, face_visible=False,
                  awaiting_answer=False, loudness=0.0):
    """The weighted sum of every addressing signal. Pure — all context comes
    in as arguments, so tests don't need shared state or a camera."""
    return sum(explain(text, conf, kb_score, face_visible,
                       awaiting_answer, loudness).values())


def has_signal(text, kb_score=0.0, face_visible=False, awaiting_answer=False,
               loudness=0.0):
    """The ADDR_REQUIRE_SIGNAL rule: is there at least one sign the utterance
    was aimed at Luna, beyond it merely being clear speech? Confidence is not
    a signal — a crisp remark to a friend is still a remark to a friend."""
    return (mentions_name(text)
            or is_question(text)
            or has_second_person(text)
            or _unit(kb_score) >= SIGNAL_KB_MIN
            or bool(face_visible)
            or _unit(loudness) >= SIGNAL_LOUDNESS_MIN
            or bool(awaiting_answer))


def decide(text, conf, kb_score=0.0, face_visible=False,
           awaiting_answer=False, loudness=0.0):
    """The full verdict: (accepted, reason). `reason` is a short phrase for
    the log — "no addressing signal", "score 1.61 >= 1.35", … Keeps the
    minimum-words floor from the old gate: one stray word is never a
    conversation."""
    if len(text.split()) < STT_MIN_WORDS:
        return False, "too few words"
    if ADDR_REQUIRE_SIGNAL and not has_signal(text, kb_score, face_visible,
                                              awaiting_answer, loudness):
        return False, "no addressing signal"
    if ADDR_FACE_OR_NAME and not (face_visible or awaiting_answer
                                  or mentions_name(text)):
        return False, "no face, name or pending answer"
    score = address_score(text, conf, kb_score, face_visible,
                          awaiting_answer, loudness)
    if score >= ADDR_THRESHOLD:
        return True, f"score {score:.2f} >= {ADDR_THRESHOLD:.2f}"
    return False, f"score {score:.2f} < {ADDR_THRESHOLD:.2f}"


def is_addressed(text, conf, kb_score=0.0, face_visible=False,
                 awaiting_answer=False, loudness=0.0):
    """True when the utterance is worth answering (see decide())."""
    return decide(text, conf, kb_score, face_visible,
                  awaiting_answer, loudness)[0]
