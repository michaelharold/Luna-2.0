# learning.py
"""
learning.py — how Luna gets better from daily life, on the device itself.

WHY not fine-tuning: a Pi 5 can't train a language model, and a desk robot
that rewrites its own weights from booth chatter would drift in ways nobody
could audit. What CAN learn cheaply and safely is everything AROUND the
model — and that turns out to be most of what people notice:

  manner     A contextual bandit (Thompson sampling over Beta posteriors)
             picks a reply STYLE — length × tone × follow-up question, 8 arms
             — per context (known person or stranger, KB / general / out-of-
             scope question, time of day). The style is one line at the end
             of the prompt; the rewards below teach it which style people
             actually respond to. Until a context has LEARNING_MIN_TURNS_TO_
             STEER rewarded turns it explores uniformly.
  rewards    Scored against the PREVIOUS answered turn when the next
             utterance arrives: explicit praise / "that's wrong", thanks or
             laughter, simply carrying on (engagement), asking the same thing
             again (a sign the answer missed), a thumbs-up gesture, or walking
             off right after the answer. Summed, clipped to [-1, 1], appended
             to the conversation log, and fed to the bandit, exemplars and
             the person's profile.
  exemplars  Replies that earned a high reward are kept (cap 200) and the
             most similar ones are shown to the model as examples for new
             questions — few-shot style transfer without touching weights.
  people     Per-person profiles (visits, topics, which style they like),
             summarised in one sentence when that person is recognised.
  facts      "Remember that…" and corrections go to a PENDING file; a human
             approves them into knowledge.txt (scripts/approve_facts.py).
             Luna never teaches herself facts unsupervised.
  diary      After DIARY_WRITE_HOUR, the first idle moment turns today's log
             into a 3-sentence diary entry; later questions can retrieve the
             most relevant diary sentence.

Rules: numpy + stdlib only; all state lives in LEARNING_DIR (JSON / npz,
written atomically) so deleting the folder = Luna forgets what she learned;
EVERY public function never raises (first failure of a kind is printed once)
because learning must never cost a voice turn. Randomness is injectable
(configure(rng=...)) so tests are deterministic.
"""

import datetime as _dt
import functools
import json
import os
import re
import threading
import time

import numpy as np

import conversation_log
import console

try:
    import config as _config
except Exception:                                      # noqa: BLE001
    _config = None


def _c(name, default):
    return getattr(_config, name, default) if _config is not None else default


_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Module tunables (not in config.py — listed in the report) ────────────────
REWARD_MAX_GAP      = 60.0    # s: an utterance later than this after a turn
                              # says nothing about that turn
EXPLICIT_MAX_AGE    = 180.0   # s: "good answer" still refers to the last turn
ABANDON_SECS        = 10.0    # s: leaving within this of a reply = abandon
DEPART_SECS         = 2.0     # s: no face this long = the person left
PRESENT_SLACK_SECS  = 5.0     # a face seen this close to a turn = they were there
VISIT_GAP_SECS      = 1800.0  # s away before a return counts as a new visit
PROFILE_SAVE_SECS   = 60.0    # throttle for last_seen-only profile saves
EXEMPLAR_CAP        = 200
EXEMPLAR_MIN_SIM    = 0.50    # cosine for an exemplar to be relevant
EXEMPLAR_DEDUP_SIM  = 0.92    # a question this close replaces the older one
KB_KIND_MIN_SIM     = 0.50    # KB similarity for a question to count as "kb"
DIARY_MIN_SIM       = 0.45    # diary sentence relevance threshold
DIARY_MAX_FILES     = 60      # only the most recent diary days are searched
DIARY_MIN_TURNS     = 3
DIARY_MAX_ATTEMPTS  = 3       # LLM failures per day before giving up
FEEDBACK_MAX_WORDS  = 7       # longer utterances are questions, not feedback
ANSWER_PATHS        = ("kb", "llm", "vision", "fallback")
EXEMPLAR_PATHS      = ("llm", "vision")

# ── Style arms ────────────────────────────────────────────────────────────────
LENGTHS = {
    "one_sentence": "Answer in one short sentence",
    "two_or_three": "Answer in two or three short sentences",
}
TONES = {
    "playful":    "playful and upbeat",
    "plain_warm": "plain, calm and warm",
}
FOLLOWUPS = {
    "ask":    "end with one short, specific follow-up question",
    "no_ask": "do not ask a follow-up question",
}
ARMS = tuple(f"{l}|{t}|{f}" for l in LENGTHS for t in TONES for f in FOLLOWUPS)

_SHORT_DESC = {"one_sentence": "one-sentence", "two_or_three": "fuller",
               "playful": "playful", "plain_warm": "calm",
               "ask": "with a question", "no_ask": "without a question"}


def style_instruction(arm):
    """The one-line prompt instruction for an arm ("" for unknown arms)."""
    try:
        l, t, f = arm.split("|")
        return f"{LENGTHS[l]}, keep the tone {TONES[t]}, and {FOLLOWUPS[f]}."
    except Exception:                                  # noqa: BLE001
        return ""


# ── Feedback phrases ──────────────────────────────────────────────────────────
POSITIVE_PHRASES = (
    "good answer", "great answer", "nice answer", "perfect answer",
    "good reply", "great reply", "well answered", "good job", "well done",
    "that helped", "that was helpful", "very helpful", "helpful answer",
    "correct answer", "right answer", "good one luna", "nice one",
)
# Also what you'd say answering a yes/no follow-up — only feedback when
# Luna's last reply did NOT end in a question.
WEAK_POSITIVE_PHRASES = (
    "that's right", "that is right", "that's correct", "that is correct",
    "exactly right", "you're right", "you are right", "that's true",
)
NEGATIVE_PHRASES = (
    "that's wrong", "that is wrong", "wrong answer", "that's not right",
    "that is not right", "that's not correct", "that is not correct",
    "that's incorrect", "that is incorrect", "not correct", "you're wrong",
    "you are wrong", "bad answer", "that's not true", "that is not true",
    "that was wrong", "incorrect answer", "not the right answer",
)
THANKS_WORDS = {"thanks", "thank", "thankyou", "haha", "hahaha", "lol",
                "cool", "awesome", "nice", "great", "wonderful", "amazing",
                "lovely", "brilliant", "sweet", "interesting", "perfect"}
_SCOPE_RE = re.compile(
    r"\b(weather|temperature outside|rain|news|headlines|stock|share price|"
    r"bitcoin|crypto|price of|cost of|score|match today|what time is it|"
    r"the time now|today's date|what day is it|what is the date|"
    r"book (a|me)|order (a|me)|call (my|someone)|send (an? )?(email|message)|"
    r"turn (on|off) the|set an? (alarm|reminder|timer)|play (a |some )?"
    r"(song|music)|phone number of|address of (my|his|her))\b")

# ── State ─────────────────────────────────────────────────────────────────────
_lock = threading.RLock()
_cfg = {}
_rng = np.random.default_rng()
_embed = None                  # fn(list[str]) -> normalised rows, or None

_loaded = False
_bandit = {}                   # context -> {"_n": int, arm: [alpha, beta]}
_exemplars = []                # [{q, reply, reward, arm, context, ts}]
_ex_emb = None                 # np.ndarray rows aligned with _exemplars
_profiles = {}                 # name -> dict
_profile_saved_at = 0.0
_pending_thumbs = None         # ts of an unconsumed thumbs-up
_presence = {"last_face": None, "absent_since": None, "name": None}
_diary_cache = {"key": None, "sents": [], "emb": None}
_diary_attempts = {}           # date -> failed attempts
_errors_printed = set()


def _defaults():
    return {
        "enabled":            _c("LEARNING_ENABLED", True),
        "dir":                _c("LEARNING_DIR", "data/learning"),
        "style_enabled":      _c("LEARNING_STYLE_ENABLED", True),
        "exemplars_k":        _c("LEARNING_EXEMPLARS_K", 2),
        "exemplar_min_reward": _c("LEARNING_EXEMPLAR_MIN_REWARD", 0.5),
        "engagement_window":  _c("LEARNING_ENGAGEMENT_WINDOW", 20.0),
        "repeat_sim":         _c("LEARNING_REPEAT_SIM", 0.80),
        "reward_explicit":    _c("LEARNING_REWARD_EXPLICIT", 1.0),
        "reward_thumbs":      _c("LEARNING_REWARD_THUMBS", 0.6),
        "reward_thanks":      _c("LEARNING_REWARD_THANKS", 0.5),
        "reward_engaged":     _c("LEARNING_REWARD_ENGAGED", 0.25),
        "reward_repeat":      _c("LEARNING_REWARD_REPEAT", -0.5),
        "reward_correction":  _c("LEARNING_REWARD_CORRECTION", -1.0),
        "reward_abandon":     _c("LEARNING_REWARD_ABANDON", -0.2),
        "min_turns_to_steer": _c("LEARNING_MIN_TURNS_TO_STEER", 8),
        "pending_path":       _c("PENDING_FACTS_PATH",
                                 "data/learning/pending_facts.txt"),
        "knowledge_path":     _c("KNOWLEDGE_PATH", "data/knowledge.txt"),
        "diary_enabled":      _c("DIARY_ENABLED", True),
        "diary_dir":          _c("DIARY_DIR", "data/diary"),
        "diary_hour":         _c("DIARY_WRITE_HOUR", 22),
    }


_cfg.update(_defaults())


def _print_once(where, exc):
    key = (where, type(exc).__name__)
    if key in _errors_printed:
        return
    _errors_printed.add(key)
    print(f"[learn] {where} failed ({type(exc).__name__}: {exc}) — "
          f"ignoring further errors of this kind")


def _safe(default=None):
    """Public functions never raise: print once, return `default`."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            try:
                return fn(*a, **kw)
            except Exception as e:                     # noqa: BLE001
                _print_once(fn.__name__, e)
                return default() if callable(default) else default
        return wrapper
    return deco


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(_ROOT, path)


def _p(name):
    return os.path.join(_abs(_cfg["dir"]), name)


def _now(now):
    return time.time() if now is None else float(now)


def _atomic_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except Exception as e:                             # noqa: BLE001
        _print_once(f"read {os.path.basename(path)}", e)
        return default


# ══════════════════════════════════════════════════════════════════════════════
# Setup
# ══════════════════════════════════════════════════════════════════════════════
@_safe()
def configure(rng=None, **kw):
    """Override settings (tests, the training script): any key of the config
    table above (dir, pending_path, diary_dir, diary_hour, reward_*, …) plus
    rng (a numpy Generator or an int seed). Clears the in-memory state so the
    next call loads from the (new) directory."""
    global _rng
    with _lock:
        for k, v in kw.items():
            if k not in _cfg:
                raise KeyError(f"unknown learning setting {k!r}")
            _cfg[k] = v
        if rng is not None:
            _rng = (rng if isinstance(rng, np.random.Generator)
                    else np.random.default_rng(rng))
        _unload()


def _unload():
    global _loaded, _bandit, _exemplars, _ex_emb, _profiles
    global _pending_thumbs, _profile_saved_at
    _loaded = False
    _bandit, _exemplars, _ex_emb, _profiles = {}, [], None, {}
    _pending_thumbs = None
    _profile_saved_at = 0.0
    _presence.update(last_face=None, absent_since=None, name=None)
    _diary_cache.update(key=None, sents=[], emb=None)
    _diary_attempts.clear()


@_safe()
def set_embedder(fn):
    """fn(list[str]) -> L2-normalised rows (np.ndarray) or None — brain.embed.
    Without one, exemplars / repeat detection / diary retrieval are off."""
    global _embed
    _embed = fn


def _embed_texts(texts):
    """Rows for `texts`, or None (no embedder, or it failed — a broken
    embedder costs exemplars / repeats / diary lookups, never a reward)."""
    if _embed is None or not texts:
        return None
    try:
        out = _embed(list(texts))
    except Exception as e:                             # noqa: BLE001
        _print_once("embedder", e)
        return None
    if out is None:
        return None
    return np.asarray(out, dtype=np.float32)


def _ensure_loaded():
    global _loaded, _bandit, _exemplars, _ex_emb, _profiles
    if _loaded:
        return
    _loaded = True
    _bandit = _read_json(_p("bandit.json"), {})
    _profiles = _read_json(_p("profiles.json"), {})
    _exemplars = _read_json(_p("exemplars.json"), [])
    _ex_emb = None
    try:
        with np.load(_p("exemplars.npz")) as z:
            emb = z["emb"]
        if len(emb) == len(_exemplars):
            _ex_emb = emb.astype(np.float32)
        else:
            print("[learn] exemplar embeddings out of sync — re-embedding")
    except FileNotFoundError:
        pass
    except Exception as e:                             # noqa: BLE001
        _print_once("load exemplars.npz", e)
    if _ex_emb is None and _exemplars:          # None without an embedder
        _ex_emb = _embed_texts([e["q"] for e in _exemplars])


@_safe()
def reset(delete_files=False):
    """Forget everything in memory; with delete_files also wipe LEARNING_DIR's
    bandit / exemplars / profiles (pending facts and the diary are kept)."""
    with _lock:
        if delete_files:
            for name in ("bandit.json", "exemplars.json", "exemplars.npz",
                         "profiles.json"):
                try:
                    os.remove(_p(name))
                except FileNotFoundError:
                    pass
        _unload()


# ══════════════════════════════════════════════════════════════════════════════
# Contexts + the style bandit
# ══════════════════════════════════════════════════════════════════════════════
def daypart(now=None):
    h = _dt.datetime.fromtimestamp(_now(now)).hour
    if 5 <= h < 12:
        return "morning"
    if 12 <= h < 17:
        return "afternoon"
    if 17 <= h < 22:
        return "evening"
    return "night"


def _norm(text):
    t = (text or "").lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    t = re.sub(r"\bthats\b", "that's", t)
    t = re.sub(r"\byoure\b", "you're", t)
    return re.sub(r"\s+", " ", t).strip()


@_safe("general")
def question_kind(text, kb_score=0.0):
    """"kb" (the knowledge base covers it), "scope" (live data / devices /
    things Luna can't do) or "general"."""
    if kb_score is not None and kb_score >= KB_KIND_MIN_SIM:
        return "kb"
    if _SCOPE_RE.search(_norm(text)):
        return "scope"
    return "general"


@_safe("stranger|general|day")
def context_key(person, kind, now=None):
    """"known|stranger" + "|" + kind + "|" + daypart."""
    who = "known" if person else "stranger"
    return f"{who}|{kind or 'general'}|{daypart(now)}"


@_safe((None, None))
def choose_style(context, rng=None):
    """(arm, instruction) for this context — uniform exploration until the
    context has LEARNING_MIN_TURNS_TO_STEER rewarded turns, then Thompson
    sampling. (None, None) when learning / style steering is off."""
    if not (_cfg["enabled"] and _cfg["style_enabled"]):
        return None, None
    r = rng if rng is not None else _rng
    with _lock:
        _ensure_loaded()
        ctx = _bandit.get(context) or {}
        n = int(ctx.get("_n", 0))
        if n < int(_cfg["min_turns_to_steer"]):
            arm = ARMS[int(r.integers(len(ARMS)))]
        else:
            samples = []
            for a in ARMS:
                alpha, beta = ctx.get(a, (1.0, 1.0))
                samples.append(r.beta(alpha, beta))
            arm = ARMS[int(np.argmax(samples))]
    return arm, style_instruction(arm)


def _bandit_update(arm, context, r, save=True):
    if arm not in ARMS or not context:
        return
    r = float(np.clip(r, -1.0, 1.0))
    ctx = _bandit.setdefault(context, {"_n": 0})
    alpha, beta = ctx.get(arm, (1.0, 1.0))
    ctx[arm] = [round(alpha + (r + 1.0) / 2.0, 4),
                round(beta + (1.0 - r) / 2.0, 4)]
    ctx["_n"] = int(ctx.get("_n", 0)) + 1
    if save:
        _atomic_json(_p("bandit.json"), _bandit)


@_safe()
def update(arm, context, r):
    """Bandit update: alpha += (r+1)/2, beta += (1-r)/2, r in [-1, 1]."""
    with _lock:
        _ensure_loaded()
        _bandit_update(arm, context, r)


@_safe(dict)
def bandit_state():
    with _lock:
        _ensure_loaded()
        return json.loads(json.dumps(_bandit))


@_safe(dict)
def arm_means(context):
    """{arm: posterior mean} for one context (for reports)."""
    with _lock:
        _ensure_loaded()
        ctx = _bandit.get(context) or {}
        return {a: ctx[a][0] / (ctx[a][0] + ctx[a][1])
                for a in ARMS if a in ctx}


# ══════════════════════════════════════════════════════════════════════════════
# Rewards
# ══════════════════════════════════════════════════════════════════════════════
def _has(norm, phrases):
    return any(re.search(r"\b" + re.escape(p) + r"\b", norm) for p in phrases)


@_safe(0)
def feedback_sign(text, last_reply_asked=False):
    """+1 praise, -1 "that's wrong", 0 neither. Weak positives ("that's
    right") only count when Luna's last reply wasn't itself a question."""
    n = _norm(text)
    if _has(n, NEGATIVE_PHRASES):
        return -1
    if _has(n, POSITIVE_PHRASES):
        return 1
    if not last_reply_asked and _has(n, WEAK_POSITIVE_PHRASES):
        return 1
    return 0


@_safe(0)
def feedback_command(text):
    """What commands.py consumes: a SHORT utterance that is explicit feedback
    on the last answered turn (+1 / -1), else 0."""
    n = _norm(text)
    if not n or len(n.split()) > FEEDBACK_MAX_WORDS:
        return 0
    prev = _prev_answered()
    if prev is None:
        return 0
    return feedback_sign(n, _asked(prev))


def _asked(turn):
    return bool((turn.get("reply") or "").rstrip().endswith("?"))


def _is_thanks(norm):
    words = set(norm.split())
    return bool(words & THANKS_WORDS) or "love it" in norm


def _turn_start(turn):
    total = (turn.get("timings") or {}).get("total_s") or 0.0
    return float(turn.get("ts", 0.0)) - float(total)


def similarity(a, b):
    """Cosine of two texts, or None (no embedder / it failed)."""
    emb = _embed_texts([a, b])
    if emb is None or len(emb) < 2:
        return None
    return float(emb[0] @ emb[1])


def score_signals(prev, text, person=None, now=None, thumbs_ts=None,
                  similarity=None):
    """Pure: the reward signals `text` (said at `now`) gives the previous
    turn `prev`. similarity(a, b) -> cosine or None. Returns
    [(signal, value)] — empty when the utterance says nothing about it."""
    now = _now(now)
    end = float(prev.get("ts", now))
    dt = now - end
    if dt > REWARD_MAX_GAP:
        return []
    prev_person = prev.get("person")
    if prev_person and person and prev_person != person:
        return []                                   # someone else now
    n = _norm(text)
    out = []
    sign = feedback_sign(n, _asked(prev))
    if sign > 0:
        out.append(("explicit", float(_cfg["reward_explicit"])))
    elif sign < 0:
        out.append(("correction", float(_cfg["reward_correction"])))
    elif _is_thanks(n):
        out.append(("thanks", float(_cfg["reward_thanks"])))
    repeat = False
    q = prev.get("heard_clean") or ""
    if q and n and similarity is not None and sign == 0:
        sim = similarity(n, q)
        if sim is not None and sim >= float(_cfg["repeat_sim"]):
            repeat = True
            out.append(("repeat", float(_cfg["reward_repeat"])))
    if (not repeat and sign >= 0 and n
            and dt <= float(_cfg["engagement_window"])):
        out.append(("engaged", float(_cfg["reward_engaged"])))
    if thumbs_ts is not None and _thumbs_valid(prev, thumbs_ts):
        out.append(("thumbs", float(_cfg["reward_thumbs"])))
    return out


def _thumbs_valid(prev, thumbs_ts):
    return (_turn_start(prev) - 1.0 <= thumbs_ts
            <= float(prev.get("ts", 0)) + float(_cfg["engagement_window"]))


def _prev_answered():
    t = conversation_log.last_turn()
    if not t or t.get("path") not in ANSWER_PATHS:
        return None
    if t.get("reward") is not None:
        return None
    return t


def _learn_from(turn, r, now, save=True):
    """Everything a rewarded turn teaches: bandit, exemplars, profile."""
    _bandit_update(turn.get("style_arm"), turn.get("context"), r, save=save)
    if (r >= float(_cfg["exemplar_min_reward"])
            and turn.get("path") in EXEMPLAR_PATHS
            and turn.get("heard_clean") and turn.get("reply")):
        _admit_exemplar(turn["heard_clean"], turn["reply"], r,
                        turn.get("style_arm"), turn.get("context"), now,
                        save=save)
    name = turn.get("person")
    if name:
        p = _profile(name)
        p["rewarded"] = int(p.get("rewarded", 0)) + 1
        p["reward_sum"] = round(float(p.get("reward_sum", 0.0)) + r, 4)
        p["mean_reward"] = round(p["reward_sum"] / p["rewarded"], 3)
        arm = turn.get("style_arm")
        if arm in ARMS:
            s = p.setdefault("arm_stats", {}).setdefault(arm, [0.0, 0])
            s[0], s[1] = round(s[0] + r, 4), s[1] + 1
            p["best_arm"] = _best_arm(p["arm_stats"])
        if save:
            _save_profiles()


def _best_arm(stats):
    cands = [(v[0] / v[1], a) for a, v in stats.items() if v[1] >= 2]
    return max(cands)[1] if cands else None


def _apply(turn, signals, now):
    r = float(np.clip(sum(v for _, v in signals), -1.0, 1.0))
    r = round(r, 3)
    conversation_log.update_turn(
        turn.get("turn_id"), reward=r,
        reward_signals=[[s, round(v, 3)] for s, v in signals])
    _learn_from(turn, r, now)
    sig = ", ".join(f"{s} {v:+.2f}" for s, v in signals)
    console.detail(f"[learn] reward {r:+.2f} for \"{(turn.get('heard_clean') or '')[:40]}\""
          f" ({sig}) arm={turn.get('style_arm')}")
    return r


def _take_thumbs(prev):
    """The pending thumbs-up timestamp if it belongs to `prev` (consumed)."""
    global _pending_thumbs
    ts = _pending_thumbs
    if ts is None:
        return None
    _pending_thumbs = None
    return ts if (prev is not None and _thumbs_valid(prev, ts)) else None


@_safe()
def on_utterance(text, person=None, now=None, extra_signals=()):
    """A new utterance arrived: score the PREVIOUS answered turn from it.
    Returns the reward applied, or None when there was nothing to score."""
    if not _cfg["enabled"]:
        return None
    now = _now(now)
    with _lock:
        _ensure_loaded()
        prev = _prev_answered()
        if prev is None:
            return None
        thumbs = _take_thumbs(prev)
        signals = score_signals(prev, text, person, now, thumbs, similarity)
        signals += [(s, float(v)) for s, v in extra_signals]
        if not signals:
            return None
        return _apply(prev, signals, now)


@_safe()
def add_signals(signals, now=None, max_age=EXPLICIT_MAX_AGE):
    """Apply explicit signals to the last unrewarded answered turn (if it
    is at most `max_age` old). Returns the reward or None."""
    if not _cfg["enabled"]:
        return None
    now = _now(now)
    with _lock:
        _ensure_loaded()
        prev = _prev_answered()
        if prev is None or now - float(prev.get("ts", now)) > max_age:
            return None
        signals = [(s, float(v)) for s, v in signals]
        thumbs = _take_thumbs(prev)
        if thumbs is not None:
            signals.append(("thumbs", float(_cfg["reward_thumbs"])))
        return _apply(prev, signals, now)


@_safe(False)
def explicit_feedback(sign, now=None):
    """"Good answer" (+1) / "that's wrong" (-1) about the last reply. True if
    a turn was rewarded."""
    sig = (("explicit", _cfg["reward_explicit"]) if sign > 0
           else ("correction", _cfg["reward_correction"]))
    return add_signals([sig], now=now) is not None


@_safe()
def last_answered_question():
    """What the person asked in the last answered turn (rewarded or not)."""
    t = conversation_log.last_turn()
    if t and t.get("path") in ANSWER_PATHS:
        return t.get("heard_clean")
    return None


@_safe()
def note_gesture(kind, now=None):
    """A held gesture from gesture_module. A thumbs-up is kept pending and
    counted with the next reward of the turn it followed."""
    global _pending_thumbs
    if kind == "thumbs_up" and _cfg["enabled"]:
        with _lock:
            _pending_thumbs = _now(now)


@_safe()
def on_person_left(name=None, now=None):
    """The person walked off. An unrewarded reply under ABANDON_SECS old gets
    LEARNING_REWARD_ABANDON — unless a thumbs-up is pending for it."""
    if not _cfg["enabled"]:
        return None
    now = _now(now)
    with _lock:
        _ensure_loaded()
        prev = _prev_answered()
        if prev is None or now - float(prev.get("ts", now)) >= ABANDON_SECS:
            return None
        thumbs = _take_thumbs(prev)
        if thumbs is not None:
            return _apply(prev, [("thumbs", float(_cfg["reward_thumbs"]))], now)
        return _apply(prev, [("abandon", float(_cfg["reward_abandon"]))], now)


@_safe()
def departure_tick(face_count, name=None, now=None):
    """Called ~1/s with the vision state. Notices a departure (no face for
    DEPART_SECS after someone was there during the last reply) and calls
    on_person_left; also counts visits of recognised people. No camera →
    no face ever seen → never penalises anything."""
    now = _now(now)
    with _lock:
        if face_count and face_count > 0:
            _presence["last_face"] = now
            _presence["absent_since"] = None
            if name:
                _presence["name"] = name
                note_visit(name, now)
            return None
        if _presence["last_face"] is None:
            return None
        if _presence["absent_since"] is None:
            _presence["absent_since"] = now
        if now - _presence["absent_since"] < DEPART_SECS:
            return None
        prev = _prev_answered()
        if prev is None:
            return None
        if _presence["last_face"] < _turn_start(prev) - PRESENT_SLACK_SECS:
            return None                       # nobody was there for that turn
        return on_person_left(_presence["name"], now)


# ══════════════════════════════════════════════════════════════════════════════
# Exemplars
# ══════════════════════════════════════════════════════════════════════════════
def _save_exemplars():
    _atomic_json(_p("exemplars.json"), _exemplars)
    if _ex_emb is not None:
        path = _p("exemplars.npz")
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(f, emb=_ex_emb)
        os.replace(tmp, path)


def _admit_exemplar(q, reply, r, arm, context, now, save=True):
    global _exemplars, _ex_emb
    if _exemplars and (_ex_emb is None or len(_ex_emb) != len(_exemplars)):
        _ex_emb = _embed_texts([e["q"] for e in _exemplars])
        if _ex_emb is None:
            return False                 # never drop stored ones blindly
    v = _embed_texts([q])
    if v is None:
        return False
    v = v[0]
    item = {"q": q, "reply": reply, "reward": round(float(r), 3),
            "arm": arm, "context": context, "ts": round(now, 1)}
    if _ex_emb is not None and len(_ex_emb) == len(_exemplars) and _exemplars:
        sims = _ex_emb @ v
        j = int(np.argmax(sims))
        if sims[j] >= EXEMPLAR_DEDUP_SIM:
            if r >= _exemplars[j]["reward"]:
                _exemplars[j] = item
                _ex_emb[j] = v
                if save:
                    _save_exemplars()
            return True
        _exemplars.append(item)
        _ex_emb = np.vstack([_ex_emb, v[None, :]])
    else:
        _exemplars, _ex_emb = [item], v[None, :].astype(np.float32)
    if len(_exemplars) > EXEMPLAR_CAP:
        keep = sorted(range(len(_exemplars)),
                      key=lambda i: (_exemplars[i]["reward"],
                                     _exemplars[i]["ts"]),
                      reverse=True)[:EXEMPLAR_CAP]
        keep.sort()
        _exemplars = [_exemplars[i] for i in keep]
        _ex_emb = _ex_emb[keep]
    if save:
        _save_exemplars()
    return True


@_safe(list)
def exemplars_for(text, k=None):
    """[(question, reply)] of liked replies to similar questions, best
    first, cosine >= EXEMPLAR_MIN_SIM."""
    if not _cfg["enabled"]:
        return []
    k = int(_cfg["exemplars_k"] if k is None else k)
    if k <= 0:
        return []
    with _lock:
        _ensure_loaded()
        if _ex_emb is None or not _exemplars or len(_ex_emb) != len(_exemplars):
            return []
        v = _embed_texts([text])
        if v is None:
            return []
        sims = _ex_emb @ v[0]
        order = np.argsort(sims)[::-1][:k]
        return [(_exemplars[i]["q"], _exemplars[i]["reply"])
                for i in order if sims[i] >= EXEMPLAR_MIN_SIM]


@_safe(list)
def all_exemplars():
    with _lock:
        _ensure_loaded()
        return [dict(e) for e in _exemplars]


# ══════════════════════════════════════════════════════════════════════════════
# People
# ══════════════════════════════════════════════════════════════════════════════
def _profile(name):
    return _profiles.setdefault(name, {
        "visits": 0, "last_seen": None, "prev_visit": None, "topics": {},
        "turns": 0, "rewarded": 0, "reward_sum": 0.0, "mean_reward": None,
        "arm_stats": {}, "best_arm": None})


def _save_profiles():
    global _profile_saved_at
    _atomic_json(_p("profiles.json"), _profiles)
    _profile_saved_at = time.time()


@_safe()
def note_visit(name, now=None):
    """Recognised `name` in front of the camera. Counts a new visit after
    VISIT_GAP_SECS away; keeps last_seen fresh (saved at most once a
    minute unless a visit was counted)."""
    if not name or not _cfg["enabled"]:
        return
    now = _now(now)
    with _lock:
        _ensure_loaded()
        p = _profile(name)
        last = p.get("last_seen")
        if last is None or now - last > VISIT_GAP_SECS:
            p["prev_visit"] = last
            p["visits"] = int(p.get("visits", 0)) + 1
            p["last_seen"] = now
            _save_profiles()
            return
        p["last_seen"] = now
        if time.time() - _profile_saved_at > PROFILE_SAVE_SECS:
            _save_profiles()


def _topic_of(turn):
    ctx = turn.get("context") or ""
    kind = ctx.split("|")[1] if ctx.count("|") >= 2 else None
    matches = turn.get("kb_matches") or []
    if kind == "kb" and matches:
        words = str(matches[0][0]).split()
        return " ".join(words[:6])
    if turn.get("path") == "vision":
        return "what you can see"
    return None


@_safe()
def observe_turn(turn, save=True):
    """An answered turn was logged: count it (and its topic) for the person."""
    name = turn.get("person")
    if not name or not _cfg["enabled"] or turn.get("path") not in ANSWER_PATHS:
        return
    with _lock:
        _ensure_loaded()
        p = _profile(name)
        p["turns"] = int(p.get("turns", 0)) + 1
        topic = _topic_of(turn)
        if topic:
            p["topics"][topic] = int(p["topics"].get(topic, 0)) + 1
        if p.get("last_seen") is None:
            p["last_seen"] = float(turn.get("ts") or time.time())
            p["visits"] = max(1, int(p.get("visits", 0)))
        if save:
            _save_profiles()


def _human_ago(secs):
    secs = max(0.0, secs)
    if secs < 3600:
        return "earlier today" if secs > 600 else "just now"
    if secs < 86400:
        return "earlier today"
    days = int(secs / 86400 + 0.5)
    if days < 1:
        return "earlier today"
    if days == 1:
        return "yesterday"
    if days < 14:
        return f"{days} days ago"
    return f"{days // 7} weeks ago"


@_safe()
def profile_context(name, now=None):
    """One short sentence about `name` for the prompt, or None."""
    if not name or not _cfg["enabled"]:
        return None
    now = _now(now)
    with _lock:
        _ensure_loaded()
        p = _profiles.get(name)
        if not p:
            return None
        visits = int(p.get("visits", 0))
        parts = []
        if visits > 1:
            s = f"You have met {name} on {visits} visits"
            if p.get("prev_visit"):
                s += f" (the last one {_human_ago(now - p['prev_visit'])})"
            parts.append(s)
        topics = sorted(p.get("topics", {}).items(), key=lambda kv: -kv[1])
        if topics and topics[0][1] >= 2:
            parts.append(f"they often ask about \"{topics[0][0]}\"")
        arm = p.get("best_arm")
        if arm in ARMS and (p.get("mean_reward") or 0) > 0:
            l, t, f = arm.split("|")
            parts.append(f"they respond best to {_SHORT_DESC[l]}, "
                         f"{_SHORT_DESC[t]} replies")
        if not parts:
            return None
        s = "; ".join(parts)
        if not s.startswith("You"):
            s = f"About {name}: " + s
        return s[0].upper() + s[1:] + "."


@_safe(dict)
def profiles():
    with _lock:
        _ensure_loaded()
        return json.loads(json.dumps(_profiles))


# ══════════════════════════════════════════════════════════════════════════════
# Facts awaiting approval
# ══════════════════════════════════════════════════════════════════════════════
def _clean_line(text, limit=300):
    return re.sub(r"\s+", " ", str(text or "")).strip()[:limit]


@_safe(False)
def teach(statement, person=None, question=None, now=None):
    """Queue a fact for human approval (PENDING_FACTS_PATH, one JSON per
    line). `question` is the suggested knowledge.txt question (the one Luna
    got wrong, for corrections). True if queued."""
    statement = _clean_line(statement)
    if not statement:
        return False
    rec = {"ts": round(_now(now), 1), "person": person,
           "statement": statement,
           "question": _clean_line(question) or None}
    path = _abs(_cfg["pending_path"])
    with _lock:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[learn] fact queued for approval: {statement!r}")
    return True


@_safe(list)
def pending_facts():
    path = _abs(_cfg["pending_path"])
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    out.append({"statement": line, "question": None,
                                "person": None, "ts": None})
    except FileNotFoundError:
        pass
    return out


def _write_pending(items):
    path = _abs(_cfg["pending_path"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


@_safe()
def approve_fact(index, question=None, knowledge_path=None):
    """Move pending fact #index (1-based) into knowledge.txt as
    "question:statement". Returns the line written, or None."""
    with _lock:
        items = pending_facts()
        if not 1 <= index <= len(items):
            return None
        item = items[index - 1]
        q = _clean_line(question or item.get("question") or "")
        q = q.replace(":", " ").strip().rstrip("?")
        if not q:
            return None
        line = f"{q}:{_clean_line(item['statement'])}"
        kp = _abs(knowledge_path or _cfg["knowledge_path"])
        needs_nl = False
        if os.path.exists(kp) and os.path.getsize(kp) > 0:
            with open(kp, "rb") as f:
                f.seek(-1, os.SEEK_END)
                needs_nl = f.read(1) != b"\n"
        with open(kp, "a", encoding="utf-8") as f:
            f.write(("\n" if needs_nl else "") + line + "\n")
        del items[index - 1]
        _write_pending(items)
        return line


@_safe()
def reject_fact(index):
    """Drop pending fact #index (1-based). Returns the dropped item."""
    with _lock:
        items = pending_facts()
        if not 1 <= index <= len(items):
            return None
        item = items.pop(index - 1)
        _write_pending(items)
        return item


# ══════════════════════════════════════════════════════════════════════════════
# Diary
# ══════════════════════════════════════════════════════════════════════════════
DIARY_SYSTEM = (
    "You are Luna, a small friendly desk robot at TKM College of Engineering. "
    "Write today's diary entry: exactly three short, warm sentences in the "
    "first person, plain text, no lists. Mention what people asked about "
    "most, anything you got wrong or couldn't answer, and one thing you want "
    "to do better. Use only the facts in the notes.")


def diary_path(date):
    if isinstance(date, (int, float)):
        date = _dt.datetime.fromtimestamp(date).strftime("%Y-%m-%d")
    return os.path.join(_abs(_cfg["diary_dir"]), f"{date}.txt")


@_safe("")
def build_digest(turns, date=None):
    """A compact plain-text summary of one day's log for the diary prompt."""
    answered = [t for t in turns if t.get("path") in ANSWER_PATHS]
    ignored = [t for t in turns if t.get("path") == "ignored"]
    people = sorted({t["person"] for t in turns if t.get("person")})
    liked = [t for t in answered if (t.get("reward") or 0) > 0.3]
    disliked = [t for t in answered if (t.get("reward") or 0) < -0.3]
    lines = [f"Notes for {date or 'today'}: I answered {len(answered)} "
             f"questions and ignored {len(ignored)} remarks not meant for me."]
    if people:
        lines.append("People I recognised: " + ", ".join(people) + ".")
    lines.append(f"Replies people liked: {len(liked)}; replies that went "
                 f"badly: {len(disliked)}.")
    lines.append("Some of the conversations:")
    pick = (disliked[:4] + liked[:4]
            + [t for t in answered if t not in liked and t not in disliked])
    for t in pick[:12]:
        r = t.get("reward")
        tag = "" if r is None else f" [reward {r:+.1f}]"
        q = (t.get("heard_clean") or "")[:80]
        a = (t.get("reply") or "")[:120]
        lines.append(f"- Asked: {q} | I said: {a}{tag}")
    return "\n".join(lines)


def _first_sentences(text, n=3):
    t = re.sub(r"\s+", " ", (text or "").replace("*", "")).strip()
    sents = re.split(r"(?<=[.!?])\s+", t)
    return " ".join(s for s in sents[:n] if s).strip()


@_safe(False)
def maybe_write_diary(llm_ask, now=None, force=False):
    """After DIARY_WRITE_HOUR (or with force): if today's diary file is
    missing and today has >= DIARY_MIN_TURNS answered turns, digest the log
    and ask the LLM for a 3-sentence entry. llm_ask(messages) -> str|None.
    True when a diary file was written."""
    if not (_cfg["enabled"] and _cfg["diary_enabled"]):
        return False
    now = _now(now)
    day = _dt.datetime.fromtimestamp(now)
    date = day.strftime("%Y-%m-%d")
    if not force and day.hour < int(_cfg["diary_hour"]):
        return False
    path = diary_path(date)
    if os.path.exists(path):
        return False
    if _diary_attempts.get(date, 0) >= DIARY_MAX_ATTEMPTS:
        return False
    turns = conversation_log.iter_day(date)
    answered = [t for t in turns
                if t.get("path") in ANSWER_PATHS and t.get("reply")]
    if len(answered) < DIARY_MIN_TURNS:
        return False
    digest = build_digest(turns, date)
    reply = None
    try:
        reply = llm_ask([{"role": "system", "content": DIARY_SYSTEM},
                         {"role": "user", "content": digest}])
    except Exception as e:                             # noqa: BLE001
        _print_once("diary llm", e)
    text = _first_sentences(reply or "")
    if not text:
        _diary_attempts[date] = _diary_attempts.get(date, 0) + 1
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    os.replace(tmp, path)
    console.detail(f"[learn] diary written: {path}")
    return True


def _diary_sentences():
    """(sentences, embeddings) over the recent diary files, cached by the
    directory's (name, mtime) listing."""
    d = _abs(_cfg["diary_dir"])
    try:
        names = sorted(n for n in os.listdir(d) if n.endswith(".txt"))
    except FileNotFoundError:
        return [], None
    names = names[-DIARY_MAX_FILES:]
    key = tuple((n, os.path.getmtime(os.path.join(d, n))) for n in names)
    if key == _diary_cache["key"]:
        return _diary_cache["sents"], _diary_cache["emb"]
    sents = []
    for n in names:
        with open(os.path.join(d, n), "r", encoding="utf-8") as f:
            txt = f.read()
        for s in re.split(r"(?<=[.!?])\s+", txt.strip()):
            if len(s.split()) >= 4:
                sents.append((n[:-4], s.strip()))
    emb = _embed_texts([s for _, s in sents]) if sents else None
    _diary_cache.update(key=key, sents=sents, emb=emb)
    return sents, emb


@_safe()
def diary_context(text):
    """The diary sentence most similar to `text` ("On 2026-09-23: …"), or
    None when nothing is relevant enough."""
    if not (_cfg["enabled"] and _cfg["diary_enabled"]):
        return None
    with _lock:
        sents, emb = _diary_sentences()
        if not sents or emb is None:
            return None
        v = _embed_texts([text])
        if v is None:
            return None
        sims = emb @ v[0]
        i = int(np.argmax(sims))
        if sims[i] < DIARY_MIN_SIM:
            return None
        date, s = sents[i]
        return f"On {date}: {s}"


# ══════════════════════════════════════════════════════════════════════════════
# Replay (scripts/train_learning.py)
# ══════════════════════════════════════════════════════════════════════════════
@_safe(0)
def replay_turns(turns):
    """Rebuild bandit / exemplars / profiles from logged turns that carry a
    reward (and count every answered turn for profiles). Saves once at the
    end. Returns the number of rewarded turns applied."""
    n = 0
    with _lock:
        _ensure_loaded()
        for t in turns:
            if t.get("path") not in ANSWER_PATHS:
                continue
            observe_turn(t, save=False)
            r = t.get("reward")
            if r is None:
                continue
            _learn_from(t, float(r), float(t.get("ts") or time.time()),
                        save=False)
            n += 1
        _atomic_json(_p("bandit.json"), _bandit)
        _save_exemplars()
        _save_profiles()
    return n
