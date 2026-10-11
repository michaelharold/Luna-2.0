# brain.py
"""
brain.py — how Luna decides what to say. Fully offline.

The answer pipeline, in priority order:

  1. Voice commands   — volume, sleep, face enrollment, "who am I"… handled
                        instantly by commands.py, no models involved.
  2. Vision questions — "what do you see?" builds a live scene description
                        (scene_context.py) and answers through the local LLM.
  3. KB direct match  — a question the knowledge base is confident about is
                        answered verbatim from it (ANSWER_OFFLINE_FIRST).
  4. Local LLM        — everything else goes to the on-device llama.cpp model
                        (llm_local.py), with the top-K most relevant KB entries
                        retrieved into the prompt instead of the whole file —
                        short prompts keep the Pi's prefill fast.
  5. KB fallback      — if the LLM is missing/broken, the closest KB entry.

The only optional network use is web_rag.py: when WEB_RAG_URL is set and the
Pi happens to be online it refreshes a cached copy of that website, and the
most relevant cached chunks join the knowledge facts in the prompt. No keys,
no quotas, and it keeps working offline from the cache.
knowledge.txt still hot-reloads while Luna is running.
"""

import os
import re
import threading
import time
import numpy as np
from sentence_transformers import SentenceTransformer

from text_to_speech import speak
from servo_module import servo
import face
import llm_local
import commands
import scene_context
import persona
from shared_state import state
from config import (
    KNOWLEDGE_PATH,
    LLM_MAX_HISTORY,
    LLM_STREAMING,
    LLM_KB_TOP_K,
    LLM_KB_MIN_SIM,
    LLM_BACKEND,
    OLLAMA_BAKE_KNOWLEDGE,
    ANSWER_OFFLINE_FIRST,
    LOCAL_ANSWER_THRESH,
    LOCAL_MATCH_THRESH,
    ENABLE_SCENE_CONTEXT,
    ENABLE_SCENE_MEMORY,
    SCENE_MEMORY_IN_PROMPT,
    WEB_RAG_ENABLED,
    WEB_RAG_URL,
    WIKI_RAG_ENABLED,
)

# Website retrieval (web_rag.py) is optional: it only does anything when
# WEB_RAG_URL is set, and only reaches the network when the Pi is online.
try:
    import web_rag
except Exception as _e:                      # missing module / import error
    print(f"[brain] web_rag unavailable ({_e}) — website retrieval off")
    web_rag = None

# Live Wikipedia fallback for genuine general-knowledge questions — see
# wiki_rag.py's module docstring for why this only ever runs when the
# knowledge base found nothing relevant at all (_turn_context below).
try:
    import wiki_rag
except Exception as _e:                      # missing module / import error
    print(f"[brain] wiki_rag unavailable ({_e}) — general-knowledge fallback off")
    wiki_rag = None

# ── Knowledge base (hot-reloadable) ───────────────────────────────────────────
# Loaded at startup AND re-read automatically whenever data/knowledge.txt changes
# on disk (see _maybe_reload_knowledge, called on every question), so edits take
# effect on a running Luna without a restart.
print("[brain] Loading knowledge base...")

questions           = []
answers             = []
knowledge_text      = ""
question_embeddings = None
_knowledge_mtime    = None

# ── Sentence transformer — retrieval + offline matching ──────────────────────
# config.py switches huggingface_hub to offline mode before anything imports it
# whenever this model is already cached (setting HF_HUB_OFFLINE here, after the
# import above, is silently ignored), so a disconnected Pi loads it straight from
# disk instead of retrying huggingface.co first. On a fresh install with internet
# it downloads once; if it is neither cached nor downloadable, Luna degrades to
# plain keyword matching instead of crashing.
print("[brain] Loading sentence transformer...")
_st_model = None
try:
    _st_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"[brain] Could not load sentence transformer ({e}) — "
          f"falling back to plain keyword matching for the knowledge base.")


def embed(texts):
    """L2-normalised sentence embeddings (rows) for a list of strings, or
    None when the sentence transformer isn't available. Shared with
    web_rag.py so the website index uses the very same model — one copy in
    RAM, and KB facts and website chunks are comparable on the same scale."""
    if _st_model is None:
        return None
    raw   = _st_model.encode(list(texts), batch_size=32, show_progress_bar=False)
    raw   = np.asarray(raw, dtype=np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / np.maximum(norms, 1e-9)


# ── Conversation log + learning (optional — never break a turn) ─────────────
# conversation_log records every turn (and learning's rewards for it);
# learning.py steers reply style, keeps liked replies as examples, remembers
# people and queues taught facts. Both share this module's embedder.
try:
    import conversation_log
except Exception as _e:                      # noqa: BLE001
    print(f"[learn] conversation log unavailable ({_e})")
    conversation_log = None
try:
    import learning
    learning.set_embedder(embed)
except Exception as _e:                      # noqa: BLE001
    print(f"[learn] learning unavailable ({_e})")
    learning = None

_system = persona.system_prompt()


def _load_knowledge():
    """(Re)read knowledge.txt and rebuild everything derived from it."""
    global questions, answers, knowledge_text, question_embeddings
    global _knowledge_mtime
    qs, ans = [], []
    with open(KNOWLEDGE_PATH, "r") as f:
        raw = f.read()
    for line in raw.splitlines():
        if ":" not in line:
            continue
        q, a = line.split(":", 1)
        qs.append(q.strip().lower())
        ans.append(a.strip())
    questions, answers, knowledge_text = qs, ans, raw.strip()

    if _st_model is not None and questions:
        question_embeddings = embed(questions)
    else:
        question_embeddings = None

    # The static system prompt (bake the whole KB in on the Ollama path — it
    # is prefix-cached there, see persona.py; llama.cpp's 2048 ctx can't).
    global _system
    bake = (LLM_BACKEND == "ollama" and OLLAMA_BAKE_KNOWLEDGE)
    _system = persona.system_prompt(knowledge_text if bake else None)

    try:
        _knowledge_mtime = os.path.getmtime(KNOWLEDGE_PATH)
    except OSError:
        _knowledge_mtime = None


def _maybe_reload_knowledge():
    """Re-read knowledge.txt if it changed since the last load, so edits to the
    file take effect on a running Luna without a restart."""
    try:
        m = os.path.getmtime(KNOWLEDGE_PATH)
    except OSError:
        return
    if m != _knowledge_mtime:
        print("[brain] knowledge.txt changed — reloading")
        _load_knowledge()
        print(f"[brain] reloaded {len(questions)} knowledge entries")


_load_knowledge()
print(f"[brain] {len(questions)} knowledge entries loaded")

# Website index: loads its cache (if any) and, when online, refreshes it in a
# background thread. Shares this module's embedder — nothing loads twice.
if web_rag is not None and WEB_RAG_ENABLED and WEB_RAG_URL:
    try:
        web_rag.start(embed)
    except Exception as _e:
        print(f"[brain] web_rag failed to start: {_e}")

# Start the first connectivity probe now, in the background, so the first
# general-knowledge question after boot can already use Wikipedia instead of
# finding "not probed yet" and skipping it.
if wiki_rag is not None and WIKI_RAG_ENABLED:
    wiki_rag.connectivity.state()

# ── The prompt ────────────────────────────────────────────────────────────────
# Persona + scope rules live in persona.py (shared with the Ollama Modelfile
# and the benchmark). The system prompt is STATIC per knowledge.txt version;
# everything that changes per turn — retrieved facts, website extracts, the
# live scene, who is talking — travels in the user message, so the Pi only
# prefills the new tokens each question (Ollama caches the static prefix).

def _turn_context(text, scene=None, with_kb=True):
    """Per-turn context dict for persona.build_user_turn().

    Scene memory rides in the `scene` slot: on an ordinary turn it is the
    whole scene (marked as background, so the model doesn't narrate it
    unprompted); on a vision question it follows the live description."""
    with state.lock:
        person = state.person_name
    memory = ""
    if ENABLE_SCENE_MEMORY and SCENE_MEMORY_IN_PROMPT:
        try:                                 # deferred + guarded: remembering
            import scene_memory              # must never break a voice turn
            memory = scene_memory.summary() or ""
        except Exception as e:               # noqa: BLE001
            print(f"[brain] scene memory unavailable: {e}")
    if memory:
        if scene:
            scene = f"{scene} Recently: {memory}"
        else:
            scene = f"(Background — mention only if it helps.) {memory}"
    kb_snippets = _top_k_kb(text) if with_kb else ()
    web_snippets = _top_k_web(text)
    # Wikipedia is only ever tried when the knowledge base found NOTHING —
    # otherwise a loosely campus-flavored question (even one about something
    # that doesn't exist, like a made-up college) would risk pulling in a
    # real but irrelevant article alongside genuine KB context. See
    # wiki_rag.py's module docstring.
    wiki_snippets = _top_k_wiki(text) if not kb_snippets else ()
    _turn["kb_matches"] = [[q, round(float(s), 3)] for q, _, s in kb_snippets]
    _turn["web_matches"] = [[src, round(float(s), 3)]
                            for src, _, s in web_snippets]
    _turn["wiki_matches"] = [[src, round(float(s), 3)]
                             for src, _, s in wiki_snippets]
    return dict(
        kb_snippets=kb_snippets,
        web_snippets=web_snippets,
        wiki_snippets=wiki_snippets,
        scene=scene,
        person=person,
        **_learning_context(text, person),
    )


# ── Learning hooks ────────────────────────────────────────────────────────────
# Per-turn bookkeeping for the conversation log: reset at the top of
# process(), filled in along the way, written once by _log_turn().
_turn = {}


def _begin_turn(text):
    with state.lock:
        person = state.person_name
        heard  = state.heard_text or ""
        loud   = state.utterance_loudness
    _turn.clear()
    _turn.update(t0=time.time(), text=text, person=person, loudness=loud,
                 heard_raw=heard if text.lower() in heard.lower() else text,
                 kb_matches=[], web_matches=[], wiki_matches=[], style_arm=None,
                 context=None, ttft=None)


def _set_learning_context(text, kb_score):
    """The bandit's context key for this turn (who × kind × daypart)."""
    if learning is None:
        return
    try:
        kind = learning.question_kind(text, kb_score)
        _turn["context"] = learning.context_key(_turn.get("person"), kind)
    except Exception as e:                   # noqa: BLE001
        print(f"[learn] context failed: {e}")


def _learning_context(text, person):
    """Style / exemplars / profile / diary for persona.format_context — {}
    when learning is off or fails (the prompt is then exactly as before)."""
    if learning is None:
        return {}
    try:
        extra = {}
        arm, instruction = learning.choose_style(_turn.get("context"))
        _turn["style_arm"] = arm
        if instruction:
            extra["style"] = instruction
        ex = learning.exemplars_for(text)
        if ex:
            extra["exemplars"] = ex
        prof = learning.profile_context(person) if person else None
        if prof:
            extra["profile"] = prof
        diary = learning.diary_context(text)
        if diary:
            extra["diary"] = diary
        return extra
    except Exception as e:                   # noqa: BLE001
        print(f"[learn] context extras failed: {e}")
        return {}


def _mark_first_token():
    if _turn.get("ttft") is None and "t0" in _turn:
        _turn["ttft"] = time.time() - _turn["t0"]


def _log_turn(path, reply=None):
    """One conversation-log record for the turn that just finished."""
    if conversation_log is None or "t0" not in _turn:
        return
    try:
        now  = time.time()
        ttft = _turn.get("ttft")
        rec = dict(
            heard_raw=_turn.get("heard_raw"), heard_clean=_turn.get("text"),
            loudness=_turn.get("loudness"), person=_turn.get("person"),
            path=path, kb_matches=_turn.get("kb_matches") or [],
            web_matches=_turn.get("web_matches") or [],
            wiki_matches=_turn.get("wiki_matches") or [], reply=reply,
            timings={"ttft_s": round(ttft, 3) if ttft is not None else None,
                     "total_s": round(now - _turn["t0"], 3)},
            # a style only shaped LLM-written replies (kb / fallback are
            # verbatim), so only those may credit or blame it
            style_arm=(_turn.get("style_arm")
                       if path in ("llm", "vision") else None),
            context=_turn.get("context"))
        conversation_log.new_turn(ts=now, **rec)
        if learning is not None:
            learning.observe_turn(conversation_log.last_turn() or {})
    except Exception as e:                   # noqa: BLE001
        print(f"[learn] could not log turn: {e}")


# ── Conversation history ──────────────────────────────────────────────────────
_history = []


def _push_user(text):
    _history.append({"role": "user", "content": text})
    while len(_history) > LLM_MAX_HISTORY:
        _history.pop(0)


def _drop_user():
    """Undo _push_user after a failed generation, so a dropped question doesn't
    poison the next request's history."""
    if _history and _history[-1]["role"] == "user":
        _history.pop()


def reset_history():
    """Clear the LLM conversation history. Called when a conversation window
    ends (see speech_to_text._expire_conversation) so one visitor's turns
    never carry over into the next visitor's session."""
    _history.clear()


# ── Local LLM ─────────────────────────────────────────────────────────────────

def _messages():
    return [{"role": "system", "content": _system}, *_history]


def _stream_llm(text, context):
    """Yield reply deltas from the local model, maintaining history the same
    way the old streaming path did: the user turn is pushed up front and
    rolled back if nothing came out; the assistant turn is appended once the
    generator is exhausted (which speak() does synchronously)."""
    _push_user(persona.build_user_turn(text, **context))
    parts = []
    for delta in llm_local.stream(_messages()):
        parts.append(delta)
        yield delta

    reply = "".join(parts).strip()
    if reply:
        _history.append({"role": "assistant", "content": reply})
        print(f"[brain] LLM: {reply}")
    else:
        _drop_user()


def _llm_stream_or_none(text, context):
    """Start a streaming reply and pull the first delta so failures surface
    before anything is spoken. Returns an iterator of deltas, or None to use
    the KB fallback."""
    if not llm_local.available() or not LLM_STREAMING:
        return None
    gen = _stream_llm(text, context)
    try:
        first = next(gen)
    except StopIteration:
        return None                      # empty reply / generation failed
    _mark_first_token()
    import itertools
    return itertools.chain([first], gen)


# ── Knowledge-base matching ───────────────────────────────────────────────────

def _best_local_match_keywords(text):
    """Word-overlap fallback used when the sentence transformer couldn't be
    loaded (no internet on a machine that never cached the model)."""
    words = set(re.findall(r"[a-z']+", text.lower()))
    best_score, best_index = 0.0, 0
    for i, q in enumerate(questions):
        q_words = set(re.findall(r"[a-z']+", q))
        if not q_words:
            continue
        overlap = len(words & q_words) / len(q_words)
        if overlap > best_score:
            best_score, best_index = overlap, i
    return best_score, best_index


_query_cache = {}          # text -> normalised embedding (tiny LRU)
_QUERY_CACHE_MAX = 32


def embed_query(text):
    """Normalised embedding of one utterance, memoised. The SAME text is
    embedded up to four times per question (addressing gate, KB match, top-K
    retrieval, website retrieval) — on the Pi each encode is tens of ms, so
    this cache is a straight latency win with no behaviour change."""
    if _st_model is None:
        return None
    vec = _query_cache.get(text)
    if vec is None:
        vec = embed([text])[0]
        if len(_query_cache) >= _QUERY_CACHE_MAX:
            _query_cache.pop(next(iter(_query_cache)))
        _query_cache[text] = vec
    return vec


def _kb_scores(text):
    """Similarity of `text` to every KB question, or None without embeddings."""
    if _st_model is None or question_embeddings is None:
        return None
    return question_embeddings @ embed_query(text)


def _best_local_match(text):
    """Return (best_score, best_index) of the closest knowledge-base question."""
    scores = _kb_scores(text)
    if scores is None:
        return _best_local_match_keywords(text)
    best_index = int(np.argmax(scores))
    return float(scores[best_index]), best_index


def _top_k_kb(text):
    """The LLM_KB_TOP_K most relevant KB entries as (question, answer, score),
    best first — the retrieval half of RAG-lite. Entries below LLM_KB_MIN_SIM
    are noise, not context, and are dropped."""
    scores = _kb_scores(text)
    if scores is None or not len(scores):
        return []
    order = np.argsort(scores)[::-1][:LLM_KB_TOP_K]
    return [(questions[i], answers[i], float(scores[i]))
            for i in order if scores[i] >= LLM_KB_MIN_SIM]


def _top_k_web(text):
    """Relevant website chunks as (source, text, score) from web_rag — empty
    when no site is configured, nothing is cached yet, or the module is
    missing. Never raises: a broken website index must not break answers."""
    if web_rag is None or not WEB_RAG_ENABLED or not WEB_RAG_URL:
        return []
    try:
        return web_rag.retrieve(text)
    except Exception as e:
        print(f"[brain] web_rag retrieve failed: {e}")
        return []


def _top_k_wiki(text):
    """The one relevant Wikipedia extract as (source, text, score) from
    wiki_rag — empty when disabled, the question looks campus-specific, the
    device is offline, or the module is missing. Never raises: a broken
    lookup must not break answers. Caller (_turn_context) only invokes this
    when the knowledge base found nothing at all — see wiki_rag.py."""
    if wiki_rag is None or not WIKI_RAG_ENABLED:
        return []
    try:
        return wiki_rag.retrieve(text)
    except Exception as e:
        print(f"[brain] wiki_rag retrieve failed: {e}")
        return []


def knowledge_match_score(text):
    """Semantic similarity of `text` to the closest knowledge-base entry (0..1).
    Used by the STT layer's meaning gate / addressing score."""
    if not text.strip():
        return 0.0
    return _best_local_match(text)[0]


def _local_fallback(text):
    best_score, best_index = _best_local_match(text)
    print(f"[brain] Local match score: {best_score:.2f}")
    if best_score > LOCAL_MATCH_THRESH:
        return answers[best_index]
    return "I am sorry, I do not have an answer for that."


# ── Spoken commands that move the body ────────────────────────────────────────
#
# At most ONE gesture is triggered per utterance, and only for an explicit
# request — the talking sway is the only motion during a normal answer.

_HANDSHAKE_PHRASES = (
    "shake hand", "shake hands", "shake my hand", "shake your hand",
    "handshake", "hand shake", "give me your hand", "give me a hand",
    "let's shake", "lets shake", "nice to meet you", "put your hand",
)

_GREETING_WORDS = {"hi", "hello", "hey", "bye", "goodbye"}


def _wants_handshake(lower, words):
    if any(p in lower for p in _HANDSHAKE_PHRASES):
        return True
    return "shake" in words and bool(words & {"hand", "hands"})


def _gesture_for(lower, words):
    """Return the single gesture this utterance asks for, or None."""
    if _wants_handshake(lower, words):
        return servo.handshake
    if words & _GREETING_WORDS:
        return servo.wave
    return None


def _mark_followup(reply):
    """If Luna's reply ended in a question she is now expecting an answer, so
    hold the conversation window open and let speech_to_text give the person
    the longer post-question timeout."""
    asked = bool(reply) and reply.rstrip().endswith("?")
    with state.lock:
        state.awaiting_answer = asked
        if asked:
            state.conversation_active = True
    if asked:
        print("[brain] Asked a follow-up — holding the conversation open")


# ── Vision questions ──────────────────────────────────────────────────────────

def _answer_vision_query(text):
    """Describe the live camera scene and answer through the LLM — or speak
    the description directly when the LLM isn't available (it's written to be
    speakable on its own)."""
    with state.lock:
        frame      = state.frame
        face_count = state.face_count
        person     = state.person_name
        emotion    = state.emotion
        face_x     = state.face_x

    objects = scene_context.detect_objects(frame)
    desc = scene_context.describe(face_count, person, emotion, face_x,
                                  objects=objects, frame_bgr=frame)
    print(f"[brain] Scene: {desc}")

    stream = _llm_stream_or_none(text, _turn_context(text, scene=desc,
                                                     with_kb=False))
    if stream is not None:
        speak(stream)
        reply = (_history[-1]["content"]
                 if _history and _history[-1]["role"] == "assistant" else "")
        _mark_followup(reply)
        _log_turn("vision", reply)
        return

    face.curious()
    _mark_first_token()
    _turn["style_arm"] = None                # no LLM → no style was applied
    speak(desc)
    _log_turn("vision", desc)


# ── Main process ──────────────────────────────────────────────────────────────

def process(text):
    text = text.strip()
    if not text:
        return

    # pick up any edits to knowledge.txt without needing a restart
    _maybe_reload_knowledge()

    print(f"[brain] Processing: {text}")
    _begin_turn(text)

    lower = text.lower()
    words = set(re.findall(r"[a-z']+", lower))

    # 1 — instant voice commands (volume, sleep, enrollment, forget, style…)
    if commands.handle(text):
        _log_turn("command")
        return

    # learning: this utterance is the reaction to Luna's previous answer
    if learning is not None:
        try:
            learning.on_utterance(text, person=_turn.get("person"))
        except Exception as e:               # noqa: BLE001
            print(f"[learn] on_utterance failed: {e}")

    # one gesture at most — the servo worker runs it to completion before the
    # talking sway is allowed to start, so the two never fight over an arm
    gesture = _gesture_for(lower, words)
    if gesture is not None:
        gesture()

    # compliments → heart-eyes face for a few seconds
    if ("love you" in lower or "good job" in lower or "well done" in lower
            or words & {"cute", "awesome", "amazing", "beautiful"}):
        face.love()

    # 2 — questions about what the camera sees
    if ENABLE_SCENE_CONTEXT and scene_context.is_vision_query(lower):
        _set_learning_context(text, 0.0)
        _answer_vision_query(text)
        return

    kb_score, kb_index = (_best_local_match(text) if answers else (0.0, 0))
    _set_learning_context(text, kb_score)

    # 3 — offline-first: answer straight from the knowledge base. A confident
    # match is instant and exactly what the KB author wrote; only questions
    # the KB can't confidently answer fall through to the LLM below.
    if ANSWER_OFFLINE_FIRST and answers:
        if kb_score >= LOCAL_ANSWER_THRESH:
            reply = answers[kb_index]
            print(f"[brain] Answered from knowledge base "
                  f"(match {kb_score:.2f}) — no LLM")
            _mark_followup(reply)
            _turn["kb_matches"] = [[questions[kb_index], round(kb_score, 3)]]
            _mark_first_token()
            speak(reply)
            _log_turn("kb", reply)
            return

    # 4 — the local LLM, with only the RELEVANT KB entries in the prompt.
    # Streaming: Luna starts speaking the first sentence while the model is
    # still writing the rest.
    context = _turn_context(text)
    stream  = _llm_stream_or_none(text, context)
    if stream is not None:
        print("[brain] Using local LLM (streaming)")
        speak(stream)
        reply = (_history[-1]["content"]
                 if _history and _history[-1]["role"] == "assistant" else "")
        _mark_followup(reply)
        _log_turn("llm", reply)
        return

    if not LLM_STREAMING and llm_local.available():
        _push_user(persona.build_user_turn(text, **context))
        reply = llm_local.ask(_messages())
        if reply:
            _history.append({"role": "assistant", "content": reply})
            print(f"[brain] LLM: {reply}")
            _mark_followup(reply)
            _mark_first_token()
            speak(reply)
            _log_turn("llm", reply)
            return
        _drop_user()

    # 5 — no LLM (model missing / still loading / failed): closest KB entry.
    print("[brain] LLM unavailable — using local knowledge base")
    reply = _local_fallback(text)
    _mark_followup(reply)
    if not _turn.get("kb_matches") and answers:
        _turn["kb_matches"] = [[questions[kb_index], round(kb_score, 3)]]
    _mark_first_token()
    speak(reply)
    _log_turn("fallback", reply)


# ── Background learning tick ──────────────────────────────────────────────────
# Once a second: tell learning who is in front of the camera (departures
# after a reply = "walked away"; recognised faces = visits). Every
# _DIARY_CHECK_SECS, while Luna is idle, let it write tonight's diary.
_LEARN_TICK_SECS  = 1.0
_DIARY_CHECK_SECS = 600.0


def _learning_loop():
    last_diary = time.time() - _DIARY_CHECK_SECS + 60.0   # first check ~1 min
    warned = False
    while True:
        time.sleep(_LEARN_TICK_SECS)
        try:
            now = time.time()
            with state.lock:
                faces  = state.face_count
                name   = state.person_name
                busy   = (state.speaking or state.tts_active
                          or state.conversation_active)
            learning.departure_tick(faces, name, now)
            if now - last_diary >= _DIARY_CHECK_SECS and not busy:
                last_diary = now
                if llm_local.available():
                    learning.maybe_write_diary(llm_local.ask, now)
        except Exception as e:               # noqa: BLE001
            if not warned:
                warned = True
                print(f"[learn] background tick failed: {e}")


if learning is not None and getattr(learning, "_cfg", {}).get("enabled"):
    threading.Thread(target=_learning_loop, daemon=True,
                     name="learning").start()
