# brain.py
"""
brain.py — how Luna decides what to say. Fully offline.

The answer pipeline, in priority order:

  1. Voice commands   — volume, sleep, face enrollment, "who am I"… handled
                        instantly by commands.py, no models involved.
  2. Website request  — "look into the website" re-asks the previous question
                        against the college website, fetching pages live.
  3. Vision questions — "what do you see?" builds a live scene description
                        (scene_context.py) and answers through the local LLM.
  4. KB direct match  — a question the knowledge base is confident about is
                        answered verbatim from it (ANSWER_OFFLINE_FIRST).
  5. Local LLM        — everything else goes to the on-device llama.cpp model
                        (llm_local.py), with the top-K most relevant KB entries
                        retrieved into the prompt instead of the whole file —
                        short prompts keep the Pi's prefill fast.
  6. KB fallback      — if the LLM is missing/broken, the closest KB entry.

The only optional network use is web_rag.py: when WEB_RAG_URL is set and the
Pi happens to be online it refreshes a cached copy of that website, and the
most relevant cached chunks join the knowledge facts in the prompt. A college
question the cache can't answer fetches the best-matching pages live, and
"check the website" re-asks the previous question that way. No keys, no
quotas, and it keeps working offline from the cache.
knowledge.txt still hot-reloads while Luna is running.
"""

import os
import datetime
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
import intents
import people
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
    WEB_RAG_LIVE_MIN_KB,
    WEB_RAG_LIVE_TIMEOUT,
    WEB_RAG_DEPARTMENT,
    WEB_RAG_RELATED,
    PERSON_CARDS,
)
import console

# Website retrieval (web_rag.py) is optional: it only does anything when
# WEB_RAG_URL is set, and only reaches the network when the Pi is online.
try:
    import web_rag
except Exception as _e:                      # missing module / import error
    print(f"[brain] web_rag unavailable ({_e}) — website retrieval off")
    web_rag = None

# ── Knowledge base (hot-reloadable) ───────────────────────────────────────────
# Loaded at startup AND re-read automatically whenever data/knowledge.txt changes
# on disk (see _maybe_reload_knowledge, called on every question), so edits take
# effect on a running Luna without a restart.
console.detail("[brain] Loading knowledge base...")

questions           = []
answers             = []
knowledge_text      = ""
question_embeddings = None
_knowledge_mtime    = None

# ── Sentence transformer — retrieval + offline matching ──────────────────────
# Loaded from the local cache first (HF_HUB_OFFLINE forces huggingface_hub to
# skip its network reachability check entirely) so a disconnected Pi never
# depends on internet just to boot. Only reaches for the network if the model
# isn't cached yet, and if that also fails (never downloaded + no internet)
# Luna degrades to plain keyword matching instead of crashing.
console.detail("[brain] Loading sentence transformer...")
_st_model = None
try:
    os.environ["HF_HUB_OFFLINE"]      = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    _st_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
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
console.detail(f"[brain] {len(questions)} knowledge entries loaded")

# Website index: loads its cache (if any) and, when online, refreshes it in a
# background thread. Shares this module's embedder — nothing loads twice.
if web_rag is not None and WEB_RAG_ENABLED and WEB_RAG_URL:
    try:
        web_rag.start(embed)
    except Exception as _e:
        print(f"[brain] web_rag failed to start: {_e}")

# ── The prompt ────────────────────────────────────────────────────────────────
# Persona + scope rules live in persona.py (shared with the Ollama Modelfile
# and the benchmark). The system prompt is STATIC per knowledge.txt version;
# everything that changes per turn — retrieved facts, website extracts, the
# live scene, who is talking — travels in the user message, so the Pi only
# prefills the new tokens each question (Ollama caches the static prefix).

def _turn_context(text, scene=None, with_kb=True, live_web=False,
                  web_snippets=None):
    """Per-turn context dict for persona.build_user_turn().

    Scene memory: on a vision question it follows the live description in
    the `scene` slot, moods and objects included — the person asked what
    Luna sees. On an ordinary turn it goes in the `background` slot, people
    only (who is here, who came by, what Luna just asked): the camera's mood
    guesses and desk objects are exactly what the model used to bring up out
    of nowhere. live_web lets the website lookup fetch pages live;
    web_snippets, when given, are used as they are (the website path already
    looked them up)."""
    with state.lock:
        person = state.person_name
    memory = ""
    if ENABLE_SCENE_MEMORY and SCENE_MEMORY_IN_PROMPT:
        try:                                 # deferred + guarded: remembering
            import scene_memory              # must never break a voice turn
            if scene:                        # "what do you see?"
                memory = scene_memory.summary() or ""
            elif (intents.needs_scene_context(text)
                  or scene_memory.asked_recently()):
                # Only when the question is about who is / was around, or is
                # the reply to a question Luna just asked — otherwise the
                # model gets no scene notes to wander off into.
                memory = scene_memory.summary(moods=False,
                                              objects=False) or ""
                if not memory and intents.needs_scene_context(text):
                    # Say so: with nothing at all the model invents people
                    # ("a student studying, a professor typing…").
                    memory = ("You have not seen anyone in front of you "
                              "recently. Do not describe or invent people.")
        except Exception as e:               # noqa: BLE001
            print(f"[brain] scene memory unavailable: {e}")
    background = None
    if memory:
        if scene:
            scene = f"{scene} Recently: {memory}"
        else:
            background = memory
    kb_snippets = _top_k_kb(text) if with_kb else ()
    if web_snippets is None:
        web_snippets = _top_k_web(text, live=live_web)
    # the best knowledge-base answer first: "who is the HOD?" names the
    # person only in the answer, and that is who the second hop follows
    top_kb = [h for h in kb_snippets[:1] if h[2] >= LOCAL_ANSWER_THRESH]
    web_related = _related_web(text, top_kb + list(web_snippets))
    _turn["kb_matches"] = [[q, round(float(s), 3)] for q, _, s in kb_snippets]
    _turn["web_matches"] = [[src, round(float(s), 3)]
                            for src, _, s in web_snippets]
    return dict(
        kb_snippets=kb_snippets,
        web_snippets=web_snippets,
        web_related=web_related,
        scene=scene,
        background=background,
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
                 kb_matches=[], web_matches=[], style_arm=None, context=None,
                 ttft=None)


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
        # "You have met Michael on 4 visits" in every prompt made the model
        # open answers with "glad you're here again!" — only for questions
        # about the person themself / who is around
        prof = (learning.profile_context(person)
                if person and intents.needs_scene_context(text) else None)
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
            web_matches=_turn.get("web_matches") or [], reply=reply,
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


class _StageFilter:
    """Drops *stage directions* ("*waves little arms*") from a streamed
    reply: the small model writes them despite the persona, and TTS would
    read them out. Works across delta boundaries; a lone "*" that isn't
    closed within STAGE_MAX chars is let through as plain text."""
    STAGE_MAX = 80

    def __init__(self):
        self.buf = None                      # text after an opening "*"

    def __call__(self, delta):
        out = []
        for ch in delta:
            if self.buf is None:
                if ch == "*":
                    self.buf = []
                else:
                    out.append(ch)
            elif ch == "*":
                self.buf = None              # closed: drop the direction
            else:
                self.buf.append(ch)
                if len(self.buf) > self.STAGE_MAX:   # not a direction
                    out.extend(self.buf)
                    self.buf = None
        return "".join(out)


_STAGE_RE = re.compile(r"\*[^*]{0,80}\*")


def _stream_llm(text, context):
    """Yield reply deltas from the local model, maintaining history the same
    way the old streaming path did: the user turn is pushed up front and
    rolled back if nothing came out; the assistant turn is appended once the
    generator is exhausted (which speak() does synchronously)."""
    _push_user(persona.build_user_turn(text, **context))
    parts = []
    strip = _StageFilter()
    for delta in llm_local.stream(_messages()):
        delta = strip(delta)
        if not delta:
            continue
        parts.append(delta)
        yield delta

    reply = "".join(parts).strip()
    if reply:
        _history.append({"role": "assistant", "content": reply})
        console.detail(f"[brain] LLM: {reply}")
    else:
        _drop_user()


_BOOT_T = time.time()
LLM_BOOT_WAIT = 120.0   # s after start-up in which a question waits for the
                        # model to finish loading instead of falling back


def _wait_for_llm(max_wait=15.0):
    """Right after boot the model may still be loading (main.py warms it in
    the background). A visitor's first "hello" should wait a few seconds
    for it, not get "I am sorry, I do not have an answer for that"."""
    if llm_local.available() or time.time() - _BOOT_T > LLM_BOOT_WAIT:
        return
    print("[brain] LLM still loading — waiting for it")
    t0 = time.time()
    while time.time() - t0 < max_wait and not llm_local.available():
        time.sleep(0.25)


def _llm_stream_or_none(text, context):
    """Start a streaming reply and pull the first delta so failures surface
    before anything is spoken. Returns an iterator of deltas, or None to use
    the KB fallback."""
    _wait_for_llm()
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


def _kb_text(text):
    """`text` with college abbreviations spelled out ("hod of cse" -> "... head
    of department HOD ... computer science and engineering") — the embedder
    barely links "hod" to "head of department" (0.43 vs 1.00)."""
    try:
        return web_rag.expand_query(text) if web_rag is not None else text
    except Exception:                        # noqa: BLE001
        return text


def _kb_scores(text):
    """Similarity of `text` to every KB question, or None without embeddings."""
    if _st_model is None or question_embeddings is None:
        return None
    return question_embeddings @ embed_query(_kb_text(text))


_STOP = set("""a an the is are was were be been am do does did of in on at to
for from by with and or what who whom whose which where when why how can could
would will shall should may might me my you your i we our they their it its
this that these those there here tell about please any all some much many
college tkm tkmce engineering kollam department dept""".split())


def _stem(w):
    return w[:5]


def _kb_direct_ok(text, index, score):
    """May the KB answer `text` verbatim? Clear matches (>= KB_SURE) always;
    borderline ones only if every content word of the question appears in
    that KB entry — "who are the TEACHERS in computer science" scores 0.72
    against "who is the head of computer science department" (and would be
    answered with the HOD's name), while "where is the canteen" scores 0.72
    against "where is the college canteen and store" and is right."""
    if score >= KB_SURE:
        return True
    entry = set(_stem(w) for w in re.findall(r"[a-z]+",
                (questions[index] + " " + answers[index]).lower()))
    expansions = getattr(web_rag, "_SLUG_EXPANSIONS", {}) if web_rag else {}

    def covered(w):
        if _stem(w) in entry:
            return True
        full = [x for x in re.findall(r"[a-z]+", expansions.get(w, "").lower())
                if x not in _STOP and x != w]      # "hod" -> head (department)
        return bool(full) and all(_stem(x) in entry for x in full)

    words = [w for w in re.findall(r"[a-z]+", text.lower())
             if w not in _STOP and len(w) > 2]
    return all(covered(w) for w in words)


KB_SURE = 0.85


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
    hits = [(questions[i], answers[i], float(scores[i]))
            for i in order if scores[i] >= LLM_KB_MIN_SIM]
    if web_rag is not None and web_rag._names_other_department(text):
        # "who is the HOD of mechanical": facts about Luna's own department
        # ("head of computer science: Dr. A") only make the model mix them up
        own = re.compile(r"\b(cse|computer science)\b")
        hits = [h for h in hits if not own.search(f"{h[0]} {h[1]}".lower())]
    return hits


def _web_enabled():
    return web_rag is not None and WEB_RAG_ENABLED and bool(WEB_RAG_URL)


_THE_DEPARTMENT = re.compile(r"\b(the|your|this|our|my) (department|dept)\b")


def _web_query(text):
    """The website search text: "the department" means Luna's own one, or
    every department's staff list would match equally well."""
    if WEB_RAG_DEPARTMENT and _THE_DEPARTMENT.search(text.lower()):
        return f"{text} ({WEB_RAG_DEPARTMENT})"
    return text


def _top_k_web(text, live=False, force=False):
    """Relevant website chunks as (source, text, score) from web_rag — empty
    when no site is configured, nothing is cached yet, or the module is
    missing. live=True may fetch matching pages from the site right now
    (web_rag.lookup); force=True always tries. Never raises: a broken
    website index must not break answers."""
    if not _web_enabled():
        return []
    try:
        query = _web_query(text)
        if live or force:
            return web_rag.lookup(query, force=force)
        return web_rag.retrieve(query)
    except Exception as e:
        print(f"[brain] web_rag retrieve failed: {e}")
        return []


def _related_web(text, hits):
    """web_rag's second hop: what else the site says about the people or
    things in the question or its best answer — a faculty member's latest
    award alongside their designation. [] when the website or the hop is
    off. `hits` are (source, text, score); hits[0] is what the hop follows
    when the question names nobody."""
    if not WEB_RAG_RELATED or not _web_enabled():
        return []
    try:
        return web_rag.related(text, hits)
    except Exception as e:                   # noqa: BLE001
        print(f"[brain] related website search failed: {e}")
        return []


def knowledge_match_score(text):
    """Semantic similarity of `text` to the closest knowledge-base entry (0..1).
    Used by the STT layer's meaning gate / addressing score."""
    if not text.strip():
        return 0.0
    return _best_local_match(text)[0]


def _local_fallback(text):
    best_score, best_index = _best_local_match(text)
    console.detail(f"[brain] Local match score: {best_score:.2f}")
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
        console.detail("[brain] Asked a follow-up — holding the conversation open")


# ── Vision questions ──────────────────────────────────────────────────────────

def _show_source(context=None, kind=None):
    """The clean-mode [source] line: the website pages this answer was
    built from, and the closest knowledge-base entry if it really matches.
    Only the closest: the looser ones that ride along in the prompt score
    deceptively high ("what is the name of the college" for any question
    ending in "of the college") and would bury the one that matters."""
    context = context or {}
    kb = [(sc, q) for q, _, sc in context.get("kb_snippets") or ()
          if sc >= LOCAL_MATCH_THRESH]
    web = [u for u, _, _ in (list(context.get("web_snippets") or ())
                             + list(context.get("web_related") or ()))]
    console.source(web=web, kb=[max(kb)[1]] if kb else (), kind=kind)


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
    console.detail(f"[brain] Scene: {desc}")

    context = _turn_context(text, scene=desc, with_kb=False)
    stream = _llm_stream_or_none(text, context)
    _show_source(context, kind="camera")
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


# ── "Check the website" ───────────────────────────────────────────────────────

_last_question = ""        # the last question Luna tried to answer


def _last_reply():
    return (_history[-1]["content"]
            if _history and _history[-1]["role"] == "assistant" else "")


def _answer_date_time(when):
    """The date or time from the system clock, said aloud and shown as a
    calendar / clock card. A clock that was never set — a Pi offline with no
    RTC battery boots in 1970 — is admitted to rather than read out."""
    now = datetime.datetime.now()
    if not intents.clock_is_set(now):
        console.source(kind="clock (not set)")
        _say("My clock isn't set right now, so I can't tell you that.",
             path="clock")
        return
    if when == "time":
        face.time_card()
        reply = intents.say_time(now)
    else:
        face.date_card()
        reply = intents.say_date(now)
        if when == "both":
            reply += " " + intents.say_time(now)
    console.source(kind="clock")
    _say(reply, path="clock")


def _staff_sections():
    """The website's staff lists for people.directory(), [] when off."""
    if not _web_enabled():
        return []
    try:
        return web_rag.staff_sections()
    except Exception:                        # noqa: BLE001
        return []


_TITLE_WORDS = people.TITLES


def _strip_subject(sentence, person):
    """'Prof. Reshma Sheik has been awarded …' -> 'Has been awarded …' —
    the card already shows the name."""
    words = sentence.split()
    i = 0
    while i < len(words):
        w = words[i].strip(".,").lower()
        if w in _TITLE_WORDS or w in person.tokens or len(w) == 1:
            i += 1
            continue
        break
    rest = " ".join(words[i:]).strip(" .") if i < len(words) else ""
    return (rest[:1].upper() + rest[1:]) if rest else sentence.strip(" .")


def _person_highlight(person, max_chars=150):
    """One line under the name: the newest dated thing the college website
    says about them (an award, a project — a sentence, never a table row)
    when it is from the last three years, else the knowledge base's second
    fact (their PhD), else nothing."""
    if _web_enabled() and WEB_RAG_RELATED:
        try:
            year = datetime.date.today().year
            for _, _, sent in web_rag.related_sentences(person.name):
                years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", sent)]
                if (not years or max(years) < year - 3
                        or not _reads_like_a_sentence(sent)):
                    continue
                line = _strip_subject(sent, person)
                if len(line) > max_chars:
                    line = line[:max_chars].rsplit(" ", 1)[0] + "…"
                return line
        except Exception as e:               # noqa: BLE001
            console.detail(f"[brain] highlight lookup failed: {e}")
    return person.detail


def _reads_like_a_sentence(text):
    """A line fit for a card: prose, not a data row — no table pipes, no
    ';'-separated records, no 'Key: value' fields, not overlong."""
    return ("|" not in text and ";" not in text and len(text.split()) <= 40
            and len(re.findall(r"\b[\w.]+:\s", text)) < 2)


def _maybe_person_card(text, kb_score, kb_index):
    """Put up a person card when — and only when — Luna is certain who the
    question is about: it names exactly one known person ("who is reshma
    sheik"), or it confidently matches a knowledge-base entry about one
    person ("who is the principal"). Every word on the card is a stored
    fact (people.py), never the model's. Returns the Person or None."""
    if not PERSON_CARDS:
        return None
    try:
        d = people.directory(_staff_sections())
        person = d.match_question(text)
        if (person is None and answers and kb_score >= LOCAL_ANSWER_THRESH
                and _kb_direct_ok(text, kb_index, kb_score)):
            person = d.for_kb_question(questions[kb_index], text)
        if person is None:
            return None
        face.person_card(person.name, person.role, _person_highlight(person))
        console.detail(f"[brain] person card: {person.name}")
        return person
    except Exception as e:                   # noqa: BLE001
        print(f"[brain] person card failed: {e}")
        return None


def _say(reply, path="fallback"):
    """Speak a fixed sentence (no LLM) and log the turn."""
    _mark_followup(reply)
    _mark_first_token()
    _turn["style_arm"] = None
    speak(reply)
    _log_turn(path, reply)


# An LLM reply that admits it doesn't know. For a college question that is
# the cue to check the website before giving up (process(), step 5).
_UNSURE_RE = re.compile(
    r"\b(don'?t|do not|doesn'?t) (have|know|mention|say)\b"
    r"|\bnot (sure|certain|mentioned|explicitly|available|listed|specified)\b"
    r"|\bno (information|details|info)\b"
    r"|\bcouldn'?t find\b|\bunable to\b|\bdepartment office\b"
    r"|\bnot in (my|the) (facts|knowledge)\b")


def _unsure(reply):
    return bool(reply) and bool(_UNSURE_RE.search(reply.lower()))


_seen_web = []     # website extracts the unsure answer already had


def _fresh_answer(hits):
    """After an unsure answer: did the website check turn up anything the
    LLM hadn't already seen? Re-asking with the same extracts would only
    produce the same "I don't know" again."""
    return any(u not in _seen_web for u, _, _ in hits)


def _ask_web_only(question, hits):
    """A second, non-streamed answer from the website extracts only — no
    knowledge-base facts, no earlier turns. "" if the model is still unsure
    (then nothing is added to the conversation history)."""
    if not llm_local.available():
        return ""
    user = persona.build_user_turn(question, web_snippets=hits)
    try:
        reply = llm_local.ask([{"role": "system", "content": _system},
                               {"role": "user", "content": user}]) or ""
    except Exception as e:                   # noqa: BLE001
        print(f"[brain] web-only retry failed: {e}")
        return ""
    reply = _STAGE_RE.sub("", reply).strip()
    if not reply or _unsure(reply):
        return ""
    console.detail(f"[brain] LLM (website only): {reply}")
    _history.append({"role": "user", "content": question})
    _history.append({"role": "assistant", "content": reply})
    return reply


def _answer_from_website(topic, auto=False):
    """Answer `topic` — or, for a bare "look into the website", the previous
    question — from the college website, fetching matching pages live. When
    the website has nothing, Luna says so plainly instead of letting the LLM
    talk around a near-miss knowledge-base fact. auto=True: Luna's own
    follow-up after an unsure LLM answer (she has already said she doesn't
    know, so a miss is a short "couldn't find it there either")."""
    question = topic or _last_question
    if not question:
        _say("Sure! What should I look up on the college website?")
        return
    if not _web_enabled():
        _say("I don't have the college website set up, so I can only answer "
             "from what I already know.")
        return
    console.detail(f"[brain] Checking the website for: {question}")
    face.curious()
    result = {}
    worker = threading.Thread(
        target=lambda: result.update(hits=_top_k_web(question, force=True)),
        daemon=True, name="web_lookup")
    worker.start()
    _mark_first_token()
    speak("Let me check the college website.")    # while the pages download
    worker.join(timeout=WEB_RAG_LIVE_TIMEOUT + 10.0)
    hits = result.get("hits") or []
    if auto and not _fresh_answer(hits):
        # Nothing new on the site — but the model may have fumbled the right
        # extract among knowledge-base facts and history: one more, clean
        # try with the website extracts alone. Spoken only if it's sure.
        reply = _ask_web_only(question, hits) if hits else ""
        if reply:
            console.source(web=[u for u, _, _ in hits])
            _mark_followup(reply)
            _mark_first_token()
            speak(reply)
            _log_turn("llm", reply)
        else:
            _say("I checked the college website too, but couldn't find it there.")
        return
    if not hits:
        if web_rag.busy():
            _say("I'm still reading the college website for the first time. "
                 "Ask me again in a minute!")
        elif not web_rag.is_online():
            _say("I can't reach the college website right now, so I can't "
                 "check that.")
        else:
            _say("I checked the college website but couldn't find that. "
                 "The department office would know.")
        return
    context = _turn_context(question, web_snippets=hits)
    stream = _llm_stream_or_none(question, context)
    if stream is not None:
        _show_source(context)
        speak(stream)
        reply = _last_reply()
        _mark_followup(reply)
        _log_turn("llm", reply)
        return
    best = " ".join(hits[0][1].split()[:45])       # no LLM: read the extract
    console.source(web=[hits[0][0]])
    _say(f"Here's what the college website says. {best}")


# ── Main process ──────────────────────────────────────────────────────────────

def process(text):
    global _last_question
    text = text.strip()
    if not text:
        return

    # pick up any edits to knowledge.txt without needing a restart
    _maybe_reload_knowledge()

    console.detail(f"[brain] Processing: {text}")
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

    # 1b — the date / the time: from the clock, shown as a card
    when = intents.date_time(lower)
    if when:
        _answer_date_time(when)
        return

    # 2 — "look into the website", "check online for the fees", or a yes to
    # Luna's own offer to check: answer from the website, fetching live
    if web_rag is not None:
        is_web, topic = web_rag.website_request(text, _last_reply())
        if is_web:
            _set_learning_context(topic or _last_question or text, 0.0)
            _answer_from_website(topic)
            return

    # 3 — questions about what the camera sees
    if ENABLE_SCENE_CONTEXT and scene_context.is_vision_query(lower):
        _set_learning_context(text, 0.0)
        _answer_vision_query(text)
        return

    _last_question = text
    kb_score, kb_index = (_best_local_match(text) if answers else (0.0, 0))
    _set_learning_context(text, kb_score)

    # a person Luna is certain about: their card goes up while she answers
    _maybe_person_card(text, kb_score, kb_index)

    # 4 — offline-first: answer straight from the knowledge base. A confident
    # match is instant and exactly what the KB author wrote; only questions
    # the KB can't confidently answer fall through to the LLM below.
    if ANSWER_OFFLINE_FIRST and answers:
        reply = answers[kb_index]
        direct = (kb_score >= LOCAL_ANSWER_THRESH
                  and _kb_direct_ok(text, kb_index, kb_score))
        if direct and llm_local.available() and _related_web(
                text, [(questions[kb_index], reply, kb_score)]):
            # The knowledge base has the answer and the website has more
            # about who / what it names (a recent award): let the model say
            # both, below, instead of the bare entry.
            console.detail("[brain] KB answer has related website facts — "
                           "answering through the LLM")
            direct = False
        if direct:
            console.source(kb=[questions[kb_index]])
            console.detail(f"[brain] Answered from knowledge base "
                  f"(match {kb_score:.2f}) — no LLM")
            _mark_followup(reply)
            _turn["kb_matches"] = [[questions[kb_index], round(kb_score, 3)]]
            _mark_first_token()
            speak(reply)
            _log_turn("kb", reply)
            return

    # 5 — the local LLM, with only the RELEVANT KB entries (and website
    # extracts) in the prompt. A question about the college that the cached
    # website can't answer fetches matching pages live first. Streaming:
    # Luna starts speaking the first sentence while the model is still
    # writing the rest.
    # a question about who is / was around is never a website question
    college = (kb_score >= WEB_RAG_LIVE_MIN_KB
               and not intents.needs_scene_context(text))
    context = _turn_context(text, live_web=college)
    stream  = _llm_stream_or_none(text, context)
    if stream is not None:
        console.detail("[brain] Using local LLM (streaming)")
        _show_source(context)
        speak(stream)
        reply = (_history[-1]["content"]
                 if _history and _history[-1]["role"] == "assistant" else "")
        # Low confidence on a college question: check the website (live,
        # forced) before leaving the visitor with "I don't know".
        if college and _unsure(reply) and _web_enabled():
            print("[brain] Unsure answer — checking the college website")
            _seen_web[:] = [u for u, _, _ in context.get("web_snippets", ())]
            _log_turn("llm", reply)
            _begin_turn(text)
            _answer_from_website(text, auto=True)
            return
        _mark_followup(reply)
        _log_turn("llm", reply)
        return

    if not LLM_STREAMING and llm_local.available():
        _push_user(persona.build_user_turn(text, **context))
        reply = _STAGE_RE.sub("", llm_local.ask(_messages()) or "").strip()
        if reply:
            _history.append({"role": "assistant", "content": reply})
            console.detail(f"[brain] LLM: {reply}")
            _show_source(context)
            _mark_followup(reply)
            _mark_first_token()
            speak(reply)
            _log_turn("llm", reply)
            return
        _drop_user()

    # 6 — no LLM (model missing / still loading / failed): closest KB entry.
    console.detail("[brain] LLM unavailable — using local knowledge base")
    console.source(kind="knowledge base (LLM unavailable)")
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
