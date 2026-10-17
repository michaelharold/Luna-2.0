# brain.py
"""
brain.py — how Luna decides what to say. Fully offline.

The answer pipeline, in order (docs/Fetch Algo.md has the whole flow):

  1. Voice commands   — volume, sleep, registration, feedback ... handled at
                        once by commands.py, no models involved.
  2. Date / time      — from the clock, shown as a calendar / clock card.
  3. Vision questions — "what do you see?": a live scene description,
                        answered through the LLM.
  4. Knowledge base   — a question data/knowledge.txt matches very closely is
                        answered verbatim from it, instantly, no LLM.
  5. Data Fetch       — fetch.py's two LLM calls:
                          ROUTE   the model sees the LIBRARY (every page file
                                  of the college website : one line) and the
                                  query, and says PAGES: a, b / REPLY: ... /
                                  AGAIN / IGNORE
                          ANSWER  for PAGES, the model answers from those
                                  pages' data
                        REPLY is spoken as it streams (small talk, general
                        questions), AGAIN asks the visitor to repeat, IGNORE
                        stays silent (noise).
  6. KB fallback      — the LLM is missing / broken: the closest KB entry.

The system prompt (persona + LIBRARY + rules) is the same on every LLM call,
so Ollama keeps it cached and the Pi only reads each question's new tokens.
The website is never fetched here: site_index.py builds the page files once a
week. knowledge.txt still hot-reloads while Luna is running.
"""

import datetime
import os
import random
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
import fetch
from shared_state import state
from config import (
    KNOWLEDGE_PATH,
    LLM_MAX_HISTORY,
    LLM_STREAMING,
    ANSWER_OFFLINE_FIRST,
    LOCAL_ANSWER_THRESH,
    LOCAL_MATCH_THRESH,
    ENABLE_SCENE_CONTEXT,
    ENABLE_SCENE_MEMORY,
    SCENE_MEMORY_IN_PROMPT,
    FETCH_ENABLED,
    FETCH_MAX_PAGES,
    FETCH_KB_PAGE,
    FETCH_ANSWER_MAX_TOKENS,
    FETCH_PRIME_AT_BOOT,
    FETCH_REPEAT_REPLIES,
    FETCH_DIRECT,
    FETCH_ROUTE_MODEL,
    FETCH_CACHE,
    FETCH_CACHE_DAYS,
    FETCH_CACHE_SIM,
    FETCH_CACHE_PATH,
    FETCH_COLLEGE_WORDS,
    FETCH_MODE,
    FETCH_SECTION_WORDS,
    PERSON_CARDS,
    LLM_FILLER_AFTER,
    LLM_FILLERS,
    STT_UNCLEAR_CONF,
    STT_ONE_WORD_CONF,
)
import console

# ── Knowledge base (hot-reloadable) ───────────────────────────────────────────
# Loaded at startup AND re-read whenever data/knowledge.txt changes on disk
# (see _maybe_reload_knowledge, called on every question).
console.detail("[brain] Loading knowledge base...")

questions           = []
answers             = []
faq_page            = []     # per entry: the page file of an indexing FAQ
question_embeddings = None   # entry, None for a knowledge.txt line
knowledge_text      = ""     # knowledge.txt as it is (digest mode: in the
                             # system prompt)
_knowledge_mtime    = None
_faq_key            = None   # the catalog the FAQ entries came from

# ── Sentence transformer — knowledge-base matching ───────────────────────────
# Loaded straight from the cached snapshot folder first, so booting never
# touches the network (setting HF_HUB_OFFLINE here would be too late —
# huggingface_hub reads it once at import). Only reaches for the network if
# the model isn't cached yet; if that also fails Luna degrades to plain
# keyword matching instead of crashing.
console.detail("[brain] Loading sentence transformer...")
_st_model = None
try:
    from huggingface_hub import snapshot_download
    _st_model = SentenceTransformer(snapshot_download(
        "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True))
except Exception:
    try:
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"[brain] Could not load sentence transformer ({e}) — "
              f"falling back to plain keyword matching for the knowledge base.")


def embed(texts):
    """L2-normalised sentence embeddings (rows) for a list of strings, or
    None when the sentence transformer isn't available. Shared with
    learning.py (exemplars, repeats) — one model copy in RAM."""
    if _st_model is None:
        return None
    raw   = _st_model.encode(list(texts), batch_size=32, show_progress_bar=False)
    raw   = np.asarray(raw, dtype=np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / np.maximum(norms, 1e-9)


# ── Conversation log + learning (optional — never break a turn) ─────────────
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


def _load_knowledge():
    """(Re)read knowledge.txt — plus the FAQ the weekly indexing wrote for
    each website page file — and rebuild everything derived from them. A
    close match to either is answered at once, with no LLM."""
    global questions, answers, faq_page, question_embeddings
    global _knowledge_mtime, _faq_key, knowledge_text
    qs, ans, src = [], [], []
    with open(KNOWLEDGE_PATH, "r") as f:
        raw = f.read()
    knowledge_text = raw
    for line in raw.splitlines():
        if ":" not in line:
            continue
        q, a = line.split(":", 1)
        qs.append(q.strip().lower())
        ans.append(a.strip())
        src.append(None)
    for q, a, page in fetch.library.faq() if FETCH_ENABLED else ():
        qs.append(q.strip().lower())
        ans.append(a.strip())
        src.append(page)
    _faq_key = fetch.library.catalog.get("built_at")
    questions, answers, faq_page = qs, ans, src
    question_embeddings = (embed(questions)
                           if _st_model is not None and questions else None)
    try:
        _knowledge_mtime = os.path.getmtime(KNOWLEDGE_PATH)
    except OSError:
        _knowledge_mtime = None


def _maybe_reload_knowledge():
    """Re-read knowledge.txt (and the FAQ) if either changed since the
    last load."""
    try:
        m = os.path.getmtime(KNOWLEDGE_PATH)
    except OSError:
        return
    fetch.library.names()                    # re-reads a changed catalog
    if m != _knowledge_mtime or fetch.library.catalog.get("built_at") != _faq_key:
        print("[brain] knowledge.txt changed — reloading")
        _load_knowledge()
        print(f"[brain] reloaded {len(questions)} knowledge entries")


_load_knowledge()
console.detail(f"[brain] {len(questions)} knowledge entries loaded")


# ── The system prompt ─────────────────────────────────────────────────────────
# Persona + LIBRARY + FETCH_SYSTEM_RULES (fetch.system_prompt). Rebuilt only
# when the catalog changes (a weekly index deployed), so it is byte-identical
# from call to call and Ollama's prefix cache keeps it.
_system_cache = {"key": None, "text": persona.system_prompt()}


def _digest_mode():
    return FETCH_ENABLED and FETCH_MODE == "digest" and bool(
        fetch.library.digest())


def _system():
    """The system prompt — the same on every call until the index (or, in
    digest mode, knowledge.txt) changes. Digest mode: persona + the whole
    knowledge base + the digest of every page. Route mode: persona + the
    LIBRARY + the route / answer rules."""
    fetch.library.names()                    # re-reads a changed catalog
    digest = _digest_mode()
    key = (fetch.library.catalog.get("built_at"), digest,
           fetch.library.digest() if digest else None,
           knowledge_text if digest else None) if FETCH_ENABLED else None
    if key != _system_cache["key"]:
        _system_cache["key"] = key
        if digest:
            text = fetch.digest_system_prompt(
                persona.system_prompt(knowledge_text))
        elif FETCH_ENABLED:
            text = fetch.system_prompt(persona.system_prompt())
        else:
            text = persona.system_prompt()
        _system_cache["text"] = text
        console.detail(f"[brain] system prompt: {len(text.split())} words "
                       f"({'digest' if digest else 'route'} mode)")
    return _system_cache["text"]


def _library_pages():
    return fetch.library.names() if FETCH_ENABLED else []


# ── Per-turn notes: who is asking, scene memory, learning ────────────────────

def _scene_notes(text, vision=False):
    """Scene memory for this turn. A camera question gets moods and objects
    too; an ordinary one only who is / was around, and only when it is
    about that or answers a question Luna just asked — otherwise the model
    has no scene notes to wander off into."""
    if not (ENABLE_SCENE_MEMORY and SCENE_MEMORY_IN_PROMPT):
        return None
    try:                                     # never break a voice turn
        import scene_memory
        if vision:
            return scene_memory.summary() or None
        if intents.needs_scene_context(text) or scene_memory.asked_recently():
            memory = scene_memory.summary(moods=False, objects=False) or ""
            if not memory and intents.needs_scene_context(text):
                # Say so: with nothing at all the model invents people.
                memory = ("You have not seen anyone in front of you "
                          "recently. Do not describe or invent people.")
            return memory or None
    except Exception as e:                   # noqa: BLE001
        print(f"[brain] scene memory unavailable: {e}")
    return None


def _speech_unsure(text):
    """Did speech recognition doubt these words? A lone word needs
    STT_ONE_WORD_CONF, a sentence STT_UNCLEAR_CONF. None (typed text) is
    never unsure."""
    with state.lock:
        conf = state.utterance_conf
    if conf is None:
        return False
    words = re.findall(r"[a-z']+", text.lower())
    return conf < (STT_ONE_WORD_CONF if len(words) == 1 else STT_UNCLEAR_CONF)


def _notes(text, vision=False):
    """The notes fetch.route_message / answer_message take: who is asking,
    scene, learning style / profile / diary (exemplars for the answer)."""
    with state.lock:
        person = state.person_name
    extra = _learning_context(text, person)
    return dict(person=person, greeted=_greeted(person),
                scene=None if vision else _scene_notes(text),
                style=extra.get("style"), profile=extra.get("profile"),
                diary=extra.get("diary")), extra.get("exemplars") or ()


GREETED_FOR_SECS = 600.0   # Luna's own "Oh, hi Michael!" counts this long


def _greeted(person):
    """Has this conversation already had its hello — an earlier turn, or
    Luna's own proactive greeting a few minutes ago?"""
    if _history:
        return True
    try:
        import proactive
        at = proactive._last_greet.get(person) if person else None
        return bool(at) and time.time() - at < GREETED_FOR_SECS
    except Exception:                        # noqa: BLE001
        return False


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
                 ttft=None, filler=False)


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
    """Style / exemplars / profile / diary — {} when learning is off."""
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


# ── Conversation history (the "context" of the route and answer calls) ───────
# Plain questions and Luna's replies only: the page data a turn carried was
# for THAT answer, and re-sending it every later turn made prompts slow.
_history = []


def _remember(question, reply):
    _history.extend(({"role": "user", "content": question},
                     {"role": "assistant", "content": reply}))
    while len(_history) > LLM_MAX_HISTORY:
        _history.pop(0)
    if _history and _history[0]["role"] == "assistant":
        _history.pop(0)                      # keep user / assistant pairs


def reset_history():
    """Clear the LLM conversation history. Called when a conversation window
    ends (see speech_to_text._expire_conversation) so one visitor's turns
    never carry over into the next visitor's session."""
    _history.clear()


def _last_reply():
    return (_history[-1]["content"]
            if _history and _history[-1]["role"] == "assistant" else "")


def _messages(user_msg):
    return [{"role": "system", "content": _system()}, *_history,
            {"role": "user", "content": user_msg}]


# ── Streaming helpers ─────────────────────────────────────────────────────────

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


def _recorded(question, deltas):
    """Pass a reply stream on (stage directions dropped) and, once it has
    been spoken in full, add the turn to the history."""
    parts = []
    strip = _StageFilter()
    for delta in deltas:
        delta = strip(delta)
        if not delta:
            continue
        _mark_first_token()
        parts.append(delta)
        yield delta
    reply = "".join(parts).strip()
    if reply:
        _remember(question, reply)
        console.detail(f"[brain] LLM: {reply}")


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


def _with_filler(work):
    """Run work() in a thread; if it takes longer than LLM_FILLER_AFTER,
    Luna says a short "Hmm, let me think." meanwhile (once per turn) — a
    silent pause of several seconds made people repeat themselves, and the
    answer then landed out of the blue. Returns work()'s result (re-raises
    its exception)."""
    box = {}

    def run():
        try:
            box["value"] = work()
        except Exception as e:               # noqa: BLE001 — re-raised below
            box["error"] = e

    t = threading.Thread(target=run, daemon=True, name="llm_work")
    with state.lock:
        state.thinking = True                # vision slows down meanwhile
    try:
        t.start()
        t.join(LLM_FILLER_AFTER if LLM_FILLER_AFTER > 0 else None)
        if t.is_alive():
            if LLM_FILLERS and not _turn.get("filler"):
                _turn["filler"] = True
                speak(random.choice(LLM_FILLERS))    # the model keeps working
            t.join()
    finally:
        with state.lock:
            state.thinking = False
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _first_delta(gen):
    """Start a stream and pull its first delta (so failures show before
    anything is spoken). Returns an iterator with that delta put back, or
    None for an empty / failed stream."""
    try:
        first = next(gen)
    except StopIteration:
        return None
    return _chain(first, gen)


def _chain(first, rest):
    yield first
    yield from rest


# ── Data Fetch: ROUTE then ANSWER (fetch.py) ──────────────────────────────────

def _route(text, kb_score):
    """LLM call 1. Returns (fetch.Route, hint pages) — Route None when the
    model gave nothing (error)."""
    hints = fetch.library.name_hints(text) if FETCH_ENABLED else []
    if (FETCH_KB_PAGE in _library_pages() and kb_score >= LOCAL_MATCH_THRESH
            and FETCH_KB_PAGE not in hints):
        hints.append(FETCH_KB_PAGE)
    notes, _ = _notes(text)
    msg = fetch.route_message(text, unsure=_speech_unsure(text),
                              hints=hints[:FETCH_MAX_PAGES], **notes)
    stream = llm_local.stream(_messages(msg), model=FETCH_ROUTE_MODEL)

    def work():
        route = fetch.read_route(stream, _library_pages())
        if route.kind == "reply":            # its first words, too, inside
            route.reply = _first_delta(_recorded(text, route.reply))
        return route                         # the filler's wait

    route = _with_filler(work)
    console.detail(f"[brain] route: {route.kind} {route.pages or ''} "
                   f"(model said {route.raw.strip()[:60]!r})")
    if route.kind == "reply" and route.reply is None:
        return None, hints                   # nothing after "REPLY:"
    if not route.raw.strip():
        return None, hints                   # the model gave nothing
    return route, hints


def _answer_stream(text, pages):
    """LLM call 2: the query with the data of `pages` (none: a general
    question the route step did not answer). Returns a reply iterator
    (first delta already pulled) or None."""
    data, used = fetch.page_data(pages, text)
    _turn["web_matches"] = [[p, 1.0] for p in used]
    console.detail(f"[brain] data: {len(data.split())} words from {used}")
    notes, exemplars = _notes(text)
    msg = fetch.answer_message(text, data, exemplars=exemplars, **notes)
    gen = _recorded(text, llm_local.stream(_messages(msg),
                                           FETCH_ANSWER_MAX_TOKENS))
    return _with_filler(lambda: _first_delta(gen)), used


def _show_pages(pages):
    """The clean-mode [source] line for an answer from page files."""
    kb = [q for q in ("knowledge base",) if FETCH_KB_PAGE in pages]
    web = fetch.sources([p for p in pages if p != FETCH_KB_PAGE])
    console.source(web=web, kb=kb,
                   kind=None if web or kb else "general knowledge")


def _answer_in_one_call(text, kb_score):
    """Step 5, digest mode: ONE call. The digest of every page is in the
    cached system prompt; the page lines matching the query's names / rare
    words (misheard ones too) come with the question. The model answers,
    or says AGAIN (misheard) / IGNORE (noise). False when it gave nothing."""
    lines = fetch.library.lookup_lines(text)
    if len(lines) < 2 and FETCH_SECTION_WORDS and _about_college(text, kb_score):
        lines += fetch.library.best_section(text, FETCH_SECTION_WORDS)
    if lines:
        console.detail(f"[brain] matching lines: {len(lines)}")
    notes, exemplars = _notes(text)
    msg = fetch.ask_message(text, lines, exemplars=exemplars,
                            unsure=_speech_unsure(text), **notes)
    stream = llm_local.stream(_messages(msg), FETCH_ANSWER_MAX_TOKENS)

    def work():
        route = fetch.read_route(stream, [])
        if route.kind == "reply":
            route.reply = _first_delta(_recorded(text, route.reply))
        return route

    route = _with_filler(work)
    if route.kind == "pages" or (route.kind == "reply" and route.reply is None):
        return False                         # nothing usable
    if route.kind in ("ignore", "again"):
        return _speak_route(text, route)
    college = _about_college(text, kb_score)
    console.source(kind=("college facts (digest + knowledge base"
                         + (f" + {len(lines)} matching lines)" if lines else ")")
                         if college else "general knowledge"))
    speak(route.reply)
    reply = _last_reply()
    _mark_followup(reply)
    _log_turn("llm", reply)
    if college:
        _cache_store(text, reply)
    return True


def _speak_route(text, route):
    """IGNORE: stay silent. AGAIN: ask to hear it again."""
    if route.kind == "ignore":
        console.detail(f"[brain] Ignored as noise: {text!r}")
        _mark_first_token()
        with state.lock:                     # no "answering Luna's question"
            state.awaiting_answer = False    # bonus for the next stray sound
        _log_turn("ignored")
        return True
    console.detail(f"[brain] Probably misheard: {text!r}")
    _say(random.choice(FETCH_REPEAT_REPLIES), path="again")
    with state.lock:
        state.awaiting_answer = False
    return True


def _answer_with_fetch(text, kb_score):
    """Step 5, route mode: route, then reply / ask again / stay silent /
    answer from the pages. False when the LLM gave nothing (the caller
    falls back)."""
    direct = _direct_pages(text, kb_score) if FETCH_DIRECT else []
    if direct:                               # no route call needed
        console.detail(f"[brain] route: direct {direct}")
        route, hints = fetch.Route("pages", direct), []
    else:
        route, hints = _route(text, kb_score)
    if route is None:                        # nothing usable ("REPLY" and no
        route = fetch.Route("pages", [])     # answer): the ANSWER call, with
                                             # the hints' data or none
    if route.kind in ("ignore", "again"):
        return _speak_route(text, route)
    if route.kind == "reply" and _about_college(text, kb_score):
        # never the model's own knowledge for the college: it invents names
        _close_stream(route.reply)
        pages = fetch.library.word_pages(text) or (
            [FETCH_KB_PAGE] if FETCH_KB_PAGE in _library_pages() else [])
        console.detail(f"[brain] route said REPLY to a college question — "
                       f"answering from {pages}")
        route = fetch.Route("pages", pages)
    if route.kind == "reply":
        console.source(kind="general knowledge (no page needed)")
        speak(route.reply)
        reply = _last_reply()
        _mark_followup(reply)
        _log_turn("llm", reply)
        return True
    # PAGES: the model's choice, then person-name hints it did not list
    pages = route.pages + [h for h in hints
                           if h not in route.pages and h != FETCH_KB_PAGE]
    if not pages and FETCH_KB_PAGE in hints:
        pages = [FETCH_KB_PAGE]
    stream, used = _answer_stream(text, pages[:FETCH_MAX_PAGES])
    if stream is None:
        return False
    _show_pages(used)
    speak(stream)
    reply = _last_reply()
    _mark_followup(reply)
    _log_turn("llm", reply)
    _cache_store(text, reply)
    return True


_COLLEGE = frozenset(w.lower() for w in FETCH_COLLEGE_WORDS)


def _about_college(text, kb_score):
    """Is this query about the college (so its answer must come from data)?"""
    words = set(re.findall(r"[a-z]+", text.lower()))
    return bool(words & _COLLEGE) or kb_score >= LOCAL_MATCH_THRESH


def _close_stream(it):
    close = getattr(it, "close", None)
    if close:
        try:
            close()
        except Exception:                    # noqa: BLE001
            pass


def _direct_pages(text, kb_score):
    """The pages when they are obvious, so the ROUTE call can be skipped:
    "who is <a name in the index>" -> the files naming that person; else
    the ONE page file the query names ("placements"). [] = ask the model."""
    pages = (fetch.library.name_hints(text)
             if intents.asks_about_person(text) else [])
    if not pages:
        pages = fetch.library.title_match(text)
    if (pages and kb_score >= LOCAL_MATCH_THRESH
            and FETCH_KB_PAGE in _library_pages()):
        pages = pages + [FETCH_KB_PAGE]
    return pages[:FETCH_MAX_PAGES]


# ── Answer cache: the same question again is answered at once ───────────────
# Answers from page data are kept for FETCH_CACHE_DAYS (a new weekly index
# clears them). Not kept: answers about who is around (scene) or that name
# the person asking — those are about that moment.
_cache = {"key": None, "items": [], "vecs": None}
_CACHE_MAX = 300


def _cache_path():
    path = FETCH_CACHE_PATH
    return path if os.path.isabs(path) else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), path)


def _cache_ready():
    """Load (once per index) the cached answers still valid."""
    key = fetch.library.catalog.get("built_at")
    if _cache["key"] == key and _cache["vecs"] is not None:
        return True
    items = []
    try:
        import json
        with open(_cache_path(), encoding="utf-8") as f:
            items = json.load(f)
    except (OSError, ValueError):
        pass
    now = time.time()
    items = [i for i in items if i.get("key") == key
             and now - i.get("t", 0) < FETCH_CACHE_DAYS * 86400]
    _cache.update(key=key, items=items,
                  vecs=embed([i["q"] for i in items]) if items else
                  np.zeros((0, 384), dtype=np.float32))
    return _cache["vecs"] is not None


def _cache_lookup(text):
    """A cached answer to (nearly) this question, or None."""
    if not FETCH_CACHE or _st_model is None or not _cache_ready():
        return None
    if not _cache["items"] or intents.needs_scene_context(text):
        return None
    sims = _cache["vecs"] @ embed_query(text)
    best = int(np.argmax(sims))
    if sims[best] < FETCH_CACHE_SIM:
        return None
    return _cache["items"][best]["a"]


def _cache_store(question, reply):
    if (not FETCH_CACHE or _st_model is None or not reply
            or intents.needs_scene_context(question) or not _cache_ready()):
        return
    person = _turn.get("person")
    if person and person.split()[0].lower() in reply.lower():
        return
    item = {"q": question, "a": reply, "t": time.time(), "key": _cache["key"]}
    _cache["items"] = (_cache["items"] + [item])[-_CACHE_MAX:]
    _cache["vecs"] = np.vstack([_cache["vecs"], embed([question])])[-_CACHE_MAX:]
    try:
        import json
        path = _cache_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(_cache["items"], f, ensure_ascii=False)
        os.replace(path + ".tmp", path)
    except OSError as e:
        print(f"[brain] answer cache not saved: {e}")


# ── Priming the cache ─────────────────────────────────────────────────────────

def prime():
    """Have the model read the system prompt once (one token of output), so
    Ollama caches it before the first visitor asks — on the Pi reading the
    LIBRARY takes far longer than any one question."""
    if not llm_local.available():
        return False
    t0 = time.time()
    msgs = _messages(fetch.route_message("hello"))
    llm_local.ask(msgs, max_tokens=1)
    if FETCH_ROUTE_MODEL:                    # the route model keeps its own
        llm_local.ask(msgs, max_tokens=1, model=FETCH_ROUTE_MODEL)
    console.detail(f"[brain] system prompt cached ({time.time() - t0:.1f}s)")
    return True


def _prime_when_ready():
    t0 = time.time()
    while time.time() - t0 < LLM_BOOT_WAIT + 60:
        if llm_local.available():
            try:
                prime()
            except Exception as e:           # noqa: BLE001
                print(f"[brain] could not prime the LLM cache: {e}")
            return
        time.sleep(1.0)


if FETCH_PRIME_AT_BOOT:
    threading.Thread(target=_prime_when_ready, daemon=True,
                     name="llm_prime").start()


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
    """Normalised embedding of one utterance, memoised: the same text is
    embedded by the addressing gate and the KB match."""
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
    """Similarity of `text` (college short forms spelled out — "hod" barely
    matches "head of department" otherwise) to every KB question, or None
    without embeddings."""
    if _st_model is None or question_embeddings is None:
        return None
    return question_embeddings @ embed_query(intents.expand_abbreviations(text))


_STOP = set("""a an the is are was were be been am do does did of in on at to
for from by with and or what who whom whose which where when why how can could
would will shall should may might me my you your i we our they their it its
this that these those there here tell about please any all some much many
college tkm tkmce engineering kollam department dept""".split())


def _stem(w):
    return w[:5]


KB_SURE = 0.85


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

    def covered(w):
        if _stem(w) in entry:
            return True
        full = [x for x in re.findall(r"[a-z]+",
                                      intents.ABBREVIATIONS.get(w, "").lower())
                if x not in _STOP and x != w]      # "hod" -> head (department)
        return bool(full) and all(_stem(x) in entry for x in full)

    words = [w for w in re.findall(r"[a-z]+", text.lower())
             if w not in _STOP and len(w) > 2]
    return all(covered(w) for w in words)


def _best_local_match(text):
    """Return (best_score, best_index) of the closest knowledge-base question."""
    scores = _kb_scores(text)
    if scores is None:
        return _best_local_match_keywords(text)
    best_index = int(np.argmax(scores))
    return float(scores[best_index]), best_index


def knowledge_match_score(text):
    """Semantic similarity of `text` to the closest knowledge-base entry (0..1).
    Used by the STT layer's meaning gate / addressing score."""
    if not text.strip() or not questions:
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


def _say(reply, path="fallback"):
    """Speak a fixed sentence (no LLM) and log the turn."""
    _mark_followup(reply)
    _mark_first_token()
    _turn["style_arm"] = None
    speak(reply)
    _log_turn(path, reply)


# ── Vision questions ──────────────────────────────────────────────────────────

def _answer_vision_query(text):
    """Describe the live camera scene and answer through the LLM (same
    system prompt as every other call) — or speak the description directly
    when the LLM isn't available (it's written to be speakable on its own)."""
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
    memory = _scene_notes(text, vision=True)
    camera = f"{desc} Recently: {memory}" if memory else desc

    console.source(kind="camera")
    _wait_for_llm()
    if llm_local.available() and LLM_STREAMING:
        notes, exemplars = _notes(text, vision=True)
        msg = fetch.answer_message(text, None, exemplars=exemplars,
                                   camera=camera, **notes)
        stream = _first_delta(_recorded(text, llm_local.stream(_messages(msg))))
        if stream is not None:
            speak(stream)
            reply = _last_reply()
            _mark_followup(reply)
            _log_turn("vision", reply)
            return

    face.curious()
    _mark_first_token()
    _turn["style_arm"] = None                # no LLM → no style was applied
    speak(desc)
    _log_turn("vision", desc)


# ── Date and time ─────────────────────────────────────────────────────────────

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


# ── Person cards ──────────────────────────────────────────────────────────────

def _maybe_person_card(text, kb_score, kb_index):
    """Put up a person card when — and only when — Luna is certain who the
    question is about: it names exactly one known person ("who is reshma
    sheik"), or it confidently matches a knowledge-base entry about one
    person ("who is the principal"). Every word on the card is a stored
    fact (people.py: the knowledge base + the website's staff lists from
    the index), never the model's. Returns the Person or None."""
    if not PERSON_CARDS:
        return None
    try:
        d = people.directory(fetch.library.staff())
        person = d.match_question(text)
        if (person is None and answers and kb_score >= LOCAL_ANSWER_THRESH
                and _kb_direct_ok(text, kb_index, kb_score)):
            person = d.for_kb_question(questions[kb_index], text)
        if person is None:
            return None
        face.person_card(person.name, person.role, person.detail)
        console.detail(f"[brain] person card: {person.name}")
        return person
    except Exception as e:                   # noqa: BLE001
        print(f"[brain] person card failed: {e}")
        return None


# ── Main process ──────────────────────────────────────────────────────────────

def process(text):
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

    # 2 — the date / the time: from the clock, shown as a card
    when = intents.date_time(lower)
    if when:
        _answer_date_time(when)
        return

    # 3 — questions about what the camera sees
    if ENABLE_SCENE_CONTEXT and scene_context.is_vision_query(lower):
        _set_learning_context(text, 0.0)
        _answer_vision_query(text)
        return

    kb_score, kb_index = (_best_local_match(text) if answers else (0.0, 0))
    _set_learning_context(text, kb_score)

    # a person Luna is certain about: their card goes up while she answers
    _maybe_person_card(text, kb_score, kb_index)

    # 4 — a knowledge-base entry that matches very closely: said verbatim,
    # instantly, exactly as the KB author wrote it
    if (ANSWER_OFFLINE_FIRST and answers and kb_score >= LOCAL_ANSWER_THRESH
            and _kb_direct_ok(text, kb_index, kb_score)):
        reply = answers[kb_index]
        page = faq_page[kb_index] if kb_index < len(faq_page) else None
        if page:                             # an indexing FAQ entry
            console.source(web=fetch.sources([page]), kind="FAQ")
        else:
            console.source(kb=[questions[kb_index]])
        console.detail(f"[brain] Answered from {'FAQ' if page else 'knowledge base'} "
                       f"(match {kb_score:.2f}) — no LLM")
        _turn["kb_matches"] = [[questions[kb_index], round(kb_score, 3)]]
        _remember(text, reply)
        _say(reply, path="kb")
        return

    # 4b — asked (nearly) the same before: the answer it got, at once
    cached = _cache_lookup(text)
    if cached:
        console.source(kind="answer cache")
        console.detail("[brain] Answered from the answer cache — no LLM")
        _remember(text, cached)
        _say(cached, path="kb")
        return

    # 5 — Data Fetch: route (LLM call 1), then reply / ask again / stay
    # silent / answer from the chosen pages (LLM call 2), streamed into the
    # voice sentence by sentence
    _wait_for_llm()
    if llm_local.available() and LLM_STREAMING:
        try:
            answer = (_answer_in_one_call if _digest_mode()
                      else _answer_with_fetch)
            if answer(text, kb_score):
                return
        except Exception as e:               # noqa: BLE001
            print(f"[brain] data fetch failed: {e}")

    # 6 — no LLM (missing / still loading / failed): closest KB entry
    console.detail("[brain] LLM unavailable — using local knowledge base")
    console.source(kind="knowledge base (LLM unavailable)")
    reply = _local_fallback(text)
    if not _turn.get("kb_matches") and answers:
        _turn["kb_matches"] = [[questions[kb_index], round(kb_score, 3)]]
    _say(reply, path="fallback")


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
                if (llm_local.available()
                        and learning.maybe_write_diary(llm_local.ask, now)
                        and FETCH_PRIME_AT_BOOT):
                    prime()          # the diary prompt pushed ours out
        except Exception as e:               # noqa: BLE001
            if not warned:
                warned = True
                print(f"[learn] background tick failed: {e}")


if learning is not None and getattr(learning, "_cfg", {}).get("enabled"):
    threading.Thread(target=_learning_loop, daemon=True,
                     name="learning").start()
