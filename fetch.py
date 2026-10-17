# fetch.py
"""
fetch.py — Data Fetch(): what Luna reads before she answers a question.
The full algorithm, with the indexing side, is in docs/Fetch Algo.md.

  ROUTE   LLM call 1. The system prompt holds Luna's persona, the LIBRARY
          (every page file : one-line description, from catalog.json) and
          FETCH_SYSTEM_RULES. It is the same for every call, so Ollama keeps
          it cached and the Pi only reads each question's new tokens. The
          user message (FETCH_ROUTE_TASK) carries the query, who is asking,
          scene notes, "speech recognition was unsure" and name hints. The
          model answers with one line:
              PAGES: a, b     -> ANSWER below
              REPLY: ...      -> spoken as it streams (no data needed)
              AGAIN           -> ask the visitor to repeat
              IGNORE          -> noise: say nothing
  ANSWER  LLM call 2, same system prompt: the query, the data of those
          pages (whole while they fit FETCH_DATA_WORDS, else their sections
          and lines that share words with the query), who is asking, scene
          notes and FETCH_ANSWER_TASK's instructions.

brain.py runs the calls (it owns the conversation history, streaming and
speech); this module builds the prompts and reads the replies. Pure text +
files: no model is loaded here, so it is unit-testable anywhere.

    python fetch.py "who is the hod of cse"   # route + answer, with timings
    python fetch.py --library                 # the LIBRARY the model sees
"""

import difflib
import itertools
import json
import math
import os
import re
import threading
from dataclasses import dataclass, field

import intents

try:
    import config as _config
except Exception:                       # standalone use without config.py
    _config = None

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

_DEFAULTS = dict(
    SITE_DIR="data/site", KNOWLEDGE_PATH="data/knowledge.txt",
    FETCH_ENABLED=True, FETCH_MAX_PAGES=3, FETCH_DATA_WORDS=700,
    FETCH_MODE="digest", FETCH_LOOKUP_LINES=6, FETCH_SECTION_WORDS=120,
    FETCH_DIGEST_RULES="FACTS:\n{digest}",
    FETCH_DIGEST_TASK="Query: \"{query}\"\n{lines}\n{notes}",
    FETCH_KB_PAGE="kb", FETCH_SYSTEM_RULES="{library}",
    FETCH_ROUTE_TASK="TASK: ROUTE\nQuery: \"{query}\"\n{notes}",
    FETCH_ANSWER_TASK="TASK: ANSWER\nQuery: \"{query}\"\nData:\n{data}\n{notes}",
)


def _cfg(name):
    return getattr(_config, name, _DEFAULTS[name])


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(_PROJECT_DIR, path)


# ── The library: catalog.json + the page files ───────────────────────────────

_STOP = frozenset("""a an the is are was were be been am do does did of in on
at to for from by with and or not no what who whom whose which where when why
how can could would will shall should may might me my you your i we our they
their it its this that these those there here tell about please any all some
much many give list name names know get got has have had also just more
luna college tkm tkmce kollam department dept""".split())


def query_words(text):
    """The words of a query worth matching inside a page: no stop words,
    college short forms spelled out ("hod" -> head, department, hod)."""
    text = intents.expand_abbreviations(text or "").lower()
    return {w for w in re.findall(r"[a-z0-9]+", text)
            if w not in _STOP and (len(w) > 2 or w.isdigit())}


LOOKUP_COMMON = 250         # a word on more page lines than this ("tech",
                            # "students") points at nothing; below it a
                            # word counts more the rarer it is
LOOKUP_WORDS  = 35          # a longer matching line is cut around the match
LOOKUP_KEEP   = 0.5         # lines scoring under this share of the best go


def _around(line, near_words, max_words):
    """`line`, or its max_words words around the first one that starts like
    a query word (a whole paragraph is one "line" on some pages)."""
    words = line.split()
    if len(words) <= max_words:
        return line
    at = next((i for i, w in enumerate(words)
               if any(w.lower().strip(".,;:()").startswith(q[:4])
                      for q in near_words)), 0)
    start = max(0, min(at - max_words // 3, len(words) - max_words))
    return ("… " if start else "") + " ".join(words[start:start + max_words]) + " …"


@dataclass
class Page:
    name: str
    title: str
    description: str
    sections: list = field(default_factory=list)   # [(heading, [lines])]
    sources: list = field(default_factory=list)

    @property
    def words(self):
        return sum(len(h.split()) + sum(len(l.split()) for l in ls)
                   for h, ls in self.sections)


def parse_page(text):
    """(title, [(heading, [lines])]) of a page file; the Sources / Updated
    header lines are for people, not for the model."""
    title, sections, cur = "", [], None
    for line in (text or "").split("\n"):
        if line.startswith("# ") and not title:
            title = line[2:].strip()
            continue
        if cur is None and line.startswith(("Sources: ", "Updated: ")):
            continue
        if line.startswith(("## ", "### ")):  # sub-headings too: finer
            cur = (line.lstrip("#").strip(), [])       # pieces to pick from
            sections.append(cur)
            continue
        if line.strip():
            if cur is None:
                cur = ("", [])
                sections.append(cur)
            cur[1].append(line.rstrip())
    return title, sections


def _kb_sections(path):
    """knowledge.txt ("question: answer" per line) as one section — read on
    every use, so edits are live like the rest of the knowledge base."""
    lines = []
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                raw = " ".join(raw.split())
                if ":" in raw:
                    q, a = raw.split(":", 1)
                    lines.append(f"- {q.strip()}: {a.strip()}")
    except OSError:
        pass
    return [("Curated facts", lines)] if lines else []


class Library:
    """catalog.json and its page files, re-read when the catalog changes
    (after the weekly indexing and a deploy — no restart needed)."""

    def __init__(self, site_dir=None):
        self.dir = _abs(site_dir or _cfg("SITE_DIR"))
        self._lock = threading.Lock()
        self._mtime = None
        self.catalog = {}
        self._pages = {}
        self._lines = None
        self._digest = (None, "")

    @property
    def catalog_path(self):
        return os.path.join(self.dir, "catalog.json")

    def _refresh(self):
        try:
            m = os.path.getmtime(self.catalog_path)
        except OSError:
            m = None
        if m == self._mtime:
            return
        cat = {}
        if m is not None:
            try:
                with open(self.catalog_path, encoding="utf-8") as f:
                    cat = json.load(f)
            except (OSError, ValueError) as e:
                print(f"[fetch] catalog unreadable ({e}) — no website pages")
        self.catalog, self._pages, self._mtime = cat, {}, m
        self._lines = None                       # lookup index, rebuilt lazily

    def names(self):
        with self._lock:
            self._refresh()
            return list((self.catalog.get("pages") or {}).keys())

    def entries(self):
        with self._lock:
            self._refresh()
            return dict(self.catalog.get("pages") or {})

    def page(self, name):
        """The Page `name`, or None (unknown / file missing)."""
        with self._lock:
            self._refresh()
            e = (self.catalog.get("pages") or {}).get(name)
            if e is None:
                return None
            if name == _cfg("FETCH_KB_PAGE"):    # always the live file
                return Page(name, e.get("title", name), e.get("description", ""),
                            _kb_sections(_abs(_cfg("KNOWLEDGE_PATH"))),
                            e.get("sources") or [])
            if name not in self._pages:
                try:
                    with open(os.path.join(self.dir, "pages", e["file"]),
                              encoding="utf-8") as f:
                        title, sections = parse_page(f.read())
                except OSError:
                    return None
                self._pages[name] = Page(name, title or e.get("title", name),
                                         e.get("description", ""), sections,
                                         e.get("sources") or [])
            return self._pages[name]

    def name_hints(self, text, limit=None):
        """Page files naming a person the query names ("who is saheer" ->
        cse_people), best first: the more of the query's name words a page
        has, the better."""
        with self._lock:
            self._refresh()
            idx = self.catalog.get("names") or {}
        score = {}
        vocab = list(idx)
        for w in query_words(text):
            # a misheard name ("rishma" for reshma) by its closest spelling
            for m in ([w] if w in idx else difflib.get_close_matches(
                    w, vocab, n=1, cutoff=0.8) if len(w) >= 4 else ()):
                for p in idx.get(m, ()):
                    score[p] = score.get(p, 0) + 1
        ranked = sorted(score, key=lambda p: (-score[p], p))
        return ranked[:limit or _cfg("FETCH_MAX_PAGES")]

    def digest(self):
        """data/site/digest.txt (site_index.py): the key facts of every
        page file in one compact text — "" when there is none yet."""
        path = os.path.join(self.dir, "digest.txt")
        try:
            m = os.path.getmtime(path)
        except OSError:
            return ""
        with self._lock:
            if self._digest[0] != m:
                with open(path, encoding="utf-8") as f:
                    self._digest = (m, f.read().strip())
            return self._digest[1]

    def _line_index(self):
        """(lines, {word: [line ids]}) over every page file's lines, for
        lookup_lines(). Built once per catalog."""
        if self._lines is not None:
            return self._lines
        lines, words = [], {}
        for name in self.names():
            if name == _cfg("FETCH_KB_PAGE"):
                continue
            page = self.page(name)
            for head, ls in (page.sections if page else ()):
                for line in ls:
                    i = len(lines)
                    lines.append((page.title, head, line.lstrip("- ").strip()))
                    for w in set(re.findall(r"[a-z]{4,}", line.lower())):
                        words.setdefault(w, []).append(i)
        self._lines = (lines, words, sorted(words))
        return self._lines

    def lookup_lines(self, text, limit=None):
        """The page lines that share the query's distinctive words — a
        name, a club, a lab — matched by their closest spelling too
        ("rishma sheikh" -> "Dr. Reshma Sheik, Asst. Professor"). Words on
        more than LOOKUP_COMMON lines ("professor") say nothing and are
        skipped. Best lines first, as "line (page / section)"."""
        limit = limit if limit is not None else _cfg("FETCH_LOOKUP_LINES")
        if not limit:
            return []
        lines, words, vocab = self._line_index()
        count = {}                     # line -> query words it matches
        score = {}                     # line -> summed similarity of the
        for w in set(re.findall(r"[a-z]{4,}", (text or "").lower())) - _STOP:
            sims = {m: difflib.SequenceMatcher(None, w, m).ratio()
                    for m in difflib.get_close_matches(   # misheard names:
                        w, [v for v in vocab if abs(len(v) - len(w)) <= 2],
                        n=25, cutoff=0.78)}            # closest spellings
            if w in words:
                sims[w] = 1.0
            best_of = {}                 # line -> this word's best match
            for m, sim in sims.items():
                if len(words[m]) > LOOKUP_COMMON:
                    continue
                weight = sim * math.log(len(lines) / len(words[m]))
                for i in words[m]:
                    best_of[i] = max(best_of.get(i, 0.0), weight)
            for i, weight in best_of.items():
                score[i] = score.get(i, 0.0) + weight
                count[i] = count.get(i, 0) + 1
        for i in score:                  # both of the name's words beat one;
            score[i] *= count[i]         # a staff entry beats a paragraph
            score[i] /= 1 + len(lines[i][2].split()) / 40
        best = sorted(score, key=lambda i: (-score[i], i))[:limit]
        if best:                         # only the lines nearly as good as
            top = score[best[0]]         # the best: "Vaishnav V Bishoy", not
            best = [i for i in best if score[i] >= LOOKUP_KEEP * top]  # every Vaishnavi
        out = []
        for i in best:
            title, head, line = lines[i]
            out.append(_around(line, near_words=set(re.findall(
                r"[a-z]{4,}", (text or "").lower())), max_words=LOOKUP_WORDS)
                + (f" ({head})" if head and head.lower() not in line.lower()
                   else ""))
        return out

    def best_section(self, text, budget):
        """The section(s) of the ONE page that shares most of the query's
        words, within `budget` words, as lines — for a college question
        whose words match no distinctive line ("when does b tech admission
        start" -> the admissions schedule)."""
        pages = self.word_pages(text, limit=1)
        page = self.page(pages[0]) if pages else None
        if not page:
            return []
        out = []
        for head, ls in fit_page(page, query_words(text), budget):
            out += [f"{l.lstrip('- ').strip()}" + (f" ({head})" if head else "")
                    for l in ls]
        return out

    def title_match(self, text):
        """The ONE page file whose name the query says ("placements",
        "examinations", "admission") — by its words of 6+ letters, compared
        on their first 6 — or [] when none or several match."""
        stems = {w[:6] for w in query_words(text) if len(w) >= 6}
        hits = [n for n in self.names() if n != _cfg("FETCH_KB_PAGE")
                and any(len(p) >= 6 and p[:6] in stems for p in n.split("_"))]
        return hits if len(hits) == 1 else []

    def word_pages(self, text, limit=None):
        """Page files ranked by how many of the query's words their name and
        description share (first 5 letters) — the fallback when the model
        must not answer from its own knowledge."""
        stems = {w[:5] for w in query_words(text)}
        scores = {}
        for n, e in self.entries().items():
            have = {w[:5] for w in re.findall(r"[a-z0-9]+", (n.replace("_", " ")
                    + " " + (e.get("description") or "")).lower())}
            hit = len(stems & have)
            if hit:
                scores[n] = hit
        ranked = sorted(scores, key=lambda n: (-scores[n], n))
        best = scores[ranked[0]] if ranked else 0
        return [n for n in ranked if scores[n] == best][:limit or 2]

    def faq(self):
        """[(question, answer, page)] written at indexing time."""
        with self._lock:
            self._refresh()
            return [tuple(f) for f in self.catalog.get("faq") or ()]

    def staff(self):
        """[(title, text)] staff lists for people.py's person cards."""
        with self._lock:
            self._refresh()
            return [tuple(s) for s in self.catalog.get("staff") or ()]

    def available(self):
        return bool(self.names())


library = Library()


# ── Prompts ───────────────────────────────────────────────────────────────────

def library_lines(lib=None):
    """"name: description" per page — the LIBRARY in the system prompt."""
    lib = lib or library
    return "\n".join(f"{n}: {e.get('description') or e.get('title') or n}"
                     for n, e in lib.entries().items())


def system_prompt(persona_text, lib=None):
    """Persona + LIBRARY + routing / answering rules. Identical on every
    call until the catalog changes, so Ollama's prefix cache keeps it."""
    lines = library_lines(lib)
    if not lines or not _cfg("FETCH_ENABLED"):
        return persona_text
    rules = _cfg("FETCH_SYSTEM_RULES").format(
        library=lines, max_pages=_cfg("FETCH_MAX_PAGES"))
    return f"{persona_text}\n\n{rules}"


def _notes(person=None, scene=None, unsure=False, hints=(), style=None,
           profile=None, diary=None, greeted=False):
    out = []
    if person:
        out.append(f"The person asking is {person} (you recognised their "
                   f"face)." + (" You are mid-conversation: do not greet "
                                "them again." if greeted else ""))
    if profile:
        out.append(profile)
    if scene:                    # under "answer from this" the model used to
        out.append("Scene context, for your understanding only — do not "
                   "bring it up, comment on it or greet anyone because of it "
                   f"unless the query is about it: {scene}")   # narrate it
    if diary:
        out.append(f"From your diary: {diary}")
    if unsure:
        out.append("Speech recognition was unsure of some of these words.")
    if hints:
        out.append("Names in the query appear in: " + ", ".join(hints) + ".")
    if style:
        out.append(f"Style for a reply: {style}")
    return "\n".join(out)


def digest_system_prompt(persona_text, lib=None):
    """FETCH_MODE "digest": persona (+ knowledge base) + the digest + how to
    answer. Identical on every call until the index or knowledge.txt
    changes, so Ollama reads it once and keeps it cached."""
    digest = (lib or library).digest()
    if not digest or not _cfg("FETCH_ENABLED"):
        return persona_text
    return f"{persona_text}\n\n" + _cfg("FETCH_DIGEST_RULES").format(
        digest=digest)


def ask_message(query, lines=(), exemplars=(), **notes):
    """FETCH_MODE "digest": the one message of a question — the query, the
    matching page lines and the notes (FETCH_DIGEST_TASK)."""
    block = ("Lines from the college pages that match the query:\n"
             + "\n".join(f"- {l}" for l in lines)) if lines else ""
    msg = _cfg("FETCH_DIGEST_TASK").format(query=query, lines=block,
                                           notes=_notes(**notes))
    msg = "\n".join(l for l in msg.split("\n") if l.strip())
    if exemplars:
        ex = "\n".join(f"- Q: {q}\n  A: {a}" for q, a in exemplars)
        msg += ("\n\nReplies people liked for similar questions (copy the "
                "manner, not the facts):\n" + ex)
    return msg


def route_message(query, **notes):
    """The ROUTE task (FETCH_ROUTE_TASK) for one query: no blank lines,
    every token counts on the Pi."""
    msg = _cfg("FETCH_ROUTE_TASK").format(query=query, notes=_notes(**notes))
    return "\n".join(l for l in msg.split("\n") if l.strip())


def answer_message(query, data, exemplars=(), camera=None, **notes):
    """The ANSWER task (FETCH_ANSWER_TASK): query + data + notes + the
    instructions. camera: a live scene description ("what do you see?"),
    answered from instead of data."""
    if camera:
        data = f"What your camera sees right now:\n{camera}"
    msg = _cfg("FETCH_ANSWER_TASK").format(
        query=query, data=data or "(no pages were needed)",
        notes=_notes(**notes)).strip()
    msg = re.sub(r"\n{3,}", "\n\n", msg)       # no notes: no gap
    if exemplars:
        ex = "\n".join(f"- Q: {q}\n  A: {a}" for q, a in exemplars)
        msg += ("\n\nReplies people liked for similar questions (copy the "
                "manner, not the facts):\n" + ex)
    return msg


# ── Reading the route ─────────────────────────────────────────────────────────

@dataclass
class Route:
    kind: str                     # "pages" | "reply" | "again" | "ignore"
    pages: list = field(default_factory=list)
    reply: object = None          # "reply": an iterator of text deltas
    raw: str = ""


_KINDS = {"PAGES": "pages", "PAGE": "pages", "REPLY": "reply",
          "ANSWER": "reply", "AGAIN": "again", "REPEAT": "again",
          "IGNORE": "ignore", "NOISE": "ignore", "NONE": "ignore"}


def match_pages(text, names, limit=None):
    """Page names from the model's "PAGES: a, b" line: exact names first,
    then close spellings ("cse_faculty" for "cse_people" is not close
    enough; "cse people" is). Unknown names are dropped."""
    import difflib
    out = []
    norm = {re.sub(r"[^a-z0-9]+", "_", n.lower()).strip("_"): n for n in names}
    for raw in re.split(r"[,;|\n]+", text or ""):
        key = re.sub(r"[^a-z0-9]+", "_", raw.lower().replace(".txt", "")).strip("_")
        if not key:
            continue
        hit = norm.get(key)
        if hit is None:
            close = difflib.get_close_matches(key, list(norm), n=1, cutoff=0.8)
            hit = norm[close[0]] if close else None
        if hit and hit not in out:
            out.append(hit)
    return out[:limit or _cfg("FETCH_MAX_PAGES")]


def read_route(deltas, names):
    """Read the ROUTE reply from a stream of text deltas only as far as
    needed: the first word decides. For "pages" / "again" / "ignore" the
    stream is closed (Ollama stops writing); for "reply" the rest of the
    stream — "REPLY:" stripped — is Route.reply, ready to be spoken. A reply
    that ignores the format is taken as a REPLY of its whole text."""
    it = iter(deltas)
    buf = ""
    word = None
    for d in it:
        buf += d
        m = re.match(r"[\s*_#>\"'`]*([A-Za-z]+)(.?)", buf, re.S)
        if m and m.group(2):                     # the first word is complete
            word = m.group(1).upper()
            break
        if not m and re.match(r"[\s*_#>\"'`]*[^\s*_#>\"'`A-Za-z]", buf):
            break                                # starts with a digit / symbol
    else:
        m = re.match(r"[\s*_#>\"'`]*([A-Za-z]+)", buf)
        word = m.group(1).upper() if m else None
    kind = _KINDS.get(word or "")
    if kind is None and buf.strip().strip(".") == "0":
        kind = "ignore"
    if kind in ("pages", "again", "ignore"):
        if kind == "pages":
            while "\n" not in buf.split(":", 1)[-1].lstrip():
                d = next(it, None)
                if d is None:
                    break
                buf += d
        _close(it)
        body = re.sub(r"^[^:]*:", "", buf.strip(), count=1) if kind == "pages" else ""
        # one line, up to anything the model tacked on ("a, b / REPLY: ...")
        line = re.split(r"\s/\s|\b(?:REPLY|AGAIN|IGNORE|PAGES)\b",
                        body.strip().split("\n")[0])[0]
        pages = match_pages(line, names) if kind == "pages" else []
        return Route(kind, pages, None, buf)
    if kind == "reply":                          # strip the label, keep going
        rest = re.sub(r"^[\s*_#>\"'`]*[A-Za-z]+[\s*_]*:?\s*", "", buf, count=1)
    else:
        rest = buf
    return Route("reply", [], _lstripped(rest, it), buf)


def _lstripped(first, it):
    """`first` then the rest of `it`, without the spaces / newlines the
    reply starts with ("REPLY: " ends with one)."""
    started = False
    for piece in itertools.chain([first], it):
        if not started:
            piece = piece.lstrip()
            if not piece:
                continue
            started = True
        yield piece


def _close(it):
    close = getattr(it, "close", None)
    if close is not None:
        try:
            close()
        except Exception:                        # noqa: BLE001
            pass


# ── The data of the chosen pages ──────────────────────────────────────────────

def _score(text, words):
    low = text.lower()
    return sum(1 for w in words if w in low)


def _fit_section(heading, lines, words, budget):
    """A section cut to `budget` words: its lines that share words with the
    query (best first, then back in page order), else its first lines."""
    head_n = len(heading.split())
    size = head_n + sum(len(l.split()) for l in lines)
    if size <= budget:
        return lines
    scored = [(i, _score(l, words)) for i, l in enumerate(lines)]
    hits = [i for i, s in sorted(scored, key=lambda x: -x[1]) if s > 0]
    order = hits or list(range(len(lines)))
    keep, n = [], head_n
    for i in order:
        w = len(lines[i].split())
        if n + w > budget:
            if keep:
                break
            continue
        keep.append(i)
        n += w
    return [lines[i] for i in sorted(keep)]


def fit_page(page, words, budget):
    """[(heading, lines)] of `page` within `budget` words: the whole page
    when it fits, else its sections that share words with the query (best
    first) — a long one cut to its matching lines — kept in page order;
    with no match at all, the page from the top."""
    if page.words <= budget:
        return page.sections
    scored = []
    for i, (h, lines) in enumerate(page.sections):
        s = 2 * _score(h, words) + _score(" ".join(lines), words)
        scored.append((i, s))
    hits = [i for i, s in sorted(scored, key=lambda x: -x[1]) if s > 0]
    order = hits or [i for i, _ in scored]
    chosen, n = {}, 0
    for i in order:
        h, lines = page.sections[i]
        left = budget - n
        if left < 15:
            break
        part = _fit_section(h, lines, words, left)
        if not part and lines:
            continue
        chosen[i] = part
        n += len(h.split()) + sum(len(l.split()) for l in part)
    return [(page.sections[i][0], chosen[i]) for i in sorted(chosen)]


def page_data(names, query, budget=None, lib=None):
    """The Data block for the ANSWER task: each page under its title,
    sharing `budget` words (FETCH_DATA_WORDS) by rank — the first (most
    useful) page the biggest share, a small page whole and its unused share
    passed on. Returns (data text, [page names used])."""
    lib = lib or library
    budget = budget or _cfg("FETCH_DATA_WORDS")
    pages = [p for p in (lib.page(n) for n in names) if p and p.sections]
    if not pages:
        return "", []
    words = query_words(query)
    weight = {p.name: len(pages) - i for i, p in enumerate(pages)}  # 3, 2, 1
    share, left, wleft = {}, budget, sum(weight.values())
    for p in sorted(pages, key=lambda p: p.words):
        share[p.name] = min(p.words, left * weight[p.name] // wleft)
        left -= share[p.name]
        wleft -= weight[p.name]
    blocks = []
    for p in pages:
        parts = fit_page(p, words, max(share[p.name], 40))
        lines = [f"=== {p.title} ==="]
        for h, ls in parts:
            if h:
                lines.append(f"{h}:")
            lines += ls
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks), [p.name for p in pages]


def sources(names, lib=None):
    """The website URLs behind page files (for the console's [source])."""
    lib = lib or library
    out = []
    for n in names:
        p = lib.page(n)
        if p:
            out += [u for u in p.sources if u not in out]
    return out


# ── CLI: try a question against the index (needs Ollama) ─────────────────────

def _cli(question, show_library, model=None):
    import time
    import persona
    import llm_ollama
    sysmsg = system_prompt(persona.system_prompt())
    if show_library:
        print(sysmsg)
        return
    if model:                                    # compare models / formats
        llm_ollama.set_model(model)
    llm_ollama.warmup()
    if _cfg("FETCH_MODE") == "digest" and library.digest():
        with open(_abs(_cfg("KNOWLEDGE_PATH")), encoding="utf-8") as f:
            sysmsg = digest_system_prompt(persona.system_prompt(f.read()))
    t0 = time.time()
    llm_ollama.ask([{"role": "system", "content": sysmsg},   # cache it, as
                    {"role": "user", "content": route_message("hello")}],
                   max_tokens=1)                             # at boot
    print(f"(system prompt read: {time.time() - t0:.1f}s — once per boot)")
    if _cfg("FETCH_MODE") == "digest" and library.digest():
        lines = library.lookup_lines(question)
        if len(lines) < 2:                       # as brain does for a
            lines += library.best_section(question, 120)  # college question
        msgs = [{"role": "system", "content": sysmsg},
                {"role": "user", "content": ask_message(question, lines)}]
        print(f"ONE CALL with {len(lines)} matching lines: {lines}")
        t1, first, parts = time.time(), None, []
        for d in llm_ollama.stream(msgs, 400):
            first = first or time.time()
            parts.append(d)
        print(f"ANSWER (first word after {(first or time.time()) - t1:.1f}s, "
              f"all {time.time() - t1:.1f}s): {''.join(parts).strip()}")
        return
    names = library.names()
    hints = library.name_hints(question)
    msgs = [{"role": "system", "content": sysmsg},
            {"role": "user", "content": route_message(question, hints=hints)}]
    t0 = time.time()
    r = read_route(llm_ollama.stream(msgs), names)
    t1 = time.time()
    print(f"ROUTE ({t1 - t0:.1f}s): {r.raw.strip()!r} -> {r.kind} {r.pages}"
          + (f"  hints {hints}" if hints else ""))
    if r.kind == "reply":
        print("REPLY:", "".join(r.reply).strip())
        return
    if r.kind != "pages":
        return
    chosen = r.pages + [h for h in hints if h not in r.pages]
    data, used = page_data(chosen[:_cfg("FETCH_MAX_PAGES")], question)
    msgs[-1] = {"role": "user", "content": answer_message(question, data)}
    print(f"DATA: {len(data.split())} words from {used}")
    t2 = time.time()
    first, parts = None, []
    for d in llm_ollama.stream(msgs, 400):
        first = first or time.time()
        parts.append(d)
    print(f"ANSWER (first word after {(first or time.time()) - t2:.1f}s, "
          f"all {time.time() - t2:.1f}s): {''.join(parts).strip()}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Data Fetch: route a question "
                                 "and answer it from the index")
    ap.add_argument("question", nargs="*")
    ap.add_argument("--library", action="store_true",
                    help="print the system prompt (persona + LIBRARY + rules)")
    ap.add_argument("--model", help="another pulled Ollama model, e.g. "
                    "llama3.2:3b-instruct-q4_0 (to compare speed)")
    a = ap.parse_args()
    _cli(" ".join(a.question) or "who is the head of the cse department",
         a.library, a.model)
