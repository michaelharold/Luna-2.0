# site_index.py
"""
site_index.py — Indexing(): turn the college website into a small library of
clean page files that Luna's Data Fetch (fetch.py) can route questions to.

Run by hand about once a week (docs/Fetch Algo.md):

    python site_index.py              # crawl + group + organize + describe
    python site_index.py --offline    # skip the crawl, re-use crawl.json
    python site_index.py --plan       # show the page files it would build
    python site_index.py --status     # what is indexed now
    python site_index.py --force      # ignore the Llama cache

  1. CRAWL     site_crawler.crawl(): every page, its ETLab data and its
               PDFs                                -> <SITE_DIR>/crawl.json
  2. GROUP     each page / data section / PDF goes to ONE page file, by the
               rules in SITE_PAGES; SITE_SKIP_PAGES, duplicates, PDFs named
               only by a number and near-empty text are dropped
  3. ORGANIZE  prose (page text, PDFs) goes to Llama with
               SITE_ORGANIZE_PROMPT, in pieces of SITE_LLM_CHUNK_WORDS; its
               notes must keep SITE_MIN_FACT_KEEP of the numbers and names
               and invent no numbers, else the cleaned text is kept as it is.
               Lists (staff, students, publications, table rows) are
               formatted without the LLM. Each Llama answer is cached by the
               hash of its input (llm_cache.json), so a weekly re-run only
               sends changed pages      -> <SITE_DIR>/pages/<name>.txt
  4. DESCRIBE  one line per page file from Llama (SITE_DESCRIBE_PROMPT),
               plus the knowledge base as page "kb", a name index (person
               name -> page files) and the staff lists for the person cards
                                           -> <SITE_DIR>/catalog.json

A page file reads:

    # CSE people: faculty and staff
    Sources: https://tkmce.ac.in/faculty-cse.php, ...
    Updated: 2026-10-13

    ## Regular teaching staff (faculty, teachers), 13 in total
    - Dr. Aneesh G Nath, Associate Professor
    ...
"""

import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
from datetime import datetime

import console

try:
    import config as _config
except Exception:                       # standalone use without config.py
    _config = None

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

_DEFAULTS = dict(
    SITE_DIR="data/site", SITE_INDEX_MODEL="llama3.2:3b",
    SITE_INDEX_NUM_CTX=8192, SITE_LLM_CHUNK_WORDS=700,
    SITE_MIN_PAGE_WORDS=25, SITE_PDF_MAX_WORDS=1500,
    SITE_GROUP_MAX_WORDS=12000, SITE_MIN_FACT_KEEP=0.8,
    SITE_PAGES=(), SITE_SKIP_PAGES=(), SITE_STUDENTS_PAGE="students",
    SITE_AFFILIATES_PAGE="tkm_institutions", SITE_SKIP_TITLES=(),
    SITE_OTHER_DEPT_DATA=("staff", "hod", "overview"),
    SITE_OTHER_DEPT_MAX_WORDS=3000,
    SITE_ORGANIZE_PROMPT="{title}\n{text}", SITE_DESCRIBE_PROMPT="{title}\n{text}",
    KNOWLEDGE_PATH="data/knowledge.txt", FETCH_KB_PAGE="kb",
    SITE_FAQ_PER_PAGE=0, SITE_FAQ_PROMPT="{title}\n{n}\n{text}",
    SITE_DIGEST_WORDS={"": 80}, SITE_DIGEST_MAX_WORDS=3500,
    SITE_DIGEST_PROMPT="{title}\n{words}\n{text}",
    FETCH_KB_DESCRIPTION="Luna's own curated facts about the college.",
)


def _cfg(name):
    return getattr(_config, name, _DEFAULTS[name])


CATALOG_VERSION = 1
DESCRIBE_WORDS  = 600     # words of a page Llama reads to describe it
DESCRIBE_MAX_WORDS = 40   # a longer "one line" is cut here
SAVE_CACHE_EVERY = 10     # Llama answers between cache saves (a crash or
                          # Ctrl+C loses at most this many)
NAME_MAX_PAGES  = 4       # a name word on more page files than this is too
                          # common to be a routing hint ("mohammed")
FRAGMENT_MIN_LINES  = 30  # prose with this many lines and fewer than
FRAGMENT_LINE_WORDS = 5   # this many words per line is a broken-up table


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(_PROJECT_DIR, path)


def site_dir():
    return _abs(_cfg("SITE_DIR"))


def paths():
    d = site_dir()
    return dict(crawl=os.path.join(d, "crawl.json"),
                pages=os.path.join(d, "pages"),
                catalog=os.path.join(d, "catalog.json"),
                cache=os.path.join(d, "llm_cache.json"))


def _write_json(path, data):
    """Atomic: a crash mid-write never leaves half a file behind."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _words(text):
    return len((text or "").split())


# ── 2. Group ──────────────────────────────────────────────────────────────────
# A "source" is one piece of one page file: the text of a page, one data
# section, the rows of one table, the student rows of one class list, or a
# PDF — {kind, title, url, text}. kind decides how it is organized: "prose"
# goes to Llama, "list" is formatted line by line.

def _fragment(url):
    return url.partition("#")[2].lower()


def _url_match(pattern, url):
    """A URL pattern: a piece of the URL, or "...$" for its end."""
    url = url.lower().partition("#")[0]
    if pattern.endswith("$"):
        return url.endswith(pattern[:-1])
    return pattern in url


def _matches(pattern, url):
    """A SITE_PAGES / SITE_SKIP_PAGES pattern against one URL: "#key"
    matches a data section whose name contains key ("#staff" ->
    non_teaching_staff); any other pattern a piece of the URL."""
    pattern = pattern.lower()
    if pattern.startswith("#"):
        frag = _fragment(url)
        return bool(frag) and pattern[1:] in frag
    return _url_match(pattern, url)


def _slug_name(url):
    """'https://tkmce.ac.in/seed-money.php' -> 'seed_money'."""
    parts = urllib.parse.urlsplit(url)
    slug = re.sub(r"\.(php|html?|aspx?|jsp|pdf)$", "", parts.path.strip("/"),
                  flags=re.I).split("/")[-1]
    if not slug:
        host = parts.netloc.lower()
        slug = (host[4:] if host.startswith("www.") else host).split(".")[0]
    return re.sub(r"[^a-z0-9]+", "_", slug.lower()).strip("_") or "page"


def _meaningful_pdf(title):
    """A PDF named by words ("TKM Academic Policy", "let final 26-27"), not
    by a code ("16-11869642-OBC-2021", "01") — the scholarship page alone
    links some 100 government orders."""
    title = title or ""
    letters = len(re.findall(r"[A-Za-z]", title))
    return letters >= 4 and not re.search(r"\d{5,}", title.replace(" ", ""))


def _skip_reason(doc):
    """Why `doc` is never indexed (SITE_SKIP_PAGES / SITE_SKIP_TITLES), or
    None."""
    page, url = doc.get("page") or doc["url"], doc["url"]
    for pattern in _cfg("SITE_SKIP_PAGES"):
        if _matches(pattern, url) or _matches(pattern, page):
            return f"SITE_SKIP_PAGES ({pattern})"
    title = (doc.get("title") or "").lower()
    for words in _cfg("SITE_SKIP_TITLES"):
        if words.lower() in title:
            return f"SITE_SKIP_TITLES ({words})"
    return None


_DEPT_STOP = {"and", "of", "the", "department", "dept", "engineering",
              "tkm", "tkmce", "college", "centre", "center", "for"}


def _dept_words(text):
    """'Electronics & Communication Engineering' and the URL slug
    'electronics-and-communication-engineering' -> {electronics,
    communication}: what names a department either way."""
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", (text or "").lower())
            if w not in _DEPT_STOP}


def _dept_name(text):
    """'electronics-and-communication-engineering' -> 'dept_electronics_
    communication'."""
    words = [w for w in re.findall(r"[a-z]+", (text or "").lower())
             if w not in _DEPT_STOP]
    return "dept_" + "_".join(words) if words else "dept_other"


class _Departments:
    """The other departments' page files: one per department page in the
    catalog ("dept_mechanical"), and the ETLab sections the home page loads
    for a department matched to it by name."""

    def __init__(self, catalog):
        self.pages = {}                       # page url -> (name, title)
        for url, info in catalog.items():
            if info.get("scope") == "other":
                slug = urllib.parse.urlsplit(url).path.strip("/").split("/")[-1]
                slug = re.sub(r"\.(php|html?)$", "", slug)
                label = " ".join(w.capitalize()
                                 for w in slug.replace("-", " ").split())
                self.pages[url] = (_dept_name(slug), f"{label}: another "
                                                     f"department of the college",
                                   _dept_words(slug.replace("-", " ")))

    def for_page(self, url):
        hit = self.pages.get(url)
        return hit[:2] if hit else None

    def for_label(self, label):
        words = _dept_words(label)
        best, score = None, 0
        for name, title, have in self.pages.values():
            n = len(words & have)
            if n > score:
                best, score = (name, title), n
        return best or (_dept_name(label),
                        f"{label}: another department of the college")


def _group_of(doc, scope, depts):
    """(page file name, its title or None) for one crawled doc, or (None,
    why) when it is not kept."""
    if doc.get("topic"):                       # a student row
        return _cfg("SITE_STUDENTS_PAGE"), None
    page, url = doc.get("page") or doc["url"], doc["url"]
    if doc.get("scope") == "other":            # another department's data
        frag = _fragment(url)
        if not any(k in frag for k in _cfg("SITE_OTHER_DEPT_DATA")):
            return None, "another department's data (SITE_OTHER_DEPT_DATA)"
        return depts.for_label((doc.get("title") or "").split(" — ")[0])
    # a data section by its own name first ("#publications"), then any
    # piece by its own URL (a PDF), then by its page's URL
    rules = _cfg("SITE_PAGES")
    if "#" in url and scope != "affiliate":
        for name, title, patterns in rules:
            if any(p.startswith("#") and _matches(p, url) for p in patterns):
                return name, title
    for target in (url, page):
        for name, title, patterns in rules:
            if any(not p.startswith("#") and _url_match(p.lower(), target)
                   for p in patterns):
                return name, title
    if scope == "other":
        return depts.for_page(page) or depts.for_label(_slug_name(page))
    if scope == "affiliate":
        return _cfg("SITE_AFFILIATES_PAGE"), "Other TKM institutions"
    return _slug_name(page), None


_STUDENT_ROW = re.compile(r"^Name: (?P<name>[^;]+); Batch: (?P<batch>[^;.]+)"
                          r"(?:; Admission No: (?P<adm>[^;.]+))?\.?$")


def _student_line(text, role="student"):
    """'Name: Pranav P; Batch: CSE 2K23A; Admission No: 220032.' ->
    'Pranav P — B.Tech student, batch CSE 2K23A, admission no. 220032':
    said in words, so the model reads it as the fact it is."""
    m = _STUDENT_ROW.match(text.strip())
    if not m:
        return text.strip().rstrip(".")
    line = f"{m['name'].strip()} — {role}, batch {m['batch'].strip()}"
    return line + (f", admission no. {m['adm'].strip()}" if m["adm"] else "")


_IN_TOTAL = re.compile(r"^(?P<head>[^:]{3,160}?, \d+ in total): (?P<items>.*)$",
                       re.S)


def _data_lines(text):
    """(heading suffix, lines) for one ETLab data section. "Publications
    (research papers), 166 in total: A; B; C." becomes the heading
    "..., 166 in total" and one line per item; anything else is one line."""
    text = text.strip().rstrip(".")
    m = _IN_TOTAL.match(text)
    if m:
        return m["head"], [_tidy_caps(i.strip()) for i in m["items"].split("; ")
                           if i.strip()]
    if ". " in text and text.count(": ") >= 2:          # overview fields
        return "", [p.strip() for p in re.split(r"(?<=\.) (?=[A-Z][\w ()]*: )",
                                                text) if p.strip()]
    return "", [text]


_KEEP_CAPS = {"AND": "and", "OF": "of", "THE": "the", "FOR": "for",
              "IN": "in", "AT": "at"}


def _tidy_caps(text):
    """'Dr. ANEESH G NATH, COMPUTER SCIENCE AND ENGINEERING' -> 'Dr. Aneesh
    G Nath, Computer Science and Engineering': read as names, not spelled
    out letter by letter. Short acronyms (CSE, IIT, KTU) stay."""
    def fix(m):
        w = m.group(0)
        if w in _KEEP_CAPS:
            return _KEEP_CAPS[w]
        return w.capitalize() if len(w) >= 4 else w
    return re.sub(r"\b[A-Z]{2,}\b", fix, text) if re.search(
        r"\b[A-Z]{4,}\b", text) else text


def short_title(title):
    """A source's heading without the parts every section of a file
    repeats: "Computer Science & Engineering (CSE) — Regular staff, 13 in
    total" -> "Regular staff, 13 in total"; "CSE Department > Vision Mission
    / Department Vision and Mission" -> "Department Vision and Mission";
    "Placement | TKM College of Engineering" -> "Placement"."""
    t = (title or "").strip()
    if t.startswith("PDF: "):
        return t
    if " — " in t:
        t = t.split(" — ", 1)[1]
    if " / " in t:
        t = t.rsplit(" / ", 1)[1]
    if " | " in t:
        t = t.split(" | ", 1)[0]
    return t.strip() or title


def _strip_table_text(text):
    """A page's text repeats its tables cell by cell ("a | b | c"); the rows
    are kept as their own source, so those lines go."""
    return "\n".join(l for l in text.split("\n") if " | " not in l)


def group(crawl):
    """{name: {title, sources: [source], urls: [...]}} plus a report
    {dropped: [(url, why)]}. Deterministic: same crawl, same groups."""
    catalog = crawl.get("catalog") or {}
    report = crawl.get("report") or {}
    docs = crawl.get("docs") or []
    groups, dropped = {}, []
    depts = _Departments(catalog)
    tabled = {d.get("page") or d["url"] for d in docs
              if re.search(r"#t\d+r\d+$", d["url"])}
    rows = {}          # (group, table key) -> source being filled

    def add(name, title, src, page):
        g = groups.setdefault(name, {"title": title, "sources": [],
                                     "urls": []})
        if title and not g["title"]:
            g["title"] = title
        g["sources"].append(src)
        if page not in g["urls"]:
            g["urls"].append(page)
        return src

    for d in docs:
        page, url = d.get("page") or d["url"], d["url"]
        why = _skip_reason(d)
        if why:
            dropped.append((url, why))
            continue
        scope = (catalog.get(page) or {}).get("scope", "general")
        name, title = _group_of(d, scope, depts)
        if name is None:
            dropped.append((url, title))
            continue
        text = d.get("text") or ""
        if d.get("topic"):                                  # student row
            key = (name, d.get("title") or "Students")
            src = rows.get(key)
            if src is None:
                src = rows[key] = add(name, None, {
                    "kind": "list", "title": key[1], "url": d["topic"],
                    "lines": []}, page)
            src["lines"].append(_student_line(text, short_title(key[1])))
            continue
        m = re.search(r"#t(\d+)r\d+$", url)
        if m:                                                # a table row
            key = (name, f"{page}#t{m.group(1)}")
            src = rows.get(key)
            if src is None:
                src = rows[key] = add(name, title, {
                    "kind": "list", "title": d.get("title") or "Table",
                    "url": key[1], "lines": []}, page)
            src["lines"].append(text.rstrip("."))
            continue
        if "#" in url:                                       # data section
            head, lines = _data_lines(text)
            t = d.get("title") or _fragment(url)
            if head:
                base = t.rsplit(" — ", 1)[0] if " — " in t else ""
                t = f"{base} — {head}" if base else head
            add(name, title, {"kind": "list", "title": t, "url": url,
                              "lines": lines}, page)
            continue
        is_pdf = urllib.parse.urlsplit(url).path.lower().endswith(".pdf")
        if is_pdf and url != page:
            pdf_title = (d.get("title") or "").rsplit(" — ", 1)[-1]
            if not _meaningful_pdf(pdf_title):
                dropped.append((url, "PDF named only by a number / code"))
                continue
            words = text.split()
            if len(words) > _cfg("SITE_PDF_MAX_WORDS"):
                dropped.append((url, f"PDF cut at {_cfg('SITE_PDF_MAX_WORDS')} "
                                     f"of {len(words)} words"))
                text = " ".join(words[:_cfg("SITE_PDF_MAX_WORDS")])
            add(name, title, {"kind": "prose", "title": f"PDF: {pdf_title}",
                              "url": url, "text": text}, page)
            continue
        if page in tabled:
            text = _strip_table_text(text)
        add(name, title, {"kind": "prose", "title": d.get("title") or url,
                          "url": url, "text": text}, page)

    # cap each file's input, keeping its own pages' text and data first
    for name, g in groups.items():
        if name == _cfg("SITE_STUDENTS_PAGE"):
            continue
        cap = (_cfg("SITE_OTHER_DEPT_MAX_WORDS") if name.startswith("dept_")
               else _cfg("SITE_GROUP_MAX_WORDS"))
        order = sorted(range(len(g["sources"])), key=lambda i: (
            g["sources"][i]["title"].startswith("PDF:"), i))
        total, keep = 0, set()
        for i in order:
            s = g["sources"][i]
            n = _words(s.get("text")) + sum(_words(l) for l in s.get("lines", ()))
            if total + n > cap and keep:
                dropped.append((s["url"], f"over SITE_GROUP_MAX_WORDS "
                                          f"({cap}) for {name}"))
                continue
            total += n
            keep.add(i)
        g["sources"] = [s for i, s in enumerate(g["sources"]) if i in keep]
        if not g["title"]:
            g["title"] = _default_title(name, g)
    return groups, {"dropped": dropped,
                    "crawl": {u: s for u, s in report.items()}}


def _default_title(name, g):
    for s in g["sources"]:
        if s["kind"] == "prose" and not s["title"].startswith("PDF:"):
            return s["title"]
    return name.replace("_", " ").capitalize()


# ── 3. Organize ───────────────────────────────────────────────────────────────

_URL_RE  = re.compile(r"(?:https?://|www\.)\S+", re.I)
_NUM_RE  = re.compile(r"\d[\d,]*(?:\.\d+)?")
# proper nouns: a capitalised word in mid-sentence, or an acronym (NAAC)
_NAME_RE = re.compile(r"(?<=[a-z0-9,;:()] )[A-Z][A-Za-z]{2,}|\b[A-Z]{3,}\b")


def clean_text(text):
    """Deterministic clean-up: links gone, whitespace collapsed, lines with
    no letter or digit and repeated lines dropped."""
    out, seen = [], set()
    for line in (text or "").split("\n"):
        line = " ".join(_URL_RE.sub("", line).split()).strip(" |-•·*❮❯")
        if not re.search(r"[A-Za-z0-9]", line):
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return "\n".join(out)


def _chunks(text, max_words):
    """Line-aligned pieces of at most ~max_words words (a single longer line
    is split by words)."""
    pieces, cur, n = [], [], 0
    for line in text.split("\n"):
        words = line.split()
        while len(words) > max_words:
            if cur:
                pieces.append("\n".join(cur))
                cur, n = [], 0
            pieces.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if n + len(words) > max_words and cur:
            pieces.append("\n".join(cur))
            cur, n = [], 0
        if words:
            cur.append(" ".join(words))
            n += len(words)
    if cur:
        pieces.append("\n".join(cur))
    return pieces


def _numbers(text):
    return {n.replace(",", "") for n in _NUM_RE.findall(text or "")}


def fact_check(src, out, min_keep):
    """(ok, kept fraction, invented numbers) for Llama's notes `out` of
    `src`: every number and capitalised name of the source should still be
    there (case-insensitive), and no number of two or more digits may
    appear that the source does not have."""
    low = (out or "").lower().replace(",", "")
    facts = {n for n in _numbers(src)} | {w.lower() for w in _NAME_RE.findall(src)}
    kept = sum(1 for f in facts if f.lower() in low) / len(facts) if facts else 1.0
    src_nums = _numbers(src)
    invented = sorted(n for n in _numbers(out) - src_nums
                      if len(n.replace(".", "")) >= 2 and
                      not any(n in s for s in src_nums))
    return (kept >= min_keep and not invented), kept, invented


class Llama:
    """The indexing model: llm_ollama.complete() on SITE_INDEX_MODEL with
    SITE_INDEX_NUM_CTX, temperature 0. Every answer is cached by the hash
    of (model, prompt) in llm_cache.json."""

    def __init__(self, cache_path, force=False, client=None):
        self.cache_path = cache_path
        self.cache = {} if force else (_read_json(cache_path, {}) or {})
        self.used = set()
        self.calls = self.hits = self.unsaved = 0
        self.model = _cfg("SITE_INDEX_MODEL")
        self.client = client

    def _complete(self, prompt, max_tokens):
        if self.client is not None:              # tests
            return self.client(prompt, max_tokens)
        import llm_ollama
        return llm_ollama.complete(
            [{"role": "user", "content": prompt}], self.model,
            {"num_ctx": _cfg("SITE_INDEX_NUM_CTX"), "temperature": 0,
             "seed": 0, "num_predict": max_tokens})

    def ask(self, prompt, max_tokens):
        key = hashlib.sha1(f"{self.model}\n{prompt}".encode("utf-8")).hexdigest()
        self.used.add(key)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        out = self._complete(prompt, max_tokens)
        self.calls += 1
        self.cache[key] = out
        self.unsaved += 1
        if self.unsaved >= SAVE_CACHE_EVERY:
            self.save()
        return out

    def save(self, prune=False):
        """Write the cache; prune=True (end of a full run) keeps only the
        answers this run used, so it never grows week after week."""
        if prune:
            self.cache = {k: v for k, v in self.cache.items() if k in self.used}
        _write_json(self.cache_path, self.cache)
        self.unsaved = 0


def _strip_wrapper(out):
    """Llama sometimes frames its notes ("Here are the notes:", ``` fences)."""
    lines = [l for l in (out or "").strip().split("\n")
             if not l.strip().startswith("```")]
    if lines and re.match(r"(?i)^(here (are|is)|sure|below)\b.*:\s*$",
                          lines[0].strip()):
        lines = lines[1:]
    return "\n".join(lines).strip()


def organize_prose(title, text, llm, stats):
    """Llama's compact notes for one prose source (pieces of
    SITE_LLM_CHUNK_WORDS), each fact-checked; a piece that fails keeps its
    cleaned text instead. Returns the notes ("" when nothing useful)."""
    text = clean_text(text)
    if _words(text) < 8:
        return text
    lines = text.count("\n") + 1
    if lines >= FRAGMENT_MIN_LINES and _words(text) / lines < FRAGMENT_LINE_WORDS:
        # a PDF table read column by column (dates on some lines, events
        # on others): Llama can't put it back together, and its notes would
        # fail the fact check anyway — the cleaned text, without a call
        stats["fragmented"] += 1
        return text
    parts = []
    for piece in _chunks(text, _cfg("SITE_LLM_CHUNK_WORDS")):
        prompt = _cfg("SITE_ORGANIZE_PROMPT").format(title=title, text=piece)
        max_tokens = min(2048, int(_words(piece) * 2.2) + 64)
        try:
            out = _strip_wrapper(llm.ask(prompt, max_tokens))
        except Exception as e:                   # noqa: BLE001
            print(f"[index] Llama failed on {title!r}: {e} — kept as text")
            stats["failed"] += 1
            parts.append(piece)
            continue
        if out.strip().upper().rstrip(".") == "NONE":
            facts = len(_numbers(piece)) + len(set(_NAME_RE.findall(piece)))
            if facts < 8:                        # really nothing in it
                stats["empty"] += 1
                continue
            stats["fallback"] += 1               # "NONE" for real facts:
            parts.append(piece)                  # keep them
            continue
        ok, kept, invented = fact_check(piece, out, _cfg("SITE_MIN_FACT_KEEP"))
        if ok:
            stats["organized"] += 1
            parts.append(out)
        else:
            stats["fallback"] += 1
            console.detail(f"[index] kept text of {title!r}: Llama kept "
                           f"{kept:.0%} of its facts"
                           + (f", invented {invented[:3]}" if invented else ""))
            parts.append(piece)
    return "\n".join(p for p in parts if p.strip())


def _notes_lines(text):
    """Organized text as page-file lines: headings stay, everything else
    becomes a "- " line."""
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.lstrip("-*• ").strip().upper().rstrip(".") == "NONE":
            continue
        if line.startswith("#"):
            # a heading INSIDE a source sits one level below the source's own
            out.append("### " + line.lstrip("#").strip())
        elif line.startswith(("- ", "* ", "• ")):
            out.append("- " + line[2:].strip())
        else:
            out.append("- " + line)
    # a heading left with nothing under it (its only line was "NONE")
    return [l for i, l in enumerate(out)
            if not (l.startswith("#") and (i + 1 == len(out)
                                           or out[i + 1].startswith("#")))]


def build_page(name, g, llm, stats, when):
    """The text of one page file (header + one "## " section per source)."""
    body, seen = [], []                        # seen: line sets of lists
    for s in g["sources"]:
        if s["kind"] == "list":
            lines = [l for l in dict.fromkeys(l.strip() for l in s["lines"])
                     if l]
            have = set(lines)
            if not lines or any(have <= old for old in seen):
                continue                       # "Regular staff" = "Regular
            seen.append(have)                  # teaching staff": once
            body += ["", f"## {short_title(s['title'])}",
                     *(f"- {l}" for l in lines)]
            continue
        notes = organize_prose(short_title(s["title"]), s["text"], llm, stats)
        lines = _notes_lines(notes)
        if _words("\n".join(l for l in lines if not l.startswith("#"))) < 3:
            continue
        body += ["", f"## {short_title(s['title'])}", *lines]
    if _words("\n".join(body)) < _cfg("SITE_MIN_PAGE_WORDS"):
        return ""
    head = [f"# {g['title']}",
            "Sources: " + ", ".join(g["urls"]),
            f"Updated: {when:%Y-%m-%d}"]
    return "\n".join(head + body).strip() + "\n"


# ── 4. Describe ───────────────────────────────────────────────────────────────

def parse_page(text):
    """(title, [(heading, [lines])]) of a page file. Header lines
    (Sources / Updated) are not content."""
    title, sections = "", []
    cur = None
    for line in (text or "").split("\n"):
        if line.startswith("# ") and not title:
            title = line[2:].strip()
            continue
        if line.startswith(("Sources: ", "Updated: ")) and cur is None:
            continue
        if line.startswith("## "):
            cur = (line[3:].strip(), [])
            sections.append(cur)
            continue
        if line.strip():
            if cur is None:
                cur = ("", [])
                sections.append(cur)
            cur[1].append(line.rstrip())
    return title, sections


def describe(name, title, text, llm):
    """One line saying what the page file holds (Llama), or its title and
    section headings when Llama gives nothing usable."""
    _, sections = parse_page(text)
    heads = [h for h, _ in sections if h]
    fallback = f"{title}: " + "; ".join(heads[:8]) if heads else title
    body = " ".join(text.split()[:DESCRIBE_WORDS])
    prompt = _cfg("SITE_DESCRIBE_PROMPT").format(
        title=title, headings="; ".join(heads[:30]), text=body)
    try:
        out = llm.ask(prompt, 80)
    except Exception as e:                       # noqa: BLE001
        print(f"[index] describe {name} failed: {e}")
        return fallback
    line = " ".join(_strip_wrapper(out).split()).strip(' "\'')
    line = re.sub(r"(?i)^(description\s*[:\-]\s*)?((this|the) (file|notes?|"
                  r"page)( also)? (holds|contains|has|covers|includes|"
                  r"provides|lists)( information| details| facts)?"
                  r"( on| about| regarding| of)?\s*)?", "", line).strip()
    line = line[:1].upper() + line[1:]
    if _words(line) < 4:
        return fallback
    words = line.split()
    if len(words) > DESCRIBE_MAX_WORDS:
        line = " ".join(words[:DESCRIBE_MAX_WORDS]).rstrip(",;") + "…"
    return line


# ── The digest: every page file's key facts in ONE file ─────────────────────

DIGEST_READ_WORDS = 1500  # words of a page Llama reads to pick its key facts


def _digest_budget(name):
    """SITE_DIGEST_WORDS for a page file: the longest matching prefix."""
    table = _cfg("SITE_DIGEST_WORDS")
    best = max((p for p in table if name.startswith(p)), key=len, default="")
    return table.get(best, 80)


def _cut_words(lines, budget):
    out, n = [], 0
    for line in lines:
        w = _words(line)
        if n + w > budget and out:
            break
        out.append(line)
        n += w
    return out


def build_digest(texts, entries, llm):
    """digest.txt: for each page file (Luna's department first) its key
    facts in SITE_DIGEST_WORDS — the page's own lines when it is that short
    already, else Llama's pick (SITE_DIGEST_PROMPT; a line with a number
    the page lacks is dropped) — the whole cut at SITE_DIGEST_MAX_WORDS.
    brain sends it with every question in digest mode (FETCH_MODE)."""
    rank = {n: i for i, (n, _, _) in enumerate(_cfg("SITE_PAGES"))}
    names = sorted(texts, key=lambda n: (rank.get(n, len(rank)), n))
    parts, total = [], 0
    for name in names:
        title, text = entries[name]["title"], texts[name]
        budget = _digest_budget(name)
        _, sections = parse_page(text)
        own = [l for _, ls in sections for l in ls if l.startswith("- ")]
        if name == _cfg("SITE_STUDENTS_PAGE"):
            lines = [l.split(" (")[0] for l in own if "Number of" in l]
        elif sum(_words(l) for l in own) <= budget:
            lines = own
        else:
            body = " ".join(text.split()[:DIGEST_READ_WORDS])
            try:
                out = llm.ask(_cfg("SITE_DIGEST_PROMPT").format(
                    title=title, words=budget, text=body), budget * 3)
            except Exception as e:               # noqa: BLE001
                print(f"[index] digest of {name} failed: {e}")
                out = ""
            have = _numbers(text)
            lines = [l for l in _notes_lines(_strip_wrapper(out))
                     if l.startswith("- ") and not (_numbers(l) - have)]
            lines = lines or own
        lines = _cut_words(lines, min(budget, _cfg("SITE_DIGEST_MAX_WORDS") - total))
        if not lines:
            continue
        total += sum(_words(l) for l in lines)
        parts.append(f"## {title}\n" + "\n".join(lines))
    return "\n\n".join(parts) + "\n"


# ── FAQ: the questions each page answers, answered once, here ────────────────

FAQ_WORDS = 900           # words of a page Llama reads to write its FAQ
_QA = re.compile(r"^\s*(?:\d+[.)]\s*)?Q\s*[:.]\s*(?P<q>.+?)\s*(?:\n\s*)?A\s*[:.]"
                 r"\s*(?P<a>.+?)\s*$", re.M | re.I)


def faq(name, title, text, llm):
    """Up to SITE_FAQ_PER_PAGE (question, answer, page) a visitor may ask
    about this page file, written by Llama. An answer with a number the
    page does not have, or of a size unfit to be spoken, is dropped. brain
    answers a close match to one of these at once, with no LLM call."""
    n = _cfg("SITE_FAQ_PER_PAGE")
    if not n:
        return []
    body = " ".join(text.split()[:FAQ_WORDS])
    try:
        out = llm.ask(_cfg("SITE_FAQ_PROMPT").format(title=title, n=n,
                                                     text=body), 700)
    except Exception as e:                       # noqa: BLE001
        print(f"[index] FAQ for {name} failed: {e}")
        return []
    have, seen, got = _numbers(text), set(), []
    for m in _QA.finditer(out or ""):
        q, a = " ".join(m["q"].split()), " ".join(m["a"].split())
        if (not 3 <= _words(q) <= 25 or not 3 <= _words(a) <= 60
                or q.lower() in seen or _numbers(a) - have):
            continue
        seen.add(q.lower())
        got.append([q.rstrip("?") + "?", a, name])
    return got[:n]


# ── Name index + staff (for fetch.py's routing hints and people.py) ───────────

_TITLED = re.compile(r"\b(?:Dr|Prof|Mr|Mrs|Ms|Smt|Sri|Shri)\.?\s+"
                     r"((?:[A-Z][A-Za-z.'-]*\s?){1,4})")
_NAME_STOP = frozenset("""the and for with from this that department college
engineering professor assistant associate science computer head office
student students faculty staff research school institute university""".split())


def name_words(name):
    return {w.lower() for w in re.findall(r"[A-Za-z]{4,}", name or "")
            if w.lower() not in _NAME_STOP}


def name_index(pages):
    """{name word: [page names]} from the titled names ("Dr. Manu J Pillai")
    and student rows ("Pranav P (CSE 2K23A, ...)") in the page files. A
    word on more than NAME_MAX_PAGES files is dropped as too common."""
    idx = {}
    for page, text in pages.items():
        found = set()
        for m in _TITLED.finditer(text):
            found |= name_words(m.group(1))
        if page == _cfg("SITE_STUDENTS_PAGE"):
            for line in text.split("\n"):
                if line.startswith("- ") and " — " in line:
                    found |= name_words(line[2:].split(" — ", 1)[0])
        for w in found:
            idx.setdefault(w, []).append(page)
    return {w: sorted(p) for w, p in sorted(idx.items())
            if len(p) <= NAME_MAX_PAGES}


def staff_sections(crawl):
    """[(section title, text)] of the website's staff lists (ETLab data
    sections named "...staff" or "hod") — people.py's directory."""
    out, seen = [], set()
    for d in crawl.get("docs") or ():
        frag = _fragment(d["url"])
        if frag and ("staff" in frag or frag == "hod") and d["text"] not in seen:
            seen.add(d["text"])
            out.append([d.get("title") or "", d["text"]])
    return out


# ── The run ───────────────────────────────────────────────────────────────────

def _load_crawl(offline):
    p = paths()["crawl"]
    if offline:
        data = _read_json(p)
        if not data:
            raise RuntimeError(f"no {p} yet — run once without --offline")
        print(f"[index] using the crawl of "
              f"{datetime.fromtimestamp(data['fetched_at']):%Y-%m-%d %H:%M}")
        return data
    import site_crawler
    t0 = time.time()
    print("[index] 1/4 crawling the website (a few minutes)…")
    data = site_crawler.crawl()
    _write_json(p, data)
    print(f"[index]     {len(data['docs'])} pieces from "
          f"{len(data['catalog'])} pages in {time.time() - t0:.0f}s")
    return data


def plan(groups):
    """Printable list of the page files and their input size."""
    rows = []
    for name, g in sorted(groups.items()):
        n = sum(_words(s.get("text")) + sum(_words(l) for l in s.get("lines", ()))
                for s in g["sources"])
        prose = sum(_words(s.get("text")) for s in g["sources"]
                    if s["kind"] == "prose")
        rows.append(f"  {name:28} {n:7} words ({prose} to Llama)  "
                    f"{len(g['urls'])} page(s)  {g['title'][:50]}")
    return "\n".join(rows)


def run(offline=False, force=False, only=None, show_plan=False, client=None):
    """Indexing(). Returns the catalog dict it wrote (None for --plan)."""
    p = paths()
    crawl = _load_crawl(offline)
    groups, report = group(crawl)
    print(f"[index] 2/4 grouped into {len(groups)} page files "
          f"({len(report['dropped'])} pieces dropped or cut)")
    if show_plan:
        print(plan(groups))
        for url, why in report["dropped"]:
            console.detail(f"    dropped {url} — {why}")
        return None
    llm = Llama(p["cache"], force=force, client=client)
    stats = dict(organized=0, fallback=0, empty=0, failed=0, fragmented=0)
    when = datetime.fromtimestamp(crawl.get("fetched_at") or time.time())
    os.makedirs(p["pages"], exist_ok=True)
    old = _read_json(p["catalog"], {}) or {}
    texts, entries = {}, {}
    # SITE_PAGES order (Luna's department first), then the rest: the
    # files that matter most are ready first if a run is cut short
    rank = {n: i for i, (n, _, _) in enumerate(_cfg("SITE_PAGES"))}
    names = sorted(groups, key=lambda n: (rank.get(n, len(rank)), n))
    if only:
        names = [n for n in names if n in only]
    t0 = time.time()
    print(f"[index] 3/4 organizing {len(names)} page files with "
          f"{llm.model} (cached answers are reused)…")
    for i, name in enumerate(names, 1):
        g = groups[name]
        t1 = time.time()
        text = build_page(name, g, llm, stats, when)
        if not text:
            report["dropped"].append((name, "too little left to be useful"))
            continue
        texts[name] = text
        with open(os.path.join(p["pages"], f"{name}.txt"), "w",
                  encoding="utf-8") as f:
            f.write(text)
        print(f"[index]     {i:3}/{len(names)} {name:28} "
              f"{_words(text):6} words  {time.time() - t1:5.1f}s")
    if only:                                   # keep the other pages as they are
        for name, e in (old.get("pages") or {}).items():
            if name not in texts and name != _cfg("FETCH_KB_PAGE"):
                path = os.path.join(p["pages"], e["file"])
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as f:
                        texts[name] = f.read()
                    entries[name] = e
    print(f"[index]     Llama: {stats['organized']} pieces organized, "
          f"{stats['fallback']} kept as text (fact check), "
          f"{stats['fragmented']} broken-up tables kept as text, "
          f"{stats['empty']} empty, {stats['failed']} failed; {llm.calls} new calls, "
          f"{llm.hits} cached ({time.time() - t0:.0f}s)")

    print("[index] 4/4 describing each page file…")
    for name in sorted(texts):
        if name in entries:                    # --only: untouched page
            continue
        g = groups[name]
        title = parse_page(texts[name])[0] or g["title"]
        entries[name] = {
            "title": title, "file": f"{name}.txt",
            "description": describe(name, title, texts[name], llm),
            "words": _words(texts[name]),
            "list": name == _cfg("SITE_STUDENTS_PAGE"),
            "sources": g["urls"]}
    print("[index]     … and the digest of them all (one file)")
    digest = build_digest(texts, entries, llm)
    with open(os.path.join(site_dir(), "digest.txt"), "w", encoding="utf-8") as f:
        f.write(digest)
    print(f"[index]     digest: {_words(digest)} words "
          f"(~{int(_words(digest) * 1.4)} tokens)")
    faqs = []
    if _cfg("SITE_FAQ_PER_PAGE"):
        print("[index]     … and the questions each one answers (FAQ)")
        for name in sorted(texts):
            if name != _cfg("SITE_STUDENTS_PAGE"):
                faqs += faq(name, entries[name]["title"], texts[name], llm)
    entries[_cfg("FETCH_KB_PAGE")] = {
        "title": "Luna's knowledge base", "file": _cfg("KNOWLEDGE_PATH"),
        "description": _cfg("FETCH_KB_DESCRIPTION"), "words": 0,
        "list": False, "sources": [_cfg("KNOWLEDGE_PATH")]}
    llm.save(prune=not only)

    catalog = {
        "version": CATALOG_VERSION, "built_at": time.time(),
        "crawled_at": crawl.get("fetched_at"), "model": llm.model,
        "pages": dict(sorted(entries.items())),
        "names": name_index(texts),
        "staff": staff_sections(crawl),
        "faq": faqs,
        "dropped": report["dropped"],
    }
    _write_json(p["catalog"], catalog)
    keep = {e["file"] for e in entries.values()}
    for f in os.listdir(p["pages"]):           # page files no longer built
        if f.endswith(".txt") and f not in keep:
            os.remove(os.path.join(p["pages"], f))
    total = sum(e["words"] for e in entries.values())
    print(f"[index] done: {len(entries)} pages ({total} words) in "
          f"{p['pages']}, catalog {p['catalog']}")
    return catalog


def status():
    """Printable summary of what is indexed."""
    p = paths()
    cat = _read_json(p["catalog"])
    if not cat:
        return f"No index yet ({p['catalog']}). Run: python site_index.py"
    built = datetime.fromtimestamp(cat["built_at"])
    crawled = datetime.fromtimestamp(cat.get("crawled_at") or cat["built_at"])
    lines = [f"Index built {built:%Y-%m-%d %H:%M} by {cat.get('model')}, "
             f"website crawled {crawled:%Y-%m-%d %H:%M}",
             f"{len(cat['pages'])} pages, {len(cat.get('names') or {})} name "
             f"words, {len(cat.get('staff') or [])} staff lists", ""]
    for name, e in cat["pages"].items():
        lines.append(f"  {name:28} {e['words']:6} words  {e['description']}")
    dropped = cat.get("dropped") or []
    if dropped:
        lines += ["", f"{len(dropped)} pieces dropped or cut "
                      f"(python site_index.py --plan -v lists them)"]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Indexing(): the college website "
                                 "-> Luna's page files (docs/Fetch Algo.md)")
    ap.add_argument("--offline", action="store_true",
                    help="skip the crawl, re-use the last crawl.json")
    ap.add_argument("--force", action="store_true",
                    help="ignore cached Llama answers (after a prompt change)")
    ap.add_argument("--plan", action="store_true",
                    help="only show the page files that would be built")
    ap.add_argument("--only", nargs="+", metavar="PAGE",
                    help="rebuild just these page files")
    ap.add_argument("--status", action="store_true",
                    help="show what is indexed now")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.verbose:
        console.LOG_VERBOSE = True
    if args.status:
        print(status())
        sys.exit(0)
    try:
        run(offline=args.offline, force=args.force, only=args.only,
            show_plan=args.plan)
    except KeyboardInterrupt:
        print("\n[index] stopped — Llama answers so far are cached; "
              "re-run to continue")
        sys.exit(1)
    except RuntimeError as e:
        print(f"[index] failed: {e}")
        sys.exit(1)
