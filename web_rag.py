# web_rag.py
"""
web_rag.py — the college website as a knowledge source that keeps working
offline.

Luna is offline-forever by design, but the college website already holds most
of what visitors ask about (principal, admissions, exams, the department's
staff). When the Pi HAPPENS to be online this module reads it, and from then on
answers come from a cache on disk (<cache_dir>/<host>.json + .npz) with no
network at all. How:

  1. CATALOG. The site's menus ARE its topic map: every link sits under a menu
     path ("ACADEMICS > Admissions > B.Tech Lateral Entry", "People > Staff").
     discover() reads the menus of the seed pages (the home page and the focus
     department's page) and keeps only the general college pages and the focus
     department's pages; other departments, files and other sites are skipped,
     each with a reason (verify()).
  2. FETCH. refresh() downloads every catalog page in parallel, verifies it
     (broken / not a web page / duplicate), and indexes its text passages, its
     table rows ("Name: Dr. A; Department: Mechanical") and the JSON data the
     page loads with JavaScript (department staff lists — the names never
     appear in the HTML). Student rosters, e-mails and phone numbers are left
     out. Everything is embedded with brain.py's sentence transformer.
  3. ROUTE. A question is matched against the catalog's TOPICS first — one per
     page (its menu path, link text, title) and one per data section ("Computer
     Science and Engineering — Non teaching staff"). Only the passages of the
     best-matching topics are searched (route()), so an answer comes from the
     page that is about the question, not from whatever page shares its words.
     lookup() fetches a routed page live if it isn't cached yet.

Everything degrades instead of failing:
  - no WEB_RAG_URL              -> brain.py never calls us
  - offline + no cache          -> available() False, retrieve() -> []
  - offline + cache             -> answers from the cache (no live fetch)
  - a refresh fails             -> old index kept, last_error set
  - a live fetch is slow        -> answered from the cache; the page is cached
                                   when it lands
  - embedder missing            -> retrieve() -> []
start() never raises and does all network work in ONE daemon thread, after a
short socket probe, so boot never blocks on a flaky Wi-Fi.

The runtime path gets brain.py's embed() and must NOT import
sentence_transformers itself (one model copy in RAM); only the CLI in __main__
builds its own embedder:

    python web_rag.py --verify                 # catalog + link check, no index
    python web_rag.py --refresh                # fetch everything, rebuild cache
    python web_rag.py --route "exam calendar"  # which pages a question goes to
    python web_rag.py --query "who are the non teaching staff"
    python web_rag.py --status
"""

import concurrent.futures
import gzip
import html as htmllib
import json
import os
import re
import zlib
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from html.parser import HTMLParser

import numpy as np

_DEFAULTS = dict(                       # used when config.py lacks a setting
    WEB_RAG_URL="", WEB_RAG_SEEDS=(), WEB_RAG_FOCUS_NAME="",
    WEB_RAG_FOCUS_SLUGS=(), WEB_RAG_SKIP_SECTIONS=(), WEB_RAG_SKIP_SLUGS=(),
    WEB_RAG_MAX_PAGES=250, WEB_RAG_FETCH_WORKERS=8, WEB_RAG_CHUNK_WORDS=120,
    WEB_RAG_TOP_K=3, WEB_RAG_MIN_SIM=0.35, WEB_RAG_ROUTE_TOPICS=5,
    WEB_RAG_ROUTE_MIN=0.35, WEB_RAG_CACHE_DIR="data/rag_cache",
    WEB_RAG_REFRESH_HOURS=24, WEB_RAG_FETCH_TIMEOUT=8.0,
    WEB_RAG_ONLINE_TIMEOUT=1.5, WEB_RAG_LIVE_PAGES=2, WEB_RAG_LIVE_TIMEOUT=6.0,
)
try:
    import config as _config
except Exception:                       # standalone use without config.py
    _config = None


def _cfg(name):
    return getattr(_config, name, _DEFAULTS[name])


USER_AGENT = "LunaBot/1.0 (+offline desk robot)"

# Not in config.py (module-level tunables):
CHUNK_OVERLAP_WORDS = 20           # words shared by neighbouring chunks
MAX_PAGE_BYTES      = 2_000_000    # never read more than this from one page
MIN_CHUNK_WORDS     = 8            # a trailing stub shorter than this is dropped
MIN_PAGE_WORDS      = 10           # a page with fewer words carries no facts
EMBED_BATCH         = 64           # chunks per embed_fn() call during a refresh
EMBED_BATCH_PAUSE_S = 0.05         # breather between batches so the daemon
                                   # thread doesn't starve vosk/LLM on the Pi
MAX_DATA_ENDPOINTS  = 40           # JSON endpoints fetched per refresh
JSON_MAX_ITEMS      = 40           # list entries kept per JSON section
JSON_ITEM_WORDS     = 40           # words kept per JSON text value
DATA_EMBED_WORDS    = 12           # words of a data section that get embedded
THIN_PAGE_WORDS     = 60           # a page that loads its content as data and
                                   # has fewer words of its own is just a shell
TABLE_MAX_ROWS      = 60           # rows indexed one by one per table
MAX_HITS_PER_PAGE   = 2            # retrieve(): rows/sections per page; its
                                   # plain text gets 1 slot, so one long page
                                   # can't fill the top-K with near-duplicates
KEYWORD_BONUS       = 0.05         # routing: per question word in a topic's name
PASSAGE_KEYWORD_BONUS = 0.05       # per question word found in a passage
                                   # ("mechanical" picks the Mechanical row
                                   # of the HODs table over the CSE HOD)
GROUP_FACTOR        = 0.85         # a group sibling's share of a routed topic's score
GENERIC_TITLE_PAGES = 3            # a <title> on this many pages is generic
BOILERPLATE_PAGES   = 5            # a text line on this many pages is template
ROUTE_WEIGHT        = 0.5          # final score = passage match + this x topic
                                   # match, / (1 + this)
PASSAGE_QUERY       = "raw"        # passages are matched by the question as
                                   # asked ("raw"), the spelled-out one ("cse"
                                   # -> "computer science and engineering",
                                   # "expanded") or the better of both ("max")
ROUTE_ALSO_TOP      = 0            # also consider the N best passages outside
                                   # the routed topics (0 = routed only)
RELATIVE_CUT        = 0.12         # drop passages this far below the best one:
                                   # near-misses only confuse the small LLM
SPELL_MIN_LEN       = 6            # shorter words are never "corrected"
SPELL_CUTOFF        = 0.82         # difflib ratio a correction must reach
SPELL_SAME_STEM     = 0.85         # ... when the word starts like a site word
CACHE_VERSION       = 2            # bump when the index layout changes
ONLINE_RECHECK_S    = 60.0         # lookup() re-probes connectivity this often

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Module-level settings: read from config at import, overridable by tests and
# the CLI through configure(). One dict so nothing here reads config later.
_settings = dict(
    url=_cfg("WEB_RAG_URL"),
    seeds=list(_cfg("WEB_RAG_SEEDS")),
    focus_name=_cfg("WEB_RAG_FOCUS_NAME"),
    focus_slugs=list(_cfg("WEB_RAG_FOCUS_SLUGS")),
    skip_sections=list(_cfg("WEB_RAG_SKIP_SECTIONS")),
    skip_slugs=list(_cfg("WEB_RAG_SKIP_SLUGS")),
    cache_dir=_cfg("WEB_RAG_CACHE_DIR"),
    max_pages=_cfg("WEB_RAG_MAX_PAGES"),
    workers=_cfg("WEB_RAG_FETCH_WORKERS"),
    chunk_words=_cfg("WEB_RAG_CHUNK_WORDS"),
    chunk_overlap=CHUNK_OVERLAP_WORDS,
    top_k=_cfg("WEB_RAG_TOP_K"),
    min_sim=_cfg("WEB_RAG_MIN_SIM"),
    route_topics=_cfg("WEB_RAG_ROUTE_TOPICS"),
    route_min=_cfg("WEB_RAG_ROUTE_MIN"),
    refresh_hours=_cfg("WEB_RAG_REFRESH_HOURS"),
    fetch_timeout=_cfg("WEB_RAG_FETCH_TIMEOUT"),
    online_timeout=_cfg("WEB_RAG_ONLINE_TIMEOUT"),
    live_pages=_cfg("WEB_RAG_LIVE_PAGES"),
    live_timeout=_cfg("WEB_RAG_LIVE_TIMEOUT"),
)

_lock         = threading.Lock()   # guards _index swaps and reads
_refresh_lock = threading.Lock()   # one refresh at a time
_live_lock    = threading.Lock()   # one live fetch at a time
_save_lock    = threading.Lock()   # refresh and live fetch both write the cache
_embed_fn     = None
_index        = None               # {url, host, fetched_at, pages, chunks,
                                   #  matrix, links (= topics), link_matrix,
                                   #  report}
_thread       = None
_online       = None               # None = not probed yet
_online_at    = 0.0                # when _online was last probed
_robots       = ([], [])           # the site's rules, from the last refresh
_tried        = set()              # live-fetched URLs that gave nothing
_last_error   = None
_dim_warned   = False              # retrieve() logs a dim mismatch once

_SKIP_EXT = {
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".bmp",
    ".tif", ".tiff", ".zip", ".rar", ".gz", ".tgz", ".tar", ".7z", ".mp3",
    ".mp4", ".m4a", ".avi", ".mov", ".wmv", ".mkv", ".webm", ".doc", ".docx",
    ".xls", ".xlsx", ".ppt", ".pptx", ".css", ".js", ".json", ".xml", ".rss",
    ".exe", ".dmg", ".apk", ".iso", ".woff", ".woff2", ".ttf", ".eot", ".otf",
}
_SKIP_SCHEMES = ("javascript:", "mailto:", "tel:", "data:", "ftp:")
_HTML_TYPES   = ("text/html", "application/xhtml+xml")

# URL slugs are terse ("nts-cse.php"); spelling the common college ones out
# lets a spoken question ("non teaching staff in computer science") find the
# link even when its anchor text is just "Staff".
_SLUG_EXPANSIONS = {
    "cse": "computer science and engineering",
    "cs": "computer science",
    "ece": "electronics and communication engineering",
    "eee": "electrical and electronics engineering",
    "me": "mechanical engineering", "mech": "mechanical engineering",
    "ce": "civil engineering", "civil": "civil engineering",
    "chem": "chemical engineering", "arch": "architecture",
    "mca": "computer applications MCA",
    "nts": "non teaching staff", "hod": "head of department HOD",
    "hods": "heads of department HODs", "ug": "undergraduate",
    "pg": "postgraduate", "phd": "PhD research", "rs": "research scholars",
    "btech": "BTech", "mtech": "MTech", "faq": "frequently asked questions",
    "nss": "national service scheme NSS", "iqac": "quality assurance IQAC",
}


def configure(**kw):
    """Override settings (tests / CLI). Unknown keys raise so typos surface.
    Changing the url drops the in-memory index — it belonged to another site."""
    global _index
    for k in kw:
        if k not in _settings:
            raise ValueError(f"web_rag: unknown setting {k!r}")
    global _online
    url_changed = "url" in kw and kw["url"] != _settings["url"]
    _settings.update(kw)
    if url_changed:
        with _lock:
            _index = None
        _tried.clear()
        _online = None                    # that answer was about another host


# ── URL helpers ───────────────────────────────────────────────────────────────

def _host_of(url):
    return (urllib.parse.urlsplit(url).netloc or "").lower()


def _same_site(host_a, host_b):
    """Same host, treating 'www.' as decoration (sites mix the two freely)."""
    strip = lambda h: h[4:] if h.startswith("www.") else h
    return bool(host_a) and strip(host_a) == strip(host_b)


def _normalize(url):
    """Canonical form: no fragment, lower-case scheme/host, quoted path so a
    space in an href never produces a malformed request line."""
    parts = urllib.parse.urlsplit(url)
    path  = urllib.parse.quote(parts.path or "/", safe="/%:@&=+$,;~!*'()")
    query = urllib.parse.quote(parts.query, safe="/%:@&=+$,;~!*'()?")
    return urllib.parse.urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def _looks_binary(url):
    path = urllib.parse.urlsplit(url).path.lower()
    return os.path.splitext(path)[1] in _SKIP_EXT


def _cache_paths():
    host = re.sub(r"[^A-Za-z0-9.-]+", "_", _host_of(_settings["url"]) or "site")
    d = _settings["cache_dir"]
    if not os.path.isabs(d):
        d = os.path.join(_PROJECT_DIR, d)
    return os.path.join(d, host + ".json"), os.path.join(d, host + ".npz")


# ── HTML -> text (stdlib only) ────────────────────────────────────────────────

_SKIP_TAGS  = {"script", "style", "noscript", "svg", "nav", "footer", "header",
               "form", "template", "iframe"}
# Page furniture (menus, mastheads, footers, search forms) is skipped — but
# HTML5 puts headings in <article><header>, and sloppy pages never close a
# <nav>. So these "soft" skips are dropped again when a content tag starts
# (<main> can't legally sit inside any of them), and header/footer are not
# skipped at all while inside a content tag (nav/form there are still menus).
_SOFT_SKIP    = {"nav", "footer", "header", "form"}
_SECTION_PART = {"header", "footer"}
_CONTENT_TAGS = {"main", "article", "section"}
# Table cells are NOT blocks: a row stays one line, "Dr. A | Mechanical |
# …", so a name keeps its department once the text is chunked.
_CELL_TAGS  = {"td", "th"}
_BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4",
               "h5", "h6", "tr", "table", "section", "article",
               "aside", "main", "blockquote", "pre", "hr", "dt", "dd", "dl",
               "figure", "figcaption", "address", "details", "summary",
               "body", "title", "nav", "header", "footer", "form"}


_CF_EMAIL_RE = re.compile(r"\[email\s*protected\]", re.I)
# The site abbreviates ("Mechanical Engg. Dept"); spelled out, a row says
# what a question says ("Mechanical Engineering Department") — for the
# embedder and for the small LLM, which matches on surface wording.
_ABBREVIATIONS = ((re.compile(r"\bEngg\b\.?"), "Engineering"),
                  (re.compile(r"\bDept\b\.?"), "Department"))


def _unabbreviate(text):
    for rx, full in _ABBREVIATIONS:
        text = rx.sub(full, text)
    return text


class _TextExtractor(HTMLParser):
    """Collects visible text, the <title>, every href, each link's anchor
    text (menus included — that is where a site's map lives) and the inline
    <script> bodies (data endpoints). Unclosed tags are tolerated: only the
    skip-tags need matching end tags, a content tag (<main>/<article>/
    <section>) ends any soft skip, and </body> or </html> resets any skip
    that a sloppy page never closed."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip  = []
        self._content = []              # open main/article/section tags
        self._title = []
        self._in_title = False
        self._anchor = None             # [href, [text parts]] inside an <a>
        self._tables = []               # open <table>s: [rows]
        self._heads = []                # per open table: first row all <th>?
        self._row = None                # cells of the open <tr>
        self._row_th = True             # every cell of the open row is <th>
        self._cell = None               # text parts of the open <td>/<th>
        self.parts = []
        self.links = []
        self.anchors = []               # (href, anchor text)
        self.scripts = []               # inline <script> bodies
        self.tables = []                # finished tables: [[cell, ...], ...]
        self.table_heads = []           # per finished table: has a header row

    def _close_cell(self):
        if self._cell is not None and self._row is not None:
            self._row.append(_unabbreviate(_CF_EMAIL_RE.sub(
                "", " ".join("".join(self._cell).split())).strip()))
        self._cell = None

    def _close_row(self):
        self._close_cell()
        if self._row is not None and self._tables:
            if any(self._row):
                if self._heads[-1] is None:
                    self._heads[-1] = self._row_th
                self._tables[-1].append(self._row)
        self._row = None

    def _close_table(self):
        t, head = self._tables.pop(), self._heads.pop()
        if t:
            self.tables.append(t)
            self.table_heads.append(bool(head))

    def close_tables(self):
        self._close_row()
        while self._tables:
            self._close_table()

    def close_anchor(self):
        if self._anchor is not None:
            href, parts = self._anchor
            self.anchors.append((href, " ".join("".join(parts).split())))
            self._anchor = None

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.close_anchor()          # <a> never closed before the next
        if tag in _CONTENT_TAGS:
            self._content.append(tag)
            self._skip = [t for t in self._skip if t not in _SOFT_SKIP]
        if tag in _SKIP_TAGS and not (tag in _SECTION_PART and self._content):
            self._skip.append(tag)
            return
        if tag == "title":
            self._in_title = True
        if tag == "table":
            self._tables.append([])
            self._heads.append(None)
        elif tag == "tr" and self._tables:
            self._close_row()
            self._row, self._row_th = [], True
        elif tag in _CELL_TAGS and self._row is not None:
            self._close_cell()
            self._cell = []
            if tag == "td":
                self._row_th = False
        if tag in ("a", "area"):
            a = dict(attrs)
            href = (a.get("href") or "").strip()
            if href:
                self.links.append(href)
                if tag == "a":
                    self._anchor = [href, [a.get("title") or ""]]
                else:
                    self.anchors.append((href, a.get("alt") or ""))
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")
        elif tag in _CELL_TAGS and not self._skip:
            self.parts.append(" | ")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if tag == "a":
            self.close_anchor()
        if tag in _CELL_TAGS:
            self._close_cell()
        elif tag == "tr":
            self._close_row()
        elif tag == "table" and self._tables:
            self._close_row()
            self._close_table()
        if tag in _SKIP_TAGS and tag in self._skip:
            while self._skip and self._skip.pop() != tag:
                pass
            return
        if tag in _CONTENT_TAGS and tag in self._content:
            while self._content and self._content.pop() != tag:
                pass
        if tag in ("body", "html"):
            self._skip.clear()
        if tag == "title":
            self._in_title = False
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self._title.append(data)
            return
        if self._anchor is not None:
            self._anchor[1].append(" " + data)
        if self._skip and self._skip[-1] == "script":
            self.scripts.append(data)
            return
        if not self._skip:              # source line breaks are just spaces;
            self.parts.append(data.replace("\n", " "))   # blocks make lines
            if self._cell is not None:
                self._cell.append(data)

    @property
    def title(self):
        return " ".join("".join(self._title).split())


def parse_page(html):
    """dict(title, text, links, anchors, scripts, tables) from an HTML
    document. Text keeps a newline at block boundaries and has all other
    whitespace collapsed; anchors are (href, anchor text) pairs; tables are
    lists of rows of cell texts."""
    p = _TextExtractor()
    try:
        p.feed(html)
        p.close()
    except Exception:                   # html.parser is lenient; be more so
        pass
    p.close_anchor()
    p.close_tables()
    raw   = _unabbreviate(_CF_EMAIL_RE.sub("", "".join(p.parts)))
    lines = [" ".join(l.split()).strip(" |") for l in raw.split("\n")]
    lines = [re.sub(r"(\|\s*)+\|", "|", l) for l in lines]   # empty cells
    text  = "\n".join(l for l in lines if l)
    return dict(title=p.title, text=text, links=p.links, anchors=p.anchors,
                scripts=p.scripts, tables=p.tables, table_heads=p.table_heads)


def html_to_text(html):
    """(title, text, links) from an HTML document — see parse_page()."""
    page = parse_page(html)
    return page["title"], page["text"], page["links"]


def chunk_text(text, chunk_words, overlap):
    """Split into chunk_words-word windows sharing `overlap` words with their
    neighbour, so a fact straddling a boundary still lands whole in one."""
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - max(0, overlap))
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_words]
        if len(piece) < MIN_CHUNK_WORDS and chunks:
            break                         # trailing stub: already covered
        chunks.append(" ".join(piece))
        if start + chunk_words >= len(words):
            break
    return chunks


# ── Link labels and query expansion ───────────────────────────────────────────────────────────

# Anchor texts that say nothing about the page they point to.
_JUNK_ANCHORS = {"read more", "more", "click here", "here", "view", "view more",
                 "view all", "details", "know more", "learn more", "link",
                 "open", "download", "go", "continue", "next", "previous"}


def link_label(url, anchor_texts=()):
    """What a link is about, for matching against a question: its anchor
    texts plus the URL slug spelled out ("Staff | non teaching staff
    computer science and engineering" for nts-cse.php)."""
    parts = urllib.parse.urlsplit(url)
    slug  = re.sub(r"\.(php|html?|aspx?|jsp)$", "", parts.path.lower())
    words = [w for w in re.split(r"[^a-z0-9]+", f"{slug} {parts.query.lower()}")
             if w and not w.isdigit() and w not in ("index", "www", "id",
                                                    "page", "slug")]
    slug_text = " ".join(_SLUG_EXPANSIONS.get(w, w) for w in words)
    anchors = " / ".join(dict.fromkeys(
        t for t in anchor_texts
        if t and t.lower().strip(" .>»…") not in _JUNK_ANCHORS))[:200]
    return " | ".join(s for s in (anchors, slug_text) if s) or url


def expand_query(text):
    """Spell out the college abbreviations a question uses ("hod of mech"
    -> "hod head of department HOD of mech mechanical engineering"): the
    embedder barely connects "hod" with "head of department"."""
    out = []
    for w in text.split():
        out.append(w)
        key = w.lower().strip(".,;:!?()'\"")
        if key in _SLUG_EXPANSIONS and len(key) > 2:
            out.append(_SLUG_EXPANSIONS[key])
    return " ".join(out)


# ── Data endpoints: pages that fill themselves in with JavaScript ─────────────

_JS_COMMENT_RE = re.compile(r"(?m)(?<![:'\"\\])//[^\n]*")
_JS_URL_RE     = re.compile(r"""\burl\s*:\s*(['"])([^'"\s]+)\1""")
_JS_CALL_RE    = re.compile(r"""(?:\bfetch|\$\.getJSON|\$\.get|\$\.post)\(\s*"""
                            r"""(['"])([^'"\s]+)\1""")
_JS_DATA_RE    = re.compile(r"""\bdata\s*:\s*\{([^{}]*)\}""")
_JS_NUM_RE     = re.compile(r"-?\d+(?:\.\d+)?")
_DATA_SKIP_EXT = _SKIP_EXT - {".json"}


def _js_params(body):
    """{key: value} from a JS object literal body like " id: 2, t: 'x' ",
    or None when a value is a variable (we can't know what it holds)."""
    params = {}
    for piece in body.split(","):
        if not piece.strip():
            continue
        if ":" not in piece:
            return None
        k, v = (s.strip() for s in piece.split(":", 1))
        k = k.strip("'\"")
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        elif not _JS_NUM_RE.fullmatch(v):
            return None
        params[k] = v
    return params


def find_data_endpoints(scripts, page_url):
    """Same-site URLs the page's inline scripts load data from — $.ajax({url:
    …, data: {…}}), fetch('…'), $.getJSON('…') — with literal data params
    folded into the query string, in order of appearance."""
    out, host = [], _host_of(page_url)
    for js in scripts:
        js = _JS_COMMENT_RE.sub("", js)
        datas = [(m.start(), m.group(1)) for m in _JS_DATA_RE.finditer(js)]
        for rx in (_JS_URL_RE, _JS_CALL_RE):
            for m in rx.finditer(js):
                raw = m.group(2)
                if raw.lower().startswith(_SKIP_SCHEMES) or raw.startswith("#"):
                    continue
                params = {}
                near = [d for d in datas if abs(d[0] - m.start()) <= 300]
                if near:                  # the data:{} of this same call
                    body = min(near, key=lambda d: abs(d[0] - m.start()))[1]
                    params = _js_params(body)
                    if params is None:
                        continue
                try:
                    url = urllib.parse.urljoin(page_url, raw)
                    if params:
                        url += (("&" if "?" in url else "?")
                                + urllib.parse.urlencode(params))
                    url = _normalize(url)
                except ValueError:
                    continue
                if not url.startswith(("http://", "https://")):
                    continue
                path = urllib.parse.urlsplit(url).path.lower()
                if os.path.splitext(path)[1] in _DATA_SKIP_EXT:
                    continue
                if _same_site(_host_of(url), host) and url not in out:
                    out.append(url)
    return out


# JSON fields that are plumbing or personal contact details, never facts to
# speak; "students" holds whole class rosters (children's names).
_JSON_SKIP_FIELDS = {
    "id", "sno", "slug", "photo", "image", "images", "img", "url", "link",
    "href", "file", "files", "status", "email", "phone", "mobile", "address",
    "password", "staff_id", "dept_id", "dept_name", "department", "type",
    "added_on", "valid_upto", "ending_on", "created_at", "updated_at",
    "joindate", "join_date", "students",
}
_JSON_SKIP_KEYS  = ("popup", "photo", "gallery", "image", "banner", "slider")
_JSON_LABEL_KEYS = ("dept_name", "department_name", "department", "dept")


def _humanize(key):
    words = re.split(r"[_\-\s]+", str(key).strip().lower())
    out = " ".join(_SLUG_EXPANSIONS.get(w, w) if w in ("hod", "hods", "nts",
                                                       "ug", "pg", "rs")
                   else w for w in words if w)
    return out[:1].upper() + out[1:]


def _json_scalar(v):
    """Readable text for one JSON leaf, or "" for ids, placeholders, file
    names, links and embedded record dumps."""
    if v is None or isinstance(v, bool):
        return ""
    if isinstance(v, (int, float)):
        return str(v)
    if not isinstance(v, str):
        return ""
    s = v.strip()
    low = s.lower()
    if (not s or low in ("none", "null", "undefined", "nan", "-")
            or s.startswith(("0000-00-00", "[", "{"))
            or low.startswith(("http://", "https://", "/uploads", "uploads/"))):
        return ""
    if " " not in s and os.path.splitext(low)[1] in _SKIP_EXT:
        return ""                                   # a bare file name
    if "<" in s and ">" in s:
        s = parse_page(s)["text"]
    words = s.split()
    if len(words) > JSON_ITEM_WORDS:
        words = words[:JSON_ITEM_WORDS] + ["…"]
    return " ".join(words)


def _json_item(v):
    """One list entry as a short phrase: "Dr. A B, Professor, Computer
    Science". Number-only values keep their field name ("count 77")."""
    if not isinstance(v, dict):
        return "" if isinstance(v, list) else _json_scalar(v)
    parts = []
    for k, x in v.items():
        k = str(k).lower()
        if (k in _JSON_SKIP_FIELDS or k.endswith(("_id", "_url", "photo"))
                or isinstance(x, (dict, list))):
            continue
        s = _json_scalar(x)
        if not s:
            continue
        if not re.search(r"[A-Za-z]", s):
            s = f"{_humanize(k).lower()} {s}"
        parts.append(s)
    parts = list(dict.fromkeys(parts))
    # "Dr." next to "Dr. A B": keep only the value that says more
    parts = [p for p in parts if not any(p != q and p in q for q in parts)]
    return ", ".join(parts)


def _json_label(data, depth=0):
    """The department (or similar) the JSON describes, for section titles."""
    if depth > 3:
        return ""
    if isinstance(data, dict):
        for k in _JSON_LABEL_KEYS:
            v = data.get(k)
            if isinstance(v, str) and v.strip() and v.strip().lower() != "none":
                v = " ".join(v.split())
                return v.title() if v.isupper() else v
        values = list(data.values())
    elif isinstance(data, list):
        values = data[:3]
    else:
        return ""
    for v in values:
        if isinstance(v, (dict, list)):
            found = _json_label(v, depth + 1)
            if found:
                return found
    return ""


def json_to_sections(data, url, fallback_title=""):
    """Flatten a JSON document into page-like sections [{url, title, text}]:
    one per list/object field ("Non teaching staff, 11 in total: …") plus an
    overview of the plain fields (vision, mission, counts)."""
    if isinstance(data, list):
        data = {"items": data}
    if not isinstance(data, dict):
        return []
    label = _json_label(data) or fallback_title

    def title(name):
        return f"{label} — {name}" if label else name

    sections, overview = [], []
    for key, val in data.items():
        k = str(key).lower()
        if k in _JSON_SKIP_FIELDS or any(s in k for s in _JSON_SKIP_KEYS):
            continue
        name = _humanize(key)
        if isinstance(val, list):
            items = [s for s in (_json_item(v) for v in val[:JSON_MAX_ITEMS])
                     if s]
            if not items:
                continue
            more = (f" (and {len(val) - JSON_MAX_ITEMS} more)"
                    if len(val) > JSON_MAX_ITEMS else "")
            text = f"{name}, {len(val)} in total: " + "; ".join(items) + more
        elif isinstance(val, dict):
            text = _json_item(val)
            if not text:
                continue
            text = f"{name}: {text}"
        else:
            s = _json_scalar(val)
            if s:
                overview.append(f"{name}: {s}")
            continue
        sections.append({"url": f"{url}#{key}", "title": title(name),
                         "text": text + ".", "data": True})
    if overview:                          # named after what it holds:
        names = ", ".join(o.split(":", 1)[0] for o in overview[:4])
        sections.insert(0, {"url": f"{url}#overview", "title": title(names),
                            "text": ". ".join(overview) + ".", "data": True})
    return sections


# ── robots.txt (User-agent: * only, prefix rules) ────────────────────────────

def parse_robots(text):
    """(disallow, allow) prefix lists from the groups that apply to '*'."""
    disallow, allow = [], []
    agents, in_star, seen_rule = [], False, False
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, val = (s.strip() for s in line.split(":", 1))
        key = key.lower()
        if key == "user-agent":
            if seen_rule:                 # a new group starts
                agents, seen_rule = [], False
            agents.append(val.lower())
            in_star = "*" in agents
        elif key in ("disallow", "allow"):
            seen_rule = True
            if in_star and val:
                (disallow if key == "disallow" else allow).append(val)
    return disallow, allow


def robots_allows(rules, url):
    """Prefix matching; a longer matching Allow beats a Disallow."""
    disallow, allow = rules
    parts = urllib.parse.urlsplit(url)
    path  = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    dis = max((len(p) for p in disallow if path.startswith(p)), default=-1)
    if dis < 0:
        return True
    alw = max((len(p) for p in allow if path.startswith(p)), default=-1)
    return alw >= dis


# ── Fetching ──────────────────────────────────────────────────────────────────

def _open(url, timeout):
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
        "Accept-Encoding": "gzip, deflate",   # smaller transfers on slow sites
    })
    return urllib.request.urlopen(req, timeout=timeout)


def _decode_body(data, content_encoding):
    """Undo a Content-Encoding. CDNs (python.org's, for one) gzip the body
    even when asked for identity, and urllib does not decompress — indexing
    the raw bytes would fill the cache with noise. Unknown codings (br,
    zstd) raise so the page is skipped."""
    enc = (content_encoding or "identity").strip().lower()
    if enc in ("", "identity"):
        return data
    if enc in ("gzip", "x-gzip"):
        return gzip.decompress(data)
    if enc == "deflate":
        try:
            return zlib.decompress(data)
        except zlib.error:                    # raw deflate without header
            return zlib.decompress(data, -zlib.MAX_WBITS)
    raise ValueError(f"unsupported Content-Encoding {enc!r}")


def _read_body(resp, limit):
    return _decode_body(resp.read(limit), resp.headers.get("Content-Encoding"))


def _fetch_robots(base_url):
    """(disallow, allow) rules from the site's robots.txt (none if missing)."""
    parts = urllib.parse.urlsplit(base_url)
    robots_url = urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, "/robots.txt", "", ""))
    try:
        with _open(robots_url, _settings["fetch_timeout"]) as resp:
            body = _read_body(resp, 200_000).decode("utf-8", "replace")
    except Exception:
        return [], []                     # unavailable -> allow everything
    return parse_robots(body)


def _fetch_html(url, timeout=None):
    """(final_url, html) or (final_url, None) when the response isn't HTML.
    Raises on network errors, truncated/undecodable bodies and unknown
    Content-Encodings so the caller can log and move on."""
    with _open(url, timeout or _settings["fetch_timeout"]) as resp:
        final = resp.geturl()
        ctype = (resp.headers.get_content_type() or "").lower()
        if ctype not in _HTML_TYPES:
            return final, None
        charset = resp.headers.get_content_charset() or "utf-8"
        data = _read_body(resp, MAX_PAGE_BYTES)
    return final, data.decode(charset, "replace")


def _fetch_json(url, timeout=None):
    """The decoded JSON at `url`, or None when the body isn't JSON. Raises
    on network errors like _fetch_html."""
    with _open(url, timeout or _settings["fetch_timeout"]) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        data = _read_body(resp, MAX_PAGE_BYTES)
    try:
        return json.loads(data.decode(charset, "replace"))
    except ValueError:
        return None


_SERIAL_HEADER = re.compile(r"^(sl\.?|s\.?)?\s*no\.?$|^#$|^sno$", re.I)
_PRIVATE_HEADER = re.compile(r"e-?mail|phone|mobile|contact", re.I)
_NUMBER_CELL    = re.compile(r"^[\d.,\s%+-]+$")
_NAME_HEADER    = re.compile(r"\bname\b", re.I)


def table_rows(tables, url, title, heads=None):
    """One small section per row of each real table (a header row plus at
    least two rows): "Name: Dr. A; Department: Mechanical Engg. Dept". A
    staff directory chunked as one block mixes every department into one
    embedding (0.43 for "hod of mechanical"); one row scores 0.59. Serial
    numbers and e-mail/phone columns are left out."""
    out = []
    for ti, rows in enumerate(tables):
        rows = [r for r in rows if any(r)]
        if len(rows) < 3 or len(rows[0]) < 2:
            continue
        if heads is not None and not heads[ti]:
            # no <th> header row: its first row is data, not column names —
            # reading it as names turns "Chairman | Janab A | Chairman, TKM
            # Trust" into "Janab A: Dr. B" nonsense the LLM then believes
            for ri, row in enumerate(rows[:TABLE_MAX_ROWS]):
                cells = [c for c in row if c and not _NUMBER_CELL.match(c)]
                text = " | ".join(cells)
                if len(cells) >= 2 and len(text.split()) >= 3:
                    out.append({"url": f"{url}#t{ti}r{ri}", "title": title,
                                "text": text + "."})
            continue
        header = rows[0]
        if any(_NUMBER_CELL.match(h) for h in header):
            continue                  # a stats box ("UG students | 877")
        for ri, row in enumerate(rows[1:TABLE_MAX_ROWS + 1]):
            cells, about = [], []
            for i, cell in enumerate(row):
                h = header[i] if i < len(header) else ""
                if not cell or _SERIAL_HEADER.match(h) or _PRIVATE_HEADER.search(h):
                    continue
                cells.append(f"{h}: {cell}" if h else cell)
                if not _NAME_HEADER.search(h):
                    about.append(cells[-1])
            text = "; ".join(cells)
            if len(cells) >= 2 and len(text.split()) >= 3:
                # embedded WITHOUT the name: "Designation: Dean, Research"
                # is what a question matches, a person's name only drowns it
                out.append({"url": f"{url}#t{ti}r{ri}", "title": title,
                            "text": text + ".",
                            "embed": "; ".join(about) or text})
    return out


def _page_docs(page, url, title, endpoints):
    """What to index for one fetched page: its text — unless it has too few
    words, or is only a shell that loads its content as data (its menu text
    would crowd the real sections out of the top-K) — plus its table rows."""
    docs = []
    n = len(page["text"].split())
    if n >= MIN_PAGE_WORDS and not (endpoints and n < THIN_PAGE_WORDS):
        docs.append({"url": url, "title": title, "text": page["text"]})
    return docs + table_rows(page["tables"], url, title,
                             page.get("table_heads"))


def _data_pages(endpoints, rules, data_seen, title_hint="", delay=0.0,
                skip=frozenset(), failures=None):
    """Sections of the JSON endpoints a page's scripts load, each endpoint
    fetched once per crawl (`data_seen`, at most MAX_DATA_ENDPOINTS), never
    when already indexed (`skip`), and only if robots.txt allows. Endpoints
    that fail are appended to `failures` (a list) when given."""
    out = []
    for ep in endpoints:
        if (ep in data_seen or ep in skip
                or len(data_seen) >= MAX_DATA_ENDPOINTS):
            continue
        data_seen.add(ep)
        if not robots_allows(rules, ep):
            continue
        if delay:
            time.sleep(delay)
        try:
            data = _fetch_json(ep)
        except Exception as e:
            print(f"[web_rag] skip data {ep}: {type(e).__name__}: {e}")
            if failures is not None:
                failures.append(ep)
            continue
        if data is not None:
            out.extend(json_to_sections(data, ep, title_hint))
    return out


# ── Site catalog: the menus are the topic map ────────────────────────────────

class _MenuParser(HTMLParser):
    """Every <a> on a page with its MENU PATH: the labels of the <li> items it
    sits in. "B.Tech Lateral Entry" under ACADEMICS > Admissions comes out as
    (href, "B.Tech Lateral Entry", ["ACADEMICS", "Admissions"]). An <li>'s
    label is its own first link's text — "#" links included, which is how
    menu headings ("ACADEMICS") name their sub-menus."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._li, self._a, self._skip = [], None, 0
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag == "li":
            self._li.append(None)
        elif tag == "a":
            self._close_a()
            self._a = [(dict(attrs).get("href") or "").strip(), []]

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        elif tag == "a":
            self._close_a()
        elif tag == "li" and self._li:
            self._li.pop()

    def handle_data(self, data):
        if self._a is not None and not self._skip:
            self._a[1].append(data)

    def _close_a(self):
        if self._a is None:
            return
        href, parts = self._a
        self._a = None
        text = " ".join("".join(parts).split())
        path = [l for l in self._li[:-1] if l]
        if self._li and self._li[-1] is None and text:
            self._li[-1] = text
        self.links.append((href, text, path))


def menu_links(html, base_url):
    """[(url, link text, menu path)] for the same-site links of a page."""
    p = _MenuParser()
    try:
        p.feed(html)
        p.close()
    except Exception:
        pass
    p._close_a()
    host, out = _host_of(base_url), []
    for href, text, path in p.links:
        if not href or href.lower().startswith(_SKIP_SCHEMES) or href.startswith("#"):
            continue
        try:
            url = _normalize(urllib.parse.urljoin(base_url, href))
        except ValueError:
            continue
        if url.startswith(("http://", "https://")) and _same_site(_host_of(url), host):
            out.append((url, text, path))
    return out


_FILE_DIRS = re.compile(r"^/(uploads|images|img|files|assets|downloads?)/", re.I)


def _slug_words(url):
    parts = urllib.parse.urlsplit(url.lower())
    return set(re.findall(r"[a-z]+", f"{parts.path} {parts.query}"))


def classify(url, texts=(), paths=()):
    """("focus" | "general", "") for a page to index, or (None, reason).
    Focus = the configured department (URL words like "cse", or its name in
    the link text). Skipped: files, pages under a skipped menu section
    ("DEPARTMENTS" — other departments' home pages), and URLs naming another
    department ("mech", "civil", …)."""
    path = urllib.parse.urlsplit(url).path
    if _looks_binary(url) or _FILE_DIRS.search(path):
        return None, "file (PDF, image or document)"
    words = _slug_words(url)
    amp = lambda t: " ".join(t.lower().replace("&", " and ").split())
    label = amp(" ".join(list(texts) + list(paths)))
    name = amp(_settings["focus_name"] or "")
    if words & set(_settings["focus_slugs"]) or (name and name in label):
        return "focus", ""
    sections = {p.split(">")[0].strip().lower() for p in paths}
    if sections & {s.lower() for s in _settings["skip_sections"]}:
        return None, "other department"
    if words & set(_settings["skip_slugs"]):
        return None, "other department"
    return "general", ""


def discover():
    """(catalog, skipped): catalog = {url: {texts, paths, scope}} from the
    menus of the seed pages (plus the seeds themselves), skipped = {url:
    reason}. Network: one request per seed."""
    seeds = _settings["seeds"] or [_settings["url"]]
    found = {}
    for seed in seeds:
        try:
            final, html = _fetch_html(_normalize(seed))
        except Exception as e:
            print(f"[web_rag] seed {seed} unavailable: {type(e).__name__}: {e}")
            continue
        if html is None:
            continue
        found.setdefault(_normalize(final), {"texts": [], "paths": []})
        for url, text, path in menu_links(html, final):
            e = found.setdefault(url, {"texts": [], "paths": []})
            if text and text not in e["texts"]:
                e["texts"].append(text)
            p = " > ".join(path)
            if p and p not in e["paths"]:
                e["paths"].append(p)
    catalog, skipped = {}, {}
    for url, info in found.items():
        scope, why = classify(url, info["texts"], info["paths"])
        if scope:
            catalog[url] = dict(info, scope=scope)
        else:
            skipped[url] = why
    if len(catalog) > _settings["max_pages"]:     # focus pages first
        keep = sorted(catalog, key=lambda u: catalog[u]["scope"] != "focus")
        for u in keep[_settings["max_pages"]:]:
            skipped[u] = "over WEB_RAG_MAX_PAGES"
            del catalog[u]
    return catalog, skipped


def topic_label(url, info, title=""):
    """What a catalog page is about, for routing: where it lives (the focus
    department or the college), its menu path, link text and URL words
    spelled out, and its title."""
    where = (_settings["focus_name"] + " department"
             if info.get("scope") == "focus" and _settings["focus_name"]
             else "college")
    paths = " / ".join(info.get("paths") or [])
    return " | ".join(s for s in (where, paths,
                                  link_label(url, info.get("texts") or []),
                                  title) if s)


def _display_title(url, info, page_title):
    """A passage's title: the page's <title>, else its menu name ("Computer
    Science and Engineering — People / Staff")."""
    if page_title:
        return page_title
    names = [t for t in info.get("texts") or []
             if t.lower().strip(" .>»…") not in _JUNK_ANCHORS]
    path = (info.get("paths") or [""])[0]
    name = " / ".join(s for s in (path, names[0] if names else "") if s)
    lead = _settings["focus_name"] if info.get("scope") == "focus" else ""
    return " — ".join(s for s in (lead, name) if s) or urllib.parse.urlsplit(url).path


# ── Fetching the catalog ──────────────────────────────────────────────────────

def _fetch_page(url, info, rules, data_seen, data_lock):
    """One catalog page: (docs, status). docs = its passages, table rows and
    the sections of the JSON it loads; status = the verification line
    ("ok", "HTTP 404", "not a web page", …). Never raises."""
    if not robots_allows(rules, url):
        return [], "blocked by robots.txt"
    try:
        final, html = _fetch_html(url)
    except urllib.error.HTTPError as e:
        return [], f"broken (HTTP {e.code})"
    except Exception as e:
        return [], f"unreachable ({type(e).__name__})"
    if html is None:
        return [], "not a web page"
    if not _same_site(_host_of(final), _host_of(_settings["url"])):
        return [], "redirects to another site"
    page = parse_page(html)
    title = _display_title(url, info, page["title"])
    endpoints = find_data_endpoints(page["scripts"], final)
    with data_lock:                       # each JSON endpoint once per refresh
        mine = [ep for ep in endpoints if ep not in data_seen]
        data_seen.update(mine)
    docs = _page_docs(page, final, title, endpoints)
    failures = []
    docs += _data_pages(mine, rules, set(), failures=failures)
    for d in docs:
        d["page"] = url                   # route by the catalog page
    words = len(page["text"].split())
    status = f"ok ({words} words"
    if mine:
        status += f", {len(mine)} data source{'s' * (len(mine) > 1)}"
    status += ")"
    if failures:
        status += f" — data failed: {len(failures)}"
    elif not docs:
        status = f"ok but empty ({words} words)"
    return docs, status


def _strip_boilerplate(results):
    """Remove the lines a site repeats on many pages — the CSE pages all
    open with "Home UG students PG students Research Scholars 16 Faculty
    Staff Publications …" — from page text: they drown the page's own
    subject in its first passage. A page left with too few words goes."""
    counts = {}
    for page_docs, _ in results.values():
        for d in page_docs:
            if "#" not in d["url"] and not d.get("data"):
                for line in set(d["text"].split("\n")):
                    counts[line] = counts.get(line, 0) + 1
    common = {l for l, n in counts.items() if n >= BOILERPLATE_PAGES}
    for url, (page_docs, status) in results.items():
        kept = []
        for d in page_docs:
            if "#" not in d["url"] and not d.get("data"):
                d["text"] = "\n".join(l for l in d["text"].split("\n")
                                      if l not in common)
                if len(d["text"].split()) < MIN_PAGE_WORDS:
                    continue
            kept.append(d)
        page_docs[:] = kept


def _fetch_catalog(catalog, rules):
    """Fetch every catalog page in parallel. Returns (docs, report): the
    deduplicated docs and {url: status}. "/principal" and "/principal.php"
    are one page under two links: the second is reported as a duplicate."""
    data_seen, data_lock = set(), threading.Lock()
    workers = max(1, int(_settings["workers"]))
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="web_rag_fetch") as pool:
        futs = {u: pool.submit(_fetch_page, u, info, rules, data_seen, data_lock)
                for u, info in catalog.items()}
        results = {u: f.result() for u, f in futs.items()}
        # a slow college server drops a few under parallel load: one retry
        retry = [u for u, (_, s) in results.items() if s.startswith("unreachable")]
        futs = {u: pool.submit(_fetch_page, u, catalog[u], rules, data_seen,
                               data_lock) for u in retry}
        results.update({u: f.result() for u, f in futs.items()})
    # A <title> many pages share ("TKM College Of Engineering") says nothing
    # about the page: use its menu name ("Placement") instead.
    counts = {}
    for page_docs, _ in results.values():
        for t in {d["title"] for d in page_docs if not d.get("data")}:
            counts[t] = counts.get(t, 0) + 1
    for url, (page_docs, _) in results.items():
        for d in page_docs:
            if not d.get("data") and counts.get(d["title"], 0) >= GENERIC_TITLE_PAGES:
                d["title"] = _display_title(url, catalog[url], "")
    _strip_boilerplate(results)
    docs, report, seen_text = [], {}, {}
    for url in catalog:                   # catalog order: deterministic dedup
        page_docs, status = results[url]
        fresh = []
        for d in page_docs:
            key = d["text"]
            if key in seen_text:
                continue
            seen_text[key] = url
            fresh.append(d)
        if page_docs and not fresh:
            first = seen_text[page_docs[0]["text"]]
            status = f"duplicate of {first}"
        report[url] = status
        docs += fresh
    return docs, report


def verify(print_report=True):
    """Discover the catalog and fetch every page once, without indexing:
    {url: status} for kept pages plus {url: "skipped: reason"}."""
    rules = _fetch_robots(_settings["url"])
    catalog, skipped = discover()
    _, report = _fetch_catalog(catalog, rules)
    full = {u: f"[{catalog[u]['scope']}] {s}" for u, s in report.items()}
    full.update({u: f"skipped: {why}" for u, why in skipped.items()})
    if print_report:
        for u, s in sorted(full.items(), key=lambda kv: (kv[1].startswith("skipped"), kv[1][:12], kv[0])):
            print(f"  {s:44.44}  {u}")
        ok = sum(1 for s in report.values() if s.startswith("ok"))
        print(f"[web_rag] verify: {len(catalog)} pages in scope ({ok} ok), "
              f"{len(skipped)} links skipped")
    return full


# ── Index build / cache ───────────────────────────────────────────────────────

def _chunk_pages(pages):
    """(chunks, texts-to-embed) for a list of {url, title, text} docs.
    A JSON data section is mostly names, which drown what it IS in the
    embedding ("non teaching staff" scores 0.47 as a full list, 0.78 as its
    title) — so it is embedded by title plus the first few words; a table
    row by its non-name cells."""
    chunks, texts = [], []
    for pg in pages:
        for piece in chunk_text(pg["text"], _settings["chunk_words"],
                                _settings["chunk_overlap"]):
            chunks.append({"url": pg["url"], "title": pg["title"],
                           "text": piece, "page": pg.get("page", pg["url"])})
            if pg.get("embed"):           # a table row: matched by its role
                piece = pg["embed"]
            elif pg.get("data"):
                piece = " ".join(piece.split()[:DATA_EMBED_WORDS])
            texts.append(f"{pg['title']}: {piece}" if pg["title"] else piece)
    return chunks, texts


def _embed_rows(texts):
    """L2-normalised embedding rows for `texts` (batched, see below)."""
    matrix = _embed_batched(texts)
    if matrix.ndim != 2 or matrix.shape[0] != len(texts):
        raise RuntimeError(f"embedding shape {matrix.shape} != {len(texts)}")
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
    return matrix


_ROLE_WORDS = {"regular", "adhoc", "ad", "hoc", "permanent", "guest",
               "visiting", "contract", "temporary", "assistant", "associate"}


def _data_group(url):
    """Data sections of one kind share a group: "regular_teaching_staff" and
    "adhoc_teaching_staff" are both "teaching staff"."""
    base, _, key = url.partition("#")
    words = [w for w in re.split(r"[_\-\s]+", key.lower()) if w]
    while len(words) > 1 and words[0] in _ROLE_WORDS:
        words = words[1:]
    return f"data:{base}:{'_'.join(words)}"


def _topics(catalog, docs):
    """The routing table: one topic per catalog page (topic_label) and one
    per JSON data section (its title) — [{url, label, scope, group}].
    Pages under the same sub-menu ("People > Faculty": Regular AND Adhoc)
    share a group, and so do data sections of one kind; a question
    routed to one member of a group searches all of them (retrieve)."""
    titles = {}
    for d in docs:
        titles.setdefault(d.get("page", d["url"]), d["title"])
    out = []
    for u, info in catalog.items():
        # grouped by SUB-menu ("People > Faculty": Regular + Adhoc) — a
        # whole top-level section ("ADMINISTRATION") is far too broad
        path = (info.get("paths") or [""])[0]
        out.append({"url": u, "label": topic_label(u, info, titles.get(u, "")),
                    "scope": info.get("scope", "general"),
                    "group": f"{info.get('scope')}:{path}" if ">" in path else ""})
    seen = set()
    for d in docs:
        if d.get("data") and d["url"] not in seen:
            seen.add(d["url"])
            out.append({"url": d["url"], "label": d["title"],
                        "scope": catalog.get(d.get("page"), {}).get("scope",
                                                                   "general"),
                        "group": _data_group(d["url"])})
    return out


def _build_index(docs, catalog=None, report=None):
    """Chunk + embed the docs and the routing topics into a fresh index
    dict (no disk)."""
    chunks, texts = _chunk_pages(docs)
    if not chunks:
        raise RuntimeError("no text chunks produced")
    matrix = _embed_rows(texts)
    topics = _topics(catalog or {}, docs)
    topic_matrix = _embed_rows([t["label"] for t in topics]) if topics else None
    # every page that answered counts as fetched — also those whose content
    # was all duplicate or JSON — so lookup() never re-downloads them
    pages = sorted({d.get("page", d["url"]) for d in docs}
                   | {u for u, s in (report or {}).items()
                      if s.startswith(("ok", "duplicate"))})
    now = time.time()
    return {
        "url": _settings["url"], "host": _host_of(_settings["url"]),
        "fetched_at": now,
        "pages": [{"url": u, "fetched_at": now} for u in pages],
        "chunks": chunks, "matrix": matrix,
        "links": topics, "link_matrix": topic_matrix,
        "report": report or {},
    }


def _known_urls(index):
    return {_normalize(p["url"]) for p in index["pages"]} if index else set()

def _embed_batched(texts):
    """embed_fn over `texts` in EMBED_BATCH slices with a short pause between
    them: a 40-page refresh is ~1000 chunks, and one giant encode() from the
    daemon thread would pin all four Pi cores while Luna is listening."""
    rows = []
    for i in range(0, len(texts), EMBED_BATCH):
        if i:
            time.sleep(EMBED_BATCH_PAUSE_S)
        part = _embed_fn(texts[i:i + EMBED_BATCH])
        if part is None:
            raise RuntimeError("embedder returned None")
        rows.append(np.asarray(part, dtype=np.float32))
    return np.vstack(rows)


def _embed_dim():
    """Width of the live embedder's vectors (one tiny encode), or None."""
    try:
        v = _embed_fn(["luna"])
        return None if v is None else int(np.asarray(v).shape[-1])
    except Exception as e:
        print(f"[web_rag] embedder probe failed: {e}")
        return None


def _save_cache(index):
    """Write <host>.json + <host>.npz atomically (tmp + os.replace)."""
    jpath, npath = _cache_paths()
    os.makedirs(os.path.dirname(jpath), exist_ok=True)
    meta = {k: index[k] for k in ("url", "host", "fetched_at", "pages",
                                  "chunks")}
    meta["version"] = CACHE_VERSION
    meta["links"] = index.get("links") or []
    meta["report"] = index.get("report") or {}
    meta["dim"] = int(index["matrix"].shape[1])
    arrays = {"matrix": index["matrix"]}
    if index.get("link_matrix") is not None and meta["links"]:
        arrays["link_matrix"] = index["link_matrix"]
    with _save_lock:
        tmp = jpath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        os.replace(tmp, jpath)
        tmp = npath + ".tmp"
        with open(tmp, "wb") as f:           # file object: numpy adds no .npz
            np.savez(f, **arrays)
        os.replace(tmp, npath)


def _load_cache():
    """The cached index for the configured host, or None (missing/corrupt)."""
    jpath, npath = _cache_paths()
    if not (os.path.exists(jpath) and os.path.exists(npath)):
        return None
    try:
        with open(jpath, "r", encoding="utf-8") as f:
            meta = json.load(f)
        with np.load(npath) as z:
            matrix = np.asarray(z["matrix"], dtype=np.float32)
            link_matrix = (np.asarray(z["link_matrix"], dtype=np.float32)
                           if "link_matrix" in z.files else None)
        links = meta.get("links") or []      # caches before the link map
        if (link_matrix is None or link_matrix.ndim != 2
                or link_matrix.shape[0] != len(links)
                or link_matrix.shape[1] != matrix.shape[1]):
            links, link_matrix = [], None
        meta["links"], meta["link_matrix"] = links, link_matrix
        if meta.get("version") != CACHE_VERSION:
            print("[web_rag] cache is from an older version — rebuilding "
                  "when online")
            return None
        if not _same_site(meta.get("host", ""), _host_of(_settings["url"])):
            print("[web_rag] cache belongs to another host — ignoring")
            return None
        if matrix.ndim != 2 or matrix.shape[0] != len(meta["chunks"]):
            print("[web_rag] cache json/npz disagree — ignoring")
            return None
        live = _embed_dim() if _embed_fn is not None else None
        if live is not None and matrix.shape[1] != live:
            # Built with another model (e.g. the CLI's): unusable with this
            # embedder. Dropping it makes the background worker re-index as
            # soon as we are online, instead of warning on every question.
            print(f"[web_rag] cache dim {matrix.shape[1]} != embedder dim "
                  f"{live} — ignoring, will re-index when online")
            return None
        meta["matrix"] = matrix
        return meta
    except Exception as e:
        print(f"[web_rag] cache unreadable ({e}) — ignoring")
        return None


def _is_stale(index):
    age_h = (time.time() - float(index.get("fetched_at", 0))) / 3600.0
    return age_h >= float(_settings["refresh_hours"])


# ── Public API ────────────────────────────────────────────────────────────────

def start(embed_fn, background=True):
    """Remember the embedder, load the cache synchronously, then (in ONE
    daemon thread) probe connectivity and refresh when online and stale.
    Never raises."""
    global _embed_fn, _index, _thread
    try:
        _embed_fn = embed_fn
        if not _settings["url"]:
            print("[web_rag] no WEB_RAG_URL configured — idle")
            return
        cached = _load_cache()
        with _lock:
            _index = cached
        if cached:
            print(f"[web_rag] cache loaded: {len(cached['pages'])} pages, "
                  f"{len(cached['chunks'])} chunks, {len(cached['links'])} "
                  f"topics from {cached['host']}")
        else:
            print(f"[web_rag] no cache for {_host_of(_settings['url'])} yet")
        if not background:
            return
        if _thread is not None and _thread.is_alive():
            print("[web_rag] background worker already running")
            return
        _thread = threading.Thread(target=_background, name="web_rag",
                                   daemon=True)
        _thread.start()
    except Exception as e:
        print(f"[web_rag] start failed: {e}")


def _background():
    try:
        if not probe_online():
            print("[web_rag] offline — using cache only")
            return
        with _lock:
            idx = _index
        if idx is not None and not _is_stale(idx):
            print("[web_rag] online, cache fresh — no refresh")
            return
        refresh()
    except Exception as e:
        print(f"[web_rag] background worker failed: {e}")


def wait(timeout=None):
    """Block until the background worker finishes (start-up wait, tests,
    CLI). True when nothing is running any more."""
    t = _thread
    if t is not None and t.is_alive():
        t.join(timeout)
        return not t.is_alive()
    return True


def probe_online():
    """Can we reach the site's host:port? A short socket connect, never a
    long block. Records the answer for status()."""
    global _online, _online_at
    ok = False
    try:
        parts = urllib.parse.urlsplit(_settings["url"])
        host = parts.hostname
        port = parts.port or (443 if parts.scheme == "https" else 80)
        if host:
            with socket.create_connection((host, port),
                                          timeout=_settings["online_timeout"]):
                ok = True
    except (OSError, ValueError):             # refused, DNS, bad port, ...
        ok = False
    _online, _online_at = ok, time.time()
    return ok


def is_online():
    """probe_online(), but at most once every ONLINE_RECHECK_S — lookup()
    asks on every question and must not stall on a dead link each time."""
    if _online is None or time.time() - _online_at >= ONLINE_RECHECK_S:
        return probe_online()
    return _online


def busy():
    """True while a refresh is running (first boot: nothing cached yet)."""
    return _refresh_lock.locked()


def refresh(force=False):
    """Catalog + parallel fetch + embed + cache, then swap the live index.
    Synchronous. With force=False a fresh cache is left alone. True on
    success; on failure the old index stays and last_error says why."""
    global _index, _last_error, _dim_warned, _robots
    if not _settings["url"]:
        _last_error = "no url configured"
        return False
    if _embed_fn is None:
        _last_error = "no embedder (call start first)"
        print(f"[web_rag] refresh skipped: {_last_error}")
        return False
    with _refresh_lock:
        with _lock:
            idx = _index
        if not force and idx is not None and not _is_stale(idx):
            return True
        t0 = time.time()
        try:
            _robots = _fetch_robots(_settings["url"])
            catalog, skipped = discover()
            if not catalog:
                raise RuntimeError("no pages found in the site's menus")
            print(f"[web_rag] catalog: {len(catalog)} pages "
                  f"({sum(i['scope'] == 'focus' for i in catalog.values())} "
                  f"{_settings['focus_name'] or 'focus'}), {len(skipped)} "
                  f"links skipped — fetching {_settings['workers']} at a time")
            docs, report = _fetch_catalog(catalog, _robots)
            if not docs:
                raise RuntimeError("no page could be fetched")
            report.update({u: f"skipped: {w}" for u, w in skipped.items()})
            new = _build_index(docs, catalog, report)
        except Exception as e:
            _last_error = str(e)
            print(f"[web_rag] refresh failed: {e} — keeping old index")
            return False
        with _lock:
            _index = new
        try:
            _save_cache(new)
        except Exception as e:
            print(f"[web_rag] could not write cache: {e}")
        _tried.clear()
        _last_error = None
        _dim_warned = False
        bad = sum(1 for s in report.values()
                  if s.startswith(("broken", "unreachable")))
        print(f"[web_rag] indexed {len(new['pages'])} pages / "
              f"{len(new['chunks'])} chunks / {len(new['links'])} topics in "
              f"{time.time() - t0:.1f}s ({bad} broken)")
        return True


def _query_vec(text):
    q = _embed_fn([expand_query(text)])
    if q is None:
        return None
    q = np.asarray(q, dtype=np.float32)[0]
    return q / max(float(np.linalg.norm(q)), 1e-9)


_GENERIC_WORDS = set("""what who whom whose which where when why how is are was
were the a an of in on at to for from by with and or do does did can could
would will tell me about please there any all some this that these those your
our you college tkm tkmce department dept cse computer science engineering
kollam""".split())


def _keywords(text):
    """The question's distinctive words (5-letter stems): no question words,
    no department / college names — those match every topic alike."""
    return {w[:5] for w in re.findall(r"[a-z]+", (text or "").lower())
            if len(w) > 2 and w not in _GENERIC_WORDS}


def _vocabulary(idx):
    """Every word (4+ letters) in the topic names — the site's own words."""
    vocab = idx.get("_vocab")
    if vocab is None or idx.get("_vocab_n") != len(idx.get("links") or []):
        words = set()
        for t in idx.get("links") or []:
            words.update(w for w in re.findall(r"[a-z]+", t["label"].lower())
                         if len(w) >= 4)
        vocab = idx["_vocab"] = words
        idx["_vocab_n"] = len(idx.get("links") or [])
    return vocab


def correct_spelling(text, idx=None):
    """Fix misspelt words against the site's own vocabulary before routing:
    "acheivement of faculties" -> "achievement of faculty". The embedder
    knows "acheivement" no better than a random word (it routed to the
    exam pages). Only words of SPELL_MIN_LEN+ letters that aren't site words
    are touched, and only towards a very close match (SPELL_CUTOFF)."""
    import difflib
    if idx is None:
        with _lock:
            idx = _index
    if idx is None or not text:
        return text
    vocab = _vocabulary(idx)
    if not vocab:
        return text
    choices = sorted(vocab)
    stems = {v[:5] for v in vocab}

    def fix(m):
        w = m.group(0)
        low = w.lower()
        if len(low) < SPELL_MIN_LEN or low in vocab:
            return w
        if low[:5] in stems:
            # a form of a site word ("faculties" for "faculty") is not a
            # typo — difflib would "correct" it to "facilities" — so only a
            # near-identical word with the same start replaces it
            # ("admision" -> "admission")
            same = [v for v in choices if v[:5] == low[:5]]
            best = difflib.get_close_matches(low, same, 1, SPELL_SAME_STEM)
        else:
            best = difflib.get_close_matches(low, choices, 1, SPELL_CUTOFF)
        return best[0] if best else w

    return re.sub(r"[A-Za-z]+", fix, text)


def _names_other_department(text):
    """Does the question name a department other than the focus one
    ("mechanical", "civil engineering")? Uses WEB_RAG_SKIP_SLUGS' longer
    words — two-letter codes ("ce", "ca") would match ordinary words."""
    words = set(re.findall(r"[a-z]+", (text or "").lower()))
    return bool(words & {w for w in _settings["skip_slugs"] if len(w) > 3})


def _route_scores(idx, q, words=()):
    """[(topic url, score)] — the topics the question is about, best first:
    at most route_topics, each >= route_min. `words` (from _keywords) add
    KEYWORD_BONUS per distinctive word found in a topic's name (max 2): the
    embedding alone ranks "About CSE" (0.65) over "Vision / Mission" (0.64)
    for "what is the vision of the cse department"."""
    tm = idx.get("link_matrix")
    if tm is None or not idx.get("links") or tm.shape[1] != q.shape[0]:
        return []
    s = tm @ q
    if words:                             # exact keyword in the topic's name
        names = idx.get("_names")
        if names is None or len(names) != len(idx["links"]):
            names = idx["_names"] = [
                {w[:5] for w in re.findall(r"[a-z]+", t["label"].lower())}
                for t in idx["links"]]
        bonus = np.array([min(2, sum(w in n for w in words)) for n in names],
                         dtype=np.float32)
        s = s + KEYWORD_BONUS * bonus
    out = []
    for i in np.argsort(s)[::-1][:int(_settings["route_topics"])]:
        if float(s[i]) < float(_settings["route_min"]):
            break
        out.append((idx["links"][int(i)]["url"], float(s[i])))
    return out


def _with_groups(idx, routed):
    """{url: score} for the routed topics plus the other members of their
    groups (at GROUP_FACTOR of the member's score): "faculty" routed to
    Regular also searches Adhoc."""
    out = dict(routed)
    groups = {}
    for t in idx.get("links") or []:
        if t.get("group"):
            groups.setdefault(t["group"], []).append(t["url"])
    member = {t["url"]: t.get("group") for t in idx.get("links") or []}
    for u, s in routed:
        g = member.get(u) or ""
        factor = 1.0 if g.startswith("data:") else GROUP_FACTOR
        for v in groups.get(g, ()):
            out[v] = max(out.get(v, 0.0), s * factor)
    return out


def route(text):
    """Which pages / data sections `text` is routed to: [(url, score)]."""
    with _lock:
        idx = _index
    if idx is None or _embed_fn is None or not text.strip():
        return []
    fixed = correct_spelling(text, idx)
    if fixed != text:
        print(f"[web_rag] spelling: {text!r} -> {fixed!r}")
    q = _query_vec(fixed)
    return [] if q is None else _route_scores(idx, q, _keywords(fixed))


# ── Live fetch (a routed page that isn't cached yet) ─────────────────────────

def _merge_live(results):
    """Add live results [(url, docs)] to the index and cache. Pages that
    gave no new text are still recorded, so they aren't fetched again.
    Returns the number of chunks added."""
    global _index
    pages, text_seen = [], set()
    for url, docs in results:
        for d in docs:
            if d["text"] not in text_seen:
                text_seen.add(d["text"])
                pages.append(d)
    chunks, texts = _chunk_pages(pages)
    rows = _embed_rows(texts) if chunks else None
    with _lock:                           # merge into whatever is live now
        cur = _index
        if cur is None or (rows is not None
                           and cur["matrix"].shape[1] != rows.shape[1]):
            return 0
        have = {c["text"] for c in cur["chunks"]}
        keep = [i for i, c in enumerate(chunks) if c["text"] not in have]
        known = _known_urls(cur)
        new_pages = [u for u, _ in results if _normalize(u) not in known]
        if not keep and not new_pages:
            return 0
        new = dict(cur)
        now = time.time()
        new["pages"] = cur["pages"] + [{"url": u, "fetched_at": now}
                                       for u in new_pages]
        if keep:
            new["chunks"] = cur["chunks"] + [chunks[i] for i in keep]
            new["matrix"] = np.vstack([cur["matrix"], rows[keep]])
        _index = new
    try:
        _save_cache(new)
    except Exception as e:
        print(f"[web_rag] could not write cache: {e}")
    return len(keep)


def _late(f, url):
    """A live page that missed the question's wait: cache it when it lands."""
    try:
        docs, status = f.result()
        if status.startswith("ok"):
            n = _merge_live([(url, docs)])
            print(f"[web_rag] live: late page {url} cached ({n} chunks)")
        else:
            _tried.add(url)
    except Exception as e:
        print(f"[web_rag] live: late merge failed: {e}")


def _fetch_routed(urls, wait_s):
    """Fetch these catalog pages now (in parallel), waiting at most wait_s;
    slower ones are cached when they land. Returns chunks added in time."""
    with _lock:
        idx = _index
    infos = {}
    for t in (idx or {}).get("links", []):
        infos.setdefault(t["url"], {"scope": t.get("scope", "general")})
    data_seen, data_lock = set(_known_urls(idx)), threading.Lock()
    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=len(urls), thread_name_prefix="web_rag_live")
    futs = {pool.submit(_fetch_page, u, infos.get(u, {}), _robots, data_seen,
                        data_lock): u for u in urls}
    done, _ = concurrent.futures.wait(futs, timeout=max(0.0, wait_s))
    pool.shutdown(wait=False)
    results = []
    for f, url in futs.items():
        if f not in done:
            print(f"[web_rag] live: {url} still downloading — cached when it lands")
            f.add_done_callback(lambda f, url=url: _late(f, url))
            continue
        docs, status = f.result()
        if status.startswith("ok"):
            results.append((url, docs))
        else:
            print(f"[web_rag] live: {url}: {status}")
            _tried.add(url)
    return _merge_live(results) if results else 0


def lookup(text, force=False, top_k=None):
    """retrieve(), plus a live fetch of the routed pages that aren't cached
    yet (at most live_pages; waits live_timeout, twice that with `force` —
    an explicit "check the website"). Online only. Never raises."""
    hits = retrieve(text, top_k)
    try:
        if int(_settings["live_pages"]) <= 0 or not _settings["url"]:
            return hits
        with _lock:
            idx = _index
        routed = route(text)
        if routed:
            print("[web_rag] route: " + ", ".join(
                f"{u.replace('https://', '').replace('http://', '')[-40:]} "
                f"({s:.2f})" for u, s in routed[:3]))
        known = _known_urls(idx)
        missing = []
        for u, _ in routed:
            if "#" in u:                  # a data section: not a page to fetch
                continue
            if u not in known and u not in _tried and u not in missing:
                missing.append(u)
        missing = missing[:int(_settings["live_pages"])]
        if not missing or not is_online():
            return hits
        with _live_lock:
            wait_s = float(_settings["live_timeout"]) * (2 if force else 1)
            print(f"[web_rag] live: fetching {', '.join(missing)}")
            if _fetch_routed(missing, wait_s):
                hits = retrieve(text, top_k)
    except Exception as e:
        print(f"[web_rag] live lookup failed: {e}")
    return hits


def retrieve(text, top_k=None):
    """[(source_url, passage, score)] best first. The question is ROUTED
    first: only passages of its best-matching topics (pages / data sections)
    are candidates, scored by passage match + ROUTE_WEIGHT x topic match.
    With no topic above route_min it searches every passage. [] without an
    index or embedder; top_k=None means the configured default."""
    global _dim_warned
    with _lock:
        idx = _index
    if idx is None or _embed_fn is None or not text or not text.strip():
        return []
    k = int(_settings["top_k"] if top_k is None else top_k)
    if k <= 0:
        return []
    text = correct_spelling(text, idx)
    try:
        q = _query_vec(text)              # abbreviations spelled out: routing
        if q is None:
            return []
        matrix = idx["matrix"]
        if matrix.shape[1] != q.shape[0]:
            if not _dim_warned:           # once, not on every question
                print(f"[web_rag] index dim {matrix.shape[1]} != query dim "
                      f"{q.shape[0]} — re-index needed")
                _dim_warned = True
            return []
        scores = matrix @ q
        if PASSAGE_QUERY != "expanded" and expand_query(text) != text:
            qr = np.asarray(_embed_fn([text]), dtype=np.float32)[0]
            qr /= max(float(np.linalg.norm(qr)), 1e-9)
            raw = matrix @ qr
            scores = raw if PASSAGE_QUERY == "raw" else np.maximum(raw, scores)
        words = _keywords(text)
        if words and PASSAGE_KEYWORD_BONUS:   # exact words in the passage
            stems = idx.get("_stems")
            if stems is None or len(stems) != len(idx["chunks"]):
                stems = idx["_stems"] = [
                    {w[:5] for w in re.findall(r"[a-z]+", c["text"].lower())}
                    for c in idx["chunks"]]
            scores = scores + PASSAGE_KEYWORD_BONUS * np.array(
                [min(2, sum(w in st for w in words)) for st in stems],
                dtype=np.float32)
        routed = _with_groups(idx, _route_scores(idx, q, words))
        if routed and _names_other_department(text):
            # "HOD of mechanical": the focus department's pages are not
            # about that — only the general (college-wide) pages are
            scope = {t["url"]: t.get("scope") for t in idx.get("links") or []}
            routed = {u: v for u, v in routed.items() if scope.get(u) != "focus"}
    except Exception as e:
        print(f"[web_rag] retrieve failed: {e}")
        return []
    min_sim = float(_settings["min_sim"])
    final = scores
    if routed:
        # a data section / table row with a topic of its own is routed by
        # it alone — not by the page that happened to load it (all the CSE
        # staff data is loaded by every CSE page)
        own = {t["url"] for t in idx.get("links") or []}
        topic = np.array([routed.get(c["url"], 0.0) if c["url"] in own
                          else routed.get(c.get("page", c["url"]), 0.0)
                          for c in idx["chunks"]], dtype=np.float32)
        candidate = topic > 0
        if ROUTE_ALSO_TOP:                # plus the best passages anywhere
            candidate[np.argsort(scores)[::-1][:ROUTE_ALSO_TOP]] = True
        combined = np.where(candidate,
                            (scores + ROUTE_WEIGHT * topic) / (1 + ROUTE_WEIGHT),
                            -1.0)
        # a routed page has already shown it is on topic: its passages need
        # only clear route_min, not the stricter min_sim of a blind search
        if np.any(combined >= float(_settings["route_min"])):
            final, min_sim = combined, float(_settings["route_min"])
    out, per_page = [], {}
    best = None
    for i in np.argsort(final)[::-1]:
        s = float(final[i])
        best = s if best is None else best
        if s < min_sim or len(out) >= k or s < best - RELATIVE_CUT:
            break
        c = idx["chunks"][int(i)]
        # a page's text chunks repeat one topic: 1 slot; its table rows /
        # JSON sections are separate facts: up to MAX_HITS_PER_PAGE
        base, _, frag = c["url"].partition("#")
        key, cap = (base, bool(frag)), (MAX_HITS_PER_PAGE if frag else 1)
        n = per_page.get(key, 0)
        if n >= cap:
            continue
        per_page[key] = n + 1
        title = c.get("title") or ""
        text_out = c["text"] if not title or title.startswith("/") \
            else f"{title}: {c['text']}"
        out.append((c["url"], text_out, s))
    return out

def available():
    with _lock:
        idx = _index
    return idx is not None and len(idx["chunks"]) > 0


# ── "Check the website" requests ──────────────────────────────────────────────

_SITE_RE    = re.compile(r"\b(web ?sites?|web ?pages?|online|internet|site)\b")
_LOOK_WORDS = {"check", "look", "search", "see", "find", "browse", "open",
               "try", "go", "refer", "read", "looking", "checking",
               "searching", "google"}
_YES_WORDS  = {"yes", "yeah", "yep", "yup", "sure", "okay", "ok", "please",
               "go", "ahead", "do", "it", "luna", "fine", "alright"}
_FILLER = _LOOK_WORDS | _YES_WORDS | {
    "can", "could", "would", "will", "you", "hey", "the", "a", "an", "college",
    "website", "websites", "web", "site", "sites", "page", "pages", "online",
    "internet", "on", "in", "into", "at", "its", "for", "me", "and", "then",
    "now", "just", "about", "there", "from", "of", "to", "your", "our",
    "official", "tkm", "tkmce", "so", "that", "this", "out", "up", "info",
    "information", "details", "it's"}


def website_request(text, last_reply=""):
    """Is `text` asking Luna to check the college website? Returns
    (True, topic) — topic is "" when the request names nothing to look up
    ("look into the website", or a "yes" to Luna's own offer to check), so
    the caller re-asks the previous question — or (False, "")."""
    lower = " ".join(text.lower().split())
    words = re.findall(r"[a-z']+", lower)
    wset = set(words)
    if _SITE_RE.search(lower) and (wset & _LOOK_WORDS or len(words) <= 3):
        topic = [w for w in words if w not in _FILLER]
        return True, (" ".join(topic) if len(topic) >= 2 else "")
    reply = (last_reply or "").lower().rstrip()
    if (words and len(words) <= 4 and wset <= _YES_WORDS
            and reply.endswith("?") and _SITE_RE.search(reply)):
        return True, ""
    return False, ""


def status():
    with _lock:
        idx = _index
    fetched = None
    if idx is not None and idx.get("fetched_at"):
        fetched = datetime.fromtimestamp(float(idx["fetched_at"])).isoformat(
            timespec="seconds")
    return {
        "url": _settings["url"],
        "n_pages": len(idx["pages"]) if idx else 0,
        "n_chunks": len(idx["chunks"]) if idx else 0,
        "n_topics": len(idx.get("links") or []) if idx else 0,
        "n_broken": sum(1 for v in (idx or {}).get("report", {}).values()
                        if v.startswith(("broken", "unreachable"))),
        "fetched_at": fetched,
        "online": _online,
        "last_error": _last_error,
        "cache_dir": os.path.dirname(_cache_paths()[0]),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Luna website index (web_rag)")
    ap.add_argument("--url", help="site (default: config)")
    ap.add_argument("--cache-dir", help="cache directory (default: config)")
    ap.add_argument("--verify", action="store_true",
                    help="list the catalog and check every link (no index)")
    ap.add_argument("--refresh", action="store_true", help="fetch + index now")
    ap.add_argument("--route", action="append", default=[],
                    help="show which pages a question is routed to")
    ap.add_argument("--query", action="append", default=[],
                    help="question to retrieve for (repeatable)")
    ap.add_argument("--live", action="store_true",
                    help="--query may fetch routed pages that aren't cached")
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    over = {k: v for k, v in dict(url=args.url, cache_dir=args.cache_dir,
                                  top_k=args.top_k).items() if v is not None}
    configure(**over)
    if args.url:
        configure(seeds=[args.url])

    if args.verify:
        verify()

    def _cli_embed(texts):
        """Standalone embedder — only here, never on the runtime path."""
        global _cli_model
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if "_cli_model" not in globals():
            from sentence_transformers import SentenceTransformer
            _cli_model = SentenceTransformer("all-MiniLM-L6-v2")
        raw = np.asarray(_cli_model.encode(list(texts), batch_size=32,
                                           show_progress_bar=False),
                         dtype=np.float32)
        return raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True),
                                1e-9)

    if args.refresh or args.route or args.query or args.status \
            or not args.verify:
        start(_cli_embed, background=False)
    if args.refresh:
        ok = refresh(force=True)
        print(f"[web_rag] refresh {'ok' if ok else 'FAILED'}")
    for q in args.route:
        print(f"\n[web_rag] route: {q!r}")
        for u, s in route(q):
            label = next((t["label"] for t in _index["links"] if t["url"] == u), "")
            print(f"  {s:.3f}  {u}\n         {label[:150]}")
    for q in args.query:
        hits = lookup(q) if args.live else retrieve(q)
        print(f"\n[web_rag] query: {q!r} -> {len(hits)} hit(s)")
        for url, txt, score in hits:
            print(f"  {score:.3f}  {url}\n         {txt[:300]}")
    if args.status or not (args.verify or args.refresh or args.route
                           or args.query):
        probe_online()                  # so "online" in the report is real
        print(json.dumps(status(), indent=2))
