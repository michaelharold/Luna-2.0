# site_crawler.py
"""
site_crawler.py — Indexing() step 1: read the college website.

Used only by site_index.py (the weekly, run-by-hand indexing — see
docs/Fetch Algo.md). Nothing here runs while Luna is answering questions.

  CATALOG  The site's menus are its topic map: discover() reads the menus of
           SITE_SEEDS and keeps the general college pages and the focus
           department's pages (other departments, files and other sites are
           skipped, each with a reason). The hand-picked list in
           SITE_LINKS_FILE (data/website_links.txt) is always added,
           including other departments ("other") and the other TKM
           institutions ("affiliate").
  FETCH    Every catalog page is downloaded in parallel and verified
           (broken / not a web page / duplicate). From each page: its text,
           its headlines, its table rows ("Name: Dr. A; Department: ..."),
           the JSON data it loads with JavaScript — also from SITE_DATA_HOSTS
           (the college's ETLab serves staff, student lists, publications,
           news) — and the text of the PDFs it links to (a scanned PDF has
           no text: "needs OCR"). E-mails and phone numbers are left out.

crawl() returns all of it as plain "docs" ({url, title, text, page, ...});
site_index.py groups and organizes them. Standard library only (+ pypdf).

    python site_crawler.py --verify      # catalog + link check, no output
"""

import concurrent.futures
import gzip
import io
import json
import os
import re
import zlib
import socket
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

import console
from intents import ABBREVIATIONS

_DEFAULTS = dict(                       # used when config.py lacks a setting
    SITE_URL="", SITE_SEEDS=(), SITE_FOCUS_NAME="", SITE_FOCUS_SLUGS=(),
    SITE_SKIP_SECTIONS=(), SITE_SKIP_SLUGS=(), SITE_MAX_PAGES=250,
    SITE_FETCH_WORKERS=8, SITE_FETCH_TIMEOUT=8.0, SITE_LINKS_FILE="",
    SITE_DATA_HOSTS=(), SITE_STUDENT_ROSTERS=False, SITE_PDFS=False,
    SITE_MAX_PDFS=80,
)
try:
    import config as _config
except Exception:                       # standalone use without config.py
    _config = None


def _cfg(name):
    return getattr(_config, name, _DEFAULTS[name])


USER_AGENT = "LunaBot/1.0 (+offline desk robot)"

# Module-level tunables (not in config.py):
MAX_PAGE_BYTES      = 2_000_000    # never read more than this from one page
MIN_PAGE_WORDS      = 10           # a page with fewer words carries no facts
MAX_DATA_ENDPOINTS  = 40           # JSON endpoints fetched per crawl
JSON_MAX_ITEMS      = 400          # list entries kept per JSON section (all
                                   # 166 CSE publications)
JSON_ITEM_WORDS     = 120          # words kept per JSON text value (the
                                   # department's mission in full)
THIN_PAGE_WORDS     = 60           # a page that loads its content as data and
                                   # has fewer words of its own is just a shell
TABLE_MAX_ROWS      = 400          # rows kept per table
PDF_MAX_BYTES       = 15_000_000   # never read more than this from one PDF
PDF_MAX_PAGES       = 60           # pages of text read per PDF
PDF_MIN_WORDS       = 20           # fewer words = a scanned PDF (needs OCR)
GENERIC_TITLE_PAGES = 3            # a <title> on this many pages is generic
BOILERPLATE_PAGES   = 5            # a text line on this many pages is template
ONLINE_TIMEOUT      = 1.5          # s for the "is the site reachable?" probe

# Read from config at import, overridable by tests and the CLI through
# configure(). One dict so nothing here reads config later.
_settings = dict(
    url=_cfg("SITE_URL"),
    seeds=list(_cfg("SITE_SEEDS")),
    focus_name=_cfg("SITE_FOCUS_NAME"),
    focus_slugs=list(_cfg("SITE_FOCUS_SLUGS")),
    skip_sections=list(_cfg("SITE_SKIP_SECTIONS")),
    skip_slugs=list(_cfg("SITE_SKIP_SLUGS")),
    max_pages=_cfg("SITE_MAX_PAGES"),
    workers=_cfg("SITE_FETCH_WORKERS"),
    fetch_timeout=_cfg("SITE_FETCH_TIMEOUT"),
    online_timeout=ONLINE_TIMEOUT,
    links_file=_cfg("SITE_LINKS_FILE"),
    data_hosts=[h.lower() for h in _cfg("SITE_DATA_HOSTS")],
    rosters=_cfg("SITE_STUDENT_ROSTERS"),
    pdfs=_cfg("SITE_PDFS"),
    max_pdfs=_cfg("SITE_MAX_PDFS"),
)


def configure(**kw):
    """Override settings (tests / CLI). Unknown keys raise so typos surface."""
    for k in kw:
        if k not in _settings:
            raise ValueError(f"site_crawler: unknown setting {k!r}")
    _settings.update(kw)


_SKIP_EXT = {
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".bmp",
    ".tif", ".tiff", ".zip", ".rar", ".gz", ".tgz", ".tar", ".7z", ".mp3",
    ".mp4", ".m4a", ".avi", ".mov", ".wmv", ".mkv", ".webm", ".doc", ".docx",
    ".xls", ".xlsx", ".ppt", ".pptx", ".css", ".js", ".json", ".xml", ".rss",
    ".exe", ".dmg", ".apk", ".iso", ".woff", ".woff2", ".ttf", ".eot", ".otf",
}
_SKIP_SCHEMES = ("javascript:", "mailto:", "tel:", "data:", "ftp:")
_HTML_TYPES   = ("text/html", "application/xhtml+xml")


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


def _is_pdf(url):
    return urllib.parse.urlsplit(url).path.lower().endswith(".pdf")


def _data_host_ok(url, page_url):
    """A data endpoint the crawler may fetch: the page's own site, or one of
    SITE_DATA_HOSTS (the college's ETLab serves its JSON)."""
    host = _host_of(url)
    strip = lambda h: h[4:] if h.startswith("www.") else h
    return (_same_site(host, _host_of(page_url))
            or strip(host) in {strip(h) for h in _settings["data_hosts"]})


def _looks_binary(url):
    path = urllib.parse.urlsplit(url).path.lower()
    return os.path.splitext(path)[1] in _SKIP_EXT


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


_HEADING_TAGS = {"h1", "h2", "h3", "h4"}
OUTLINE_MAX_HEADINGS = 12          # headlines kept per page
OUTLINE_MAX_WORDS    = 16          # a longer "heading" is a paragraph

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
        self.content_links = []         # hrefs outside nav / header / footer
        self.anchors = []               # (href, anchor text)
        self.scripts = []               # inline <script> bodies
        self.tables = []                # finished tables: [[cell, ...], ...]
        self.table_heads = []           # per finished table: has a header row
        self._heading = None            # text parts of the open <h1>-<h4>
        self.headings = []              # finished headings, in page order

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
        if tag in _HEADING_TAGS:
            self._heading = []
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
                if not self._skip:
                    self.content_links.append(href)
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
        if tag in _HEADING_TAGS and self._heading is not None:
            self.headings.append("".join(self._heading))
            self._heading = None
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
            if self._heading is not None:
                self._heading.append(data)
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
                content_links=p.content_links, scripts=p.scripts, tables=p.tables, table_heads=p.table_heads,
                headings=_clean_headings(p.headings, p.title))


def _clean_headings(headings, title=""):
    """The page's own headlines: whitespace collapsed, styled Unicode
    letters made plain ("𝐀 𝐠𝐢𝐚𝐧𝐭 𝐥𝐞𝐚𝐩" -> "A giant leap"), duplicates, the
    title, template placeholders ("'+data['x'].heading+'") and paragraph-
    length "headings" dropped."""
    out, seen = [], {(title or "").strip().lower()}
    for h in headings:
        h = _unabbreviate(" ".join(unicodedata.normalize("NFKC", h).split()))
        key = h.lower()
        if (not h or key in seen or "+" in h and "'" in h
                or len(h.split()) > OUTLINE_MAX_WORDS):
            continue
        seen.add(key)
        out.append(h)
    return out


def html_to_text(html):
    """(title, text, links) from an HTML document — see parse_page()."""
    page = parse_page(html)
    return page["title"], page["text"], page["links"]


# Anchor texts that say nothing about the page they point to.
_JUNK_ANCHORS = {"read more", "more", "click here", "here", "view", "view more",
                 "view all", "details", "know more", "learn more", "link",
                 "open", "download", "go", "continue", "next", "previous"}


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
    """Same-site (or SITE_DATA_HOSTS) URLs the page's inline scripts load
    data from — $.ajax({url:
    …, data: {…}}), fetch('…'), $.getJSON('…') — with literal data params
    folded into the query string, in order of appearance."""
    out = []
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
                if _data_host_ok(url, page_url) and url not in out:
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
    out = " ".join(ABBREVIATIONS.get(w, w) if w in ("hod", "hods", "nts",
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


# ETLab's section names next to the words a question uses ("who are the
# faculty" never matched "Regular teaching staff"): a section's title is
# its routing topic.
_SECTION_ALIASES = {
    "regular_teaching_staff": "faculty, teachers",
    "adhoc_teaching_staff": "guest faculty, teachers",
    "regular_staff": "faculty, teachers",
    "professors": "faculty", "associate_professors": "faculty",
    "assistant_professors": "faculty",
    "non_teaching_staff": "lab staff, technical and office staff",
    "non_staff": "lab staff, technical and office staff",
    "publications": "research papers",
    "dept_news": "department news", "general_news": "college news",
    "dept_events": "department events",
}

_ROSTER_NAMES = {"btech": "B.Tech", "mtech": "M.Tech", "mca": "MCA",
                 "barch": "B.Arch", "mplan": "M.Plan",
                 "research_scholars": "PhD research scholar"}


def _is_roster(val):
    """A class list: [{batch, count, students: [{name, admission_no}, …]}]."""
    return any(isinstance(v, dict) and isinstance(v.get("students"), list)
               for v in val)


def _person_name(s):
    s = " ".join(s.split())
    return s.title() if s.isupper() else s


def _roster_sections(batches, url, key, name, title):
    """A class list ("btech", "mtech", "research_scholars"): one data section
    with every batch and its size, plus — with SITE_STUDENT_ROSTERS — one
    row per student ("Name: Pranav Prasanth; Batch: CSE 2K23A; Admission No:
    220032"). The rows are routed by that section (their "topic"), so 700
    students don't become 700 topics."""
    topic = f"{url}#{key}"
    name = _ROSTER_NAMES.get(str(key).lower(), name)
    sizes, rows, total = [], [], 0
    for b in batches:
        if not isinstance(b, dict):
            continue
        batch = _json_scalar(b.get("batch")) or name
        studs = [s for s in b.get("students") or () if isinstance(s, dict)]
        try:
            n = int(b.get("count"))
        except (TypeError, ValueError):
            n = len(studs)
        total += n
        sizes.append(f"{batch}, {n} students")
        if not _settings["rosters"]:
            continue
        for s in studs:
            who = _json_scalar(s.get("name"))
            if not who:
                continue
            text = f"Name: {_person_name(who)}; Batch: {batch}"
            adm = _json_scalar(s.get("admission_no"))
            if adm:
                text += f"; Admission No: {adm}"
            slug = re.sub(r"[^A-Za-z0-9]+", "-", batch).strip("-")
            rows.append({"url": f"{topic}/{slug}/{len(rows)}",
                         "title": title(f"{name} student"),
                         "text": text + ".", "topic": topic})
    if not sizes:
        return []
    # the number first, said plainly: "Btech, 8 batches, 604 students in
    # total" left the 3B model unsure 604 counted the B.Tech students
    head = {"url": topic, "title": title(f"{name} students"), "data": True,
            "text": (f"Number of {name} students: {total}, in {len(sizes)} "
                     f"batches ({'; '.join(sizes)}).")}
    return [head] + rows


def json_to_sections(data, url, fallback_title=""):
    """Flatten a JSON document into page-like sections [{url, title, text}]:
    one per list/object field ("Non teaching staff, 11 in total: …") plus an
    overview of the plain fields (vision, mission, counts)."""
    if isinstance(data, list):
        data = {"items": data}
    if not isinstance(data, dict):
        return []
    label = _json_label(data) or fallback_title
    if (label and _settings["focus_name"] and _settings["focus_slugs"]
            and not _other_department(f"{label} — x")):
        # "Computer Science & Engineering (CSE)": the name people say
        label += f" ({'/'.join(_settings['focus_slugs']).upper()})"

    def title(name):
        return f"{label} — {name}" if label else name

    sections, overview = [], []
    for key, val in data.items():
        k = str(key).lower()
        if k in _JSON_SKIP_FIELDS or any(s in k for s in _JSON_SKIP_KEYS):
            continue
        name = _humanize(key)
        if k in _SECTION_ALIASES:         # the words a visitor uses for it
            name += f" ({_SECTION_ALIASES[k]})"
        if isinstance(val, list) and _is_roster(val):
            sections += _roster_sections(val, url, key, name, title)
            continue
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
            cells = []
            for i, cell in enumerate(row):
                h = header[i] if i < len(header) else ""
                if not cell or _SERIAL_HEADER.match(h) or _PRIVATE_HEADER.search(h):
                    continue
                cells.append(f"{h}: {cell}" if h else cell)
            text = "; ".join(cells)
            if len(cells) >= 2 and len(text.split()) >= 3:
                out.append({"url": f"{url}#t{ti}r{ri}", "title": title,
                            "text": text + "."})
    return out


def _page_docs(page, url, title, endpoints):
    """What to index for one fetched page: its text — unless it has too few
    words, or is only a shell that loads its content as data (its menu text
    would crowd the real sections out of the top-K) — plus its table rows."""
    docs = []
    n = len(page["text"].split())
    if n >= MIN_PAGE_WORDS and not (endpoints and n < THIN_PAGE_WORDS):
        docs.append({"url": url, "title": title, "text": page["text"],
                     "outline": [h for h in page.get("headings") or ()
                                 if h.lower() != (title or "").lower()]})
    return docs + table_rows(page["tables"], url, title,
                             page.get("table_heads"))


def _data_pages(endpoints, rules, data_seen, title_hint="", delay=0.0,
                skip=frozenset(), failures=None, statuses=None):
    """Sections of the JSON endpoints a page's scripts load, each endpoint
    fetched once per crawl (`data_seen`, at most MAX_DATA_ENDPOINTS), never
    when already indexed (`skip`), and only if robots.txt allows. Endpoints
    that fail are appended to `failures` (a list) when given; `statuses`
    (a dict) gets each endpoint's verification line."""
    out = []
    for ep in endpoints:
        if (ep in data_seen or ep in skip
                or len(data_seen) >= MAX_DATA_ENDPOINTS):
            continue
        data_seen.add(ep)
        note = statuses.__setitem__ if statuses is not None else (lambda *a: None)
        if not robots_allows(rules, ep):
            note(ep, "blocked by robots.txt")
            continue
        if delay:
            time.sleep(delay)
        try:
            data = _fetch_json(ep)
        except Exception as e:
            console.detail(f"[crawl] skip data {ep}: {type(e).__name__}: {e}")
            note(ep, f"unreachable ({type(e).__name__})")
            if failures is not None:
                failures.append(ep)
            continue
        if data is None:
            note(ep, "not JSON")
            continue
        secs = json_to_sections(data, ep, title_hint)
        note(ep, f"ok ({len(secs)} sections, JSON data)" if secs
             else "ok but empty (JSON data)")
        out.extend(secs)
    return out


# ── PDFs a page links to ──────────────────────────────────────────────────────

def _pdf_title(url):
    """"Batchwise placement status 2022 to 2025" from its file name."""
    name = urllib.parse.unquote(os.path.basename(urllib.parse.urlsplit(url).path))
    name = re.sub(r"(?i)\.pdf$", "", name)
    name = re.sub(r"(?i)[_ -]*11zon$", "", name)          # a compressor's tag
    return " ".join(name.replace("_", " ").split()) or url


def _fetch_pdf(url):
    """(text, status) for one PDF. A scanned PDF is pictures of pages: no
    text to extract, reported as "needs OCR"."""
    try:
        import logging
        from pypdf import PdfReader
        logging.getLogger("pypdf").setLevel(logging.ERROR)
    except ImportError:
        return "", "skipped: pypdf not installed (pip install pypdf)"
    try:
        with _open(url, _settings["fetch_timeout"] * 3) as resp:
            data = _read_body(resp, PDF_MAX_BYTES + 1)
    except urllib.error.HTTPError as e:
        return "", f"broken (HTTP {e.code})"
    except Exception as e:
        return "", f"unreachable ({type(e).__name__})"
    if len(data) > PDF_MAX_BYTES:
        return "", f"skipped: PDF over {PDF_MAX_BYTES // 1_000_000} MB"
    if not data.lstrip()[:5].startswith(b"%PDF"):
        return "", "not a PDF"
    try:
        reader = PdfReader(io.BytesIO(data))
        n_pages = len(reader.pages)
        parts = [pg.extract_text() or "" for pg in reader.pages[:PDF_MAX_PAGES]]
    except Exception as e:                   # noqa: BLE001 — a broken PDF
        return "", f"unreadable PDF ({type(e).__name__})"
    lines = (" ".join(l.split()) for t in parts for l in t.splitlines())
    text = "\n".join(l for l in lines if l)
    words = len(text.split())
    if words < PDF_MIN_WORDS:
        return "", f"needs OCR (scanned PDF, {n_pages} pages, no text)"
    return text, f"ok ({words} words, PDF, {n_pages} pages)"


def _pdf_links(page, page_url):
    """Same-site PDFs linked from the page's content (not its menus, which
    every page repeats), in order."""
    out, home = [], _host_of(_settings["url"])
    for href in page.get("content_links") or ():
        if href.lower().startswith(_SKIP_SCHEMES) or href.startswith("#"):
            continue
        try:
            u = _normalize(urllib.parse.urljoin(page_url, href))
        except ValueError:
            continue
        if _is_pdf(u) and _same_site(_host_of(u), home) and u not in out:
            out.append(u)
    return out


def _pdf_docs(pdfs, rules, title, statuses=None):
    """Docs for the PDFs (one per PDF with text); `statuses` gets each one's
    verification line."""
    out = []
    for u in pdfs:
        if not robots_allows(rules, u):
            text, status = "", "blocked by robots.txt"
        else:
            text, status = _fetch_pdf(u)
        if statuses is not None:
            statuses[u] = status
        if text:
            name = _pdf_title(u)
            out.append({"url": u, "text": text,
                        "title": f"{title} — {name}" if title else name})
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
    if _settings["pdfs"] and _is_pdf(url):
        pass                              # read with pypdf (_fetch_pdf)
    elif _looks_binary(url) or _FILE_DIRS.search(path):
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
            print(f"[crawl] seed {seed} unavailable: {type(e).__name__}: {e}")
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
    groups = {}                           # url -> its groups, in list order
    for url, group, label in curated_links():
        url = _normalize(url)
        e = found.setdefault(url, {"texts": [], "paths": []})
        if group and group not in groups.setdefault(url, []):
            groups[url].append(group)
        if label and label not in e["texts"]:
            e["texts"].insert(0, label)
        e["curated"] = True
    for url, gs in groups.items():        # the list's grouping comes first
        e = found[url]
        e["paths"] = gs + [p for p in e["paths"] if p not in gs]
    for url, e in found.items():
        if e.get("curated") and not e["texts"]:
            e["texts"].append(_slug_title(url))   # "computer science and …"
    home = _host_of(_settings["url"])
    catalog, skipped = {}, {}
    for url, info in found.items():
        if not _same_site(_host_of(url), home):
            # only the curated list reaches other hosts: the TKM institutions
            scope, why = ((None, "file (PDF, image or document)")
                          if _looks_binary(url) else ("affiliate", ""))
        else:
            scope, why = classify(url, info["texts"], info["paths"])
            if scope is None and info.get("curated") and why == "other department":
                scope = "other"           # wanted, but kept apart (retrieve)
        if scope:
            catalog[url] = dict(info, scope=scope)
        else:
            skipped[url] = why
    if len(catalog) > _settings["max_pages"]:     # curated, then focus first
        keep = sorted(catalog, key=lambda u: (not catalog[u].get("curated"),
                                              catalog[u]["scope"] != "focus"))
        for u in keep[_settings["max_pages"]:]:
            skipped[u] = "over SITE_MAX_PAGES"
            del catalog[u]
    return catalog, skipped


def parse_links(text):
    """[(url, group, label)] from a curated links file: a "[Group > Sub]"
    line starts a group; every http(s) URL after it belongs to that group,
    optionally followed by a label ("... student-chapter-cse.php  Student
    Chapters"); "#" starts a comment; anything else is ignored. A page may be
    listed under several groups (faculty achievements is both "Faculty" and
    "Achievements"); the same page twice in one group counts once. Pure."""
    out, seen, group = [], set(), ""
    for line in (text or "").splitlines():
        line = line.split(" #", 1)[0].strip()       # trailing comment
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            group = " > ".join(p.strip() for p in line[1:-1].split(">")
                               if p.strip())
            continue
        if line.startswith(("http://", "https://")):
            url, _, label = line.partition(" ")
            if (url, group) not in seen:
                seen.add((url, group))
                out.append((url, group, " ".join(label.split())))
    return out


def _links_path():
    path = _settings.get("links_file") or ""
    if path and not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    return path


def curated_links():
    """parse_links() of SITE_LINKS_FILE, or [] when unset / unreadable."""
    path = _links_path()
    if not path:
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return parse_links(f.read())
    except OSError as e:
        print(f"[crawl] links file unreadable ({e}) — using the site menus only")
        return []


def _slug_title(url):
    """A readable name from a URL: ".../computer-science-and-engineering" ->
    "computer science and engineering", "https://tkmit.ac.in/" -> "tkmit"."""
    parts = urllib.parse.urlsplit(url)
    slug = re.sub(r"\.(php|html?|aspx?|jsp)$", "", parts.path.strip("/"))
    if not slug:
        host = parts.netloc.lower()
        return (host[4:] if host.startswith("www.") else host).split(".")[0]
    return " ".join(w for w in re.split(r"[^a-z0-9]+", slug.lower()) if w)


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

def _fetch_page(url, info, rules, data_seen, data_lock, sources=None):
    """One catalog page: (docs, status). docs = its passages, table rows, the
    sections of the JSON it loads and the text of the PDFs it links to;
    status = the verification line ("ok", "HTTP 404", "not a web page", …).
    Never raises. An "affiliate" page (another TKM institution's site, from
    the curated list) may live on another host; the college's robots.txt
    says nothing about it. `sources` (a dict, optional) collects
    {"loads": {page: [data / PDF urls]}, "status": {that url: line}}."""
    affiliate = info.get("scope") == "affiliate"
    if not affiliate and not robots_allows(rules, url):
        return [], "blocked by robots.txt"
    if _settings["pdfs"] and _is_pdf(url):        # a PDF straight from a menu
        statuses = {}
        docs = _pdf_docs([url], rules, _display_title(url, info, ""), statuses)
        for d in docs:
            d["page"] = url
        return docs, statuses.get(url, "ok but empty (PDF)")
    try:
        final, html = _fetch_html(url)
    except urllib.error.HTTPError as e:
        return [], f"broken (HTTP {e.code})"
    except Exception as e:
        return [], f"unreachable ({type(e).__name__})"
    if html is None:
        return [], "not a web page"
    if not affiliate and not _same_site(_host_of(final),
                                        _host_of(_settings["url"])):
        return [], "redirects to another site"
    page = parse_page(html)
    title = _display_title(url, info, page["title"])
    endpoints = find_data_endpoints(page["scripts"], final)
    pdfs = _pdf_links(page, final) if _settings["pdfs"] else []
    with data_lock:                       # each JSON endpoint / PDF once
        mine = [ep for ep in endpoints if ep not in data_seen]
        data_seen.update(mine)
        my_pdfs = []
        if sources is not None:           # a full crawl reads the PDFs after
            # every page, most wanted first (_read_pdfs)
            sources.setdefault("loads", {})[url] = endpoints + pdfs
        else:                             # a live fetch: this page's own
            budget = int(_settings["max_pdfs"]) - sum(map(_is_pdf, data_seen))
            my_pdfs = [u for u in pdfs if u not in data_seen][:max(0, budget)]
            data_seen.update(my_pdfs)
    statuses = {}
    docs = _page_docs(page, final, title, endpoints)
    failures = []
    docs += _data_pages(mine, rules, set(), failures=failures,
                        statuses=statuses)
    docs += _pdf_docs(my_pdfs, rules, title, statuses)
    if sources is not None:
        with data_lock:
            sources.setdefault("status", {}).update(statuses)
    for d in docs:
        d["page"] = url                   # route by the catalog page
        if (d.get("data") or d.get("topic")) and _other_department(d["title"]):
            d["scope"] = "other"          # the home page loads Mechanical's
    words = len(page["text"].split())
    status = f"ok ({words} words"
    if mine:
        status += f", {len(mine)} data source{'s' * (len(mine) > 1)}"
    if my_pdfs:
        status += f", {len(my_pdfs)} PDF{'s' * (len(my_pdfs) > 1)}"
    status += ")"
    if failures:
        status += f" — data failed: {len(failures)}"
    ocr = sum(1 for u in my_pdfs if statuses.get(u, "").startswith("needs OCR"))
    if ocr:
        status += f" — {ocr} scanned PDF{'s' * (ocr > 1)} need OCR"
    elif not docs:
        status = f"ok but empty ({words} words)"
    return docs, status


def _other_department(title):
    """A data section about another department: its title starts with that
    department's name ("Mechanical Engineering — Btech")."""
    label, sep, _ = title.partition(" — ")
    focus = _settings["focus_name"]
    if not (sep and focus):
        return False
    label = re.sub(r"\s*\([^)]*\)$", "", label)   # "… (CSE)"
    norm = lambda t: " ".join(t.lower().replace("&", " and ").split())
    return norm(label) != norm(focus)


def _read_pdfs(catalog, results, rules, sources, data_seen):
    """The PDFs the pages link to, read after every page so the cap
    (SITE_MAX_PDFS) goes to the most wanted first: the focus department's
    pages, the hand-picked list, general pages, then other departments.
    Each PDF's docs join its page's results; every PDF gets a status."""
    rank = {"focus": 0, "general": 2, "other": 3, "affiliate": 4}
    order = sorted(catalog, key=lambda u: (
        1 if catalog[u].get("curated") and catalog[u]["scope"] != "focus"
        else rank.get(catalog[u]["scope"], 3)))
    loads, statuses = sources.get("loads") or {}, sources.setdefault("status", {})
    queue = []
    for u in order:
        for pdf in loads.get(u, ()):
            if _is_pdf(pdf) and pdf not in data_seen and pdf not in dict(queue):
                queue.append((pdf, u))
    cap = max(0, int(_settings["max_pdfs"]))
    for pdf, _ in queue[cap:]:
        statuses[pdf] = f"skipped: over SITE_MAX_PDFS ({cap})"
    queue = queue[:cap]
    if not queue:
        return
    console.detail(f"[crawl] reading {len(queue)} PDFs")

    def one(pdf, page):
        docs = results[page][0]
        title = next((d["title"] for d in docs if not d.get("data")),
                     _display_title(page, catalog[page], ""))
        got = {}
        out = _pdf_docs([pdf], rules, title, got)
        return pdf, page, out, got

    workers = max(1, int(_settings["workers"]))
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="crawl_pdf") as pool:
        for pdf, page, out, got in pool.map(lambda q: one(*q), queue):
            statuses.update(got)
            for d in out:
                d["page"] = page
            results[page][0].extend(out)


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
                if d.get("outline"):      # "Quick Links" on every page
                    d["outline"] = [h for h in d["outline"] if h not in common]
                if len(d["text"].split()) < MIN_PAGE_WORDS:
                    continue
            kept.append(d)
        page_docs[:] = kept


def _fetch_catalog(catalog, rules, sources=None):
    """Fetch every catalog page in parallel. Returns (docs, report): the
    deduplicated docs and {url: status}. "/principal" and "/principal.php"
    are one page under two links: the second is reported as a duplicate.
    `sources` (a dict) collects the data / PDF urls each page loads."""
    # a PDF that is a catalog page itself is never re-read for a page
    # linking to it
    data_seen, data_lock = {u for u in catalog if _is_pdf(u)}, threading.Lock()
    workers = max(1, int(_settings["workers"]))
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="crawl_fetch") as pool:
        futs = {u: pool.submit(_fetch_page, u, info, rules, data_seen, data_lock,
                               sources)
                for u, info in catalog.items()}
        results = {u: f.result() for u, f in futs.items()}
        # a slow college server drops a few under parallel load: one retry
        retry = [u for u, (_, s) in results.items() if s.startswith("unreachable")]
        futs = {u: pool.submit(_fetch_page, u, catalog[u], rules, data_seen,
                               data_lock, sources) for u in retry}
        results.update({u: f.result() for u, f in futs.items()})
    if _settings["pdfs"] and sources is not None:
        _read_pdfs(catalog, results, rules, sources, data_seen)
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
    docs, report, seen_text, seen_doc = [], {}, {}, {}
    alias = {}                            # dropped source -> the kept copy
    for url in catalog:                   # catalog order: deterministic dedup
        page_docs, status = results[url]
        fresh = []
        for d in page_docs:
            key = d["text"]
            if key in seen_text:
                kept = seen_doc[key].partition("#")[0]
                base = d["url"].partition("#")[0]
                if base != kept:          # get-staff.php = ETLab's ?id=2
                    alias.setdefault(base, kept)
                continue
            seen_text[key], seen_doc[key] = url, d["url"]
            fresh.append(d)
        if page_docs and not fresh:
            first = seen_text[page_docs[0]["text"]]
            status = f"duplicate of {first}"
        report[url] = status
        docs += fresh
    if sources is not None and alias:     # a page loading the dropped copy
        for page, srcs in (sources.get("loads") or {}).items():   # loads
            srcs += [alias[u] for u in srcs           # the kept one too
                     if u in alias and alias[u] not in srcs]
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
        print(f"[crawl] verify: {len(catalog)} pages in scope ({ok} ok), "
              f"{len(skipped)} links skipped")
    return full


def probe_online():
    """Can we reach the site's host:port? A short socket connect, never a
    long block."""
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
    return ok



# ── The whole crawl ───────────────────────────────────────────────────────────

def crawl():
    """Indexing step 1 (network): every catalog page with its data and PDFs.

    Returns {url, fetched_at, catalog, docs, report, loads}:
      catalog  {page url: {texts, paths, scope, curated}} (menu names)
      docs     [{url, title, text, page, ...}]: page text (with "outline",
               its headlines), table rows / data sections (url with a
               "#fragment", "data": True), student rows ("topic"), PDFs
      report   {url: status} for every page, data source, PDF and skipped
               link ("ok (512 words)", "broken (HTTP 404)", "needs OCR" ...)
      loads    {page url: [data / PDF urls it loads]}
    Raises RuntimeError when nothing could be read (offline, no menus)."""
    if not _settings["url"]:
        raise RuntimeError("no SITE_URL configured")
    if not probe_online():
        raise RuntimeError(f"cannot reach {_host_of(_settings['url'])} "
                           f"(offline?)")
    rules = _fetch_robots(_settings["url"])
    catalog, skipped = discover()
    if not catalog:
        raise RuntimeError("no pages found in the site's menus")
    console.detail(f"[crawl] catalog: {len(catalog)} pages, {len(skipped)} "
                   f"links skipped — fetching {_settings['workers']} at a time")
    sources = {}
    docs, report = _fetch_catalog(catalog, rules, sources)
    if not docs:
        raise RuntimeError("no page could be fetched")
    report.update({u: f"skipped: {w}" for u, w in skipped.items()})
    for u, st in (sources.get("status") or {}).items():
        report.setdefault(u, st)          # each data source / PDF, verified
    return {"url": _settings["url"], "fetched_at": time.time(),
            "catalog": catalog, "docs": docs, "report": report,
            "loads": sources.get("loads") or {}}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Luna's website crawler (Indexing step 1). The full "
                    "indexing is: python site_index.py")
    ap.add_argument("--verify", action="store_true",
                    help="list the catalog and check every link, write nothing")
    args = ap.parse_args()
    if args.verify:
        verify()
    else:
        ap.print_help()
