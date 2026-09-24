# web_rag.py
"""
web_rag.py — optional website retrieval that keeps working offline.

Luna is offline-forever by design, but the college website already holds
most of what visitors ask about (principal, departments, admissions, events)
and nobody wants to retype it into knowledge.txt. So, when the Pi HAPPENS to
be online, this module crawls a bounded slice of ONE configured site, turns
each page into overlapping word chunks, embeds them with the very same
sentence transformer brain.py uses for the knowledge base, and caches all of
it on disk (<cache_dir>/<host>.json + .npz). From then on retrieval is a
local matrix product — no network needed — and the cache is only re-fetched
when online AND stale (WEB_RAG_REFRESH_HOURS).

Everything degrades instead of failing:
  - no WEB_RAG_URL              -> brain.py never calls us
  - offline + no cache          -> available() False, retrieve() -> []
  - offline + cache             -> answers from the cache
  - crawl breaks half-way       -> old index kept, last_error set
  - embedder missing            -> retrieve() -> []
start() never raises and does all network work in ONE daemon thread, after a
short socket probe, so boot never blocks on a flaky Wi-Fi.

The runtime path gets brain.py's embed() and must NOT import
sentence_transformers itself (one model copy in RAM); only the CLI in
__main__ builds its own embedder for standalone use:

    python web_rag.py --url https://example.org --refresh
    python web_rag.py --query "who is the principal"
    python web_rag.py --status
"""

import gzip
import json
import os
import re
import zlib
import socket
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime
from html.parser import HTMLParser

import numpy as np

try:
    from config import (WEB_RAG_URL, WEB_RAG_MAX_PAGES, WEB_RAG_MAX_DEPTH,
                        WEB_RAG_CHUNK_WORDS, WEB_RAG_TOP_K, WEB_RAG_MIN_SIM,
                        WEB_RAG_CACHE_DIR, WEB_RAG_REFRESH_HOURS,
                        WEB_RAG_FETCH_TIMEOUT, WEB_RAG_ONLINE_TIMEOUT)
except Exception:                       # standalone use without config.py
    WEB_RAG_URL, WEB_RAG_MAX_PAGES, WEB_RAG_MAX_DEPTH = "", 40, 2
    WEB_RAG_CHUNK_WORDS, WEB_RAG_TOP_K, WEB_RAG_MIN_SIM = 120, 3, 0.35
    WEB_RAG_CACHE_DIR, WEB_RAG_REFRESH_HOURS = "data/rag_cache", 24
    WEB_RAG_FETCH_TIMEOUT, WEB_RAG_ONLINE_TIMEOUT = 8.0, 1.5

USER_AGENT = "LunaBot/1.0 (+offline desk robot)"

# Not in config.py (module-level tunables, see the result's new_tunables):
CRAWL_DELAY_S       = 0.2          # polite pause between page fetches
CHUNK_OVERLAP_WORDS = 20           # words shared by neighbouring chunks
MAX_PAGE_BYTES      = 2_000_000    # never read more than this from one page
MIN_CHUNK_WORDS     = 8            # a trailing stub shorter than this is dropped
MIN_PAGE_WORDS      = 10           # a page with fewer words carries no facts
EMBED_BATCH         = 64           # chunks per embed_fn() call during a refresh
EMBED_BATCH_PAUSE_S = 0.05         # breather between batches so the daemon
                                   # thread doesn't starve vosk/LLM on the Pi

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Module-level settings: read from config at import, overridable by tests and
# the CLI through configure(). One dict so nothing here reads config later.
_settings = dict(
    url=WEB_RAG_URL,
    cache_dir=WEB_RAG_CACHE_DIR,
    max_pages=WEB_RAG_MAX_PAGES,
    max_depth=WEB_RAG_MAX_DEPTH,
    chunk_words=WEB_RAG_CHUNK_WORDS,
    chunk_overlap=CHUNK_OVERLAP_WORDS,
    top_k=WEB_RAG_TOP_K,
    min_sim=WEB_RAG_MIN_SIM,
    refresh_hours=WEB_RAG_REFRESH_HOURS,
    fetch_timeout=WEB_RAG_FETCH_TIMEOUT,
    online_timeout=WEB_RAG_ONLINE_TIMEOUT,
    crawl_delay=CRAWL_DELAY_S,
)

_lock         = threading.Lock()   # guards _index swaps and reads
_refresh_lock = threading.Lock()   # one crawl at a time
_embed_fn     = None
_index        = None               # {url, host, fetched_at, pages, chunks, matrix}
_thread       = None
_online       = None               # None = not probed yet
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


def configure(**kw):
    """Override settings (tests / CLI). Unknown keys raise so typos surface.
    Changing the url drops the in-memory index — it belonged to another site."""
    global _index
    for k in kw:
        if k not in _settings:
            raise ValueError(f"web_rag: unknown setting {k!r}")
    url_changed = "url" in kw and kw["url"] != _settings["url"]
    _settings.update(kw)
    if url_changed:
        with _lock:
            _index = None


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
_BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4",
               "h5", "h6", "tr", "td", "th", "table", "section", "article",
               "aside", "main", "blockquote", "pre", "hr", "dt", "dd", "dl",
               "figure", "figcaption", "address", "details", "summary",
               "body", "title", "nav", "header", "footer", "form"}


class _TextExtractor(HTMLParser):
    """Collects visible text, the <title> and every href. Unclosed tags are
    tolerated: only the skip-tags need matching end tags, a content tag
    (<main>/<article>/<section>) ends any soft skip, and </body> or </html>
    resets any skip that a sloppy page never closed."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip  = []
        self._content = []              # open main/article/section tags
        self._title = []
        self._in_title = False
        self.parts = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag in _CONTENT_TAGS:
            self._content.append(tag)
            self._skip = [t for t in self._skip if t not in _SOFT_SKIP]
        if tag in _SKIP_TAGS and not (tag in _SECTION_PART and self._content):
            self._skip.append(tag)
            return
        if tag == "title":
            self._in_title = True
        if tag in ("a", "area"):
            for k, v in attrs:
                if k == "href" and v:
                    self.links.append(v.strip())
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
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
        if not self._skip:
            self.parts.append(data)

    @property
    def title(self):
        return " ".join("".join(self._title).split())


def html_to_text(html):
    """(title, text, links) from an HTML document. Text keeps a newline at
    block boundaries and has all other whitespace collapsed."""
    p = _TextExtractor()
    try:
        p.feed(html)
        p.close()
    except Exception:                   # html.parser is lenient; be more so
        pass
    raw   = "".join(p.parts)
    lines = [" ".join(l.split()) for l in raw.split("\n")]
    text  = "\n".join(l for l in lines if l)
    return p.title, text, p.links


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
    parts = urllib.parse.urlsplit(base_url)
    robots_url = urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, "/robots.txt", "", ""))
    try:
        with _open(robots_url, _settings["fetch_timeout"]) as resp:
            body = _read_body(resp, 200_000).decode("utf-8", "replace")
        return parse_robots(body)
    except Exception:
        return [], []                     # unavailable -> allow everything


def _fetch_html(url):
    """(final_url, html) or (final_url, None) when the response isn't HTML.
    Raises on network errors, truncated/undecodable bodies and unknown
    Content-Encodings so the caller can log and move on."""
    with _open(url, _settings["fetch_timeout"]) as resp:
        final = resp.geturl()
        ctype = (resp.headers.get_content_type() or "").lower()
        if ctype not in _HTML_TYPES:
            return final, None
        charset = resp.headers.get_content_charset() or "utf-8"
        data = _read_body(resp, MAX_PAGE_BYTES)
    return final, data.decode(charset, "replace")


def _crawl():
    """Breadth-first crawl of the configured site. Returns
    [{url, title, text}] — pages that were HTML and carried some words."""
    seed  = _normalize(_settings["url"])
    host  = _host_of(seed)
    rules = _fetch_robots(seed)
    if rules[0]:
        print(f"[web_rag] robots.txt disallows {len(rules[0])} prefix(es)")

    queue   = [(seed, 0)]
    seen    = {seed}
    pages   = []
    fetched = 0
    text_seen = set()
    while queue and fetched < _settings["max_pages"]:
        url, depth = queue.pop(0)
        if not robots_allows(rules, url):
            continue
        if fetched:
            time.sleep(_settings["crawl_delay"])
        fetched += 1
        try:
            final, html = _fetch_html(url)
        except Exception as e:
            # URLError, timeouts, http.client.HTTPException (truncated
            # chunked body, garbage status line), LookupError (bogus
            # charset), ... — ONE bad page must never sink the whole crawl.
            print(f"[web_rag] skip {url}: {type(e).__name__}: {e}")
            continue
        if html is None:
            continue
        try:
            final_n = _normalize(final)
        except ValueError:
            continue
        if final_n != url:                # a redirect landed somewhere else
            if not _same_site(_host_of(final_n), host):
                continue
            if not robots_allows(rules, final_n):
                print(f"[web_rag] skip {url}: redirects to disallowed {final_n}")
                continue
            if final_n in seen:           # target queued too: fetch it once
                queue = [(u, d) for (u, d) in queue if u != final_n]
            seen.add(final_n)
        title, text, links = html_to_text(html)
        words = text.split()
        if len(words) >= MIN_PAGE_WORDS:
            key = hash(text)
            if key not in text_seen:      # "/" and "/index.php" are one page
                text_seen.add(key)
                pages.append({"url": final, "title": title or
                              urllib.parse.urlsplit(final).path, "text": text})
        if depth >= _settings["max_depth"]:
            continue
        for href in links:
            if href.lower().startswith(_SKIP_SCHEMES) or href.startswith("#"):
                continue
            try:
                nxt = _normalize(urllib.parse.urljoin(final, href))
            except ValueError:
                continue
            if not nxt.startswith(("http://", "https://")):
                continue
            if nxt in seen or _looks_binary(nxt):
                continue
            if not _same_site(_host_of(nxt), host):
                continue
            seen.add(nxt)
            queue.append((nxt, depth + 1))
    return pages


# ── Index build / cache ───────────────────────────────────────────────────────

def _build_index(pages):
    """Chunk + embed the crawled pages into a fresh index dict (no disk)."""
    chunks, texts = [], []
    for pg in pages:
        for piece in chunk_text(pg["text"], _settings["chunk_words"],
                                _settings["chunk_overlap"]):
            chunks.append({"url": pg["url"], "title": pg["title"],
                           "text": piece})
            texts.append(f"{pg['title']}: {piece}" if pg["title"] else piece)
    if not chunks:
        raise RuntimeError("no text chunks produced")
    matrix = _embed_batched(texts)
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
        raise RuntimeError(f"embedding shape {matrix.shape} != {len(chunks)}")
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
    return {
        "url": _settings["url"], "host": _host_of(_settings["url"]),
        "fetched_at": time.time(),
        "pages": [{"url": p["url"], "title": p["title"]} for p in pages],
        "chunks": chunks, "matrix": matrix,
    }


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
    meta["dim"] = int(index["matrix"].shape[1])
    tmp = jpath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    os.replace(tmp, jpath)
    tmp = npath + ".tmp"
    with open(tmp, "wb") as f:               # file object: numpy adds no .npz
        np.savez(f, matrix=index["matrix"])
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
                  f"{len(cached['chunks'])} chunks from {cached['host']}")
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
    """Block until the background worker finishes (tests / CLI). True when
    nothing is running any more."""
    t = _thread
    if t is not None and t.is_alive():
        t.join(timeout)
        return not t.is_alive()
    return True


def probe_online():
    """Can we reach the site's host:port? A short socket connect, never a
    long block. Records the answer for status()."""
    global _online
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
    _online = ok
    return ok


def refresh(force=False):
    """Crawl + embed + cache, then swap the live index. Synchronous. With
    force=False a fresh cache is left alone. True on success; on failure the
    old index stays and last_error says why."""
    global _index, _last_error, _dim_warned
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
        print(f"[web_rag] crawling {_settings['url']} "
              f"(max {_settings['max_pages']} pages, depth "
              f"{_settings['max_depth']})")
        try:
            pages = _crawl()
            if not pages:
                raise RuntimeError("no HTML pages fetched")
            new = _build_index(pages)
            _save_cache(new)
        except Exception as e:
            _last_error = str(e)
            print(f"[web_rag] refresh failed: {e} — keeping old index")
            return False
        with _lock:
            _index = new
        _last_error = None
        _dim_warned = False
        print(f"[web_rag] indexed {len(new['pages'])} pages / "
              f"{len(new['chunks'])} chunks in {time.time() - t0:.1f}s")
        return True


def retrieve(text, top_k=None):
    """[(source_url, chunk_text, score)] best first — the chunks whose cosine
    similarity to `text` clears min_sim. [] without an index or embedder.
    top_k=None means the configured default; top_k=0 really means none."""
    global _dim_warned
    with _lock:
        idx = _index
    if idx is None or _embed_fn is None or not text or not text.strip():
        return []
    k = int(_settings["top_k"] if top_k is None else top_k)
    if k <= 0:
        return []
    try:
        q = _embed_fn([text])
        if q is None:
            return []
        q = np.asarray(q, dtype=np.float32)[0]
        q /= max(float(np.linalg.norm(q)), 1e-9)
        matrix = idx["matrix"]
        if matrix.shape[1] != q.shape[0]:
            if not _dim_warned:           # once, not on every question
                print(f"[web_rag] index dim {matrix.shape[1]} != query dim "
                      f"{q.shape[0]} — re-index needed")
                _dim_warned = True
            return []
        scores = matrix @ q
    except Exception as e:
        print(f"[web_rag] retrieve failed: {e}")
        return []
    order = np.argsort(scores)[::-1][:k]
    min_sim = float(_settings["min_sim"])
    out = []
    for i in order:
        s = float(scores[i])
        if s < min_sim:
            break
        c = idx["chunks"][int(i)]
        out.append((c["url"], c["text"], s))
    return out


def available():
    with _lock:
        idx = _index
    return idx is not None and len(idx["chunks"]) > 0


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
        "fetched_at": fetched,
        "online": _online,
        "last_error": _last_error,
        "cache_dir": os.path.dirname(_cache_paths()[0]),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Luna website index (web_rag)")
    ap.add_argument("--url", help="site to crawl (default: config)")
    ap.add_argument("--cache-dir", help="cache directory (default: config)")
    ap.add_argument("--max-pages", type=int)
    ap.add_argument("--max-depth", type=int)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--min-sim", type=float)
    ap.add_argument("--refresh", action="store_true", help="crawl now")
    ap.add_argument("--query", action="append", default=[],
                    help="question to retrieve for (repeatable)")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    over = {k: v for k, v in dict(
        url=args.url, cache_dir=args.cache_dir, max_pages=args.max_pages,
        max_depth=args.max_depth, top_k=args.top_k, min_sim=args.min_sim,
    ).items() if v is not None}
    configure(**over)

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

    start(_cli_embed, background=False)
    if args.refresh:
        ok = refresh(force=True)
        print(f"[web_rag] refresh {'ok' if ok else 'FAILED'}")
    for q in args.query:
        hits = retrieve(q)
        print(f"\n[web_rag] query: {q!r} -> {len(hits)} hit(s)")
        for url, txt, score in hits:
            print(f"  {score:.3f}  {url}\n         {txt[:300]}")
    if args.status or not (args.refresh or args.query):
        probe_online()                  # so "online" in the report is real
        print(json.dumps(status(), indent=2))
