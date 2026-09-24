# tests/test_web_rag.py
"""Unit tests for web_rag.py — the optional website index.

A tiny site is served from a thread (http.server on an ephemeral 127.0.0.1
port) and embedded with a deterministic hashed bag-of-words function, so no
network and no real model are ever touched:

    python -m unittest discover tests -v
"""

import gzip
import hashlib
import http.server
import os
import shutil
import sys
import tempfile
import threading
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import web_rag  # noqa: E402

DIM = 4096


def fake_embed(texts):
    """Hashed bag-of-words, L2-normalised — deterministic across processes.
    Each word lights two of 4096 dims (two slices of its md5), so a single
    hash collision between unrelated words can't fake a full match."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            w = w.strip(".,;:!?()\"'")
            if w:
                d = hashlib.md5(w.encode()).digest()
                out[i, int.from_bytes(d[:4], "big") % DIM] += 1.0
                out[i, int.from_bytes(d[4:8], "big") % DIM] += 1.0
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


PAGES = {
    "/": ("text/html", """<html><head><title>Test College</title></head><body>
        <h1>Welcome to Test College</h1>
        <p>The college campus sits by the river. It teaches engineering.</p>
        <a href="a.html">About</a> <a href="b.html">Courses</a>
        <a href="a.html#staff">Staff</a> <a href="b.html">Courses again</a>
        <a href="messy.html">News</a>
        <a href="/private/secret.html">Private</a>
        <a href="http://127.0.0.1:1/">External</a>
        <a href="uploads/brochure.pdf">Brochure</a>
        <a href="mailto:x@y.z">Mail</a> <a href="#">Top</a>
        <a href="feed">Feed</a>
        </body></html>"""),
    "/a.html": ("text/html", """<html><head><title>About</title></head><body>
        <p>The principal of the college is Doctor Zorblax Quintrell, appointed
        in 2019. The college has a library and a canteen.</p>
        <a href="/">Home</a> <a href="deep.html">History</a>
        </body></html>"""),
    "/b.html": ("text/html", """<html><head><title>Courses</title></head><body>
        <p>The college offers computer science, electronics and mechanical
        engineering. Admissions to the college open every June.</p>
        </body></html>"""),
    "/deep.html": ("text/html", """<html><head><title>History</title></head>
        <body><p>Founded in 1958, the college started with fifty students in
        one building. It grew steadily over the decades that followed.</p>
        </body></html>"""),
    "/messy.html": ("text/html", """<html><head><title>News</title>
        <style>.x{content:"STYLEWORD"}</style>
        <script>var secret = "SCRIPTWORD";</script></head>
        <body><nav>NAVWORD <a href="a.html">nav link</a></nav>
        <header>HEADERWORD</header>
        <p>The college won the Kalpathi robotics trophy this year
        <li>first item <li>second item <b>bold never closed
        <p>Students built a rover named Pemberly that maps corridors.
        <script>document.write("SCRIPTWORD2")</script>
        <footer>FOOTERWORD</footer></body></html>"""),
    "/private/secret.html": ("text/html", "<html><body><p>SECRETWORD "
                             "should never be indexed by anyone at all here."
                             "</p></body></html>"),
    "/feed": ("application/json", '{"college": "FEEDWORD not html"}'),
    "/uploads/brochure.pdf": ("application/pdf", "%PDF-1.4 PDFWORD"),
    "/robots.txt": ("text/plain", "User-agent: Googlebot\nDisallow: /\n\n"
                    "User-agent: *\nDisallow: /private/\nDisallow: /cgi-bin\n"),
    # A second seed (not linked from "/"): one good page among broken ones.
    "/flaky/": ("text/html", """<html><head><title>Flaky</title></head><body>
        <p>This index page of the flaky section carries enough words to be
        indexed on its own.</p>
        <a href="badcs.html">bad charset</a> <a href="chunked.html">cut</a>
        <a href="garbage.html">garbage</a> <a href="error500.html">500</a>
        <a href="brotli.html">br</a> <a href="gzip.html">gz</a>
        <a href="ok.html">ok</a></body></html>"""),
    "/flaky/ok.html": ("text/html", """<html><head><title>OK</title></head>
        <body><p>The Quorvane laboratory on this page survived the crawl
        even though its neighbours were broken.</p></body></html>"""),
    "/flaky/gzip.html": ("text/html", """<html><head><title>GZ</title></head>
        <body><p>The Gzipwell archive page is served compressed by the CDN
        whatever the client asked for, like python.org.</p></body></html>"""),
    # A third seed: links that redirect (302) — one to a robots-disallowed
    # page, one to a page that is also linked directly.
    "/r/": ("text/html", """<html><head><title>Redirects</title></head><body>
        <p>This index page of the redirect section carries enough words to
        be indexed on its own.</p>
        <a href="to-private">secret</a> <a href="to-ok">alias</a>
        <a href="ok.html">direct</a></body></html>"""),
    "/r/ok.html": ("text/html", """<html><head><title>ROK</title></head>
        <body><p>The Vexlorn observatory is described here once and must be
        indexed exactly once as well.</p></body></html>"""),
}

REDIRECTS = {"/r/to-private": "/private/secret.html", "/r/to-ok": "/r/ok.html"}
BROKEN = {"/flaky/badcs.html", "/flaky/chunked.html", "/flaky/garbage.html",
          "/flaky/error500.html", "/flaky/brotli.html", "/flaky/gzip.html"}


class _Handler(http.server.BaseHTTPRequestHandler):
    requests = []
    lock = threading.Lock()

    def do_GET(self):
        with _Handler.lock:
            _Handler.requests.append(self.path)
        path = self.path.split("?", 1)[0]
        if path in REDIRECTS:
            self.send_response(302)
            self.send_header("Location", REDIRECTS[path])
            self.end_headers()
            return
        if path in BROKEN:
            self._serve_broken(path)
            return
        hit = PAGES.get(path)
        if hit is None:
            self.send_response(404)
            self.end_headers()
            return
        ctype, body = hit
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_broken(self, path):
        """Malformed responses a real site produces: an unknown charset
        (LookupError in decode), a chunked body cut mid-chunk
        (IncompleteRead), a non-HTTP reply (BadStatusLine), a 500, a body
        in a Content-Encoding we can't undo (br) and a gzip body that must
        be undone (a CDN compresses regardless of Accept-Encoding)."""
        if path.endswith("gzip.html") or path.endswith("brotli.html"):
            if path.endswith("gzip.html"):
                data = gzip.compress(PAGES["/flaky/gzip.html"][1].encode())
                enc = "gzip"
            else:
                data, enc = b"\x8b\x00BROTLIWORD not really brotli", "br"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Encoding", enc)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif path.endswith("badcs.html"):
            data = b"<html><body><p>BADCSWORD and enough words to index</p></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=x-bogus-charset")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif path.endswith("chunked.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.write(b"10\r\n<html><p>CHUNKW")   # 16 announced, 14 sent
            self.wfile.flush()
        elif path.endswith("garbage.html"):
            self.wfile.write(b"this is not http\r\n\r\n")
        else:
            self.send_response(500)
            self.end_headers()
        self.close_connection = True

    def log_message(self, *args):       # keep the test output clean
        pass


def _start_server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=lambda: srv.serve_forever(poll_interval=0.05),
                         daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/"


def _stop_server(srv):
    srv.shutdown()
    srv.server_close()


class TestPureHelpers(unittest.TestCase):
    def test_html_to_text_drops_script_style_and_keeps_title(self):
        title, text, links = web_rag.html_to_text(PAGES["/messy.html"][1])
        self.assertEqual(title, "News")
        for bad in ("STYLEWORD", "SCRIPTWORD", "SCRIPTWORD2", "NAVWORD",
                    "HEADERWORD", "FOOTERWORD"):
            self.assertNotIn(bad, text)
        self.assertIn("Kalpathi", text)
        self.assertIn("Pemberly", text)
        self.assertIn("first item", text)
        self.assertIn("a.html", links)   # link inside <nav> is still a link

    def test_html_to_text_soft_skips(self):
        # An unclosed <nav>/<header> ends when a content tag starts.
        for html in ("<body><nav>MENU<p>item</p><main><p>Real fact</p></main>",
                     "<body><nav>MENU<section><p>Real fact</p></section>",
                     "<body><header>MENU<article><p>Real fact</p></article>"):
            _, text, _ = web_rag.html_to_text(html)
            self.assertEqual(text, "Real fact", html)
        # HTML5 <article><header> holds the heading: keep it. The page-level
        # <header>/<footer> and a <nav>/<form> inside content stay dropped.
        _, text, _ = web_rag.html_to_text(
            "<html><body><header><h1>SITEWORD</h1></header>"
            "<article><header><h1>Principal message</h1></header>"
            "<nav>ARTNAVWORD</nav><form>FORMWORD</form>"
            "<p>Dr X is principal</p></article>"
            "<footer>FOOTERWORD</footer></body></html>")
        self.assertEqual(text, "Principal message\nDr X is principal")

    def test_embedding_runs_in_batches(self):
        calls = []

        def counting_embed(texts):
            calls.append(len(texts))
            return fake_embed(texts)

        old = (web_rag._embed_fn, web_rag.EMBED_BATCH, web_rag.EMBED_BATCH_PAUSE_S)
        old_cfg = {k: web_rag._settings[k] for k in ("chunk_words", "chunk_overlap")}
        web_rag._embed_fn, web_rag.EMBED_BATCH = counting_embed, 4
        web_rag.EMBED_BATCH_PAUSE_S = 0.0
        web_rag.configure(chunk_words=10, chunk_overlap=0)
        try:
            words = " ".join(f"w{i}" for i in range(100))      # 10 chunks
            idx = web_rag._build_index([{"url": "u", "title": "t",
                                         "text": words}])
        finally:
            (web_rag._embed_fn, web_rag.EMBED_BATCH,
             web_rag.EMBED_BATCH_PAUSE_S) = old
            web_rag.configure(**old_cfg)
        self.assertEqual(calls, [4, 4, 2])
        self.assertEqual(idx["matrix"].shape, (10, DIM))
        self.assertEqual(len(idx["chunks"]), 10)

    def test_chunking_overlaps(self):
        words = [f"w{i}" for i in range(130)]
        chunks = web_rag.chunk_text(" ".join(words), 50, 20)
        self.assertEqual(len(chunks), 4)             # starts 0, 30, 60, 90
        first, second = chunks[0].split(), chunks[1].split()
        self.assertEqual(len(first), 50)
        self.assertEqual(first[30:], second[:20])    # shared 20 words
        self.assertEqual(chunks[-1].split()[-1], "w129")
        self.assertEqual(web_rag.chunk_text("", 50, 20), [])
        self.assertEqual(len(web_rag.chunk_text("just a few words", 50, 20)), 1)

    def test_robots_parsing(self):
        rules = web_rag.parse_robots(PAGES["/robots.txt"][1])
        self.assertEqual(rules[0], ["/private/", "/cgi-bin"])
        self.assertTrue(web_rag.robots_allows(rules, "http://h/a.html"))
        self.assertFalse(web_rag.robots_allows(rules, "http://h/private/x"))
        self.assertFalse(web_rag.robots_allows(rules, "http://h/cgi-bin/x"))
        allow_rules = web_rag.parse_robots(
            "User-agent: *\nDisallow: /p/\nAllow: /p/open\n")
        self.assertTrue(web_rag.robots_allows(allow_rules, "http://h/p/open/1"))
        self.assertFalse(web_rag.robots_allows(allow_rules, "http://h/p/x"))
        self.assertTrue(web_rag.robots_allows(([], []), "http://h/anything"))


class TestCrawlAndRetrieve(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="luna-test-webrag-")
        self.srv, self.base = _start_server()
        _Handler.requests = []
        web_rag.configure(url=self.base, cache_dir=self.tmp, max_pages=10,
                          max_depth=2, chunk_words=40, chunk_overlap=10,
                          top_k=3, min_sim=0.01, crawl_delay=0.0,
                          online_timeout=0.5, fetch_timeout=3.0,
                          refresh_hours=24)

    def tearDown(self):
        web_rag.wait(5)
        try:
            _stop_server(self.srv)
        except Exception:
            pass
        web_rag.configure(url="")          # drop the index for the next test
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _paths(self):
        with _Handler.lock:
            return sorted(set(_Handler.requests))

    def test_crawl_visits_only_allowed_html_pages(self):
        web_rag.start(fake_embed, background=False)
        self.assertTrue(web_rag.refresh(force=True), web_rag.status())
        self.assertTrue(web_rag.available())
        st = web_rag.status()
        self.assertEqual(st["n_pages"], 5)             # /, a, b, messy, deep
        self.assertIsNone(st["last_error"])
        paths = self._paths()
        self.assertEqual(paths, ["/", "/a.html", "/b.html", "/deep.html",
                                 "/feed", "/messy.html", "/robots.txt"])
        self.assertNotIn("/private/secret.html", paths)   # robots
        self.assertNotIn("/uploads/brochure.pdf", paths)  # binary extension
        with _Handler.lock:
            self.assertEqual(_Handler.requests.count("/robots.txt"), 1)
            self.assertEqual(_Handler.requests.count("/a.html"), 1)  # dedup
        # /feed was fetched (no extension) but rejected on Content-Type.
        urls = {p["url"] for p in web_rag._index["pages"]}
        self.assertNotIn(self.base + "feed", urls)
        # Chunks never carry script/style/nav content.
        all_text = " ".join(c["text"] for c in web_rag._index["chunks"])
        for bad in ("STYLEWORD", "SCRIPTWORD", "NAVWORD", "FOOTERWORD",
                    "SECRETWORD", "FEEDWORD", "PDFWORD"):
            self.assertNotIn(bad, all_text)
        self.assertIn("Zorblax", all_text)
        # Both cache files landed in the temp dir.
        files = sorted(os.listdir(self.tmp))
        self.assertTrue(any(f.endswith(".json") for f in files), files)
        self.assertTrue(any(f.endswith(".npz") for f in files), files)
        self.assertFalse(any(f.endswith(".tmp") for f in files), files)

    def test_max_pages_and_max_depth(self):
        web_rag.configure(max_pages=3)
        web_rag.start(fake_embed, background=False)
        self.assertTrue(web_rag.refresh(force=True))
        html_fetched = [p for p in self._paths() if p != "/robots.txt"]
        self.assertEqual(len(html_fetched), 3)
        self.assertEqual(web_rag.status()["n_pages"], 3)

        _Handler.requests = []
        web_rag.configure(max_pages=10, max_depth=1)
        self.assertTrue(web_rag.refresh(force=True))
        self.assertNotIn("/deep.html", self._paths())   # depth 2 from seed
        self.assertEqual(web_rag.status()["n_pages"], 4)

    def test_retrieve_finds_distinctive_word(self):
        web_rag.start(fake_embed, background=False)
        self.assertTrue(web_rag.refresh(force=True))
        hits = web_rag.retrieve("who is zorblax")
        self.assertTrue(hits)
        url, text, score = hits[0]
        self.assertEqual(url, self.base + "a.html")
        self.assertIn("Zorblax", text)
        self.assertGreater(score, 0.0)
        hits = web_rag.retrieve("tell me about pemberly")
        self.assertEqual(hits[0][0], self.base + "messy.html")

    def test_retrieve_honours_min_sim_and_top_k(self):
        web_rag.start(fake_embed, background=False)
        self.assertTrue(web_rag.refresh(force=True))
        self.assertGreaterEqual(len(web_rag.retrieve("the college")), 3)
        self.assertEqual(len(web_rag.retrieve("the college", top_k=1)), 1)
        web_rag.configure(top_k=2)
        self.assertEqual(len(web_rag.retrieve("the college")), 2)
        self.assertEqual(web_rag.retrieve("the college", top_k=0), [])
        web_rag.configure(min_sim=0.99)
        self.assertEqual(web_rag.retrieve("the college"), [])
        self.assertEqual(web_rag.retrieve(""), [])

    def test_one_bad_page_does_not_abort_crawl(self):
        web_rag.configure(url=self.base + "flaky/")
        web_rag.start(fake_embed, background=False)
        self.assertTrue(web_rag.refresh(force=True), web_rag.status())
        st = web_rag.status()
        self.assertIsNone(st["last_error"])
        self.assertEqual(st["n_pages"], 3)      # /flaky/ + gzip.html + ok.html
        for p in ("/flaky/badcs.html", "/flaky/chunked.html",
                  "/flaky/garbage.html", "/flaky/error500.html",
                  "/flaky/brotli.html"):
            self.assertIn(p, self._paths())              # attempted, skipped
        all_text = " ".join(c["text"] for c in web_rag._index["chunks"])
        self.assertIn("Quorvane", all_text)
        self.assertIn("Gzipwell", all_text)              # gzip undone
        for bad in ("BADCSWORD", "CHUNKW", "BROTLIWORD", "�"):
            self.assertNotIn(bad, all_text)
        self.assertIn("Quorvane", web_rag.retrieve("where is quorvane")[0][1])
        self.assertIn("Gzipwell", web_rag.retrieve("the gzipwell archive")[0][1])

    def test_redirects_are_robots_checked_and_fetched_once(self):
        web_rag.configure(url=self.base + "r/")
        web_rag.start(fake_embed, background=False)
        self.assertTrue(web_rag.refresh(force=True), web_rag.status())
        self.assertEqual(web_rag.status()["n_pages"], 2)  # /r/ + ok.html
        urls = {p["url"] for p in web_rag._index["pages"]}
        self.assertNotIn(self.base + "private/secret.html", urls)
        all_text = " ".join(c["text"] for c in web_rag._index["chunks"])
        self.assertNotIn("SECRETWORD", all_text)       # 302 -> disallowed
        self.assertIn("Vexlorn", all_text)
        with _Handler.lock:
            # Reached via the redirect first, so the direct link is skipped.
            self.assertEqual(_Handler.requests.count("/r/ok.html"), 1)

    def test_cache_with_other_dim_is_discarded_then_rebuilt(self):
        web_rag.start(fake_embed, background=False)
        self.assertTrue(web_rag.refresh(force=True))
        web_rag.configure(url="")                      # forget the index
        web_rag.configure(url=self.base)

        def embed64(texts):                            # same words, 64 dims
            v = fake_embed(texts).reshape(len(texts), 64, 64).sum(axis=2)
            return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

        web_rag.start(embed64, background=False)       # cache is 4096-wide
        self.assertFalse(web_rag.available())
        self.assertEqual(web_rag.retrieve("who is zorblax"), [])
        web_rag.start(embed64)                         # online -> re-index
        self.assertTrue(web_rag.wait(10))
        self.assertTrue(web_rag.available())
        self.assertEqual(web_rag._index["matrix"].shape[1], 64)
        self.assertTrue(web_rag.retrieve("who is zorblax"))

    def test_offline_start_uses_cache(self):
        web_rag.start(fake_embed, background=False)
        self.assertTrue(web_rag.refresh(force=True))
        n_chunks = web_rag.status()["n_chunks"]
        _stop_server(self.srv)
        web_rag.configure(url="")                      # forget everything
        self.assertFalse(web_rag.available())

        _Handler.requests = []
        web_rag.configure(url=self.base)
        web_rag.start(fake_embed)                      # real path, thread
        self.assertTrue(web_rag.wait(10))
        self.assertTrue(web_rag.available())
        st = web_rag.status()
        self.assertEqual(st["n_chunks"], n_chunks)
        self.assertFalse(st["online"])
        self.assertEqual(self._paths(), [])            # nothing fetched
        hits = web_rag.retrieve("who is zorblax")
        self.assertIn("Zorblax", hits[0][1])

    def test_background_refresh_when_online_and_no_cache(self):
        web_rag.start(fake_embed)                      # server up, no cache
        self.assertTrue(web_rag.wait(10))
        self.assertTrue(web_rag.available())
        self.assertTrue(web_rag.status()["online"])
        self.assertEqual(web_rag.status()["n_pages"], 5)

    def test_unreachable_host_never_raises(self):
        empty = tempfile.mkdtemp(prefix="luna-test-webrag-empty-")
        try:
            web_rag.configure(url="http://127.0.0.1:1/", cache_dir=empty)
            web_rag.start(fake_embed)
            self.assertTrue(web_rag.wait(10))
            self.assertFalse(web_rag.available())
            self.assertEqual(web_rag.retrieve("anything"), [])
            self.assertFalse(web_rag.status()["online"])
            self.assertFalse(web_rag.refresh(force=True))   # keeps working
            self.assertIsNotNone(web_rag.status()["last_error"])
            self.assertEqual(os.listdir(empty), [])
        finally:
            shutil.rmtree(empty, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
