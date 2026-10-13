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
import time
import unittest
from unittest import mock

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
                    "User-agent: *\nDisallow: /private/\nDisallow: /cgi-bin\n"
                    "Disallow: /s/private/\n"),
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
    # A fourth seed for live lookups: the crawl (depth 0) only reads this
    # page; its menu links are what lookup() can fetch later.
    "/live/": ("text/html", """<html><head><title>Department</title></head>
        <body><nav><a href="people.html">Staff</a>
        <a href="labs.html">Laboratories</a>
        <a href="alumni.html">Alumni</a></nav>
        <p>The department of quantum studies was founded long ago and teaches
        many courses to many students every single year.</p></body></html>"""),
    # A shell page: the list arrives from JSON, like tkmce.ac.in's staff pages.
    "/live/people.html": ("text/html", """<html><head><title></title></head>
        <body><p>Technical Staff</p><div id="list"></div>
        <script>
        $(document).ready(function() {
          $.ajax({ success: function(d) { $("#list").append(d); },
                   type: 'get',
                   url: 'get-people.php', // <-- local PHP bridge
                   data: { id: 7 }, dataType: 'json' });
          $.ajax({ url: 'get-other.php', data: { id: deptId } });
        });
        </script></body></html>"""),
    "/live/get-people.php": ("application/json", """{
        "vision": "To be excellent at quantum things.",
        "hod": [{"dept_name": "QUANTUM STUDIES", "title": "Dr. ",
                 "name": "Dr. Ilvana Brisk", "designation": "Professor",
                 "email": "ilvana@q.edu", "photo": "U-1.jpg"}],
        "non_teaching_staff": [
            {"sno": "1", "id": "9", "name": "Mr. Tobrin Valesko",
             "designation": "Lab Assistant", "email": "tobrin@q.edu",
             "phone": "12345", "slug": "mr-tobrin"}],
        "btech": [{"batch": "QS 2K23", "count": "60",
                   "students": "[{'sno': 1, 'name': 'SECRETSTUDENT'}]"}],
        "popups": [{"name": "POPUPWORD", "image": "/uploads/x.jpg"}]
    }"""),
    "/live/labs.html": ("text/html", """<html><head><title>Labs</title></head>
        <body><p>The Xantharic laboratory has forty machines and opens at nine
        in the morning on every working day of the week.</p></body></html>"""),
    # Its data endpoint answers slowly (see SLOW): lands after the deadline.
    "/live/alumni.html": ("text/html", """<html><body><p>Alumni</p><script>
        $.ajax({url: 'get-alumni.php', data: {id: 3}});</script></body></html>"""),
    "/live/get-alumni.php": ("application/json", """{"alumni": [
        {"name": "Ms. Quillon Dreya", "designation": "Rocket engineer"}]}"""),
}
# A small college site for the catalog / routing tests (served under /s/):
# a home page whose MENU is the catalog, a focus department ("Quantum
# Studies", URL word "qs") whose staff lists arrive as JSON, another
# department to skip, a duplicate, a broken link, a robots-blocked page.
_QS_HEAD = ("<p>UG students PG students Faculty Staff Publications</p>"
            "<script>$.ajax({url: 'get-qs.php', data: {id: 7}});</script>")
SITE = {
    "/s/": ("text/html", """<html><head><title>Test College</title></head><body>
        <ul>
        <li><a href="#">ABOUT</a><ul>
            <li><a href="about.html">The College</a></li></ul></li>
        <li><a href="#">ADMINISTRATION</a><ul>
            <li><a href="principal.html">Principal</a></li>
            <li><a href="principal.php">Principal</a></li></ul></li>
        <li><a href="#">ACADEMICS</a><ul>
            <li><a href="#">Admissions</a><ul>
                <li><a href="admission.html">UG/PG</a></li></ul></li>
            <li><a href="exam.html">Exam Calendar</a></li></ul></li>
        <li><a href="#">DEPARTMENTS</a><ul>
            <li><a href="dept-qs.html">Quantum Studies</a></li>
            <li><a href="mechanical.html">Mechanical Engineering</a></li></ul></li>
        <li><a href="broken.html">Notices</a></li>
        <li><a href="/s/private/x.html">Staff only</a></li>
        <li><a href="uploads/brochure.pdf">Brochure</a></li>
        <li><a href="http://other.example/x.html">Elsewhere</a></li>
        </ul>
        <p>Welcome to the test college, a place of learning by the river.</p>
        </body></html>"""),
    "/s/about.html": ("text/html", """<html><head><title>About</title></head>
        <body><p>The test college was founded in 1958 by Janab Zorblax and
        teaches engineering to a thousand students every year.</p></body></html>"""),
    "/s/principal.html": ("text/html", """<html><head><title>Principal</title>
        </head><body><p>The principal of the college is Doctor Quintrell, who
        joined in 2019 and leads the academic council.</p></body></html>"""),
    "/s/principal.php": ("text/html", """<html><head><title>Principal</title>
        </head><body><p>The principal of the college is Doctor Quintrell, who
        joined in 2019 and leads the academic council.</p></body></html>"""),
    "/s/admission.html": ("text/html", """<html><head><title>Admissions</title>
        </head><body><p>Admission to the BTech programme is through the state
        entrance examination; applications open every June.</p></body></html>"""),
    "/s/exam.html": ("text/html", """<html><head><title>Exam Calendar</title>
        </head><body><table><tr><th>Exam</th><th>Date</th></tr>
        <tr><td>Semester one finals</td><td>December tenth</td></tr>
        <tr><td>Semester two finals</td><td>May twentieth</td></tr>
        </table></body></html>"""),
    "/s/mechanical.html": ("text/html", "<html><body><p>MECHWORD gears and "
                           "engines are taught in this other department.</p>"
                           "</body></html>"),
    "/s/private/x.html": ("text/html", "<html><body><p>SECRETWORD should "
                          "never be fetched by anyone at all.</p></body></html>"),
    "/s/dept-qs.html": ("text/html", """<html><head><title></title></head><body>
        <ul><li><a href="#">People</a><ul>
            <li><a href="#">Faculty</a><ul>
                <li><a href="faculty-qs.html">Regular</a></li>
                <li><a href="adhoc-qs.html">Adhoc</a></li></ul></li>
            <li><a href="nts-qs.html">Staff</a></li></ul></li>
        <li><a href="vision-qs.html">Vision / Mission</a></li></ul>""" + _QS_HEAD +
        """<p>The department of quantum studies teaches qubits, entanglement and
        error correction to undergraduates and research scholars. It was
        started twenty years ago with a single laboratory and now runs an
        undergraduate programme, a masters programme and a doctoral research
        centre, with students working on algorithms, hardware and the theory
        of computation, and it hosts a yearly workshop open to everyone.</p>
        </body></html>"""),
    "/s/faculty-qs.html": ("text/html", "<html><body>" + _QS_HEAD + "</body></html>"),
    "/s/adhoc-qs.html": ("text/html", "<html><body>" + _QS_HEAD + "</body></html>"),
    "/s/nts-qs.html": ("text/html", "<html><body>" + _QS_HEAD + "</body></html>"),
    "/s/vision-qs.html": ("text/html", "<html><body>" + _QS_HEAD + """
        <p>Vision: to lead the world in quantum computing education and to
        train engineers who build fault tolerant machines.</p>
        <p>Mission: to teach the theory and practice of quantum information,
        to run laboratories where students build and test real devices, to
        work with industry on useful problems, and to keep the curriculum in
        step with research so that every graduate can contribute from the
        first day of their career in this young and fast moving field.</p>
        </body></html>"""),
    "/s/get-qs.php": ("application/json", """{
        "vision": "To lead the world in quantum computing education.",
        "hod": [{"dept_name": "QUANTUM STUDIES", "name": "Dr. Ilvana Brisk",
                 "designation": "Professor", "email": "ilvana@q.edu"}],
        "regular_teaching_staff": [{"name": "Dr. Orren Vask",
                                    "designation": "Professor"}],
        "adhoc_teaching_staff": [{"name": "Ms. Pell Ardane",
                                  "designation": "Asst. Professor"}],
        "non_teaching_staff": [{"name": "Mr. Tobrin Valesko",
                                "designation": "Lab Assistant",
                                "phone": "12345"}],
        "btech": [{"batch": "QS 2K23", "count": "60",
                   "students": "[{'name': 'SECRETSTUDENT'}]"}]}"""),
}

PAGES.update(SITE)
SLOW = {"/live/get-alumni.php": 1.0}
FAIL_ONCE = set()      # paths that answer 500 once (down at refresh time)

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
        if path in FAIL_ONCE:
            FAIL_ONCE.discard(path)
            self.send_response(500)
            self.end_headers()
            return
        if path in SLOW:
            time.sleep(SLOW[path])
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
        data = body if isinstance(body, bytes) else body.encode("utf-8")
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


def _pdf(lines):
    """A minimal one-page PDF whose text is `lines` (no lines = a page with
    nothing to extract, like a scan)."""
    ops = "BT /F1 12 Tf 72 720 Td 14 TL " + " ".join(
        f"({l}) Tj T*" for l in lines) + " ET"
    objs = ["<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            f"<< /Length {len(ops)} >>\nstream\n{ops}\nendstream",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = "%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n"
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += (f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n")
    return out.encode("latin-1")


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
        self.assertEqual(rules[0], ["/private/", "/cgi-bin", "/s/private/"])
        self.assertTrue(web_rag.robots_allows(rules, "http://h/a.html"))
        self.assertFalse(web_rag.robots_allows(rules, "http://h/private/x"))
        self.assertFalse(web_rag.robots_allows(rules, "http://h/cgi-bin/x"))
        allow_rules = web_rag.parse_robots(
            "User-agent: *\nDisallow: /p/\nAllow: /p/open\n")
        self.assertTrue(web_rag.robots_allows(allow_rules, "http://h/p/open/1"))
        self.assertFalse(web_rag.robots_allows(allow_rules, "http://h/p/x"))
        self.assertTrue(web_rag.robots_allows(([], []), "http://h/anything"))


class TestDataAndLinks(unittest.TestCase):
    """Pure helpers behind live lookups: data endpoints, JSON flattening,
    link labels and "check the website" detection."""

    def test_data_endpoints_from_inline_scripts(self):
        page = web_rag.parse_page(PAGES["/live/people.html"][1])
        eps = web_rag.find_data_endpoints(page["scripts"],
                                          "http://h.org/live/people.html")
        # literal params fold into the query; a JS variable can't be resolved
        self.assertEqual(eps, ["http://h.org/live/get-people.php?id=7"])
        js = ["fetch('/api/list.json'); $.getJSON(\"https://h.org/x.php\");"
              "fetch('https://elsewhere.com/a.php'); url: 'pic.png'"]
        self.assertEqual(web_rag.find_data_endpoints(js, "https://www.h.org/"),
                         ["https://www.h.org/api/list.json",
                          "https://h.org/x.php"])

    def test_parse_page_keeps_menu_anchor_text(self):
        page = web_rag.parse_page(PAGES["/live/"][1])
        self.assertIn(("people.html", "Staff"), page["anchors"])
        self.assertNotIn("Staff", page["text"])            # nav isn't content

    def test_json_sections_are_readable_and_private(self):
        import json
        data = json.loads(PAGES["/live/get-people.php"][1])
        secs = web_rag.json_to_sections(data, "http://h/get-people.php?id=7")
        # " (QS)" when an earlier test left Quantum Studies as the focus
        by_title = {s["title"].replace(" (QS)", ""): s["text"] for s in secs}
        nts = "Non teaching staff (lab staff, technical and office staff)"
        self.assertEqual(by_title["Quantum Studies — " + nts],
                         nts + ", 1 in total: Mr. Tobrin Valesko, "
                         "Lab Assistant.")
        self.assertIn("Dr. Ilvana Brisk, Professor",
                      by_title["Quantum Studies — Head of department HOD"])
        self.assertIn("To be excellent", by_title["Quantum Studies — Vision"])
        self.assertIn("count 60", by_title["Quantum Studies — Btech"])
        everything = " ".join(by_title) + " ".join(by_title.values())
        for bad in ("SECRETSTUDENT", "@q.edu", "12345", "U-1.jpg", "POPUPWORD",
                    "mr-tobrin", "Dr. ,"):
            self.assertNotIn(bad, everything)
        self.assertTrue(all(s["data"] for s in secs))

    def test_table_rows_become_sections(self):
        html = ("<table><tr><th>Sl. No.</th><th>Name</th><th>Department</th>"
                "<th>Email</th><th>Contact Number</th></tr>"
                "<tr><td>1</td><td>Dr. Adarsh S</td><td>Civil</td>"
                "<td>[email&#160;protected]</td><td>9446915388</td></tr>"
                "<tr><td>2</td><td>Dr. Sudheer A </td><td>Mechanical</td>"
                "<td>a@b.c</td><td>9447472142</td></tr></table>")
        page = web_rag.parse_page(html)
        self.assertIn("2 | Dr. Sudheer A | Mechanical", page["text"])
        rows = web_rag.table_rows(page["tables"], "http://h/hods", "HODs")
        self.assertEqual([r["text"] for r in rows],
                         ["Name: Dr. Adarsh S; Department: Civil.",
                          "Name: Dr. Sudheer A; Department: Mechanical."])
        self.assertEqual(rows[1]["url"], "http://h/hods#t0r1")
        self.assertEqual(rows[1]["embed"], "Department: Mechanical")
        self.assertEqual(page["table_heads"], [True])
        # no <th> row: rows stay rows, the first one is NOT column names
        plain = web_rag.parse_page(
            "<table><tr><td>Chairman</td><td>Janab A</td><td>Trust head</td></tr>"
            "<tr><td>Member</td><td>Janab B</td><td>Treasurer</td></tr>"
            "<tr><td>Member</td><td>Dr. C</td><td>Faculty member</td></tr></table>")
        self.assertEqual(plain["table_heads"], [False])
        got = web_rag.table_rows(plain["tables"], "u", "GB", plain["table_heads"])
        self.assertEqual(got[0]["text"], "Chairman | Janab A | Trust head.")
        self.assertNotIn("embed", got[0])
        # a two-row layout table and a stats box are not directories
        self.assertEqual(web_rag.table_rows([[["a", "b"], ["c", "d"]]],
                                            "u", "t"), [])
        self.assertEqual(web_rag.table_rows(
            [[["UG students", "877"], ["PG students", "48"],
              ["Staff", "37"]]], "u", "t"), [])

    def test_a_named_person_finds_their_row(self):
        saved = (web_rag._embed_fn, web_rag._index, dict(web_rag._settings))
        web_rag._embed_fn = fake_embed
        web_rag._settings.update(chunk_words=40, chunk_overlap=0, top_k=3,
                                 min_sim=0.01)
        try:
            web_rag._index = web_rag._build_index([
                {"url": "p", "title": "", "text": "zorp " * 20},
                {"url": "p#t0r1", "title": "PTA",
                 "text": "Name: Sri. Saheer H; Designation: Member from parents."},
                {"url": "r#btech/A/0", "title": "Btech student",
                 "text": "Name: Pranav P; Batch: A."},
                {"url": "r#btech/A/1", "title": "Btech student",
                 "text": "Name: Pranav Prasanth; Batch: A; Admission No: 9."},
                {"url": "r#btech/A/2", "title": "Btech student",
                 "text": "Name: Anu K; Batch: A."}])
            top = web_rag.retrieve("is pranav prasanth a student here")[0]
            self.assertEqual(top[0], "r#btech/A/1")      # not every Pranav
            self.assertEqual(top[2], web_rag.NAME_MATCH_SCORE)
            self.assertEqual(web_rag.retrieve("who is saheer")[0][0], "p#t0r1")
            # a short first name alone matches nobody in particular
            self.assertNotIn("r#btech/A/2",
                             [h[0] for h in web_rag.retrieve("is anu here")
                              if h[2] == web_rag.NAME_MATCH_SCORE])
        finally:
            web_rag._embed_fn, web_rag._index = saved[0], saved[1]
            web_rag._settings.clear()
            web_rag._settings.update(saved[2])

    @mock.patch.object(web_rag, "RELATIVE_CUT", 1.0)
    @mock.patch.object(web_rag, "SECTION_CUT", 1.0)
    def test_one_page_cannot_fill_the_top_k(self):
        saved = (web_rag._embed_fn, web_rag._index, dict(web_rag._settings))
        web_rag._embed_fn = fake_embed
        web_rag._settings.update(chunk_words=10, chunk_overlap=0, top_k=3,
                                 min_sim=0.01)
        try:
            web_rag._index = web_rag._build_index([
                {"url": "u1", "title": "", "text": " ".join(["zorp alpha"] * 30)},
                {"url": "u1#t0r1", "title": "", "text": "zorp alpha zorp"},
                {"url": "u2", "title": "", "text": "zorp beta gamma delta"}])
            # u1's text gets one slot, its table row another, then u2
            self.assertEqual([h[0] for h in web_rag.retrieve("zorp alpha")],
                             ["u1", "u1#t0r1", "u2"])
        finally:
            web_rag._embed_fn, web_rag._index = saved[0], saved[1]
            web_rag._settings.clear()
            web_rag._settings.update(saved[2])

    @mock.patch.object(web_rag, "RELATIVE_CUT", 1.0)
    @mock.patch.object(web_rag, "SECTION_CUT", 1.0)
    def test_best_page_comes_as_a_section_not_a_scrap(self):
        saved = (web_rag._embed_fn, web_rag._index, dict(web_rag._settings))
        web_rag._embed_fn = fake_embed
        web_rag._settings.update(chunk_words=10, chunk_overlap=2, top_k=2,
                                 min_sim=0.01, page_words=30)
        filler = " ".join(f"w{i}" for i in range(42))   # no short tail
        try:
            web_rag._index = web_rag._build_index([
                {"url": "u1", "title": "News",
                 "outline": ["Robot team wins", "Patent granted",
                             "New lab opens"],
                 "text": "zorp alpha one two three four five six seven "
                         "eight " + filler + " zorp beta nine ten eleven"},
                {"url": "u2", "title": "", "text": "zorp gamma delta"}])
            hits = web_rag.retrieve("zorp alpha beta")
            top = hits[0][1]
            # the headlines first, then the page's best passages in reading
            # order (a gap marked), not its single best chunk
            self.assertTrue(top.startswith("News: Headlines on this page: "
                                           "Robot team wins; Patent granted; "
                                           "New lab opens. Details: "), top)
            self.assertIn("zorp alpha", top)
            self.assertIn("zorp beta", top)
            self.assertLess(top.index("alpha"), top.index("beta"))
            self.assertIn(" … ", top)
            self.assertEqual(hits[1][0], "u2")     # others stay as passages
            # off: the old single chunk
            web_rag._settings["page_words"] = 0
            self.assertNotIn("Headlines",
                             web_rag.retrieve("zorp alpha beta")[0][1])
        finally:
            web_rag._embed_fn, web_rag._index = saved[0], saved[1]
            web_rag._settings.clear()
            web_rag._settings.update(saved[2])

    @mock.patch.object(web_rag, "RELATIVE_CUT", 1.0)
    def test_a_section_drops_much_weaker_extracts(self):
        saved = (web_rag._embed_fn, web_rag._index, dict(web_rag._settings))
        web_rag._embed_fn = fake_embed
        web_rag._settings.update(chunk_words=10, chunk_overlap=2, top_k=3,
                                 min_sim=0.01, page_words=40)
        try:
            web_rag._index = web_rag._build_index([
                {"url": "u1", "title": "",
                 "text": "zorp alpha beta one two three four five six seven "
                         "zorp alpha beta eight nine ten eleven twelve"},
                {"url": "u2", "title": "",
                 "text": "zorp other words entirely here and there"}])
            self.assertEqual([h[0] for h in web_rag.retrieve("zorp alpha beta")],
                             ["u1"])
        finally:
            web_rag._embed_fn, web_rag._index = saved[0], saved[1]
            web_rag._settings.clear()
            web_rag._settings.update(saved[2])

    def test_page_text_rejoins_chunks_without_overlap(self):
        saved = (web_rag._embed_fn, web_rag._index, dict(web_rag._settings))
        web_rag._embed_fn = fake_embed
        web_rag._settings.update(chunk_words=10, chunk_overlap=3)
        words = " ".join(f"w{i}" for i in range(31))   # no short tail
        try:
            idx = web_rag._build_index([{"url": "u", "title": "", "text": words}])
            self.assertGreater(len(idx["chunks"]), 2)
            self.assertEqual(web_rag.page_text(idx, "u"), words)
        finally:
            web_rag._embed_fn, web_rag._index = saved[0], saved[1]
            web_rag._settings.clear()
            web_rag._settings.update(saved[2])

    def test_headings_become_the_page_outline(self):
        page = web_rag.parse_page(
            "<html><head><title>Achievements</title></head><body>"
            "<h2>Achievements</h2><h2>Team Jager wins   karting race</h2>"
            "<p>Long story.</p><h4>𝐀 𝐠𝐢𝐚𝐧𝐭 𝐥𝐞𝐚𝐩</h4>"
            "<h4>'+data['achievements'][i].heading+'</h4>"
            "<h3>Team Jager wins karting race</h3>"
            "<footer><h3>Quick Links</h3></footer></body></html>")
        self.assertEqual(page["headings"],
                         ["Team Jager wins karting race", "A giant leap"])

    def test_expand_query(self):
        self.assertEqual(web_rag.expand_query("who is the HOD of cse?"),
                         "who is the HOD head of department HOD of cse? "
                         "computer science and engineering")
        self.assertEqual(web_rag.expand_query("tell me about me"),
                         "tell me about me")     # 2-letter keys never expand

    def test_link_label_spells_out_slugs(self):
        self.assertEqual(
            web_rag.link_label("https://h/nts-cse.php", ["Staff"]),
            "Staff | non teaching staff computer science and engineering")
        self.assertEqual(web_rag.link_label("https://h/staff-profile.php?id=9"),
                         "staff profile")

    def test_website_request(self):
        req = web_rag.website_request
        self.assertEqual(req("look into the website"), (True, ""))
        self.assertEqual(req("check online"), (True, ""))
        self.assertEqual(req("can you check the website for the fee structure"),
                         (True, "fee structure"))
        self.assertEqual(req("website"), (True, ""))
        self.assertEqual(req("what is the college website address"),
                         (False, ""))
        self.assertEqual(req("who is the principal"), (False, ""))
        offer = "Shall I check the college website for that?"
        self.assertEqual(req("yes please", offer), (True, ""))
        self.assertEqual(req("yes please", "Do you like robots?"), (False, ""))
        self.assertEqual(req("yes I study here", offer), (False, ""))


class _SiteCase(unittest.TestCase):
    """The /s/ test college, configured like tkmce.ac.in: home + focus
    department as seeds, "qs" as the focus department's URL word."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="luna-test-webrag-site-")
        self.srv, base = _start_server()
        self.base = base + "s/"
        _Handler.requests = []
        web_rag.configure(
            url=self.base, seeds=[self.base, self.base + "dept-qs.html"],
            focus_name="Quantum Studies", focus_slugs=["qs"],
            skip_sections=["departments"], skip_slugs=["mechanical", "mech"],
            cache_dir=self.tmp, max_pages=50, workers=4, chunk_words=40,
            chunk_overlap=10, top_k=3, min_sim=0.05, route_topics=5,
            route_min=0.05, online_timeout=0.5, fetch_timeout=3.0,
            refresh_hours=24, live_pages=2, live_timeout=5.0,
            links_file="",                # never the real college list here
            pdfs=False, rosters=False, data_hosts=[])

    def tearDown(self):
        web_rag.wait(5)
        try:
            _stop_server(self.srv)
        except Exception:
            pass
        web_rag.configure(url="")          # drop the index for the next test
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _requests(self):
        with _Handler.lock:
            return list(_Handler.requests)

    def _build(self):
        web_rag.start(fake_embed, background=False)
        self.assertTrue(web_rag.refresh(force=True), web_rag.status())
        _Handler.requests = []


class TestCatalog(_SiteCase):
    def test_menu_links_carry_their_menu_path(self):
        links = web_rag.menu_links(PAGES["/s/"][1], self.base)
        self.assertIn((self.base + "admission.html", "UG/PG",
                       ["ACADEMICS", "Admissions"]), links)
        self.assertIn((self.base + "principal.html", "Principal",
                       ["ADMINISTRATION"]), links)
        self.assertFalse(any("other.example" in u for u, _, _ in links))

    def test_classify(self):
        c = web_rag.classify
        self.assertEqual(c(self.base + "nts-qs.html", ["Staff"], ["People"]),
                         ("focus", ""))
        self.assertEqual(c(self.base + "dept-qs.html", ["Quantum Studies"],
                           ["DEPARTMENTS"]), ("focus", ""))
        self.assertEqual(c(self.base + "mechanical.html",
                           ["Mechanical Engineering"], ["DEPARTMENTS"]),
                         (None, "other department"))
        self.assertEqual(c(self.base + "exam.html", ["Exam Calendar"],
                           ["ACADEMICS"]), ("general", ""))
        self.assertEqual(c(self.base + "uploads/brochure.pdf")[0], None)

    def test_verify_reports_every_link(self):
        report = web_rag.verify(print_report=False)
        st = lambda page: report[self.base + page]
        self.assertTrue(st("about.html").startswith("[general] ok"))
        self.assertTrue(st("nts-qs.html").startswith("[focus]"))
        self.assertIn("duplicate of", st("principal.php"))
        self.assertIn("broken (HTTP 404)", st("broken.html"))
        self.assertIn("blocked by robots.txt", st("private/x.html"))
        self.assertEqual(st("mechanical.html"), "skipped: other department")
        self.assertTrue(st("uploads/brochure.pdf").startswith("skipped: file"))
        reqs = self._requests()
        self.assertNotIn("/s/private/x.html", reqs)       # never fetched
        self.assertNotIn("/s/mechanical.html", reqs)      # out of scope
        self.assertEqual(reqs.count("/s/get-qs.php?id=7"), 1)   # JSON once


class TestCuratedLinks(_SiteCase):
    """data/website_links.txt: a hand-picked, grouped list of pages that are
    always indexed, each routed under its group."""

    def _links(self, text):
        path = os.path.join(self.tmp, "links.txt")
        with open(path, "w") as f:
            f.write(text)
        web_rag.configure(links_file=path)
        self.addCleanup(web_rag.configure, links_file="")
        return path

    def test_parse_links(self):
        got = web_rag.parse_links("""# the college's important pages
            https://a.example/            # before any group
            [Main]
            https://a.example/
            [Administration > General]
            https://a.example/governing-body
            https://a.example/governing-body
            not a link
            https://a.example/x.php
            """)
        self.assertEqual(got, [
            ("https://a.example/", "", ""),
            ("https://a.example/", "Main", ""),        # listed in two groups
            ("https://a.example/governing-body", "Administration > General", ""),
            ("https://a.example/x.php", "Administration > General", "")])

    def test_labels_and_pages_in_several_groups(self):
        got = web_rag.parse_links("""[CSE > Students]
            https://a.example/student-achievements-cse.php
            [CSE > Others]
            https://a.example/student-chapter-cse.php   Student Chapters
            [CSE > Achievements]
            https://a.example/student-achievements-cse.php""")
        self.assertEqual(got, [
            ("https://a.example/student-achievements-cse.php", "CSE > Students", ""),
            ("https://a.example/student-chapter-cse.php", "CSE > Others",
             "Student Chapters"),
            ("https://a.example/student-achievements-cse.php",
             "CSE > Achievements", "")])

    def test_every_group_of_a_page_is_a_path(self):
        self._links(f"[About]\n{self.base}about.html   About us\n"
                    f"[History]\n{self.base}about.html\n")
        catalog, _ = web_rag.discover()
        info = catalog[self.base + "about.html"]
        self.assertEqual(info["paths"][:2], ["About", "History"])
        self.assertIn("About us", info["texts"])

    def test_curated_pages_join_the_catalog_under_their_group(self):
        self._links(f"[About]\n{self.base}about.html\n"
                    f"[Departments]\n{self.base}mechanical.html\n")
        catalog, skipped = web_rag.discover()
        self.assertEqual(catalog[self.base + "about.html"]["paths"][0], "About")
        mech = catalog[self.base + "mechanical.html"]     # no longer skipped
        self.assertEqual(mech["scope"], "other")
        self.assertNotIn(self.base + "mechanical.html", skipped)

    def test_other_department_answers_only_when_named(self):
        self._links(f"[Departments]\n{self.base}mechanical.html\n")
        self._build()
        text = lambda q: " ".join(t for _, t, _ in web_rag.retrieve(q, top_k=5))
        self.assertNotIn("MECHWORD", text("gears and engines taught"))
        self.assertIn("MECHWORD", text("mechanical gears and engines taught"))

    def test_another_tkm_site_is_indexed_but_kept_apart(self):
        PAGES["/s/sister.html"] = ("text/html",
            "<html><head><title>Sister Institute</title></head><body><p>"
            "ZEBRAWORD the sister institute teaches gears and engines too, "
            "on its own campus with its own staff and students."
            "</p></body></html>")
        self.addCleanup(PAGES.pop, "/s/sister.html", None)
        other = self.base.replace("127.0.0.1", "localhost") + "sister.html"
        self._links(f"[TKM institutions]\n{other}\n")
        self._build()
        scope = {t["url"]: t["scope"] for t in web_rag._index["links"]}
        self.assertEqual(scope.get(other), "affiliate")
        text = lambda q: " ".join(t for _, t, _ in web_rag.retrieve(q, top_k=5))
        self.assertIn("ZEBRAWORD", text("sister institute"))
        self.assertNotIn("ZEBRAWORD", text("when are the exams"))

    def test_changing_the_list_triggers_a_rebuild(self):
        path = self._links(f"[About]\n{self.base}about.html\n")
        self._build()
        self.assertFalse(web_rag._is_stale(web_rag._index))
        with open(path, "a") as f:
            f.write(f"[Exams]\n{self.base}exam.html\n")
        self.assertTrue(web_rag._is_stale(web_rag._index))


class TestRelatedSearch(unittest.TestCase):
    """Second hop: passages elsewhere on the site about the people / things a
    question (or its best answer) names — a faculty member's achievements, a
    student's placement — found from the data, not a hard-coded page map."""

    PAGES = {
        "u/faculty":  ("Faculty", "Head of department HOD, 1 in total: "
                       "Dr. MIRA K VANCE, Associate Professor. Dr. OREN DALE, "
                       "Assistant Professor."),
        "u/achieve":  ("Faculty achievements", "Prof. Oren Dale was awarded "
                       "a Ph.D. in 2019. Prof. Mira Vance received the Best "
                       "Teacher award in 2019. Prof. Mira Vance won the "
                       "national innovation grant in 2025."),
        "u/students": ("Student achievements", "Lena Fox got placed in "
                       "Zentrix with a package of 12 lakhs. The students "
                       "attended a workshop."),
        "u/labs":     ("Labs", "The labs are open to all students. Computer "
                       "labs have sixty systems for the students. Zentrix "
                       "donated two servers to the labs."),
        "u/fest":     ("Arts fest", "Prof. Mira Vance was the convenor of "
                       "the arts fest."),
    }

    def setUp(self):
        self.idx = {"chunks": [{"url": u, "page": u, "title": t, "text": x}
                               for u, (t, x) in self.PAGES.items()],
                    "links": [{"url": u, "scope": "general" if u == "u/fest"
                               else "focus"} for u in self.PAGES]}

    def related(self, q, hits=()):
        return web_rag._related(self.idx, q, list(hits), now_year=2026)

    def text(self, q, hits=()):
        return " ".join(t for _, t, _ in self.related(q, hits))

    def test_a_named_person_brings_their_achievements(self):
        out = self.text("tell me about mira vance")
        self.assertIn("national innovation grant in 2025", out)
        self.assertNotIn("Oren Dale", out)

    def test_recent_first(self):
        out = self.text("what has mira vance done")
        self.assertLess(out.index("2025"), out.index("2019"))

    def test_hop_from_the_answer(self):
        hod = ("u/faculty", "Faculty: Head of department HOD, 1 in total: "
                            "Dr. MIRA K VANCE, Associate Professor.", 0.8)
        out = self.text("who is the hod", [hod])
        self.assertIn("national innovation grant", out)

    def test_the_department_s_own_pages_come_first(self):
        out = self.text("tell me about mira vance")
        fest = out.find("convenor")             # may be past the cap entirely
        self.assertTrue(fest == -1 or out.index("innovation grant") < fest)

    def test_sentences_sharing_the_question_s_words_come_first(self):
        out = self.text("did anyone get placed in zentrix")
        self.assertLess(out.index("Lena Fox"), out.index("donated"))

    def test_a_long_row_is_cut_around_the_name(self):
        row = " | ".join(f"Talk {i} by Speaker{i}" for i in range(40))
        self.idx["chunks"].append({"url": "u/talks", "page": "u/talks",
                                   "title": "Talks", "text": row +
                                   " | Talk 41 by Mira Vance on robots"})
        out = self.text("what talks did mira vance give")
        self.assertIn("Mira Vance on robots", out)

    def test_no_hop_unless_the_question_asks_who(self):
        hod = ("u/faculty", "Faculty: Head of department HOD, 1 in total: "
                            "Dr. MIRA K VANCE, Associate Professor.", 0.8)
        self.assertEqual(self.related("where is the department office", [hod]),
                         [])

    def test_sentences_one_by_one(self):
        out = web_rag._related_sentences(self.idx, "tell me about mira vance",
                                         [], now_year=2026)
        self.assertTrue(all(len(r) == 3 for r in out))
        self.assertIn("national innovation grant in 2025", out[0][2])

    def test_staff_sections(self):
        idx = {"chunks": [
            {"url": "u/staff#regular_teaching_staff", "page": "u/staff",
             "title": "CSE — Regular teaching staff",
             "text": "Regular teaching staff, 1 in total: Dr. MIRA K VANCE, Professor"},
            {"url": "u/staff#hod", "page": "u/staff", "title": "CSE — HOD",
             "text": "Head of department HOD, 1 in total: Dr. MIRA K VANCE"},
            {"url": "u/news#dept_news", "page": "u/news", "title": "News",
             "text": "Dept news, 3 in total: something"}], "links": []}
        with mock.patch.object(web_rag, "_index", idx):
            got = web_rag.staff_sections()
        self.assertEqual([t for t, _ in got], ["CSE — Regular teaching staff",
                                               "CSE — HOD"])

    def test_things_too_not_just_people(self):
        self.assertIn("Lena Fox", self.text("did anyone get placed in zentrix"))

    def test_ordinary_words_are_not_names(self):
        self.assertEqual(self.related("what labs are there for students"), [])

    def test_nothing_already_in_the_answer_is_repeated(self):
        hit = ("u/students", "Student achievements: Lena Fox got placed in "
                             "Zentrix with a package of 12 lakhs.", 0.9)
        out = self.text("who got placed in zentrix", [hit])
        self.assertNotIn("Lena Fox", out)              # already said
        self.assertIn("donated two servers", out)      # still related


class TestNewSources(_SiteCase):
    """Data from another host (the college's ETLab), class lists, the PDFs a
    page links to, and the one-.txt-per-link mirror."""

    ROSTER = """{"btech": [{"batch": "QS 2K23", "count": "2", "students": [
        {"sno": 1, "id": "7", "name": "VELLAN  ORSIK", "admission_no": "230011"},
        {"sno": 2, "id": "8", "name": "MIRA TOLLEN", "admission_no": "230012"}]}],
        "publications": [%s]}""" % ", ".join(
            '{"staff": "Dr. Orren Vask", "publication": "Paper %d"}' % i
            for i in range(60))

    def setUp(self):
        super().setUp()
        port = self.base.split(":")[2].split("/")[0]
        self.other = f"http://localhost:{port}/"
        self._swap("/s/about.html", ("text/html", """<html><head>
            <title>About</title></head><body><p>Zorblax college was founded
            long ago by people who liked teaching very much indeed.</p>
            <p><a href="files/placement stats.pdf">Placement statistics</a>
            <a href="files/scan.pdf">Scanned notice</a></p>
            <script>$.ajax({url: '%sx/roster.php', data: {id: 2}});</script>
            </body></html>""" % self.other))
        self._swap("/x/roster.php", ("application/json", self.ROSTER))
        self._swap("/s/files/placement%20stats.pdf", ("application/pdf", _pdf(
            ["Placement statistics 2024-25", "Ninety one students were placed",
             "in forty companies with a highest package of 44 lakhs per year",
             "and a median package of six lakhs per year in that batch"])))
        self._swap("/s/files/scan.pdf", ("application/pdf", _pdf([])))
        web_rag.configure(data_hosts=[self.other.split("//")[1].rstrip("/")],
                          pdfs=True, rosters=True, pages_txt=True)
        self.addCleanup(web_rag.configure, pdfs=False, rosters=False,
                        data_hosts=[], pages_txt=False)

    def _swap(self, path, value):
        old = PAGES.get(path)
        PAGES[path] = value
        self.addCleanup(lambda: PAGES.__setitem__(path, old) if old
                        else PAGES.pop(path, None))

    def test_data_endpoint_on_a_data_host(self):
        js = [f"$.ajax({{url: '{self.other}x/roster.php', data: {{id: 2}}}})",
              "fetch('https://elsewhere.com/a.php')"]
        self.assertEqual(web_rag.find_data_endpoints(js, self.base + "a.html"),
                         [self.other + "x/roster.php?id=2"])

    def test_class_lists_by_student_or_counts_only(self):
        import json
        data = json.loads(self.ROSTER)
        secs = web_rag.json_to_sections(data, "http://h/r.php", "QS")
        head = [s for s in secs if s["url"] == "http://h/r.php#btech"][0]
        self.assertEqual(head["text"], "Number of B.Tech students: 2, in 1 "
                                       "batches (QS 2K23, 2 students).")
        rows = [s for s in secs if s.get("topic") == "http://h/r.php#btech"]
        self.assertEqual([r["text"] for r in rows],
                         ["Name: Vellan Orsik; Batch: QS 2K23; Admission No: 230011.",
                          "Name: Mira Tollen; Batch: QS 2K23; Admission No: 230012."])
        self.assertFalse(any(r.get("data") for r in rows))     # not topics
        pubs = [s for s in secs if s["url"].endswith("#publications")][0]
        self.assertIn("Paper 59", pubs["text"])                # all, not 40
        web_rag.configure(rosters=False)
        secs = web_rag.json_to_sections(data, "http://h/r.php", "QS")
        text = " ".join(s["text"] for s in secs)
        self.assertIn("Number of B.Tech students: 2", text)
        self.assertNotIn("Orsik", text)

    def test_pdfs_rosters_and_a_txt_per_link(self):
        self._build()
        rep = web_rag._index["report"]
        pdf = self.base + "files/placement%20stats.pdf"
        scan = self.base + "files/scan.pdf"
        self.assertTrue(rep[pdf].startswith("ok ("), rep.get(pdf))
        self.assertTrue(rep[scan].startswith("needs OCR"), rep.get(scan))
        chunks = web_rag._index["chunks"]
        placed = [c for c in chunks if c["url"] == pdf]
        self.assertTrue(placed and "44 lakhs" in placed[0]["text"])
        self.assertEqual(placed[0]["page"], self.base + "about.html")
        mira = [c for c in chunks if "Mira Tollen" in c["text"]]
        self.assertEqual(mira[0]["topic"], self.other + "x/roster.php?id=2#btech")
        topics = [t["url"] for t in web_rag._index["links"]]
        self.assertIn(mira[0]["topic"], topics)
        self.assertNotIn(mira[0]["url"], topics)          # rows aren't topics
        d = os.path.join(self.tmp, "pages")
        names = os.listdir(d)
        about = open(os.path.join(d, [n for n in names
                                      if n.endswith("about.html.txt")][0]),
                     encoding="utf-8").read()
        for want in ("Status: ok", "Mira Tollen", "=== PDF: placement stats",
                     "44 lakhs", "needs OCR"):
            self.assertIn(want, about)
        # skipped links get a file too, saying why
        mech = [n for n in names if "mechanical" in n]
        self.assertTrue(mech, names)
        self.assertIn("skipped", open(os.path.join(d, mech[0]),
                                      encoding="utf-8").read())


    def test_pdf_cap_reports_what_was_left_out(self):
        web_rag.configure(max_pdfs=1)
        self.addCleanup(web_rag.configure, max_pdfs=80)
        self._build()
        rep = web_rag._index["report"]
        got = [rep[u] for u in rep if u.endswith(".pdf")]
        self.assertEqual(sum(s.startswith("skipped: over WEB_RAG_MAX_PDFS")
                             for s in got), 1, got)

    def test_other_departments_data_is_marked(self):
        self.assertTrue(web_rag._other_department(
            "Mechanical Engineering — Btech student"))
        self.assertFalse(web_rag._other_department(
            "Quantum Studies — Btech"))           # the focus department
        self.assertFalse(web_rag._other_department("Placement statistics"))


class TestRouting(_SiteCase):
    def test_refresh_indexes_the_catalog(self):
        self._build()
        st = web_rag.status()
        self.assertEqual(st["n_broken"], 1)                # broken.html
        text = " ".join(c["text"] for c in web_rag._index["chunks"])
        for bad in ("SECRETWORD", "MECHWORD", "SECRETSTUDENT", "12345",
                    "ilvana@q.edu"):
            self.assertNotIn(bad, text)
        for good in ("Quintrell", "Tobrin Valesko", "December tenth",
                     "fault tolerant"):
            self.assertIn(good, text)
        # shells (faculty/adhoc/staff pages) count as fetched: no live refetch
        known = web_rag._known_urls(web_rag._index)
        self.assertIn(self.base + "nts-qs.html", known)

    def test_questions_are_routed_to_their_page(self):
        self._build()
        top = lambda q: web_rag.route(q)[0][0]
        self.assertEqual(top("exam calendar dates"), self.base + "exam.html")
        self.assertIn(top("who is the principal"),
                      (self.base + "principal.html", self.base + "principal.php"))
        self.assertTrue(top("non teaching staff").endswith("#non_teaching_staff"))
        hits = web_rag.retrieve("when are the semester exam finals")
        self.assertIn("December tenth", " ".join(t for _, t, _ in hits))

    def test_misspelt_questions_are_corrected(self):
        self._build()
        self.assertEqual(web_rag.correct_spelling("examm calender dates"),
                         "examm calendar dates")       # 5 letters: left alone
        self.assertEqual(web_rag.correct_spelling("admision for btech"),
                         "admission for btech")
        self.assertEqual(web_rag.correct_spelling("what is photosynthesis"),
                         "what is photosynthesis")     # nothing close: kept
        self.assertEqual(web_rag.correct_spelling("admissions please"),
                         "admissions please")          # a form of a site word
        self.assertEqual(web_rag.route("admision")[0][0],
                         self.base + "admission.html")

    @mock.patch.object(web_rag, "RELATIVE_CUT", 1.0)
    def test_faculty_group_brings_regular_and_adhoc(self):
        self._build()
        hits = " ".join(t for _, t, _ in web_rag.retrieve("regular faculty"))
        self.assertIn("Orren Vask", hits)                  # regular
        self.assertIn("Pell Ardane", hits)                 # adhoc, same group

    def test_boilerplate_lines_are_stripped(self):
        with mock.patch.object(web_rag, "BOILERPLATE_PAGES", 2):
            self._build()
        vision = [c["text"] for c in web_rag._index["chunks"]
                  if c["url"].endswith("vision-qs.html")]
        self.assertTrue(vision and vision[0].startswith("Vision"), vision)

    def test_every_page_gets_a_readable_txt_mirror(self):
        was = web_rag._settings["pages_txt"]
        web_rag.configure(pages_txt=True)
        try:
            self._build()
            d = os.path.join(self.tmp, "pages")
            names = os.listdir(d)
            self.assertTrue(names and all(n.endswith(".txt") for n in names))
            about = [n for n in names if n.endswith("about.html.txt")]
            self.assertEqual(len(about), 1, names)
            with open(os.path.join(d, about[0]), encoding="utf-8") as f:
                body = f.read()
            self.assertIn(self.base + "about.html", body)
            self.assertIn("Zorblax", body)
            # a page that left the index loses its file on the next save
            stale = os.path.join(d, "gone.example_x.txt")
            open(stale, "w").close()
            web_rag._save_cache(web_rag._index)
            self.assertFalse(os.path.exists(stale))
        finally:
            web_rag.configure(pages_txt=was)

    def test_cache_survives_restart_offline(self):
        self._build()
        web_rag.configure(url="")
        web_rag.configure(url=self.base)
        _stop_server(self.srv)
        web_rag.start(fake_embed)                          # offline now
        self.assertTrue(web_rag.wait(5))
        self.assertFalse(web_rag.status()["online"])
        self.assertIn("Quintrell", " ".join(
            t for _, t, _ in web_rag.retrieve("who is the principal")))

    def test_routed_page_missing_from_cache_is_fetched_live(self):
        FAIL_ONCE.add("/s/admission.html")                 # down at refresh
        self._build()
        self.assertNotIn(self.base + "admission.html",
                         web_rag._known_urls(web_rag._index))
        hits = web_rag.lookup("admission btech applications june")
        self.assertIn("/s/admission.html", self._requests())
        self.assertIn("state entrance", " ".join(t for _, t, _ in hits))
        _Handler.requests = []                             # now cached
        web_rag.lookup("admission btech applications june")
        self.assertNotIn("/s/admission.html", self._requests())

    def test_background_refresh_when_online_and_no_cache(self):
        web_rag.start(fake_embed)
        self.assertTrue(web_rag.wait(15))
        self.assertTrue(web_rag.available())
        self.assertTrue(web_rag.status()["online"])

    def test_cache_with_other_dim_is_discarded(self):
        self._build()
        web_rag.configure(url="")
        web_rag.configure(url=self.base)
        other = lambda texts: fake_embed(texts)[:, :64]    # another model
        web_rag.start(other, background=False)
        self.assertFalse(web_rag.available())

    def test_unreachable_host_never_raises(self):
        empty = tempfile.mkdtemp(prefix="luna-test-webrag-empty-")
        try:
            web_rag.configure(url="http://127.0.0.1:1/", seeds=[],
                              cache_dir=empty)
            web_rag.start(fake_embed)
            self.assertTrue(web_rag.wait(10))
            self.assertFalse(web_rag.available())
            self.assertEqual(web_rag.retrieve("anything"), [])
            self.assertEqual(web_rag.lookup("anything"), [])
            self.assertFalse(web_rag.refresh(force=True))
            self.assertIsNotNone(web_rag.status()["last_error"])
        finally:
            shutil.rmtree(empty, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
