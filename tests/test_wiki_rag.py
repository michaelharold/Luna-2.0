# tests/test_wiki_rag.py
"""Unit tests for wiki_rag.py — the live general-knowledge fallback.

A tiny fake Wikipedia (search + summary endpoints) is served from a thread
(http.server on an ephemeral 127.0.0.1 port), and wiki_rag._api_base is
monkeypatched to point at it — no real network, no real Wikipedia, ever
touched:

    python -m unittest discover tests -v
"""

import http.server
import json
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import wiki_rag  # noqa: E402

SEARCH_RESULTS = {
    "capital of france": [{"title": "List of capitals of France"}],
    "quantum entanglement": [{"title": "Quantum entanglement"}],
    "no such article at all": [],
}

SUMMARIES = {
    "List_of_capitals_of_France": {
        "extract": "This is a chronological list of capitals of France. "
                   "The capital of France has been Paris since 1944. "
                   "A third sentence that should be dropped.",
        "content_urls": {"desktop": {"page": "https://fake.wiki/capitals"}},
    },
    "Quantum_entanglement": {
        "extract": "Quantum entanglement is a physical phenomenon.",
        "content_urls": {"desktop": {"page": "https://fake.wiki/entanglement"}},
    },
}


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/w/api.php"):
            from urllib.parse import urlsplit, parse_qs
            q = parse_qs(urlsplit(self.path).query).get("srsearch", [""])[0].lower()
            hits = SEARCH_RESULTS.get(q, [])
            self._json({"query": {"search": hits}})
        elif self.path.startswith("/api/rest_v1/page/summary/"):
            from urllib.parse import urlsplit
            title = urlsplit(self.path).path.rsplit("/", 1)[-1]
            body = SUMMARIES.get(title)
            if body is None:
                self.send_response(404)
                self.end_headers()
            else:
                self._json(body)
        else:
            self.send_response(404)
            self.end_headers()

    def _json(self, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):       # keep test output clean
        pass


def _start_server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=lambda: srv.serve_forever(poll_interval=0.05),
                         daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


class TestPureHelpers(unittest.TestCase):
    def test_strips_question_wrapper_to_keywords(self):
        self.assertEqual(wiki_rag._to_search_keywords("What is the capital of France?"),
                         "capital of France")
        self.assertEqual(wiki_rag._to_search_keywords("Who is the president?"),
                         "president")
        self.assertEqual(wiki_rag._to_search_keywords("Tell me about gravity"),
                         "gravity")
        self.assertEqual(wiki_rag._to_search_keywords("gravity"), "gravity")

    def test_domain_adjacent_terms_block_lookup(self):
        for q in ("Where is P.K.M. college located?",
                  "Tell me about the chemistry department",
                  "Who is the HOD?", "Who is the principal?",
                  "where is the library"):
            self.assertEqual(wiki_rag.retrieve(q), [], q)

    def test_disabled_returns_empty(self):
        old = wiki_rag.WIKI_RAG_ENABLED
        wiki_rag.WIKI_RAG_ENABLED = False
        try:
            self.assertEqual(wiki_rag.retrieve("what is gravity"), [])
        finally:
            wiki_rag.WIKI_RAG_ENABLED = old

    def test_blank_query_returns_empty(self):
        self.assertEqual(wiki_rag.retrieve(""), [])
        self.assertEqual(wiki_rag.retrieve("   "), [])


class TestRetrieveAgainstFakeWikipedia(unittest.TestCase):
    def setUp(self):
        self.srv, self.base = _start_server()
        self._orig_api_base = wiki_rag._api_base
        wiki_rag._api_base = lambda: self.base
        # Never let a test start a real connectivity probe.
        self._orig_state = wiki_rag.connectivity.state
        self._orig_mark = wiki_rag.connectivity.mark_offline
        self.online = True
        self.marked_offline = 0
        wiki_rag.connectivity.state = lambda: self.online
        wiki_rag.connectivity.mark_offline = self._mark

    def _mark(self):
        self.marked_offline += 1

    def tearDown(self):
        wiki_rag._api_base = self._orig_api_base
        wiki_rag.connectivity.state = self._orig_state
        wiki_rag.connectivity.mark_offline = self._orig_mark
        self.srv.shutdown()
        self.srv.server_close()

    def test_finds_and_summarizes_a_real_topic(self):
        hits = wiki_rag.retrieve("What is the capital of France?")
        self.assertEqual(len(hits), 1)
        src, text, score = hits[0]
        self.assertEqual(src, "https://fake.wiki/capitals")
        self.assertEqual(score, 1.0)
        # Only the first two sentences — the third is dropped.
        self.assertIn("Paris", text)
        self.assertNotIn("third sentence", text)

    def test_no_search_hit_returns_empty(self):
        self.assertEqual(wiki_rag.retrieve("no such article at all"), [])

    def test_unreachable_server_never_raises_and_marks_offline(self):
        wiki_rag._api_base = lambda: "http://127.0.0.1:1"
        self.assertEqual(wiki_rag.retrieve("what is gravity"), [])
        self.assertEqual(self.marked_offline, 1)

    def test_known_offline_skips_without_any_request(self):
        def boom():
            raise AssertionError("must not touch the network when offline")
        wiki_rag._api_base = boom
        self.online = False
        self.assertEqual(wiki_rag.retrieve("What is the capital of France?"), [])
        self.online = None                   # not probed yet: also skip
        self.assertEqual(wiki_rag.retrieve("What is the capital of France?"), [])

    def test_http_error_is_not_treated_as_offline(self):
        # The search hit exists but its summary 404s: Wikipedia answered, so
        # the device is online — only the lookup failed.
        SEARCH_RESULTS["missing summary"] = [{"title": "No Such Summary"}]
        try:
            self.assertEqual(wiki_rag.retrieve("missing summary"), [])
        finally:
            del SEARCH_RESULTS["missing summary"]
        self.assertEqual(self.marked_offline, 0)


if __name__ == "__main__":
    unittest.main()
