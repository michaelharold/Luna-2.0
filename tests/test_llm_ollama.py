# tests/test_llm_ollama.py
"""Unit tests for the Ollama LLM backend and the Modelfile builder — pure
logic, the HTTP layer is a fake server swapped in for llm_ollama's opener
that serves canned NDJSON:

    python -m unittest discover tests -v

Covers: streaming delta assembly + stop-at-done, stats capture, the warmup
load call carrying the SAME options as a real question (else Ollama reloads
the model on the first answer), the "think" flag being sent only to
thinking-capable models, graceful failure when the server is down and the
three recovery paths (boot-time backoff in warmup(), the throttled background
re-warm from available(), the synchronous retry in stream()), the proxy-free
transport, model resolution (custom → base → none, explicit override), and
make_luna_model's Modelfile rendering.
"""

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
import urllib.parse
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import llm_ollama  # noqa: E402
import persona     # noqa: E402


# ── A fake Ollama server behind llm_ollama._opener ────────────────────────────

class FakeResponse:
    """Stand-in for urllib's HTTPResponse: context manager, line iterator
    (what the streaming path uses) and read() (the non-streaming path)."""

    def __init__(self, lines):
        self._lines = [l if isinstance(l, bytes) else l.encode("utf-8")
                       for l in lines]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._lines)

    def read(self):
        return b"".join(self._lines)


def ndjson(*objs):
    return [json.dumps(o) + "\n" for o in objs]


DONE = {"message": {"role": "assistant", "content": ""}, "done": True,
        "eval_count": 42, "eval_duration": 3_100_000_000,
        "prompt_eval_count": 210, "prompt_eval_duration": 900_000_000,
        "load_duration": 12_000_000, "total_duration": 4_100_000_000}


def delta(text):
    return {"message": {"role": "assistant", "content": text}, "done": False}


class FakeServer:
    """Looks like an OpenerDirector (open(req, timeout)); routes by path and
    records every JSON body sent."""

    def __init__(self, models=("luna:latest",), capabilities=("completion",),
                 chat_lines=None, fail=False):
        self.models = list(models)
        self.capabilities = list(capabilities)
        self.chat_lines = chat_lines or ndjson(delta("Hi"), DONE)
        self.fail = fail
        self.requests = []            # [(path, payload-or-None)]

    def open(self, req, timeout=None):
        if self.fail:
            raise urllib.error.URLError("connection refused")
        path = urllib.parse.urlparse(req.full_url).path
        payload = json.loads(req.data) if req.data else None
        self.requests.append((path, payload))
        if path == "/api/tags":
            return FakeResponse(ndjson(
                {"models": [{"name": n} for n in self.models]}))
        if path == "/api/show":
            return FakeResponse(ndjson({"capabilities": self.capabilities}))
        if path == "/api/chat":
            if payload.get("messages") == []:          # the warmup load call
                return FakeResponse(ndjson({"done": True,
                                            "done_reason": "load"}))
            return FakeResponse(self.chat_lines)
        raise AssertionError(f"unexpected path {path}")

    def chat_payloads(self):
        """Bodies of the real (non-empty) chat requests."""
        return [p for path, p in self.requests
                if path == "/api/chat" and p.get("messages")]

    def load_payloads(self):
        """Bodies of the warmup load calls (empty messages)."""
        return [p for path, p in self.requests
                if path == "/api/chat" and p.get("messages") == []]


MESSAGES = [{"role": "user", "content": "hello"}]


class OllamaCase(unittest.TestCase):
    """Common plumbing: reset the module, install the fake server, make
    warmup() single-shot (no boot backoff) unless a test opts back in, and
    mute the "[llm] ..." prints so the test run stays readable."""

    def setUp(self):
        llm_ollama.set_model(None)
        for name, value in (("WARMUP_BOOT_WAIT", 0.0),):
            patcher = mock.patch.object(llm_ollama, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self._quiet = contextlib.redirect_stdout(io.StringIO())
        self._quiet.__enter__()
        self.addCleanup(self._quiet.__exit__, None, None, None)

    def serve(self, **kw):
        server = FakeServer(**kw)
        patcher = mock.patch.object(llm_ollama, "_opener", server)
        patcher.start()
        self.addCleanup(patcher.stop)
        return server


class TestStreaming(OllamaCase):
    def test_concatenates_deltas_and_stops_at_done(self):
        after_done = delta("NOT THIS")
        self.serve(chat_lines=ndjson(delta("Hel"), delta("lo"), delta(" there"),
                                     DONE, after_done))
        llm_ollama.warmup()
        self.assertTrue(llm_ollama.available())
        self.assertEqual("".join(llm_ollama.stream(MESSAGES)), "Hello there")
        self.assertEqual(llm_ollama.ask(MESSAGES), "Hello there")

    def test_stats_recorded_from_final_line(self):
        self.serve()
        llm_ollama.warmup()
        self.assertEqual(llm_ollama.last_stats(), {})
        list(llm_ollama.stream(MESSAGES))
        stats = llm_ollama.last_stats()
        self.assertEqual(stats["eval_count"], 42)
        self.assertEqual(stats["eval_duration"], 3_100_000_000)
        self.assertEqual(stats["prompt_eval_count"], 210)
        self.assertEqual(stats["prompt_eval_duration"], 900_000_000)
        self.assertEqual(stats["load_duration"], 12_000_000)
        self.assertEqual(stats["total_duration"], 4_100_000_000)

    def test_request_carries_options_and_keep_alive(self):
        server = self.serve()
        llm_ollama.warmup()
        list(llm_ollama.stream(MESSAGES))
        payload = server.chat_payloads()[-1]
        self.assertEqual(payload["model"], "luna")
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["keep_alive"], llm_ollama.OLLAMA_KEEP_ALIVE)
        self.assertEqual(payload["options"]["num_ctx"], llm_ollama.OLLAMA_NUM_CTX)
        self.assertEqual(payload["options"]["num_thread"],
                         llm_ollama.OLLAMA_NUM_THREAD)
        self.assertEqual(payload["options"]["num_predict"],
                         llm_ollama.LLM_MAX_TOKENS)
        self.assertEqual(payload["options"]["temperature"],
                         llm_ollama.LLM_TEMPERATURE)

    def test_error_line_stops_stream(self):
        self.serve(chat_lines=ndjson({"error": "model exploded"}, delta("x")))
        llm_ollama.warmup()
        self.assertEqual(list(llm_ollama.stream(MESSAGES)), [])
        self.assertIsNone(llm_ollama.ask(MESSAGES))
        self.assertTrue(llm_ollama.available())   # one bad reply ≠ dead server


class TestWarmupLoad(OllamaCase):
    def test_load_call_carries_same_options_as_a_question(self):
        """Ollama reloads the model when runner options (num_ctx, num_thread,
        ...) differ from the resident instance's — so the warmup load must
        send exactly what a real question sends, or the first answer after
        boot pays the load again."""
        server = self.serve()
        llm_ollama.warmup()
        list(llm_ollama.stream(MESSAGES))
        loads = server.load_payloads()
        self.assertEqual(len(loads), 1)
        load, chat = loads[0], server.chat_payloads()[-1]
        self.assertEqual(load["model"], chat["model"])
        self.assertEqual(load["keep_alive"], chat["keep_alive"])
        self.assertEqual(load["options"], chat["options"])
        for key in ("num_ctx", "num_thread", "temperature", "num_predict"):
            self.assertIn(key, load["options"])

    def test_load_call_precedes_availability(self):
        server = self.serve()
        llm_ollama.warmup()
        self.assertTrue(llm_ollama.available())
        self.assertEqual([path for path, _ in server.requests],
                         ["/api/tags", "/api/show", "/api/chat"])


class TestThinkFlag(OllamaCase):
    def test_sent_only_when_model_supports_thinking(self):
        server = self.serve(capabilities=("completion", "thinking"))
        llm_ollama.warmup()
        list(llm_ollama.stream(MESSAGES))
        self.assertIs(server.chat_payloads()[-1].get("think"), False)

    def test_not_sent_to_plain_models(self):
        server = self.serve(capabilities=("completion", "tools"))
        llm_ollama.warmup()
        list(llm_ollama.stream(MESSAGES))
        self.assertNotIn("think", server.chat_payloads()[-1])
        # and the warmup load call never carries it either
        load = server.load_payloads()
        self.assertEqual(len(load), 1)
        self.assertNotIn("think", load[0])


class TestFailure(OllamaCase):
    def test_server_down_never_raises(self):
        self.serve(fail=True)
        llm_ollama.warmup()                       # must not raise
        self.assertFalse(llm_ollama.available())
        self.assertIsNone(llm_ollama.model_name())
        self.assertEqual(list(llm_ollama.stream(MESSAGES)), [])
        self.assertIsNone(llm_ollama.ask(MESSAGES))
        self.assertFalse(llm_ollama.available())

    def test_stream_retries_warmup_after_cooldown(self):
        server = self.serve(fail=True)
        llm_ollama.warmup()
        self.assertFalse(llm_ollama.available())
        server.fail = False                        # the service came up
        with mock.patch.object(llm_ollama, "WARMUP_RETRY_SECS", 1e9):
            self.assertEqual(list(llm_ollama.stream(MESSAGES)), [])
        with mock.patch.object(llm_ollama, "WARMUP_RETRY_SECS", 0.0):
            self.assertEqual("".join(llm_ollama.stream(MESSAGES)), "Hi")
        self.assertTrue(llm_ollama.available())

    def test_generation_exception_yields_nothing(self):
        server = self.serve()
        llm_ollama.warmup()
        server.fail = True                         # dies mid-conversation
        self.assertEqual(list(llm_ollama.stream(MESSAGES)), [])

    def test_disabled_by_config(self):
        server = self.serve()
        with mock.patch.object(llm_ollama, "LLM_ENABLED", False):
            llm_ollama.warmup()
            self.assertFalse(llm_ollama.available())
        self.assertEqual(server.requests, [])


class TestRecovery(OllamaCase):
    """Luna only ever calls warmup() once (main.py, daemon thread) and then
    available() before each question — so those two must be enough to
    recover from an ollama.service that was not up yet at boot."""

    def _fake_sleep(self, server, come_up_after=None):
        """A time.sleep stand-in that records the backoff and, optionally,
        brings the fake server up after the N-th sleep."""
        calls = []

        def _sleep(secs):
            calls.append(secs)
            if come_up_after is not None and len(calls) >= come_up_after:
                server.fail = False
        patcher = mock.patch.object(llm_ollama, "_sleep", _sleep)
        patcher.start()
        self.addCleanup(patcher.stop)
        return calls

    def test_warmup_keeps_trying_while_server_is_unreachable(self):
        server = self.serve(fail=True)
        sleeps = self._fake_sleep(server, come_up_after=2)
        with mock.patch.object(llm_ollama, "WARMUP_BOOT_WAIT", 120.0):
            llm_ollama.warmup()
        self.assertTrue(llm_ollama.available())
        self.assertEqual(llm_ollama.model_name(), "luna")
        self.assertEqual(sleeps, [2.0, 4.0])      # exponential backoff
        self.assertEqual("".join(llm_ollama.stream(MESSAGES)), "Hi")

    def test_warmup_gives_up_after_boot_wait(self):
        server = self.serve(fail=True)
        sleeps = self._fake_sleep(server)          # never comes up
        with mock.patch.object(llm_ollama, "WARMUP_BOOT_WAIT", 5.0):
            llm_ollama.warmup()                    # must return, not spin
        self.assertFalse(llm_ollama.available())
        self.assertEqual(sleeps, [2.0, 4.0])      # 2 + 4 fit in 5 s; 8 would not

    def test_warmup_does_not_retry_other_failures(self):
        server = self.serve(models=("gemma2:2b",))     # server up, no model
        sleeps = self._fake_sleep(server)
        with mock.patch.object(llm_ollama, "WARMUP_BOOT_WAIT", 120.0), \
             mock.patch.object(llm_ollama, "OLLAMA_MODEL", "luna"), \
             mock.patch.object(llm_ollama, "OLLAMA_BASE_MODEL", "qwen2.5:1.5b"):
            llm_ollama.warmup()
        self.assertFalse(llm_ollama.available())
        self.assertEqual(sleeps, [])
        self.assertEqual([p for p, _ in server.requests], ["/api/tags"])

    def test_available_starts_background_rewarm_after_cooldown(self):
        server = self.serve(fail=True)
        llm_ollama.warmup()
        self.assertFalse(llm_ollama.available())   # inside cooldown: no retry
        server.fail = False                        # the service came up later
        self.assertFalse(llm_ollama.available())
        self.assertEqual(server.requests, [])      # ...and still no request
        with mock.patch.object(llm_ollama, "WARMUP_RETRY_SECS", 0.0):
            llm_ollama.available()                 # kicks it off, never blocks
            thread = llm_ollama._rewarm_thread
            self.assertIsNotNone(thread)
            thread.join(5.0)
        self.assertFalse(thread.is_alive())
        self.assertTrue(llm_ollama.available())
        self.assertEqual(llm_ollama.model_name(), "luna")
        self.assertEqual("".join(llm_ollama.stream(MESSAGES)), "Hi")

    def test_available_rewarm_is_single_and_throttled(self):
        server = self.serve(fail=True)
        llm_ollama.warmup()
        with mock.patch.object(llm_ollama, "WARMUP_RETRY_SECS", 0.0):
            llm_ollama.available()
            first = llm_ollama._rewarm_thread
            first.join(5.0)
        # still down: the next calls inside the cooldown spawn nothing new
        self.assertFalse(llm_ollama.available())
        self.assertFalse(llm_ollama.available())
        self.assertIs(llm_ollama._rewarm_thread, first)
        self.assertEqual(server.requests, [])

    def test_available_never_rewarms_when_disabled(self):
        server = self.serve()
        with mock.patch.object(llm_ollama, "LLM_ENABLED", False), \
             mock.patch.object(llm_ollama, "WARMUP_RETRY_SECS", 0.0):
            llm_ollama.warmup()
            before = llm_ollama._rewarm_thread
            self.assertFalse(llm_ollama.available())
            self.assertIs(llm_ollama._rewarm_thread, before)
        self.assertEqual(server.requests, [])


class TestTransport(unittest.TestCase):
    @staticmethod
    def _proxied(opener):
        return [h for h in opener.handlers
                if isinstance(h, urllib.request.ProxyHandler) and h.proxies]

    def test_opener_ignores_environment_proxies(self):
        with mock.patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:9"}):
            # sanity: a default opener (what urlopen uses) WOULD proxy
            self.assertTrue(self._proxied(urllib.request.build_opener()))
            # ours never routes through a proxy, whatever the environment
            self.assertEqual(self._proxied(llm_ollama._opener), [])

    def test_requests_go_through_the_opener_not_urlopen(self):
        with mock.patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:9",
                                          "HTTP_PROXY": "http://127.0.0.1:9"}), \
             mock.patch("urllib.request.urlopen",
                        side_effect=AssertionError("urlopen must not be used")), \
             mock.patch.object(llm_ollama._opener, "open",
                               return_value=FakeResponse(
                                   ndjson({"models": []}))) as opened:
            self.assertEqual(llm_ollama._json("/api/tags"), {"models": []})
        opened.assert_called_once()
        req = opened.call_args[0][0]
        self.assertTrue(req.full_url.startswith(llm_ollama.OLLAMA_HOST))


class TestModelResolution(OllamaCase):
    def test_custom_model_preferred(self):
        self.serve(models=("qwen2.5:1.5b", "luna:latest"))
        with mock.patch.object(llm_ollama, "OLLAMA_MODEL", "luna"), \
             mock.patch.object(llm_ollama, "OLLAMA_BASE_MODEL", "qwen2.5:1.5b"):
            llm_ollama.warmup()
        self.assertEqual(llm_ollama.model_name(), "luna")

    def test_falls_back_to_base_when_only_base_listed(self):
        self.serve(models=("qwen2.5:1.5b",))
        with mock.patch.object(llm_ollama, "OLLAMA_MODEL", "luna"), \
             mock.patch.object(llm_ollama, "OLLAMA_BASE_MODEL", "qwen2.5:1.5b"):
            llm_ollama.warmup()
        self.assertTrue(llm_ollama.available())
        self.assertEqual(llm_ollama.model_name(), "qwen2.5:1.5b")

    def test_unavailable_when_neither_listed(self):
        self.serve(models=("gemma2:2b",))
        with mock.patch.object(llm_ollama, "OLLAMA_MODEL", "luna"), \
             mock.patch.object(llm_ollama, "OLLAMA_BASE_MODEL", "qwen2.5:1.5b"):
            llm_ollama.warmup()
        self.assertFalse(llm_ollama.available())
        self.assertIsNone(llm_ollama.model_name())

    def test_set_model_override(self):
        server = self.serve(models=("gemma2:2b", "luna:latest"))
        llm_ollama.set_model("gemma2:2b")
        llm_ollama.warmup()
        self.assertEqual(llm_ollama.model_name(), "gemma2:2b")
        list(llm_ollama.stream(MESSAGES))
        self.assertEqual(server.chat_payloads()[-1]["model"], "gemma2:2b")

    def test_set_model_override_missing_is_unavailable(self):
        self.serve(models=("luna:latest",))
        llm_ollama.set_model("nope:7b")
        llm_ollama.warmup()
        self.assertFalse(llm_ollama.available())


# ── scripts/make_luna_model.py ────────────────────────────────────────────────

def _load_make_luna_model():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "scripts", "make_luna_model.py")
    spec = importlib.util.spec_from_file_location("make_luna_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestMakeLunaModel(unittest.TestCase):
    def setUp(self):
        self.mod = _load_make_luna_model()
        self.tmp = tempfile.mkdtemp(prefix="luna-test-kb-")
        self.kb = os.path.join(self.tmp, "knowledge.txt")
        with open(self.kb, "w") as f:
            f.write("When was the college established:It was established in 1958\n"
                    "\n"
                    'Quote test:She said """hi""" once\n'
                    "no separator on this line\n")

    def test_modelfile_contents(self):
        system_text = self.mod.build_system(bake=True, knowledge_path=self.kb)
        text = self.mod.render_modelfile("qwen2.5:1.5b", system_text)
        self.assertTrue(text.startswith("FROM qwen2.5:1.5b\n"))
        self.assertIn('SYSTEM """', text)
        self.assertIn(persona.PERSONALITY, text)
        self.assertIn("Q: When was the college established  "
                      "A: It was established in 1958", text)
        self.assertNotIn("no separator on this line", text)
        self.assertIn(f"PARAMETER temperature {self.mod.config.LLM_TEMPERATURE}",
                      text)
        self.assertIn(f"PARAMETER num_ctx {self.mod.config.OLLAMA_NUM_CTX}", text)
        self.assertIn(f"PARAMETER num_predict {self.mod.config.LLM_MAX_TOKENS}",
                      text)

    def test_triple_quotes_are_escaped(self):
        system_text = self.mod.build_system(bake=True, knowledge_path=self.kb)
        text = self.mod.render_modelfile("qwen2.5:1.5b", system_text)
        # only the two SYSTEM delimiters survive as a bare triple quote
        self.assertEqual(text.count('"""'), 2)
        self.assertIn('\\"\\"\\"hi\\"\\"\\"', text)

    def test_no_bake_is_persona_only(self):
        system_text = self.mod.build_system(bake=False, knowledge_path=self.kb)
        self.assertEqual(system_text, persona.PERSONALITY)
        self.assertNotIn("1958", self.mod.render_modelfile("x:y", system_text))

    def test_modelfile_path_is_file_safe(self):
        p = self.mod.modelfile_path("ns/luna:v2")
        self.assertTrue(p.endswith(os.path.join("models", "llm",
                                                "Modelfile.ns_luna_v2")))

    def test_dry_run_prints_modelfile(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = self.mod.main(["--dry-run", "--base", "gemma2:2b",
                                "--name", "luna-test", "--no-bake"])
        self.assertEqual(rc, 0)
        self.assertIn("FROM gemma2:2b", out.getvalue())
        self.assertIn(persona.PERSONALITY.splitlines()[0], out.getvalue())


if __name__ == "__main__":
    unittest.main()
