# tests/test_commands_realtime.py
"""Unit tests for commands.py's date/time/weather handlers.

commands.py itself is light to import (face.py/face_store.py never touch
pygame at import time — only the renderer does), so this runs on any machine:

    python -m unittest discover tests -v
"""

import http.server
import json
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import commands  # noqa: E402


class _FakeSpeak:
    """Swap in for commands._speak so tests never touch real TTS; records
    every utterance."""

    def __init__(self):
        self.said = []

    def __call__(self, text):
        self.said.append(text)


class TestDateTime(unittest.TestCase):
    def setUp(self):
        self._orig_speak = commands._speak
        self.speak = _FakeSpeak()
        commands._speak = self.speak

    def tearDown(self):
        commands._speak = self._orig_speak

    def test_date_question_consumed(self):
        self.assertTrue(commands._datetime_command("what day is it", set()))
        self.assertEqual(len(self.speak.said), 1)
        self.assertTrue(self.speak.said[0].startswith("It's "))
        self.assertNotIn(":", self.speak.said[0])   # date only, no clock time

    def test_time_question_consumed(self):
        self.assertTrue(commands._datetime_command("what's the time", set()))
        self.assertIn(":", self.speak.said[0])       # a clock time was spoken

    def test_both_date_and_time_when_both_asked(self):
        self.assertTrue(commands._datetime_command(
            "what's the date and time right now", set()))
        self.assertIn(":", self.speak.said[0])

    def test_unrelated_utterance_not_consumed(self):
        self.assertFalse(commands._datetime_command("volume up", set()))
        self.assertFalse(commands._datetime_command("who are you", set()))
        self.assertEqual(self.speak.said, [])

    def test_disabled_never_consumes(self):
        old = commands.ENABLE_DATETIME_COMMAND
        commands.ENABLE_DATETIME_COMMAND = False
        try:
            self.assertFalse(commands._datetime_command("what day is it", set()))
        finally:
            commands.ENABLE_DATETIME_COMMAND = old


class _WeatherHandler(http.server.BaseHTTPRequestHandler):
    response_body = {"current_weather": {"temperature": 24.4, "weathercode": 3}}
    status = 200

    def do_GET(self):
        data = json.dumps(self.response_body).encode("utf-8")
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class TestWeather(unittest.TestCase):
    def setUp(self):
        self._orig_speak = commands._speak
        self.speak = _FakeSpeak()
        commands._speak = self.speak
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _WeatherHandler)
        threading.Thread(target=lambda: self.srv.serve_forever(poll_interval=0.05),
                         daemon=True).start()
        self._orig_urlopen = commands.urllib.request.urlopen
        base = f"http://127.0.0.1:{self.srv.server_address[1]}"
        # Redirect the weather call at the transport level: same trick as
        # wiki_rag's tests, but urllib.request.urlopen instead of a rebuilt
        # base — commands.py builds the whole Open-Meteo URL inline, so the
        # simplest seam is to swap urlopen itself for one that ignores the
        # real host and hits our fake server instead.
        def fake_urlopen(url, timeout=None):
            return self._orig_urlopen(base, timeout=timeout)
        commands.urllib.request.urlopen = fake_urlopen

    def tearDown(self):
        commands._speak = self._orig_speak
        commands.urllib.request.urlopen = self._orig_urlopen
        self.srv.shutdown()
        self.srv.server_close()

    def test_weather_question_consumed_and_spoken(self):
        self.assertTrue(commands._weather_command("how is the weather today", set()))
        self.assertEqual(len(self.speak.said), 1)
        self.assertIn("24 degrees", self.speak.said[0])
        self.assertIn("overcast", self.speak.said[0])   # code 3

    def test_unrelated_utterance_not_consumed(self):
        self.assertFalse(commands._weather_command("what is the vision of the "
                                                    "department", set()))
        self.assertEqual(self.speak.said, [])

    def test_server_failure_gives_honest_reply_not_an_exception(self):
        commands.urllib.request.urlopen = lambda url, timeout=None: (_ for _ in ()).throw(
            OSError("connection refused"))
        self.assertTrue(commands._weather_command("what's the weather like", set()))
        self.assertIn("couldn't reach", self.speak.said[0].lower())

    def test_disabled_never_consumes(self):
        old = commands.ENABLE_WEATHER_COMMAND
        commands.ENABLE_WEATHER_COMMAND = False
        try:
            self.assertFalse(commands._weather_command("how's the weather", set()))
        finally:
            commands.ENABLE_WEATHER_COMMAND = old


if __name__ == "__main__":
    unittest.main()
