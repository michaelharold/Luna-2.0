# tests/test_face_cards.py
"""Full-screen cards in place of the face — date, time, person, greeting:

    python -m unittest tests.test_face_cards -v

  * face_cards draws each card (pure surfaces, no shared state)
  * CardTimer decides when a card is up: a fixed time, or "while Luna speaks
    the answer + a short linger", optionally after a delay
  * face.py's card API writes the request; RobotFace cross-fades face <-> card
  * a tap dismisses a card (and makes Luna angry, like any tap)
"""

import datetime
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

_tmpdir = tempfile.mkdtemp(prefix="luna-test-faces-")
import config  # noqa: E402
config.FACE_DB_DIR = _tmpdir
config.RENDER_FPS = 1_000_000

import numpy as np  # noqa: E402
import pygame  # noqa: E402

import face  # noqa: E402
import face_cards  # noqa: E402
import robot_face  # noqa: E402
from face_style import WIDTH, HEIGHT  # noqa: E402
from shared_state import state  # noqa: E402

robot_face.RENDER_FPS = config.RENDER_FPS
NOW = datetime.datetime(2026, 10, 12, 14, 7, 38)

_RESET = dict(face_override=None, face_override_until=0.0, speaking=False,
              tts_active=False, face_detected=False, face_x=0.5,
              audio_energy=0.0, banner_text="", banner_until=0.0,
              banner_since=0.0, recog_event_time=0.0, card=None,
              card_set_at=0.0, card_since=0.0, card_until=None,
              touch_enroll_request=False)


def _set(**kw):
    with state.lock:
        for k, v in kw.items():
            setattr(state, k, v)


def _px(surf, x, y):
    return tuple(surf.get_at((int(x), int(y))))[:3]


class FakeClock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def tick(self, secs=1.0 / 60.0):
        self.t += secs


# ══════════════════════════════════════════════════════════════════════════════
# Drawing
# ══════════════════════════════════════════════════════════════════════════════
class TestCardDrawing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        pygame.display.set_mode((1, 1))

    def test_every_card_fills_the_screen(self):
        for card in ({"kind": "date"}, {"kind": "time"},
                     {"kind": "person", "name": "Dr. Mira Vance",
                      "role": "Professor", "highlight": "PhD, IIT"},
                     {"kind": "greeting", "name": "Michael", "line": "Hi"}):
            with self.subTest(kind=card["kind"]):
                s = face_cards.render(card, NOW)
                self.assertEqual(s.get_size(), (WIDTH, HEIGHT))
                self.assertEqual(_px(s, 5, 5), (0, 0, 0))      # black edges

    def test_unknown_card_is_none(self):
        self.assertIsNone(face_cards.render({"kind": "nope"}, NOW))
        self.assertIsNone(face_cards.render(None, NOW))

    def test_date_card_has_the_red_weekday(self):
        s = face_cards.date_card(NOW)
        arr = pygame.surfarray.array3d(s)[:, :HEIGHT // 2]   # top half
        red = (arr[..., 0] > 200) & (arr[..., 1] < 110) & (arr[..., 2] < 110)
        self.assertGreater(red.sum(), 500)

    def test_clock_hands_move(self):
        a = pygame.surfarray.array3d(face_cards.clock_card(NOW))
        b = pygame.surfarray.array3d(face_cards.clock_card(
            NOW + datetime.timedelta(seconds=15)))
        self.assertGreater(np.abs(a.astype(int) - b.astype(int)).sum(), 0)

    def test_long_names_fit(self):
        name = "Dr. Anantha Padmanabhan Narayanan Kutty Krishnan"
        f = face_cards.fit_font(name, WIDTH - 160, 124, 64)
        self.assertLessEqual(min(f.size(name)[0], WIDTH), WIDTH)
        lines = face_cards.wrap("word " * 80, face_cards.font(42, "semibold"),
                                900, 2)
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[-1].endswith("…"))

    def test_cache_key(self):
        k = face_cards.cache_key
        self.assertNotEqual(k({"kind": "time"}, NOW),
                            k({"kind": "time"}, NOW + datetime.timedelta(seconds=1)))
        self.assertEqual(k({"kind": "date"}, NOW),
                         k({"kind": "date"}, NOW + datetime.timedelta(hours=1)))
        self.assertNotEqual(k({"kind": "person", "name": "A"}, NOW),
                            k({"kind": "person", "name": "B"}, NOW))

    def test_a_font_is_always_found(self):
        self.assertIsNotNone(face_cards.font(40, "bold"))
        self.assertIsNotNone(face_cards.font(40, "semibold"))


# ══════════════════════════════════════════════════════════════════════════════
# When a card is up (pure)
# ══════════════════════════════════════════════════════════════════════════════
class TestCardTimer(unittest.TestCase):
    def setUp(self):
        self.t = robot_face.CardTimer(linger=3.0, grace=8.0)
        self.card = {"kind": "date"}

    def vis(self, now, talking=False, since=100.0, until=None, set_at=100.0):
        return self.t.visible(self.card, set_at, since, until, talking, now)

    def test_no_card(self):
        self.assertFalse(self.t.visible(None, 0, 0, None, False, 100.0))

    def test_fixed_duration(self):
        self.assertTrue(self.vis(101.0, until=104.0))
        self.assertFalse(self.vis(104.5, until=104.0))

    def test_delayed_start(self):
        self.assertFalse(self.vis(100.5, since=101.0, until=105.0))
        self.assertTrue(self.vis(101.5, since=101.0, until=105.0))

    def test_while_speaking_then_linger(self):
        self.assertTrue(self.vis(101.0))                  # waiting for the voice
        self.assertTrue(self.vis(103.0, talking=True))    # speaking
        self.assertTrue(self.vis(110.0, talking=True))    # long answer
        self.assertTrue(self.vis(111.0))                  # stopped: linger...
        self.assertTrue(self.vis(113.5))
        self.assertFalse(self.vis(114.1))                 # ...then gone

    def test_voice_that_never_starts(self):
        self.assertTrue(self.vis(107.0))
        self.assertFalse(self.vis(108.5))

    def test_a_new_card_starts_fresh(self):
        self.vis(103.0, talking=True)
        self.vis(111.0)
        self.assertTrue(self.vis(120.5, since=120.0, set_at=120.0))


# ══════════════════════════════════════════════════════════════════════════════
# face.py's card API
# ══════════════════════════════════════════════════════════════════════════════
class TestCardApi(unittest.TestCase):
    def setUp(self):
        _set(**_RESET)

    def tearDown(self):
        _set(**_RESET)

    def test_date_card_follows_speech(self):
        face.date_card()
        with state.lock:
            self.assertEqual(state.card, {"kind": "date"})
            self.assertIsNone(state.card_until)
            self.assertLessEqual(state.card_since, state.card_set_at + 1e-6)

    def test_person_card_content(self):
        face.person_card("Dr. Mira Vance", "Professor", "PhD, IIT")
        with state.lock:
            self.assertEqual(state.card, {"kind": "person",
                                          "name": "Dr. Mira Vance",
                                          "role": "Professor",
                                          "highlight": "PhD, IIT"})

    def test_greeting_card_is_delayed_and_timed(self):
        face.greeting_card("Michael", "Good afternoon", delay=1.0, seconds=4.0)
        with state.lock:
            self.assertAlmostEqual(state.card_since - state.card_set_at, 1.0)
            self.assertAlmostEqual(state.card_until - state.card_since, 4.0)

    def test_clear(self):
        face.time_card()
        face.clear_card()
        with state.lock:
            self.assertIsNone(state.card)

    def test_cards_can_be_switched_off(self):
        with mock.patch.object(face, "SCREEN_CARDS", False):
            self.assertFalse(face.date_card())
        with state.lock:
            self.assertIsNone(state.card)


# ══════════════════════════════════════════════════════════════════════════════
# Rendered: the cross-fade
# ══════════════════════════════════════════════════════════════════════════════
class TestRenderedCards(unittest.TestCase):
    def setUp(self):
        _set(**_RESET)
        self.clock = FakeClock()
        self.patch = mock.patch("time.time", self.clock)
        self.patch.start()
        self.face = robot_face.RobotFace()
        self.face.motes = []

    def tearDown(self):
        self.patch.stop()
        _set(**_RESET)

    def frames(self, n):
        for _ in range(n):
            self.face.draw()
            self.clock.tick()

    def test_face_fades_to_the_card_and_back(self):
        face.time_card(seconds=2.0)
        self.frames(3)
        self.assertGreater(self.face.card_mix, 0.0)
        self.assertLess(self.face.card_mix, 1.0)          # mid-fade
        self.frames(40)
        self.assertEqual(self.face.card_mix, 1.0)
        # the white clock dial fills the middle of the screen
        arr = pygame.surfarray.array3d(self.face.screen)[
            WIDTH // 2 - 150:WIDTH // 2 + 150, HEIGHT // 2 - 150:HEIGHT // 2 + 150]
        self.assertGreater((arr.min(axis=2) > 240).mean(), 0.6)
        self.frames(120)                                  # past 2 s
        self.assertEqual(self.face.card_mix, 0.0)
        with state.lock:
            self.assertIsNone(state.card)                 # expired card cleared

    def test_card_waits_for_the_voice_and_lingers(self):
        face.date_card()                                   # speech-bound
        self.frames(30)
        _set(tts_active=True, speaking=True)
        self.frames(120)
        self.assertEqual(self.face.card_mix, 1.0)
        _set(tts_active=False, speaking=False)
        self.frames(int(60 * (config.CARD_LINGER_SECS - 0.5)))
        self.assertEqual(self.face.card_mix, 1.0)          # still lingering
        self.frames(90)
        self.assertEqual(self.face.card_mix, 0.0)

    def test_tap_dismisses_the_card_and_angers(self):
        face.time_card(seconds=10.0)
        self.frames(40)
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                             pos=(80, 80), button=1))
        self.frames(1)
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP,
                                             pos=(80, 80), button=1))
        self.frames(2)
        with state.lock:
            self.assertIsNone(state.card)
            self.assertEqual(state.face_override, "angry")


if __name__ == "__main__":
    unittest.main()
