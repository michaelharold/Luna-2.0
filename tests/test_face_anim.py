# tests/test_face_anim.py
"""Tests for the "I know you!" recognition moment and touch registration:

    python -m unittest tests.test_face_anim -v

Pure pieces (no screen): double_take_offset, slide_params, in_enroll_zone,
TouchHold, the "recognised" registry entry, and the state that
face.recognition_moment / face.banner / face.recognised write.
Rendered pieces (SDL dummy driver): 60 frames of the whole moment, and a
touch-and-hold / short tap driven through real pygame events.

Set LUNA_FACE_SHOT_DIR to keep the rendered PNGs (default: a temp dir).
"""

import os
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Point the face store at a throwaway dir BEFORE anything imports config.
_tmpdir = tempfile.mkdtemp(prefix="luna-test-faces-")
import config  # noqa: E402
config.FACE_DB_DIR = _tmpdir
config.RENDER_FPS = 1_000_000          # never let clock.tick() sleep in tests

import expressions  # noqa: E402
import face  # noqa: E402
import robot_face  # noqa: E402
from shared_state import state  # noqa: E402

robot_face.RENDER_FPS = config.RENDER_FPS

_RESET = dict(face_override=None, face_override_until=0.0, banner_text="",
              banner_until=0.0, banner_since=0.0, recog_event_time=0.0,
              touch_enroll_request=False, speaking=False, tts_active=False,
              face_detected=False, face_x=0.5, audio_energy=0.0, card=None,
              card_set_at=0.0, card_since=0.0, card_until=None)


def _reset_state():
    with state.lock:
        for k, v in _RESET.items():
            setattr(state, k, v)
    face._moments.clear()              # one moment per arrival — per test


class FakeClock:
    """Stands in for time.time(): 60 fps of fake time per tick()."""

    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def tick(self, secs=1.0 / 60.0):
        self.t += secs


# ══════════════════════════════════════════════════════════════════════════════
# Pure functions
# ══════════════════════════════════════════════════════════════════════════════
class TestDoubleTake(unittest.TestCase):
    f = staticmethod(robot_face.double_take_offset)

    def test_outside_timeline_is_neutral(self):
        for p in (-0.5, -1e-9, 1.0, 1.5, 99.0):
            self.assertEqual(self.f(p), (0.0, 1.0))

    def test_starts_popped_and_centred(self):
        dx, scale = self.f(0.0)
        self.assertEqual(dx, 0.0)
        self.assertAlmostEqual(scale, 1.08)

    def test_darts_left_then_right(self):
        # 0.15 s in 0.6 s: the left dart spans phase 0-0.25, right 0.25-0.5.
        self.assertAlmostEqual(self.f(0.2)[0], -robot_face.DOUBLE_TAKE_DART)
        self.assertAlmostEqual(self.f(0.45)[0], robot_face.DOUBLE_TAKE_DART)
        self.assertEqual(robot_face.DOUBLE_TAKE_DART, 28.0)
        self.assertAlmostEqual(robot_face.DOUBLE_TAKE_SECS, 0.6)

    def test_eases_back_to_zero(self):
        self.assertAlmostEqual(self.f(0.999)[0], 0.0, places=2)
        self.assertAlmostEqual(self.f(0.999)[1], 1.0, places=3)
        xs = [self.f(0.5 + i / 200)[0] for i in range(100)]
        self.assertTrue(all(a >= b for a, b in zip(xs, xs[1:])))

    def test_scale_settles_monotonically(self):
        ss = [self.f(i / 100)[1] for i in range(100)]
        self.assertTrue(all(a >= b for a, b in zip(ss, ss[1:])))
        self.assertTrue(all(1.0 <= s <= 1.08 + 1e-9 for s in ss))

    def test_continuous_at_segment_joins(self):
        for join in (0.25, 0.5):
            a, b = self.f(join - 1e-6)[0], self.f(join)[0]
            self.assertAlmostEqual(a, b, places=3)


class TestSlide(unittest.TestCase):
    f = staticmethod(robot_face.slide_params)

    def test_start(self):
        dy, scale, alpha = self.f(0.0)
        self.assertAlmostEqual(dy, 40.0)
        self.assertAlmostEqual(scale, 1.0)
        self.assertAlmostEqual(alpha, 0.0)

    def test_settled(self):
        for t in (0.35, 0.5, 10.0, -1.0):
            self.assertEqual(self.f(t), (0.0, 1.0, 1.0))

    def test_rises_fades_in_and_overshoots(self):
        ts = [i * 0.35 / 50 for i in range(50)]
        dys = [self.f(t)[0] for t in ts]
        alphas = [self.f(t)[2] for t in ts]
        scales = [self.f(t)[1] for t in ts]
        self.assertTrue(all(a >= b for a, b in zip(dys, dys[1:])))
        self.assertTrue(all(a <= b for a, b in zip(alphas, alphas[1:])))
        self.assertAlmostEqual(max(scales), 1.08, places=2)
        self.assertGreaterEqual(min(scales), 1.0)
        # The peak is mid-slide, not at either end.
        peak = ts[scales.index(max(scales))]
        self.assertTrue(0.1 < peak < 0.25)


class TestEnrollZone(unittest.TestCase):
    def test_zone(self):
        z = robot_face.in_enroll_zone
        cx, cy = 700, 400
        self.assertTrue(z(cx, cy, cx, cy))                    # dead centre
        self.assertTrue(z(cx + 60, cy - 60, cx, cy))          # between the eyes
        self.assertFalse(z(cx - 190, cy - 30, cx, cy))        # on the left eye
        self.assertFalse(z(cx + 190, cy - 30, cx, cy))        # on the right eye
        self.assertFalse(z(cx, cy + 155, cx, cy))             # on the mouth
        self.assertFalse(z(cx + 400, cy, cx, cy))


class TestTouchHold(unittest.TestCase):
    def setUp(self):
        self.h = robot_face.TouchHold(1.2)

    def test_short_tap_returns_position(self):
        self.h.press(10, 20, now=100.0)
        self.assertFalse(self.h.poll(100.5))
        self.assertEqual(self.h.release(100.6), (10, 20))
        self.assertFalse(self.h.pending)

    def test_hold_fires_exactly_once(self):
        self.h.press(10, 20, now=100.0)
        self.assertFalse(self.h.poll(101.19))
        self.assertTrue(self.h.poll(101.2))
        self.assertFalse(self.h.poll(101.3))
        self.assertFalse(self.h.poll(105.0))
        self.assertIsNone(self.h.release(105.1))      # not also a tap

    def test_drag_cancels(self):
        self.h.press(10, 20, now=100.0)
        self.h.move(15, 25)                           # a wobble is fine
        self.assertTrue(self.h.pending)
        self.h.move(200, 20)                          # a drag is not
        self.assertFalse(self.h.pending)
        self.assertFalse(self.h.poll(102.0))
        self.assertIsNone(self.h.release(102.1))

    def test_duplicate_press_ignored(self):
        self.h.press(10, 20, now=100.0)
        self.h.press(11, 21, now=101.0)               # synthesised mouse event
        self.assertTrue(self.h.poll(101.2))           # timed from the first

    def test_release_without_press(self):
        self.assertIsNone(self.h.release(1.0))
        self.assertFalse(self.h.poll(9.0))


# ══════════════════════════════════════════════════════════════════════════════
# Registry + face API state
# ══════════════════════════════════════════════════════════════════════════════
class TestRecognisedExpression(unittest.TestCase):
    def test_registered(self):
        e = expressions.get("recognised")
        self.assertEqual(e.name, "recognised")
        self.assertIn("recognised", expressions.EXPRESSIONS)
        self.assertIn("recognised", expressions.NAMES)

    def test_design(self):
        e = expressions.get("recognised")
        self.assertTrue(e.arc_eyes)
        self.assertAlmostEqual(e.widen, 0.5)
        self.assertAlmostEqual(e.pupil_scale, 1.3)
        self.assertEqual(e.brow, "happy")
        self.assertEqual(e.mouth, "smile")
        self.assertAlmostEqual(e.mouth_visible, 0.9)
        self.assertAlmostEqual(e.corner_lift, 0.9)
        self.assertAlmostEqual(e.blush, 170.0)
        self.assertAlmostEqual(e.bounce, 28.0)
        self.assertEqual(e.particles, (("star", 0), ("heart", 30)))

    def test_others_unchanged(self):
        self.assertEqual(expressions.get("happy").blush, 140.0)
        self.assertEqual(expressions.get("excited").particles,
                         (("star", 0), ("star", 25)))


class TestFaceApi(unittest.TestCase):
    def setUp(self):
        _reset_state()

    tearDown = setUp

    def test_exported(self):
        self.assertIn("recognised", face.__all__)
        self.assertIn("recognition_moment", face.__all__)

    def test_banner_stamps_since(self):
        before = time.time()
        face.banner("Hello", 3.0)
        with state.lock:
            self.assertEqual(state.banner_text, "Hello")
            self.assertGreaterEqual(state.banner_since, before)
            self.assertAlmostEqual(state.banner_until - state.banner_since, 3.0)

    def test_recognised_default_duration(self):
        face.recognised()
        with state.lock:
            self.assertEqual(state.face_override, "recognised")
            left = state.face_override_until - time.time()
        self.assertAlmostEqual(left, config.RECOG_EXPRESSION_SECS, delta=0.5)
        face.recognised(hold=True)
        with state.lock:
            self.assertEqual(state.face_override_until, float("inf"))

    def test_recognition_moment_writes(self):
        with mock.patch.object(face, "RECOG_ANIMATION", True):
            before = time.time()
            face.recognition_moment("Alice", line="Good afternoon")
            after = time.time()
        with state.lock:
            # the configured greeting face (RECOG_EXPRESSION, "excited")
            self.assertEqual(state.face_override, face.RECOG_EXPRESSION)
            self.assertTrue(before <= state.recog_event_time <= after)
            # the name card, AFTER the double-take, for GREET_CARD_SECS
            self.assertEqual(state.card, {"kind": "greeting", "name": "Alice",
                                          "line": "Good afternoon"})
            self.assertEqual(state.card_set_at, state.recog_event_time)
            self.assertAlmostEqual(state.card_since - state.card_set_at,
                                   config.GREET_CARD_DELAY)
            self.assertAlmostEqual(state.card_until - state.card_since,
                                   config.GREET_CARD_SECS)
            self.assertAlmostEqual(
                state.face_override_until - state.recog_event_time,
                config.RECOG_EXPRESSION_SECS)
            self.assertEqual(state.banner_text, "")

    def test_banner_when_cards_are_off(self):
        with mock.patch.object(face, "RECOG_ANIMATION", True), \
                mock.patch.object(face, "GREET_CARD", False):
            face.recognition_moment("Alice")
        with state.lock:
            self.assertEqual(state.banner_text, "Hi Alice!")
            self.assertEqual(state.banner_since, state.recog_event_time)
            self.assertAlmostEqual(state.banner_until - state.banner_since,
                                   config.GREET_BANNER_SECS)
            self.assertIsNone(state.card)

    def test_once_per_arrival(self):
        """Vision plays the moment on sight; the spoken greeting asks for it
        again a moment later — the double-take must not play twice."""
        self.assertTrue(face.recognition_moment("Alice", line=""))
        with state.lock:
            first = state.recog_event_time
        self.assertFalse(face.recognition_moment("Alice", line=""))
        with state.lock:
            self.assertEqual(state.recog_event_time, first)
        self.assertTrue(face.recognition_moment("Bob", line=""))   # someone else

    def test_greeting_line(self):
        noon = time.mktime((2026, 10, 12, 14, 0, 0, 0, 0, -1))
        with mock.patch.object(face, "_met_before", lambda n: True):
            self.assertEqual(face.greeting_line("Zed Quill", noon),
                             "Good afternoon · welcome back")
        with mock.patch.object(face, "_met_before", lambda n: False):
            self.assertEqual(face.greeting_line("Zed Quill", noon - 5 * 3600),
                             "Good morning · nice to see you")

    def test_recognition_moment_without_animation(self):
        with mock.patch.object(face, "RECOG_ANIMATION", False):
            face.recognition_moment("Bob", line="")
        with state.lock:
            self.assertEqual(state.face_override, "happy")
            self.assertEqual(state.card["name"], "Bob")
            self.assertEqual(state.card_since, state.card_set_at)  # no delay
            self.assertEqual(state.recog_event_time, 0.0)   # no double-take


# ══════════════════════════════════════════════════════════════════════════════
# Rendered
# ══════════════════════════════════════════════════════════════════════════════
class TestRenderedMoment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shot_dir = os.environ.get("LUNA_FACE_SHOT_DIR") or tempfile.mkdtemp(
            prefix="luna-face-shots-")
        os.makedirs(cls.shot_dir, exist_ok=True)

    def setUp(self):
        _reset_state()
        self.clock = FakeClock()
        self.patch = mock.patch("time.time", self.clock)
        self.patch.start()
        self.face = robot_face.RobotFace()

    def tearDown(self):
        self.patch.stop()
        _reset_state()

    def _frames(self, n, override=True):
        for _ in range(n):
            if override:
                with state.lock:
                    name = state.face_override
                    live = name and self.clock() <= state.face_override_until
                if live:
                    self.face.set_state(name)
            self.face.draw()
            self.clock.tick()

    def test_recognition_moment_60_frames(self):
        import pygame
        with mock.patch.object(face, "RECOG_ANIMATION", True):
            face.recognition_moment("Alice")
        darts = []
        for i in range(60):
            self._frames(1)
            darts.append(self.face._nudge[0])
            if i == 14:
                pygame.image.save(self.face.screen, os.path.join(
                    self.shot_dir, "recognised_mid.png"))
        pygame.image.save(self.face.screen,
                          os.path.join(self.shot_dir, "recognised.png"))

        self.assertEqual(self.face.state, face.RECOG_EXPRESSION)
        self.assertLess(min(darts), -20)             # looked left...
        self.assertGreater(max(darts), 20)           # ...then right...
        self.assertEqual(darts[-1], 0.0)             # ...and settled
        self.assertEqual(self.face._nudge, (0.0, 1.0))
        self.assertFalse(self.face.double_take_active)
        self.assertGreater(self.face.eye_arc, 0.5)   # arcs AFTER the take
        self.assertTrue(any(self.face.particles[k] for k in ("star", "heart")))

    def test_stale_event_is_ignored(self):
        with state.lock:
            state.recog_event_time = self.clock() - 30.0
        self._frames(3, override=False)
        self.assertFalse(self.face.double_take_active)

    def _centre(self):
        return (self.face.face_cx + self.face.face_ox,
                self.face.face_cy + self.face.face_oy)

    def _post(self, etype, pos):
        import pygame
        kw = {"pos": pos}
        if etype != pygame.MOUSEMOTION:
            kw["button"] = 1
        else:
            kw.update(rel=(0, 0), buttons=(1, 0, 0))
        pygame.event.post(pygame.event.Event(etype, **kw))

    def test_touch_and_hold_requests_enrollment(self):
        import pygame
        self._frames(5, override=False)
        cx, cy = self._centre()
        self._post(pygame.MOUSEBUTTONDOWN, (int(cx), int(cy)))
        self._frames(int(config.TOUCH_ENROLL_HOLD_SECS * 60) - 6, override=False)
        with state.lock:
            self.assertFalse(state.touch_enroll_request)   # not yet
        self._frames(12, override=False)
        with state.lock:
            self.assertTrue(state.touch_enroll_request)
            self.assertEqual(state.face_override, "curious")
            self.assertTrue(state.banner_text.startswith("Registering"))
            state.touch_enroll_request = False               # voice loop consumed
        self._frames(30, override=False)                     # still held
        self._post(pygame.MOUSEBUTTONUP, (int(cx), int(cy)))
        self._frames(2, override=False)
        with state.lock:
            self.assertFalse(state.touch_enroll_request)     # only once
            self.assertEqual(state.face_override, "curious")  # no tap on release

    def test_short_tap_in_the_middle_angers(self):
        """A quick tap in the registration zone is a tap, not a hold."""
        import pygame
        self._frames(5, override=False)
        cx, cy = self._centre()
        pos = (int(cx), int(cy) + 60)          # in the zone, near the mouth
        self._post(pygame.MOUSEBUTTONDOWN, pos)
        self._frames(6, override=False)
        self._post(pygame.MOUSEBUTTONUP, pos)
        self._frames(1, override=False)
        with state.lock:
            self.assertEqual(state.face_override, "angry")
            self.assertFalse(state.touch_enroll_request)

    def test_tap_on_an_eye_angers_at_once(self):
        import pygame
        self._frames(5, override=False)
        cx, cy = self._centre()
        self._post(pygame.MOUSEBUTTONDOWN, (int(cx) - 190, int(cy) - 30))
        self._frames(1, override=False)
        with state.lock:
            self.assertEqual(state.face_override, "angry")
            self.assertAlmostEqual(state.face_override_until - self.clock(),
                                   config.TOUCH_ANGRY_SECS, delta=0.1)

    def test_no_anger_while_speaking(self):
        import pygame
        self._frames(5, override=False)
        with state.lock:
            state.speaking = True
        self._post(pygame.MOUSEBUTTONDOWN, (80, 80))
        self._frames(1, override=False)
        with state.lock:
            self.assertIsNone(state.face_override)


if __name__ == "__main__":
    unittest.main()
