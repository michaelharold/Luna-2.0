# tests/test_face_render.py
"""Headless render tests for the face (SDL dummy driver — no window needed):

    python -m unittest tests.test_face_render -v

Covers the render-speed work: every expression survives tilt + speaking
frames, the rotated face layer is big enough for everything it must hold, the
angle quantiser, and the sprite cache (identity + invalidation on a style
switch).
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Point the face store at a throwaway dir BEFORE anything imports config.
_tmpdir = tempfile.mkdtemp(prefix="luna-test-faces-")
import config  # noqa: E402
config.FACE_DB_DIR = _tmpdir
config.RENDER_FPS = 1_000_000          # never let clock.tick() sleep in tests

import robot_face  # noqa: E402
import expressions  # noqa: E402
import face_style  # noqa: E402
from shared_state import state  # noqa: E402

robot_face.RENDER_FPS = config.RENDER_FPS


def _set(**kw):
    with state.lock:
        for k, v in kw.items():
            setattr(state, k, v)


class TestRenderSmoke(unittest.TestCase):
    """30 frames of every expression, tilted and talking, must not raise."""

    @classmethod
    def setUpClass(cls):
        cls.face = robot_face.RobotFace()

    def tearDown(self):
        _set(face_detected=False, face_x=0.5, speaking=False, audio_energy=0.0,
             conversation_active=False, luna_mode="idle", banner_text="",
             banner_until=0.0)

    def test_every_expression_tilted_and_speaking(self):
        _set(face_detected=True, face_x=0.15, speaking=True, audio_energy=0.7,
             frozen_emotion=None, conversation_active=True,
             luna_mode="processing", banner_text="Hi Test!",
             banner_until=1e12)
        for name in expressions.NAMES:
            with self.subTest(expression=name):
                self.face.set_state(name)
                for _ in range(30):
                    self.face.draw()

    def test_actually_tilts(self):
        """With a face at the left edge the head angle must leave the dead
        band, so the tilted (layer) path is really what the smoke test ran."""
        _set(face_detected=True, face_x=0.15)
        self.face.set_state("neutral")
        for _ in range(60):
            self.face.draw()
        self.assertNotEqual(robot_face.quantize_angle(self.face.head_angle), 0.0)

    def test_every_style(self):
        _set(face_detected=True, face_x=0.85, speaking=True, audio_energy=0.7)
        try:
            for style in (1, 2, 3):
                face_style.apply_style(style)
                for name in ("happy", "angry", "sleeping"):
                    self.face.set_state(name)
                    for _ in range(10):
                        self.face.draw()
        finally:
            face_style.apply_style(config.FACE_STYLE)


class TestLayerExtent(unittest.TestCase):
    """face_layer_extent() is pure: it must contain every part at its biggest."""

    def setUp(self):
        self.l, self.t, self.r, self.b = robot_face.face_layer_extent()

    def assertInside(self, x0, y0, x1, y1):
        self.assertLessEqual(self.l, x0)
        self.assertLessEqual(self.t, y0)
        self.assertGreaterEqual(self.r, x1)
        self.assertGreaterEqual(self.b, y1)

    def test_biggest_eyes(self):
        # Largest target eye (any expression, bigger asym eye), drawn as a
        # Face 3 solid eye (x1.06 wide + blink bulge, x1.3 tall + widen), at
        # +-EYE_SPREAD, plus its glow halo.
        exprs = expressions.EXPRESSIONS.values()
        w = max(e.eye_w * max(e.asym) for e in exprs) * 1.06 * 1.12
        h = max(e.eye_h * max(e.asym) for e in exprs) * 1.30 * 1.22
        glow = 30
        x = robot_face.EYE_SPREAD + w / 2 + glow
        y0 = robot_face.EYE_OFFSET_Y - h / 2 - glow
        y1 = robot_face.EYE_OFFSET_Y + h / 2 + glow
        self.assertInside(-x, y0, x, y1)
        # The spec's reference sizes: 240x190 eyes at +-190 px.
        self.assertInside(-190 - 120 - 35, -30 - 95 - 35, 190 + 120 + 35, 0)

    def test_mouth(self):
        m = robot_face.MOUTH_OFFSET_Y
        self.assertInside(-150, m - 60, 150, m + 110)

    def test_particle_spawns(self):
        # Spawns reach +-260 x and -180..+120 y around the face centre; the
        # sprites themselves stick out ~36 px beyond that.
        self.assertInside(-260 - 36, -180 - 36, 260 + 36, 120 + 36)

    def test_blush(self):
        bx, by = robot_face.BLUSH_OFFSET
        self.assertInside(-bx - 38, by - 38, bx + 38, by + 38)

    def test_smaller_than_canvas(self):
        # The whole point: rotating this must be cheaper than the canvas.
        area = (self.r - self.l) * (self.b - self.t)
        self.assertLess(area, face_style.WIDTH * face_style.HEIGHT * 0.6)


class TestAngle(unittest.TestCase):
    def test_dead_band(self):
        q = robot_face.quantize_angle
        self.assertEqual(q(0.0), 0.0)
        self.assertEqual(q(0.1), 0.0)
        self.assertEqual(q(-0.1), 0.0)

    def test_quantised(self):
        q = robot_face.quantize_angle
        self.assertEqual(q(7.9), 8.0)
        self.assertEqual(q(3.13), 3.25)
        self.assertEqual(q(-2.6), -2.5)
        for a in (0.3, 1.11, 5.49, -8.0):
            self.assertEqual(q(a) % 0.25, 0.0)

    def test_rotate_point_matches_pygame_sense(self):
        # pygame.transform.rotate turns counter-clockwise on screen: a point
        # to the right of centre moves UP (smaller y) for a positive angle.
        cx, cy = face_style.WIDTH / 2, face_style.HEIGHT / 2
        x, y = robot_face.rotate_point(cx + 100, cy, 90)
        self.assertAlmostEqual(x, cx, places=6)
        self.assertAlmostEqual(y, cy - 100, places=6)
        self.assertEqual(robot_face.rotate_point(cx, cy, 33), (cx, cy))


class TestSpriteCache(unittest.TestCase):
    def setUp(self):
        import pygame
        self.pygame = pygame
        pygame.display.init()
        if pygame.display.get_surface() is None:
            pygame.display.set_mode((8, 8))
        self.builds = 0

    def _build(self):
        self.builds += 1
        return self.pygame.Surface((4, 4), self.pygame.SRCALPHA)

    def test_identity(self):
        a = face_style.sprite(("test", 1), self._build)
        b = face_style.sprite(("test", 1), self._build)
        self.assertIs(a, b)
        self.assertEqual(self.builds, 1)
        c = face_style.sprite(("test", 2), self._build)
        self.assertIsNot(a, c)

    def test_invalidated_by_clear_caches(self):
        a = face_style.sprite(("test", "clear"), self._build)
        face_style.clear_caches()
        b = face_style.sprite(("test", "clear"), self._build)
        self.assertIsNot(a, b)
        self.assertEqual(self.builds, 2)

    def test_invalidated_by_style_switch(self):
        a = face_style.sprite(("test", "style"), self._build)
        try:
            face_style.apply_style(1)
        finally:
            face_style.apply_style(config.FACE_STYLE)
        self.assertIsNot(a, face_style.sprite(("test", "style"), self._build))

    def test_alpha_bucket(self):
        ab = face_style.alpha_bucket
        self.assertEqual(ab(0), 0)
        self.assertEqual(ab(-5), 0)
        self.assertEqual(ab(255), 255)
        self.assertEqual(ab(300), 255)
        levels = {ab(a) for a in range(256)}
        self.assertLessEqual(len(levels), 16)
        for a in range(256):
            self.assertLessEqual(abs(ab(a) - a), 9)      # half a step

    def test_with_alpha_scales_pixels(self):
        pygame = self.pygame
        s = pygame.Surface((2, 2), pygame.SRCALPHA)
        s.fill((200, 100, 50, 200))
        out = face_style.with_alpha(s, 128)
        r, g, b, a = out.get_at((0, 0))
        self.assertEqual((r, g, b), (200, 100, 50))
        self.assertAlmostEqual(a, 200 * 128 / 255, delta=2)
        self.assertEqual(s.get_at((0, 0)).a, 200)        # source untouched


if __name__ == "__main__":
    unittest.main()
