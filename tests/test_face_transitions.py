# tests/test_face_transitions.py
"""Tests for calm, realistic face transitions and voice-synced talking:

    python -m unittest tests.test_face_transitions -v

  * EmotionFilter — a vision mood must settle before the face wears it, and
    once worn it is held for a minimum time (no one-second smiles).
  * Speaking keeps the face Luna was wearing — it no longer snaps to the raw
    vision label the moment audio starts, and the eyes keep the mood's shape.
  * Angry eyes are red, steadily, in every face style.
  * A mood change fades the old mouth out and lets old particles fade away
    instead of deleting them in one frame.
  * Lip sync — the mouth is driven by the real audio envelope, and the
    talking animation starts when the audio is audible, not before.
"""

import math
import os
import sys
import tempfile
import time
import unittest
import wave
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Point the face store at a throwaway dir BEFORE anything imports config.
_tmpdir = tempfile.mkdtemp(prefix="luna-test-faces-")
import config  # noqa: E402
config.FACE_DB_DIR = _tmpdir
config.RENDER_FPS = 1_000_000          # never let clock.tick() sleep in tests

import numpy as np  # noqa: E402

import face_renderer  # noqa: E402
import face_style  # noqa: E402
import robot_face  # noqa: E402
from shared_state import state  # noqa: E402

robot_face.RENDER_FPS = config.RENDER_FPS

_RESET = dict(face_override=None, face_override_until=0.0, speaking=False,
              tts_active=False, face_detected=False, face_x=0.5,
              audio_energy=0.0, emotion="Neutral", listening=False,
              conversation_active=False, luna_mode="idle", banner_text="",
              banner_until=0.0)


def _set(**kw):
    with state.lock:
        for k, v in kw.items():
            setattr(state, k, v)


class FakeClock:
    """Stands in for time.time(): 60 fps of fake time per tick()."""

    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def tick(self, secs=1.0 / 60.0):
        self.t += secs


# ══════════════════════════════════════════════════════════════════════════════
# Emotion debouncing (pure)
# ══════════════════════════════════════════════════════════════════════════════
class TestEmotionFilter(unittest.TestCase):
    def setUp(self):
        self.f = face_renderer.EmotionFilter(onset=0.8, hold=2.5)
        self.t = 100.0

    def feed(self, name, secs, step=1 / 60):
        """Report `name` every frame for `secs`; return the moods worn."""
        seen = []
        end = self.t + secs
        while self.t < end:
            seen.append(self.f.update(name, self.t))
            self.t += step
        return seen

    def test_starts_neutral(self):
        self.assertEqual(self.f.update("neutral", self.t), "neutral")

    def test_brief_blip_is_ignored(self):
        self.feed("neutral", 1.0)
        self.assertEqual(set(self.feed("happy", 0.5)), {"neutral"})
        self.assertEqual(set(self.feed("neutral", 1.0)), {"neutral"})

    def test_settled_mood_is_adopted(self):
        self.feed("neutral", 3.0)
        seen = self.feed("happy", 1.2)
        self.assertEqual(seen[-1], "happy")
        # ...but only once it had persisted for the onset time
        self.assertEqual(seen[int(0.7 * 60)], "neutral")

    def test_adopted_mood_is_held(self):
        """The 'smile for a second' bug: vision flips straight back, and the
        face must still hold the smile for the minimum hold."""
        self.feed("neutral", 3.0)
        self.feed("happy", 0.9)
        self.assertEqual(self.f.current, "happy")
        seen = self.feed("neutral", 1.5)
        self.assertEqual(set(seen), {"happy"})
        seen = self.feed("neutral", 2.0)
        self.assertEqual(seen[-1], "neutral")

    def test_flicker_never_commits(self):
        self.feed("neutral", 3.0)
        for _ in range(10):
            self.assertEqual(set(self.feed("happy", 0.3)), {"neutral"})
            self.assertEqual(set(self.feed("sad", 0.3)), {"neutral"})

    def test_choose_expression_uses_mood(self):
        fake = mock.Mock(idle_frames=0)
        self.assertEqual(
            face_renderer.choose_expression(fake, None, "happy", False), "happy")
        self.assertIsNone(
            face_renderer.choose_expression(fake, None, "happy", True))
        self.assertEqual(
            face_renderer.choose_expression(fake, "love", "sad", True), "love")


# ══════════════════════════════════════════════════════════════════════════════
# Rendered behaviour
# ══════════════════════════════════════════════════════════════════════════════
class _Rendered(unittest.TestCase):
    def setUp(self):
        import random
        random.seed(7)          # blinks / glances are random: keep it repeatable
        face_style.apply_style(3)
        _set(**_RESET)
        self.clock = FakeClock()
        self.patch = mock.patch("time.time", self.clock)
        self.patch.start()
        self.face = robot_face.RobotFace()
        self.face.motes = []                 # nothing drifting through boxes

    def tearDown(self):
        self.patch.stop()
        _set(**_RESET)
        face_style.apply_style(3)

    def frames(self, n):
        for _ in range(n):
            self.face.draw()
            self.clock.tick()

    def centre(self):
        return (int(self.face.face_cx + self.face.face_ox),
                int(self.face.face_cy + self.face.face_oy))

    def box(self, x0, y0, x1, y1):
        """Pixels of the screen box as an (h, w, 3) array."""
        import pygame
        arr = pygame.surfarray.array3d(self.face.screen)   # (w, h, 3)
        x0, x1 = max(0, x0), min(arr.shape[0], x1)
        y0, y1 = max(0, y0), min(arr.shape[1], y1)
        return arr[x0:x1, y0:y1].astype(int)

    def lit(self, px):
        """Mask of pixels that differ from the background."""
        bg = np.array(face_style.BG)
        return np.abs(px - bg).sum(axis=2) > 30


class TestSpeakingKeepsMood(_Rendered):
    def start_speaking(self):
        """What speak() + the audio-start callback do to the shared state."""
        _set(speaking=True, tts_active=True, luna_mode="speaking",
             audio_energy=0.6)

    def test_override_face_survives_audio_start(self):
        _set(emotion="Neutral")
        self.face.set_state("happy")         # e.g. face.happy() before speak()
        self.frames(10)
        self.start_speaking()
        self.frames(20)
        expr, speaking, _ = self.face._frame
        self.assertTrue(speaking)
        self.assertEqual(expr.name, "happy")
        self.assertGreater(self.face.left_blush.alpha, 60)   # still blushing

    def test_eyes_keep_the_mood_shape_while_talking(self):
        self.face.set_state("angry")
        self.start_speaking()
        self.frames(90)
        # angry eyes are 78 px tall; the old code forced 118 while talking
        self.assertLess(self.face.left_eye.h, 95)
        self.assertGreater(self.face.left_eye.squint, 0.3)


class TestAngryRedEyes(_Rendered):
    def red_fraction(self):
        cx, cy = self.centre()
        eye = self.face.left_eye
        x, y = cx + eye.rel_x, cy + eye.rel_y
        # Stops above the cheek: a happy blush is pink, and is not the eye.
        px = self.box(int(x - eye.w * 0.6), int(y - eye.h * 0.7),
                      int(x + eye.w * 0.6), int(y + eye.h * 0.45))
        lit = self.lit(px)
        red = (px[..., 0] > px[..., 1] + 60) & (px[..., 0] > px[..., 2] + 60)
        return (red & lit).sum() / max(1, lit.sum())

    def test_angry_eyes_are_steadily_red_in_every_style(self):
        for style in (1, 2, 3):
            with self.subTest(style=style):
                face_style.apply_style(style)
                self.face.set_state("angry")
                self.frames(60)
                worst = 1.0
                for _ in range(90):          # 1.5 s — a full throb cycle
                    self.frames(1)
                    worst = min(worst, self.red_fraction())
                # the worst frame is mid-blink (the glow outweighs the shrunken
                # eye): ~0.45 in the robo style. The bug this guards against —
                # a pulse that dropped to no red at all — measured 0.0.
                self.assertGreater(worst, 0.35)

    def test_other_moods_are_not_red(self):
        for name in ("neutral", "happy", "sad"):
            with self.subTest(mood=name):
                self.face.set_state(name)
                self.frames(90)
                self.assertLess(self.red_fraction(), 0.05)

    def test_red_fades_in_and_out(self):
        self.face.set_state("angry")
        self.frames(1)
        self.assertLess(self.red_fraction(), 0.2)    # not an instant switch
        self.frames(90)
        self.face.set_state("neutral")
        self.frames(90)
        self.assertLess(self.red_fraction(), 0.05)


class TestSmoothMoodExit(_Rendered):
    def mouth_lit(self):
        cx, cy = self.centre()
        my = cy + robot_face.MOUTH_OFFSET_Y
        return self.lit(self.box(cx - 170, my - 30, cx + 170, my + 90)).sum()

    def test_smile_fades_instead_of_vanishing(self):
        self.face.set_state("happy")
        self.frames(90)
        full = self.mouth_lit()
        self.assertGreater(full, 500)
        self.face.set_state("neutral")
        self.frames(1)
        self.assertGreater(self.mouth_lit(), full * 0.5)
        self.frames(90)
        self.assertEqual(self.mouth_lit(), 0)

    def test_new_mouth_appears_after_the_old_one_fades(self):
        self.face.set_state("happy")
        self.frames(90)
        self.face.set_state("sad")
        self.frames(120)
        self.assertEqual(self.face.mouth.shape, "frown")

    def test_particles_fade_out_on_mood_change(self):
        self.face.set_state("angry")
        self.frames(120)
        self.assertTrue(self.face.particles["steam"])
        self.face.set_state("neutral")
        self.frames(1)
        self.assertTrue(self.face.particles["steam"])       # still fading
        self.frames(60)
        self.assertFalse(self.face.particles["steam"])      # and gone


# ══════════════════════════════════════════════════════════════════════════════
# Lip sync
# ══════════════════════════════════════════════════════════════════════════════
def _burst(rate=22050, secs=1.0, on=(0.4, 0.6), amp=0.5):
    t = np.arange(int(rate * secs)) / rate
    x = np.zeros_like(t)
    m = (t >= on[0]) & (t < on[1])
    x[m] = amp * np.sin(2 * math.pi * 220 * t[m])
    return x.astype(np.float32)


def _write_wav(path, samples, rate=22050):
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


class TestEnvelope(unittest.TestCase):
    def setUp(self):
        import lip_sync
        self.ls = lip_sync

    def test_silence_is_closed(self):
        env = self.ls.envelope(np.zeros(22050, np.float32), 22050, fps=60)
        self.assertEqual(len(env), 60)
        self.assertEqual(max(env), 0.0)

    def test_burst_opens_only_while_sounding(self):
        env = self.ls.envelope(_burst(), 22050, fps=60)
        self.assertEqual(len(env), 60)
        self.assertGreater(min(env[26:34]), 0.8)       # 0.43-0.57 s: sounding
        self.assertEqual(max(env[:22]), 0.0)           # before
        self.assertEqual(max(env[40:]), 0.0)           # after

    def test_quiet_voice_still_opens_the_mouth(self):
        env = self.ls.envelope(_burst(amp=0.02), 22050, fps=60)
        self.assertGreater(max(env), 0.8)

    def test_load_wav_round_trip(self):
        path = os.path.join(_tmpdir, "rt.wav")
        _write_wav(path, _burst())
        samples, rate = self.ls.load_wav(path)
        self.assertEqual(rate, 22050)
        self.assertEqual(len(samples), 22050)
        self.assertAlmostEqual(float(np.abs(samples).max()), 0.5, places=2)


class TestLipSyncTrack(unittest.TestCase):
    def setUp(self):
        import lip_sync
        self.track = lip_sync.LipSync()

    def test_no_track_means_no_opinion(self):
        self.assertIsNone(self.track.energy(10.0))

    def test_follows_the_envelope_in_time(self):
        self.track.add([0.1, 0.9, 0.5], fps=10, start=10.0)
        self.assertEqual(self.track.energy(9.95), 0.0)     # not audible yet
        self.assertEqual(self.track.energy(10.05), 0.1)
        self.assertEqual(self.track.energy(10.15), 0.9)
        self.assertEqual(self.track.energy(10.25), 0.5)
        self.assertEqual(self.track.energy(10.5), 0.0)     # gap: mouth shuts

    def test_reset_forgets(self):
        self.track.add([1.0], fps=10, start=0.0)
        self.track.reset()
        self.assertIsNone(self.track.energy(0.05))


class _FakeProc:
    def __init__(self, log, secs):
        self.log, self.secs, self.returncode = log, secs, None
        log.append("spawn")

    def wait(self, timeout=None):
        time.sleep(self.secs)
        self.returncode = 0
        return 0

    def poll(self):
        return self.returncode


class TestTalkingStartsWithTheVoice(unittest.TestCase):
    def setUp(self):
        import piper_tts
        self.pt = piper_tts
        self.tts = piper_tts.PiperTTS()
        self.log = []
        self.wav_secs = 0.3

    def synth(self, text):
        path = tempfile.mktemp(suffix=".wav", dir=_tmpdir)
        _write_wav(path, _burst(secs=self.wav_secs, on=(0.1, 0.2)))
        return path

    def fake_run(self, cmd, *a, **kw):
        self.log.append("spawn")
        time.sleep(self.wav_secs)
        return mock.Mock(returncode=0)

    def test_external_player_starts_animation_after_spawn(self):
        self.tts._mode = "persistent"
        tracks = []
        with mock.patch.object(self.tts, "_synth_persistent", self.synth), \
             mock.patch.object(self.pt, "TTS_INPROCESS_PLAYBACK", False, create=True), \
             mock.patch.object(self.pt.subprocess, "run", self.fake_run), \
             mock.patch.object(self.pt.subprocess, "Popen",
                               lambda *a, **k: _FakeProc(self.log, self.wav_secs)):
            self.tts.speak_chunks(
                ["Hello there."],
                on_start=lambda: self.log.append("start"),
                on_audio=lambda env, fps, at: tracks.append((env, fps, at)))
        self.assertEqual(self.log, ["spawn", "start"])
        self.assertEqual(len(tracks), 1)
        env, fps, _ = tracks[0]
        self.assertAlmostEqual(len(env) / fps, self.wav_secs, places=1)

    def test_player_latency_self_calibrates(self):
        est = self.tts._player_latency
        for _ in range(20):
            self.tts._calibrate(wall=1.0 + 0.35, duration=1.0)
        self.assertAlmostEqual(self.tts._player_latency, 0.35, places=2)
        self.tts._calibrate(wall=60.0, duration=1.0)          # a stall
        self.assertLessEqual(self.tts._player_latency, 1.0)
        self.assertNotEqual(est, self.tts._player_latency)

    def test_mac_synthesises_a_wav_before_playing(self):
        """`say` used to synthesise AND play in one call after on_start had
        already fired — the ~1 s head start on the Mac."""
        self.tts._mode = "mac"

        def fake_say(cmd, *a, **kw):
            self.assertEqual(cmd[0], "say")
            _write_wav(cmd[cmd.index("-o") + 1],
                       _burst(secs=self.wav_secs, on=(0.1, 0.2)))
            return mock.Mock(returncode=0)

        with mock.patch.object(self.pt.subprocess, "run", fake_say):
            prepared = self.tts._prepare("Hello there.")
        self.assertIsInstance(prepared, str)
        self.assertTrue(os.path.exists(prepared))

        played = []
        stream = mock.MagicMock(latency=0.02)
        stream.__enter__.return_value = stream
        stream.write.side_effect = lambda data: played.append("write")
        fake_sd = mock.Mock()
        fake_sd.OutputStream.return_value = stream
        with mock.patch.object(self.pt, "TTS_INPROCESS_PLAYBACK", True, create=True), \
             mock.patch.object(self.pt, "_sounddevice", lambda: fake_sd, create=True):
            self.tts._render(prepared,
                             lambda env, fps, at: played.append("audible"))
        # audio is queued before the talking animation is released
        self.assertEqual(played[:2], ["write", "audible"])
        fake_sd.play.assert_not_called()       # never the Python-callback path


if __name__ == "__main__":
    unittest.main()
