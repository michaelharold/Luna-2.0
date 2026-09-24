# tests/test_audio_preprocess.py
"""Unit tests for audio_preprocess.py — the high-pass / gate / AGC stage in
front of Vosk. Pure numpy signals, no mic, no models:

    python -m unittest discover tests -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import audio_preprocess as ap  # noqa: E402

RATE = 16000
FS = 32767.0
BLOCK = int(RATE * 0.25)      # 4000 samples — Luna's live block size
T = np.arange(2 * RATE) / RATE  # 2 s


def _sine(freq, ampl):
    return ampl * FS * np.sin(2 * np.pi * freq * T)


def _to_i16(x):
    return np.clip(np.rint(x), -32768, 32767).astype(np.int16)


def _run_blocks(pre, x):
    """Feed x through `pre` in 250 ms blocks, the way listen() does."""
    out = []
    for i in range(0, x.size, BLOCK):
        out.append(np.frombuffer(pre.process(x[i:i + BLOCK].tobytes()),
                                 dtype=np.int16))
    return np.concatenate(out)


def _level_db(x, freq):
    """Magnitude (dB) of one frequency, measured on the middle 1 s so the
    filter's start-up transient is excluded. 1 s at 16 kHz -> 1 Hz bins."""
    seg = x[RATE // 2: RATE // 2 + RATE].astype(np.float64)
    spec = np.abs(np.fft.rfft(seg))
    return 20.0 * np.log10(spec[int(freq)] + 1e-9)


class TestHighpass(unittest.TestCase):
    def setUp(self):
        # filter only: no gate, no gain — so we measure the filter itself
        self.pre = ap.AudioPreprocessor(RATE, normalize=False, gate_db=None)

    def test_hum_removed_speech_kept(self):
        x = _to_i16(_sine(50, 0.3) + _sine(1000, 0.3))
        y = _run_blocks(self.pre, x)
        drop_50 = _level_db(x, 50) - _level_db(y, 50)
        drop_1k = _level_db(x, 1000) - _level_db(y, 1000)
        self.assertGreater(drop_50, 20.0, f"50 Hz only dropped {drop_50:.1f} dB")
        self.assertLess(abs(drop_1k), 1.5, f"1 kHz changed by {drop_1k:.2f} dB")

    def test_dc_offset_removed(self):
        x = _to_i16(5000.0 + _sine(1000, 0.2))
        y = _run_blocks(self.pre, x)
        self.assertLess(abs(float(np.mean(y[-RATE:]))), 50.0)
        self.assertAlmostEqual(float(np.mean(x[-RATE:])), 5000.0, delta=5)

    def test_block_boundary_continuity(self):
        rng = np.random.default_rng(1)
        x = _to_i16(_sine(50, 0.2) + _sine(300, 0.2) + _sine(2500, 0.1)
                    + rng.normal(0, 0.02 * FS, T.size))
        whole = np.frombuffer(
            ap.AudioPreprocessor(RATE, normalize=False, gate_db=None)
            .process(x.tobytes()), dtype=np.int16)
        blocks = _run_blocks(self.pre, x)
        diff = np.max(np.abs(whole.astype(np.int64) - blocks.astype(np.int64)))
        self.assertLessEqual(diff, 1, f"block seams differ by up to {diff}")

    def test_numpy_fallback_matches_scipy(self):
        """The scipy-free biquad cascade is the same filter (used only when
        scipy is missing, so prove it here rather than on the Pi)."""
        x = _to_i16(_sine(50, 0.3) + _sine(1000, 0.3))
        sos_np = ap._design_highpass_numpy(RATE, 100.0, ap.HIGHPASS_ORDER)
        zi = np.zeros((sos_np.shape[0], 2))
        y_np, _ = ap._sosfilt_numpy(sos_np, x.astype(np.float64), zi)
        y_np = _to_i16(y_np)
        y_sp = _run_blocks(self.pre, x)
        self.assertGreater(_level_db(x, 50) - _level_db(y_np, 50), 20.0)
        self.assertLessEqual(
            np.max(np.abs(y_np.astype(np.int64) - y_sp.astype(np.int64))), 2)


class TestGateAndGain(unittest.TestCase):
    def _speechlike(self, peak):
        """Three harmonics under a syllable-rate envelope, scaled to `peak` FS."""
        env = np.abs(np.sin(2 * np.pi * 3 * T))
        tone = (np.sin(2 * np.pi * 200 * T) + np.sin(2 * np.pi * 800 * T)
                + np.sin(2 * np.pi * 2000 * T))
        sig = env * tone
        return _to_i16(sig / np.max(np.abs(sig)) * peak * FS)

    def test_quiet_speech_boosted_up_to_max_gain(self):
        pre = ap.AudioPreprocessor(RATE)          # config defaults
        x = self._speechlike(0.02)                # ~-34 dBFS peak: quiet
        gains = []
        out = []
        for i in range(0, x.size, BLOCK):
            out.append(np.frombuffer(pre.process(x[i:i + BLOCK].tobytes()),
                                     dtype=np.int16))
            gains.append(pre.last_gain)
        y = np.concatenate(out)
        in_peak, out_peak = np.abs(x).max(), np.abs(y).max()
        self.assertGreater(out_peak, 2.0 * in_peak, "quiet speech not boosted")
        self.assertLessEqual(out_peak, pre.max_gain * in_peak * 1.01 + 2)
        self.assertLessEqual(max(gains), pre.max_gain)
        self.assertFalse(pre.last_gated)

    def test_loud_speech_not_pushed_past_target(self):
        pre = ap.AudioPreprocessor(RATE)
        x = self._speechlike(0.6)
        y = _run_blocks(pre, x)
        self.assertLessEqual(np.abs(y).max(), pre.target_peak * FS * 1.02 + 2)
        self.assertLess(pre.last_gain, pre.max_gain)

    def test_near_silence_is_gated_to_zeros(self):
        pre = ap.AudioPreprocessor(RATE)
        rng = np.random.default_rng(0)
        x = _to_i16(rng.normal(0, 10 ** (-70 / 20) * FS, T.size))   # -70 dBFS
        y = _run_blocks(pre, x)
        self.assertFalse(y.any(), "gated block should be all zeros")
        self.assertTrue(pre.last_gated)
        self.assertEqual(pre.last_gain, 0.0)
        self.assertLess(pre.last_rms_db, pre.gate_db)

    def test_peak_hold_keeps_gain_constant_within_sentence(self):
        """Loud and quiet syllables alternate: with the hold window the gain
        is set by the loudest one and stays put; with hold 0 it pumps."""
        loud = _to_i16(_sine(300, 0.6)[:BLOCK]).tobytes()
        soft = _to_i16(_sine(300, 0.15)[:BLOCK]).tobytes()
        pattern = [loud, soft, soft, loud, soft, soft, loud, soft]

        # filter off: an abruptly starting tone gives the high-pass a start-up
        # overshoot, and this test is about the AGC alone
        held = ap.AudioPreprocessor(RATE, highpass_hz=None, hold_s=1.5)
        gains_held = []
        for blk in pattern:
            held.process(blk)
            gains_held.append(held.last_gain)
        self.assertLess(max(gains_held) - min(gains_held), 0.05)
        self.assertAlmostEqual(gains_held[0], 0.85 / 0.6, places=2)

        pumping = ap.AudioPreprocessor(RATE, highpass_hz=None,
                                       hold_s=0.0)             # hold 0 must not crash
        gains_pump = []
        for blk in pattern:
            pumping.process(blk)
            gains_pump.append(pumping.last_gain)
        self.assertGreater(max(gains_pump) - min(gains_pump), 0.2)

    def test_hold_releases_for_quieter_talker(self):
        loud = _to_i16(_sine(300, 0.6)[:BLOCK]).tobytes()
        soft = _to_i16(_sine(300, 0.05)[:BLOCK]).tobytes()
        pre = ap.AudioPreprocessor(RATE, highpass_hz=None, hold_s=1.0,
                                   release_db_per_s=6.0)
        pre.process(loud)
        g0 = pre.last_gain
        for _ in range(40):                                   # 10 s of quiet talker
            pre.process(soft)
        self.assertGreater(pre.last_gain, g0 * 4)
        self.assertLessEqual(pre.last_gain, pre.max_gain)

    def test_gate_disabled_keeps_quiet_audio(self):
        pre = ap.AudioPreprocessor(RATE, gate_db=None, normalize=False)
        rng = np.random.default_rng(0)
        x = _to_i16(rng.normal(0, 10 ** (-60 / 20) * FS, BLOCK))
        y = np.frombuffer(pre.process(x.tobytes()), dtype=np.int16)
        self.assertTrue(y.any())
        self.assertFalse(pre.last_gated)


class TestShapeAndPassthrough(unittest.TestCase):
    def test_output_length_and_dtype(self):
        pre = ap.AudioPreprocessor(RATE)
        for n in (BLOCK, 1234, 1, 0):
            x = _to_i16(_sine(500, 0.3)[:n])
            out = pre.process(x.tobytes())
            self.assertIsInstance(out, bytes)
            self.assertEqual(len(out), 2 * n)
            self.assertEqual(np.frombuffer(out, dtype=np.int16).size, n)

    def test_everything_off_is_identity(self):
        x = _to_i16(_sine(50, 0.4) + _sine(700, 0.3) + 3000.0)[:BLOCK]
        for hp in (None, 0, 0.0):
            pre = ap.AudioPreprocessor(RATE, highpass_hz=hp, normalize=False,
                                       gate_db=None)
            self.assertEqual(pre.process(x.tobytes()), x.tobytes())
            self.assertEqual(pre.last_gain, 1.0)

    def test_preprocess_wav_helper(self):
        x = _to_i16(_sine(50, 0.3) + _sine(1000, 0.2))
        y = ap.preprocess_wav(x, RATE, normalize=False, gate_db=None)
        self.assertEqual(y.dtype, np.int16)
        self.assertEqual(y.size, x.size)
        self.assertGreater(_level_db(x, 50) - _level_db(y, 50), 20.0)

    def test_stats_and_reset(self):
        pre = ap.AudioPreprocessor(RATE)
        _run_blocks(pre, _to_i16(_sine(400, 0.1)))
        s = pre.stats()
        self.assertEqual(set(s), {"gain", "rms_db", "gated"})
        self.assertGreater(s["gain"], 1.0)
        pre.reset()
        self.assertEqual(pre.last_gain, 1.0)
        self.assertEqual(pre.last_rms_db, -np.inf)


if __name__ == "__main__":
    unittest.main()
