# audio_preprocess.py
"""
audio_preprocess.py — cleans each mic block before Vosk hears it.

WHY this exists: the Pi's USB sound dongle has no earth ground, so the
collar mic picks up 50 Hz mains hum (plus its 100 Hz harmonic). The hum
rides on top of the speech, clips the ADC on loud words and floods the low
end of the spectrum — Vosk's recognition falls apart even though the person
sounded perfectly clear (docs/PROJECT_PIPELINE.md, "Known constraints"). On top of
that a collar mic at arm's length gives a quiet, level-varying signal, and
room noise between sentences turns into garbage words.

Three stages, in this order, on every 250 ms block Luna captures:

  (a) high-pass  — a Butterworth IIR removes DC and the 50/100 Hz hum. The
                   filter STATE is carried from block to block, so the
                   stream is filtered exactly as if it were one long signal
                   (no clicks at the block seams).
  (b) noise gate — a block whose RMS sits below AUDIO_GATE_DB is replaced by
                   digital silence. This runs BEFORE the gain stage so
                   quiet room noise is never amplified into "speech".
  (c) slow AGC   — a running peak envelope (instant attack, slow release)
                   sets one gain per block, capped at AUDIO_MAX_GAIN, that
                   brings the loudest sample to AUDIO_TARGET_PEAK. Slow
                   release means the gain does not pump between words; a
                   rising gain is ramped across the block so it never steps,
                   a falling gain applies at once so a loud onset never clips.

Everything is float64 numpy on 4000-sample blocks — microseconds on the Pi.
scipy provides the filter design and the C-speed recursion; if scipy were
ever missing, a pure-numpy biquad cascade takes over so this module never
raises at import time. Tunables live in config.py (AUDIO_*).
"""

import math

import numpy as np

from config import (AUDIO_HIGHPASS_HZ, AUDIO_NORMALIZE, AUDIO_TARGET_PEAK,
                    AUDIO_MAX_GAIN, AUDIO_GATE_DB)

try:                                    # scipy is a transitive dependency
    from scipy.signal import butter, sosfilt, sosfilt_zi
    _HAVE_SCIPY = True
except Exception:                       # pragma: no cover — Pi always has it
    _HAVE_SCIPY = False

FULL_SCALE = 32767.0
BLOCK_SECONDS = 0.25          # Luna's capture block length (speech_to_text.py)

# Butterworth order of the high-pass. Not in config.py on purpose: it is an
# implementation detail of how steep the roll-off is, not a "what sound do I
# want" knob. 2nd order only takes 12 dB off 50 Hz with the 100 Hz cutoff in
# config; 4th order takes 24 dB and still leaves 200 Hz+ untouched (<0.02 dB).
HIGHPASS_ORDER = 4

# AGC envelope: the gain follows the loudest block of the last AGC_HOLD_S
# seconds of speech (silence does not count), so within a sentence the gain
# is one constant — measured: normalising every 250 ms block to its own peak
# modulated the level 1-3 dB between syllables and doubled Vosk's errors on
# noisy loud speech. Once the hold expires the envelope releases at
# AGC_RELEASE_DB_PER_S, i.e. ~3 s from a loud talker to full gain for a
# quiet one.
AGC_HOLD_S = 1.5
AGC_RELEASE_DB_PER_S = 6.0


# ── Filter design ─────────────────────────────────────────────────────────────

def _design_highpass(rate, cutoff_hz, order):
    """Second-order sections (n, 6) for a Butterworth high-pass, or None when
    the cutoff is disabled / out of range for this sample rate."""
    if not cutoff_hz or cutoff_hz <= 0 or cutoff_hz >= rate / 2:
        return None
    if _HAVE_SCIPY:
        return butter(order, cutoff_hz / (rate / 2), btype="highpass",
                      output="sos")
    return _design_highpass_numpy(rate, cutoff_hz, order)


def _design_highpass_numpy(rate, cutoff_hz, order):
    """scipy-free Butterworth design: bilinear-transform each pole pair into an
    RBJ-style high-pass biquad. Odd orders round up to the next even order."""
    n = max(2, order + (order % 2))
    w0 = 2.0 * math.pi * cutoff_hz / rate
    cos_w0, sin_w0 = math.cos(w0), math.sin(w0)
    sections = []
    for k in range(n // 2):
        # Butterworth pole angles give the Q of each biquad section.
        q = 1.0 / (2.0 * math.sin(math.pi * (2 * k + 1) / (2 * n)))
        alpha = sin_w0 / (2.0 * q)
        b0 = (1 + cos_w0) / 2
        b1 = -(1 + cos_w0)
        b2 = (1 + cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
        sections.append([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0])
    return np.array(sections, dtype=np.float64)


def _sosfilt_numpy(sos, x, zi):
    """Direct-form-II-transposed biquad cascade in pure numpy/python. Slow
    (per-sample loop) but only ever used when scipy is unavailable."""
    y = np.array(x, dtype=np.float64, copy=True)
    for s in range(sos.shape[0]):
        b0, b1, b2, _, a1, a2 = sos[s]
        z0, z1 = zi[s]
        out = np.empty_like(y)
        for i in range(y.shape[0]):
            xi = y[i]
            yi = b0 * xi + z0
            z0 = b1 * xi - a1 * yi + z1
            z1 = b2 * xi - a2 * yi
            out[i] = yi
        zi[s, 0], zi[s, 1] = z0, z1
        y = out
    return y, zi


# ── The preprocessor ──────────────────────────────────────────────────────────

class AudioPreprocessor:
    """Streaming high-pass + gate + slow AGC for int16 mono blocks.

    One instance per audio stream — it keeps the filter state, the peak
    envelope and the last gain between calls, so feed it consecutive blocks.
    `process()` never raises on ordinary input and always returns bytes of
    the same length as it was given.
    """

    def __init__(self, rate, highpass_hz=AUDIO_HIGHPASS_HZ,
                 normalize=AUDIO_NORMALIZE, target_peak=AUDIO_TARGET_PEAK,
                 max_gain=AUDIO_MAX_GAIN, gate_db=AUDIO_GATE_DB,
                 filter_order=HIGHPASS_ORDER, hold_s=AGC_HOLD_S,
                 release_db_per_s=AGC_RELEASE_DB_PER_S):
        self.rate = int(rate)
        self.highpass_hz = highpass_hz
        self.normalize = bool(normalize)
        self.target_peak = float(target_peak)
        self.max_gain = float(max_gain)
        self.gate_db = gate_db
        self.hold_s = float(hold_s)
        self.release_db_per_s = float(release_db_per_s)

        self._sos = _design_highpass(self.rate, highpass_hz, filter_order)
        self.reset()

    def reset(self):
        """Forget all state (filter memory, envelope, gain) — e.g. on a new stream."""
        if self._sos is not None:
            # rest state: zeros (silence in, silence out), same shape scipy uses
            self._zi = np.zeros((self._sos.shape[0], 2), dtype=np.float64)
        else:
            self._zi = None
        self._envelope = 0.0        # running peak, full-scale units
        self._peaks = []            # (speech-sample clock, peak) of recent blocks
        self._clock = 0             # speech samples seen (gated blocks don't count)
        self._prev_gain = None      # gain at the end of the previous block
        self.last_gain = 1.0        # stats for logging
        self.last_rms_db = -math.inf
        self.last_gated = False

    # ── stages ────────────────────────────────────────────────────────────────

    def _highpass(self, x):
        if self._sos is None:
            return x
        if _HAVE_SCIPY:
            y, self._zi = sosfilt(self._sos, x, zi=self._zi)
            return y
        y, self._zi = _sosfilt_numpy(self._sos, x, self._zi)
        return y

    def _gated(self, x):
        """Update last_rms_db; True if the block is below the gate."""
        rms = math.sqrt(float(np.mean(x * x))) if x.size else 0.0
        self.last_rms_db = (20.0 * math.log10(rms / FULL_SCALE)
                            if rms > 0 else -math.inf)
        return self.gate_db is not None and self.last_rms_db < self.gate_db

    def _release_factor(self, n_samples):
        """Envelope decay for a block this long (release is set in dB/s)."""
        seconds = n_samples / self.rate
        return 10.0 ** (-(self.release_db_per_s * seconds) / 20.0)

    def _held_peak(self, x):
        """Loudest block peak within the last hold_s seconds of speech."""
        self._clock += x.size
        horizon = self._clock - int(self.hold_s * self.rate)
        self._peaks = [(t, p) for (t, p) in self._peaks if t > horizon]
        # appended AFTER pruning so the current block is always in the window
        self._peaks.append((self._clock, float(np.max(np.abs(x))) if x.size else 0.0))
        return max(p for _, p in self._peaks)

    def _gain_for(self, x):
        """Slow AGC: instant attack (a louder block is never over-boosted),
        peak hold (one gain per sentence), then slow release, capped at
        max_gain."""
        held = self._held_peak(x)
        if held >= self._envelope:
            self._envelope = held
        else:
            self._envelope = max(held,
                                 self._envelope * self._release_factor(x.size))
        if self._envelope <= 0.0:
            return 1.0
        return min(self.max_gain, self.target_peak * FULL_SCALE / self._envelope)

    def _apply_gain(self, x, gain):
        """Attack is instant: a LOWER gain applies to the whole block at once,
        because ramping down from the previous (higher) gain would clip the
        loud start of this block — measured: that turned "where is" into
        "where's". Release is ramped across the block so the gain never
        steps upward at a seam (a step is a click)."""
        if self._prev_gain is None or gain <= self._prev_gain:
            y = x * gain
        else:
            y = x * np.linspace(self._prev_gain, gain, x.size, endpoint=True)
        self._prev_gain = gain
        return y

    # ── public ────────────────────────────────────────────────────────────────

    def process(self, block_bytes):
        """int16 mono bytes in → int16 mono bytes out, same length."""
        x = np.frombuffer(block_bytes, dtype=np.int16)
        if x.size == 0:
            return block_bytes
        x = x.astype(np.float64)

        x = self._highpass(x)                                   # (a)

        if self._gated(x):                                       # (b)
            # The envelope HOLDS through silence: the next word after a pause
            # then gets the same gain as the last one, instead of a boosted
            # onset that the attack has to yank back down.
            self.last_gated = True
            self.last_gain = 0.0
            return np.zeros(x.size, dtype=np.int16).tobytes()
        self.last_gated = False

        if self.normalize:                                       # (c)
            gain = self._gain_for(x)
            x = self._apply_gain(x, gain)
            self.last_gain = gain
        else:
            self.last_gain = 1.0

        return np.clip(np.rint(x), -32768, 32767).astype(np.int16).tobytes()

    def stats(self):
        """Last-block numbers for a log line: {'gain', 'rms_db', 'gated'}."""
        return {"gain": self.last_gain, "rms_db": self.last_rms_db,
                "gated": self.last_gated}


# ── Whole-array helper (bench / tests) ────────────────────────────────────────

def preprocess_wav(int16_array, rate, block_seconds=BLOCK_SECONDS, **kwargs):
    """Run a FRESH preprocessor over a whole int16 array in 250 ms blocks —
    exactly the way the live mic feed reaches it. Returns an int16 array of
    the same length. Extra kwargs go to AudioPreprocessor."""
    x = np.asarray(int16_array, dtype=np.int16)
    pre = AudioPreprocessor(rate, **kwargs)
    block = max(1, int(rate * block_seconds))
    out = np.empty(x.size, dtype=np.int16)
    for i in range(0, x.size, block):
        chunk = x[i:i + block]
        out[i:i + chunk.size] = np.frombuffer(pre.process(chunk.tobytes()),
                                              dtype=np.int16)
    return out
