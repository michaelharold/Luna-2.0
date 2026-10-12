# lip_sync.py
"""
lip_sync.py — drives the talking mouth from the voice Luna is actually playing.

The mouth used to flap to a made-up sine envelope from the moment a reply was
handed to TTS, so it moved before any sound came out, kept moving through the
silence between sentences, and never matched a single syllable.

Now every chunk of speech is a wav before it is played. Its loudness is turned
into an envelope — one 0-1 value per animation frame — and the TTS engine
registers that envelope with the wall-clock time the chunk becomes AUDIBLE.
The mouth then reads the envelope at "now": open on loud syllables, nearly
shut on soft ones, closed in pauses and between sentences.

    envelope(samples, rate)   pure: audio -> per-frame 0..1 mouth opening
    load_wav(path)            16-bit PCM wav -> (float samples, rate)
    LipSync                   thread-safe timeline of envelopes being played
    track                     the one LipSync that text_to_speech feeds

No pygame and no audio device here, so all of it is unit-testable.
"""

import threading
import wave

import numpy as np

#: Envelope resolution — one value per rendered frame at 60 fps.
FPS = 60

#: Loudness range mapped onto the mouth: a syllable at the utterance's loud
#: level opens it fully, anything this many dB quieter leaves it shut.
RANGE_DB = 30.0

#: Below this absolute level (dBFS) a frame is silence however the rest of
#: the utterance is normalised — keeps hiss and dither from moving the mouth.
SILENCE_DB = -55.0


def envelope(samples, rate, fps=FPS):
    """Per-frame mouth opening (0..1) for mono float `samples` in -1..1.

    RMS per 1/fps window, in dB, normalised to the utterance's own loud level
    (90th percentile of the sounding frames) — so a quiet voice still opens
    the mouth fully and a loud one doesn't pin it open."""
    samples = np.asarray(samples, dtype=np.float32)
    n = int(np.ceil(len(samples) * fps / rate)) if len(samples) else 0
    if n == 0:
        return []
    starts = (np.arange(n) * rate / fps).astype(np.int64)
    sq = samples.astype(np.float64) ** 2
    sums = np.add.reduceat(sq, starts)
    counts = np.diff(np.append(starts, len(sq)))
    rms = np.sqrt(sums / np.maximum(counts, 1))
    db = 20.0 * np.log10(rms + 1e-9)

    sounding = db > SILENCE_DB
    if not sounding.any():
        return [0.0] * n
    loud = float(np.percentile(db[sounding], 90))
    env = np.clip((db - (loud - RANGE_DB)) / RANGE_DB, 0.0, 1.0)
    env[~sounding] = 0.0
    return [float(v) for v in env]


def load_wav(path):
    """Read a PCM wav as (mono float32 samples in -1..1, sample rate)."""
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width {width}")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


class LipSync:
    """The envelopes of the speech being played, placed on the wall clock.

    `energy(now)` is None until something has been added since the last
    reset() — the caller then knows there is no real audio to follow (e.g. a
    fallback TTS path) and can animate some other way. Once a track exists,
    times outside every track are silence (0.0): the mouth closes between
    sentences rather than inventing movement."""

    def __init__(self):
        self._lock = threading.Lock()
        self._tracks = []          # (start, fps, envelope)
        self._armed = False

    def reset(self):
        with self._lock:
            self._tracks = []
            self._armed = False

    def add(self, env, fps, start):
        """`env` becomes audible at wall-clock `start`, one value per 1/fps."""
        with self._lock:
            # Only the current and the next chunk ever matter.
            self._tracks = self._tracks[-1:] + [(start, fps, list(env))]
            self._armed = True

    def energy(self, now):
        with self._lock:
            if not self._armed:
                return None
            for start, fps, env in self._tracks:
                i = int((now - start) * fps)
                if 0 <= i < len(env) and now >= start:
                    return env[i]
        return 0.0


#: The timeline text_to_speech feeds and its mouth-energy loop reads.
track = LipSync()
