#!/usr/bin/env python3
# scripts/bench_stt_noise.py
"""
bench_stt_noise.py — how well Luna hears in a noisy room, as a word error rate.

Known sentences are spoken by macOS `say` (Indian English voices, at the
mic's sample rate), mixed with room noise at a given signal-to-noise ratio,
and fed through the capture pipeline in 250 ms blocks exactly like
speech_to_text.py, then into Vosk. Three pipelines on the SAME audio:

  before  the old capture path: np.interp resample with no anti-alias
          filter, then the preprocessor (hum filter, gate, AGC)
  now     audio_preprocess.Resampler + AudioPreprocessor as configured
  +NR     "now" with noise suppression forced on (AUDIO_NOISE_SUPPRESS)

Then one column per downloaded stt_engines.py engine (1 Parakeet,
2 Whisper small, 3 Whisper base): each whole clip re-transcribed the way
Luna does with STT_ENGINE > 0 — resampled to 16 kHz only, no gate / AGC
(Parakeet scored better on that than on the preprocessed audio). Each
engine's compute time per second of audio is printed at the end.
ENGINES=1,3 picks some; SHOW_ERRORS=1 prints every wrong transcription.

Noise: "babble" (four other voices talking at once — a corridor, a crowd)
and "fan" (pink noise plus hiss — fans, AC). Needs macOS for `say`; run it
on the Mac, with the Pi's 48000 in RATES to model the Pi's dongle.

    venv/bin/python scripts/bench_stt_noise.py
    RATES=48000 CONDS=clean:0,fan:0,babble:10 venv/bin/python scripts/bench_stt_noise.py

Measured 2026-10-12 (vosk-model-en-in-0.5, 24 sentences, 3 voices; runs
vary by ~2 points): the anti-aliasing resampler is equal or better
everywhere (clean 3.7 -> 2.6 %, chatter at 10 dB 29 -> 27 %, fan at 0 dB
20 -> 15-18 %). Noise suppression halves errors in loud fan noise (0 dB:
~20 % -> ~10 %) but hurts chatter (5 dB: ~57 % -> ~71 %), so it is off by
default. Chatter is the hard case — no filter removes speech from speech;
a closer or directional microphone is what helps there.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import config  # noqa: E402
from audio_preprocess import AudioPreprocessor, Resampler  # noqa: E402

import stt_engines  # noqa: E402

CACHE = os.path.join(tempfile.gettempdir(), "luna-bench-stt-noise")
SENTENCES = [
    "what are the latest achievements of the college",
    "who is the head of the computer science department",
    "what is the fee for the b tech programme",
    "tell me about the placement record of the department",
    "where is the central library",
    "what are the hostel facilities for students",
    "when do the semester exams start",
    "how many students are there in the computer science department",
]
SPEAKERS = ["Rishi", "Aman", "Tara"]
BABBLE_VOICES = ["Samantha", "Daniel", "Flo (English (US))",
                 "Reed (English (UK))"]
BABBLE_TEXT = ("Did you see the match yesterday, it was really close until the "
               "last over. I think the canteen closes early today so we should "
               "go now. The lab assignment is due next week and I have not even "
               "started it yet. Can you send me the notes from the morning class. ")


def say(text, voice, rate, name):
    path = os.path.join(CACHE, f"{name}_{rate}.wav")
    if not os.path.exists(path):
        subprocess.run(["say", "-v", voice, "-r", "165", "-o", path,
                        f"--data-format=LEI16@{rate}", "--", text], check=True)
    with wave.open(path) as w:
        return (np.frombuffer(w.readframes(w.getnframes()), "<i2")
                .astype(np.float64) / 32768.0)


def babble(rate, n):
    out, rng = np.zeros(n), np.random.default_rng(7)
    for i, v in enumerate(BABBLE_VOICES):
        x = say(BABBLE_TEXT * 2, v, rate, f"babble{i}")
        start = int(rng.integers(0, max(1, len(x) - n)))
        seg = x[start:start + n]
        out[:len(seg)] += seg
    return out


def fan(rate, n):
    rng = np.random.default_rng(3)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / rate)
    spec[1:] /= np.sqrt(f[1:] / 100.0)                 # pink: fans, AC
    x = np.fft.irfft(spec, n)
    x = x / np.std(x) + 0.3 * rng.standard_normal(n)   # plus hiss
    return x / np.std(x)


def mix(speech, kind, snr_db, rate):
    pad = np.zeros(int(0.8 * rate))                    # room noise around it
    s = np.concatenate([pad, speech, pad])
    if kind == "clean":
        return s
    noise = (babble if kind == "babble" else fan)(rate, len(s))
    noise *= np.sqrt(np.mean(speech ** 2)
                     / (np.mean(noise ** 2) * 10 ** (snr_db / 10)))
    y = s + noise
    return y * min(1.0, 0.95 / np.max(np.abs(y)))


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = np.arange(len(h) + 1)
    for i, rw in enumerate(r, 1):
        prev, d[0] = d.copy(), i
        for j, hw in enumerate(h, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (rw != hw))
    return int(d[-1]), len(r)


def _interp_16k(block, rate):
    """The old resampler, verbatim: no anti-alias filter."""
    a = np.frombuffer(block, dtype=np.int16)
    n = int(len(a) * 16000 / rate)
    return np.interp(np.linspace(0, 1, n, endpoint=False),
                     np.linspace(0, 1, len(a), endpoint=False),
                     a).astype(np.int16).tobytes()


def before(rate):
    pre = AudioPreprocessor(16000, noise_floor_db=None)
    return lambda b: pre.process(_interp_16k(b, rate))


def now(rate):
    rs, pre = Resampler(rate), AudioPreprocessor(16000)
    return lambda b: pre.process(rs(b))


def with_nr(rate):
    rs = Resampler(rate)
    pre = AudioPreprocessor(16000, noise_floor_db=config.AUDIO_NOISE_FLOOR_DB)
    return lambda b: pre.process(rs(b))


def transcribe(model, pipeline, signal, rate):
    from vosk import KaldiRecognizer
    rec = KaldiRecognizer(model, 16000)
    pcm = (np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes()
    step = int(rate * 0.25) * 2
    words = []
    for i in range(0, len(pcm), step):
        if rec.AcceptWaveform(pipeline(pcm[i:i + step])):
            words.append(json.loads(rec.Result()).get("text", ""))
    words.append(json.loads(rec.FinalResult()).get("text", ""))
    return " ".join(w for w in words if w)


def resampled(rate):
    rs = Resampler(rate)
    return lambda b: rs(b)


_secs = {}     # engine -> [compute time, audio time]


def transcribe_engine(engine, rec, pipeline, signal, rate):
    pcm = (np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes()
    step = int(rate * 0.25) * 2
    audio = b"".join(pipeline(pcm[i:i + step]) for i in range(0, len(pcm), step))
    t = time.time()
    text = stt_engines.transcribe(rec, audio)
    acc = _secs.setdefault(engine, [0.0, 0.0])
    acc[0] += time.time() - t
    acc[1] += len(audio) / 2 / 16000
    return text


def main():
    os.makedirs(CACHE, exist_ok=True)
    rates = [int(r) for r in os.environ.get("RATES", "44100,48000").split(",")]
    conds = [c.split(":") for c in os.environ.get(
        "CONDS", "clean:0,fan:10,fan:5,fan:0,babble:10,babble:5").split(",")]
    from vosk import Model, SetLogLevel
    SetLogLevel(-1)
    model = Model(config.VOSK_MODEL_PATH)
    wanted = [int(e) for e in os.environ.get("ENGINES", "1,2,3").split(",")]
    engines = [(e, stt_engines.load(e, int(os.environ.get("THREADS", "4"))))
               for e in wanted if stt_engines.available(e)]
    show = os.environ.get("SHOW_ERRORS") == "1"
    print(f"{'rate':>6} {'noise':<12} {'before':>8} {'now':>8} {'+NR':>8}"
          + "".join(f" {'[' + str(e) + ']':>8}" for e, _ in engines))
    for rate in rates:
        clips = [(s, say(s, v, rate, f"{v}_{i}"))
                 for v in SPEAKERS for i, s in enumerate(SENTENCES)]
        for kind, snr in conds:
            sig = [(s, mix(x, kind, float(snr), rate)) for s, x in clips]
            row = []
            for make in (before, now, with_nr):
                errs = total = 0
                for ref, y in sig:
                    e, n = wer(ref, transcribe(model, make(rate), y, rate))
                    errs, total = errs + e, total + n
                row.append(100.0 * errs / total)
            for engine, rec in engines:
                errs = total = 0
                for ref, y in sig:
                    hyp = transcribe_engine(engine, rec, resampled(rate), y, rate)
                    e, n = wer(ref, hyp)
                    errs, total = errs + e, total + n
                    if show and e:
                        print(f"    [{engine}] {kind} {snr}: {hyp!r}  (said: {ref!r})")
                row.append(100.0 * errs / total)
            label = kind if kind == "clean" else f"{kind} {snr} dB"
            print(f"{rate:>6} {label:<12} "
                  + " ".join(f"{v:>7.1f}%" for v in row), flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    for engine, (busy, audio) in sorted(_secs.items()):
        print(f"[{engine}] {stt_engines.name(engine)}: {busy / audio:.3f} s of "
              f"compute per second of audio")
    print(f"({time.time() - t0:.0f}s)")
