#!/usr/bin/env python3
"""
bench_vosk.py — measure Vosk STT cost on THIS machine (meant to be run on the Pi).

Answers the two questions that decide small-en-in-0.4 vs en-in-0.5:
  • how long after you stop talking does Luna have the text?  (finalization delay)
  • can the model keep up with real time?                     (real-time factor)
  • how much RAM does it hold resident?                       (peak RSS)

Run the SAME audio through both models for a fair A/B:

    # record one clip once, reuse it for every model
    python3 scripts/bench_vosk.py --record sample.wav --seconds 6 --mic-rate 32000

    # then benchmark each model against that clip
    python3 scripts/bench_vosk.py --model vosk-model-small-en-in-0.4 --wav sample.wav
    python3 scripts/bench_vosk.py --model vosk-model-en-in-0.5       --wav sample.wav

Robustness A/B for the audio preprocessing stage (audio_preprocess.py): inject
the dongle's 50 Hz mains hum or white noise into the clip, optionally clean it
the way Luna does live, and score the transcript against what was said:

    python3 scripts/bench_vosk.py --model vosk-model-en-in-0.5 --wav s1.wav \
        --ref "where is the artificial intelligence lab located" \
        --add-hum 0.5 --preprocess          # vs. the same without --preprocess
    python3 scripts/bench_vosk.py ... --add-noise -35 --preprocess

Real-time factor (RTF) < 1.0 means the model decodes faster than audio plays —
i.e. it keeps up in real time. RTF above ~1.0 means it lags.
"""

import argparse
import json
import os
import re
import sys
import time
import wave

import numpy as np

VOSK_RATE = 16000
CHUNK = 4000  # samples per AcceptWaveform call — matches a typical live feed
FULL_SCALE = 32767.0


def _peak_rss_bytes():
    """Peak resident set size of this process, normalized to bytes.
    Linux (the Pi) reports ru_maxrss in KiB; macOS reports it in bytes."""
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss * 1024 if sys.platform.startswith("linux") else rss


def _resample_to_16k(x, src_rate):
    if src_rate == VOSK_RATE:
        return x.astype(np.int16)
    n = int(len(x) * VOSK_RATE / src_rate)
    xi = np.interp(
        np.linspace(0, len(x), n, endpoint=False),
        np.arange(len(x)),
        x.astype(np.float32),
    )
    return xi.astype(np.int16)


def record(path, seconds, mic_rate):
    """Capture one clip from the mic so every model is benchmarked on it."""
    import sounddevice as sd
    print(f"[bench] Recording {seconds}s at {mic_rate} Hz — speak now...")
    audio = sd.rec(int(seconds * mic_rate), samplerate=mic_rate,
                   channels=1, dtype="int16")
    sd.wait()
    audio = _resample_to_16k(audio.flatten(), mic_rate)
    save_wav_16k(path, audio)
    print(f"[bench] Saved {len(audio) / VOSK_RATE:.2f}s to {path}")


def load_wav_16k(path):
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16)
    if rate != VOSK_RATE:
        print(f"[bench] Resampling {rate} Hz -> {VOSK_RATE} Hz")
        x = _resample_to_16k(x, rate)
    return x


def save_wav_16k(path, audio):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(VOSK_RATE)
        w.writeframes(np.asarray(audio, dtype=np.int16).tobytes())


# ── Degradations (what the Pi's dongle does to the signal) ────────────────────

def _to_int16_clipped(x, label):
    """Clip like the ADC would and say how much clipped — that clipping is the
    part of the hum problem no filter can undo, so it is worth seeing."""
    clipped = int(np.sum((x > FULL_SCALE) | (x < -FULL_SCALE - 1)))
    if clipped:
        print(f"[bench] {label}: {clipped} samples "
              f"({100.0 * clipped / max(1, x.size):.1f}%) clipped")
    return np.clip(np.rint(x), -32768, 32767).astype(np.int16)


def add_hum(audio, ampl):
    """Add 50 Hz mains hum at `ampl` x full scale plus a 100 Hz harmonic at a
    quarter of that — the ungrounded-dongle signature from the Pi."""
    t = np.arange(audio.size) / VOSK_RATE
    hum = (ampl * FULL_SCALE * np.sin(2 * np.pi * 50.0 * t)
           + (ampl / 4.0) * FULL_SCALE * np.sin(2 * np.pi * 100.0 * t))
    print(f"[bench] Adding 50 Hz hum at {ampl:.2f} FS (+100 Hz at {ampl / 4:.3f} FS)")
    return _to_int16_clipped(audio.astype(np.float64) + hum, "hum")


def add_noise(audio, dbfs):
    """Add white noise with the given RMS level in dBFS (e.g. -35)."""
    sigma = 10.0 ** (dbfs / 20.0) * FULL_SCALE
    rng = np.random.default_rng(0)          # repeatable across runs
    print(f"[bench] Adding white noise at {dbfs:.1f} dBFS RMS")
    return _to_int16_clipped(audio.astype(np.float64)
                             + rng.normal(0.0, sigma, audio.size), "noise")


def preprocess(audio):
    """Run the clip through Luna's live preprocessing (audio_preprocess.py),
    250 ms block by 250 ms block, exactly like speech_to_text.py does."""
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    if root not in sys.path:
        sys.path.insert(0, root)
    from audio_preprocess import preprocess_wav
    print("[bench] Preprocessing (high-pass / gate / AGC)")
    return preprocess_wav(audio, VOSK_RATE)


# ── Word error rate ───────────────────────────────────────────────────────────

def _words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def word_errors(ref_words, hyp_words):
    """Levenshtein distance over words: substitutions + insertions + deletions."""
    prev = list(range(len(hyp_words) + 1))
    for i, r in enumerate(ref_words, 1):
        cur = [i]
        for j, h in enumerate(hyp_words, 1):
            cur.append(min(prev[j] + 1,                    # deletion
                           cur[j - 1] + 1,                 # insertion
                           prev[j - 1] + (0 if r == h else 1)))  # substitution
        prev = cur
    return prev[-1]


def word_error_rate(ref, hyp):
    """WER as a fraction of the reference length (can exceed 1.0)."""
    ref_w, hyp_w = _words(ref), _words(hyp)
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    return word_errors(ref_w, hyp_w) / len(ref_w)


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark(model_path, audio, ref=None):
    from vosk import Model, KaldiRecognizer

    rss_before = _peak_rss_bytes()
    t0 = time.perf_counter()
    model = Model(model_path)
    load_s = time.perf_counter() - t0
    rec = KaldiRecognizer(model, VOSK_RATE)

    audio_s = len(audio) / VOSK_RATE

    # Stream the clip through in real-time-sized chunks, timing only the decode.
    decode_s = 0.0
    text_parts = []
    for i in range(0, len(audio), CHUNK):
        chunk = audio[i:i + CHUNK].tobytes()
        t = time.perf_counter()
        if rec.AcceptWaveform(chunk):
            decode_s += time.perf_counter() - t
            text_parts.append(json.loads(rec.Result()).get("text", ""))
        else:
            decode_s += time.perf_counter() - t

    # Finalization: the delay you actually feel after you stop talking.
    t = time.perf_counter()
    final = json.loads(rec.FinalResult()).get("text", "")
    final_delay_s = time.perf_counter() - t
    if final:
        text_parts.append(final)

    rss_after = _peak_rss_bytes()
    transcript = " ".join(p for p in text_parts if p).strip()

    print("\n" + "=" * 60)
    print(f"model               {model_path}")
    print(f"transcript          {transcript!r}")
    if ref is not None:
        wer = word_error_rate(ref, transcript)
        print(f"reference           {ref!r}")
        print(f"word error rate     {wer:6.2f}   "
              f"({word_errors(_words(ref), _words(transcript))} errors / "
              f"{len(_words(ref))} words)")
    print("-" * 60)
    print(f"audio duration      {audio_s:6.2f} s")
    print(f"model load time     {load_s:6.2f} s")
    print(f"decode wall time    {decode_s:6.2f} s")
    print(f"real-time factor    {decode_s / audio_s:6.2f}   (<1.0 = keeps up)")
    print(f"finalization delay  {final_delay_s:6.2f} s   (felt after you stop)")
    print(f"peak RSS            {rss_after / 1e6:6.0f} MB "
          f"(+{(rss_after - rss_before) / 1e6:.0f} MB for this model)")
    print("=" * 60)
    return transcript


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", metavar="OUT.wav",
                    help="record a clip to this path and exit")
    ap.add_argument("--seconds", type=float, default=6.0,
                    help="record length (with --record)")
    ap.add_argument("--mic-rate", type=int, default=32000,
                    help="mic sample rate for --record (see config.MIC_SAMPLE_RATE)")
    ap.add_argument("--model", help="path to a Vosk model dir to benchmark")
    ap.add_argument("--wav", help="16k mono wav to run through the model")
    ap.add_argument("--ref", metavar="TEXT",
                    help="what was actually said — prints the word error rate")
    ap.add_argument("--add-hum", type=float, metavar="AMPL",
                    help="inject 50 Hz hum at AMPL x full scale (e.g. 0.4) "
                         "plus a 100 Hz harmonic at a quarter of it")
    ap.add_argument("--add-noise", type=float, metavar="DBFS",
                    help="inject white noise at this RMS level (e.g. -35)")
    ap.add_argument("--preprocess", action="store_true",
                    help="clean the audio with audio_preprocess.py first")
    ap.add_argument("--save-wav", metavar="OUT.wav",
                    help="also write the (degraded, preprocessed) audio here")
    args = ap.parse_args()

    if args.record:
        record(args.record, args.seconds, args.mic_rate)
        return

    if not args.model or not args.wav:
        ap.error("give --model and --wav (or --record to capture a clip first)")

    audio = load_wav_16k(args.wav)
    if args.add_hum is not None:
        audio = add_hum(audio, args.add_hum)
    if args.add_noise is not None:
        audio = add_noise(audio, args.add_noise)
    if args.preprocess:
        audio = preprocess(audio)
    if args.save_wav:
        save_wav_16k(args.save_wav, audio)
        print(f"[bench] Wrote benchmarked audio to {args.save_wav}")
    benchmark(args.model, audio, ref=args.ref)


if __name__ == "__main__":
    main()
