# stt_engines.py
"""
stt_engines.py — the speech-to-text engines Luna can use, picked by number
with config.STT_ENGINE:

  0  Vosk only (vosk-model-en-in-0.5) — streaming, the original path
  1  Parakeet-TDT 0.6B v2 (NVIDIA, int8)          ~630 MB
  2  Whisper small.en (OpenAI, int8)               ~370 MB
  3  Whisper base.en  (OpenAI, int8, lighter)      ~160 MB

1-3 run through sherpa-onnx, fully offline. They are NOT streaming: Vosk
still listens live (it decides when an utterance has ended, and an
utterance it hears as silence is not re-read), then the chosen engine
re-transcribes that utterance's audio and its text is what Luna uses.
Vosk's confidence still feeds the addressing / misheard gates.

Measured with scripts/bench_stt_noise.py on 2026-10-13 (Mac M3, 8
sentences x 3 Indian-English `say` voices at 48 kHz, word error rate;
"babble" = other people talking; compute = seconds per second of audio):

                     clean  fan 10  fan 5  fan 0  babble 10  babble 5  compute
  0 Vosk             2.6 %   5.3 %  5.8 %  13.8 %   26.5 %    56.6 %   (live)
  1 Parakeet         0.5 %   0.0 %  2.1 %   4.2 %    1.6 %     4.8 %   0.036
  2 Whisper small    1.6 %   0.5 %  0.5 %   5.3 %    2.1 %     2.1 %   0.144
  3 Whisper base     1.6 %   2.1 %  3.7 %   6.3 %    4.2 %    10.6 %   0.048

Synthetic voices, not real people: re-check with your own voice and on the
Pi (expect several times the compute there).

Download the models with scripts/download_models.sh (into models/stt/).
"""

import os
import re
import threading

import numpy as np

_STT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "models", "stt")

# number -> (name, loader kind, model folder, file prefix)
ENGINES = {
    0: ("Vosk", None, None, None),
    1: ("Parakeet-TDT 0.6B", "transducer",
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8", ""),
    2: ("Whisper small.en", "whisper", "sherpa-onnx-whisper-small.en",
        "small.en-"),
    3: ("Whisper base.en", "whisper", "sherpa-onnx-whisper-base.en",
        "base.en-"),
}

_cache = {}
_lock = threading.Lock()


def name(engine):
    return ENGINES.get(engine, ("unknown",))[0]


def _files(engine):
    _, kind, folder, pre = ENGINES[engine]
    f = lambda n: os.path.join(_STT_DIR, folder, pre + n)   # noqa: E731
    if kind == "transducer":
        return dict(encoder=f("encoder.int8.onnx"),
                    decoder=f("decoder.int8.onnx"),
                    joiner=f("joiner.int8.onnx"), tokens=f("tokens.txt"))
    return dict(encoder=f("encoder.int8.onnx"), decoder=f("decoder.int8.onnx"),
                tokens=f("tokens.txt"))


def available(engine):
    """True when `engine` is a re-transcribing engine whose files exist."""
    if not ENGINES.get(engine, (None, None))[1]:
        return False
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False
    return all(os.path.exists(p) for p in _files(engine).values())


def load(engine, threads=4):
    """The sherpa-onnx recognizer for `engine` (cached), or None for Vosk /
    a missing model. Loading takes a few seconds — call it at start-up."""
    if not available(engine):
        return None
    with _lock:
        if engine not in _cache:
            import sherpa_onnx
            kind, files = ENGINES[engine][1], _files(engine)
            if kind == "transducer":
                rec = sherpa_onnx.OfflineRecognizer.from_transducer(
                    **files, num_threads=threads,
                    model_type="nemo_transducer",
                    decoding_method="greedy_search")
            else:
                rec = sherpa_onnx.OfflineRecognizer.from_whisper(
                    **files, language="en", task="transcribe",
                    num_threads=threads)
            _cache[engine] = rec
        return _cache[engine]


# Whisper marks non-speech: "[BLANK_AUDIO]", "(laughs)", "*coughs*", "♪"
_NON_SPEECH = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*|♪")


# The engines learnt American English and general vocabulary; Vosk's
# en-in model (and the knowledge base) say "programme" and "b tech". Only
# spellings and college terms — never real words that merely sound alike
# ("hostile" stays "hostile": guessing would make new mistakes).
_FIXES = [(re.compile(rf"\b{a}\b"), b) for a, b in (
    (r"programs?", lambda m: "programmes" if m.group().endswith("s")
     else "programme"),
    (r"b ?tech?|b ?tec|b ?tac|b ?take|btek|be tech", "b tech"),
    (r"m ?tech?|m ?tec|em tech", "m tech"),
    (r"b ?arch", "b arch"),
    (r"enrol+ment", "enrolment"),
)]


def normalize(text):
    """The engines write "B.Tech programme?" — the rest of Luna expects
    Vosk's style, "b tech programme": lower case, no punctuation, dots
    between letters as spaces, non-speech marks dropped, college terms
    spelt the way Vosk and the knowledge base spell them."""
    text = _NON_SPEECH.sub(" ", text or "")
    text = text.lower().replace(".", " ").replace("-", " ")
    text = " ".join("".join(c for c in text if c.isalnum() or c in " '")
                    .split())
    for rx, sub in _FIXES:
        text = rx.sub(sub, text)
    return text


def transcribe(rec, audio_16k):
    """Text of one utterance. audio_16k: 16 kHz mono, int16 bytes or a
    float32 array in -1..1. "" on an empty clip."""
    return transcribe_conf(rec, audio_16k)[0]


def transcribe_conf(rec, audio_16k):
    """(text, confidence): confidence is the mean token probability (0..1)
    where the engine reports one (Parakeet), else None (Whisper)."""
    if isinstance(audio_16k, (bytes, bytearray)):
        x = np.frombuffer(audio_16k, "<i2").astype(np.float32) / 32768.0
    else:
        x = np.asarray(audio_16k, dtype=np.float32)
    if not len(x):
        return "", None
    st = rec.create_stream()
    st.accept_waveform(16000, x)
    rec.decode_stream(st)
    r = st.result
    lp = list(getattr(r, "ys_log_probs", None) or ())
    conf = float(np.mean(np.exp(lp))) if lp else None
    return normalize(r.text), conf
