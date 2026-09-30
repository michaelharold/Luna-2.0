import re
import queue
import datetime
from pathlib import Path

import numpy as np
import noisereduce as nr
import requests
import sounddevice as sd
from faster_whisper import WhisperModel
from piper import PiperVoice
from silero_vad import load_silero_vad, VADIterator

OLLAMA_MODEL = "luna-brain"
SAMPLE_RATE = 16000
VAD_CHUNK_SAMPLES = 512  # required chunk size for Silero VAD at 16kHz

# Multilingual (no ".en" suffix) so Malayalam speech transcribes as Malayalam instead
# of being forced through an English-only decoder, which just produces English-sounding
# gibberish for non-English audio. "small" trades some speed for the accuracy Malayalam
# needs on a CPU-bound target like a Raspberry Pi 5 (swap to "base" if that's too slow).
STT_MODEL = "small"

PIPER_MODEL = "models/piper/en_US-lessac-medium.onnx"
PIPER_CONFIG = "models/piper/en_US-lessac-medium.onnx.json"

STOP_PHRASES = {"bye", "goodbye", "good bye", "stop", "exit", "quit", "stop listening"}

# Software gain applied to the raw mic signal before VAD/denoise/STT ever see it —
# compensates for a quiet mic/room without needing to touch Windows input settings.
# Raise if it's still missing quieter speech, lower if quiet parts start clipping/distorting.
MIC_GAIN = 2.5

# Whisper's base.en model reliably mangles names and acronyms it hasn't seen (e.g. it
# heard "TKM" as "P.K.M." and "Aneesh G Nath" as "Anish G. Nuth" in testing). Passing
# the actual vocabulary as an initial_prompt biases the decoder toward these spellings.
DOMAIN_VOCABULARY = (
    "TKM College of Engineering, Thangal Kunju Musaliar, Karikode, Kollam, Kerala, "
    "Kilikollur, Dr. Aneesh G Nath, Dr. Dimple A Shajahan, Dr. Thushara A, "
    "Dr. Manu J Pillai, Dr. Anantha Padmanabhan N K, Dr. Shyna A, Dr. Reshma Sheik, "
    "Reena Mary George, Rehannara Begum T, Dr. Jini Raju, B. Mohammed Siyad, Nisa A K, "
    "Dr. Shameem Ansar, NAAC, KIRF, NBA, KTU, CGPU, HOD, Luna, Michael Harold Sony, "
    "Sreenandana, Devadathan, Raspberry Pi."
)

# Campus location (Kollam), hardcoded so weather lookups skip a geocoding round-trip.
WEATHER_LOCATION_NAME = "Kollam"
WEATHER_LAT, WEATHER_LON = 8.8932, 76.6141

# WMO weather codes used by Open-Meteo's current_weather field, collapsed to plain
# spoken descriptions. https://open-meteo.com/en/docs has the full table.
WMO_WEATHER_DESCRIPTIONS = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy with frost",
    51: "light drizzle", 53: "drizzle", 55: "dense drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "light freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "violent rain showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with light hail", 99: "thunderstorm with heavy hail",
}

# Wikipedia's API rejects requests without a descriptive User-Agent (returns 403).
WIKIPEDIA_HEADERS = {"User-Agent": "LunaVoiceAssistant/1.0 (offline campus assistant; contact: n/a)"}

LOG_PATH = Path("logs/voice_session.log")

_DATE_RE = re.compile(r"\b(what('?s| is) the )?(date|day)\b", re.IGNORECASE)
_TIME_RE = re.compile(r"\bwhat('?s| is) the time\b|\bwhat time\b", re.IGNORECASE)
_WEATHER_RE = re.compile(r"\bweather\b|\bhow(’|')?s it (looking|outside)\b|\bis it raining\b", re.IGNORECASE)

# Luna's refusals come in different flavors (see data/train_data.jsonl), and only one of
# them is safe to hand to the web fallback. "Didn't catch that" means the STT transcript
# itself may be garbage — searching it risks a spurious match on gibberish. Refusals that
# name a specific entity type ("...by that name", "...about that person", "...about the
# X department") mean Luna recognizes this as a question about a specific named thing in
# her domain that she doesn't have — that thing very likely doesn't exist (see: the
# P.K.M. college test earlier), and Wikipedia's keyword search will confidently return
# *something* for almost any query, so falling back there just re-introduces confident
# wrong answers instead of an honest "I don't know" for made-up local entities. Only the
# fully generic "I don't have that information" — a genuine open-domain knowledge gap
# (capital of France, quantum entanglement) — is safe to look up.
_GARBLED_MARKERS = ("didn't catch that", "did not catch that")
_CAPABILITY_REFUSAL_MARKERS = ("i can't do that", "i cannot do that")
_ENTITY_REFUSAL_MARKERS = ("by that name", "about that person", "about the")
_GENERIC_UNKNOWN_MARKERS = ("i don't have that information", "i do not have that information")


# Relying only on which refusal phrasing the fine-tuned model happens to pick is too
# fragile — it sometimes gives the fully generic refusal even for a campus-domain
# question (e.g. "Tell me about the chemistry department" got the generic "I don't have
# that information" instead of an entity-specific refusal), which would otherwise pass
# the check above. As a second, more reliable guard: if the *question itself* mentions
# campus-domain vocabulary, treat it as almost certainly about this specific campus, not
# real-world general knowledge — a generic real Wikipedia article that happens to share
# those words (a different college, a different department) is a false-confidence risk,
# not a genuine answer.
_DOMAIN_ADJACENT_TERMS = re.compile(
    r"\b(department|college|campus|hod|advisor|professor|faculty|principal|hostel|club|"
    r"fest|placement|lab|laboratory|canteen|library|hall|office|dean|lecturer|staff)\b",
    re.IGNORECASE,
)


def should_try_web_fallback(query, reply):
    if _DOMAIN_ADJACENT_TERMS.search(query):
        return False
    lowered = reply.lower()
    if any(m in lowered for m in _GARBLED_MARKERS + _CAPABILITY_REFUSAL_MARKERS + _ENTITY_REFUSAL_MARKERS):
        return False
    return any(m in lowered for m in _GENERIC_UNKNOWN_MARKERS)


def is_datetime_question(text):
    return bool(_DATE_RE.search(text) or _TIME_RE.search(text))


def answer_datetime(text):
    now = datetime.datetime.now()
    wants_time = bool(_TIME_RE.search(text))
    wants_date = bool(_DATE_RE.search(text))
    if wants_time and not wants_date:
        return f"It's {now.strftime('%I:%M %p').lstrip('0')}."
    if wants_date and not wants_time:
        return f"It's {now.strftime('%A, %B %d, %Y')}."
    return f"It's {now.strftime('%A, %B %d, %Y')}, {now.strftime('%I:%M %p').lstrip('0')}."


def is_weather_question(text):
    return bool(_WEATHER_RE.search(text))


def answer_weather():
    """Live current-weather lookup for the campus location via Open-Meteo (free, no API
    key). Requires internet — this is the one place voice_loop.py depends on a network
    connection; if the request fails, we say so honestly rather than guessing."""
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": WEATHER_LAT, "longitude": WEATHER_LON, "current_weather": "true"},
            timeout=5,
        )
        r.raise_for_status()
        current = r.json()["current_weather"]
        description = WMO_WEATHER_DESCRIPTIONS.get(current["weathercode"], "unclear conditions")
        temp_c = round(current["temperature"])
        return f"It's currently {temp_c} degrees Celsius in {WEATHER_LOCATION_NAME}, with {description}."
    except Exception:
        return "Sorry, I couldn't reach the weather service right now."


# Wikipedia's search index ranks on keywords, not natural-language questions — leaving
# interrogative phrasing in (see: "What is the capital of France?" matching the article
# "Closed-ended question" ahead of "France") measurably hurts relevance. Stripping the
# question wrapper down to its keywords before searching fixes that in testing.
_QUESTION_PREFIX_RE = re.compile(
    r"^(what('?s| is| are)|who('?s| is| are)|where('?s| is| are)|when('?s| is| are)|"
    r"why('?s| is| are)|how('?s| is| are)|tell me about|can you tell me)\s+(the\s+)?",
    re.IGNORECASE,
)


def _to_search_keywords(query):
    stripped = _QUESTION_PREFIX_RE.sub("", query).strip()
    return stripped.rstrip("?").strip() or query


def web_fallback(query):
    """Best-effort general-knowledge lookup via Wikipedia's free, no-key API, used only
    when Luna's fine-tuned model has no domain answer. This is not a real search engine —
    it only helps for encyclopedic/factual questions, not opinions or current events, and
    fails closed (returns None) on anything else so the caller can keep the honest refusal
    instead of speaking nothing or an error."""
    try:
        search = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": _to_search_keywords(query),
                     "format": "json", "srlimit": 1},
            headers=WIKIPEDIA_HEADERS,
            timeout=5,
        )
        search.raise_for_status()
        hits = search.json()["query"]["search"]
        if not hits:
            return None
        title = hits[0]["title"]

        summary = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}",
            headers=WIKIPEDIA_HEADERS,
            timeout=5,
        )
        summary.raise_for_status()
        extract = summary.json().get("extract", "").strip()
        if not extract:
            return None

        # Keep spoken replies short — Piper will read the whole string aloud, and a full
        # Wikipedia summary paragraph is too long for a voice answer.
        sentences = re.split(r"(?<=[.!?])\s+", extract)
        return " ".join(sentences[:2])
    except Exception:
        return None


def log_exchange(heard, reply):
    LOG_PATH.parent.mkdir(exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now().isoformat(timespec='seconds')}]\n")
        f.write(f"You said: {heard}\n")
        f.write(f"Luna: {reply}\n\n")

# Piper's phonemizer regularly mangles acronyms and initialisms it hasn't seen — it
# guesses a "word" pronunciation instead of reading the letters. Spelling them out
# with periods ("T. K. M.") is a reliable way to force letter-by-letter reading.
# This is a starting guess at what should be spelled out — adjust freely, and note
# some (like IEEE) have an idiomatic spoken form ("eye triple ee") rather than a
# strict letter-by-letter one if you want to hand-tune those further.
ACRONYMS = ["TKM", "NAAC", "NBA", "KIRF", "KTU", "CGPU", "HOD", "IEEE", "ISTE", "IEI", "CSI", "LPA", "PIN"]
_ACRONYM_RE = re.compile(r"\b(" + "|".join(re.escape(a) for a in ACRONYMS) + r")\b")


def _spell_out(word):
    return " ".join(f"{c}." for c in word)


def fix_pronunciation(text):
    return _ACRONYM_RE.sub(lambda m: _spell_out(m.group(0)), text)


def is_stop_phrase(text):
    normalized = re.sub(r"[^\w\s]", "", text.lower()).strip()
    words = normalized.split()
    if not words:
        return False
    # Treat it as a stop command only if the whole utterance is short and IS a stop
    # phrase, so "why did the bus stop" doesn't accidentally end the program.
    return normalized in STOP_PHRASES or (len(words) <= 3 and words[-1] in {"bye", "stop", "exit", "quit"})


def ask_luna(text):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": text, "stream": False},
    )
    body = r.json()
    if "response" not in body:
        return f"(Ollama error: {body.get('error', body)})"
    return body["response"].strip()


def speak(tts, text):
    for chunk in tts.synthesize(fix_pronunciation(text)):
        sd.play(chunk.audio_int16_array, chunk.sample_rate)
        sd.wait()


def listen_for_utterance(audio_queue, vad_iterator, max_seconds=20):
    """Blocks until it captures one spoken utterance (speech start -> sustained
    silence), using Silero VAD to find the boundaries. Returns a float32 numpy array."""
    buffer = []
    triggered = False
    max_chunks = int(max_seconds * SAMPLE_RATE / VAD_CHUNK_SAMPLES)

    while True:
        chunk = audio_queue.get()
        event = vad_iterator(chunk, return_seconds=False)

        if event and "start" in event and not triggered:
            triggered = True
            buffer = [chunk]
        elif triggered:
            buffer.append(chunk)
            if (event and "end" in event) or len(buffer) >= max_chunks:
                vad_iterator.reset_states()
                return np.concatenate(buffer)


def drain_queue(audio_queue):
    try:
        while True:
            audio_queue.get_nowait()
    except queue.Empty:
        pass


def main():
    print("Loading speech-to-text model...")
    stt = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")

    print("Loading text-to-speech voice...")
    tts = PiperVoice.load(PIPER_MODEL, PIPER_CONFIG)

    print("Loading voice activity detector...")
    vad_model = load_silero_vad()
    vad_iterator = VADIterator(
        vad_model, sampling_rate=SAMPLE_RATE, threshold=0.3,
        min_silence_duration_ms=600, speech_pad_ms=300,
    )

    audio_queue = queue.Queue()

    def callback(indata, frame_count, time_info, status):
        boosted = np.clip(indata[:, 0] * MIC_GAIN, -1.0, 1.0)
        audio_queue.put(boosted.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        blocksize=VAD_CHUNK_SAMPLES, callback=callback,
    )

    print("Luna voice loop ready. Speak any time; say 'bye' or 'stop' to exit.\n")
    with stream:
        speak(tts, "Hi, I'm listening.")
        # The mic picks up Luna's own greeting through the speakers; without this,
        # that echo gets transcribed as the first "utterance" from the user.
        drain_queue(audio_queue)
        vad_iterator.reset_states()
        while True:
            audio = listen_for_utterance(audio_queue, vad_iterator)

            # Denoise before transcription — helps separate the speaker from steady
            # background noise (fans, hum, room tone). prop_decrease < 1.0 keeps this
            # gentle so it doesn't eat quieter speech along with the noise.
            try:
                audio = nr.reduce_noise(y=audio, sr=SAMPLE_RATE, stationary=False, prop_decrease=0.6)
            except Exception as e:
                print(f"(denoise skipped: {e})")

            # No forced language: auto-detect per utterance so English and Malayalam
            # (and code-switching between them, common in campus speech) both transcribe
            # in their own script instead of Malayalam being mis-read as English.
            segments, _info = stt.transcribe(audio, initial_prompt=DOMAIN_VOCABULARY)
            heard = " ".join(seg.text for seg in segments).strip()

            # Discard whatever the mic picked up while Luna was talking, and reset
            # VAD state, so playback audio can't be misread as the next utterance.
            drain_queue(audio_queue)
            vad_iterator.reset_states()

            if not heard:
                continue

            print(f"You said: {heard}")

            if is_stop_phrase(heard):
                log_exchange(heard, "(stop phrase — session ended)")
                speak(tts, "Goodbye.")
                drain_queue(audio_queue)
                break

            # Date/time and weather are answered directly instead of going through the
            # fine-tuned model — they're not domain facts it was trained on, and date/time
            # in particular needs no network round-trip at all.
            if is_datetime_question(heard):
                reply = answer_datetime(heard)
            elif is_weather_question(heard):
                reply = answer_weather()
            else:
                reply = ask_luna(heard)
                if should_try_web_fallback(heard, reply):
                    fallback = web_fallback(heard)
                    if fallback:
                        reply = fallback

            print(f"Luna: {reply}\n")
            log_exchange(heard, reply)
            speak(tts, reply)
            drain_queue(audio_queue)
            vad_iterator.reset_states()


if __name__ == "__main__":
    main()
