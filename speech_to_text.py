# speech_to_text.py
"""
speech_to_text.py — Vosk STT with strict mic blocking + robust device handling.

Improvements over the old version:
  • Persistent input stream (opened once, not per-listen) — lower latency,
    no device re-open glitches between turns.
  • Auto-detects the mic's native sample rate and resamples to 16 kHz for
    Vosk with numpy interpolation — works with 16k/32k/44.1k/48k mics.
  • Graceful "microphone not found" handling: Luna keeps running (face,
    vision, gestures) and retries the mic every 5 seconds instead of crashing.
  • Multiple wake words (WAKE_WORDS in config), matched on word boundaries.
  • Respects state.mic_unblock_time — never listens while Luna speaks.
"""

import queue
import json
import re
import time
import threading
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

from shared_state import state
from config import (
    WAKE_WORDS,
    AUDIO_INPUT_DEVICE,
    VOSK_MODEL_PATH,
    MIC_SAMPLE_RATE,
    VOSK_SAMPLE_RATE,
    CONVO_TIMEOUT,
    CONVO_TIMEOUT_AFTER_QUESTION,
    POST_SPEAK_DELAY,
    DOUBLE_FLUSH,
    STT_CONFIDENCE_MIN,
    STT_MIN_WORDS,
    STT_RESCUE_WITH_KNOWLEDGE,
    STT_RESCUE_MATCH_MIN,
    STT_REQUIRE_WAKE_EACH_TURN,
    ADDRESSED_SCORING,
    AUDIO_PREPROCESS,
)
import addressing

_model = Model(VOSK_MODEL_PATH)
_q     = queue.Queue(maxsize=50)   # bounded — audio can't pile up unbounded

_mic_rate     = None    # actual sample rate the stream runs at
_mic_ok       = False
_stream       = None
_help_printed = False

# Sentinel returned by listen() when a wake word was heard with no question
# attached ("Luna!") — main.py answers with a short acknowledgement.
WAKE_ACK = "__WAKE_ACK__"

# ── Password-protected voice shutdown ─────────────────────────────────────────
# Say "exit" → Luna asks for the password → say "exit" again to quit.
# listen() returns one of these sentinels; main.py speaks the prompt / quits.
EXIT_PROMPT  = "__EXIT_PROMPT__"    # heard "exit" — ask for the password
TERMINATE    = "__TERMINATE__"     # correct password — shut down
BAD_PASSWORD = "__BAD_PASSWORD__"  # wrong password — stay on
_awaiting_terminate = False


def _shutdown_command(text):
    """Two-step voice shutdown, checked BEFORE wake-word / meaning gating so the
    single word "exit" isn't filtered out. Returns a sentinel if this utterance
    was part of the shutdown flow, else None."""
    global _awaiting_terminate
    words = set(re.findall(r"[a-z]+", text.lower()))
    if _awaiting_terminate:
        _awaiting_terminate = False
        return TERMINATE if ("exit" in words) else BAD_PASSWORD
    if "exit" in words:
        _awaiting_terminate = True
        with state.lock:                       # keep listening for the password
            state.conversation_active = True
            state.last_activity_time  = time.time()
        return EXIT_PROMPT
    return None


# ── Device + sample rate detection ────────────────────────────────────────────

def _pick_input_device():
    """Return (device_index_or_None, sample_rate) or (None, None) if no mic."""
    if AUDIO_INPUT_DEVICE is not None:
        try:
            dev = sd.query_devices(AUDIO_INPUT_DEVICE)
            if dev["max_input_channels"] > 0:
                rate = MIC_SAMPLE_RATE or int(dev["default_samplerate"])
                print(f"[STT] Using configured device {AUDIO_INPUT_DEVICE}: "
                      f"{dev['name']} @ {rate} Hz")
                return AUDIO_INPUT_DEVICE, rate
            print(f"[STT] Configured device {AUDIO_INPUT_DEVICE} has no "
                  f"input channels — falling back to auto-detect")
        except Exception as e:
            print(f"[STT] Configured device {AUDIO_INPUT_DEVICE} error: {e} "
                  f"— falling back to auto-detect")

    # auto-detect: prefer the system default input, then any input device
    candidates = []
    try:
        default_in = sd.default.device[0]
        if default_in is not None and default_in >= 0:
            candidates.append(default_in)
    except Exception:
        pass

    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"[STT] Could not query audio devices: {e}")
        return None, None

    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0 and i not in candidates:
            candidates.append(i)

    for i in candidates:
        try:
            dev  = sd.query_devices(i)
            rate = MIC_SAMPLE_RATE or int(dev["default_samplerate"])
            # verify the device actually accepts this configuration
            sd.check_input_settings(device=i, samplerate=rate,
                                    channels=1, dtype="int16")
            print(f"[STT] Auto-detected input device {i}: "
                  f"{dev['name']} @ {rate} Hz")
            return i, rate
        except Exception:
            # try the device's own default rate before giving up on it
            try:
                dev  = sd.query_devices(i)
                rate = int(dev["default_samplerate"])
                sd.check_input_settings(device=i, samplerate=rate,
                                        channels=1, dtype="int16")
                print(f"[STT] Auto-detected input device {i}: "
                      f"{dev['name']} @ {rate} Hz (device default)")
                return i, rate
            except Exception:
                continue

    return None, None


def _print_mic_help():
    global _help_printed
    if _help_printed:
        return
    _help_printed = True
    print("=" * 60)
    print("[STT] NO MICROPHONE FOUND — voice input disabled.")
    print("[STT] Luna keeps running (face + vision + gestures).")
    print("[STT] Will retry the microphone every 5 seconds.")
    print("[STT] To fix, see the Troubleshooting section in README.md:")
    print("[STT]   python3 -c \"import sounddevice as sd; print(sd.query_devices())\"")
    print("[STT]   then set AUDIO_INPUT_DEVICE in config.py")
    print("=" * 60)


# ── Persistent stream (background thread) ─────────────────────────────────────

def _callback(indata, frames, time_info, status):
    try:
        _q.put_nowait(bytes(indata))
    except queue.Full:
        pass   # drop oldest-style: consumer is behind, losing a block is fine


def _stream_keeper():
    """Keeps a mic stream open forever; retries every 5 s if the mic vanishes."""
    global _stream, _mic_ok, _mic_rate, _help_printed

    while True:
        if _mic_ok:
            time.sleep(1.0)
            continue

        device, rate = _pick_input_device()
        if device is None and rate is None:
            _print_mic_help()
            with state.lock:
                state.mic_ok = False
            time.sleep(5.0)
            continue

        try:
            stream = sd.RawInputStream(
                samplerate=rate,
                blocksize=int(rate * 0.25),   # 250 ms blocks
                dtype="int16",
                channels=1,
                callback=_callback,
                device=device,
            )
            stream.start()
            _stream       = stream
            _mic_rate     = rate
            _mic_ok       = True
            _help_printed = False
            with state.lock:
                state.mic_ok = True
            print(f"[STT] Microphone stream running @ {rate} Hz")
        except Exception as e:
            print(f"[STT] Could not open microphone: {e} — retrying in 5s")
            _print_mic_help()
            with state.lock:
                state.mic_ok = False
            time.sleep(5.0)


threading.Thread(target=_stream_keeper, daemon=True, name="mic").start()


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _resample_to_16k(raw_bytes):
    """Resample int16 mono audio from _mic_rate to VOSK_SAMPLE_RATE."""
    audio = np.frombuffer(raw_bytes, dtype=np.int16)
    if _mic_rate == VOSK_SAMPLE_RATE or _mic_rate is None:
        return raw_bytes
    ratio   = VOSK_SAMPLE_RATE / _mic_rate
    n_out   = int(len(audio) * ratio)
    if n_out == 0:
        return b""
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out,      endpoint=False)
    resampled = np.interp(x_new, x_old, audio).astype(np.int16)
    return resampled.tobytes()


# ── Audio preprocessing (audio_preprocess.py) ─────────────────────────────────
# High-pass (kills the dongle's 50 Hz hum + DC), noise gate and slow AGC on
# every 16 kHz block before Vosk sees it. Created lazily on the first block;
# any failure falls back to the raw block so preprocessing can never take
# the mic down. Off entirely with AUDIO_PREPROCESS = False in config.
_pre          = None
_pre_disabled = False
_pre_warned   = False


def _preprocess(data):
    global _pre, _pre_disabled, _pre_warned
    if not AUDIO_PREPROCESS or _pre_disabled:
        return data
    try:
        if _pre is None:
            from audio_preprocess import AudioPreprocessor
            _pre = AudioPreprocessor(VOSK_SAMPLE_RATE)
            print(f"[audio] Preprocessing on: high-pass {_pre.highpass_hz} Hz, "
                  f"gate {_pre.gate_db} dBFS, AGC x{_pre.max_gain:g} max")
        return _pre.process(data)
    except Exception as e:
        if not _pre_warned:
            _pre_warned = True
            print(f"[audio] Preprocessing failed ({e}) — using raw audio")
        if _pre is None:              # couldn't even build it: stop retrying
            _pre_disabled = True
        return data


# ── Utterance loudness (addressing's "standing at her mic" signal) ─────────────
# People talking TO Luna stand at her collar mic; room chatter arrives
# quieter. Each block's level is measured AFTER the hum filter but BEFORE the
# AGC (audio_preprocess's last_rms_db), so gain can't fake closeness. At the
# end of an utterance the loudest half of its blocks (speech, not the pauses)
# is averaged in power terms and mapped to 0..1 by addressing.loudness_score.
_utt_levels = []          # per-block dBFS since the last final result
_UTT_MAX_BLOCKS = 120     # 30 s of 250 ms blocks — plenty for one sentence


def _block_db(raw16k):
    """Level of one block: the preprocessor's measurement when it ran,
    else computed from the raw 16 kHz block."""
    if _pre is not None and AUDIO_PREPROCESS and not _pre_disabled:
        db = getattr(_pre, "last_rms_db", None)
        if db is not None:
            return db
    x = np.frombuffer(raw16k, dtype=np.int16).astype(np.float64)
    if not len(x):
        return -120.0
    rms = float(np.sqrt(np.mean(x * x)))
    return 20.0 * np.log10(max(rms, 1e-9) / 32768.0)


def _track_level(db):
    _utt_levels.append(db)
    if len(_utt_levels) > _UTT_MAX_BLOCKS:
        del _utt_levels[0]


def _utterance_db():
    """Loudness of the utterance just finished (dBFS), then reset."""
    levels = sorted((d for d in _utt_levels if d > -119.0), reverse=True)
    _utt_levels.clear()
    if not levels:
        return -120.0
    top = levels[:max(1, len(levels) // 2)]
    power = np.mean([10.0 ** (d / 10.0) for d in top])
    return float(10.0 * np.log10(max(power, 1e-12)))


def _note_heard(text, conf, loudness, why):
    """Hand the accepted utterance's confidence + addressing reason to the
    conversation log, so the answered turn records WHY it was accepted."""
    try:
        import conversation_log
        conversation_log.note_heard(text, conf=conf, loudness=loudness,
                                    reason=why)
    except Exception:
        pass


def _flush_queue():
    _utt_levels.clear()        # levels from before the flush (Luna's own
                               # voice, echo) must not count toward the next
    while True:
        try:
            _q.get_nowait()
        except queue.Empty:
            break


def _match_wake_word(text):
    """Word-boundary wake word match. Returns cleaned text or None.
    Longest wake phrase first, so "hey luna ..." strips the whole phrase
    instead of leaving a stray "hey" in the question."""
    words = text.split()
    for wake in sorted(WAKE_WORDS, key=lambda w: -len(w.split())):
        wake_parts = wake.split()
        n = len(wake_parts)
        for i in range(len(words) - n + 1):
            if words[i:i + n] == wake_parts:
                cleaned = " ".join(words[:i] + words[i + n:]).strip()
                return cleaned
    return None


# ── Recognizer — created once, Reset() between turns (cheap on the Pi) ────────
_rec = None


def _get_recognizer():
    global _rec
    if _rec is None:
        _rec = KaldiRecognizer(_model, VOSK_SAMPLE_RATE)
        _rec.SetWords(True)   # emit per-word confidences for the meaning gate
    else:
        _rec.Reset()
    return _rec


# ── Meaning gate — ignore surrounding chatter / garbled other-language speech ──

def _avg_conf(result):
    """Average Vosk word-confidence (0..1) of a full result. Background and
    other-language audio that Vosk mistranscribes tends to score low here.
    Returns 1.0 when no per-word info is available (so we never over-reject)."""
    words = result.get("result")
    if not words:
        return 1.0
    confs = [w.get("conf", 1.0) for w in words]
    return sum(confs) / len(confs) if confs else 1.0


def _is_meaningful(text, conf):
    """True if `text` is worth answering: clear speech (high confidence), or —
    even if garbled — it still matches something in the knowledge base."""
    if len(text.split()) < STT_MIN_WORDS:
        return False
    if conf >= STT_CONFIDENCE_MIN:
        return True
    # low confidence — optionally rescue, but only if it CLEARLY matches the
    # knowledge base (STT_RESCUE_MATCH_MIN, stricter than the offline-answer
    # threshold). Turn the rescue off entirely for maximum strictness.
    if not STT_RESCUE_WITH_KNOWLEDGE:
        return False
    try:
        from brain import knowledge_match_score   # deferred: brain loads heavy
        return knowledge_match_score(text) >= STT_RESCUE_MATCH_MIN
    except Exception:
        return False


def _accept_utterance(text, conf, awaiting, loudness=0.0):
    """In-conversation gate: is this utterance aimed at Luna?
    Returns (accepted, reason) — the reason carries the per-signal breakdown
    so tuning on the Pi can be done straight from the log.

    With ADDRESSED_SCORING on, the decision is addressing.decide(): the
    weighted multi-signal score (name, question shape, "you", KB relevance, a
    visible face, loudness, continuity) plus the rule that clear speech alone
    is never enough. Off, it's the original confidence-plus-meaning gate."""
    if not ADDRESSED_SCORING:
        ok = _is_meaningful(text, conf)
        return ok, f"legacy gate, conf={conf:.2f}"

    with state.lock:
        face_visible = state.face_detected
    try:
        from brain import knowledge_match_score
        kb = knowledge_match_score(text)
    except Exception:
        kb = 0.0
    ok, why = addressing.decide(text, conf, kb_score=kb,
                                face_visible=face_visible,
                                awaiting_answer=awaiting, loudness=loudness)
    parts = addressing.explain(text, conf, kb_score=kb,
                               face_visible=face_visible,
                               awaiting_answer=awaiting, loudness=loudness)
    return ok, f"{why}; {addressing.format_explain(parts)}"


def _convo_timeout():
    """How long silence may last before the conversation window closes. After
    Luna asks a follow-up she gives the person longer to think (see
    brain._mark_followup, which sets awaiting_answer), so her own question
    doesn't time out before they can answer it."""
    with state.lock:
        awaiting = state.awaiting_answer
    return CONVO_TIMEOUT_AFTER_QUESTION if awaiting else CONVO_TIMEOUT


def _expire_conversation():
    """Conversation window ran out — back to wake-word mode.
    Stamps convo_expired_time so the face can play its subtle 'rest' cue."""
    print("[STT] Conversation timed out — waiting for wake word")
    with state.lock:
        state.conversation_active = False
        state.awaiting_answer     = False
        state.convo_expired_time  = time.time()
        state.listening           = False
    # End of session: wipe the LLM history so the next visitor starts fresh
    # and nothing from this conversation leaks into their answers. Deferred
    # import mirrors _is_meaningful — brain is already loaded by this point.
    try:
        from brain import reset_history
        reset_history()
    except Exception:
        pass


# ── Main listen ───────────────────────────────────────────────────────────────

def listen():
    """
    Waits until the mic is allowed (Luna not speaking, echo-guard passed),
    then listens. In conversation mode no wake word is needed until timeout.
    Returns WAKE_ACK when a wake word was heard alone, the recognised text
    when actionable, "" otherwise.
    """

    # no mic? — don't spin, just idle politely (keeper thread is retrying)
    if not _mic_ok:
        time.sleep(1.0)
        return ""

    # ── Wait until mic is allowed ─────────────────────────────────────────────
    # `tts_active` covers synthesis as well as playback — Luna must not listen
    # while a reply is being generated, or she hears the tail of her own
    # question echo and the first words of her own answer.
    was_blocked = False
    while True:
        with state.lock:
            busy         = state.speaking or state.tts_active
            unblock_time = state.mic_unblock_time
        now = time.time()

        if busy or now < unblock_time:
            was_blocked = True
            remaining = unblock_time - now
            time.sleep(min(0.1, remaining) if remaining > 0 else 0.05)
            continue
        break

    # Only pay the settle delay when Luna actually just spoke — the old code
    # added POST_SPEAK_DELAY to EVERY listen cycle, even silent ones.
    if was_blocked:
        _flush_queue()
        time.sleep(POST_SPEAK_DELAY)
        if DOUBLE_FLUSH:
            _flush_queue()

    rec = _get_recognizer()

    # check conversation timeout
    with state.lock:
        active        = state.conversation_active
        last_activity = state.last_activity_time

    if active and (time.time() - last_activity) > _convo_timeout():
        _expire_conversation()
        active = False

    # "listening" (blue pulse on the face) means attentive conversation mode;
    # passively waiting for a wake word stays visually idle
    with state.lock:
        state.listening = active
        state.luna_mode = "listening" if active else "idle"

    while True:
        try:
            data = _q.get(timeout=0.5)
        except queue.Empty:
            # mic may have vanished mid-listen
            if not _mic_ok:
                with state.lock:
                    state.listening = False
                    state.luna_mode = "idle"
                return ""
            # conversation window can also run out mid-silence
            if active and (time.time() - last_activity) > _convo_timeout():
                _expire_conversation()
                active = False
                with state.lock:
                    state.luna_mode = "idle"
            continue

        # if Luna started replying, abort this listen (mic gets flushed after)
        with state.lock:
            if state.speaking or state.tts_active:
                state.listening = False
                state.luna_mode = "idle"
                return ""

        data = _resample_to_16k(data)
        if not data:
            continue
        raw_block = data
        data = _preprocess(data)
        _track_level(_block_db(raw_block))

        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text   = result.get("text", "").strip()

            with state.lock:
                state.listening  = False
                state.luna_mode  = "processing"
                state.heard_text = text

            utt_db   = _utterance_db()
            loudness = addressing.loudness_score(utt_db)
            if text:
                with state.lock:
                    state.utterance_loudness = loudness

            if not text:
                # silence chunk — also the spot to notice a timed-out window
                if active and (time.time() - last_activity) > _convo_timeout():
                    _expire_conversation()
                    active = False
                with state.lock:
                    state.luna_mode = "idle" if not active else "listening"
                    state.listening = active
                continue   # keep listening — don't restart the whole cycle

            conf = _avg_conf(result)
            print(f"[Luna heard] {text}  (conf={conf:.2f}, "
                  f"level={utt_db:.0f} dBFS, loud={loudness:.2f})")

            # password-protected voice shutdown — before any gating
            cmd = _shutdown_command(text)
            if cmd is not None:
                with state.lock:
                    state.luna_mode = "processing"
                return cmd

            cleaned = _match_wake_word(text)
            if cleaned is not None:
                print("[STT] Wake word — conversation active")
                _note_heard(cleaned or text, conf, loudness, "wake word")
                with state.lock:
                    state.conversation_active = True
                    state.awaiting_answer     = False
                    state.last_activity_time  = time.time()
                return cleaned if cleaned else WAKE_ACK

            if active:
                # in-window follow-up: only answer if it's real speech aimed at
                # Luna — otherwise it's surrounding chatter, so ignore it and do
                # NOT extend the window (garbage must not keep the window alive).
                with state.lock:
                    awaiting = state.awaiting_answer
                # Strict mode requires a wake word every turn — EXCEPT right
                # after Luna asked a question (so continuity still works).
                strict_block = STT_REQUIRE_WAKE_EACH_TURN and not awaiting
                accepted, why = (False, "wake word required") if strict_block \
                    else _accept_utterance(text, conf, awaiting, loudness)
                if accepted:
                    print(f"[STT] Accepted ({why}): \"{text}\"")
                    _note_heard(text, conf, loudness, why)
                    with state.lock:
                        # they answered — drop back to the normal timeout until
                        # Luna asks something again
                        state.awaiting_answer    = False
                        state.last_activity_time = time.time()
                    return text

                print(f"[STT] Ignored ({why}): \"{text}\"")
                try:                  # a wrongly ignored question is fixable
                    import conversation_log      # only if it's on record
                    conversation_log.log_ignored(text, conf, loudness, why)
                except Exception:                # noqa: BLE001
                    pass
                with state.lock:
                    state.luna_mode = "listening"
                    state.listening = True
                continue

            with state.lock:
                state.luna_mode = "idle"
            return ""

        partial = json.loads(rec.PartialResult()).get("partial", "")
        if partial:
            with state.lock:
                state.heard_text = partial
