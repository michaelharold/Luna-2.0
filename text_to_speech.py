"""
text_to_speech.py — Luna TTS front end.

Responsibilities
================
  • serialize speech so two voices never overlap
  • hold the microphone shut for the whole of synthesis + playback
  • drive the mouth animation and the arm sway **in time with the audio**
  • record what was said for the self-echo guard

Animation/voice sync
--------------------
The old version set `state.speaking = True` and started the arm sway *before*
handing the text to the TTS engine. Because synthesis took seconds, the face
started mouthing and the arms started swaying long before any sound came out —
the "animation starts 3-4 s before the voice" symptom.

Now there are two distinct states:

  state.tts_active  — from the moment a reply is handed to TTS. Blocks the mic.
                      Nothing visible happens.
  state.speaking    — set by the `on_start` callback the instant the audio
                      device actually begins playing. Drives the mouth energy
                      (see `_energy_loop`) and the servo arm sway.

So the mic is protected for the whole window, while the body only moves while
there is genuinely sound coming out.
"""

import threading
import time
import random
import math

from piper_tts import tts, chunk_stream

from shared_state import state
from servo_module import servo
from config import MIC_BLOCK_AFTER_SPEAK


# ============================================================================
# Talking mouth energy animation
# ============================================================================
# Purely cosmetic: a smooth pseudo-random envelope that looks like speech.
# Gated on state.speaking, which is now only true while audio is really playing.

_energy_phase = 0.0


def _energy_loop():
    global _energy_phase

    while True:
        with state.lock:
            speaking = state.speaking

        if speaking:
            _energy_phase += 0.18

            slow = abs(math.sin(_energy_phase * 0.9))
            mid  = abs(math.sin(_energy_phase * 2.3)) * 0.5
            fast = abs(math.sin(_energy_phase * 5.1)) * 0.2

            raw    = (slow + mid + fast) / 1.7
            jitter = random.uniform(-0.08, 0.08)
            energy = max(0.0, min(1.0, raw + jitter))

            with state.lock:
                state.audio_energy = energy
        else:
            _energy_phase = 0.0
            with state.lock:
                state.audio_energy = 0.0

        time.sleep(0.016)


threading.Thread(target=_energy_loop, daemon=True, name="mouth").start()


# ============================================================================
# Speech lock
# ============================================================================
# Brain responses wait their turn; gesture reactions (can_drop=True) are
# skipped rather than queued, so a wave never talks over an answer.

_speak_lock = threading.Lock()


def _on_audio_start():
    """Fired by the TTS engine the moment the first audio chunk hits the
    speaker. Everything that should look/move like talking starts here."""
    with state.lock:
        state.speaking = True
    servo.start_talking()


def speak(text, can_drop=False):
    """Speak `text`, blocking until playback finishes.

    `text` is either a plain string, or an iterable of text chunks that may
    still be arriving (a live LLM stream) — in that case Luna starts speaking
    the first sentence while the rest is still being generated.
    """
    is_stream = not isinstance(text, str)
    if not is_stream and not text.strip():
        return

    if can_drop:
        if not _speak_lock.acquire(blocking=False):
            return
    else:
        _speak_lock.acquire()

    spoken_parts = []
    start_time = time.time()

    try:
        with state.lock:
            # mic shut from here (synthesis included); body still motionless
            state.tts_active     = True
            state.luna_mode      = "speaking"
            state.frozen_emotion = state.emotion

        def _tap(chunks):
            """Pass chunks through to the engine, logging and recording them."""
            for chunk in chunks:
                if chunk and chunk.strip():
                    spoken_parts.append(chunk.strip())
                    print(f"[Luna] {chunk.strip()}")
                yield chunk

        if is_stream:
            tts.speak_chunks(_tap(chunk_stream(text)), on_start=_on_audio_start)
        else:
            print(f"[Luna] {text}")
            spoken_parts.append(text)
            tts.speak(text, on_start=_on_audio_start)

    finally:
        elapsed = time.time() - start_time

        # stop the arm sway immediately — the worker walks the arms back to
        # rest and then releases them, so they end up completely still
        servo.stop_talking()

        full_text = " ".join(spoken_parts).strip()

        with state.lock:
            state.speaking       = False
            state.tts_active     = False
            state.audio_energy   = 0.0
            state.frozen_emotion = None
            state.luna_mode      = "idle"

            # The mic was blocked for the whole of synthesis+playback; this
            # only has to cover the speaker's own tail and room reverb.
            state.mic_unblock_time = time.time() + MIC_BLOCK_AFTER_SPEAK

            state.last_spoken_text = full_text.lower()
            state.last_spoken_time = time.time()

            # Don't let Luna's own speaking time count against the
            # conversation timeout.
            if state.conversation_active:
                state.last_activity_time = time.time()

        print(f"[TTS] finished in {elapsed:.1f}s")

        _speak_lock.release()
