import os
import sys
import time
import signal
import threading

# First, before the libraries that print on import / load (pygame, Vosk,
# MediaPipe, TensorFlow Lite): clean terminal unless LOG_VERBOSE is on.
import console
console.quiet_libraries()

from config import TORCH_THREADS, PI_MODEL, IS_MAC
import torch
torch.set_num_threads(TORCH_THREADS)

console.detail("[Luna] Running in Mac dev mode" if IS_MAC else
               f"[Luna] Running on Raspberry Pi {PI_MODEL}")
console.detail(f"[Luna] PyTorch threads: {TORCH_THREADS}")

# Load the piper voice model in the background, overlapped with the camera /
# vision / brain startup below. Without this the model load (several seconds on
# the Pi) landed on the FIRST answer instead, which is most of why the first
# reply used to take ~11 s to make a sound.
from config import TTS_WARMUP
if TTS_WARMUP:
    from piper_tts import tts
    threading.Thread(target=tts.warmup, daemon=True, name="tts-warmup").start()

# Same trick for the local LLM: the gguf takes several seconds to load on the
# Pi, so it loads in the background during startup instead of on the first
# question. Until it's ready, questions fall back to the knowledge base.
from config import LLM_WARMUP
if LLM_WARMUP:
    import llm_local
    threading.Thread(target=llm_local.warmup, daemon=True,
                     name="llm-warmup").start()

from camera_thread import start_camera
start_camera()
time.sleep(0.5)

from vision_module import start_vision
start_vision()
time.sleep(1.0)

from config import ENABLE_GESTURES
if ENABLE_GESTURES:
    # Importing gesture_module builds the MediaPipe hand tracker (an OpenGL
    # context and a TensorFlow Lite model), so it is only imported when used.
    from gesture_module import start_gesture
    start_gesture()
    time.sleep(0.5)
else:
    console.detail("[gesture] Disabled (ENABLE_GESTURES=False) — hand "
                   "gestures ignored")

from behavior_engine import start_behavior
start_behavior()
time.sleep(0.2)

from proactive import start_proactive
start_proactive()

import random
import difflib

from speech_to_text import (listen, WAKE_ACK,
                            EXIT_PROMPT, TERMINATE, BAD_PASSWORD)
from text_to_speech import speak
from brain import process
from shared_state import state
from servo_module import servo
from config import (WAKE_REPLIES, ECHO_GUARD_WINDOW,
                    ECHO_RUN_THRESH, ECHO_OVERLAP_THRESH)

# Everything above this point (camera/vision/gesture threads, TTS engine,
# brain's models, STT model) has finished loading — only now is it safe for
# gesture reactions to start speaking. Before this, camera warm-up (auto
# exposure/white balance still settling) plus someone's hands near the
# camera during setup can trick gesture detection into firing early.
with state.lock:
    state.ready = True
print("[Luna] Ready")


def _is_self_echo(text):
    """Defense-in-depth against Luna hearing her own voice through the
    speaker (see text_to_speech.py docstring for the timing side of this —
    that's the primary fix; this is cleanup for trailing echo/reverb that
    slips past the mic-blocking window).

    Requires BOTH a decent contiguous word-run match AND decent overall
    word overlap with Luna's last reply — either check alone is too easy
    to trigger on ordinary short sentences full of common words ("can",
    "you", "i"...), which would wrongly swallow real user speech."""
    with state.lock:
        last      = state.last_spoken_text
        last_time = state.last_spoken_time

    if not last or (time.time() - last_time) > ECHO_GUARD_WINDOW:
        return False

    heard_words = text.lower().split()
    last_words  = last.split()
    if not heard_words:
        return False

    overlap = sum(1 for w in heard_words if w in set(last_words)) / len(heard_words)
    sm      = difflib.SequenceMatcher(None, heard_words, last_words, autojunk=False)
    run     = sm.find_longest_match(0, len(heard_words), 0, len(last_words)).size
    run_ratio = run / len(heard_words)

    return run_ratio >= ECHO_RUN_THRESH and overlap >= ECHO_OVERLAP_THRESH


def voice_loop():
    while True:
        text = listen()
        try:
            if text == WAKE_ACK:
                # wake word alone ("Luna!") — short acknowledgement
                speak(random.choice(WAKE_REPLIES))
            elif text == EXIT_PROMPT:
                speak("Say the password to terminate.")
            elif text == BAD_PASSWORD:
                speak("Wrong password. I will stay on.")
            elif text == TERMINATE:
                speak("Password accepted. Shutting down. Goodbye!")
                signal.raise_signal(signal.SIGTERM)   # triggers _shutdown() cleanly
                return
            elif text:
                if _is_self_echo(text):
                    console.detail(f"[Luna] Ignoring self-echo: \"{text}\"")
                else:
                    # speak() serializes internally — an answer is never
                    # dropped, even if a gesture reaction is mid-sentence
                    process(text)
        finally:
            # never leave the mode stuck on "processing" (e.g. empty input)
            with state.lock:
                if not state.speaking and state.luna_mode == "processing":
                    state.luna_mode = "idle"
        time.sleep(0.05)


threading.Thread(target=voice_loop, daemon=True).start()


_shutting_down = False


def _shutdown(*_):
    """Clean up once, then leave with os._exit. Ctrl+C used to run this twice
    (the signal, then the `finally` below) and then crash Python's own
    teardown ("segmentation fault") while the camera / mic / model threads
    were still inside native code."""
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True
    print("\n[Luna] Shutting down — centering servos, releasing GPIO")
    try:
        servo.cleanup()
    except Exception:
        pass
    try:
        from piper_tts import tts as _tts
        _tts.shutdown()
    except Exception:
        pass
    try:
        import llm_local as _llm
        _llm.shutdown()
    except Exception:
        pass
    try:
        import pygame
        pygame.quit()
    except Exception:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

from face_renderer import renderer_loop

try:
    renderer_loop()
finally:
    _shutdown()
