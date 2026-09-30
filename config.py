# config.py

import os
import platform
import time

# Load .env file if present (simple parser — no extra dependency needed).
# .env is git-ignored, so secrets stay out of version control.
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# ── Run mode: one config, two machines ────────────────────────────────────────
# "auto" — detect: a Mac runs in mac mode, anything else runs in pi mode.
# "mac"  — force Mac-dev mode  (windowed face, USB/FaceTime camera, `say` TTS,
#          simulated servos, osascript volume).
# "pi"   — force Pi product mode (fullscreen kiosk, CSI camera, Piper TTS,
#          real servos, pactl volume).
# Everything platform-dependent below derives from IS_MAC / IS_PI, so the SAME
# config file deploys to both machines — the established pattern, made explicit.
RUN_MODE = "auto"
IS_MAC = (RUN_MODE == "mac") or (RUN_MODE == "auto"
                                 and platform.system() == "Darwin")
IS_PI  = not IS_MAC

# ── Pi 5 fixed config ─────────────────────────────────────────────────────────
PI_MODEL        = 5
VISION_FPS      = 8
GESTURE_FPS     = 6
RENDER_FPS      = 60
DEEPFACE_EVERY  = 4
TORCH_THREADS   = 4

# ── Camera ────────────────────────────────────────────────────────────────────
# Where Luna's vision gets its frames:
#   "picam" — Raspberry Pi CSI camera (ribbon cable) via libcamera/rpicam-vid.
#             This is the on-board Pi Camera. Captured with a subprocess so it
#             works regardless of the venv's Python version.
#   "usb"   — USB / V4L2 webcam through OpenCV (cv2.VideoCapture).
#   "esp32" — ESP32-CAM MJPEG stream over WiFi (ESP32_STREAM below).
# Mode-aware default: the Pi uses its CSI camera, a Mac uses the built-in /
# USB webcam. Replace the expression with a plain string to force one.
CAMERA_SOURCE  = "usb" if IS_MAC else "picam"

# Pi CSI camera (rpicam-vid) settings
PICAM_WIDTH    = 640
PICAM_HEIGHT   = 480
PICAM_FPS      = 15
PICAM_HFLIP    = False   # set True if the image is left-right mirrored
PICAM_VFLIP    = False   # set True if the camera is mounted upside down

# USB / V4L2 webcam settings (used when CAMERA_SOURCE = "usb")
CAMERA_ID      = 0
FRAME_WIDTH    = 320
FRAME_HEIGHT   = 240
FPS            = 30

# ESP32-CAM stream (used when CAMERA_SOURCE = "esp32")
ESP32_STREAM   = "http://10.240.252.155/stream"
USE_ESP32_CAM  = False   # legacy flag, kept for reference; CAMERA_SOURCE wins

# ── Models ────────────────────────────────────────────────────────────────────
MODEL_PATH     = "models/emotion_raf_mobilenet_finetuned.pth"
KNOWLEDGE_PATH = "data/knowledge.txt"

# ── Robot identity ────────────────────────────────────────────────────────────
ROBOT_NAME = "Luna"
WAKE_WORDS = ["hello", "luna", "hey luna"]  # any of these activates conversation

# ── Noise / false-trigger filtering ───────────────────────────────────────────
# Luna should ignore surrounding chatter and other-language speech that Vosk
# mistranscribes into random, meaningless words. Once a wake word has opened the
# conversation window, each follow-up utterance must pass a meaning gate before
# Luna answers — otherwise she stays silent instead of replying to nonsense.
#
# Two signals decide "is this real speech aimed at Luna":
#   1) confidence — average Vosk word-confidence. Garbled / other-language audio
#      scores low here. Utterances at or above STT_CONFIDENCE_MIN are accepted.
#   2) meaning    — a low-confidence utterance is still (optionally) rescued if
#      it clearly matches the knowledge base, i.e. it has real content Luna
#      knows about. Random words that match nothing are dropped.
#
# ── FINE-TUNING GUIDE (tweak these live, then re-run) ──
#   • Luna replies to background noise / other languages  → RAISE STT_CONFIDENCE_MIN
#     (e.g. 0.62 → 0.72) and/or RAISE STT_MIN_WORDS (2 → 3).
#   • She still answers garbled nonsense that happens to match the KB
#     → set STT_RESCUE_WITH_KNOWLEDGE = False, or RAISE STT_RESCUE_MATCH_MIN.
#   • She ignores YOU / misses real questions → LOWER STT_CONFIDENCE_MIN.
#   • You want her to answer ONLY when addressed by name every time (no bystander
#     pickup at all) → set STT_REQUIRE_WAKE_EACH_TURN = True. She'll still catch
#     the answer to a question SHE just asked (continuity), but otherwise stays
#     silent until she hears a wake word again.
STT_CONFIDENCE_MIN = 0.62   # 0..1 main strictness knob (was 0.55)
STT_MIN_WORDS      = 2      # ignore utterances with fewer real words than this

STT_RESCUE_WITH_KNOWLEDGE = True   # rescue a low-confidence utterance if it
                                   # strongly matches the knowledge base
STT_RESCUE_MATCH_MIN      = 0.70   # how strong that match must be (higher =
                                   # stricter; independent of LOCAL_MATCH_THRESH)
STT_REQUIRE_WAKE_EACH_TURN = False # True = only answer wake-word utterances
                                   # (plus the reply to Luna's own question)

# ── Conversation mode ─────────────────────────────────────────────────────────
CONVO_TIMEOUT      = 10    # seconds of silence before deactivating conversation
POST_SPEAK_DELAY   = 0.45  # settle time after Luna speaks before listening again
DOUBLE_FLUSH       = True  # flush audio queue twice (before and after delay)
MIC_BLOCK_AFTER_SPEAK = 0.4   # echo-guard after speech ends (the mic is already
                              # held shut for the whole of synthesis+playback via
                              # state.tts_active, so this only has to cover the
                              # speaker's own tail/reverb)

# (TTS_EARLY_RETURN_RATIO used to live here, padding the mic block when
# pyttsx3 returned before its audio had finished. pyttsx3 is gone; piper's
# player and macOS `say` both block until playback really ends, so there is
# nothing left to compensate for.)

# Extra defense: if what the mic just heard closely matches what Luna just
# said (within this many seconds), it's almost certainly speaker echo/room
# reverb, not a real new utterance from the user — discard it. Both checks
# must pass (AND, not OR) to stay high-precision — heavily garbled ASR
# echoes won't always clear this, but the mic being held shut for the whole
# of synthesis and playback prevents most of those from being heard in the
# first place; this is just cleanup for trailing echo/reverb.
ECHO_GUARD_WINDOW  = 8.0    # seconds after speaking to check for self-echo
ECHO_RUN_THRESH    = 0.5    # longest contiguous word-run shared with the reply
ECHO_OVERLAP_THRESH = 0.6   # fraction of heard words that appear in the reply

# Spoken when a wake word is heard on its own ("Luna!") with no question attached
WAKE_REPLIES = ["Yes?", "I'm listening!", "Hi! How can I help?"]

# ── Emotion smoothing ─────────────────────────────────────────────────────────
EMOTION_HISTORY_LEN  = 5
SAD_CONFIDENCE       = 55.0

# DeepFace (TensorFlow) is the heaviest thing Luna can load (~1-2 GB RAM and a
# slow import) and only improves "Sad" detection. That RAM is now budgeted to
# the local LLM instead, so DeepFace is OFF by default — the PyTorch emotion
# model covers everything else. Flip to True to bring Sad detection back if
# you're not running the LLM.
ENABLE_DEEPFACE = False

# ── Sleep / wake ──────────────────────────────────────────────────────────────
# Luna sleeps (closed eyes + zzz) after this many seconds with NO face in view,
# and wakes the instant a face reappears. Wake is immediate; only the drop-off
# to sleep is delayed, so a brief detection flicker doesn't nod her off.
SLEEP_AFTER_SECONDS  = 15
SLEEP_AFTER_FRAMES   = RENDER_FPS * SLEEP_AFTER_SECONDS

# ── Servos ────────────────────────────────────────────────────────────────────
ENABLE_SERVOS = True    # False = simulation (Mac/dev), True = drive real servos.
                        # On a Mac the GPIO backends fail to load and it falls
                        # back to simulation automatically, so True is safe there.

# ── Gesture recognition ───────────────────────────────────────────────────────
# False = ignore hand gestures entirely: no thumbs-up / peace / wave / open-palm
# detection or reactions, and the MediaPipe hand tracker isn't even started
# (saves CPU). The detector isn't reliable yet, so it's off for now — flip to
# True to bring gesture reactions back.
ENABLE_GESTURES = False

# Which BCM GPIO each arm's SIGNAL wire is on — set these to match YOUR wiring.
# NOTE: BCM GPIO numbers, NOT physical header pin numbers (BCM17=pin11,
# BCM18=pin12, BCM5=pin29 …). The RIGHT arm is the one that waves and shakes
# hands, so put your best / most-reachable servo there. If unsure which pin an
# arm is on, sweep candidates with:  python3 /tmp/one.py <bcm-gpio>
SERVO_LEFT_PIN  = 17
SERVO_RIGHT_PIN = 18

# Pulse generation backend:
#   "auto"    — use pigpio if its daemon is running, else RPi.GPIO
#   "pigpio"  — force pigpio (needs `sudo pigpiod`)
#   "rpigpio" — force RPi.GPIO software PWM
# pigpio times the pulses with DMA instead of a Python thread. Software PWM
# jitters by a few tens of microseconds whenever the scheduler is busy, and the
# servo physically chases that jitter — that is the "buzzing / twitching a few
# degrees non-stop" symptom. Prefer pigpio on the Pi.
SERVO_BACKEND = "auto"

# Stop pulsing a servo once it has reached its target and settled. A servo with
# no pulse is not actively holding, so it cannot hunt or buzz — this is what
# makes Luna genuinely motionless while she is not talking. Set False only if
# an arm sags under its own weight at rest (then fix it mechanically instead).
SERVO_DETACH_AT_REST = True
SERVO_SETTLE_SECS    = 0.35   # keep driving this long after a move, then release
SERVO_STEP_DELAY     = 0.02   # seconds between interpolation steps

# Global SPEED multiplier for all arm GESTURE moves (wave, handshake, arm
# raises, arms up). 1.0 = normal, 2.0 = twice as fast, 0.5 = half speed/gentler.
# This scales every gesture at once; the per-gesture *_SECS knobs below still let
# you fine-tune individual moves. (Talk-sway speed is set by TALK_SWAY_PERIOD.)
SERVO_SPEED = 1.0

# Pulse width range for the full 0-180° sweep. 500-2500 µs is the usual
# SG90/MG90 range and matches the 2.5-12.5% duty cycle in docs/SERVO_SETUP.md.
# Narrow this (e.g. 600-2400) if an arm strains against its end stop.
SERVO_MIN_PULSE_US = 500
SERVO_MAX_PULSE_US = 2500

# Arm positions, in degrees
ARM_REST_ANGLE = 90    # arms down at the sides — the motionless default
ARM_UP_ANGLE   = 160   # arm raised (wave, celebrate)
ARM_DOWN_ANGLE = 30    # arm lowered below rest

# How long a plain arm raise/lower takes (arm_up, arm_down, arms_up gestures).
# Smaller = snappier, larger = slower/gentler.
ARM_MOVE_SECS = 0.45

# Talking: a small synchronized sway of both arms, and nothing else.
# Raise TALK_SWAY_DEGREES for more animated speech, lower it for subtler.
TALK_SWAY_DEGREES = 8      # peak degrees either side of rest
TALK_SWAY_PERIOD  = 1.6    # seconds for one full up-down-up sway

# Wave: right arm up, small wiggle at the top, back down
WAVE_WIGGLES        = 3      # number of wiggles at the top
WAVE_WIGGLE_DEGREES = 14     # how far each wiggle travels
WAVE_RAISE_SECS     = 0.5    # time to raise/lower the arm
WAVE_WIGGLE_SECS    = 0.18   # time per wiggle (smaller = faster wiggle)

# Handshake: right arm swings out perpendicular to the body to meet the
# person, holds while they take it, then shakes gently up and down.
# Tune HANDSHAKE_ANGLE to whatever "arm horizontal, pointing at the visitor"
# is on your build (between ARM_REST_ANGLE and ARM_UP_ANGLE).
HANDSHAKE_ANGLE      = 125
HANDSHAKE_HOLD_SECS  = 2.0   # pause before shaking, so a hand can be offered
HANDSHAKE_SHAKES     = 3
HANDSHAKE_DEGREES    = 12    # travel either side during the shake
HANDSHAKE_RAISE_SECS = 0.6   # time to swing the arm out / back
HANDSHAKE_SHAKE_SECS = 0.16  # time per shake (smaller = faster/firmer shake)

# Luna has no head servo, so there is no nod. Left here as documentation of a
# deliberate removal: servo.nod() is now a no-op.
SERVO_NOD_ON_REPLY = False

# ── Local LLM (fully offline — replaces Groq) ─────────────────────────────────
# Luna's generative brain is a llama.cpp model running ON the Pi. Nothing ever
# leaves the device and there are no tokens, keys or daily limits — she runs
# indefinitely with no internet. The model is a single .gguf file; swap
# LLM_MODEL_PATH for any other GGUF (bigger = smarter but slower) and restart.
# Qwen2.5-1.5B-Instruct Q4_K_M (~1.0 GB) is the shipped default: good short
# conversational answers at ~8-12 tokens/s on the Pi 5 CPU.
# scripts/install_pi.sh downloads it once; after that Luna is offline forever.
LLM_ENABLED     = True
LLM_MODEL_PATH  = "models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf"
LLM_CTX         = 2048   # context window; 2048 keeps prefill fast on the Pi
LLM_THREADS     = TORCH_THREADS
LLM_WARMUP      = True   # load the model in the background at boot (like TTS)

# Instead of stuffing the WHOLE knowledge base into every prompt (slow prefill
# on the Pi), only the LLM_KB_TOP_K entries most similar to the question are
# included. The sentence-transformer that already powers offline matching does
# the retrieval, so this costs nothing extra.
LLM_KB_TOP_K    = 6
LLM_KB_MIN_SIM  = 0.30   # retrieved entries below this similarity are dropped

# ── LLM backend: Ollama (default) or in-process llama.cpp ─────────────────────
# "ollama"   — talk to a local Ollama server (http://127.0.0.1:11434). Still
#              fully offline: Ollama runs ON the Pi as a service, models are
#              pulled once at install (scripts/install_pi.sh). Swapping models is
#              `ollama pull <tag>` + editing OLLAMA_BASE_MODEL; the custom "luna"
#              model (persona + knowledge baked in) is rebuilt with
#              scripts/make_luna_model.py. Benchmarked with scripts/bench_llm.py.
# "llamacpp" — the V2 path: one .gguf via llama-cpp-python (LLM_MODEL_PATH).
# Either way brain.py sees the same llm_local.stream()/ask() contract; if the
# chosen backend is unavailable Luna degrades to knowledge-base-only answers.
LLM_BACKEND       = "ollama"
OLLAMA_HOST       = "http://127.0.0.1:11434"
OLLAMA_MODEL      = "luna"          # the model brain.py talks to (built by
                                    # scripts/make_luna_model.py from the base)
OLLAMA_BASE_MODEL = "llama3.2:3b"   # base for "luna" — benchmark winner: best
                                    # knowledge + scope scores, zero invented
                                    # facts (docs/llm_benchmark.md). Faster,
                                    # weaker option: "qwen2.5:1.5b".
OLLAMA_NUM_CTX    = 4096            # context window for the Ollama model
OLLAMA_NUM_THREAD = TORCH_THREADS   # CPU threads Ollama may use on the Pi
OLLAMA_KEEP_ALIVE = -1              # -1 = keep the model resident forever
                                    # (never pay the load again mid-conversation)
OLLAMA_TIMEOUT    = 90.0            # seconds before a generation is abandoned
OLLAMA_BAKE_KNOWLEDGE = False       # make_luna_model.py puts the WHOLE
                                    # knowledge.txt in the model's SYSTEM
                                    # prompt (Ollama caches that static prefix,
                                    # so it is prefilled once per load, not per
                                    # question). RAG-lite retrieval still adds
                                    # the top-K entries per turn. OFF: with
                                    # llama3.2:3b retrieval-only scored higher
                                    # (0.97 vs 0.91 — baked made it refuse
                                    # general questions) and keeps the Pi's
                                    # prompt short.

# ── Audio preprocessing (before Vosk) ─────────────────────────────────────────
# Cleans each mic block before it reaches the recogniser (audio_preprocess.py):
#   • high-pass — removes DC offset and the 50 Hz mains hum the dongle picks
#     up without an earth ground (see the gotchas in docs/PROJECT_PIPELINE.md); the
#     hum otherwise clips the ADC and kills recognition.
#   • normalise — brings quiet speech (collar mic, soft talker) up to a
#     consistent level so Vosk sees the same loudness every time.
#   • noise gate — blocks whose RMS sits below AUDIO_GATE_DB are silenced so
#     room noise never becomes garbage words.
AUDIO_PREPROCESS     = True
AUDIO_HIGHPASS_HZ    = 100.0   # cutoff; speech starts ~85 Hz, hum is 50/100 Hz
AUDIO_NORMALIZE      = True
AUDIO_TARGET_PEAK    = 0.85    # fraction of full scale the loudest sample is
                               # brought to (never amplify past this)
AUDIO_MAX_GAIN       = 8.0     # never boost more than this (x) — avoids
                               # blowing up silence into noise
AUDIO_GATE_DB        = -50.0   # blocks quieter than this (dBFS RMS) are muted;
                               # None disables the gate

# ── Web RAG (only when online, only if a site is configured) ─────────────────
# Point WEB_RAG_URL at a website (e.g. the department pages) and, whenever the
# Pi has internet, web_rag.py crawls it (bounded), splits it into chunks,
# embeds them with the same sentence-transformer that powers the knowledge
# base, and CACHES everything under WEB_RAG_CACHE_DIR — so the site's content
# keeps answering questions offline afterwards, and is only re-fetched when
# online and stale. Retrieved chunks join the KB facts in the prompt. An empty
# WEB_RAG_URL disables all of it (Luna's offline-forever default).
WEB_RAG_ENABLED        = True
WEB_RAG_URL            = ""            # e.g. "https://tkmce.ac.in/"
WEB_RAG_MAX_PAGES      = 40            # crawl budget (same host only)
WEB_RAG_MAX_DEPTH      = 2             # link depth from WEB_RAG_URL
WEB_RAG_CHUNK_WORDS    = 120           # words per chunk
WEB_RAG_TOP_K          = 3             # chunks added to a prompt
WEB_RAG_MIN_SIM        = 0.35          # below this a chunk is noise, not context
WEB_RAG_CACHE_DIR      = "data/rag_cache"
WEB_RAG_REFRESH_HOURS  = 24            # re-crawl when online and older than this
WEB_RAG_FETCH_TIMEOUT  = 8.0           # seconds per page
WEB_RAG_ONLINE_TIMEOUT = 1.5           # seconds for the "are we online?" probe

# ── Wiki RAG (live, only when online, no cache) ───────────────────────────────
# Unlike web_rag (one trusted, pre-indexed site), wiki_rag.py does a live
# Wikipedia search + summary fetch per question — only for genuine general
# knowledge, and only when the knowledge base found nothing relevant at all
# (brain.py never calls it otherwise). See wiki_rag.py's module docstring for
# why a campus-flavored question never reaches it even then.
WIKI_RAG_ENABLED = True
WIKI_RAG_TIMEOUT = 5.0        # seconds per request (search + summary, two calls)
WIKI_RAG_LANG    = "en"       # Wikipedia language edition (subdomain)

# ── Audio devices ─────────────────────────────────────────────────────────────
# Capture through PIPEWIRE, not the raw USB device. Direct-hw capture on this
# dongle exclusively locks the card, so PipeWire/paplay can't play Luna's reply
# out the same dongle (silent reply). Going through the "pipewire" PortAudio
# device (needs the pipewire-alsa package) lets PipeWire share the dongle for
# capture + playback at once. It's also a stable NAME, immune to the index
# reshuffling that put the mic on the webcam's slot. Fallback: "USB PnP Sound
# Device" (direct hw, but then output must not use the dongle).
AUDIO_INPUT_DEVICE  = None if IS_MAC else "pipewire"   # Mac: system default mic
AUDIO_OUTPUT_DEVICE = None

# ── STT ───────────────────────────────────────────────────────────────────────
# en-in = English (India) — trained on Indian-accented English. Benchmarked on
# the Pi 5 (scripts/bench_vosk.py): the full en-in-0.5 model transcribed a test
# sentence markedly better than the small en-in-0.4 / en-us-0.15 models, and
# still runs comfortably in real time (RTF ~0.4, finalization ~0.07 s). Its only
# real cost is RAM (~1.7 GB resident vs ~0.17 GB), which the 8 GB Pi affords.
# Drop back to "vosk-model-small-en-in-0.4" if RAM ever gets tight.
VOSK_MODEL_PATH = "vosk-model-en-in-0.5"

# None = auto-detect the device's native sample rate (recommended).
# Set to a number (e.g. 48000, 44100, 32000, 16000) only if auto-detect fails.
# Find your mic's rate: python3 -c "import sounddevice as sd; print(sd.query_devices())"
MIC_SAMPLE_RATE = None if IS_MAC else 48000   # Mac: device default. Pi: USB PnP dongle only opens at 44100/48000 via
                          # PortAudio (not 16k/32k); code resamples to 16k for Vosk

VOSK_SAMPLE_RATE = 16000    # vosk always needs 16000 — do not change

# ── TTS ───────────────────────────────────────────────────────────────────────

TTS_RATE = 155

# Mac mode ────────────────────────────────────────────────────────────────────
# Piper + paplay is a Linux/Pi setup — neither exists on macOS, so on a Mac dev
# box Luna routes speech through macOS's built-in `say` (plays on the default
# Mac output and blocks until finished, so mic echo protection is unchanged).
# Follows RUN_MODE (see the top of this file): True in mac mode (Darwin or
# forced), False in pi mode. A hardcoded True here previously got synced to
# the Pi and made it try `say`, which doesn't exist there → silent replies.
MAC_MODE = IS_MAC

# Voice for `say`. None = system default voice.
# List them with:  say -v '?'
MAC_VOICE = None

# Words per minute for `say`. macOS default is ~175; 155 matches TTS_RATE.
MAC_TTS_RATE = TTS_RATE

# Piper voice engine
PIPER_PATH = "/home/luna/piper/piper/piper"

PIPER_MODEL = (
    "/home/luna/piper/voices/"
    "en_US-lessac-medium.onnx"
)

# Higher = slower speech
# 1.0 = default
# 1.3 = natural assistant pace
PIPER_LENGTH_SCALE = 1.3

TTS_PLAYER = "paplay"

# Stream piper audio straight into the player so speech starts on the first
# audio chunk (instead of waiting for the whole file to synthesize — the old
# ~5s gap). If the streaming pipeline fails once on this machine, Luna caches
# that and permanently uses the safe synth-to-file path — it never retries the
# streaming path per reply, so there is no repeated overhead or error spam.
# Set False to force the plain file path (e.g. if streaming plays silence).
PIPER_STREAMING = True

# Keep ONE piper process alive for the whole session instead of spawning a new
# one per reply. Loading en_US-lessac-medium.onnx takes several seconds on the
# Pi, and the old code paid that cost on *every* answer — that is the bulk of
# the ~11 s first reply and the 3-4 s gap on later ones. With a persistent
# process the model is loaded once at startup (see TTS_WARMUP) and a reply only
# costs actual synthesis. Falls back automatically if the piper build in use
# doesn't support the persistent mode.
PIPER_PERSISTENT = True

# Synthesize a throwaway phrase in the background at boot so the model is
# already resident when the first real question arrives.
TTS_WARMUP = True

# Speech is synthesized and played one chunk at a time, so Luna starts talking
# after the FIRST sentence is ready rather than the whole reply. Chunks shorter
# than the minimum are merged with the next one (a lone "Sure!" isn't worth its
# own player process); anything longer than the maximum is split at a comma.
TTS_CHUNK_MIN_CHARS = 60
TTS_CHUNK_MAX_CHARS = 220

# Seconds to wait for piper to synthesize one chunk before giving up on the
# persistent process and falling back. The first call also covers model load.
TTS_SYNTH_TIMEOUT       = 20.0
TTS_FIRST_SYNTH_TIMEOUT = 60.0

# ── Brain ─────────────────────────────────────────────────────────────────────
LLM_MAX_TOKENS      = 150   # max tokens per response (keep short for speech)
LLM_TEMPERATURE     = 0.7   # creativity (0.0 = factual, 1.0 = creative)
LLM_MAX_HISTORY     = 6     # max conversation history turns kept in the prompt
LOCAL_MATCH_THRESH  = 0.55  # min similarity score for local knowledge match

# ── Offline-first answering ───────────────────────────────────────────────────
# Before waking the LLM, Luna checks the knowledge base with the sentence-
# transformer. If a question matches a known entry at/above this score she
# answers straight from the KB — instant, and exactly the wording the KB
# author wrote. Only questions the KB isn't confident about go to the LLM.
# Higher = stricter (fewer KB answers, more LLM); lower = more KB answers (risk
# of giving a KB answer to a loosely-related question). Set 1.1 to disable and
# always use the LLM first.
ANSWER_OFFLINE_FIRST = True
LOCAL_ANSWER_THRESH  = 0.70

# Stream the reply out of the LLM token-by-token instead of waiting for the
# whole completion. Combined with chunked TTS above, Luna starts speaking the
# first sentence while the model is still writing the rest of the answer.
LLM_STREAMING = True

# ── Gestures / expressions ────────────────────────────────────────────────────
GESTURE_REACT_COOLDOWN = 8.0   # seconds between spoken reactions to a gesture
FACE_OVERRIDE_SECS     = 4.0   # how long gesture-triggered faces (excited/love) last

# ── Face style ────────────────────────────────────────────────────────────────
# 1 = Luna classic (purple, soft rounded)
# 2 = Robo (cyan, sharp corners, equalizer-bar mouth — NIMO/modern-robot look)
# 3 = Aurora (Face 3 — EMO/Vector-style big gradient eyes, eyes-only at rest,
#     squash-and-stretch blinks, catchlights. The product face.)
# Press 1 / 2 / 3 on the face window to switch live; this sets the startup default.
FACE_STYLE = 3

# ── Display fit ───────────────────────────────────────────────────────────────
# The face is drawn on a fixed 1400x800 logical canvas. With pygame's SCALED
# flag it is automatically scaled to the REAL display resolution while keeping
# its aspect ratio (letterboxed with black bars if the screen ratio differs), so
# the face fits and stays centred on any screen — e.g. the Pi's 800x480 DSI panel
# — instead of being cropped by the oversized window.
# DISPLAY_FULLSCREEN = True fills the whole screen (kiosk mode, best on the Pi).
# Mode-aware: kiosk on the Pi, a normal window on the Mac. Replace with a plain
# True/False to force it.
DISPLAY_FULLSCREEN = IS_PI

# ══════════════════════════════════════════════════════════════════════════════
# Ported from the dev/Luna branch (yesterday's animation + gesture + continuity
# work) onto this Vosk base. See UPDATES.txt. These drive robot_face.py's newer
# animation, gesture_module.py's tuned hand detection, and the follow-up
# conversation-continuity behaviour in brain.py / speech_to_text.py.
# ══════════════════════════════════════════════════════════════════════════════

# ── Conversation continuity ───────────────────────────────────────────────────
# After Luna asks a follow-up question she waits longer before the conversation
# window times out, so the person has time to think without losing the thread.
CONVO_TIMEOUT_AFTER_QUESTION = 20   # seconds (vs CONVO_TIMEOUT for normal turns)

# ── Hand-gesture detection (MediaPipe) ────────────────────────────────────────
GESTURE_STABLE_FRAMES       = 3    # frames a gesture must persist to register
GESTURE_REACT_STABLE_FRAMES = 6    # stricter stability before a spoken reaction
THUMB_RISE_RATIO        = 0.50     # thumb tip above its own MCP, / hand span
THUMB_ABOVE_INDEX_RATIO = 0.35     # thumb tip above the index knuckle, / span
HAND_DETECT_CONF   = 0.75          # MediaPipe min_detection_confidence
HAND_TRACK_CONF    = 0.65          # MediaPipe min_tracking_confidence
HAND_MIN_HANDEDNESS = 0.80         # min left/right classification score
HAND_MIN_SPAN      = 0.13          # hand bbox must cover this fraction of frame
HAND_MODEL_COMPLEXITY = 1
GESTURE_DEBUG = False

# ── Face animation (robot_face.py) ────────────────────────────────────────────
MOUTH_FILLED_TALK = True   # False restores the old four-shape jaw
MOUTH_MIN_DEPTH   = 8      # bulge depth (px) at silence — the "C"
MOUTH_MAX_DEPTH   = 104    # bulge depth at full volume — the "D"
MOUTH_CLOSED_H    = 14     # thickness of the closed-mouth bar
ARC_EYES_HAPPY = True
WAKE_ARMS = True

# Conversation "listening" indicator light
CONVO_LIGHT_COL        = (255, 45, 45)   # red
CONVO_LIGHT_RADIUS     = 9               # core radius in px
CONVO_LIGHT_MARGIN     = 34              # distance from the top-left corner
CONVO_LIGHT_LABEL      = True            # show the word "LISTENING" beside it
CONVO_LIGHT_LABEL_SIZE = 20

# Ambient floating motes + "thinking" dots
AMBIENT_MOTES      = True
AMBIENT_MOTE_COUNT = 18
THINKING_DOTS = True

# ══════════════════════════════════════════════════════════════════════════════
# Luna V2 — fully-offline product features (2026-09).
# See docs/superpowers/specs/2026-09-07-luna-v2-design.md for the design.
# ══════════════════════════════════════════════════════════════════════════════

# ── Face recognition (offline — YuNet detect + SFace embed) ──────────────────
# Luna remembers faces you enroll by voice ("Luna, remember me — my name is
# Alice"), greets people by name, and logs sightings. Everything stays on the
# device: embeddings in data/faces/, the log in data/face_log.jsonl.
ENABLE_FACE_RECOGNITION = True
FACE_DETECTOR      = "yunet"   # "yunet" (better, gives landmarks) | "haar" (fallback)
YUNET_MODEL_PATH   = "models/vision/face_detection_yunet_2023mar.onnx"
SFACE_MODEL_PATH   = "models/vision/face_recognition_sface_2021dec.onnx"
FACE_DETECT_CONF   = 0.80      # YuNet detection score threshold
FACE_MATCH_THRESH  = 0.36      # SFace cosine similarity — the paper's 0.363,
                               # rounded. RAISE if strangers get named, LOWER
                               # if enrolled people aren't recognised.
RECOGNIZE_EVERY    = 3         # run the embedder every Nth vision frame
FACE_DB_DIR        = "data/faces"
FACE_LOG_PATH      = "data/face_log.jsonl"
FACE_LOG_ENABLED   = True
FACE_LOG_COOLDOWN  = 60.0      # min seconds between log entries per person

# Voice enrollment: how many embedding samples to average, and how long the
# person has to hold still-ish in front of the camera before Luna gives up.
ENROLL_FRAMES      = 8
ENROLL_TIMEOUT     = 12.0

# ── Scene context (offline "what do you see?") ───────────────────────────────
# When a question sounds like it's about what the camera sees, Luna assembles a
# description of the live scene (people, names, emotions, motion, brightness —
# plus detected objects) and answers through the local LLM. Object detection is
# YOLOv8n via OpenCV's dnn, loaded lazily and run ON DEMAND only, so it costs
# nothing while idle.
ENABLE_SCENE_CONTEXT    = True
ENABLE_OBJECT_DETECTION = True
YOLO_MODEL_PATH   = "models/vision/yolov8n.onnx"
YOLO_INPUT_SIZE   = 416        # must match the exported ONNX imgsz
YOLO_CONF         = 0.35
YOLO_NMS          = 0.45
# Phrases that route a question through the scene context. Substring match on
# the lowercased utterance.
VISION_QUERY_PHRASES = (
    "what do you see", "what can you see", "what are you seeing",
    "who am i", "who do you see", "do you see", "can you see",
    "how do i look", "how am i looking", "what am i holding",
    "what is in front of you", "what's in front of you",
    "describe what you see", "look at me", "what am i wearing",
)

# ── Addressed-speech scoring ──────────────────────────────────────────────────
# Inside a conversation window, "was that meant for Luna?" is decided by a
# weighted score instead of the old confidence-only gate. Signals that a human
# uses to know they're being spoken to: being named, being asked a question,
# being looked at, the topic being something Luna knows, and having just asked
# a question herself. Each weight is how much that signal adds; the utterance
# is accepted when the total reaches ADDR_THRESHOLD.
# Set ADDRESSED_SCORING = False to fall back to the plain confidence gate.
ADDRESSED_SCORING   = True
ADDR_THRESHOLD      = 1.35  # was 1.0: with ADDR_W_CONF=1.2 a crisp sentence
                            # passed on confidence ALONE, so clear chatter
                            # between two people got answered. Now at least one
                            # addressing signal (name / question / "you" /
                            # topic / face / loudness / continuity) is needed.
ADDR_REQUIRE_SIGNAL = True  # hard rule on top of the score: confidence alone
                            # can never pass, whatever the weights say
ADDR_W_CONF         = 1.2   # × Vosk confidence (0..1) — the base signal
ADDR_W_LOUDNESS     = 0.45  # × how close the talker sounds (0..1, from the
                            # utterance level measured by audio_preprocess):
                            # people talking TO Luna stand at her mic; room
                            # chatter is quieter. LOUDNESS_NEAR_DB / _FAR_DB
                            # map dBFS RMS onto 0..1.
LOUDNESS_NEAR_DB    = -22.0
LOUDNESS_FAR_DB     = -42.0
ADDR_FACE_OR_NAME   = False # True = inside a conversation window ALSO require
                            # a visible face, Luna's name, or a pending answer
                            # (strictest: nothing said off-camera is answered)
ADDR_W_NAME         = 0.8   # Luna's name / a wake word appears anywhere
ADDR_W_QUESTION     = 0.35  # utterance is question-shaped (what/where/can you…)
ADDR_W_SECOND_PERSON = 0.25 # "you / your" — they're talking TO someone
ADDR_W_KB           = 0.5   # × KB similarity — it's about something Luna knows
ADDR_W_FACE         = 0.3   # a face is looking at Luna right now
ADDR_W_CONTINUITY   = 0.7   # Luna just asked a question — this is likely the answer

# ── Voice commands ────────────────────────────────────────────────────────────
# Handled before the brain: instant, offline, no LLM involved. All phrase
# tables are plain substring/word matches on the lowercased utterance.
ENABLE_VOICE_COMMANDS = True
VOLUME_STEP           = 10     # percent per "volume up"/"volume down"
VOLUME_MIN, VOLUME_MAX = 20, 100

# Date & time: answered straight from the system clock, no LLM, no network.
# persona.py still tells the LLM to decline these — that's the fallback for
# any phrasing this command's regex doesn't catch, so an edge case degrades to
# an honest "outside what I can help with" instead of the LLM guessing.
ENABLE_DATETIME_COMMAND = True

# Weather: the ONE voice command that needs the internet (a live Open-Meteo
# call, free, no API key). Hardcoded to the campus location rather than doing
# a geocoding round-trip; update the coordinates if Luna ever moves campuses.
# On any failure (offline, API down) she says so honestly rather than
# guessing — see commands.py's _weather_command.
ENABLE_WEATHER_COMMAND = True
WEATHER_LOCATION_NAME  = "Kollam"
WEATHER_LAT, WEATHER_LON = 8.8932, 76.6141
WEATHER_TIMEOUT        = 5.0

# ── Proactive behaviour (EMO-style liveliness) ────────────────────────────────
# Luna acts on what she sees, without being spoken to first: greets people she
# recognises, reacts to a smile, checks in on a sad face. Everything is
# cooldown-limited so she is charming, not needy, and she never interrupts an
# active conversation or her own speech.
ENABLE_PROACTIVE          = True
PROACTIVE_GREET_RECOGNIZED = True   # "Hi Alice!" + wave + name banner
GREET_COOLDOWN_SECS       = 120.0   # per person: a quick step away and back
                                    # gets no second greeting
PROACTIVE_SMILE_REACT     = True    # smile back + a happy word
SMILE_REACT_COOLDOWN      = 120.0
PROACTIVE_SAD_REACT       = True    # gentle check-in on a sad face
SAD_REACT_COOLDOWN        = 240.0
PROACTIVE_POLL_SECS       = 0.5     # how often the proactive loop looks around
PROACTIVE_STABLE_SECS     = 2.0     # emotion must persist this long to react

# What she actually says. {name} is filled in for greetings.
GREET_PHRASES = (
    "Hi {name}! Good to see you again!",
    "Hey {name}! How's it going?",
    "Oh, hi {name}! I was hoping you'd come by.",
)
SMILE_PHRASES = (
    "I love that smile!",
    "You look happy today — it's contagious!",
    "Someone's in a good mood!",
)
SAD_PHRASES = (
    "Hey... you look a little down. Want to talk about it?",
    "Everything okay? I'm here if you want to chat.",
)

# ── Touch interactions (Face 3) ───────────────────────────────────────────────
# The DSI panel is a touchscreen — tapping Luna's eyes boops her (they pop and
# cross for a moment), tapping the mouth area gets a giggle smile. Works with a
# mouse too, for desktop testing.
ENABLE_TOUCH_REACTIONS = True
BOOP_SECONDS           = 1.4

# ── Greeting name banner ──────────────────────────────────────────────────────
# "Hi <name>!" drawn on screen when a recognised face is greeted (the sketch's
# "Hello 'name' on screen"). Also used for enrollment confirmations.
GREET_BANNER       = True
GREET_BANNER_SECS  = 4.0
GREET_BANNER_SIZE  = 44

# ── The recognition moment ────────────────────────────────────────────────────
# What happens in the second Luna realises she KNOWS the face in front of her.
# A dedicated expression ("recognised": eyes pop, sparkle burst, a hop) plus a
# quick double-take of the eyes, the name banner sliding up, and — if the
# greeting ends in a question — the conversation window opens so the person
# can answer without saying the wake word. Someone who left only a moment ago
# gets a "welcome back" instead of a full greeting.
RECOG_ANIMATION        = True
RECOG_EXPRESSION_SECS  = 4.0    # how long the "recognised" face is held
RECOG_DOUBLE_TAKE      = True   # quick left-right eye dart before locking on
RECOG_BANNER_SLIDE     = True   # banner slides up and pops (vs just appearing)
RECOG_GREET_OPENS_CONVERSATION = True
WELCOME_BACK_SECS      = 900    # left less than this long ago → "welcome back"
                                # (must exceed GREET_COOLDOWN_SECS)
WELCOME_BACK_PHRASES = (
    "Welcome back, {name}!",
    "Back again, {name}? Nice to see you.",
    "Hey {name}, that was quick! Did you forget something?",
)

# ── Face registration options ─────────────────────────────────────────────────
# Three ways to register a face, all ending in the same face_store entry:
#   voice  — "remember me, my name is Alice" (existing), now GUIDED: Luna asks
#            for a couple of head poses so the stored embedding is the average
#            of straight/left/right views (recognition survives angles), and
#            confirms the name she heard before storing it (Vosk mishears
#            names more than anything else).
#   touch  — press and hold the centre of the face for TOUCH_ENROLL_HOLD_SECS;
#            Luna asks for the name by voice.
#   photos — scripts/enroll_faces.py data/enroll/<Name>/*.jpg registers people
#            from photos with no camera (pre-load the department staff).
ENROLL_GUIDED_POSES    = True
ENROLL_POSES = (
    "Look straight at me.",
    "Now turn your head a little to your left.",
    "And a little to your right.",
)
ENROLL_FRAMES_PER_POSE = 4      # samples collected while each pose is spoken
ENROLL_MIN_FRAMES      = 4      # store if at least this many samples were good
ENROLL_CONFIRM_NAME    = True   # "Did you say Alice?" → yes / no
TOUCH_ENROLL           = True
TOUCH_ENROLL_HOLD_SECS = 1.2
ENROLL_PHOTO_DIR       = "data/enroll"

# ── Scene memory (contextual, continuous conversation) ───────────────────────
# A rolling memory of what the camera has seen recently — who arrived, who
# left, moods, objects noticed — summarised in one or two sentences and added
# to EVERY LLM turn, so Luna can say "you looked tired earlier" or "Bob was
# here a few minutes ago" without being asked "what do you see". Also drives
# conversation starters: a known person who has been standing there quietly
# for SCENE_LINGER_SECS gets a friendly opener (once per visit).
ENABLE_SCENE_MEMORY     = True
SCENE_MEMORY_EVENTS     = 12     # recent events kept
SCENE_MEMORY_MAX_AGE    = 1800   # seconds — older events are forgotten
SCENE_MEMORY_IN_PROMPT  = True   # add the summary to LLM turns
SCENE_LINGER_SECS       = 8.0
PROACTIVE_CONVERSATION_STARTERS = True
CONVERSATION_STARTERS = (
    "So {name}, what brings you to the department today?",
    "{name}, anything I can help you find today?",
    "How is your day going so far, {name}?",
)

# ── Conversation log ──────────────────────────────────────────────────────────
# Every utterance Luna hears and everything she answers, one JSON line per
# turn in data/logs/conversation_YYYY-MM-DD.jsonl: raw + cleaned text, Vosk
# confidence, loudness, the addressing-score breakdown and the decision,
# recognised person, which path answered (command / vision / kb / llm /
# fallback), the knowledge entries retrieved, the exact reply, timings, and
# any reward that arrives afterwards. IGNORED utterances are logged too, with
# the reason — a question she wrongly ignored is as fixable as a wrong answer.
# Read a day back with scripts/review_log.py. This is also the training data
# for learning.py. Files are pruned after CONVO_LOG_RETENTION_DAYS.
CONVO_LOG_ENABLED        = True
CONVO_LOG_DIR            = "data/logs"
CONVO_LOG_RETENTION_DAYS = 365 if IS_MAC else 45

# ── Learning from daily life (learning.py) ────────────────────────────────────
# Not weight training — a small on-device learning loop that runs everywhere:
#   rewards   — explicit ("good answer", "that's wrong", a thumbs-up) and
#               implicit (they kept talking / said thanks / repeated the same
#               question / walked away) signals, scored against the previous
#               turn.
#   manner    — a contextual bandit (Thompson sampling) picks a reply STYLE
#               (length × tone × follow-up) per context (known person or not,
#               question kind, time of day) and learns from the rewards which
#               style people respond to. State: data/learning/bandit.json.
#   people    — per-person profiles (visits, topics, preferred style) used
#               in the prompt when that person is recognised.
#   exemplars — replies that earned a high reward are kept and retrieved as
#               few-shot examples for similar questions (how a small model
#               picks up style without touching its weights).
#   facts     — "remember that…" / corrections go to a PENDING file and only
#               enter knowledge.txt after approval (scripts/approve_facts.py).
#   diary     — a nightly one-paragraph summary of the day's log
#               (data/diary/), retrievable like the website index.
# Bootstrap on the Mac with scripts/train_learning.py (replays logs and can
# run simulated visitor sessions), then deploy ONCE with
# `scripts/deploy.sh --go --seed-learning`; afterwards the Pi keeps learning
# and deploys never overwrite data/learning/.
LEARNING_ENABLED           = True
LEARNING_DIR               = "data/learning"
LEARNING_STYLE_ENABLED     = True    # let the bandit steer reply style
LEARNING_EXEMPLARS_K       = 2       # liked replies shown as examples per turn
LEARNING_EXEMPLAR_MIN_REWARD = 0.5   # reward needed to keep a reply as example
LEARNING_ENGAGEMENT_WINDOW = 20.0    # s: a follow-up within this = engaged (+)
LEARNING_REPEAT_SIM        = 0.80    # same question again (similarity) = (−)
LEARNING_REWARD_EXPLICIT   = 1.0     # "good answer" / "that's wrong"
LEARNING_REWARD_THUMBS     = 0.6     # thumbs-up gesture right after a reply
LEARNING_REWARD_THANKS     = 0.5     # thanks / laughter / compliment
LEARNING_REWARD_ENGAGED    = 0.25    # kept the conversation going
LEARNING_REWARD_REPEAT     = -0.5    # asked the same thing again
LEARNING_REWARD_CORRECTION = -1.0    # "no, that's not right…"
LEARNING_REWARD_ABANDON    = -0.2    # walked away mid-answer / no reply
LEARNING_MIN_TURNS_TO_STEER = 8      # before this many rewarded turns per
                                     # context, styles are explored uniformly
PENDING_FACTS_PATH         = "data/learning/pending_facts.txt"
DIARY_ENABLED              = True
DIARY_DIR                  = "data/diary"
DIARY_WRITE_HOUR           = 22      # first idle moment after this hour

# ── Vision performance ────────────────────────────────────────────────────────
# Detect faces on a downscaled copy of the frame (boxes and landmarks are
# scaled back up for SFace alignment and tracking). 0.5 = quarter the pixels
# for YuNet, the single biggest cost in the vision loop on the Pi.
VISION_DETECT_SCALE = 0.5
