# Luna V2 — Fully-Offline Product Design

**Date:** 2026-09-07
**Base:** `/Users/michaelharoldsony/Developer/Luna` (Vosk lineage — the go-forward base)
**Target:** Raspberry Pi 5 (8 GB), 800×480 DSI touchscreen, Pi Camera v1 (ov5647), USB audio dongle, 2 arm servos.
**Rollback:** `Developer/Luna_rollback_20260907_2247.tar.gz`

## Goal

Turn Luna into a polished, product-grade desk robot in the spirit of Emo / Nimo /
Vector, per the Luna V2 sketches:

1. **Fully offline, forever.** No online service anywhere in the runtime path.
   The Groq LLM is replaced with a local llama.cpp model.
2. **Face 3** — a new, professional-and-cute face design plus richer animation
   (squash-and-stretch, saccades, eased blinks, curious head tilt, touch boops).
3. **Facial recognition** — stored faces, live voice-driven enrollment
   ("my name is X"), greeting by name with an on-screen name tag, recognition log.
4. **Camera context understanding** — Luna can answer "what do you see?",
   "who am I?", "how do I look?" from live vision state (+ on-demand object
   detection), routed through the local LLM.
5. **Addressed-speech detection** — better separation of "random chatter in the
   room" vs "talking to Luna", as a configurable scoring gate.
6. **Interactive sessions** — proactive greetings for recognized people, smile /
   sad reactions, voice commands (volume, sleep, face style…), existing
   conversation-window + follow-up continuity kept.
7. **Everything adjustable in `config.py`** — the project's established single
   source of tunables; every new feature gets its own config section and enable
   flag.

## Non-goals (explicitly deferred)

- The admin **website** from the sketches (KB upload, live activity) — separate
  sub-project; the recognition log + hot-reloading knowledge.txt cover the core
  needs meanwhile.
- **Touch/FSR/IMU sensor module** — hardware not attached. Screen-touch
  interactions ARE included (the touchscreen exists).
- Wake-word neural models (openWakeWord) — Vosk text matching already works.
- **DeepFace/TensorFlow stays optional and default-OFF** (`ENABLE_DEEPFACE`):
  ~1–2 GB RAM + slow import, needed only for better "Sad" detection; that RAM
  is re-budgeted to the local LLM.

## Architecture decisions

### D1. Local LLM: llama-cpp-python + Qwen2.5-1.5B-Instruct (Q4_K_M)

- In-process `llama-cpp-python` (no daemon to manage vs Ollama; one venv).
- Model file configurable (`LLM_MODEL_PATH`); default
  `models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf` (~1.0 GB, ~8–12 tok/s on Pi 5
  CPU — fine for 1–3 spoken sentences). Swappable for any GGUF.
- **RAG-lite instead of whole-KB-in-prompt:** embed the query with the existing
  sentence-transformer, put only the top-K KB entries in the system prompt
  (`LLM_KB_TOP_K`). Keeps prompts short → fast prefill on the Pi.
- Offline-first KB answering (existing ≥0.70 direct-match path) is kept — the
  LLM is only consulted below that threshold.
- Streaming synthesis kept: LLM token stream → chunked Piper TTS.
- Graceful degradation: no gguf / import failure → KB-only mode (same as today
  with no key). `groq` dependency removed entirely.

### D2. Face recognition: OpenCV YuNet (detect) + SFace (recognize)

- Both ship as small ONNX models loadable by opencv-contrib 4.8's
  `FaceDetectorYN` / `FaceRecognizerSF` — no dlib build, fast on Pi CPU.
- YuNet **replaces Haar** as the primary detector (better accuracy, gives the
  5 landmarks SFace needs for alignment); Haar remains a config fallback
  (`FACE_DETECTOR = "yunet" | "haar"`).
- `face_store.py`: embeddings + names persisted to `data/faces/` (npz + json);
  cosine matching with `FACE_MATCH_THRESH`; enroll / forget / list APIs.
- Voice enrollment intents ("remember me, my name is Alice", "forget Alice",
  "who am I") handled in `commands.py`, capture N frames over ~2 s.
- Recognition every `RECOGNIZE_EVERY` vision frames; publishes
  `state.person_name` (+ confidence); appends sightings to
  `data/face_log.jsonl` (the sketch's "recognized face logs").

### D3. Scene context: vision-state description + on-demand YOLOv8n

- `scene_context.py` builds a text description from what Luna already knows:
  person present/name/emotion/position, motion, frame brightness.
- Vision-directed questions ("what do you see / who am I / how do I look…",
  configurable trigger phrases) additionally run **YOLOv8n ONNX via cv2.dnn**
  on the current frame (lazy-loaded, on-demand only — no idle CPU cost) and the
  combined description is fed to the local LLM as context to answer naturally.
- The ONNX is produced once at build time (ultralytics export on the Mac,
  shipped in `models/vision/`).

### D4. Face 3 — "Aurora" (style 3, new default)

Emo/Vector-inspired, eyes-carry-everything design on the existing logical
canvas / parts architecture (a third entry in `face_style.STYLES` + new
rendering modes in `face_parts`):

- Large rounded-square **solid-gradient eyes** (ice-mint→teal), no iris ring —
  expression comes from shape, not eyeball parts; dual white catchlights.
- **Squash & stretch** blinks (y squashes, x bulges) with eased two-phase
  timing; idle **micro-saccades**; breathing scale; curious asymmetric eyes
  (`eye_asym` — one eye enlarges on head tilt).
- Minimal mouth: hidden at rest (eyes-only, like Emo/Vector), soft line/smile
  appears while talking or for strong moods.
- New expressions: **curious**, **annoyed**, **amazed** (adds to the sketch's
  list; tilted-head + blink + red-eyes already exist).
- **Touch interactions** (sketch: "popping eyes on touchy eyes"): tapping an
  eye pops/crosses it briefly ("boop"); tapping the mouth area gives a giggle
  smile — config-gated (`ENABLE_TOUCH_REACTIONS`).
- **Name tag greeting**: when a recognized face is greeted, "Hi <name>!" is
  drawn as a soft banner for a few seconds (`GREET_BANNER_*` config).
- Keys 1/2/3 switch styles live; `FACE_STYLE = 3` default.

### D5. Addressed-speech scoring (`addressing.py`)

Pure-logic module (unit-testable on the Mac) replacing the binary
confidence-gate for in-conversation turns with a weighted score:

- signals: Vosk confidence, name/wake-word mention anywhere, question form,
  2nd-person cues ("you/your"), KB match, awaiting-answer continuity,
  face-visible-while-speaking; weights + threshold in config
  (`ADDR_*`). Falls back to the current behaviour when disabled.

### D6. Voice commands (`commands.py`) + proactive behaviour (`proactive.py`)

- Commands checked before the LLM: volume up/down/percent (pactl on Pi),
  sleep / wake, face style switch, enrollment intents, existing password
  shutdown kept. All phrases configurable.
- Proactive (config-gated, per-person cooldowns): greet recognized person by
  name (speech + banner + wave), respond to a detected smile, gentle check-in
  on sad — the sketch's "wait() + if not talking: prompt(smile) → reply".
  Only acts when `state.ready` and not mid-conversation.

### D7. Deployment / install

- `scripts/install_pi.sh`: one-shot Pi setup — apt packages, venv, pip
  requirements (now incl. `llama-cpp-python`, minus `groq`), downloads
  vosk model (if absent), piper + voice, LLM gguf, YuNet/SFace ONNX.
- `scripts/download_models.sh`: model fetch shared by Mac dev + Pi install.
- `deploy.sh`: additionally excludes `models/llm/` (Pi downloads its own gguf).
- requirements pinned; everything downloadable once, runs offline forever.

## Data flow (updated)

```
mic → Vosk STT → wake/commands/addressing gate → brain
brain: KB direct match ≥ thresh → answer
       vision-question → scene_context (state + YOLO) → local LLM → answer
       else → RAG-lite top-K KB → local LLM (llama.cpp) → answer
answer stream → chunked Piper TTS → speaker (mouth via audio_energy)

camera → camera_thread → vision_module (YuNet detect → emotion PT model →
SFace recognize every N frames) → state.persons/person_name/emotion →
face_renderer + proactive (greet by name / smile / sad) + face_log.jsonl
```

## Error handling

Every subsystem keeps the project's existing degrade-don't-crash contract:
missing mic/camera retry loops kept; missing gguf → KB-only; missing ONNX →
recognition off with one clear log line; servo failures never block rendering;
unknown expressions fall back to neutral.

## Testing

- `python -m py_compile` on every file (project convention).
- New pure-logic modules (`addressing`, `face_store` matching,
  expression registry, scene description) get unit tests runnable on the Mac
  (scratchpad venv — the project venv is broken; Pi venv is authoritative).
- Face 3 verified headlessly (SDL dummy driver) with rendered-frame PNGs.
- Final integration test happens on the Pi via `scripts/deploy.sh --go`.

## Judgment calls made autonomously (flag for review)

1. Groq removed outright (not kept behind a flag) — per "no online things".
2. DeepFace default-off to free RAM for the LLM (Sad detection now relies on
   the PyTorch model).
3. Qwen2.5-1.5B-Instruct chosen as default LLM (best quality/speed at ~1 GB);
   path is config, any GGUF works.
4. Face 3 made the startup default.
5. Website/admin UI deferred.
