# Luna 2.0 — Project Pipeline

> A complete, accurate overview of the Luna robot's architecture, runtime and
> deployment. Doubles as a context prompt to paste at the start of a coding
> session ("Read this full context before acting"). For setup steps see the
> README.

Luna is an **offline-first desk robot** running on a **Raspberry Pi 5 (8 GB)**.
She listens, answers questions (knowledge base first, then a **local LLM served
by Ollama**, no cloud, no keys, no quotas), shows an animated face
(**Face 3 "Aurora"**, EMO/Vector-style), **recognises registered people by
name** with a dedicated "I know you!" animation, detects emotion, remembers
what she has seen recently and uses it in conversation, starts conversations
with people she knows, answers **"what do you see?"** from the camera, and
moves two arm servos. She answers **only when spoken to**, never room
chatter. She **learns from daily life**: which reply style people respond to,
who asks about what, which answers were liked, and facts people teach her
(after approval). Built for a college CS department (booth / helpdesk use).

The only network use is the weekly website indexing (`site_index.py`), run by
hand on the Mac and deployed with the code. While Luna runs, nothing is
downloaded: she works with no internet at all.

---

## Hardware

- **Pi 5**, hostname `cse`, user `luna`, project at `/home/luna/Luna`.
  Pi venv is **Python 3.11** at `/home/luna/Luna/venv`.
- **Display:** 800×480 DSI touchscreen (touch drives boops and touch-to-register).
- **Audio:** one USB C-Media "PnP" dongle with a collar **mic** on its input
  and an **aux speaker** on its output.
- **Camera:** Pi Camera v1 (**ov5647**) on the CSI ribbon.
- **Servos:** two hobby servos (arms) on **BCM GPIO17** and **GPIO18**,
  externally powered with a common ground (docs/SERVO_SETUP.md).

## Run modes (config.py `RUN_MODE`)

The SAME config works on both machines. `"auto"` (default) detects the
platform; `"mac"` / `"pi"` force it. Mode drives: TTS (`say` vs Piper+paplay),
camera (USB vs CSI), microphone (system default vs the `pipewire` device),
fullscreen, volume backend, servo simulation, and log retention.

## Runtime stack (Pi)

- **Ollama** runs as a systemd service bound to 127.0.0.1, keeping the model
  resident (`OLLAMA_KEEP_ALIVE=-1`). Installed and configured by
  `scripts/install_pi.sh`.
- Audio via **PipeWire**; capture through the **`pipewire`** PortAudio device
  (needs `pipewire-alsa`), NOT raw ALSA `hw:`, because raw capture locks the
  dongle and playback dies. Playback via **`paplay`**.
- Launched with `venv/bin/python -u main.py` (desktop icon runs
  `~/Desktop/start-luna.sh`).

---

## Module map

**Voice in**
- `speech_to_text.py`: Vosk (`vosk-model-en-in-0.5`), 48 kHz → 16 kHz, wake
  words, conversation window, password shutdown. Measures each utterance's
  loudness and asks `addressing.py` whether it was meant for Luna. Logs
  ignored utterances with the reason.
- `audio_preprocess.py`: per 250 ms block, a 4th-order high-pass (DC + 50/100
  Hz hum), a noise gate, and a slow AGC. The AGC is the big win for quiet
  talkers.
- `addressing.py`: "was that meant for Luna?" A weighted score (confidence,
  name, question shape, "you", KB relevance, visible face, loudness,
  continuity) plus a hard rule that clear speech alone is never enough.
  `explain()` gives the per-signal breakdown printed in the log.

**Thinking**
- `brain.py`: the answer pipeline, in priority order:
  1. `commands.py`: volume, sleep, style, registration, feedback, teaching.
  2. date / time from the clock (calendar / clock card).
  3. vision questions: live scene description (+ on-demand YOLOv8n) → LLM.
  4. KB direct match ≥ `LOCAL_ANSWER_THRESH` → the verbatim KB answer.
  5. Data Fetch (`fetch.py`, [Fetch Algo.md](Fetch%20Algo.md)): ROUTE (LLM
     call 1: the LIBRARY of page files + the query → `PAGES` / `REPLY` /
     `AGAIN` / `IGNORE`), then ANSWER (LLM call 2: the query + those
     pages' data + who is asking + scene notes + learned style).
  6. KB fallback when the LLM is unavailable.
  Every answered turn goes to the conversation log.
- `fetch.py`: the LIBRARY (catalog.json), the system prompt (persona +
  LIBRARY + `FETCH_SYSTEM_RULES`, identical on every call so Ollama keeps it
  cached), the route / answer messages, reading the route reply, choosing
  page data within `FETCH_DATA_WORDS`.
- `persona.py`: Luna's persona (voice + scope rules) shared by brain, the
  Ollama Modelfile and the benchmark.
- `llm_local.py`: backend dispatcher; `llm_ollama.py` (default) or the
  in-process llama.cpp path. Same `warmup/available/stream/ask` contract.
- `site_index.py` + `site_crawler.py`: Indexing (weekly, by hand): crawl the
  website (pages, ETLab data, PDFs) → group into topic files (`SITE_PAGES`)
  → Llama organizes each (fact-checked) → one-line descriptions →
  `data/site/pages/*.txt` + `data/site/catalog.json`.

**Seeing**
- `vision_module.py`: YuNet detection on a half-scale frame, emotion model,
  SFace recognition every `RECOGNIZE_EVERY` frames (debounced), guided
  three-pose registration, sighting log, feeds `scene_memory`.
- `face_store.py`: embeddings + names in `data/faces/`.
- `scene_memory.py`: who arrived/left, moods, objects → a one-sentence
  summary for the LLM, "welcome back" timing, visit tracking.
- `scene_context.py`: the "what do you see?" description + YOLOv8n.

**Acting**
- `proactive.py`: the recognition moment (animation + greeting or welcome
  back), conversation starters for people who linger, touch-to-register
  handoff. Never interrupts speech or an open conversation.
- `text_to_speech.py` + `piper_tts.py`: Piper via `paplay`, streamed per
  sentence; `say` on a Mac.
- `servo_module.py`, `gesture_module.py`, `behavior_engine.py`.

**Face**
- `face_style.py` (palettes, sprite cache), `expressions.py` (one function per
  mood, incl. `recognised`), `face_parts.py`, `face_particles.py`,
  `robot_face.py` (animation, tilt via a cropped face layer, double-take,
  banner slide, touch), `face_renderer.py` (policy), `face.py` (API).

**Learning**
- `conversation_log.py`: one JSONL file per day in `data/logs/`. Records what
  was heard, why it was accepted or ignored, the path, the reply, timings and
  the reward.
- `learning.py`: rewards from explicit feedback, thumbs-up, engagement,
  repeats, corrections and walk-aways. It feeds a Thompson-sampling style
  bandit, liked-answer exemplars, per-person profiles, pending taught facts,
  and a nightly diary.

## Data flow

1. **Vision:** camera → `state.frame` → detect → emotion → recognise →
   face renderer + proactive + scene memory.
2. **Voice:** mic → preprocess → Vosk → wake word / addressing gate →
   `brain.process` → command / clock / vision / KB / route → answer
   (streaming) → Piper → speaker → conversation log → rewards on the next
   turn.
3. **Touch:** tap an eye → boop; tap the mouth → giggle; hold the centre →
   register.

## Data on disk (runtime, never committed, never overwritten by deploy)

| Path | What |
|---|---|
| `data/faces/` | registered people |
| `data/face_log.jsonl` | sightings |
| `data/logs/` | conversation logs (`scripts/review_log.py`) |
| `data/logs_sim/` | simulated training sessions |
| `data/learning/` | bandit, exemplars, profiles, pending facts |
| `data/diary/` | nightly summaries |
| `data/site/` | website index (built on the Mac, synced by deploy.sh) |

## Verification

- `python -m unittest discover tests`: pure-logic tests that run on any machine.
- `python scripts/bench_llm.py`: LLM quality and speed (docs/llm_benchmark.md).
- `python scripts/bench_face.py --shots out/`: render timing + PNGs.
- `python scripts/bench_vision.py` and `python scripts/bench_vosk.py`.

---

## Deployment model

- **The dev machine is the source of truth.** Edit there, then
  `scripts/deploy.sh --go` (rsync) to push. **Do not hand-edit on the Pi.**
- deploy.sh **excludes** the venv, Vosk models, `models/llm/`, and every
  runtime data path above except `data/site/` (the website index the Mac
  builds weekly; its raw crawl and Llama cache stay on the Mac). `--seed-learning` pushes a Mac-trained
  `data/learning/` once; after that the Pi keeps learning on its own.
- `knowledge.txt` **hot-reloads**; code changes need a restart. After
  editing the persona or switching models, re-run
  `scripts/make_luna_model.py`.

## Known constraints

- **Audio ground-loop hum:** without an earth ground the dongle picks up
  50 Hz hum that clips the ADC. The high-pass cannot undo clipping; the fix is
  electrical (earth ground or lower mic gain).
- **RAM budget (8 GB):** Vosk en-in-0.5 ~1.7 GB + llama3.2:3b ~2.5 GB + torch
  + pygame fits because DeepFace/TensorFlow stays off.
- **LLM latency on the Pi:** about 5 s to the first token and 5-6 tokens/s
  for llama3.2:3b (estimated, see docs/llm_benchmark.md). Streaming TTS
  speaks the first sentence while the rest generates. For more speed use
  qwen2.5:1.5b.
- **Servos** need external power and a common ground; Pi 5 uses rpi-lgpio.
- **Wi-Fi is flaky** on the unit; its DHCP IP changes.
