# Luna 2.0

Luna is an offline desk robot for a college Computer Science department. She runs on a Raspberry Pi 5 with a touchscreen face, a camera, a microphone, a speaker and two small arm servos. The same code runs on a Mac for development.

What she does:

- **Listens** with offline speech recognition (Vosk listens live, Parakeet writes down what was said). She only answers when someone is actually talking to her, not to room chatter.
- **Answers** with a local LLM served by Ollama (default `llama3.2:3b`): close matches in `data/knowledge.txt` instantly, college questions from the website index, general questions briefly. She says plainly when something is outside what she knows.
- **Recognises people** she has been introduced to, greets them with a dedicated animation, and starts conversations with people she knows.
- **Understands the scene** ("what do you see?") and remembers who came and went, so conversation carries context.
- **Learns from daily life**: which reply style people like, who asks about what, answers that worked, and facts people teach her (after approval).
- **Knows the college website** (tkmce.ac.in). Once a week you run the indexing on the Mac: it reads the whole site (pages, ETLab staff / student lists, PDFs), groups it into ~40 topic files and has Llama clean each one up. For each question, the LLM first picks the files that may hold the answer from a one-line description of each, then answers from them. Nothing is downloaded while a visitor waits. See [docs/Fetch Algo.md](docs/Fetch%20Algo.md).

No cloud APIs, keys or accounts are needed at runtime.

---

## Contents

1. [Requirements](#1-requirements)
2. [Get the code](#2-get-the-code)
3. [Set up on a Mac (development)](#3-set-up-on-a-mac-development)
4. [Set up on a Raspberry Pi 5](#4-set-up-on-a-raspberry-pi-5)
5. [Configuration and paths](#5-configuration-and-paths)
6. [Everyday use](#6-everyday-use)
7. [Choosing the LLM](#7-choosing-the-llm)
8. [Logs and learning](#8-logs-and-learning)
9. [Tests and benchmarks](#9-tests-and-benchmarks)
10. [Project layout](#10-project-layout)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Requirements

| | Mac (development) | Raspberry Pi 5 (robot) |
|---|---|---|
| OS | macOS 13 or newer, Apple Silicon | Raspberry Pi OS 64-bit (Bookworm or newer) |
| RAM | 8 GB | 8 GB |
| Python | 3.11 | 3.11 |
| Disk | ~6 GB (models) | ~6 GB (models) |
| Other | Homebrew | USB audio dongle, Pi Camera, touchscreen, servos (optional) |

Internet is needed once, during setup, to download the models. After that Luna runs offline.

## 2. Get the code

```bash
git clone https://github.com/michaelharold/Luna-2.0.git
cd Luna-2.0
```

The repository holds code, docs, scripts, the knowledge file and two small vision models. Large models (speech, LLM, face recognition) are downloaded by the setup steps below. Everything Luna learns about people stays on the machine and is never committed.

## 3. Set up on a Mac (development)

**1. Python environment**

```bash
brew install python@3.11 portaudio
python3.11 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
```

**2. Ollama and the LLM**

Install Ollama from <https://ollama.com/download>, start it, then pull the base model and build Luna's model:

```bash
ollama serve &                 # skip if the Ollama app is already running
ollama pull llama3.2:3b
venv/bin/python scripts/make_luna_model.py
ollama run luna "who are you?" # quick check
```

`make_luna_model.py` builds a model called `luna` from the base model with Luna's persona and scope rules. Re-run it after changing `persona.py`, `OLLAMA_BASE_MODEL` or `OLLAMA_BAKE_KNOWLEDGE`.

**3. Speech and vision models**

```bash
./scripts/download_models.sh
```

This downloads the Vosk speech model (`vosk-model-en-in-0.5`, about 1 GB), the SFace face-recognition model and the sentence-transformer used for retrieval. It skips anything already present.

**4. Run**

```bash
venv/bin/python main.py
```

In Mac mode Luna uses the built-in camera and microphone, speaks with macOS `say`, runs in a window and simulates the servos. Press **1 / 2 / 3** to switch face styles and **Esc** to quit. macOS will ask for camera and microphone permission the first time.

## 4. Set up on a Raspberry Pi 5

**1. Clone into the home folder of the `luna` user** (the scripts expect `/home/luna/Luna`; see [paths](#5-configuration-and-paths) to use another location):

```bash
cd ~
git clone https://github.com/michaelharold/Luna-2.0.git Luna
cd Luna
```

**2. Run the installer once**

```bash
./scripts/install_pi.sh
```

It is safe to re-run. It:

1. installs the system packages: PipeWire, PortAudio, rpicam-apps and the SDL libraries
2. creates `venv/` and installs the Python requirements
3. installs Piper TTS and its voice into `/home/luna/piper/`
4. downloads the speech and vision models
5. installs Ollama as a service (one model, always loaded, local only), pulls the base model and builds `luna`
6. compiles the code and runs the tests

**3. Audio**

Set the USB dongle as the default input and output (`wpctl status`, then `wpctl set-default <id>`). Keep the sink around 90% and the mic source around 55%. Close Firefox, because it grabs the dongle.

**4. Run**

```bash
venv/bin/python -u main.py 2>&1 | tee ~/luna.log
```

With no monitor attached (over SSH), add `SDL_VIDEODRIVER=dummy` in front.

**Updating the Pi from a development machine** (optional): `scripts/deploy.sh` rsyncs code over SSH to the host alias `luna`. Run it once without arguments for a dry run, then `scripts/deploy.sh --go`. It never overwrites the Pi's registered faces, logs or learning state.

## 5. Configuration and paths

Every setting lives in **`config.py`**, in 22 numbered sections (the contents are at the top of the file) and commented. `RUN_MODE = "auto"` picks Mac or Pi behaviour automatically, so the same file works on both. These are the settings most people change:

| Setting | Default | Change it when |
|---|---|---|
| `RUN_MODE` | `"auto"` | you want to force `"mac"` or `"pi"` behaviour |
| `KNOWLEDGE_PATH` | `data/knowledge.txt` | you keep the knowledge file elsewhere |
| `VOSK_MODEL_PATH` | `vosk-model-en-in-0.5` | you use another Vosk model (low RAM: `vosk-model-small-en-in-0.4`) |
| `LLM_BACKEND` | `"ollama"` | you want the in-process llama.cpp path (`"llamacpp"`) |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama listens elsewhere |
| `OLLAMA_MODEL` | `luna` | you want Luna to talk to another Ollama model directly |
| `OLLAMA_BASE_MODEL` | `llama3.2:3b` | you switch LLMs (then re-run `make_luna_model.py`) |
| `PIPER_PATH`, `PIPER_MODEL` | `/home/luna/piper/...` | Piper is installed somewhere else (Pi only) |
| `AUDIO_INPUT_DEVICE` | Mac: default, Pi: `"pipewire"` | your microphone has a different name |
| `CAMERA_SOURCE` | Mac: `"usb"`, Pi: `"picam"` | you use a USB camera on the Pi |
| `SITE_URL`, `SITE_PAGES` | tkmce.ac.in, ~25 topic files | you index another website, or group its pages differently (`SITE_URL = ""` = off) |
| `FETCH_DATA_WORDS` | `700` | answers should be fuller (higher) or faster on the Pi (lower) |
| `FETCH_SYSTEM_RULES`, `FETCH_ROUTE_TASK`, `FETCH_ANSWER_TASK` | see comments | you change what the LLM is told when choosing pages and answering |
| `ROBOT_NAME`, `WAKE_WORDS` | `Luna`, hello / luna / hey luna | you rename her |
| `ENABLE_SERVOS` | `True` | no servos attached (it falls back to simulation anyway) |
| `ADDR_*`, `LOUDNESS_*` | see comments | she answers too much or too little |

**Paths on the Pi.** The scripts assume the project is at `/home/luna/Luna` and Piper is at `/home/luna/piper`. To use another user or folder:

- edit `PIPER_PATH` and `PIPER_MODEL` in `config.py`
- edit `PIPER_DIR` in `scripts/install_pi.sh`
- edit `DEST` in `scripts/deploy.sh`

Every other path is relative to the project folder.

**Knowledge.** Add one line per fact to `data/knowledge.txt`, as `question:answer`. The file reloads while Luna runs, so no restart is needed.

## 6. Everyday use

**Talking.** Say a wake word ("Luna", "hey Luna", "hello"), then ask. For the next few seconds she keeps listening without the wake word. Luna only answers speech aimed at her. The name, a question, "you", a known topic, someone facing her, or someone speaking close to the mic all count. Background chatter is ignored and the reason is logged.

**Registering a face.** There are three ways:

- **Voice.** Say "remember me, my name is Alice". Luna confirms the name ("Did you say Alice?"), then asks you to look straight, left and right while she learns your face.
- **Touch.** Press and hold the middle of her face for about a second, then say your name.
- **Photos.** Put photos in `data/enroll/<Name>/` and run `venv/bin/python scripts/enroll_faces.py`. This works without a camera.

"Who am I?", "who do you know?" and "forget me" also work.

**Feedback and teaching.**

- "Good answer" or a thumbs-up rewards the last reply.
- "That's wrong" lets you give the right answer.
- "Remember that the library closes at 6" stores a fact as pending.

Pending facts only reach `knowledge.txt` after you approve them:

```bash
venv/bin/python scripts/approve_facts.py            # list pending facts
venv/bin/python scripts/approve_facts.py --approve 1 "when does the library close"
```

**Other commands.** "Volume up / down / fifty percent", "go to sleep", "wake up", "change your face", "what do you see?". To shut down, say "exit", then "exit" again as the password.

**Website knowledge (weekly).** Run the indexing on the Mac about once a week, then deploy. It reads tkmce.ac.in (the menus, `data/website_links.txt`, the ETLab data behind the CSE pages, the PDFs). It groups everything into topic files by the rules in `SITE_PAGES`, lets Llama turn each into compact notes (fact-checked), and writes a one-line description of each to `data/site/catalog.json`. A first run takes a couple of hours; later runs only send changed pages to Llama. How a question is then answered (route, then answer) is in [docs/Fetch Algo.md](docs/Fetch%20Algo.md).

```bash
venv/bin/python site_index.py              # crawl + group + organize + describe
venv/bin/python site_index.py --status     # what is indexed
venv/bin/python fetch.py "who is the hod"  # try a question: route, data, answer
scripts/deploy.sh --go                     # the Pi gets data/site/ with the code
```

## 7. Choosing the LLM

The default was picked by `scripts/bench_llm.py`, which runs 40 spoken-style questions through each model and grades them: college facts, general questions, and requests she should decline. Results and reasoning are in [docs/llm_benchmark.md](docs/llm_benchmark.md).

| Model | Quality | Speed on Pi 5 (estimated) |
|---|---|---|
| `llama3.2:3b` (default) | best, no invented facts | ~5 s to first word |
| `qwen2.5:1.5b` | good on facts, weaker at declining | ~3 s |
| `llama3.2:1b` | invents facts, not recommended | ~2.5 s |

To switch, pull the model, change `OLLAMA_BASE_MODEL`, and rebuild:

```bash
ollama pull qwen2.5:1.5b
# edit OLLAMA_BASE_MODEL in config.py
venv/bin/python scripts/make_luna_model.py
```

To benchmark models on your own hardware:

```bash
venv/bin/python scripts/bench_llm.py --models luna,qwen2.5:1.5b --variants rag
```

## 8. Logs and learning

**Conversation log.** Every utterance Luna hears and every reply she gives is written to `data/logs/conversation_YYYY-MM-DD.jsonl`, including the ones she ignored and why. Read a day back:

```bash
venv/bin/python scripts/review_log.py              # today
venv/bin/python scripts/review_log.py 2026-09-24 --ignored
venv/bin/python scripts/review_log.py --rewards
```

**Learning loop.** Rewards come from feedback, thumbs-ups, whether people keep talking, repeated questions, corrections, and people walking away. They update:

- a reply-style chooser (length, tone, follow-up question) per situation
- a bank of liked answers used as examples
- per-person profiles
- a short nightly diary

Everything is stored in `data/learning/` and `data/diary/`.

**Train on the Mac first, then continue on the Pi.** The full walkthrough is in [Train Instructions.md](Train%20Instructions.md).

```bash
# simulated visitors (a second local model plays the visitor and the judge)
venv/bin/python scripts/train_learning.py --simulate 30 --judge-model gemma3:4b
# or rebuild from real conversation logs
venv/bin/python scripts/train_learning.py --replay
# push the trained state to the Pi ONCE; later deploys never overwrite it
scripts/deploy.sh --go --seed-learning
```

## 9. Tests and benchmarks

```bash
venv/bin/python -m unittest discover tests          # 440 tests, no hardware needed
venv/bin/python scripts/bench_llm.py                 # LLM quality + speed
venv/bin/python scripts/bench_face.py --shots out/   # face render time + PNG per expression
venv/bin/python scripts/bench_vision.py              # face detection / recognition time
venv/bin/python scripts/bench_vosk.py --help         # speech recognition + preprocessing
venv/bin/python scripts/show_face.py happy           # show one expression in a window
```

## 10. Project layout

```
Luna-2.0/
├── main.py                  start here: launches every thread and the face
├── config.py                every setting
├── brain.py                 answer pipeline (commands → clock → vision → KB → route → answer)
├── fetch.py                 Data Fetch: the LIBRARY, route / answer prompts, page data
├── persona.py               Luna's persona and scope rules
├── llm_local.py, llm_ollama.py        LLM backends
├── speech_to_text.py, audio_preprocess.py, addressing.py    hearing
├── text_to_speech.py, piper_tts.py                          speaking
├── vision_module.py, face_store.py, scene_context.py, scene_memory.py, camera_thread.py
├── proactive.py, commands.py, gesture_module.py, behavior_engine.py, servo_module.py
├── robot_face.py, face_renderer.py, face.py, expressions.py, face_parts.py, face_particles.py, face_style.py
├── conversation_log.py, learning.py                          logs + learning
├── site_index.py            Indexing (weekly): group, organize, describe the website
├── site_crawler.py          Indexing step 1: read the website (pages, data, PDFs)
├── data/knowledge.txt       curated college facts (instant answers + a library page)
├── data/website_links.txt   hand-picked website pages, always indexed
├── data/site/               the website index: pages/*.txt + catalog.json
├── models/                  emotion model, vision ONNX models (LLM lives in Ollama)
├── scripts/                 install, deploy, models, benchmarks, training, tools
├── tests/                   unit tests + LLM evaluation set
└── docs/                    architecture, benchmark results, servo wiring, changelog
```

Every command in one place: [Syntaxes.md](Syntaxes.md). How Luna knows the website: [docs/Fetch Algo.md](docs/Fetch%20Algo.md). Website knowledge and training: [Train Instructions.md](Train%20Instructions.md). Architecture in depth: [docs/PROJECT_PIPELINE.md](docs/PROJECT_PIPELINE.md). Change history: [docs/UPDATES.txt](docs/UPDATES.txt). Servo wiring: [docs/SERVO_SETUP.md](docs/SERVO_SETUP.md).

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `[llm] Ollama server not reachable` | Start Ollama (`ollama serve`, or `sudo systemctl start ollama` on the Pi). Luna answers from the knowledge file until it is up. |
| `[llm] ... run scripts/make_luna_model.py` | The `luna` model is not built yet. Run the script. Luna uses the base model meanwhile. |
| No voice input | Run `venv/bin/python -c "import sounddevice as sd; print(sd.query_devices())"` and set `AUDIO_INPUT_DEVICE`. |
| Silent replies on the Pi | Capture must go through `pipewire`, not a raw `hw:` device. Check the speaker's own power switch. Close Firefox. |
| She answers background chatter | Raise `ADDR_THRESHOLD`, or set `ADDR_FACE_OR_NAME = True`. The log shows each decision's breakdown. |
| She ignores you | Lower `ADDR_THRESHOLD` or `LOUDNESS_NEAR_DB`. Check the `[STT] Ignored (...)` lines. |
| Recognition fails in dim light | Register again in better light. Tune `FACE_MATCH_THRESH`. |
| Replies are slow on the Pi | Switch to `qwen2.5:1.5b` (section 7). |
| Constant hum or garbled words | Ground loop on the USB dongle: earth the setup or lower the mic gain. |

---

Built by Michael Harold Sony, Sreenadhana and Devadathan for the Department of Computer Science, TKM College of Engineering, Kollam.
