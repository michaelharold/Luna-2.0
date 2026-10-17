# Syntaxes — Luna command reference

Every command Luna needs, what it does, and when to run it. Run them from the project folder:

```bash
cd ~/Developer/Luna      # on the Mac
cd /home/luna/Luna       # on the Pi
```

---

## 1. Run Luna

| What for | Command |
|---|---|
| Start the LLM server (skip if the Ollama app is already running) | `ollama serve &` |
| Run Luna on the Mac | `venv/bin/python main.py` |
| Run Luna on the Pi, keeping a log | `venv/bin/python -u main.py 2>&1 \| tee ~/luna.log` |
| Run on the Pi over SSH with no monitor | `SDL_VIDEODRIVER=dummy venv/bin/python -u main.py 2>&1 \| tee ~/luna.log` |

While she runs: **1 / 2 / 3** switch face styles, **Esc** quits. To shut down by voice, say "exit", then "exit" again as the password.

---

## 2. The LLM model (Ollama)

| What for | Command |
|---|---|
| Download the base model (once) | `ollama pull llama3.2:3b` |
| Build Luna's model (persona and rules baked in). Re-run after changing `persona.py`, `OLLAMA_BASE_MODEL` or `OLLAMA_BAKE_KNOWLEDGE` | `venv/bin/python scripts/make_luna_model.py` |
| Talk to the model directly, without the robot | `ollama run luna "who are you?"` |
| List the models Ollama has | `ollama list` |
| Measure answer quality and speed | `venv/bin/python scripts/bench_llm.py` |

---

## 3. Face registration (recognition)

Luna greets people she has met by name. Nobody is registered until you do one of the following.

**By voice, in front of the camera:**

1. Say **"remember me, my name is Michael"**. "Register my face", "scan my face" and a bare "remember me" (she then asks for your name; just "Michael" is enough) also work.
2. Luna asks "Did you say Michael?". Answer **yes**, or say the name again.
3. Follow her prompts: look straight, turn a little left, then a little right.

During registration the screen shows a **live mirrored camera view**. Keep your face inside the circle; it turns green when your face is detected, and the current pose prompt is shown underneath. Afterwards, whenever she recognises a registered person, the screen shows **"Hi Michael!"** with an excited face (at most once a minute per person). You can change this with `RECOG_ON_SIGHT`, `RECOG_EXPRESSION` and `ENROLL_PREVIEW` in `config.py`.

**By touch:** press and hold the centre of her face for about 1 second, then say your name.

**From photos, with no camera** (good for pre-loading staff):

```bash
# one folder per person; the folder name is the person's name
#   data/enroll/Michael/1.jpg   data/enroll/Michael/2.jpg
venv/bin/python scripts/enroll_faces.py --dir data/enroll

# or one person from specific files
venv/bin/python scripts/enroll_faces.py --name Michael a.jpg b.jpg
```

Use photos with one clear face each (crop group photos first). Registering the same name again replaces the old entry.

| Say | What happens |
|---|---|
| "who am I?" | She says who she recognises |
| "who do you know?" | Lists everyone registered |
| "forget me" / "forget Michael" | Deletes that person |

Faces are stored in `data/faces/`. Deleting that folder makes Luna forget everyone. Deploys never overwrite the Pi's faces.

---

## 4. Reinforcement learning

Luna learns which reply style works best (length, tone, whether to end with a question) for each situation, keeps replies people liked as examples, builds per-person profiles and writes a nightly diary. Everything is stored in `data/learning/` and `data/diary/`.

**Training by talking (automatic).** These count as feedback on her last reply:

| Signal | Effect |
|---|---|
| "good answer" / thumbs-up / "thanks" | reward |
| asking a follow-up question | small reward |
| "that's wrong" (she then asks for the right answer) | penalty, and the correction is queued for approval |
| asking the same thing again | penalty |
| walking away mid-answer | small penalty |

She only starts steering her style after 8 rated turns per situation. Until then she tries styles at random.

**Head start on the Mac** (before real visitors):

| What for | Command |
|---|---|
| Simulated visitors: a second model plays the visitor and scores every reply | `venv/bin/python scripts/train_learning.py --simulate 30` |
| Same, with a separate judge model (pull it first with `ollama pull gemma3:4b`) | `venv/bin/python scripts/train_learning.py --simulate 30 --judge-model gemma3:4b` |
| Rebuild learning from real conversation logs | `venv/bin/python scripts/train_learning.py --replay` |
| Replay real and simulated logs together | `venv/bin/python scripts/train_learning.py --replay --include-sim` |
| Copy the trained state to the Pi (do this **once**) | `scripts/deploy.sh --go --seed-learning` |

**Teaching her facts.** Say "remember that the lab closes at five". Facts are never added automatically; a person approves them:

| What for | Command |
|---|---|
| List facts waiting for approval | `venv/bin/python scripts/approve_facts.py` |
| Approve fact 2 under a question | `venv/bin/python scripts/approve_facts.py --approve 2 "when does the lab close"` |
| Reject fact 3 | `venv/bin/python scripts/approve_facts.py --reject 3` |

Approved facts go into `data/knowledge.txt`, and a running Luna picks them up immediately.

**Reading what happened:**

| What for | Command |
|---|---|
| Today's conversations | `venv/bin/python scripts/review_log.py` |
| A specific day, including what she ignored and why | `venv/bin/python scripts/review_log.py 2026-10-11 --ignored` |
| Only rewarded turns, with a per-style summary | `venv/bin/python scripts/review_log.py --rewards` |
| The simulated training sessions | `venv/bin/python scripts/review_log.py --sim` |

The full walkthrough is in [Train Instructions.md](Train%20Instructions.md).

---

## 5. Website knowledge (tkmce.ac.in)

**How it works** (in full: [docs/Fetch Algo.md](docs/Fetch%20Algo.md)). Two parts:

- **Indexing, about once a week, on the Mac.** It reads every page in the site's menus and in `data/website_links.txt`, plus the staff, student and publication data the pages load from ETLab and the PDFs they link to. Everything is grouped into topic files (`SITE_PAGES` in `config.py`: `cse_people`, `admissions`, `examinations`, …). Llama turns each file into compact notes, and its notes must keep the facts or the clean text is kept instead. Each file gets a one-line description in `data/site/catalog.json`.
- **Data Fetch, every question.** The LLM sees the list of files with their descriptions and picks the ones that may hold the answer (or answers straight away for small talk, asks you to repeat, or ignores noise). Then it answers from those files. Nothing is downloaded while you wait.

| What for | Command |
|---|---|
| Index the website (crawl + group + organize + describe; first run ~2 h, later runs only redo changed pages) | `venv/bin/python site_index.py` |
| Same, without downloading again (after changing `SITE_PAGES` or a prompt) | `venv/bin/python site_index.py --offline` |
| Show which topic files the pages would go to, without Llama | `venv/bin/python site_index.py --offline --plan` |
| Rebuild only some files | `venv/bin/python site_index.py --offline --only cse_people admissions` |
| Ignore the cached Llama answers (after changing `SITE_ORGANIZE_PROMPT`) | `venv/bin/python site_index.py --offline --force` |
| Show what is indexed (files, words, descriptions) | `venv/bin/python site_index.py --status` |
| Check every link in scope (ok / broken / duplicate / skipped and why) | `venv/bin/python site_crawler.py --verify` |
| Try a question: which files, how much data, the answer, timings | `venv/bin/python fetch.py "who is the hod of cse"` |
| See exactly what the LLM is told (persona + file list + rules) | `venv/bin/python fetch.py --library` |
| Compare speed with another model / format (run it on the Pi) | `venv/bin/python fetch.py --model llama3.2:3b-instruct-q4_0 "who is the hod"` |
| Send the index to the Pi | `scripts/deploy.sh --go` |

**Settings** (`config.py`, sections 10 and 11):
- `SITE_URL`, `SITE_SEEDS`, `SITE_LINKS_FILE`: what is read (`SITE_URL = ""` = no website).
- `SITE_PAGES`, `SITE_SKIP_PAGES`, `SITE_SKIP_TITLES`: how pages are grouped, and what is never kept.
- `SITE_ORGANIZE_PROMPT`, `SITE_DESCRIBE_PROMPT`: what Llama is asked while indexing.
- `FETCH_SYSTEM_RULES`, `FETCH_ROUTE_TASK`, `FETCH_ANSWER_TASK`: what the LLM is told for each question.
- `FETCH_DATA_WORDS`: how much page data an answer may read (more = fuller, slower on the Pi).

Anything she must always get right (such as the principal's name) belongs in `data/knowledge.txt`. A very close match there is answered instantly, and the file is also one of the library pages (`kb`).

## 6. Setup and deploying to the Pi

| What for | Command |
|---|---|
| Download the speech and vision models (Mac or Pi, once) | `./scripts/download_models.sh` |
| Re-download every model | `FORCE=1 ./scripts/download_models.sh` |
| Full Pi install: packages, venv, Piper voice, models, Ollama, tests. Safe to re-run | `./scripts/install_pi.sh` |
| Preview what a deploy would change (dry run) | `scripts/deploy.sh` |
| Push the code from the Mac to the Pi | `scripts/deploy.sh --go` |
| Log in to the Pi | `ssh luna` |

Deploys never overwrite the Pi's faces, logs or learning state.

---

## 7. Tests and benchmarks

| What for | Command |
|---|---|
| Run all unit tests (no hardware needed) | `venv/bin/python -m unittest discover tests` |
| Run only the website tests | `venv/bin/python -m unittest tests.test_site_crawler tests.test_site_index tests.test_fetch` |
| Time face rendering and save a PNG per expression | `venv/bin/python scripts/bench_face.py --shots out/` |
| Time face detection and recognition | `venv/bin/python scripts/bench_vision.py` |
| Speech recognition benchmark (see its options) | `venv/bin/python scripts/bench_vosk.py --help` |
| Show one face expression in a window | `venv/bin/python scripts/show_face.py happy` |

---

## 8. Voice commands

| Say | What she does |
|---|---|
| "volume up" / "volume down" / "volume fifty percent" | changes the volume |
| "go to sleep" / "wake up" | sleeps or wakes |
| "change your face" | cycles through the face styles |
| "what do you see?" | describes what the camera sees |
| "shake hands" / "hi" / "bye" | hand gesture or wave |
| "remember me, my name is …" | registers your face |
| "who am I?" / "who do you know?" / "forget me" | face recognition |
| "good answer" / "that's wrong" | rates her last reply |
| "remember that …" | teaches a fact (needs approval) |
| "exit", then "exit" | shuts down |
