# Luna 2.0 — Train Instructions

This guide covers the two ways Luna gets smarter after setup:

1. **Website knowledge.** Once a week, Luna's index of the college website is rebuilt; she answers college questions from it.
2. **Learning from daily use.** Luna logs conversations, learns which answers and reply styles work, and accepts new facts after you approve them.

Neither step changes the LLM's weights. Luna's model stays the same; what grows is the knowledge, examples and preferences that go into each prompt. That is why training is fast, runs on a Mac or the Pi, and can be undone by deleting a folder.

It assumes Luna is already installed and runs (see `README.md`). Run every command from the project folder.

---

## Part 1 — Website knowledge (weekly indexing)

The whole algorithm, with the reasons behind it, is in [docs/Fetch Algo.md](docs/Fetch%20Algo.md). In short, run this on the Mac about once a week:

```bash
venv/bin/python site_index.py            # crawl + group + organize + describe
venv/bin/python site_index.py --status   # check: files, sizes, descriptions
venv/bin/python fetch.py "who is the hod of cse"   # try a question
scripts/deploy.sh --go                   # the Pi gets data/site/ with the code
```

What it does:

1. **Crawl.** It reads every page in the site's menus and in `data/website_links.txt`, the data the CSE pages load from ETLab (staff, student lists, publications, news), and the text PDFs they link to. Scanned PDFs have no text and are reported as "needs OCR".
2. **Group.** Every piece goes to one topic file by the rules in `SITE_PAGES` (`config.py`, section 10), for example `cse_people`, `admissions` or `examinations`. Tenders, RTI pages, meeting minutes, quotation notices and PDFs named only by a number are dropped. Other departments get one small file each.
3. **Organize.** Llama (`llama3.2:3b`) rewrites each file's text as short notes. If the notes lose more than a fifth of the numbers and names, or invent a number, the cleaned text is kept instead. Staff and student lists are formatted without the LLM. Answers are cached, so next week only changed pages go to Llama again.
4. **Describe.** Llama writes one line per file. Those lines are what the LLM reads to choose files for a question.

The first run takes about two hours on an M3 Mac; later runs are much faster. It can also run on the Pi (stop Luna first), but takes many hours there.

**Changing the grouping.** Edit `SITE_PAGES`, then run `site_index.py --offline --plan` to see the files without calling Llama, and `site_index.py --offline` to rebuild them.

**What the LLM is told** is in `config.py`, section 11: `FETCH_SYSTEM_RULES` (how to choose files), `FETCH_ROUTE_TASK` and `FETCH_ANSWER_TASK`. `venv/bin/python fetch.py --library` prints the full system prompt.

**knowledge.txt still wins.** A question that matches `data/knowledge.txt` very closely is answered from it instantly, and the file is also one of the library pages (`kb`). If the website and `knowledge.txt` disagree, fix `knowledge.txt`.

## Part 2 — Learning from daily use

### How it works

Every conversation turn is logged, and every turn can earn a **reward** between −1 and +1:

| Signal | Reward | How Luna notices |
|---|---|---|
| "good answer", "that's right", "perfect" | +1.0 | the next thing said |
| thumbs-up gesture after a reply | +0.6 | camera |
| "thanks", laughter, a compliment | +0.5 | the next thing said |
| they keep talking within 20 s | +0.25 | timing |
| they ask the same question again | −0.5 | similarity to the last question |
| "that's wrong", "no, that's not right" | −1.0 | the next thing said |
| they walk away within 10 s of a reply | −0.2 | camera |

Rewards update four things, all stored in `data/learning/`:

1. **Reply style.** There are 8 styles, made from three choices: one sentence or two to three; playful or plain and warm; ending with a question or not. Luna keeps a separate score for each situation: known person or stranger, college, general or out-of-scope question, and morning, afternoon or evening. She tries styles evenly at first. After 8 rewarded turns in a situation she leans toward the style that has earned the best rewards, while still trying the others now and then.
2. **Liked answers.** Replies rewarded +0.5 or more are kept, up to 200. The two most similar are shown to the LLM as examples for new questions.
3. **People.** Recognised people get a profile: number of visits, favourite topics and preferred style. When she sees them, a one-line summary goes into the prompt.
4. **Taught facts.** "Remember that…", or the answer given after "that's wrong", goes to a pending list and only becomes knowledge after you approve it (Step 5).

After 10 pm Luna writes a three-sentence **diary** of the day to `data/diary/` and can recall it later ("lots of people asked about Hestia yesterday").

### Step 1: Check the logs are being written

Every utterance Luna hears, answered or ignored, is written to `data/logs/conversation_YYYY-MM-DD.jsonl`. Read a day as a transcript:

```bash
venv/bin/python scripts/review_log.py                 # today
venv/bin/python scripts/review_log.py yesterday
venv/bin/python scripts/review_log.py 2026-09-24 --ignored   # include ignored speech
venv/bin/python scripts/review_log.py --rewards       # rewards + style scores
venv/bin/python scripts/review_log.py --json          # raw records
```

Use the transcript to fix problems at the source:

| You see | Fix |
|---|---|
| Wrong or missing college fact | add or correct the line in `data/knowledge.txt` (reloads live) |
| She answered chatter that wasn't meant for her | raise `ADDR_THRESHOLD`, or set `ADDR_FACE_OR_NAME = True` |
| She ignored a real question (`--ignored` shows why) | lower `ADDR_THRESHOLD` or `LOUDNESS_NEAR_DB` |
| She invented something | add the right fact to `knowledge.txt`; if it keeps happening, check `docs/llm_benchmark.md` |
| Misheard words | better mic placement or gain; `scripts/bench_vosk.py` to test |

Logs are kept for 365 days on a Mac and 45 days on the Pi (`CONVO_LOG_RETENTION_DAYS`).

### Step 2: Train on the Mac with simulated visitors

Before Luna meets real people, give her a head start. A second local model plays the visitor, with varied roles and goals: a student, parent, faculty member or prospective student, who might be curious, impatient or chatty. A third model judges each reply from 1 to 5. The score becomes a reward (≤2 → −1, 3 → 0, 4 → +0.5, 5 → +1).

```bash
# a quick check first
venv/bin/python scripts/train_learning.py --simulate 2 --judge-model gemma3:4b

# a real run
venv/bin/python scripts/train_learning.py --simulate 30 --judge-model gemma3:4b
```

Options:

- `--visitor-model` sets the model that plays the visitor (default `llama3.2:3b`).
- `--judge-model` sets the judge. **Use `gemma3:4b` or a larger model.** `llama3.2:3b` was too lenient in testing and approved invented details.
- `--luna-model` sets the model that answers as Luna (default `luna`).
- `--seed` makes a run repeatable.

Simulated sessions are logged to `data/logs_sim/`, never mixed with real logs. Each session takes a minute or two on an M3 Mac, so budget up to an hour for 30 sessions. At the end you get a table of the best style per situation.

Check the result:

```bash
venv/bin/python scripts/review_log.py --sim --rewards
```

If the judge is rewarding bad answers (invented names, pretending to call someone), throw the run away and retrain with a stronger judge:

```bash
rm -rf data/learning data/logs_sim
venv/bin/python scripts/train_learning.py --simulate 30 --judge-model gemma3:4b
```

### Step 3: Train from real conversations

Once Luna has been used for a while (on the Mac, or with logs copied from the Pi), rebuild her learning state from the real logs:

```bash
venv/bin/python scripts/train_learning.py --replay                  # real logs only
venv/bin/python scripts/train_learning.py --replay --include-sim    # real + simulated
```

`--replay` wipes and rebuilds the style scores, liked answers and profiles from the logs. Pending facts and the diary are kept.

To copy the Pi's logs to the Mac first:

```bash
rsync -av luna:/home/luna/Luna/data/logs/ data/logs/
```

### Step 4: Send the trained state to the Pi once

```bash
scripts/deploy.sh --go --seed-learning
```

This copies `data/learning/` to the Pi a single time. After that, normal deploys (`scripts/deploy.sh --go`) never touch the Pi's `data/learning/`, `data/logs/` or `data/diary/`, so everything the Pi learns on its own is kept.

To reset the Pi's learning completely, stop Luna and delete that folder:

```bash
ssh luna 'rm -rf /home/luna/Luna/data/learning'
```

### Step 5: Approve facts people taught her

Nothing people tell Luna becomes "knowledge" until a person approves it. That stops pranks and mistakes from reaching visitors.

```bash
venv/bin/python scripts/approve_facts.py                  # list pending facts, numbered
venv/bin/python scripts/approve_facts.py --approve 2 "when does the library close"
venv/bin/python scripts/approve_facts.py --reject 3
```

`--approve N "question"` adds `question:fact` to `data/knowledge.txt` and removes it from the pending list.

**Important on the Pi.** Approving on the Pi changes the Pi's `knowledge.txt`, and the next `deploy.sh --go` from the Mac overwrites it. Always copy approved lines into the Mac's `data/knowledge.txt` too. The script prints a reminder.

### A routine that works

| When | What |
|---|---|
| Once, before first use | `train_learning.py --simulate 30 --judge-model gemma3:4b`, then `deploy.sh --go --seed-learning` |
| Daily (5 min) | `review_log.py yesterday --ignored` on the Pi; fix `knowledge.txt`; `approve_facts.py` |
| Weekly | copy the Pi logs to the Mac, check `review_log.py --rewards`, adjust `ADDR_*` if needed |
| Weekly, or when the website changes | `site_index.py` on the Mac, then `deploy.sh --go` |

### Settings

All in `config.py`, section "Learning from daily life":

| Setting | Default | Meaning |
|---|---|---|
| `LEARNING_ENABLED` | `True` | master switch |
| `LEARNING_STYLE_ENABLED` | `True` | let learned styles steer replies |
| `LEARNING_MIN_TURNS_TO_STEER` | 8 | rewarded turns before a situation's style is trusted |
| `LEARNING_EXEMPLARS_K` | 2 | liked answers shown per question |
| `LEARNING_EXEMPLAR_MIN_REWARD` | 0.5 | reward needed to keep an answer |
| `LEARNING_ENGAGEMENT_WINDOW` | 20 s | a follow-up within this counts as engaged |
| `LEARNING_REWARD_*` | see the reward table | the size of each signal |
| `DIARY_ENABLED`, `DIARY_WRITE_HOUR` | `True`, 22 | nightly diary |
| `CONVO_LOG_RETENTION_DAYS` | 365 Mac / 45 Pi | how long logs are kept |

### Privacy

The logs, profiles and diary record what people said and who said it. They stay on the machine: `.gitignore` keeps `data/logs/`, `data/learning/`, `data/diary/`, `data/faces/` and `data/site/` (it holds the student lists) out of the repository, and deploys never copy the logs, learning, diary or faces over. If you run Luna in a public space, put up a short notice that she records conversations, and shorten `CONVO_LOG_RETENTION_DAYS` if needed. "Forget me" removes a person's face. To delete everything Luna remembers about people:

```bash
rm -rf data/faces data/face_log.jsonl data/logs data/learning data/diary
```

---

## Optional, later: fine-tuning the model

Everything above leaves the LLM untouched. If months of logs show a consistent problem that prompting can't fix, you can fine-tune:

1. Export high-reward turns with `scripts/review_log.py --json`.
2. Review them by hand.
3. Train a LoRA adapter for `llama3.2:3b` on the Mac, for example with MLX or Unsloth.
4. Import the adapter into Ollama with an `ADAPTER` line in the Modelfile.
5. Re-run `scripts/bench_llm.py` and only switch if the score goes up.

This is not needed to get started, and a few hundred noisy turns usually make a model worse, not better.
