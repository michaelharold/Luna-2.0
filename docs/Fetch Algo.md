# Fetch Algo — how Luna knows the college website

Two parts:

| Part | When | Code | Writes / reads |
|---|---|---|---|
| **Indexing()** | by hand, about once a week (Mac recommended) | `site_index.py` (crawler: `site_crawler.py`) | writes `data/site/` |
| **Data Fetch()** | on every question | `fetch.py`, called by `brain.py` | reads `data/site/` |

Every instruction the LLM gets is a setting in `config.py`, under
**WEBSITE KNOWLEDGE: INDEXING** and **WEBSITE KNOWLEDGE: DATA FETCH**.

---

## Indexing()

```
Indexing():                                    python site_index.py

  1. Crawl                                     site_crawler.crawl()
     for each link in the website (site menus + data/website_links.txt):
         fetch the page text, its tables, the data it loads with
         JavaScript (ETLab: staff, students, publications, news) and
         the text PDFs it links to (scanned PDFs: "needs OCR")
     save everything                           -> data/site/crawl.json

  2. Group into page files                     site_index.group()
     each page / data section / PDF goes to ONE page file, chosen
     by the rules in SITE_PAGES (config): e.g. faculty-cse.php and
     ETLab's staff lists -> cse_people.txt; the 9 IQAC pages ->
     iqac_accreditation.txt
     drop what is meaningless:
         SITE_SKIP_PAGES (tenders, RTI, office orders, ...)
         duplicates, empty frames, PDFs named only by a number
         ("16-11869642-OBC-2021"), text under SITE_MIN_PAGE_WORDS
     cap each PDF at SITE_PDF_MAX_WORDS and each file's input at
     SITE_GROUP_MAX_WORDS (the rest is reported, not silently lost)

  3. Organize each page file with Llama        site_index.organize()
     prose (page text, PDFs): sent in pieces of SITE_LLM_CHUNK_WORDS
         Llama gets  Data + SITE_ORGANIZE_PROMPT
         ("keep every name, number, date ...; drop menus, links, filler;
           group under '## ' headings; '- ' one fact per line")
         checked: the output must keep >= SITE_MIN_FACT_KEEP of the
         input's numbers and names and invent none -> else the cleaned
         input is used as it is (nothing is lost to a bad rewrite)
     lists (staff, students, publications, table rows): formatted
         without the LLM, one line each -- faster and lossless
     every answer is cached by the hash of its input
     (data/site/llm_cache.json): next week only CHANGED pages go to
     Llama again
                                               -> data/site/pages/<name>.txt

  4. Describe each page file                   site_index.describe()
     Llama gets the organized page + SITE_DESCRIBE_PROMPT
         -> one line: what this file holds
     catalog = { "cse_people": "CSE faculty and staff: HOD, professors,
                                ... names and designations",
                 "admissions": "B.Tech / M.Tech / PhD / lateral entry
                                admission rules, dates, documents", ... }
     + the knowledge base (data/knowledge.txt) as page "kb"
     + a name index (person name word -> page files) for routing hints
     + the staff lists (for the person cards on screen)
                                               -> data/site/catalog.json
```

Run it:

```bash
python site_index.py              # crawl + group + organize + describe
python site_index.py --offline    # skip the crawl, re-use crawl.json
python site_index.py --status     # what is indexed, sizes, crawl report
python site_index.py --force      # ignore the Llama cache (new prompt)
```

On the Mac (llama3.2:3b on its GPU) a first full run takes about an hour.
Weekly re-runs only send changed pages to Llama. Then push the result to
the Pi with `scripts/deploy.sh --go`. Running it on the Pi itself works
too (stop Luna first), but takes many hours.

---

## Data Fetch() — default: ONE call (`FETCH_MODE = "digest"`)

The two-call design below routed 11 of 12 test questions right, but on the
robot it failed exactly where it matters: a misheard name ("rishma sheikh"
for Dr. Reshma Sheik) matched no page, and the model answered from memory
("an Indian actress"). So the default is now one call:

```
Indexing() also writes data/site/digest.txt: the key facts of every page
file in ONE compact file (~3,500 words, SITE_DIGEST_WORDS per file)

Data Fetch(query):
  0. instant paths: commands, clock, camera, knowledge.txt / FAQ match,
     answer cache
  1. lines = the page lines that share the query's distinctive words,
     matched by closest spelling too ("rishma sheikh" -> "Dr. Reshma Sheik,
     Asst. Professor")                           fetch.library.lookup_lines
  2. ONE LLM call:
       system  = persona + knowledge.txt + digest + FETCH_DIGEST_RULES
                 (the same on every call: Ollama reads it ONCE, at boot,
                  and keeps it cached — a question costs only its own
                  ~100 tokens)
       message = FETCH_DIGEST_TASK: query + lines + who is asking + scene
     -> the answer, streamed to the voice; or IGNORE (noise) / AGAIN
```

Everything the website has does NOT fit: ~200k words would take hours per
question on the Pi. The digest holds what is asked most; the matching lines
bring in the rest (a student, a publication) per question.

## Data Fetch() — `FETCH_MODE = "route"` (two calls)

```
Data Fetch(query):                             brain.process() -> fetch.py

  0. Instant paths (no LLM, or no website):
         voice commands, date / time, "what do you see?" (camera),
         a close match (>= 0.70) to knowledge.txt OR to the FAQ the
         indexing wrote (8 questions + answers per page file),
         the same question asked in the last 7 days (answer cache)

  0b. Obvious pages -> skip ROUTE (FETCH_DIRECT): "who is <a name in the
         name index>", or the one page file the query names
         ("placements" -> placements)

  1. ROUTE: LLM call 1                         fetch.route()
     system prompt (the SAME on every call, so Ollama keeps it cached
     and the Pi never re-reads it):
         Luna's persona
         + LIBRARY: every page name : description   (catalog.json)
         + FETCH_SYSTEM_RULES (how to route, how to answer)
     earlier turns of this conversation (the "context")
     user message = FETCH_ROUTE_TASK:
         Query + "The person asking is {name}" (if recognised)
         + scene notes (only for questions about who is / was around)
         + "speech recognition was unsure" (low STT confidence)
         + name hints ("Saheer" appears in: cse_people)

     the model answers with ONE line:
         PAGES: cse_people, cse_research   -> step 2
         REPLY: <answer>                   -> spoken as it streams
                                              (jokes, small talk,
                                               general knowledge, "I
                                               don't have that")
         AGAIN                             -> "Sorry, could you say
                                              that again?"
         IGNORE                            -> nothing (noise)
     a misheard query that is >50% clear is answered as corrected
     (the model is told so); PAGES stops generating as soon as the
     line is complete

     a REPLY to a question with college words (FETCH_COLLEGE_WORDS) is
     overruled — answered from the pages sharing its words instead: a 3B
     model invents names when it answers the college from memory

  2. ANSWER: LLM call 2                        fetch.answer_message()
     data = data[page1] + data[page2] + ...    (whole pages while they
            fit FETCH_DATA_WORDS; a bigger one gives its sections /
            lines that share words with the query, in page order)
     user message = FETCH_ANSWER_TASK:
         Query + Data + person + scene context + instruction
         ("answer from the data with every relevant detail, friendly,
           spoken; use the scene / person only if relevant; don't
           invent; say so if the data doesn't have it")
     streamed into the voice: Luna starts speaking the first
     sentence while the rest is still being written

  3. No LLM (Ollama down): the closest knowledge-base entry
```

Voice in to voice out:

```
mic -> audio_preprocess (high-pass, gate, AGC)
    -> Vosk (live: end of utterance, wake word)
    -> Parakeet re-reads the utterance (STT_ENGINE 1, most accurate)
    -> addressing gate ("was it meant for Luna?")
    -> brain.process()
         -> instant paths | ROUTE -> (REPLY | AGAIN | IGNORE | ANSWER)
    -> Piper TTS, sentence by sentence, while the LLM writes
    -> speaker
```

### Why this is fast on the Pi 5

- **One cached prefix.** Persona + catalog + rules form the system prompt.
  It is identical for every route and answer call, and Ollama keeps its KV
  cache. It is pre-read once at boot (`FETCH_PRIME_AT_BOOT`), so a question
  only costs its own new tokens.
- **Short route output.** `PAGES: a, b` is about 8 tokens, and generation
  is cut the moment the line ends.
- **One call for chat.** Greetings, jokes and general questions are
  answered by the route call itself (`REPLY:`), streamed straight to the
  voice.
- **Compact pages.** Indexing removes menus, links and repetition, so the
  data the Pi has to read per answer is small. `FETCH_DATA_WORDS` caps it.
- **No embeddings and no downloads at question time.** Choosing data is
  the LLM's route plus plain word matching inside oversized pages.
- **Filler.** If the first word is slow, Luna says "Hmm, let me think."
  while the model keeps working.
- **Less to read.** `FETCH_DATA_WORDS` = 250, picked by section and line
  (sub-headings are sections too), the first page getting the biggest share.
- **No LLM at all** for FAQ matches and repeat questions (answer cache).
- **No route call** when the pages are obvious.
- **The CPU for the LLM.** Vision drops to `VISION_FPS_THINKING` (1 fps)
  while a prompt is read.
- **Speaking sooner.** The first chunk of speech may end at a comma once it
  is `TTS_FIRST_CLAUSE_CHARS` (25) long.

Tried and measured: a 1B route model (`FETCH_ROUTE_MODEL`) routed 6 of 12
test questions right against 11 of 12 for the 3B, and stayed silent on two
real questions, so it is off. Q4_0 weights (faster on ARM CPUs) must be
measured on the Pi:
`python fetch.py --model llama3.2:3b-instruct-q4_0 "who is the hod"`.

### The original idea (as first written)

> Indexing(): for each link fetch all data (text + PDF + public posts), one
> .txt per page (meaningful, redundant ones merged), let Llama organise each
> .txt (minimum tokens, nothing valuable lost, no links), and keep a dict
> {page: description}.
>
> Data Fetch(): query (fast, accurate speech-to-text) -> Llama gets
> {page: description} + query + context + "the person asking is {name}" +
> "return ALL pages that may hold the answer, or answer directly (jokes,
> poems, general questions, no info), 0 for noise, ask again if misheard,
> correct it if >50% sure" -> data of those pages + query + person + scene
> context + "answer in detail from the data, friendly and structured; use
> scene / person only if relevant". Instructions configurable in config.
