# config.py — every Luna setting, in one place, grouped by what it controls.
#
#  ┌─ CONTENTS ─────────────────────────────────────────────────────────────┐
#  │  1  RUN MODE & TERMINAL           12  CAMERA                          │
#  │  2  PERFORMANCE (Pi 5)            13  FACES: detect, recognise, enrol │
#  │  3  IDENTITY & WAKE WORDS         14  EMOTION, SCENE & SCENE MEMORY   │
#  │  4  MICROPHONE & AUDIO CLEAN-UP   15  PROACTIVE BEHAVIOUR             │
#  │  5  SPEECH-TO-TEXT                16  FACE DISPLAY & ANIMATION        │
#  │  6  "WAS THAT MEANT FOR LUNA?"    17  SCREEN CARDS, BANNERS & TOUCH   │
#  │  7  CONVERSATION FLOW             18  ARMS (SERVOS)                   │
#  │  8  VOICE (TEXT-TO-SPEECH)        19  HAND GESTURES                   │
#  │  9  LLM (Ollama / llama.cpp)      20  VOICE COMMANDS                  │
#  │ 10  WEBSITE: INDEXING (weekly)    21  CONVERSATION LOG & LEARNING     │
#  │ 11  WEBSITE: DATA FETCH + KB      22  PATHS & MODEL FILES             │
#  └────────────────────────────────────────────────────────────────────────┘
#  Search for "── 10" (etc.) to jump to a section. How the website knowledge
#  works end to end: docs/Fetch Algo.md.

import os
import platform

# Load .env if present (simple parser, no extra dependency). .env is
# git-ignored, so secrets stay out of version control.
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))


# ══ 1  RUN MODE & TERMINAL ═══════════════════════════════════════════════════
# "auto" — a Mac runs in mac mode, anything else in pi mode.
# "mac"  — force Mac-dev mode (window, USB/FaceTime camera, `say`, simulated
#          servos, osascript volume).
# "pi"   — force Pi product mode (fullscreen kiosk, CSI camera, Piper,
#          real servos, pactl volume).
# Everything platform-dependent below derives from IS_MAC / IS_PI, so the SAME
# file deploys to both machines.
RUN_MODE = "auto"
IS_MAC = (RUN_MODE == "mac") or (RUN_MODE == "auto"
                                 and platform.system() == "Darwin")
IS_PI  = not IS_MAC

# False — clean: what Luna heard, where each answer came from, what she said,
#         who she recognised, and errors. Ollama's own logs go to OLLAMA_LOG.
# True  — everything: recognition scores, LLM speed, routes, timings, servo
#         moves, start-up chatter and the libraries' own logs.
LOG_VERBOSE = False


# ══ 2  PERFORMANCE (Pi 5) ════════════════════════════════════════════════════
PI_MODEL            = 5
VISION_FPS          = 8      # vision loop frames per second
VISION_FPS_THINKING = 1      # ... while the LLM reads a prompt: the camera
                             # and the LLM share the Pi's 4 cores
GESTURE_FPS         = 6
RENDER_FPS          = 60     # face animation
DEEPFACE_EVERY      = 4      # (DeepFace only) run it every Nth vision frame
TORCH_THREADS       = 4      # CPU threads for torch, the LLM and STT engines
VISION_DETECT_SCALE = 0.5    # detect faces on a downscaled frame (0.5 = a
                             # quarter of the pixels, the biggest vision cost)


# ══ 3  IDENTITY & WAKE WORDS ═════════════════════════════════════════════════
ROBOT_NAME   = "Luna"
WAKE_WORDS   = ["hello", "luna", "hey luna"]   # any of these opens a conversation
WAKE_REPLIES = ["Yes?", "I'm listening!", "Hi! How can I help?"]  # name alone


# ══ 4  MICROPHONE & AUDIO CLEAN-UP ═══════════════════════════════════════════
# Capture through PIPEWIRE, not the raw USB device: direct-hw capture locks
# the dongle and paplay can't play Luna's reply out of it (silent replies).
# The "pipewire" PortAudio device (pipewire-alsa package) shares it, and is a
# stable NAME. Fallback: "USB PnP Sound Device" (then output must not use the
# dongle).
AUDIO_INPUT_DEVICE = None if IS_MAC else "pipewire"   # Mac: system default mic

# None = the device's native rate (recommended). The Pi's USB dongle only
# opens at 44100/48000 via PortAudio; the code resamples to 16 kHz.
MIC_SAMPLE_RATE  = None if IS_MAC else 48000
VOSK_SAMPLE_RATE = 16000     # Vosk always needs 16000 — do not change

# Each mic block is cleaned before recognition (audio_preprocess.py):
#   high-pass  — DC offset and 50/100 Hz mains hum (it clips the ADC without
#                an earth ground, see docs/PROJECT_PIPELINE.md)
#   normalise  — quiet speech brought up to a steady level
#   gate       — blocks below AUDIO_GATE_DB are silenced
AUDIO_PREPROCESS  = True
AUDIO_HIGHPASS_HZ = 100.0    # speech starts ~85 Hz, hum is 50/100 Hz
AUDIO_NORMALIZE   = True
AUDIO_TARGET_PEAK = 0.85     # loudest sample brought to this (never past it)
AUDIO_MAX_GAIN    = 8.0      # never boost more than this (x)
AUDIO_GATE_DB     = -50.0    # dBFS RMS; None disables the gate
# Noise suppression: learns steady noise (fan, AC) between words and turns it
# down. OFF: it halves errors in loud steady noise (fan 0 dB: 18-24 % ->
# 9-13 %) but makes crowd chatter worse (5 dB: ~57 % -> ~71 %). Turn on only
# where the noise is a machine, not people (scripts/bench_stt_noise.py).
AUDIO_NOISE_SUPPRESS = False
AUDIO_NOISE_FLOOR_DB = -12.0  # deepest cut; -20 dB distorted speech


# ══ 5  SPEECH-TO-TEXT ════════════════════════════════════════════════════════
# Vosk listens live (end of utterance, wake word); with STT_ENGINE 1-3 that
# engine then re-reads the utterance and its words are used. No words from
# the engine = a cough / laugh / noise -> dropped. Missing model or
# sherpa-onnx -> a start-up line and plain Vosk.
#   0  Vosk only
#   1  Parakeet-TDT 0.6B v2 (NVIDIA) — most accurate overall, fast [default]
#   2  Whisper small.en (OpenAI)     — best in very loud chatter, ~4x slower
#   3  Whisper base.en  (OpenAI)     — lighter Whisper
# Word error rate (scripts/bench_stt_noise.py, Mac M3, 72 clips):
#                 quiet   fan 0 dB  chatter 10 dB  chatter 5 dB  compute/s
#   0 Vosk         2.6%    13.8%       26.5%         56.6%        (live)
#   1 Parakeet     0.5%     4.2%        1.6%          4.8%        0.036 s
#   2 Whisper sm   1.6%     5.3%        2.1%          2.1%        0.144 s
#   3 Whisper base 1.6%     6.3%        4.2%         10.6%        0.048 s
STT_ENGINE         = 1
STT_ENGINE_THREADS = 4       # CPU threads for engines 1-3

# en-in = Indian-accented English. The full en-in-0.5 model beat the small
# ones on the Pi (RTF ~0.4) at ~1.7 GB RAM; "vosk-model-small-en-in-0.4" if
# RAM ever gets tight.
VOSK_MODEL_PATH = "vosk-model-en-in-0.5"


# ══ 6  "WAS THAT MEANT FOR LUNA?" — noise filter + addressing score ══════════
# Inside a conversation window each utterance must look meant for Luna.
# TUNING:
#   • She answers chatter / other languages -> RAISE STT_CONFIDENCE_MIN or
#     ADDR_THRESHOLD, or STT_MIN_WORDS 2 -> 3.
#   • She ignores you                         -> LOWER them.
#   • Answer ONLY when named every time       -> STT_REQUIRE_WAKE_EACH_TURN.
STT_CONFIDENCE_MIN = 0.62    # 0..1 average word confidence (legacy gate)
STT_MIN_WORDS      = 2       # fewer real words than this are ignored
STT_RESCUE_WITH_KNOWLEDGE = True   # rescue a low-confidence utterance that
STT_RESCUE_MATCH_MIN      = 0.70   # matches the knowledge base this well
STT_REQUIRE_WAKE_EACH_TURN = False # True = only wake-word utterances (plus
                                   # the reply to Luna's own question)
STT_MIN_LOUDNESS   = 0.10    # 0..1 closeness. Quieter = room noise Vosk made
                             # a word of ("laughs" 0.00; people at the mic
                             # measure 0.35-0.85) — ignored unless it names Luna
STT_UNCLEAR_CONF   = 0.60    # below this the speech is "unsure": the route
                             # step is told so and may ask to repeat
STT_ONE_WORD_CONF  = 0.85    # a lone word below this is "unsure" too

# Echo guard: something heard right after Luna spoke that matches her own
# reply is her echo, not a person. Both checks must pass.
ECHO_GUARD_WINDOW   = 8.0    # s after speaking to check for self-echo
ECHO_RUN_THRESH     = 0.5    # longest shared word-run, / heard words
ECHO_OVERLAP_THRESH = 0.6    # fraction of heard words in the reply

# The addressing score (addressing.py): each signal adds its weight; the
# utterance is accepted at ADDR_THRESHOLD. False = the plain confidence gate.
ADDRESSED_SCORING   = True
ADDR_THRESHOLD      = 1.35   # clear speech alone (ADDR_W_CONF 1.2) can't pass
ADDR_REQUIRE_SIGNAL = True   # hard rule: confidence alone never passes
ADDR_FACE_OR_NAME   = False  # True = also require a face, her name or a
                             # pending answer (strictest)
ADDR_W_CONF          = 1.2   # x recognition confidence (0..1)
ADDR_W_LOUDNESS      = 0.45  # x closeness (0..1, LOUDNESS_* below)
ADDR_W_NAME          = 0.8   # Luna's name / a wake word
ADDR_W_QUESTION      = 0.35  # question-shaped (what / where / can you ...)
ADDR_W_SECOND_PERSON = 0.25  # "you / your"
ADDR_W_KB            = 0.5   # x knowledge-base similarity
ADDR_W_FACE          = 0.3   # a face is looking at Luna
ADDR_W_CONTINUITY    = 0.7   # Luna just asked a question
LOUDNESS_NEAR_DB     = -22.0 # dBFS RMS mapped to closeness 1.0
LOUDNESS_FAR_DB      = -42.0 # ... and to 0.0


# ══ 7  CONVERSATION FLOW ═════════════════════════════════════════════════════
CONVO_TIMEOUT                = 10    # s of silence that end a conversation
CONVO_TIMEOUT_AFTER_QUESTION = 20    # ... after Luna asked a follow-up
POST_SPEAK_DELAY      = 0.45  # settle time after Luna speaks before listening
DOUBLE_FLUSH          = True  # flush the audio queue before and after it
MIC_BLOCK_AFTER_SPEAK = 0.4   # echo guard after speech ends (the mic is shut
                              # during synthesis + playback already)


# ══ 8  VOICE (TEXT-TO-SPEECH) ════════════════════════════════════════════════
TTS_RATE   = 155
TTS_PLAYER = "afplay" if IS_MAC else "paplay"

# Mac: macOS `say` (Piper + paplay are a Pi setup). Follows RUN_MODE — a
# hard-coded True once reached the Pi and made it silent.
MAC_MODE     = IS_MAC
MAC_VOICE    = None          # None = system voice; list: say -v '?'
MAC_TTS_RATE = TTS_RATE      # words per minute for `say`

# Piper (the Pi's voice)
PIPER_PATH  = "/home/luna/piper/piper/piper"
PIPER_MODEL = "/home/luna/piper/voices/en_US-lessac-medium.onnx"
PIPER_LENGTH_SCALE = 1.3     # higher = slower; 1.3 = natural assistant pace
PIPER_STREAMING    = True    # play while synthesising (falls back once to the
                             # file path for good if streaming fails)
PIPER_PERSISTENT   = True    # one piper process for the session (loading the
                             # voice takes seconds on the Pi)
TTS_WARMUP         = True    # load the voice at boot

# Speech goes out sentence by sentence while the LLM still writes.
TTS_CHUNK_MIN_CHARS = 60     # shorter chunks merge with the next
TTS_FIRST_CLAUSE_CHARS = 25  # the FIRST chunk may end at a comma this soon:
                             # Luna starts talking ~2 s earlier on the Pi
TTS_CHUNK_MAX_CHARS = 220    # longer ones split at a comma
TTS_SYNTH_TIMEOUT       = 20.0   # s per chunk before falling back
TTS_FIRST_SYNTH_TIMEOUT = 60.0   # the first one also loads the voice

# Lip sync: True plays in-process (exact timing, the Mac default); False uses
# TTS_PLAYER and estimates when sound starts (the Pi default — paplay is the
# known-good path through PipeWire).
TTS_INPROCESS_PLAYBACK = IS_MAC
TTS_PLAYER_LATENCY     = 0.2  # s, first guess at TTS_PLAYER's start delay
TTS_LIPSYNC_LEAD       = 0.05 # mouth leads the voice slightly (looks natural)


# ══ 9  LLM (Ollama / llama.cpp) ══════════════════════════════════════════════
# Fully offline. "ollama": a local Ollama server keeps the model resident and
# caches the static system prompt. "llamacpp": one .gguf in-process. Either
# way a missing backend degrades to knowledge-base answers.
LLM_ENABLED       = True
LLM_BACKEND       = "ollama"
LLM_STREAMING     = True     # speak the first sentence while it writes on
LLM_MAX_TOKENS    = 150      # per reply (chat / REPLY answers)
LLM_TEMPERATURE   = 0.3      # 0.7 invented "85% placement" next to a 90% fact
LLM_MAX_HISTORY   = 6        # earlier messages kept (the conversation context)
LLM_FILLER_AFTER  = 1.2      # s: a slow first word gets a filler meanwhile
LLM_FILLERS       = ("Hmm, let me think.", "Let me see.", "One moment.")

# Ollama
OLLAMA_HOST       = "http://127.0.0.1:11434"
OLLAMA_MODEL      = "luna"         # built by scripts/make_luna_model.py
OLLAMA_BASE_MODEL = "llama3.2:3b"  # benchmark winner (docs/llm_benchmark.md);
                                   # faster, weaker: "qwen2.5:1.5b"
OLLAMA_NUM_CTX    = 8192     # persona + knowledge base + digest (~7k tokens,
                             # read ONCE and cached) + history + query + reply.
                             # KV cache ~0.9 GB on the Pi
OLLAMA_NUM_THREAD = TORCH_THREADS
OLLAMA_KEEP_ALIVE = -1       # keep the model resident forever
OLLAMA_TIMEOUT    = 90.0     # s before a generation is abandoned
OLLAMA_AUTOSTART  = True     # start `ollama serve` if none is running
OLLAMA_LOG        = "data/logs/ollama.log"
OLLAMA_BAKE_KNOWLEDGE = False  # make_luna_model.py: put knowledge.txt in the
                               # model's SYSTEM (scored lower: 0.91 vs 0.97)

# llama.cpp (LLM_BACKEND = "llamacpp")
LLM_MODEL_PATH = "models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf"
LLM_CTX        = 2048
LLM_THREADS    = TORCH_THREADS
LLM_WARMUP     = True        # load the model at boot


# ══ 10  WEBSITE KNOWLEDGE: INDEXING (weekly, by hand) ════════════════════════
# python site_index.py — crawl the site, group pages into page files, let
# Llama organize each one, describe each one. Output: SITE_DIR/pages/*.txt +
# SITE_DIR/catalog.json. Run it on the Mac, then scripts/deploy.sh --go.
# Full description: docs/Fetch Algo.md.

# What to crawl (site_crawler.py)
SITE_URL           = "https://tkmce.ac.in/"   # "" = no website knowledge
SITE_SEEDS         = (                        # pages whose menus are the map
    "https://tkmce.ac.in/",
    "https://tkmce.ac.in/computer-science-and-engineering",
    "https://tkmce.ac.in/cse.php",
)
SITE_LINKS_FILE    = "data/website_links.txt" # hand-picked pages, always read
SITE_FOCUS_NAME    = "Computer Science and Engineering"   # Luna's department
SITE_FOCUS_SLUGS   = ("cse",)                 # URL words of its pages
SITE_SKIP_SECTIONS = ("departments",)         # menu sections not crawled
SITE_SKIP_SLUGS    = (                        # URL words of other departments
    "mech", "mechanical", "civil", "eee", "ece", "electrical", "electronics",
    "chem", "chemical", "chemistry", "ce", "arch", "architecture", "ca", "mca",
    "application", "coa", "maths", "mathematics", "phy", "physics",
    "physical", "humanities", "iic", "instrumentation",
)
SITE_MAX_PAGES       = 250
SITE_DATA_HOSTS      = ("tkmce.etlab.in",)    # ETLab serves staff, student
                                              # lists, publications and news
SITE_STUDENT_ROSTERS = True    # every student's name, batch, admission no.
SITE_PDFS            = True    # read linked PDFs (pypdf; scans need OCR)
SITE_MAX_PDFS        = 300     # per crawl, CSE pages' first
SITE_FETCH_WORKERS   = 8       # pages downloaded at once
SITE_FETCH_TIMEOUT   = 8.0     # s per page

# Where the index lives
SITE_DIR = "data/site"         # crawl.json, pages/, catalog.json, llm_cache.json

# Organizing with Llama
SITE_INDEX_MODEL     = OLLAMA_BASE_MODEL  # the plain model, not the persona
SITE_INDEX_NUM_CTX   = 8192    # room for a piece + its notes
SITE_LLM_CHUNK_WORDS = 700     # prose sent to Llama per call
SITE_MIN_FACT_KEEP   = 0.8     # notes must keep this share of the piece's
                               # numbers and names (and invent no number),
                               # else the cleaned piece is kept as it is
SITE_MIN_PAGE_WORDS  = 25      # a page file with less is dropped
SITE_PDF_MAX_WORDS   = 1500    # words read from one linked PDF
SITE_GROUP_MAX_WORDS = 12000   # input words per page file (its own pages'
                               # text and data first, then PDFs)
SITE_OTHER_DEPT_MAX_WORDS = 3000  # ... for another department's file
SITE_OTHER_DEPT_DATA = ("staff", "hod", "overview", "btech", "mtech", "mca")
                               # which ETLab sections of other departments
                               # are kept (not their publications / news)

# Grouping: each page, data section and PDF goes to the FIRST page file whose
# patterns match. A pattern is a piece of the URL (of the PDF / data source,
# else of the page), "...$" = the URL ends with it, "#key" = an ETLab data
# section whose name contains key ("#staff" -> non_teaching_staff). Other
# departments get one file each (dept_<name>), the other TKM institutions
# share one, and a page no rule matches gets a file of its own — check new
# ones with: python site_index.py --plan
SITE_PAGES = (
    # name                title (what the file is)                   patterns
    ("students", "Student lists: every student's name, batch and admission "
                 "number, and how many students each programme has",
     ("#btech", "#mtech", "#mca", "#research_scholars")),
    ("cse_people", "CSE department people: head of department, teaching "
                   "staff (faculty) and non-teaching staff",
     ("#hod", "#staff", "/faculty-cse", "/adhoc-cse", "/nts-cse", "/hod-cse",
      "/research-supervisors-cse")),
    ("cse_department", "CSE department: about, history, vision and mission",
     ("#overview", "computer-science-and-engineering", "/cse.php",
      "/about-cse", "/vision-cse")),
    ("cse_programmes", "CSE programmes: B.Tech, M.Tech and PhD in Computer "
                       "Science and Engineering",
     ("/ug-cse", "/pg-cse", "/phd-cse", "/btech-cse", "/mtech-cse")),
    ("cse_research", "CSE research: faculty publications, research labs, "
                     "research scholars, funded projects, innovations",
     ("#publications", "/research-lab-cse", "/rs-cse", "/funded-projects-cse",
      "/innovations-cse")),
    ("cse_achievements", "CSE achievements: awards and achievements of CSE "
                         "faculty and students",
     ("/faculty-achievements-cse", "/student-achievements-cse")),
    ("cse_events", "CSE news and events: department news, expert talks, "
                   "workshops, student chapters and associations",
     ("#dept_news", "#dept_events", "/expert-talks-cse", "/workshop-cse",
      "/student-chapter-cse")),
    ("cse_facilities", "CSE labs and facilities",
     ("/labs-cse", "/facilities-cse")),
    ("examinations", "Examinations: exam calendar, notifications, results, "
                     "forms, exam manual, controller of examinations",
     ("#exam", "exam")),
    ("college_news", "College news and announcements: flash news, events, "
                     "achievements, blog",
     ("#news", "news.php", "/blog.php", "nanavu", "tkmce.ac.in/$",
      "#achievements", "/achievements.php", "/awards.php")),
    ("admissions", "Admissions: B.Tech, lateral entry, M.Tech, MCA and PhD "
                   "admission rules, schedule and documents",
     ("admission", "let-btech", "let%20web", "b.tech%20schedule")),
    ("programmes", "Programmes the college offers: UG, PG and PhD "
                   "programmes and intake",
     ("ug-programmes", "pg-programmes", "phd-programmes", "phd.pdf")),
    ("academic_rules", "Academic regulations, curricula and syllabus (2022, "
                       "2023, 2024 schemes), academic policy, autonomy",
     ("regulation", "curriculum", "syllabus", "academic%20policy",
      "decentralization")),
    ("academic_calendar", "Academic calendar: semester dates and events",
     ("academic-calendar",)),
    ("policies", "College policies and rules, and the code of conduct",
     ("code-of-conduct", "college-policies")),
    ("administration", "College administration: principal, deans, heads of "
                       "all departments, governing body, councils, "
                       "committees, administrative office",
     ("/principal", "/deans", "/hods", "governing-body", "academiccouncil",
      "financecommittee", "board-of-studies", "committee",
      "administrative-office")),
    ("college_about", "About TKM College of Engineering: history, founder, "
                      "chairman, vision, mission, campus, contact",
     ("about-tkm", "tkm-founder", "vision-mission", "chairman-tkm",
      "/campus.php", "/contact.php")),
    ("placements", "Placements and training: placement cell, recruiters, "
                   "industry-institute interaction",
     ("placement", "industry-institute")),
    ("research", "College research: research centre and council, "
                 "supervisors, scholars, thrust areas, facilities, awards, "
                 "conferences",
     ("research", "international-conference")),
    ("research_projects", "Funded and government projects, seed money, "
                          "patents and the IPR cell",
     ("funded-projects", "govt-projects", "seed-money", "patent",
      "ipr-cell")),
    ("collaborations", "MoUs and collaborations with industry and "
                       "institutions",
     ("/mou", "collaboration")),
    ("facilities", "Campus facilities: central library, hostels, central "
                   "computing facility, sports, IDEA lab, FAB lab, CNC lab, "
                   "workshops",
     ("central-library", "hostel", "/ccf", "sports", "idea-lab", "/fab.php",
      "/cnc", "me-workshops")),
    ("student_life", "Student life: scholarships, forms, arts, alumni, PTA, "
                     "MOOC, internships, skill programmes, student induction, "
                     "social responsibility, ISRO outreach",
     ("scholarship", "downloads", "/arts.php", "alumni", "/pta.php", "mooc",
      "events-internship", "events-skill", "/sip", "/isr.php", "/isro.php")),
    ("quality_accreditation", "Quality and accreditation: IQAC, NAAC, NBA, "
                              "NIRF ranking, AICTE / UGC approval, audit, "
                              "ARIIA, TEQIP, NITTTR, disclosures, "
                              "development plan",
     ("iqac", "nirf", "/ugc.php", "audit", "/approval.php", "ariia", "teqip",
      "nitttr", "disclosure", "institutional_development", "strategic-plan")),
    ("dept_architecture_planning", "Department of Architecture and Planning",
     ("dap.tkmce.ac.in",)),
)
SITE_STUDENTS_PAGE    = "students"          # every student row goes here
SITE_AFFILIATES_PAGE  = "tkm_institutions"  # the other TKM institutions
# Never indexed: patterns as above, or words in a page / PDF title.
SITE_SKIP_PAGES  = ("tenders", "rti-queries", "office-proceedings")
SITE_SKIP_TITLES = ("enquiry notice", "quotation", "tender", "minutes")

# What Llama is asked. {title} = the page, {text} = one piece of it.
SITE_ORGANIZE_PROMPT = """You turn text copied from a college website page \
into compact notes that a small assistant will answer visitors' questions \
from.

Page: {title}

Rules:
- Keep EVERY fact: names, designations, numbers, dates, years, fees, \
counts, places, phone numbers, programme and course names, deadlines, \
achievements, awards, rankings.
- Remove menus, navigation words, "read more", links, image captions and \
sentences that carry no fact.
- Write short "- " lines, one fact per line, in plain phrases. Group \
related lines under short "## " headings.
- Do not add, guess or explain anything that is not in the text. Do not \
summarise facts away.
- If the text has no useful fact at all, reply with exactly: NONE

Text:
{text}

Notes:"""

# {title} = the page file, {headings} = its sections, {text} = its start.
SITE_DESCRIBE_PROMPT = """These notes are one file of a college \
assistant's library.
File: {title}
Sections: {headings}

{text}

Write ONE line, at most 25 words, saying what facts this file holds, so \
someone can tell from the line alone whether a question can be answered \
from it. Name the main kinds of facts and key topics. Start with the \
topic itself, like "CSE faculty: names, designations ...", not with "This \
file". Reply with the line only."""


# The DIGEST: the key facts of every page file in ONE compact file
# (data/site/digest.txt). FETCH_MODE "digest" sends it with every question,
# in the cached system prompt. Words per page file: the longest matching
# name prefix wins; the whole digest is cut at SITE_DIGEST_MAX_WORDS.
SITE_DIGEST_WORDS     = {"cse_people": 400, "cse_": 160, "dept_": 40,
                         "students": 80, "": 80}
SITE_DIGEST_MAX_WORDS = 3500
SITE_DIGEST_PROMPT = """These notes are one file of a college assistant's \
library: {title}

{text}

Write the key facts from these notes that visitors are most likely to ask \
about, in at most {words} words, as short "- " lines. Keep every name with \
its role, and the important numbers, dates and places exactly as written. \
Nothing that is not in the notes. Reply with the lines only."""

# The FAQ: questions each page file answers, answered once at indexing
# time. A visitor's question that closely matches one is answered at once,
# with no LLM call (like knowledge.txt). 0 = no FAQ.
SITE_FAQ_PER_PAGE = 8
SITE_FAQ_PROMPT = """These notes are one file of a college assistant's \
library: {title}

{text}

Write the {n} questions visitors to the college are most likely to ask \
that these notes answer, each with a short, friendly spoken answer (one or \
two sentences) taken ONLY from the notes — no lists, no markdown. Use \
exactly this form, one pair per two lines:
Q: question
A: answer"""


# ══ 11  WEBSITE KNOWLEDGE: DATA FETCH (every question) + KNOWLEDGE BASE ══════
# brain.py -> fetch.py. ROUTE (LLM call 1): persona + LIBRARY (page: one
# line) + FETCH_SYSTEM_RULES as the system prompt (identical on every call,
# so the Pi reads it once), FETCH_ROUTE_TASK as the message -> PAGES /
# REPLY / AGAIN / IGNORE. ANSWER (LLM call 2): FETCH_ANSWER_TASK with the
# data of those pages. Full description: docs/Fetch Algo.md.
FETCH_ENABLED        = True
FETCH_MODE           = "digest"  # "digest": ONE call — the digest of every
                                 # page is in the cached system prompt, the
                                 # lines that match the query's names / rare
                                 # words are added to it. "route": two calls
                                 # (choose pages, then answer from them)
FETCH_SECTION_WORDS  = 120     # a college question with no distinctive
                               # matching line gets the best section of the
                               # best page instead (0 = never)
FETCH_LOOKUP_LINES   = 6       # matching lines from the full page files
                               # added to a question (misheard names too:
                               # "rishma sheikh" -> Dr. Reshma Sheik)
FETCH_MAX_PAGES      = 3       # page files one answer may read
FETCH_DATA_WORDS     = 250     # page data per answer (~1.4 tokens a word):
                               # the sections / lines that match the query.
                               # The Pi reads every word before the first one
                               # is spoken — the biggest speed knob (700 was
                               # ~3x slower)
FETCH_DIRECT         = True    # skip the ROUTE call when the pages are
                               # obvious: "who is <a name in the index>", or
                               # one page file named by the query ("placements")
FETCH_ROUTE_MODEL    = None    # a smaller model just for ROUTE, e.g.
                               # "llama3.2:1b" (~3x faster, +1.3 GB RAM; it
                               # keeps its own cached system prompt). None =
                               # the answer model
FETCH_CACHE          = True    # remember answers from page data: the same
FETCH_CACHE_DAYS     = 7       # question again (similarity >= FETCH_CACHE_SIM)
FETCH_CACHE_SIM      = 0.92    # is answered at once; a new index clears it
FETCH_CACHE_PATH     = "data/answer_cache.json"
FETCH_ANSWER_MAX_TOKENS = 260  # an answer from data may be longer than chat
FETCH_PRIME_AT_BOOT  = True    # read the system prompt once at boot, so the
                               # first visitor doesn't wait for it
# A REPLY (the model's own knowledge) is never allowed for a query with one
# of these words, or that matches the knowledge base: the pages whose names /
# descriptions share its words are answered from instead — a 3B model once
# REPLY'd "who are the teachers in cse" with invented names.
FETCH_COLLEGE_WORDS = (
    "college", "tkm", "tkmce", "campus", "department", "cse", "faculty",
    "teacher", "teachers", "professor", "hod", "principal", "dean", "staff",
    "student", "students", "admission", "admissions", "fee", "fees", "exam",
    "exams", "result", "syllabus", "semester", "hostel", "library", "lab",
    "labs", "placement", "placements", "course", "courses", "programme",
    "program", "btech", "mtech", "phd", "mca", "research", "scholarship",
    "club", "event", "events", "canteen", "bus", "timetable", "calendar")
FETCH_REPEAT_REPLIES = ("Sorry, I didn't quite catch that. Could you say "
                        "it again?",
                        "I missed some of that. Could you say it once more?")

# The knowledge base — curated "question: answer" lines. Also a page ("kb")
# in the LIBRARY, read live, so edits work without re-indexing.
KNOWLEDGE_PATH       = "data/knowledge.txt"
FETCH_KB_PAGE        = "kb"
FETCH_KB_DESCRIPTION = ("Curated facts: the college's name, location, "
                        "founding, principal, accreditation and ranking; the "
                        "CSE department's vision, mission, HOD and faculty "
                        "profiles")
ANSWER_OFFLINE_FIRST = True    # a very close knowledge-base match is said
LOCAL_ANSWER_THRESH  = 0.70    # verbatim at once, no LLM (1.1 = never)
LOCAL_MATCH_THRESH   = 0.55    # closest-entry fallback when the LLM is down;
                               # also a "kb" routing hint above this

# FETCH_MODE "digest": the system prompt after the persona + knowledge base
# ({digest} = data/site/digest.txt), and the message for each question
# ({query}; {lines} = matching lines from the pages; {notes} as below).
FETCH_DIGEST_RULES = """COLLEGE FACTS FROM THE WEBSITE (the key facts of \
every page; quote them, never contradict them):
{digest}

HOW TO ANSWER
- Answer college questions only from the knowledge base, these facts and \
the matching lines given with the question. If none of them has the \
answer, say you do not have that information yet and suggest the \
department office. Never make up names, numbers or dates, and never \
guess what is "usual" or "typical" (months, fees, counts).
- Say names as the facts spell them, even when the query misheard them.
- The matching lines are public facts from the college website: answer \
from them (a student's programme, batch and admission number, a staff \
member's role). A query that is only a name means "who is this?".
- Do not introduce yourself unless you are asked or greeted.
- The query comes from speech recognition and may have misheard words, \
names most of all ("rishma sheikh" for Reshma Sheik): if you are more than \
half sure what was meant, answer that.
- If the query is only noise or random words, reply with just: IGNORE
- If it is too garbled to understand, reply with just: AGAIN"""

FETCH_DIGEST_TASK = """Query: "{query}"
{lines}
{notes}"""

# FETCH_MODE "route" — the system prompt after the persona. {library} = "page: description" per
# line, {max_pages} = FETCH_MAX_PAGES.
FETCH_SYSTEM_RULES = """LIBRARY — pages of college information you can \
read (page: what it holds):
{library}

You get one of two tasks.

TASK: ROUTE — decide how to handle the visitor's query. Write ONE line, \
in one of these four forms and nothing else:
PAGES: page, page
REPLY: your whole spoken answer, on the same line
AGAIN
IGNORE
Use PAGES when the query needs facts about the college: list every page \
that may hold the answer, most useful first, at most {max_pages}, names \
copied exactly from the LIBRARY. For a person, think of every page about \
people like them (staff lists, research, achievements, student lists).
Use REPLY for greetings, thanks, small talk, jokes, poems, general \
knowledge, questions about you or about who is around you, and for a \
college question no page covers (then say you do not have that \
information yet). A query that names nothing about the college — no \
department, person, programme, exam, fee, facility or event of it — is \
general knowledge: REPLY.
Use AGAIN when the query is too garbled to understand.
Use IGNORE when the query is only noise or random words.
The query comes from speech recognition and may have misheard words: if \
you are more than half sure what was meant, treat it as that.
Examples:
"who is the head of the cse department" -> PAGES: cse_people, kb
"who are the teachers in cse" -> PAGES: cse_people
"when do b tech admissions start" -> PAGES: admissions
"tell me a joke" -> REPLY: Why did the robot go back to school? Its \
skills were getting a little rusty!
"what is photosynthesis" -> REPLY: It is how plants turn sunlight, water \
and air into their food.
"what is the capital of japan" -> REPLY: Tokyo is the capital of Japan.
"is there a hostel for boys" -> PAGES: facilities
"who is the principle of the collage" -> PAGES: administration, kb
"the uh gina blah" -> IGNORE
"what is the fees for the ah some thing" -> AGAIN

TASK: ANSWER — the message gives the query, the data from the pages and \
the instructions. Follow them."""

# LLM call 1. {query}; {notes} = who is asking, scene, "speech recognition
# was unsure", name hints, reply style — each only when there is one.
FETCH_ROUTE_TASK = """TASK: ROUTE
Query: "{query}"
{notes}
Your one line:"""

# LLM call 2. {query}; {data} = the chosen pages; {notes} as above.
FETCH_ANSWER_TASK = """TASK: ANSWER
Query: "{query}"

Data:
{data}

{notes}
Instructions: Answer the query "{query}" from the data above. Give every \
detail of the data that answers it, as a friendly, well-ordered spoken \
answer: no lists, no markdown, no links. If the data does not answer it, \
say you do not have that information yet and suggest asking at the \
department office; do not talk about other things in the data instead. \
If the query is a general question, not about the college, answer it from \
your own knowledge. \
Mention the person asking or the scene only if they matter to the query. \
The query may have misheard words: answer what was most likely meant. \
Never make up names, numbers or dates."""


# ══ 12  CAMERA ═══════════════════════════════════════════════════════════════
#   "picam" — the Pi's CSI camera via rpicam-vid (subprocess)
#   "usb"   — USB / V4L2 webcam through OpenCV
#   "esp32" — ESP32-CAM MJPEG stream over Wi-Fi (ESP32_STREAM)
CAMERA_SOURCE = "usb" if IS_MAC else "picam"

PICAM_WIDTH  = 640           # Pi CSI camera
PICAM_HEIGHT = 480
PICAM_FPS    = 15
PICAM_HFLIP  = False         # True if the image is mirrored
PICAM_VFLIP  = False         # True if mounted upside down

CAMERA_ID    = 0             # USB webcam
FRAME_WIDTH  = 320
FRAME_HEIGHT = 240
FPS          = 30

ESP32_STREAM = "http://10.240.252.155/stream"


# ══ 13  FACES: DETECT, RECOGNISE, REGISTER ═══════════════════════════════════
# YuNet detects, SFace recognises. Everything stays on the device:
# embeddings in data/faces/, sightings in data/face_log.jsonl.
ENABLE_FACE_RECOGNITION = True
FACE_DETECTOR      = "yunet"   # "yunet" (landmarks) | "haar" (fallback)
FACE_DETECT_CONF   = 0.80      # YuNet score threshold
FACE_MATCH_THRESH  = 0.36      # SFace cosine: RAISE if strangers get named,
                               # LOWER if registered people aren't
RECOGNIZE_EVERY    = 3         # embed every Nth vision frame
FACE_MAX_TEMPLATES = 6         # looks per person (day, night, glasses ...)

# Learning a new look by itself, only when VERY sure it is you: recognised
# steadily for FACE_LEARN_STABLE_SECS at >= FACE_LEARN_MIN_SIM but below
# FACE_LEARN_NOVEL_BELOW, FACE_LEARN_MARGIN clear of everyone else, one face in
# view, at most one look per FACE_LEARN_EVERY_SECS (average of
# FACE_LEARN_SAMPLES sightings).
FACE_AUTO_LEARN        = True
FACE_LEARN_MIN_SIM     = 0.65
FACE_LEARN_NOVEL_BELOW = 0.80
FACE_LEARN_MARGIN      = 0.15
FACE_LEARN_STABLE_SECS = 10.0
FACE_LEARN_EVERY_SECS  = 600.0
FACE_LEARN_SAMPLES     = 5

FACE_LOG_ENABLED   = True
FACE_LOG_COOLDOWN  = 60.0      # s between log entries per person

# Registration: voice ("remember me, my name is Alice", guided poses + name
# check), touch (hold the centre of the face) or photos
# (scripts/enroll_faces.py data/enroll/<Name>/*.jpg).
ENROLL_FRAMES          = 8
ENROLL_TIMEOUT         = 12.0
ENROLL_GUIDED_POSES    = True
ENROLL_POSES = (
    "Look straight at me.",
    "Now turn a little to your left.",
    "And a little to your right.",
)
ENROLL_FRAMES_PER_POSE = 4
ENROLL_MIN_FRAMES      = 4     # store if at least this many samples were good
ENROLL_CONFIRM_NAME    = True  # "Did you say Alice?"
ENROLL_PREVIEW         = True  # live mirrored camera view while registering
ENROLL_PREVIEW_SECS    = 40.0
TOUCH_ENROLL           = True
TOUCH_ENROLL_HOLD_SECS = 1.2


# ══ 14  EMOTION, SCENE & SCENE MEMORY ════════════════════════════════════════
EMOTION_HISTORY_LEN   = 5
SAD_CONFIDENCE        = 55.0
EMOTION_ONSET_SECS    = 0.8    # a mood must be steady this long before the
EMOTION_MIN_HOLD_SECS = 2.5    # face wears it, and is worn at least this long
ENABLE_DEEPFACE       = False  # TensorFlow, 1-2 GB RAM, only better "sad"

# "What do you see?": a live scene description (+ YOLOv8n objects on demand)
# answered through the LLM.
ENABLE_SCENE_CONTEXT    = True
ENABLE_OBJECT_DETECTION = True
YOLO_INPUT_SIZE = 416          # must match the exported ONNX imgsz
YOLO_CONF       = 0.35
YOLO_NMS        = 0.45
VISION_QUERY_PHRASES = (       # substrings that make it a camera question
    "what do you see", "what can you see", "what are you seeing",
    "who am i", "who do you see", "do you see", "can you see",
    "how do i look", "how am i looking", "what am i holding",
    "what is in front of you", "what's in front of you",
    "describe what you see", "look at me", "what am i wearing",
)

# Scene memory: who arrived / left, moods, objects — the "scene context" for
# questions about who is or was around, and conversation starters.
ENABLE_SCENE_MEMORY    = True
SCENE_MEMORY_EVENTS    = 12     # recent events kept
SCENE_MEMORY_MAX_AGE   = 1800   # s, older events are forgotten
SCENE_MEMORY_IN_PROMPT = True
SCENE_LINGER_SECS      = 8.0    # a known person standing quietly this long
                                # gets a conversation starter


# ══ 15  PROACTIVE BEHAVIOUR & THE RECOGNITION MOMENT ═════════════════════════
# Luna greets people she knows, smiles back, checks in on a sad face — all
# cooldown-limited, never during speech or an open conversation.
ENABLE_PROACTIVE           = False
PROACTIVE_GREET_RECOGNIZED = True
GREET_COOLDOWN_SECS        = 120.0  # one greeting per visit
PERSON_AWAY_SECS           = 60.0   # out of sight longer = they left
RECOG_SWITCH_SECS          = 3.0    # camera must disagree this long to
                                    # replace a recognised name
PROACTIVE_SMILE_REACT      = True
SMILE_REACT_COOLDOWN       = 120.0
PROACTIVE_SAD_REACT        = True
SAD_REACT_COOLDOWN         = 240.0
PROACTIVE_POLL_SECS        = 0.5
PROACTIVE_STABLE_SECS      = 2.0    # an emotion must last this long
PROACTIVE_CONVERSATION_STARTERS = True

GREET_PHRASES = (                   # {name} is filled in
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
CONVERSATION_STARTERS = (
    "So {name}, what brings you to the department today?",
    "{name}, anything I can help you find today?",
    "How is your day going so far, {name}?",
)

# The second Luna realises she KNOWS you: the "recognised" face, an eye
# double-take, the name sliding up, and the conversation opens.
RECOG_ANIMATION        = True
RECOG_EXPRESSION_SECS  = 4.0
RECOG_DOUBLE_TAKE      = True
RECOG_BANNER_SLIDE     = True
RECOG_GREET_OPENS_CONVERSATION = True
RECOG_EXPRESSION       = "excited"  # the face worn with "Hi {name}!"
RECOG_ON_SIGHT         = True       # also when not greeted
RECOG_SIGHT_COOLDOWN   = 60.0
RECOG_DEDUP_SECS       = 30.0       # one recognition moment per arrival
WELCOME_BACK_SECS      = 900        # left less than this ago -> welcome back
                                    # (must exceed GREET_COOLDOWN_SECS)
WELCOME_BACK_PHRASES = (
    "Welcome back, {name}!",
    "Back again, {name}? Nice to see you.",
    "Hey {name}, that was quick! Did you forget something?",
)


# ══ 16  FACE DISPLAY & ANIMATION ═════════════════════════════════════════════
# 1 Luna classic (purple) · 2 Robo (cyan) · 3 Aurora (EMO-style, the product
# face). Keys 1 / 2 / 3 switch live.
FACE_STYLE         = 3
DISPLAY_FULLSCREEN = IS_PI     # kiosk on the Pi, a window on the Mac
                               # (1400x800 canvas, scaled to fit any screen)

SLEEP_AFTER_SECONDS = 15       # no face this long -> sleep (wakes at once)
SLEEP_AFTER_FRAMES  = RENDER_FPS * SLEEP_AFTER_SECONDS

MOUTH_FILLED_TALK = True       # False = the old four-shape jaw
MOUTH_MIN_DEPTH   = 8          # px at silence
MOUTH_MAX_DEPTH   = 104        # px at full volume
MOUTH_CLOSED_H    = 14
ARC_EYES_HAPPY    = True
WAKE_ARMS         = True

CONVO_LIGHT_COL        = (255, 45, 45)   # the "listening" light
CONVO_LIGHT_RADIUS     = 9
CONVO_LIGHT_MARGIN     = 34
CONVO_LIGHT_LABEL      = True            # "LISTENING" beside it
CONVO_LIGHT_LABEL_SIZE = 20

AMBIENT_MOTES      = True                # floating motes
AMBIENT_MOTE_COUNT = 18
THINKING_DOTS      = True

FACE_OVERRIDE_SECS = 4.0       # how long a gesture-triggered face lasts


# ══ 17  SCREEN CARDS, BANNERS & TOUCH ════════════════════════════════════════
# Cards: date, time, a person (only when certain, people.py), a recognised
# name.
SCREEN_CARDS     = True
CARD_FADE_SECS   = 0.35
CARD_LINGER_SECS = 3.0         # after Luna stops speaking
CARD_START_GRACE = 8.0         # a card waiting for a voice that never starts
PERSON_CARDS     = True
GREET_CARD       = True        # recognition shows the name card
GREET_CARD_DELAY = 1.0         # s, after the eye double-take
GREET_CARD_SECS  = 4.0

GREET_BANNER      = True       # "Hi <name>!" (and enrolment confirmations)
GREET_BANNER_SECS = 4.0
GREET_BANNER_SIZE = 44

ENABLE_TOUCH_REACTIONS = True  # tap an eye: boop; tap the mouth: giggle
TOUCH_ANGRY_SECS       = 4.0   # a tap elsewhere makes Luna angry this long


# ══ 18  ARMS (SERVOS) ════════════════════════════════════════════════════════
ENABLE_SERVOS   = True     # a Mac falls back to simulation by itself
SERVO_LEFT_PIN  = 17       # BCM GPIO numbers, NOT header pins
SERVO_RIGHT_PIN = 18       # the right arm waves and shakes hands
SERVO_BACKEND   = "auto"   # "auto" | "pigpio" (DMA, no jitter) | "rpigpio"
SERVO_DETACH_AT_REST = True    # stop pulsing at rest: no buzzing
SERVO_SETTLE_SECS    = 0.35
SERVO_STEP_DELAY     = 0.02
SERVO_SPEED          = 1.0     # all gestures: 2.0 = twice as fast
SERVO_MIN_PULSE_US   = 500     # full 0-180° sweep (SG90/MG90)
SERVO_MAX_PULSE_US   = 2500

ARM_REST_ANGLE = 90
ARM_UP_ANGLE   = 160
ARM_DOWN_ANGLE = 30
ARM_MOVE_SECS  = 0.45

TALK_SWAY_DEGREES = 8          # small sway of both arms while talking
TALK_SWAY_PERIOD  = 1.6

WAVE_WIGGLES        = 3
WAVE_WIGGLE_DEGREES = 14
WAVE_RAISE_SECS     = 0.5
WAVE_WIGGLE_SECS    = 0.18

HANDSHAKE_ANGLE      = 125     # arm horizontal, pointing at the visitor
HANDSHAKE_HOLD_SECS  = 2.0
HANDSHAKE_SHAKES     = 3
HANDSHAKE_DEGREES    = 12
HANDSHAKE_RAISE_SECS = 0.6
HANDSHAKE_SHAKE_SECS = 0.16


# ══ 19  HAND GESTURES (MediaPipe) ════════════════════════════════════════════
ENABLE_GESTURES = False        # unreliable yet: off (the tracker isn't started)
GESTURE_REACT_COOLDOWN      = 8.0
GESTURE_STABLE_FRAMES       = 3
GESTURE_REACT_STABLE_FRAMES = 6
THUMB_RISE_RATIO        = 0.50
THUMB_ABOVE_INDEX_RATIO = 0.35
HAND_DETECT_CONF        = 0.75
HAND_TRACK_CONF         = 0.65
HAND_MIN_HANDEDNESS     = 0.80
HAND_MIN_SPAN           = 0.13
HAND_MODEL_COMPLEXITY   = 1
GESTURE_DEBUG           = False


# ══ 20  VOICE COMMANDS ═══════════════════════════════════════════════════════
ENABLE_VOICE_COMMANDS  = True  # volume, sleep, style, registration ... (no LLM)
VOLUME_STEP            = 10    # percent per "volume up / down"
VOLUME_MIN, VOLUME_MAX = 20, 100


# ══ 21  CONVERSATION LOG & LEARNING ══════════════════════════════════════════
# One JSON line per turn in data/logs/ (heard, why accepted / ignored, the
# path, the reply, timings, rewards). Read with scripts/review_log.py.
CONVO_LOG_ENABLED        = True
CONVO_LOG_RETENTION_DAYS = 365 if IS_MAC else 45

# learning.py: rewards -> a reply-style bandit, liked replies as examples,
# per-person profiles, taught facts (approved with scripts/approve_facts.py)
# and a nightly diary. Bootstrap on the Mac (scripts/train_learning.py), deploy
# once with scripts/deploy.sh --go --seed-learning.
LEARNING_ENABLED             = True
LEARNING_STYLE_ENABLED       = True
LEARNING_EXEMPLARS_K         = 2     # liked replies shown as examples
LEARNING_EXEMPLAR_MIN_REWARD = 0.5
LEARNING_ENGAGEMENT_WINDOW   = 20.0  # s: a follow-up within this = engaged
LEARNING_REPEAT_SIM          = 0.80  # same question again = negative
LEARNING_REWARD_EXPLICIT     = 1.0   # "good answer" / "that's wrong"
LEARNING_REWARD_THUMBS       = 0.6
LEARNING_REWARD_THANKS       = 0.5
LEARNING_REWARD_ENGAGED      = 0.25
LEARNING_REWARD_REPEAT       = -0.5
LEARNING_REWARD_CORRECTION   = -1.0
LEARNING_REWARD_ABANDON      = -0.2
LEARNING_MIN_TURNS_TO_STEER  = 8     # explore styles uniformly before this
DIARY_ENABLED                = True
DIARY_WRITE_HOUR             = 22    # first idle moment after this hour


# ══ 22  PATHS & MODEL FILES ══════════════════════════════════════════════════
MODEL_PATH         = "models/emotion_raf_mobilenet_finetuned.pth"
YUNET_MODEL_PATH   = "models/vision/face_detection_yunet_2023mar.onnx"
SFACE_MODEL_PATH   = "models/vision/face_recognition_sface_2021dec.onnx"
YOLO_MODEL_PATH    = "models/vision/yolov8n.onnx"
FACE_DB_DIR        = "data/faces"
FACE_LOG_PATH      = "data/face_log.jsonl"
ENROLL_PHOTO_DIR   = "data/enroll"
CONVO_LOG_DIR      = "data/logs"
LEARNING_DIR       = "data/learning"
PENDING_FACTS_PATH = "data/learning/pending_facts.txt"
DIARY_DIR          = "data/diary"
# (KNOWLEDGE_PATH: section 11; SITE_DIR: section 10; OLLAMA_LOG: section 9)
