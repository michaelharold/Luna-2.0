#!/usr/bin/env bash
# deploy.sh — sync this Mac working copy to the Pi over SSH.
#
# Set PI and DEST once (edit below or pass as env vars), then just run:
#     ./scripts/deploy.sh            # dry run — shows what WOULD change
#     ./scripts/deploy.sh --go       # actually sync
#     ./scripts/deploy.sh --go --seed-learning
#                                    # ALSO push data/learning/ once (the
#                                    # bandit / exemplars / profiles bootstrapped
#                                    # on the Mac by scripts/train_learning.py)
#
# What is intentionally NOT synced (see excludes): the Mac venv (wrong CPU arch
# for the Pi — the Pi keeps its own), __pycache__, the Vosk model folders (too
# big / download them on the Pi instead), recorded benchmark clips, .DS_Store.
# The website index (data/site/pages + catalog.json, from the weekly
# `python site_index.py` on the Mac) IS synced — that is how the Pi gets it.
set -euo pipefail

PI="${PI:-luna}"                          # ssh alias (see ~/.ssh/config) -> luna@172.21.31.169
DEST="${DEST:-/home/luna/Luna}"           # project path on the Pi
SRC="$(cd "$(dirname "$0")/.." && pwd)/"  # this repo, with trailing slash

RSYNC_OPTS=(
  -avz --delete
  --exclude 'venv/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude 'vosk-model-*'       # model folders AND their .zip downloads — handled on the Pi
  --exclude '*.zip'
  --exclude 'sample.wav'
  --exclude '.DS_Store'
  --exclude '.git/'
  --exclude '.python-version'    # Pi-only: pins the Pi's Python, must not be deleted
  --exclude '.lgd-nfy0'          # Pi-only runtime file
  --exclude '*.premerge_bak'     # local pre-merge backups — never push to the Pi
  --exclude '*.pre_refactor_bak'
  --exclude 'models/llm/'        # ~1 GB gguf — the Pi downloads its own copy
  --exclude 'models/stt/'        # Parakeet / Whisper (0.2-0.6 GB each) — the
                                 # Pi downloads the STT_ENGINE it uses
                                 # (scripts/download_models.sh), don't rsync it
  --exclude 'data/faces/'        # the Pi's OWN memory of people — --delete
  --exclude 'data/face_log.jsonl'  # must never wipe these runtime files
  --exclude 'face_shots/'        # local screenshot output
  --exclude 'data/site/crawl.json'     # the website index IS synced (the Mac
  --exclude 'data/site/llm_cache.json' # runs site_index.py weekly): only its
                                 # page files + catalog.json; the raw crawl
                                 # and the Llama cache stay on the Mac
  --exclude 'results/'           # benchmark output (scripts/bench_llm.py)
  --exclude 'data/logs/'         # the Pi's own conversation log (and the Mac's
                                 # dev log must never land on the Pi)
  --exclude 'data/logs_sim/'     # simulated sessions (train_learning.py)
  --exclude 'data/diary/'        # the Pi's own diary
  --exclude 'data/answer_cache.json'  # the Pi's own cached answers
  --exclude 'data/learning/'     # what the Pi has learned — seeded ONCE with
                                 # --seed-learning, never overwritten after
)

GO=0
SEED=0
for arg in "$@"; do
  case "$arg" in
    --go)            GO=1 ;;
    --seed-learning) SEED=1 ;;
    *) echo "[deploy] unknown option: $arg" >&2; exit 2 ;;
  esac
done

# --seed-learning: copy data/learning/ over the Pi's copy (no --delete, so
# anything the Pi created that the Mac lacks survives). Meant for the first
# deploy after bootstrapping; afterwards the Pi's own learning is the truth.
SEED_OPTS=(-avz --exclude '*.tmp' --exclude 'pending_facts.txt')
if [[ "$SEED" == 1 && ! -d "${SRC}data/learning" ]]; then
  echo "[deploy] --seed-learning: ${SRC}data/learning does not exist — run scripts/train_learning.py first" >&2
  exit 1
fi

if [[ "$GO" == 1 ]]; then
  echo "[deploy] syncing $SRC -> $PI:$DEST"
  rsync "${RSYNC_OPTS[@]}" "$SRC" "$PI:$DEST/"
  if [[ "$SEED" == 1 ]]; then
    echo "[deploy] seeding learning state -> $PI:$DEST/data/learning/"
    ssh "$PI" "mkdir -p '$DEST/data/learning'"
    rsync "${SEED_OPTS[@]}" "${SRC}data/learning/" "$PI:$DEST/data/learning/"
  fi
  echo "[deploy] done. If Luna runs as a service:  ssh $PI 'sudo systemctl restart luna'"
else
  echo "[deploy] DRY RUN (nothing changed). Re-run with --go to apply."
  rsync "${RSYNC_OPTS[@]}" --dry-run "$SRC" "$PI:$DEST/"
  if [[ "$SEED" == 1 ]]; then
    echo "[deploy] (dry run) would seed data/learning/:"
    rsync "${SEED_OPTS[@]}" --dry-run "${SRC}data/learning/" "$PI:$DEST/data/learning/" || true
  fi
fi
