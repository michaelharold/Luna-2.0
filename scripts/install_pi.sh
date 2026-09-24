#!/usr/bin/env bash
# install_pi.sh — one-shot setup of Luna on a fresh Raspberry Pi 5.
# Run ON the Pi, from the project directory (/home/luna/Luna):
#
#     ./scripts/install_pi.sh
#
# Idempotent: safe to re-run after a partial failure. Needs internet ONCE —
# after it finishes, Luna runs offline indefinitely (Ollama included).
#
# Piper note: the piper binary + voice live OUTSIDE the repo (see PIPER_PATH /
# PIPER_MODEL in config.py, default /home/luna/piper/...). This script installs
# them there if missing.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "── [1/6] apt packages ────────────────────────────────────────────────"
sudo apt-get update
sudo apt-get install -y \
  python3.11 python3.11-venv python3.11-dev \
  pipewire pipewire-alsa pulseaudio-utils \
  libportaudio2 portaudio19-dev \
  rpicam-apps \
  libsdl2-2.0-0 libsdl2-ttf-2.0-0 \
  cmake build-essential pkg-config \
  curl unzip

echo "── [2/6] venv + python packages ──────────────────────────────────────"
if [[ ! -x venv/bin/python ]]; then
  python3.11 -m venv venv
fi
venv/bin/pip install --upgrade pip
# The LLM runs in Ollama (step 5), so no llama.cpp compile is needed here.
venv/bin/pip install -r requirements.txt

echo "── [3/6] piper TTS ───────────────────────────────────────────────────"
PIPER_DIR="/home/luna/piper"
if [[ ! -x "$PIPER_DIR/piper/piper" ]]; then
  mkdir -p "$PIPER_DIR"
  curl -SL --fail -o /tmp/piper.tar.gz \
    "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz"
  tar -xzf /tmp/piper.tar.gz -C "$PIPER_DIR"
  rm /tmp/piper.tar.gz
fi
VOICE_DIR="$PIPER_DIR/voices"
mkdir -p "$VOICE_DIR"
if [[ ! -s "$VOICE_DIR/en_US-lessac-medium.onnx" ]]; then
  BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
  curl -SL --fail -o "$VOICE_DIR/en_US-lessac-medium.onnx" \
    "$BASE/en_US-lessac-medium.onnx"
  curl -SL --fail -o "$VOICE_DIR/en_US-lessac-medium.onnx.json" \
    "$BASE/en_US-lessac-medium.onnx.json"
fi

echo "── [4/6] models (vision + vosk + embeddings) ───────────────────"
./scripts/download_models.sh

echo "── [5/6] Ollama — the local LLM server ───────────────────────────────"
# Ollama runs as a systemd service on the Pi and keeps the model resident, so
# Luna restarts never reload it. Still 100% offline after this step: the model
# is pulled once here. The custom "luna" model (persona + knowledge.txt baked
# in) is built from OLLAMA_BASE_MODEL by scripts/make_luna_model.py — re-run
# that script whenever data/knowledge.txt changes and you want the baked copy
# refreshed (RAG-lite retrieval already picks up edits live).
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/luna.conf >/dev/null <<'UNIT'
[Service]
# One model at a time, kept loaded forever, all 4 cores — the Pi is Luna's.
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_HOST=127.0.0.1:11434"
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now ollama
for _ in $(seq 1 30); do
  curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break
  sleep 1
done
BASE_MODEL="$(venv/bin/python -c 'import config; print(config.OLLAMA_BASE_MODEL)')"
ollama pull "$BASE_MODEL"
venv/bin/python scripts/make_luna_model.py

echo "── [6/6] sanity check ────────────────────────────────────────────────"
venv/bin/python -m py_compile ./*.py
venv/bin/python -m unittest discover tests -q
echo ""
echo "Install complete. Start Luna with:  venv/bin/python -u main.py"
echo "(or via ~/Desktop/start-luna.sh as usual)"
echo "Model benchmark on this Pi:  venv/bin/python scripts/bench_llm.py --variants baked"
