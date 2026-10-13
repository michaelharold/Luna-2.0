#!/usr/bin/env bash
# download_models.sh — fetch every model Luna needs, ONCE. After this Luna
# runs fully offline, forever. Safe to re-run: existing files are skipped.
#
#   ./scripts/download_models.sh          # everything missing
#   FORCE=1 ./scripts/download_models.sh  # re-download everything
#
# What it fetches (all local files, no runtime services):
#   models/vision/  YuNet face detector + SFace recogniser (opencv_zoo)
#                   + YOLOv8n object detector (for "what do you see?")
#   models/llm/     Qwen2.5-1.5B gguf — only when LLM_BACKEND="llamacpp"
#   vosk model      speech-to-text (skipped if any vosk-model-* dir exists)
#   models/stt/     the config.STT_ENGINE re-transcription model (Parakeet /
#                   Whisper, sherpa-onnx); ALL_STT=1 fetches all three
#   sentence-transformer  cached via one python import at the end
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

fetch() {  # fetch <dest> <url>
  local dest="$1" url="$2"
  if [[ -s "$dest" && -z "${FORCE:-}" ]]; then
    echo "[models] $dest already present — skipping"
    return
  fi
  echo "[models] downloading $dest"
  mkdir -p "$(dirname "$dest")"
  curl -SL --fail --progress-bar -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

# ── Vision ────────────────────────────────────────────────────────────────────
ZOO="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
fetch models/vision/face_detection_yunet_2023mar.onnx \
      "$ZOO/face_detection_yunet/face_detection_yunet_2023mar.onnx"
fetch models/vision/face_recognition_sface_2021dec.onnx \
      "$ZOO/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# YOLOv8n: exported from ultralytics (imgsz=416, opset=12) and normally synced
# from the Mac by deploy.sh. If it's missing here, export it wherever
# ultralytics is available:  yolo export model=yolov8n.pt format=onnx imgsz=416 opset=12
if [[ ! -s models/vision/yolov8n.onnx ]]; then
  echo "[models] NOTE: models/vision/yolov8n.onnx missing — object detection"
  echo "[models] will stay off. Sync it from the Mac (deploy.sh) or export it."
fi

# ── Local LLM (llama.cpp path only — LLM_BACKEND="llamacpp") ─────────────────
# The default Ollama backend pulls its model in scripts/install_pi.sh instead.
PY="venv/bin/python"; [[ -x "$PY" ]] || PY="python3"
if [[ "$("$PY" -c 'import config; print(config.LLM_BACKEND)' 2>/dev/null)" == "llamacpp" ]]; then
  fetch models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
else
  echo "[models] LLM_BACKEND is ollama — skipping the llama.cpp gguf"
fi

# ── Vosk STT ──────────────────────────────────────────────────────────────────
if compgen -G "vosk-model-*" > /dev/null; then
  echo "[models] vosk model already present — skipping"
else
  echo "[models] downloading vosk-model-en-in-0.5 (~1 GB)"
  curl -SL --fail --progress-bar -o vosk.zip \
    "https://alphacephei.com/vosk/models/vosk-model-en-in-0.5.zip"
  unzip -q vosk.zip && rm vosk.zip
fi

# ── STT_ENGINE models (sherpa-onnx) ───────────────────────────────────────────
SHERPA="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
stt_model() {  # stt_model <engine number> <folder>
  local n="$1" dir="$2"
  if [[ -z "${ALL_STT:-}" && "$ENGINE" != "$n" ]]; then return; fi
  if [[ -d "models/stt/$dir" && -z "${FORCE:-}" ]]; then
    echo "[models] models/stt/$dir already present — skipping"
    return
  fi
  echo "[models] downloading models/stt/$dir"
  mkdir -p models/stt
  curl -SL --fail --progress-bar -o "models/stt/$dir.tar.bz2" "$SHERPA/$dir.tar.bz2"
  tar xjf "models/stt/$dir.tar.bz2" -C models/stt && rm "models/stt/$dir.tar.bz2"
  # Luna loads the int8 files only; the full-precision copies are ~1 GB
  find "models/stt/$dir" -name '*.onnx' ! -name '*.int8.onnx' -delete
}
ENGINE="$("$PY" -c 'import config; print(config.STT_ENGINE)' 2>/dev/null || echo 0)"
stt_model 1 sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8
stt_model 2 sherpa-onnx-whisper-small.en
stt_model 3 sherpa-onnx-whisper-base.en

# ── Sentence transformer cache ────────────────────────────────────────────────
PY="venv/bin/python"; [[ -x "$PY" ]] || PY="python3"
echo "[models] caching sentence-transformer (all-MiniLM-L6-v2)..."
"$PY" - <<'EOF' || echo "[models] WARNING: could not cache the sentence transformer (is it pip-installed?)"
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
print("[models] sentence-transformer cached")
EOF

echo "[models] done — Luna can now run with no internet at all."
