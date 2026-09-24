"""
piper_tts.py

Piper TTS engine for Luna.

Latency
=======
The old path spawned a **fresh piper process per reply**. Every answer paid the
cost of loading en_US-lessac-medium.onnx from scratch, and only then began
synthesizing — so the first reply after boot took ~11 s to make a sound (cold
page cache + model load + full synthesis) and later replies still lagged 3-4 s
behind the face animation.

Two changes remove almost all of it:

1. **One long-lived piper process.** The model is loaded exactly once, in the
   background at startup (`warmup()`), and every reply after that is pure
   synthesis. Piper is run with `--output_dir`, which makes it behave as a
   line-oriented server: write a line of text to stdin, and it prints the path
   of the finished wav on stdout. That printed path is a clean end-of-utterance
   delimiter, which raw-stdout streaming does not give us.

2. **Chunked, pipelined playback.** A reply is split into sentence-sized chunks.
   Chunk N plays while chunk N+1 is still being synthesized, so Luna starts
   talking as soon as the *first sentence* is ready instead of after the last
   sample of the whole answer. `speak_chunks()` also accepts a generator, so the
   text can still be arriving from Groq while the opening sentence is spoken.

`on_start` is the other half of the fix: it fires the instant real audio begins,
so the caller can start the mouth/arm animation in sync with the voice instead
of when synthesis was merely requested.

Fallbacks, in order: persistent process → per-chunk streaming → per-chunk
synth-to-file. Whichever works first is remembered for the rest of the session,
so a machine where the persistent mode isn't supported never re-pays for it.
"""

import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import json

from config import (
    PIPER_PATH,
    PIPER_MODEL,
    PIPER_LENGTH_SCALE,
    TTS_PLAYER,
    PIPER_STREAMING,
    PIPER_PERSISTENT,
    TTS_CHUNK_MIN_CHARS,
    TTS_CHUNK_MAX_CHARS,
    TTS_SYNTH_TIMEOUT,
    TTS_FIRST_SYNTH_TIMEOUT,
    MAC_MODE,
    MAC_VOICE,
    MAC_TTS_RATE,
)

_END = object()   # sentinel: no more chunks


def _detect_sample_rate(model_path):
    """Read the model's sample rate from its companion .onnx.json.
    Falls back to 22050 Hz (piper medium-model default)."""
    for cfg in (model_path + ".json", model_path.replace(".onnx", ".onnx.json")):
        try:
            with open(cfg) as f:
                data = json.load(f)
            return int(data["audio"]["sample_rate"])
        except Exception:
            continue
    return 22050


# ── Chunking ──────────────────────────────────────────────────────────────────

_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"')\]]*\s+")
_CLAUSE_END   = re.compile(r"(?<=[,;:])\s+")


def _split_long(piece):
    """Break an over-long sentence at clause boundaries so the first audio
    still arrives quickly. Falls back to a hard word split."""
    if len(piece) <= TTS_CHUNK_MAX_CHARS:
        return [piece]

    out, current = [], ""
    for part in _CLAUSE_END.split(piece):
        if current and len(current) + len(part) + 1 > TTS_CHUNK_MAX_CHARS:
            out.append(current)
            current = part
        else:
            current = f"{current} {part}".strip()
    if current:
        out.append(current)

    # a single clause longer than the max (rare) — split on words
    final = []
    for part in out:
        while len(part) > TTS_CHUNK_MAX_CHARS:
            cut = part.rfind(" ", 0, TTS_CHUNK_MAX_CHARS)
            if cut <= 0:
                cut = TTS_CHUNK_MAX_CHARS
            final.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            final.append(part)
    return final


def split_for_speech(text):
    """Split a complete reply into chunks that are worth synthesizing on their
    own: whole sentences, merged up to TTS_CHUNK_MIN_CHARS so a two-word
    sentence doesn't get its own process, split down at TTS_CHUNK_MAX_CHARS."""
    text = " ".join(text.split())
    if not text:
        return []

    chunks, current = [], ""
    for sentence in _SENTENCE_END.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        current = f"{current} {sentence}".strip()
        if len(current) >= TTS_CHUNK_MIN_CHARS:
            chunks.extend(_split_long(current))
            current = ""
    if current:
        chunks.extend(_split_long(current))
    return chunks


def chunk_stream(token_iter):
    """Turn a stream of text deltas (e.g. Groq tokens) into speakable chunks,
    yielding each as soon as it is complete.

    The first chunk is released at the first sentence end regardless of length —
    getting *something* said quickly matters more than prosody at the very start
    of a reply. After that the normal minimum applies, so the rest of the answer
    isn't chopped into unnatural fragments."""
    buffer = ""
    first  = True

    for token in token_iter:
        if not token:
            continue
        buffer += token

        while True:
            # a sentence is complete once punctuation is followed by a space
            match = None
            for m in _SENTENCE_END.finditer(buffer):
                match = m
            if match is None:
                break

            head, tail = buffer[:match.start()].strip(), buffer[match.end():]
            minimum = 1 if first else TTS_CHUNK_MIN_CHARS
            if len(head) < minimum:
                break
            for piece in _split_long(head):
                yield piece
                first = False
            buffer = tail

        if len(buffer) > TTS_CHUNK_MAX_CHARS:
            pieces = _split_long(buffer)
            for piece in pieces[:-1]:
                yield piece
                first = False
            buffer = pieces[-1]

    tail = buffer.strip()
    if tail:
        for piece in _split_long(tail):
            yield piece


# ── Engine ────────────────────────────────────────────────────────────────────

class PiperTTS:

    def __init__(self):
        self.lock = threading.Lock()
        self.sample_rate = _detect_sample_rate(PIPER_MODEL)

        self._proc     = None       # the persistent piper process
        self._out_dir  = None
        self._paths    = queue.Queue()
        self._first    = True       # next synth also pays for the model load

        # None = untried; "persistent" | "stream" | "file" | "mac" | "dead"
        self._mode = "mac" if MAC_MODE else None
        self._tts_broken_reported = False

    # ── Persistent piper ──────────────────────────────────────────────────────

    def _reader(self, proc):
        """Collect the wav paths piper prints, one per synthesized line."""
        for line in iter(proc.stdout.readline, b""):
            line = line.decode("utf-8", "replace").strip()
            if line:
                self._paths.put(line)
        self._paths.put(None)   # process died

    def _start_persistent(self):
        if self._proc is not None and self._proc.poll() is None:
            return True
        if not PIPER_PERSISTENT or MAC_MODE:
            return False
        if not os.path.exists(PIPER_PATH):
            return False

        self._out_dir = tempfile.mkdtemp(prefix="luna-tts-")
        self._proc = subprocess.Popen(
            [
                PIPER_PATH,
                "--model", PIPER_MODEL,
                "--length_scale", str(PIPER_LENGTH_SCALE),
                "--output_dir", self._out_dir,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._paths = queue.Queue()
        self._first = True
        threading.Thread(target=self._reader, args=(self._proc,),
                         daemon=True, name="piper-reader").start()
        return True

    def _kill_persistent(self):
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
        if self._out_dir:
            shutil.rmtree(self._out_dir, ignore_errors=True)
            self._out_dir = None

    def _synth_persistent(self, text):
        """Synthesize one chunk, returning the wav path (or None on failure).
        Raises nothing — the caller just falls back to another mode."""
        if not self._start_persistent():
            return None

        # discard anything left over from an earlier timeout
        while True:
            try:
                self._paths.get_nowait()
            except queue.Empty:
                break

        timeout = TTS_FIRST_SYNTH_TIMEOUT if self._first else TTS_SYNTH_TIMEOUT
        try:
            self._proc.stdin.write((" ".join(text.split()) + "\n").encode("utf-8"))
            self._proc.stdin.flush()
            path = self._paths.get(timeout=timeout)
        except Exception:
            path = None

        self._first = False
        if not path or not os.path.exists(path):
            return None
        return path

    # ── Playback ──────────────────────────────────────────────────────────────

    def _play_file(self, path):
        subprocess.run([TTS_PLAYER, path], check=True)

    # ── Per-chunk fallbacks (old behaviour, still one process per chunk) ──────

    def _speak_streaming(self, text):
        """Pipe piper raw PCM straight into the player. Used only if the
        persistent process isn't available on this build."""
        piper = subprocess.Popen(
            [
                PIPER_PATH,
                "--model", PIPER_MODEL,
                "--length_scale", str(PIPER_LENGTH_SCALE),
                "--output-raw",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        player = None
        try:
            player = subprocess.Popen(
                [
                    TTS_PLAYER,
                    "--raw",
                    f"--rate={self.sample_rate}",
                    "--channels=1",
                    "--format=s16le",
                    "/dev/stdin",
                ],
                stdin=piper.stdout,
            )
            piper.stdout.close()   # let piper see SIGPIPE if the player dies
            piper.stdin.write(text.encode("utf-8"))
            piper.stdin.close()
            player.wait()
            piper.wait()
            if player.returncode != 0:
                raise RuntimeError(f"player exited {player.returncode}")
        finally:
            for p in (player, piper):
                if p is not None and p.poll() is None:
                    try:
                        p.kill()
                    except Exception:
                        pass

    def _speak_file(self, text):
        """Synthesize to a temp wav and play it. Only used as a fallback."""
        wav_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_file = f.name
            subprocess.run(
                [
                    PIPER_PATH,
                    "--model", PIPER_MODEL,
                    "--length_scale", str(PIPER_LENGTH_SCALE),
                    "--output_file", wav_file,
                ],
                input=text.encode("utf-8"),
                check=True,
            )
            self._play_file(wav_file)
        finally:
            if wav_file and os.path.exists(wav_file):
                os.remove(wav_file)

    def _speak_mac(self, text):
        """macOS `say` — blocks until playback finishes, which is what the mic
        echo guard relies on."""
        cmd = ["say", "-r", str(int(MAC_TTS_RATE))]
        if MAC_VOICE:
            cmd += ["-v", MAC_VOICE]
        cmd += ["--", text]
        subprocess.run(cmd, check=True)

    # ── Chunk pipeline ────────────────────────────────────────────────────────
    #
    # `_prepare()` runs on the synthesis thread and `_render()` on the playback
    # thread. Only the persistent mode can genuinely pre-synthesize ahead of
    # playback; the fallback modes synthesize and play in one blocking call, so
    # they hand back a closure that the playback thread runs instead. Either
    # way the playback thread is the single place that decides "audio is
    # starting now" — so on_start fires exactly once, in one code path.

    def _prepare(self, text):
        """Runs on the synthesis thread. Returns a wav path ready to play, a
        no-argument callable that will synthesize+play when invoked, or None if
        this chunk cannot be spoken at all."""
        if self._mode in (None, "persistent"):
            path = self._synth_persistent(text)
            if path is not None:
                self._mode = "persistent"
                return path
            if self._mode == "persistent":
                # it worked before, so this is a stall/crash — restart once
                print("[Piper] persistent process stopped responding — restarting")
                self._kill_persistent()
                path = self._synth_persistent(text)
                if path is not None:
                    return path
            self._kill_persistent()
            self._mode = "stream" if PIPER_STREAMING else "file"
            print(f"[Piper] persistent mode unavailable — using {self._mode} path")

        if self._mode == "mac":
            return lambda: self._speak_mac(text)
        if self._mode == "stream":
            return lambda: self._speak_streaming(text)
        if self._mode == "file":
            return lambda: self._speak_file(text)
        return None   # mode == "dead"

    def _render(self, prepared):
        """Runs on the playback thread. Returns True if audio actually played."""
        if isinstance(prepared, str):
            try:
                self._play_file(prepared)
                return True
            except Exception as e:
                print(f"[Piper] playback failed: {e}")
                return False
            finally:
                try:
                    os.remove(prepared)
                except Exception:
                    pass

        # fallback modes: synthesize and play in one blocking call, demoting to
        # the next mode if this one turns out not to work on this machine
        try:
            prepared()
            return True
        except Exception as e:
            if self._mode == "stream":
                self._mode = "file"
                print(f"[Piper] streaming unavailable ({e}); using file path")
            elif self._mode in ("file", "mac"):
                self._mode = "dead"
                if not self._tts_broken_reported:
                    self._tts_broken_reported = True
                    print(f"[Piper] TTS unavailable on this machine ({e}) — "
                          f"Luna will run silently. This is expected off the Pi.")
            return False

    def speak_chunks(self, chunks, on_start=None):
        """Speak an iterable of text chunks, blocking until the last one has
        finished playing. `chunks` may be a generator that is still producing
        text (e.g. a live Groq stream).

        In persistent mode synthesis runs one chunk ahead of playback, so the
        wait before Luna's voice is only ever the cost of the FIRST chunk.
        `on_start` is called once, immediately before the first audio plays."""
        with self.lock:
            pipeline = queue.Queue()

            def _synth_worker():
                try:
                    for chunk in chunks:
                        chunk = " ".join((chunk or "").split())
                        if not chunk:
                            continue
                        pipeline.put((chunk, self._prepare(chunk)))
                except Exception as e:
                    print(f"[Piper] synthesis stopped: {e}")
                finally:
                    pipeline.put(_END)

            worker = threading.Thread(target=_synth_worker, daemon=True,
                                      name="piper-synth")
            worker.start()

            started, played_any = False, False
            while True:
                item = pipeline.get()
                if item is _END:
                    break
                chunk, prepared = item
                if prepared is None:
                    continue
                if not started:
                    started = True
                    if on_start is not None:
                        on_start()
                played_any |= self._render(prepared)

            worker.join(timeout=1.0)

            # No usable engine at all (dev box with neither piper nor `say`).
            # Still fire on_start and hold for roughly the length of the reply
            # so the rest of Luna's state machine behaves as it does on the Pi.
            if not played_any and self._mode == "dead":
                if on_start is not None and not started:
                    on_start()
                time.sleep(0.6)

    def speak(self, text, on_start=None):
        """Speak a complete string, chunked so the first sentence starts early."""
        chunks = split_for_speech(text)
        if chunks:
            self.speak_chunks(chunks, on_start=on_start)

    # ── Warm-up ───────────────────────────────────────────────────────────────

    def warmup(self):
        """Load the voice model before the first question arrives. Synthesizes
        a short phrase and throws the audio away — nothing is played."""
        if MAC_MODE or self._mode == "dead":
            return
        with self.lock:
            path = self._synth_persistent("Ready.")
            if path:
                try:
                    os.remove(path)
                except Exception:
                    pass
                self._mode = "persistent"
                print("[Piper] voice model loaded and ready")
            else:
                self._kill_persistent()
                self._mode = "stream" if PIPER_STREAMING else "file"
                print(f"[Piper] persistent mode unavailable at warm-up — "
                      f"will use the {self._mode} path")

    def shutdown(self):
        self._kill_persistent()


tts = PiperTTS()
