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

"Real audio begins" means AUDIBLE, not "the player was asked to play". Every
chunk is a finished wav before playback (macOS `say` included — it used to
synthesise and play in one call, after on_start had already fired, which gave
the mouth a ~1 s head start). Playback then reports when the sound will
actually be heard:
  * in-process (sounddevice, the Mac default) — the stream's own latency
  * external player (paplay on the Pi) — spawn time plus the player's start-up
    latency, self-calibrated from how long each chunk really took to play
Each chunk's loudness envelope is handed to `on_audio` with that time, so the
mouth follows the actual syllables (see lip_sync.py).

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

import lip_sync
from config import (
    PIPER_PATH,
    PIPER_MODEL,
    PIPER_LENGTH_SCALE,
    TTS_PLAYER,
    TTS_INPROCESS_PLAYBACK,
    TTS_PLAYER_LATENCY,
    PIPER_STREAMING,
    PIPER_PERSISTENT,
    TTS_CHUNK_MIN_CHARS,
    TTS_FIRST_CLAUSE_CHARS,
    TTS_CHUNK_MAX_CHARS,
    TTS_SYNTH_TIMEOUT,
    TTS_FIRST_SYNTH_TIMEOUT,
    MAC_MODE,
    MAC_VOICE,
    MAC_TTS_RATE,
)
import console

_END = object()   # sentinel: no more chunks


def _sounddevice():
    """sounddevice, imported on first use — it opens PortAudio."""
    import sounddevice
    return sounddevice


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

        if first:                            # the first words: at a comma
            clause = None                    # once long enough, not only at
            for m in _CLAUSE_END.finditer(buffer):     # the sentence's end
                if len(buffer[:m.start()].strip()) >= TTS_FIRST_CLAUSE_CHARS:
                    clause = m
                    break
            if clause is not None:
                yield buffer[:clause.start()].strip()
                buffer = buffer[clause.end():]
                first = False

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

        # Seconds from spawning the external player to the first audible
        # sample. Re-estimated after every chunk (see _calibrate).
        self._player_latency = TTS_PLAYER_LATENCY
        self._inprocess_ok = True

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

    def _play_file(self, path, audible):
        """Play a wav, calling audible(envelope, fps, at) with the wall-clock
        time `at` its first sample is heard."""
        samples, rate = lip_sync.load_wav(path)
        env = lip_sync.envelope(samples, rate)
        if TTS_INPROCESS_PLAYBACK and self._inprocess_ok:
            try:
                self._play_inprocess(samples, rate, env, audible)
                return
            except Exception as e:
                self._inprocess_ok = False
                print(f"[Piper] in-process playback unavailable ({e}) — "
                      f"using {TTS_PLAYER}")
        self._play_external(path, len(samples) / rate, env, audible)

    @staticmethod
    def _play_inprocess(samples, rate, env, audible):
        """Play through PortAudio in BLOCKING mode, which also reports how
        far behind the write the sound is heard.

        Blocking, not sd.play(): sd.play feeds the device from a Python
        callback every few milliseconds, and while the face renderer and
        vision keep the interpreter busy that callback is starved — measured:
        2 s of speech took 8 s to come out, as a stretched, juddering voice.
        Here the real-time audio thread is PortAudio's own C code and
        stream.write() runs without the GIL, so busy threads can't touch it."""
        sd = _sounddevice()
        data = samples.reshape(-1, 1)
        prime = int(rate * 0.25)          # queued before the start is reported
        with sd.OutputStream(samplerate=rate, channels=1, dtype="float32",
                             latency="high") as stream:
            t0 = time.time()
            stream.write(data[:prime])
            audible(env, lip_sync.FPS, t0 + stream.latency)
            if len(data) > prime:
                stream.write(data[prime:])

    def _play_external(self, path, duration, env, audible):
        """Play through TTS_PLAYER. It cannot say when sound starts, so the
        start is estimated from the calibrated player latency."""
        t0 = time.time()
        proc = subprocess.Popen([TTS_PLAYER, path])
        try:
            audible(env, lip_sync.FPS, t0 + self._player_latency)
            proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
        if proc.returncode != 0:
            raise RuntimeError(f"{TTS_PLAYER} exited {proc.returncode}")
        self._calibrate(time.time() - t0, duration)

    def _calibrate(self, wall, duration):
        """paplay returns once its audio has drained, so what a chunk took
        beyond its own length is the delay before it was heard. Smoothed, and
        capped so one stall can't push the mouth a second behind the voice."""
        overhead = min(max(wall - duration, 0.0), 1.0)
        self._player_latency = 0.7 * self._player_latency + 0.3 * overhead

    # ── Per-chunk synthesis for the non-persistent modes ──────────────────────

    def _synth_file(self, text):
        """Synthesize one chunk to a temp wav with a one-shot piper."""
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="luna-tts-")
        os.close(fd)
        try:
            subprocess.run(
                [
                    PIPER_PATH,
                    "--model", PIPER_MODEL,
                    "--length_scale", str(PIPER_LENGTH_SCALE),
                    "--output_file", path,
                ],
                input=text.encode("utf-8"),
                check=True,
            )
        except Exception:
            os.remove(path)
            raise
        return path

    def _synth_mac(self, text):
        """macOS `say` rendered to a temp wav (not played) — so the Mac goes
        through the same timed, lip-synced playback as the Pi."""
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="luna-say-")
        os.close(fd)
        cmd = ["say", "-r", str(int(MAC_TTS_RATE))]
        if MAC_VOICE:
            cmd += ["-v", MAC_VOICE]
        cmd += ["-o", path, "--data-format=LEI16@22050", "--", text]
        try:
            subprocess.run(cmd, check=True)
        except Exception:
            os.remove(path)
            raise
        return path

    def _engine_dead(self, e):
        self._mode = "dead"
        if not self._tts_broken_reported:
            self._tts_broken_reported = True
            print(f"[Piper] TTS unavailable on this machine ({e}) — "
                  f"Luna will run silently. This is expected off the Pi.")

    # ── Streaming fallback (one piper process per chunk) ──────────────────────

    def _speak_streaming(self, text, audible):
        """Pipe piper raw PCM into the player. Used only if the persistent
        process isn't available on this build. The PCM is pumped through
        Python so the first bytes mark when sound starts — piper loads its
        model before any of them, and the mouth must not move through that.
        No wav, so no envelope: the mouth falls back to a synthetic one."""
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
                stdin=subprocess.PIPE,
            )
            piper.stdin.write(text.encode("utf-8"))
            piper.stdin.close()

            first = threading.Event()
            pumped = [0]

            def _pump():
                try:
                    for buf in iter(lambda: piper.stdout.read(4096), b""):
                        pumped[0] += len(buf)
                        first.set()
                        player.stdin.write(buf)
                except (BrokenPipeError, OSError):
                    pass
                finally:
                    first.set()
                    try:
                        player.stdin.close()
                    except OSError:
                        pass

            pump = threading.Thread(target=_pump, daemon=True,
                                    name="piper-pump")
            pump.start()
            first.wait(timeout=TTS_FIRST_SYNTH_TIMEOUT)
            if pumped[0]:
                audible(None, lip_sync.FPS, time.time() + self._player_latency)
            pump.join()
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

    # ── Chunk pipeline ────────────────────────────────────────────────────────
    #
    # `_prepare()` runs on the synthesis thread and `_render()` on the playback
    # thread. Every mode but "stream" synthesizes a wav ahead of playback; the
    # stream fallback synthesizes and plays in one blocking call, so it hands
    # back a closure that the playback thread runs instead. Either way the
    # playback thread is the single place that decides "audio is audible now"
    # — so on_start fires exactly once, in one code path.

    def _prepare(self, text):
        """Runs on the synthesis thread. Returns a wav path ready to play, a
        callable taking `audible` that will synthesize+play when invoked, or
        None if this chunk cannot be spoken at all."""
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

        if self._mode in ("mac", "file"):
            try:
                return (self._synth_mac(text) if self._mode == "mac"
                        else self._synth_file(text))
            except Exception as e:
                if (isinstance(e, subprocess.CalledProcessError)
                        and e.returncode is not None and e.returncode < 0):
                    return None       # killed by a signal (Ctrl+C), not broken
                self._engine_dead(e)
                return None
        if self._mode == "stream":
            return lambda audible: self._speak_streaming(text, audible)
        return None   # mode == "dead"

    def _render(self, prepared, audible):
        """Runs on the playback thread. Returns True if audio actually played.
        `audible(envelope, fps, at)` is called once sound is on its way."""
        if isinstance(prepared, str):
            try:
                self._play_file(prepared, audible)
                return True
            except Exception as e:
                print(f"[Piper] playback failed: {e}")
                return False
            finally:
                try:
                    os.remove(prepared)
                except Exception:
                    pass

        # the stream fallback synthesizes and plays in one blocking call,
        # demoting to the file path if it doesn't work on this machine
        try:
            prepared(audible)
            return True
        except Exception as e:
            if self._mode == "stream":
                self._mode = "file"
                print(f"[Piper] streaming unavailable ({e}); using file path")
            return False

    def speak_chunks(self, chunks, on_start=None, on_audio=None):
        """Speak an iterable of text chunks, blocking until the last one has
        finished playing. `chunks` may be a generator that is still producing
        text (e.g. a live Groq stream).

        In persistent mode synthesis runs one chunk ahead of playback, so the
        wait before Luna's voice is only ever the cost of the FIRST chunk.
        `on_start` is called once, when the first audio becomes audible.
        `on_audio(envelope, fps, at)` is called for every chunk that has a
        loudness envelope, with the wall-clock time `at` it becomes audible."""
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

            def audible(env, fps, at):
                nonlocal started
                if on_audio is not None and env is not None:
                    on_audio(env, fps, at)
                if not started:
                    started = True
                    # Hold "she's talking" back until the sound is actually
                    # heard; the player needs a moment to get going.
                    delay = at - time.time()
                    if delay > 0:
                        time.sleep(min(delay, 1.0))
                    if on_start is not None:
                        on_start()

            while True:
                item = pipeline.get()
                if item is _END:
                    break
                chunk, prepared = item
                if prepared is None:
                    continue
                played_any |= self._render(prepared, audible)

            worker.join(timeout=1.0)

            # No usable engine at all (dev box with neither piper nor `say`).
            # Still fire on_start and hold for roughly the length of the reply
            # so the rest of Luna's state machine behaves as it does on the Pi.
            if not played_any and self._mode == "dead":
                if on_start is not None and not started:
                    on_start()
                time.sleep(0.6)

    def speak(self, text, on_start=None, on_audio=None):
        """Speak a complete string, chunked so the first sentence starts early."""
        chunks = split_for_speech(text)
        if chunks:
            self.speak_chunks(chunks, on_start=on_start, on_audio=on_audio)

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
                console.detail("[Piper] voice model loaded and ready")
            else:
                self._kill_persistent()
                self._mode = "stream" if PIPER_STREAMING else "file"
                print(f"[Piper] persistent mode unavailable at warm-up — "
                      f"will use the {self._mode} path")

    def shutdown(self):
        self._kill_persistent()


tts = PiperTTS()
