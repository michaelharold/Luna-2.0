# shared_state.py
from threading import Lock
import time


class SharedState:

    __slots__ = (
        "lock",
        "frame",
        "face_detected", "face_x", "face_y",
        "emotion",
        "frozen_emotion",        # emotion locked while speaking
        "gesture",
        "heard_text", "response",
        "speaking",              # audio is actually coming out of the speaker —
                                 # drives the mouth animation and the arm sway
        "tts_active",            # a reply is with the TTS engine (synthesis
                                 # included). Blocks the mic, but moves nothing:
                                 # keeping this separate from `speaking` is what
                                 # stops the face animating before the voice.
        "listening",
        "audio_energy",
        "look_dir",
        "servo_action",
        "conversation_active",
        "awaiting_answer",       # Luna's last reply ended in a question, so she
                                 # expects a response and gives the person longer
                                 # to think before the conversation window closes
        "convo_expired_time",    # time.time() when conversation last timed out
        "last_activity_time",
        "mic_unblock_time",      # time.time() after which mic is allowed
        "luna_mode",             # "idle" | "listening" | "processing" | "speaking"
        "face_override",         # temporary face state ("excited" | "love" | None)
        "face_override_until",   # time.time() when the override expires
        "mic_ok",                # False when no working microphone was found
        "camera_ok",             # False when no working camera was found
        "ready",                 # False until all startup loading has finished —
                                  # gesture reactions stay quiet until then so
                                  # camera warm-up / boot-time noise can't speak
        "last_spoken_text",      # Luna's most recent utterance (lowercased)
        "last_spoken_time",      # time.time() when that utterance finished
        # ── V2: face recognition ──────────────────────────────────────────
        "person_name",           # recognised person in view (None = stranger/nobody)
        "person_conf",           # cosine similarity of that match (0..1)
        "face_count",            # how many faces the detector sees right now
        "enroll_request",        # name to enroll — set by commands, consumed by vision
        "enroll_result",         # (name, ok) once vision finishes an enrollment
        # ── V2: on-screen name banner ─────────────────────────────────────
        "banner_text",           # short text drawn under the face ("Hi Alice!")
        "banner_until",          # time.time() when the banner fades
        "banner_since",          # time.time() when the banner was posted (slide-in)
        # ── V3: recognition moment / registration / attention ─────────────
        "recog_event_time",      # time.time() of the last "I know you!" moment
        "enroll_finish",         # commands → vision: stop collecting, store now
        "enroll_pose",           # which guided pose is being captured (int)
        "touch_enroll_request",  # renderer → voice loop: face was held, enrol
        "utterance_loudness",    # 0..1 closeness of the last utterance (audio)
        "person_since",          # time.time() the current person was first seen
        "enroll_preview_until",  # time.time() until which the camera preview shows
    )

    def __init__(self):
        self.lock                = Lock()
        self.frame               = None
        self.face_detected       = False
        self.face_x              = 0.5
        self.face_y              = 0.5
        self.emotion             = "Neutral"
        self.frozen_emotion      = None
        self.gesture             = None
        self.heard_text          = ""
        self.response            = ""
        self.speaking            = False
        self.tts_active          = False
        self.listening           = False
        self.audio_energy        = 0.0
        self.look_dir            = None
        self.servo_action        = None
        self.conversation_active = False
        self.awaiting_answer     = False
        self.convo_expired_time  = 0.0
        self.last_activity_time  = 0.0
        self.mic_unblock_time    = 0.0
        self.luna_mode           = "idle"
        self.face_override       = None
        self.face_override_until = 0.0
        self.mic_ok              = True
        self.camera_ok           = True
        self.ready               = False
        self.last_spoken_text    = ""
        self.last_spoken_time    = 0.0
        self.person_name         = None
        self.person_conf         = 0.0
        self.face_count          = 0
        self.enroll_request      = None
        self.enroll_result       = None
        self.banner_text         = ""
        self.banner_until        = 0.0
        self.banner_since        = 0.0
        self.recog_event_time    = 0.0
        self.enroll_finish       = False
        self.enroll_pose         = 0
        self.touch_enroll_request = False
        self.utterance_loudness  = 0.0
        self.person_since        = 0.0
        self.enroll_preview_until = 0.0


state = SharedState()