# robot_face.py
"""
robot_face.py — Luna's face renderer.

This module owns the *animation*, not the *design*. It decides when to blink,
where to look, how the head tilts and which particles are in flight — but it
never decides what "happy" looks like. That lives in expressions.py, and this
file simply renders whatever Expression it is handed.

    face_style.py    palette, geometry, glow/font caches
    expressions.py   one function per mood — the file you edit
    face_particles.py every short-lived effect
    face_parts.py    Eye, Mouth, MicroExpressions
    robot_face.py    this file — timing, tracking and composition
    face.py          the callable API: happy(), sad(), shy(), sleep(), wake()

Overlays that are states of attention rather than moods, and so are layered on
top of any expression:
    listening — eye-colour pulse, raised brows
    speaking  — talking mouth driven by audio energy, expression frozen
    thinking  — three orbiting dots while the brain works

Keys on the face window:  1 / 2 / 3 switch style,  ESC or Q quits.
Touch (or click) an eye to boop it; the mouth area giggles. Press and HOLD the
middle of the face (between the eyes, above the mouth) to register your face.

The recognition moment ("I know you!", see face.recognition_moment) adds two
one-shot animations on top of the "recognised" expression: an eye double-take
(double_take_offset) and the name banner sliding up with a little pop
(slide_params). Both are pure functions of elapsed time, so they are testable
without a screen.

Performance notes (the Pi 5 renders with no GPU, ~6-8x slower than a laptop;
scripts/bench_face.py measures all of this):
  * Head tilt used to rotate the WHOLE 1400x800 canvas every frame — 60% of
    frame time. Now only a reused "face layer" is rotated: allocated once at
    the largest extent the face can ever reach (face_layer_extent) and, each
    frame, cropped to the box the face actually occupies (_content_box), at an
    angle quantised to 0.25 deg. The untilted face (the common case) is drawn
    straight onto the screen with no layer at all.
  * The layer is OPAQUE, not per-pixel alpha. Measured: alpha-blitting the
    rotated layer cost as much as rotating it; an opaque blit is ~8x cheaper.
    What made the alpha layer tempting was letting background motes show
    behind the face — instead motes inside the crop are painted into the layer
    and ones outside it are painted at their rotated position, so they tilt
    exactly as they did when the whole canvas rotated.
  * Nothing allocates a Surface per frame any more — small sprites come from
    face_style's cache, keyed by size / colour / alpha bucket.
"""

import math
import os
import random
import time
import pygame

from shared_state import state
from config import (RENDER_FPS, FACE_STYLE, DISPLAY_FULLSCREEN,
                    CONVO_LIGHT_COL, CONVO_LIGHT_RADIUS, CONVO_LIGHT_MARGIN,
                    CONVO_LIGHT_LABEL, CONVO_LIGHT_LABEL_SIZE,
                    AMBIENT_MOTES, AMBIENT_MOTE_COUNT, THINKING_DOTS,
                    ARC_EYES_HAPPY, WAKE_ARMS,
                    ENABLE_TOUCH_REACTIONS, BOOP_SECONDS,
                    GREET_BANNER, GREET_BANNER_SIZE,
                    RECOG_ANIMATION, RECOG_DOUBLE_TAKE, RECOG_BANNER_SLIDE,
                    TOUCH_ENROLL, TOUCH_ENROLL_HOLD_SECS)

import expressions
import face as face_api
from face_style import (theme, apply_style, WIDTH, HEIGHT, BG,
                        lerp, clamp, blend, get_font,
                        sprite, alpha_bucket, with_alpha)
from face_parts import Eye, Mouth, MicroExpressions
from face_particles import (AmbientMote, SoundWave, Blush,
                            SPAWNERS, BURST_COUNT)

# Establish the palette before anything draws.
apply_style(FACE_STYLE if FACE_STYLE in (1, 2, 3) else 3)

# ── Layout ────────────────────────────────────────────────────────────────────
EYE_SPREAD = 190          # px from face centre to each eye
EYE_OFFSET_Y = -30
MOUTH_OFFSET_Y = 155
BLUSH_OFFSET = (240, 80)

# ── Head tilt ─────────────────────────────────────────────────────────────────
TILT_STEP = 0.25          # deg — rotation angle quantum
TILT_DEADBAND = 0.1       # deg — at or below this the face is drawn unrotated

#: Particles drawn in front of the face, in paint order.
FRONT_PARTICLES = ("zzz", "tear", "star", "heart")


# ── Recognition moment ────────────────────────────────────────────────────────
DOUBLE_TAKE_SECS = 0.6    # whole double-take timeline
DOUBLE_TAKE_DART = 28.0   # px the pupils dart each way
DOUBLE_TAKE_POP = 0.08    # extra eye scale at the start (1.08 -> 1.0)
BANNER_SLIDE_SECS = 0.35  # banner rise + pop duration
BANNER_SLIDE_RISE = 40.0  # px below its resting place it starts from
BANNER_SLIDE_POP = 0.08   # peak extra scale mid-slide (1.0 -> 1.08 -> 1.0)
BANNER_SCALE_STEP = 0.02  # banner sprite scale bucket

# ── Touch-and-hold registration ───────────────────────────────────────────────
# The zone between the eyes and above the mouth, relative to the face centre.
ENROLL_ZONE_HALF_W = 90
ENROLL_ZONE_TOP = -100
ENROLL_ZONE_BOTTOM = 90
ENROLL_MOVE_TOLERANCE = 60   # px a held finger may wander before it's a drag
ENROLL_BANNER = "Registering\u2026 say your name"


def _smoothstep(u):
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    return u * u * (3 - 2 * u)


def double_take_offset(phase):
    """(pupil_dx, eye_scale) for the recognition double-take, `phase` 0..1
    across DOUBLE_TAKE_SECS. Outside 0..1 it is (0.0, 1.0) — no effect.

    A quick look away and back — the cartoon "wait... it's YOU!":
      0.00-0.25  (0.15 s) dart to -28 px (quick, then holds)
      0.25-0.50  (0.15 s) dart across to +28 px
      0.50-1.00  (0.30 s) ease back to centre
    while the eyes pop to 1.08x and settle to 1.0x. Added on top of the
    normal gaze AFTER its smoothing, so the dart stays sharp."""
    if not 0.0 <= phase < 1.0:
        return 0.0, 1.0
    d = DOUBLE_TAKE_DART
    if phase < 0.25:
        dx = -d * _smoothstep(phase / 0.25 * 3.0)       # arrives in ~50 ms
    elif phase < 0.5:
        dx = -d + 2 * d * _smoothstep((phase - 0.25) / 0.25 * 3.0)
    else:
        dx = d * (1.0 - _smoothstep((phase - 0.5) / 0.5))
    scale = 1.0 + DOUBLE_TAKE_POP * (1.0 - _smoothstep(phase))
    return dx, scale


def slide_params(elapsed):
    """(dy, scale, alpha) for a banner posted `elapsed` seconds ago.

    It rises from BANNER_SLIDE_RISE px below its resting place (ease-out),
    pops 1.0 -> 1.08 -> 1.0 on the way, and fades in; after
    BANNER_SLIDE_SECS it is simply (0, 1.0, 1.0). The fade-OUT at the end of
    the banner's life is separate and unchanged (see _draw_banner)."""
    if elapsed < 0.0 or elapsed >= BANNER_SLIDE_SECS:
        return 0.0, 1.0, 1.0
    u = elapsed / BANNER_SLIDE_SECS
    rise = 1.0 - (1.0 - u) ** 3                           # ease-out cubic
    dy = BANNER_SLIDE_RISE * (1.0 - rise)
    scale = 1.0 + BANNER_SLIDE_POP * math.sin(math.pi * u)
    alpha = 1.0 - (1.0 - u) ** 2
    return dy, scale, alpha


def in_enroll_zone(x, y, fcx, fcy):
    """Is (x, y) in the press-and-hold registration zone of a face centred at
    (fcx, fcy)? Between the eyes and above the mouth."""
    dx, dy = x - fcx, y - fcy
    return (abs(dx) <= ENROLL_ZONE_HALF_W
            and ENROLL_ZONE_TOP <= dy <= ENROLL_ZONE_BOTTOM)


class TouchHold:
    """Press-and-hold detector. Pure — every call is handed `now` — so the
    timing is unit-testable without pygame or a touchscreen.

    A press in the registration zone is DEFERRED rather than handled at once:
      * released before `hold_secs`  -> release() returns the press position,
        and the caller treats it as an ordinary tap (boop / giggle)
      * still down after `hold_secs` -> poll() returns True exactly once
      * dragged more than `tolerance` px -> cancelled, neither fires
    Duplicate presses while one is pending are ignored (a touchscreen can
    deliver both a FINGERDOWN and a synthesised MOUSEBUTTONDOWN)."""

    def __init__(self, hold_secs, tolerance=ENROLL_MOVE_TOLERANCE):
        self.hold_secs = hold_secs
        self.tolerance = tolerance
        self._down_at = None
        self._pos = None
        self._fired = False

    @property
    def pending(self):
        return self._down_at is not None

    def press(self, x, y, now):
        if self._down_at is None:
            self._down_at, self._pos, self._fired = now, (x, y), False

    def move(self, x, y):
        if self._down_at is not None and self._pos is not None:
            if math.hypot(x - self._pos[0], y - self._pos[1]) > self.tolerance:
                self.cancel()

    def release(self, now):
        """End the press. Returns the press position if it was a tap."""
        if self._down_at is None:
            return None
        tap = None if self._fired else self._pos
        self.cancel()
        return tap

    def poll(self, now):
        """True exactly once, when the press has been held long enough."""
        if (self._down_at is not None and not self._fired
                and now - self._down_at >= self.hold_secs):
            self._fired = True
            return True
        return False

    def cancel(self):
        self._down_at, self._pos, self._fired = None, None, False


def quantize_angle(angle, step=TILT_STEP, dead=TILT_DEADBAND):
    """Head-tilt angle actually rendered: 0.0 inside the dead band (draw flat,
    no rotation at all), else rounded to `step` degrees. Rounding changes
    nothing you can see — 0.125 deg is under a pixel at the eye — but it keeps
    the rotation stable instead of re-sampling every frame of a slow ease."""
    if abs(angle) <= dead:
        return 0.0
    return round(angle / step) * step


def face_layer_extent():
    """(left, top, right, bottom) of everything that rotates with the head,
    in px relative to the face centre (left/top negative). Pure — it reads
    only the layout constants and the expression registry.

    Deliberately the worst case, so the layer never has to be reallocated:
      eyes    — largest eye_w/eye_h x the bigger asym eye, x the waking
                stretch, x the solid (Face 3) eye's size factors, at +-EYE_SPREAD
      gaze    — pupils / the whole solid eye slide with gaze, tracking and the
                recognition double-take
      glow    — the halo drawn around each eye
      mouth   — the widest / deepest resting or talking mouth
      blush   — both cheek patches
      spawns  — where particles are born (+-260 x, -180..+120 y) plus sprite
    Particles that later drift outside this box are painted onto the screen
    at their rotated position instead (see RobotFace._paint)."""
    exprs = expressions.EXPRESSIONS.values()
    max_w = max(max(185, max(e.eye_w * max(e.asym) for e in exprs)), 1)
    max_h = max(118, max(e.eye_h * max(e.asym) for e in exprs))
    wake_w, wake_h = 1.16, 1.10          # see _eye_size / _update_wake
    solid_w, solid_h = 1.06 * 1.12, 1.30 * 1.22   # Eye._draw_solid worst case
    gaze_x = (28 + 26 + 28 + 10) * 0.55  # look_dir + expr gaze + double-take + saccade
    gaze_y = (28 + 16 + 10) * 0.45
    glow = 35                            # halo (6 * 5 px) + 8 px size bucketing

    eye_hw = max_w * wake_w * solid_w / 2
    eye_hh = max_h * wake_h * solid_h / 2
    right = max(EYE_SPREAD + eye_hw + gaze_x + glow,       # eyes
                BLUSH_OFFSET[0] + 38 + 4,                  # blush
                260 + 40,                                  # particle spawns
                150 + 20)                                  # widest mouth
    top = min(EYE_OFFSET_Y - eye_hh - gaze_y - glow,
              -180 - 40)
    bottom = max(EYE_OFFSET_Y + eye_hh + gaze_y + glow,
                 MOUTH_OFFSET_Y + 118,                     # frown / D mouth
                 BLUSH_OFFSET[1] + 38 + 4,
                 120 + 40)
    return (-int(math.ceil(right)), int(math.floor(top)),
            int(math.ceil(right)), int(math.ceil(bottom)))


def rotate_point(x, y, angle, cx=WIDTH / 2, cy=HEIGHT / 2):
    """Where (x, y) lands when the canvas is rotated `angle` degrees counter-
    clockwise about (cx, cy) — the same sense as pygame.transform.rotate."""
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    dx, dy = x - cx, y - cy
    return cx + dx * c + dy * s, cy - dx * s + dy * c


class RobotFace:
    """The whole face. Call `set_state(name)` to change mood, `draw()` per frame."""

    def __init__(self):
        pygame.init()
        pygame.font.init()

        # SCALED draws on the fixed WIDTH x HEIGHT logical canvas and lets
        # pygame scale it to the real display, preserving aspect ratio, so the
        # face fits any screen (e.g. the Pi's 800x480 panel) instead of being
        # cropped. FULLSCREEN fills the panel; drop it for a window.
        flags = pygame.SCALED | (pygame.FULLSCREEN if DISPLAY_FULLSCREEN else 0)
        # Headless verification (SDL's dummy driver) can't do SCALED/FULLSCREEN.
        if os.environ.get("SDL_VIDEODRIVER") == "dummy":
            flags = 0
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), flags)
        pygame.display.set_caption("Luna")
        self.clock = pygame.time.Clock()

        # The face layer: reused every frame, rotated only while tilted.
        # Opaque on purpose — see the module docstring.
        self._layer_box = face_layer_extent()
        l, t, r, b = self._layer_box
        self._layer = pygame.Surface((r - l, b - t)).convert()
        # Scratch for the (rare, large) wake ripple ring.
        self._ripple = pygame.Surface((604, 604), pygame.SRCALPHA)

        # ── mood ──────────────────────────────────────────────────────────
        self.state = "neutral"
        self.prev_state = "neutral"
        self.expr = expressions.get("neutral")

        # ── parts ─────────────────────────────────────────────────────────
        self.left_eye  = Eye(-EYE_SPREAD, EYE_OFFSET_Y, 175, 125)
        self.right_eye = Eye( EYE_SPREAD, EYE_OFFSET_Y, 175, 125)
        self.mouth     = Mouth(0, MOUTH_OFFSET_Y)
        self.micro     = MicroExpressions()
        self.left_blush  = Blush(-BLUSH_OFFSET[0], BLUSH_OFFSET[1])
        self.right_blush = Blush( BLUSH_OFFSET[0], BLUSH_OFFSET[1])

        # ── position ──────────────────────────────────────────────────────
        self.face_cx, self.face_cy = WIDTH / 2, HEIGHT / 2
        self.face_ox = self.face_oy = 0.0
        self.target_ox = self.target_oy = 0.0
        self.pupil_ox = self.pupil_oy = 0.0
        self.head_angle = self.target_tilt = 0.0
        self.breath_phase = 0.0

        # ── blinking ──────────────────────────────────────────────────────
        self.blink_timer = random.randint(100, 220)
        self.blink_counter = 0
        self.blink_open = 1.0
        self.blink_dir = 0

        # ── idle curiosity ────────────────────────────────────────────────
        self.curiosity_target = (0.0, 0.0)
        self.curiosity_timer = 0
        self.idle_frames = 0

        # ── saccades ──────────────────────────────────────────────────────
        # Tiny sub-glances layered onto wherever the gaze is pointing. Real
        # eyes never hold perfectly still; without these the stare reads as
        # a screensaver rather than attention.
        self.saccade = (0.0, 0.0)
        self.saccade_next = 0.0

        # ── pulses and one-shots ──────────────────────────────────────────
        self.listen_pulse = self.listen_phase = 0.0
        self.red_pulse = self.red_pulse_phase = 0.0
        self.eye_arc = 0.0
        self.bounce_phase = self.bounce_amp = 0.0
        self.wake_ripple = 0.0
        self.wake_active = False
        self._nudge = (0.0, 1.0)          # (pupil dx, eye scale) — double-take
        self._recog_seen = 0.0            # last state.recog_event_time handled
        self._double_take_at = None       # when the current double-take began
        self._hold = TouchHold(TOUCH_ENROLL_HOLD_SECS)
        self.wake_stretch = self.wake_yawn = 0.0
        self.think_phase = self.think_glow = 0.0
        self.awake_glow = self.rest_pulse = 0.0
        self._last_expire = 0.0

        # ── particles ─────────────────────────────────────────────────────
        self.motes = ([AmbientMote() for _ in range(AMBIENT_MOTE_COUNT)]
                      if AMBIENT_MOTES else [])
        self.particles = {kind: [] for kind in SPAWNERS}
        self.spawn_timers = {kind: 0 for kind in SPAWNERS}
        self.burst_done = False
        self.sound_waves = []
        self.wave_timer = 0

    # ══════════════════════════════════════════════════════════════════════
    # Mood
    # ══════════════════════════════════════════════════════════════════════
    def set_state(self, name):
        """Change expression. Safe to call every frame with the same name."""
        if name == self.state:
            return

        was_asleep = self.state == "sleeping"
        self.prev_state = self.state
        self.state = name
        self.expr = expressions.get(name)

        # A one-shot particle burst belongs to this entry, not the last one.
        self.burst_done = False

        # Waking up is its own little performance — see _update_wake().
        if was_asleep:
            self._begin_wake()

        if self.expr.bounce:
            self.bounce_phase = 0.0
            self.bounce_amp = self.expr.bounce

    def _begin_wake(self):
        self.wake_active = True
        self.wake_ripple = 0.0
        self.wake_stretch = 1.0
        self.wake_yawn = 1.0
        if WAKE_ARMS:
            self._wake_arms()

    @staticmethod
    def _wake_arms():
        """Both arms reach up as she wakes. Servos are optional hardware — a
        missing or broken servo must never block or crash the render thread."""
        try:
            from servo_module import servo
            servo.wake_up()
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════
    # Per-frame update
    # ══════════════════════════════════════════════════════════════════════
    def update(self):
        self.breath_phase += 0.045

        with state.lock:
            face_detected = state.face_detected
            face_x, face_y = state.face_x, state.face_y
            speaking  = state.speaking
            listening = state.listening
            energy    = state.audio_energy
            look_dir  = state.look_dir
            frozen    = state.frozen_emotion
            convo_active  = state.conversation_active
            convo_expired = state.convo_expired_time
            luna_mode = state.luna_mode
            recog_at  = state.recog_event_time

        # While speaking, the mood is frozen to whatever it was when she began —
        # a face that changes emotion mid-sentence reads as a glitch.
        if speaking and frozen:
            name = frozen.lower() if frozen != "Neutral" else "neutral"
            expr = expressions.get(name)
        else:
            expr = self.expr

        listening_only = listening and not speaking

        self._update_double_take(recog_at)
        self._update_idle(face_detected, speaking, expr)
        self._update_wake()
        self._update_conversation(convo_active, convo_expired)
        self._update_thinking(luna_mode)
        self._update_tracking(expr, look_dir, face_detected, face_x, face_y)
        self._update_eyes(expr, speaking, listening_only)
        self._update_mouth(expr, speaking, energy)
        self._update_particles(expr, speaking, energy)

        for mote in self.motes:
            mote.update()

        # Stash what draw() needs, so drawing reads no shared state.
        self._frame = (expr, speaking, listening_only)

    # ── recognition double-take ───────────────────────────────────────────
    def _update_double_take(self, recog_at):
        """Start the double-take when recog_event_time moves on, and turn the
        elapsed time into this frame's (dx, scale) nudge for the eyes."""
        now = time.time()
        if recog_at != self._recog_seen:
            self._recog_seen = recog_at
            # Only a fresh event: an old timestamp (e.g. at startup) is history.
            if (RECOG_ANIMATION and RECOG_DOUBLE_TAKE and recog_at > 0
                    and 0.0 <= now - recog_at < DOUBLE_TAKE_SECS):
                self._double_take_at = recog_at
        if self._double_take_at is None:
            self._nudge = (0.0, 1.0)
            return
        phase = (now - self._double_take_at) / DOUBLE_TAKE_SECS
        self._nudge = double_take_offset(phase)
        if phase >= 1.0:
            self._double_take_at = None

    @property
    def double_take_active(self):
        return self._double_take_at is not None

    # ── idle / sleep bookkeeping ──────────────────────────────────────────
    def _update_idle(self, face_detected, speaking, expr):
        """Counts frames with nobody around. face_renderer uses this to decide
        when to fall asleep; seeing a face while asleep wakes her."""
        if not face_detected and not speaking:
            self.idle_frames += 1
        else:
            if self.idle_frames > 0 and expr.name == "sleeping":
                self.set_state("neutral")
            self.idle_frames = 0

    # ── waking performance ────────────────────────────────────────────────
    def _update_wake(self):
        """The yawn fades fastest (eyes pop open) and the stretch lingers a
        beat, so the whole face rises and settles after the eyes are open."""
        if self.wake_yawn > 0.001:
            self.wake_yawn = max(0.0, self.wake_yawn - 0.045)
        if self.wake_stretch > 0.001:
            self.wake_stretch = max(0.0, self.wake_stretch - 0.018)
        if self.wake_active:
            self.wake_ripple += 0.04
            if self.wake_ripple >= 1.0:
                self.wake_active = False
                self.wake_ripple = 0.0

    # ── conversation window indicator ─────────────────────────────────────
    def _update_conversation(self, active, expired_at):
        self.awake_glow = lerp(self.awake_glow, 1.0 if active else 0.0, 0.06)
        if expired_at > self._last_expire:
            # The window just closed — play one slow dim blink as a "back to
            # wake-word mode" cue.
            self._last_expire = expired_at
            self.rest_pulse = 1.0
            self.blink_dir = -1
            self.blink_counter = 0
        self.rest_pulse *= 0.965          # ~1.5 s decay at 60 fps

    def _update_thinking(self, luna_mode):
        """Fades in while the brain works. Without this the pause between a
        question and the answer is indistinguishable from Luna not hearing it."""
        self.think_glow = lerp(self.think_glow,
                               1.0 if luna_mode == "processing" else 0.0, 0.12)
        if self.think_glow > 0.02:
            self.think_phase += 0.09

    # ── where the face and pupils point ───────────────────────────────────
    def _update_tracking(self, expr, look_dir, face_detected, fx, fy):
        if expr.name == "sleeping":
            self.target_ox = self.target_oy = 0.0
            self.pupil_ox = lerp(self.pupil_ox, 0.0, 0.05)
            self.pupil_oy = lerp(self.pupil_oy, 0.0, 0.05)
            self.target_tilt = 0.0

        elif look_dir is not None:
            # Explicitly told where to look (pointing gesture).
            shift, pupil = 60, 28
            deltas = {"left":  (-shift, 0, -pupil, 0),
                      "right": ( shift, 0,  pupil, 0),
                      "up":    (0, -shift, 0, -pupil),
                      "down":  (0,  shift, 0,  pupil)}
            dx, dy, px, py = deltas.get(look_dir, (0, 0, 0, 0))
            if dx or px:
                self.target_ox = dx
                self.pupil_ox = lerp(self.pupil_ox, px, 0.15)
            if dy or py:
                self.target_oy = dy
                self.pupil_oy = lerp(self.pupil_oy, py, 0.15)
            self.target_tilt = 0.0

        elif face_detected:
            # Follow the person, and cock the head toward them at the edges.
            self.target_ox = (fx - 0.5) * 120
            self.target_oy = (fy - 0.5) * 60
            self.pupil_ox = lerp(self.pupil_ox, self.target_ox * 0.25, 0.15)
            self.pupil_oy = lerp(self.pupil_oy, self.target_oy * 0.25, 0.15)
            self.target_tilt = 8.0 if fx < 0.38 else (-8.0 if fx > 0.62 else 0.0)

        else:
            # Nobody about — drift the gaze around slowly so she looks alive
            # rather than switched off.
            self.curiosity_timer += 1
            if self.curiosity_timer > 140:
                self.curiosity_timer = 0
                self.curiosity_target = (random.uniform(-70, 70),
                                         random.uniform(-35, 35))
            self.target_ox, self.target_oy = self.curiosity_target
            self.pupil_ox = lerp(self.pupil_ox, self.target_ox * 0.20, 0.02)
            self.pupil_oy = lerp(self.pupil_oy, self.target_oy * 0.20, 0.02)
            self.target_tilt = 0.0

        # Saccades: every second or two the pupils dart a few px off the gaze
        # point and drift back. Suppressed while asleep.
        now = time.time()
        if expr.name != "sleeping" and now > self.saccade_next:
            self.saccade_next = now + random.uniform(0.9, 2.6)
            self.saccade = (random.uniform(-9, 9), random.uniform(-5, 5))
        sx, sy = self.saccade
        self.saccade = (sx * 0.92, sy * 0.92)          # decay back to centre
        self.pupil_ox += sx * 0.10
        self.pupil_oy += sy * 0.10

        # Breathing, the entry bounce and the waking stretch all ride on top.
        bounce_y = 0.0
        if self.bounce_amp > 0.5:
            self.bounce_phase += 0.28
            bounce_y = -abs(math.sin(self.bounce_phase)) * self.bounce_amp
            self.bounce_amp *= 0.94
        breath_y = math.sin(self.breath_phase) * 5
        wake_y   = -34.0 * self.wake_stretch

        self.face_ox = lerp(self.face_ox, self.target_ox, 0.12)
        self.face_oy = lerp(self.face_oy,
                            self.target_oy + breath_y + bounce_y + wake_y, 0.16)
        self.head_angle = lerp(self.head_angle,
                               self.target_tilt + expr.tilt, 0.07)

    # ── eyes ──────────────────────────────────────────────────────────────
    def _update_eyes(self, expr, speaking, listening):
        eye_w, eye_h = self._eye_size(expr, speaking)
        blink_t = self._blink(expr, speaking)

        # '∩' eyes only for genuinely cheerful moods, and never while speaking —
        # the talking mouth carries the expression there, and arched eyes on
        # every sentence would flatten the difference between happy and neutral.
        # During the recognition double-take the eyes stay open so the dart
        # is visible, THEN close into happy arcs — "wait... it's you!".
        want_arc = (ARC_EYES_HAPPY and expr.arc_eyes and not speaking
                    and not self.double_take_active)
        self.eye_arc = lerp(self.eye_arc, 1.0 if want_arc else 0.0, 0.14)

        self.micro.update(expr, listening, speaking)

        # An expression may deliberately look away (shy). This is added on top
        # of tracking, so she still follows you — just not with her eyes.
        gaze_x, gaze_y = expr.gaze
        for eye, scale in ((self.left_eye,  expr.asym[0]),
                           (self.right_eye, expr.asym[1])):
            eye.update(int(eye_w * scale), int(eye_h * scale), blink_t,
                       self.pupil_ox + gaze_x, self.pupil_oy + gaze_y,
                       pupil_scale=self.micro.pupil_scale,
                       squint=self.micro.squint,
                       widen=self.micro.widen,
                       droop=self.micro.droop)

        self._update_pulses(expr, listening)
        self.left_blush.update(expr.blush)
        self.right_blush.update(expr.blush)

    def _eye_size(self, expr, speaking):
        w, h = (185, 118) if speaking else (expr.eye_w, expr.eye_h)

        # Waking: squeeze then pop. The yawn squashes the eyes almost shut and
        # stretches them wide, the way you screw your eyes up mid-yawn; as it
        # decays they spring open, then the stretch adds a brief oversize.
        if self.wake_yawn > 0.001:
            h *= 1.0 - 0.72 * self.wake_yawn
            w *= 1.0 + 0.10 * self.wake_yawn
        if self.wake_stretch > 0.001:
            grow = 1.0 + 0.16 * self.wake_stretch
            w, h = w * grow, h * grow
        return int(w), int(h)

    def _blink(self, expr, speaking):
        """Two-phase blink: a fast ease shut, a softer ease open.

        Returns 0-1 eyelid openness, smoothstepped. Sleeping holds them shut."""
        if expr.name == "sleeping":
            return 0.0

        self.blink_counter += 1
        trigger = 180 if speaking else self.blink_timer

        if self.blink_open >= 1.0 and self.blink_counter > trigger:
            self.blink_counter = 0
            self.blink_dir = -1

        if self.blink_dir == -1:
            self.blink_open -= 0.34                    # ~3 frames to close
            if self.blink_open <= 0.0:
                self.blink_open = 0.0
                self.blink_dir = 1
        elif self.blink_dir == 1:
            # Reopen slower just after a conversation timeout — reads as a
            # sleepy "back to rest" blink.
            self.blink_open += 0.16 * (0.45 if self.rest_pulse > 0.35 else 1.0)
            if self.blink_open >= 1.0:
                self.blink_open = 1.0
                self.blink_dir = 0
                self.blink_timer = random.randint(100, 220)
                # Occasionally double-blink — very lifelike.
                if random.random() < 0.18 and not speaking:
                    self.blink_counter = self.blink_timer - 8

        t = clamp(self.blink_open, 0.0, 1.0)
        return t * t * (3 - 2 * t)                     # smoothstep

    def _update_pulses(self, expr, listening):
        self.listen_phase += 0.08
        self.listen_pulse = lerp(
            self.listen_pulse,
            0.5 + 0.5 * math.sin(self.listen_phase) if listening else 0.0,
            0.15 if listening else 0.10)

        self.red_pulse_phase += 0.06
        self.red_pulse = lerp(
            self.red_pulse,
            0.3 + 0.3 * math.sin(self.red_pulse_phase) if expr.red_pulse else 0.0,
            0.10 if expr.red_pulse else 0.08)

    def _eye_color(self):
        """Listening pulse > awake tint > rest dim. All deliberately subtle —
        these are status cues layered on a mood, not moods of their own."""
        col = theme.EYE_OUTER
        if self.listen_pulse > 0.05:
            col = blend(col, theme.LISTEN_COL, self.listen_pulse * 0.5)
        elif self.awake_glow > 0.05:
            col = blend(col, theme.EYE_INNER, self.awake_glow * 0.22)
        if self.rest_pulse > 0.05:
            dim = self.rest_pulse * 0.45
            col = (int(col[0] * (1 - dim)), int(col[1] * (1 - dim)),
                   int(col[2] * (1 - dim)))
        return None if col == theme.EYE_OUTER else col

    # ── mouth ─────────────────────────────────────────────────────────────
    def _update_mouth(self, expr, speaking, energy):
        self.mouth.update(energy, expr, speaking)

    # ── particles ─────────────────────────────────────────────────────────
    def _update_particles(self, expr, speaking, energy):
        """Spawns exactly what the Expression asks for and nothing else.

        `(kind, 0)` means one burst on entry; `(kind, n)` means one every n
        frames while the mood lasts. Anything the current mood does not ask
        for is dropped immediately, so switching away clears the screen."""
        fcx = int(self.face_cx + self.face_ox)
        fcy = int(self.face_cy + self.face_oy)

        wanted = {kind for kind, _ in expr.particles}
        for kind, group in self.particles.items():
            if kind not in wanted:
                group.clear()
                self.spawn_timers[kind] = 0

        for kind, period in expr.particles:
            spawn = SPAWNERS[kind]
            if period == 0:
                if not self.burst_done:
                    self.particles[kind].extend(
                        spawn(fcx, fcy) for _ in range(BURST_COUNT))
            else:
                self.spawn_timers[kind] += 1
                if self.spawn_timers[kind] > period:
                    self.spawn_timers[kind] = 0
                    self.particles[kind].append(spawn(fcx, fcy))
        self.burst_done = True

        # Sound rings while she talks. They start outside the mouth outline so
        # they read as sound leaving her, not a circle drawn on her face.
        if speaking and energy > 0.45:
            self.wave_timer += 1
            if self.wave_timer > 22:
                self.wave_timer = 0
                self.sound_waves.append(
                    SoundWave(fcx + self.mouth.rel_x, fcy + self.mouth.rel_y,
                              self.mouth.mouth_w * 0.5 + 26))

        for group in (*self.particles.values(), self.sound_waves):
            for p in group:
                p.update()
        for kind in self.particles:
            self.particles[kind] = [p for p in self.particles[kind] if p.alive()]
        self.sound_waves = [w for w in self.sound_waves if w.alive()]

    # ══════════════════════════════════════════════════════════════════════
    # Drawing
    # ══════════════════════════════════════════════════════════════════════
    def draw(self):
        self._handle_events()
        self.clock.tick(RENDER_FPS)
        self.update()

        expr, speaking, listening = self._frame
        screen = self.screen
        screen.fill(BG)

        fcx = int(self.face_cx + self.face_ox)
        fcy = int(self.face_cy + self.face_oy)

        self._draw_background(screen, expr)
        angle = quantize_angle(self.head_angle)
        if angle == 0.0:
            # Upright: straight onto the screen, no layer, no rotation.
            self._draw_face(screen, fcx, fcy, expr, speaking, listening)
            self._paint_front(screen)
            self._draw_wake_ripple(screen, fcx, fcy)
        else:
            self._draw_tilted(angle, fcx, fcy, expr, speaking, listening)

        # Status overlays are drawn AFTER the rotation, deliberately: a status
        # light that swings around with the head is hard to read at a glance,
        # and the whole point of it is "am I still listening?".
        self._draw_convo_light()
        self._draw_thinking()
        self._draw_banner()

        pygame.display.flip()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit          # main.py's finally block cleans up
            if event.type == pygame.KEYDOWN:
                # ESC / Q are the reliable way out of a fullscreen face — there
                # is no window close button in kiosk mode.
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    raise SystemExit
                if event.key in (pygame.K_1, pygame.K_KP1):
                    apply_style(1)
                elif event.key in (pygame.K_2, pygame.K_KP2):
                    apply_style(2)
                elif event.key in (pygame.K_3, pygame.K_KP3):
                    apply_style(3)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # SCALED delivers mouse positions in LOGICAL canvas coords.
                self._press(*event.pos)
            elif event.type == pygame.FINGERDOWN:
                # Touchscreens report normalised 0..1 coords.
                self._press(event.x * WIDTH, event.y * HEIGHT)
            elif event.type in (pygame.MOUSEBUTTONUP, pygame.FINGERUP):
                tap = self._hold.release(time.time())
                if tap is not None:
                    self._handle_touch(*tap)
            elif event.type == pygame.MOUSEMOTION and self._hold.pending:
                self._hold.move(*event.pos)
            elif event.type == pygame.FINGERMOTION and self._hold.pending:
                self._hold.move(event.x * WIDTH, event.y * HEIGHT)
        if self._hold.pending and self._hold.poll(time.time()):
            self._touch_enroll()

    def _press(self, x, y):
        """A press in the middle of the face might be the start of a hold, so
        its tap is deferred to the release; anywhere else reacts at once."""
        if (TOUCH_ENROLL and self.state != "sleeping"
                and in_enroll_zone(x, y, self.face_cx + self.face_ox,
                                   self.face_cy + self.face_oy)):
            self._hold.press(x, y, time.time())
        else:
            self._handle_touch(x, y)

    def _touch_enroll(self):
        """The face was held: ask the voice loop to run registration. Only
        flags it — the render thread never speaks or blocks."""
        try:
            if self.state == "sleeping":
                return
            with state.lock:
                busy = state.speaking or state.tts_active
                if not busy:
                    state.touch_enroll_request = True
            if busy:
                return
            print("[face] Face held — touch registration requested")
            face_api.curious(2)
            face_api.banner(ENROLL_BANNER)
        except Exception as e:                     # never kill the face
            print(f"[face] touch enroll failed: {e}")

    def _handle_touch(self, x, y):
        """Tapping an eye boops it (that eye pops); the mouth area giggles.
        Uses the face override channel so it wins over vision for a moment,
        then hands the face straight back."""
        if not ENABLE_TOUCH_REACTIONS:
            return
        with state.lock:
            speaking = state.speaking or state.tts_active
        if speaking or self.state == "sleeping":
            return                        # never stomp on talking / sleeping

        fcx = self.face_cx + self.face_ox
        fcy = self.face_cy + self.face_oy
        reach = 150                       # generous hit circle around each part
        if math.hypot(x - (fcx + self.left_eye.rel_x),
                      y - (fcy + self.left_eye.rel_y)) < reach:
            face_api.show("boop_left", BOOP_SECONDS)
        elif math.hypot(x - (fcx + self.right_eye.rel_x),
                        y - (fcy + self.right_eye.rel_y)) < reach:
            face_api.show("boop_right", BOOP_SECONDS)
        elif math.hypot(x - (fcx + self.mouth.rel_x),
                        y - (fcy + self.mouth.rel_y)) < reach:
            face_api.show("giggle", BOOP_SECONDS + 0.6)

    def _draw_background(self, surf, expr):
        # Motes are dimmed while sleeping so they read as a calm night sky
        # rather than competing with the z's.
        if self.motes:
            tint = theme.ZZZ_COL if expr.name == "sleeping" else theme.EYE_INNER
            for mote in self.motes:
                mote.draw(surf, tint)

    def _draw_tilted(self, angle, fcx, fcy, expr, speaking, listening):
        """Head tilt: compose the face on the layer, rotate just that, and put
        it where rotating the whole canvas about its centre would have put it.

        Everything outside the cropped layer (motes, particles that drifted
        off the face) is painted onto the screen at its rotated position, after
        the layer — it cannot overlap the face, because the crop holds all of
        the face, so the paint order behind/in front does not change."""
        l, t, r, b = self._content_box()
        L, T, _, _ = self._layer_box
        w, h = r - l, b - t
        layer = self._layer.subsurface(pygame.Rect(l - L, t - T, w, h))
        layer.fill(BG)
        ox, oy = fcx + l, fcy + t                  # crop origin on the canvas

        outside = []
        if self.motes:
            tint = theme.ZZZ_COL if expr.name == "sleeping" else theme.EYE_INNER
            for mote in self.motes:
                m = mote.reach()
                if m <= mote.x - ox <= w - m and m <= mote.y - oy <= h - m:
                    mote.draw(layer, tint, -ox, -oy)
                else:
                    outside.append(mote)

        spill = []
        self._draw_face(layer, fcx - ox, fcy - oy, expr, speaking, listening,
                        -ox, -oy, spill)
        self._paint_front(layer, -ox, -oy, spill)
        # rotate() pads the new corners with the top-left pixel's colour on an
        # opaque surface — pin it to the background.
        layer.set_at((0, 0), BG)

        screen = self.screen
        rotated = pygame.transform.rotate(layer, angle)
        lcx, lcy = rotate_point(ox + w / 2, oy + h / 2, angle)
        screen.blit(rotated, rotated.get_rect(center=(round(lcx), round(lcy))))

        if outside:
            for mote in outside:
                rx, ry = rotate_point(mote.x, mote.y, angle)
                mote.draw(screen, tint, round(rx - mote.x), round(ry - mote.y))
        for p in spill:
            rx, ry = rotate_point(p.x, p.y, angle)
            p.draw(screen, round(rx - p.x), round(ry - p.y))
        if self.wake_active:
            rx, ry = rotate_point(fcx, fcy, angle)
            self._draw_wake_ripple(screen, round(rx), round(ry))

    def _content_box(self):
        """Box (l, t, r, b), relative to the face centre, that holds every
        pixel the face can draw THIS frame — so the rotation works on, e.g.,
        ~690x290 px for a resting face instead of the ~930x550 worst case.

        Sized from the eased open-eye size (not the blink), so it only moves
        while an expression is changing, and snapped outward to 16 px so the
        rotated face never shivers by a pixel as the box resizes."""
        solid_w, solid_h = 1.06 * 1.12, 1.30 * 1.22
        _, scale = self._nudge
        l = t = 10 ** 6
        r = b = -10 ** 6
        for eye in (self.left_eye, self.right_eye):
            hw = eye.w * scale * solid_w / 2 + abs(eye.pupil_ox) + 45
            hh = eye.h * scale * solid_h / 2 + abs(eye.pupil_oy) + 45 + 30
            l, r = min(l, eye.rel_x - hw), max(r, eye.rel_x + hw)
            t, b = min(t, eye.rel_y - hh), max(b, eye.rel_y + hh)
        if self.mouth.visible >= 0.02:
            my = self.mouth.rel_y
            l, r = min(l, -165), max(r, 165)
            t, b = min(t, my - 64), max(b, my + 124)
        if max(self.left_blush.alpha, self.right_blush.alpha) >= 3:
            bx, by = BLUSH_OFFSET
            l, r = min(l, -bx - 42), max(r, bx + 42)
            t, b = min(t, by - 42), max(b, by + 42)

        L, T, R, B = self._layer_box
        q = 16
        return (max(L, int(math.floor(l / q)) * q),
                max(T, int(math.floor(t / q)) * q),
                min(R, int(math.ceil(r / q)) * q),
                min(B, int(math.ceil(b / q)) * q))

    @staticmethod
    def _paint(group, surf, dx=0, dy=0, spill=None):
        """Paint particles onto `surf` shifted by (dx, dy). With a `spill`
        list, anything whose sprite would cross the surface edge is appended
        there instead of being clipped."""
        if spill is None:
            for p in group:
                p.draw(surf, dx, dy)
            return
        w, h = surf.get_size()
        for p in group:
            m = p.reach()
            lx, ly = p.x + dx, p.y + dy
            if m <= lx <= w - m and m <= ly <= h - m:
                p.draw(surf, dx, dy)
            else:
                spill.append(p)

    @staticmethod
    def _paint_rotated(surf, p, angle):
        rx, ry = rotate_point(p.x, p.y, angle)
        p.draw(surf, round(rx - p.x), round(ry - p.y))

    def _paint_front(self, surf, dx=0, dy=0, spill=None):
        for kind in FRONT_PARTICLES:
            self._paint(self.particles[kind], surf, dx, dy, spill)

    def _draw_face(self, surf, fcx, fcy, expr, speaking, listening,
                   pdx=0, pdy=0, spill=None):
        """Blush, back particles, eyes and mouth. (fcx, fcy) is the face
        centre on `surf`; (pdx, pdy) shifts the screen-space particles."""
        self.left_blush.draw(surf, fcx, fcy)
        self.right_blush.draw(surf, fcx, fcy)

        self._paint(self.sound_waves, surf, pdx, pdy, spill)       # behind the mouth
        self._paint(self.particles["steam"], surf, pdx, pdy, spill)  # behind the eyes

        brow = "raised" if listening else expr.brow
        color = self._eye_color()
        for eye in (self.left_eye, self.right_eye):
            eye.draw(surf, fcx + eye.rel_x, fcy + eye.rel_y,
                     brow=brow, color=color,
                     red_pulse=self.red_pulse, arc=self.eye_arc,
                     nudge=self._nudge)

        self.mouth.draw(surf, fcx + self.mouth.rel_x, fcy + self.mouth.rel_y,
                        expr, speaking)

    def _draw_wake_ripple(self, surf, fcx, fcy):
        """Expanding ring as she wakes. Drawn on a reused scratch surface: it
        is translucent over the eyes, and far too big (up to 600 px) and
        short-lived to be worth caching per radius."""
        if not self.wake_active:
            return
        r = int(self.wake_ripple * 300)
        alpha = int((1.0 - self.wake_ripple) * 120)
        if r <= 0 or alpha <= 0:
            return
        size = r * 2 + 4
        area = pygame.Rect(0, 0, size, size)
        self._ripple.fill((0, 0, 0, 0), area)
        pygame.draw.circle(self._ripple, (*theme.EYE_INNER, alpha),
                           (r + 2, r + 2), r, 6)
        surf.blit(self._ripple, (fcx - r - 2, fcy - r - 2), area)

    def _draw_convo_light(self):
        """Top-left status light, lit while the conversation window is open.

        Red and cornered rather than a soft dot under the mouth, so it reads as
        a recording/live indicator at a glance instead of blending into the
        face. Breathes gently so it is clearly live and not a dead pixel."""
        if self.awake_glow <= 0.04:
            return

        pulse = 0.75 + 0.25 * math.sin(self.breath_phase * 1.6)
        a = alpha_bucket(220 * self.awake_glow * pulse, steps=32)
        r = CONVO_LIGHT_RADIUS
        pad = CONVO_LIGHT_MARGIN
        size = r * 6
        c = size // 2

        def build_dot():
            dot = pygame.Surface((size, size), pygame.SRCALPHA)
            for radius, alpha in ((r * 3, a // 8), (r * 2, a // 3), (r, a)):
                pygame.draw.circle(dot, (*CONVO_LIGHT_COL, max(0, alpha)),
                                   (c, c), radius)
            # Hot centre, so it stays vivid against a dark background.
            pygame.draw.circle(dot, (255, 235, 235, min(255, a)), (c, c),
                               max(1, r // 3))
            return dot
        self.screen.blit(sprite(("convo_dot", r, CONVO_LIGHT_COL, a), build_dot),
                         (pad - c + r, pad - c + r))

        if CONVO_LIGHT_LABEL:
            key = ("convo_label", CONVO_LIGHT_LABEL_SIZE, CONVO_LIGHT_COL)
            base = sprite(key + (255,), lambda: get_font(
                CONVO_LIGHT_LABEL_SIZE).render("LISTENING", True, CONVO_LIGHT_COL))
            label = base if a >= 255 else sprite(key + (a,),
                                                 lambda: with_alpha(base, a))
            self.screen.blit(label,
                             (pad + r * 3, pad + r - label.get_height() // 2))

    def _draw_banner(self):
        """Greeting banner ("Hi Alice!") — a soft pill of text under the face.
        Anything can post one by setting state.banner_text/banner_until; it
        fades itself out over its final second."""
        if not GREET_BANNER:
            return
        with state.lock:
            text  = state.banner_text
            until = state.banner_until
            since = state.banner_since
        now = time.time()
        remaining = until - now
        if not text or remaining <= 0:
            return

        dy, scale, fade_in = 0.0, 1.0, 1.0
        if RECOG_BANNER_SLIDE and since > 0:
            dy, scale, fade_in = slide_params(now - since)
        alpha = alpha_bucket(255 * min(fade_in,
                                       clamp(remaining / 1.0, 0.0, 1.0)))
        if alpha <= 0:
            return
        scale = round(round(scale / BANNER_SCALE_STEP) * BANNER_SCALE_STEP, 2)
        pill = banner_sprite(text, alpha, scale)
        w, h = pill.get_size()
        rest_h = h / scale                         # height at scale 1.0
        centre_y = HEIGHT - 26 - rest_h / 2 + dy
        self.screen.blit(pill, ((WIDTH - w) // 2, round(centre_y - h / 2)))

    def _draw_thinking(self):
        """Three dots orbiting below the face while the brain is working."""
        if not THINKING_DOTS or self.think_glow <= 0.02:
            return

        cx, cy = WIDTH // 2, HEIGHT - 90
        for i in range(3):
            # Each dot trails the one before it around a flat ellipse; fading
            # the trailing dots makes the leader read as the front.
            phase = self.think_phase - i * 0.55
            alpha = int(200 * self.think_glow * (1.0 - i * 0.28))
            if alpha <= 0:
                continue
            r = 7 - i
            s = sprite(("think", r, theme.EYE_INNER, alpha_bucket(alpha)),
                       lambda r=r, a=alpha_bucket(alpha):
                       _glow_dot(r, theme.EYE_INNER, a))
            self.screen.blit(s, (int(cx + math.cos(phase) * 34) - r * 3,
                                 int(cy + math.sin(phase) * 9) - r * 3))


def _glow_dot(r, col, alpha):
    """A dot with a soft 3x halo — one thinking dot."""
    s = pygame.Surface((r * 6, r * 6), pygame.SRCALPHA)
    pygame.draw.circle(s, (*col, alpha // 4), (r * 3, r * 3), r * 3)
    pygame.draw.circle(s, (*col, alpha), (r * 3, r * 3), r)
    return s


def banner_sprite(text, alpha, scale=1.0):
    """The greeting pill for `text` at a bucketed alpha / scale, cached — it
    used to be re-rendered (font + two rounded rects) on every frame it was up."""
    key = ("banner", text, theme.EYE_INNER, theme.GLOW_COL, alpha, scale)

    def build():
        label = get_font(GREET_BANNER_SIZE).render(text, True, theme.EYE_INNER)
        pad_x, pad_y = 34, 16
        w = label.get_width() + pad_x * 2
        h = label.get_height() + pad_y * 2
        pill = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(pill, (*theme.GLOW_COL, int(alpha * 0.55)),
                         pill.get_rect(), border_radius=h // 2)
        pygame.draw.rect(pill, (*theme.EYE_INNER, int(alpha * 0.5)),
                         pill.get_rect(), width=3, border_radius=h // 2)
        label.set_alpha(alpha)
        pill.blit(label, (pad_x, pad_y))
        if scale != 1.0:
            pill = pygame.transform.smoothscale(
                pill, (max(1, round(w * scale)), max(1, round(h * scale))))
        return pill
    return sprite(key, build)
