"""
servo_module.py — Luna servo controller.

ENABLE_SERVOS = False  → silent simulation mode (Mac / dev)
ENABLE_SERVOS = True   → real servo pulses on the Pi

GPIO pins (BCM numbering) — two arm servos only:
  LEFT_ARM  → GPIO 17
  RIGHT_ARM → GPIO 18

There is no head servo. `nod()` is kept as an explicit no-op so older call
sites can't crash, but it never moves anything.

Why this file was rewritten
===========================
Three separate bugs made the servos twitch constantly and fight each other:

1. **The pulse never stopped.** The old code called `pwm.start(duty(90))` at
   boot and then only ever *changed* the duty cycle. A servo that is being
   pulsed is actively holding position, and RPi.GPIO's PWM is generated in
   software by a Python-scheduled thread — every scheduling hiccup shifts the
   pulse width slightly, and the servo chases it. That is the "moving a small
   degree all the time" behaviour, and it happened even when Luna was silent.
   Now a servo is *released* (pulse stopped) once it has reached its target and
   settled, so at rest it receives nothing at all and is mechanically still.

2. **Two threads drove the same servos.** A queue worker ran wave/arms_up while
   a second "talk worker" swayed the very same arm pins. Both called
   ChangeDutyCycle on the same pin, interleaved, so a wave during speech came
   out as a jitter of half-finished moves. There is now exactly ONE worker
   thread and one mode at a time: a queued gesture always runs to completion,
   the talking sway is suspended while it does, and resumes afterwards.

3. **Moves assumed they started at 90°.** `_smooth_move(pwm, 90, 160)` snapped
   from wherever the arm actually was to 90° on its first step. Moves now
   interpolate from the servo's real last-commanded angle.

If `pigpio` is installed and its daemon is running (`sudo pigpiod`) it is used
in preference to RPi.GPIO: pigpio generates the pulses with DMA rather than a
Python thread, which removes the remaining jitter almost entirely.

Motion contract (the ONLY things Luna's arms are allowed to do)
---------------------------------------------------------------
  • not talking, no gesture → completely motionless (servos released)
  • talking                 → small synchronized sway of both arms
  • wave                    → right arm up, small wiggle held up there, down
  • handshake               → right arm out to meet the person, 2 s pause,
                              small up/down shakes, then back down
Exactly one of these is active at a time — a gesture always finishes before the
talking sway resumes, and the sway never runs while a gesture is queued.
"""

import math
import threading
import time
from collections import deque

from config import (
    ENABLE_SERVOS,
    SERVO_BACKEND,
    SERVO_SPEED,
    SERVO_LEFT_PIN,
    SERVO_RIGHT_PIN,
    SERVO_DETACH_AT_REST,
    SERVO_SETTLE_SECS,
    SERVO_STEP_DELAY,
    SERVO_MIN_PULSE_US,
    SERVO_MAX_PULSE_US,
    ARM_REST_ANGLE,
    ARM_UP_ANGLE,
    ARM_DOWN_ANGLE,
    ARM_MOVE_SECS,
    TALK_SWAY_DEGREES,
    TALK_SWAY_PERIOD,
    WAVE_WIGGLES,
    WAVE_WIGGLE_DEGREES,
    WAVE_RAISE_SECS,
    WAVE_WIGGLE_SECS,
    HANDSHAKE_ANGLE,
    HANDSHAKE_HOLD_SECS,
    HANDSHAKE_SHAKES,
    HANDSHAKE_DEGREES,
    HANDSHAKE_RAISE_SECS,
    HANDSHAKE_SHAKE_SECS,
    IS_MAC,
)
import console

# ── Pin assignments ───────────────────────────────────────────────────────────
PIN_LEFT_ARM  = SERVO_LEFT_PIN
PIN_RIGHT_ARM = SERVO_RIGHT_PIN

ARM_PINS = (PIN_LEFT_ARM, PIN_RIGHT_ARM)

PWM_FREQ = 50   # Hz — standard servo frequency
_PERIOD_US = 1_000_000 / PWM_FREQ


def angle_to_pulse(angle):
    """Convert 0-180° to a pulse width in microseconds."""
    angle = max(0.0, min(180.0, float(angle)))
    span = SERVO_MAX_PULSE_US - SERVO_MIN_PULSE_US
    return SERVO_MIN_PULSE_US + (angle / 180.0) * span


def angle_to_duty(angle):
    """Convert 0-180° to a duty cycle percentage at 50 Hz.
    With the default 500-2500 µs pulse range this is the same 2.5-12.5%
    mapping the wiring guide documents."""
    return angle_to_pulse(angle) / _PERIOD_US * 100.0


# ── Hardware backends ─────────────────────────────────────────────────────────
#
# Each backend only has to do three things: hold a servo at an angle, stop
# driving it entirely (release), and shut down. "Release" is the important one
# — it is what lets the servos be genuinely still when Luna is not moving.


class _SimBackend:
    """Dev machine: no GPIO. Silent — the controller prints the gesture name
    once instead of the old ~40 lines of per-step spam."""

    name = "simulation"

    def write(self, pin, angle):
        pass

    def release(self, pin):
        pass

    def close(self):
        pass


class _PigpioBackend:
    """DMA-timed pulses — no jitter from Python thread scheduling.
    Needs the daemon: `sudo pigpiod` (enable at boot with
    `sudo systemctl enable pigpiod`)."""

    name = "pigpio"

    def __init__(self):
        import pigpio
        self._pi = pigpio.pi()
        if not self._pi.connected:
            raise RuntimeError("pigpio daemon not running — start it with "
                               "`sudo pigpiod`")

    def write(self, pin, angle):
        self._pi.set_servo_pulsewidth(pin, int(angle_to_pulse(angle)))

    def release(self, pin):
        self._pi.set_servo_pulsewidth(pin, 0)   # 0 = stop pulsing this pin

    def close(self):
        for pin in ARM_PINS:
            try:
                self._pi.set_servo_pulsewidth(pin, 0)
            except Exception:
                pass
        self._pi.stop()


class _RPiGPIOBackend:
    """Software PWM fallback. Works everywhere RPi.GPIO does, but the pulse is
    generated by a Python thread so a small amount of jitter is unavoidable —
    which is exactly why we release the servo when it is not moving."""

    name = "RPi.GPIO"

    def __init__(self):
        import RPi.GPIO as GPIO
        self._GPIO = GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        self._pwm = {}
        for pin in ARM_PINS:
            GPIO.setup(pin, GPIO.OUT)
            pwm = GPIO.PWM(pin, PWM_FREQ)
            # start(0) = no pulse at all. Boot leaves the servos limp and quiet
            # instead of buzzing at 90° from the moment Luna launches.
            pwm.start(0)
            self._pwm[pin] = pwm

    def write(self, pin, angle):
        self._pwm[pin].ChangeDutyCycle(angle_to_duty(angle))

    def release(self, pin):
        self._pwm[pin].ChangeDutyCycle(0)

    def close(self):
        for pwm in self._pwm.values():
            try:
                pwm.ChangeDutyCycle(0)
                pwm.stop()
            except Exception:
                pass
        self._GPIO.cleanup()


def _make_backend():
    if not ENABLE_SERVOS:
        console.detail("[SERVO] Servos disabled (ENABLE_SERVOS=False) — simulation mode")
        return _SimBackend()

    wanted = (SERVO_BACKEND or "auto").lower()
    attempts = []
    if wanted in ("auto", "pigpio"):
        attempts.append(("pigpio", _PigpioBackend))
    if wanted in ("auto", "rpigpio", "rpi.gpio"):
        attempts.append(("RPi.GPIO", _RPiGPIOBackend))

    for label, cls in attempts:
        try:
            backend = cls()
            console.detail(f"[SERVO] Using {backend.name} backend "
                  f"(left=GPIO{PIN_LEFT_ARM}, right=GPIO{PIN_RIGHT_ARM})")
            return backend
        except Exception as e:
            console.detail(f"[SERVO] {label} unavailable ({e})")

    # Expected on a Mac (no GPIO at all); worth a warning on the Pi.
    (console.detail if IS_MAC else print)(
        "[SERVO] No servo backend available — falling back to simulation")
    return _SimBackend()


# ── Controller ────────────────────────────────────────────────────────────────

class ServoController:
    """One worker thread owns both servos. Nothing else writes to the hardware,
    so two motions can never interleave on the same pin."""

    def __init__(self):
        self.running = True

        self._backend     = _make_backend()
        self._lock        = threading.Lock()
        self._queue       = deque()
        self._pending     = set()     # gesture names already queued (dedupe)
        self._talking     = False
        self._was_talking = False

        # Last angle actually commanded per pin — moves interpolate from here,
        # so a gesture starting mid-sway doesn't snap.
        self._angle      = {pin: ARM_REST_ANGLE for pin in ARM_PINS}
        self._release_at = {}         # pin → time.time() when it may be released
        self._live       = set()      # pins currently being pulsed

        threading.Thread(target=self._worker, daemon=True, name="servo").start()

    # ── Low-level ─────────────────────────────────────────────────────────────

    def _write(self, pin, angle):
        angle = max(0.0, min(180.0, float(angle)))
        self._angle[pin] = angle
        self._release_at.pop(pin, None)     # it's moving — cancel any release
        self._live.add(pin)
        self._backend.write(pin, angle)

    def _release(self, pin):
        self._release_at.pop(pin, None)
        self._live.discard(pin)
        self._backend.release(pin)

    def _schedule_release(self, pins):
        """Let the servo finish travelling, then stop pulsing it so it holds
        still without the software-PWM jitter."""
        if not SERVO_DETACH_AT_REST:
            return
        deadline = time.time() + SERVO_SETTLE_SECS
        for pin in pins:
            if pin in self._live:
                self._release_at[pin] = deadline

    def _service_releases(self):
        """Release any servo whose settle time has elapsed. Called from
        everywhere the worker waits — including during a long gesture, so the
        arm that ISN'T taking part goes limp instead of sitting energized (and
        therefore jittering) for the whole handshake."""
        now = time.time()
        for pin, deadline in list(self._release_at.items()):
            if now >= deadline:
                self._release(pin)

    def _hold(self, secs):
        """Gesture pause that keeps servicing releases while it waits."""
        end = time.time() + secs
        while self.running and time.time() < end:
            self._service_releases()
            time.sleep(min(0.02, max(0.0, end - time.time())))

    def _move(self, targets, secs):
        """Interpolate `targets` ({pin: angle}) from the current angles.
        Returns False if shutdown cut the move short."""
        secs   = secs / max(SERVO_SPEED, 0.05)   # global speed multiplier
        starts = {pin: self._angle[pin] for pin in targets}
        steps  = max(2, int(secs / SERVO_STEP_DELAY))
        for i in range(1, steps + 1):
            if not self.running:
                return False
            self._service_releases()
            f = i / steps
            for pin, target in targets.items():
                self._write(pin, starts[pin] + (target - starts[pin]) * f)
            time.sleep(SERVO_STEP_DELAY)
        return True

    def _park(self, pins=ARM_PINS, secs=0.3):
        """Return `pins` to the rest angle and let them go limp.

        Only pins that have actually drifted are driven — parking after a
        right-arm-only wave must not re-energize the left arm, which would make
        the idle arm twitch for no reason."""
        moving = [p for p in pins if abs(self._angle[p] - ARM_REST_ANGLE) > 0.5]
        if moving:
            self._move({p: ARM_REST_ANGLE for p in moving}, secs)
        self._schedule_release(pins)

    # ── Queueing ──────────────────────────────────────────────────────────────

    def _enqueue(self, name, fn):
        """Queue a one-shot gesture. Repeats of a gesture that is already
        waiting are dropped — a burst of WAVE detections used to stack up
        several waves that then played back-to-back for ten seconds."""
        with self._lock:
            if name in self._pending or len(self._queue) >= 3:
                return
            self._pending.add(name)
            self._queue.append((name, fn))

    def queue(self, fn):
        """Back-compat: queue an arbitrary callable as a one-off gesture."""
        self._enqueue(f"custom-{id(fn)}", fn)

    def _next_action(self):
        with self._lock:
            if not self._queue:
                return None
            name, fn = self._queue.popleft()
            self._pending.discard(name)
            return name, fn

    # ── Worker ────────────────────────────────────────────────────────────────

    def _worker(self):
        # Drive both arms to a known rest position once, then release: from
        # here on, silence means no pulses at all.
        for pin in ARM_PINS:
            self._write(pin, ARM_REST_ANGLE)
        time.sleep(SERVO_SETTLE_SECS)
        if SERVO_DETACH_AT_REST:
            for pin in ARM_PINS:
                self._release(pin)

        while self.running:
            action = self._next_action()

            if action is not None:
                name, fn = action
                console.detail(f"[SERVO] {name}()")
                # Always begin from rest. If the sway was mid-stroke when this
                # gesture arrived, an arm would otherwise sit frozen a few
                # degrees off-centre for the whole gesture.
                self._park(secs=0.15)
                try:
                    fn()
                except Exception as e:
                    print(f"[SERVO] {name} failed: {e}")
                finally:
                    # every gesture ends at rest; make that true even if it
                    # raised part-way through
                    self._park(secs=0.25)
                continue

            with self._lock:
                talking = self._talking

            if talking:
                self._was_talking = True
                self._talk_cycle()
                continue

            if self._was_talking:
                self._was_talking = False
                self._park(secs=0.25)

            self._idle_tick()

    def _idle_tick(self):
        self._service_releases()
        time.sleep(0.02)

    # ── Talking motion ────────────────────────────────────────────────────────

    def start_talking(self):
        """Begin the small both-arms sway. Called the moment audio actually
        starts playing (not when synthesis begins) so the body moves in time
        with the voice."""
        with self._lock:
            self._talking = True

    def stop_talking(self):
        """Stop the sway; the worker walks the arms back to rest and releases."""
        with self._lock:
            self._talking = False

    def _talk_cycle(self):
        """One gentle sine sway of both arms. Bails out immediately if speech
        ends or a gesture is queued, so nothing ever overlaps."""
        steps = max(6, int(TALK_SWAY_PERIOD / SERVO_STEP_DELAY))
        for i in range(steps):
            with self._lock:
                if not self._talking or self._queue:
                    return
            if not self.running:
                return
            offset = TALK_SWAY_DEGREES * math.sin(2 * math.pi * i / steps)
            self._write(PIN_LEFT_ARM,  ARM_REST_ANGLE + offset)
            self._write(PIN_RIGHT_ARM, ARM_REST_ANGLE + offset)
            time.sleep(SERVO_STEP_DELAY)

    # ── Gestures ──────────────────────────────────────────────────────────────

    def wave(self):
        """Right arm rises, wiggles a little while held up, then comes down."""
        def _do():
            self._move({PIN_RIGHT_ARM: ARM_UP_ANGLE}, WAVE_RAISE_SECS)
            for _ in range(WAVE_WIGGLES):
                self._move({PIN_RIGHT_ARM: ARM_UP_ANGLE - WAVE_WIGGLE_DEGREES}, WAVE_WIGGLE_SECS)
                self._move({PIN_RIGHT_ARM: ARM_UP_ANGLE}, WAVE_WIGGLE_SECS)
            self._move({PIN_RIGHT_ARM: ARM_REST_ANGLE}, WAVE_RAISE_SECS)
        self._enqueue("wave", _do)

    def handshake(self):
        """Right arm swings out to meet the person (perpendicular to the body),
        waits HANDSHAKE_HOLD_SECS for them to take it, then makes a few small
        up/down shakes before returning to rest."""
        def _do():
            self._move({PIN_RIGHT_ARM: HANDSHAKE_ANGLE}, HANDSHAKE_RAISE_SECS)
            self._hold(HANDSHAKE_HOLD_SECS)
            for _ in range(HANDSHAKE_SHAKES):
                self._move({PIN_RIGHT_ARM: HANDSHAKE_ANGLE + HANDSHAKE_DEGREES}, HANDSHAKE_SHAKE_SECS)
                self._move({PIN_RIGHT_ARM: HANDSHAKE_ANGLE - HANDSHAKE_DEGREES}, HANDSHAKE_SHAKE_SECS)
            self._move({PIN_RIGHT_ARM: HANDSHAKE_ANGLE}, HANDSHAKE_SHAKE_SECS)
            self._hold(0.3)
            self._move({PIN_RIGHT_ARM: ARM_REST_ANGLE}, HANDSHAKE_RAISE_SECS)
        self._enqueue("handshake", _do)

    def nod(self):
        """No-op — Luna has no head servo. Kept so any leftover call site is
        harmless rather than an AttributeError."""
        return

    def arm_up(self, side="right"):
        """Raise one arm, hold, then lower."""
        pin = PIN_RIGHT_ARM if side == "right" else PIN_LEFT_ARM
        def _do():
            self._move({pin: ARM_UP_ANGLE}, ARM_MOVE_SECS)
            self._hold(0.6)
            self._move({pin: ARM_REST_ANGLE}, ARM_MOVE_SECS)
        self._enqueue(f"arm_up_{side}", _do)

    def arm_down(self, side="right"):
        """Lower one arm below rest, hold, then return."""
        pin = PIN_RIGHT_ARM if side == "right" else PIN_LEFT_ARM
        def _do():
            self._move({pin: ARM_DOWN_ANGLE}, ARM_MOVE_SECS)
            self._hold(0.4)
            self._move({pin: ARM_REST_ANGLE}, ARM_MOVE_SECS)
        self._enqueue(f"arm_down_{side}", _do)

    def arms_up(self):
        """Raise both arms together — celebration / thank-you gesture."""
        def _do():
            self._move({PIN_LEFT_ARM: ARM_UP_ANGLE,
                        PIN_RIGHT_ARM: ARM_UP_ANGLE}, ARM_MOVE_SECS)
            self._hold(0.8)
            self._move({PIN_LEFT_ARM: ARM_REST_ANGLE,
                        PIN_RIGHT_ARM: ARM_REST_ANGLE}, ARM_MOVE_SECS)
        self._enqueue("arms_up", _do)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def cleanup(self):
        self.running = False
        with self._lock:
            self._talking = False
            self._queue.clear()
            self._pending.clear()
        time.sleep(0.1)
        try:
            for pin in ARM_PINS:
                self._backend.write(pin, ARM_REST_ANGLE)
            time.sleep(SERVO_SETTLE_SECS)
        except Exception:
            pass
        try:
            self._backend.close()
        except Exception:
            pass


# Singleton — import this everywhere
servo = ServoController()
