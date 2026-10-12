#!/usr/bin/env python3
"""bench_face.py — measure how long the face takes to render, per expression.

Why this exists: on the Pi 5 there is no GPU for pygame, so every millisecond
the Mac spends in RobotFace.draw() is roughly 6-8 ms on the robot. A face that
is smooth on the laptop can stutter on the desk. This gives a repeatable
number to optimise against instead of eyeballing frame drops.

    python scripts/bench_face.py                 # table of ms/frame
    python scripts/bench_face.py --profile       # + cProfile top 25
    python scripts/bench_face.py --shots DIR     # + one PNG per expression/mode
    python scripts/bench_face.py --repeat 5      # min of 5 runs per cell
    python scripts/bench_face.py --style 1       # another palette/geometry

Modes (each expression is rendered in every mode):
    plain     nobody in view, idle gaze drift
    tilt      a face at the left edge (face_x 0.15) -> head tilted ~8 deg
    speaking  talking mouth at audio_energy 0.7
    overlays  conversation light + thinking dots + greeting banner

Runs headless (SDL dummy driver). time.time() is replaced by a fake 60 fps
clock and `random` is re-seeded per run, so the workload — and the PNGs — are
identical from run to run; that is what makes before/after comparisons fair.
Timing is per-process CPU time (time.process_time), not wall time: the dev
Mac is usually busy with other work, and wall-clock numbers swung 2x between
identical runs, while CPU time only counts the rendering itself. Each cell is
the MIN over --repeat runs (default 3) — noise only ever adds time.
"""

import argparse
import cProfile
import io
import os
import pstats
import random
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402

# clock.tick() must never throttle the benchmark.
config.RENDER_FPS = 1_000_000

SETTLE = 60
TIMED = 240
MODES = ("plain", "tilt", "speaking", "overlays")


class FakeClock:
    """Deterministic time.time(): advances exactly 1/60 s per rendered frame."""

    def __init__(self):
        self.t = 1_700_000_000.0

    def __call__(self):
        return self.t

    def tick(self):
        self.t += 1.0 / 60.0


def _set_mode(state, mode, now):
    with state.lock:
        state.face_detected = mode == "tilt"
        state.face_x = 0.15 if mode == "tilt" else 0.5
        state.face_y = 0.5
        state.speaking = mode == "speaking"
        state.audio_energy = 0.7 if mode == "speaking" else 0.0
        state.listening = False
        state.conversation_active = mode == "overlays"
        state.luna_mode = "processing" if mode == "overlays" else "idle"
        state.banner_text = "Hi Alice!" if mode == "overlays" else ""
        state.banner_since = now if mode == "overlays" else 0.0
        state.banner_until = now + 1e6 if mode == "overlays" else 0.0


def run(names, modes, shots_dir=None, profiler=None, repeat=1):
    import pygame
    import robot_face
    from shared_state import state

    robot_face.RENDER_FPS = config.RENDER_FPS
    clock = FakeClock()
    real_time = time.time
    time.time = clock
    results = {}
    try:
        for mode in modes:
            for name in names:
                random.seed(1234)
                face = robot_face.RobotFace()
                _set_mode(state, mode, clock())
                face.set_state(name)
                for _ in range(SETTLE):
                    face.draw()
                    clock.tick()
                best = float("inf")
                for i in range(max(1, repeat)):
                    if profiler:
                        profiler.enable()
                    t0 = time.process_time()
                    for _ in range(TIMED):
                        face.draw()
                        clock.tick()
                    best = min(best, time.process_time() - t0)
                    if profiler:
                        profiler.disable()
                    if shots_dir and i == 0:
                        # Always frame SETTLE + TIMED, whatever --repeat is.
                        pygame.image.save(face.screen, os.path.join(
                            shots_dir, f"{mode}_{name}.png"))
                results[(mode, name)] = best * 1000.0 / TIMED
    finally:
        time.time = real_time
        _set_mode(state, "plain", 0.0)
    return results


def print_table(results, names, modes):
    head = f"{'expression':<12}" + "".join(f"{m:>10}" for m in modes)
    print(head)
    print("-" * len(head))
    for name in names:
        print(f"{name:<12}" + "".join(f"{results[(m, name)]:>10.3f}" for m in modes))
    print("-" * len(head))
    means = {m: sum(results[(m, n)] for n in names) / len(names) for m in modes}
    print(f"{'MEAN':<12}" + "".join(f"{means[m]:>10.3f}" for m in modes))
    worst = max(results.items(), key=lambda kv: kv[1])
    core = [m for m in modes if m != "overlays"] or list(modes)
    overall = sum(means[m] for m in core) / len(core)
    print(f"\nmean of {'/'.join(core)}: {overall:.3f} ms/frame   "
          f"worst: {worst[1]:.3f} ms ({worst[0][0]}/{worst[0][1]})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", action="store_true", help="cProfile top 25")
    ap.add_argument("--shots", metavar="DIR", help="save a PNG per expression/mode")
    ap.add_argument("--only", nargs="*", help="limit to these expressions")
    ap.add_argument("--repeat", type=int, default=3,
                    help="timed runs per cell; the minimum is reported")
    ap.add_argument("--modes", nargs="*", default=list(MODES), choices=MODES)
    ap.add_argument("--style", type=int, choices=(1, 2, 3),
                    help="face style (default: config.FACE_STYLE)")
    args = ap.parse_args()
    if args.style:
        config.FACE_STYLE = args.style      # robot_face applies it on import

    from expressions import NAMES
    names = [n for n in NAMES if not args.only or n in args.only]
    if args.shots:
        os.makedirs(args.shots, exist_ok=True)

    profiler = cProfile.Profile() if args.profile else None
    results = run(names, args.modes, args.shots, profiler, args.repeat)
    print(f"[bench] CPU ms/frame, {SETTLE} settle + {TIMED} timed frames each "
          f"(min of {args.repeat}), headless")
    print_table(results, names, args.modes)

    if profiler:
        buf = io.StringIO()
        pstats.Stats(profiler, stream=buf).sort_stats("tottime").print_stats(25)
        print(buf.getvalue())


if __name__ == "__main__":
    main()
