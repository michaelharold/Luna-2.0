#!/usr/bin/env python3
# scripts/show_face.py
"""
Quick visual check of a single expression, without starting the rest of Luna.

    python scripts/show_face.py            # neutral
    python scripts/show_face.py shy        # hold one expression
    python scripts/show_face.py --list     # what's available

For a tour of every expression in turn, run `python face.py` instead.
Keys: 1 / 2 switch style, ESC or Q quits.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from expressions import NAMES  # noqa: E402
from robot_face import RobotFace  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "neutral"

if name in ("--list", "-l"):
    print("Expressions:", ", ".join(NAMES))
    raise SystemExit(0)

if name not in NAMES:
    print(f"Unknown expression {name!r}. Try one of: {', '.join(NAMES)}")
    raise SystemExit(1)

print(f"[show_face] Showing {name!r} — ESC or Q to quit.")
face = RobotFace()
try:
    while True:
        face.set_state(name)
        face.draw()
except (SystemExit, KeyboardInterrupt):
    print("\n[show_face] Closed")
