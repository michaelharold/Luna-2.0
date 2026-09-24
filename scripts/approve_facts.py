#!/usr/bin/env python3
# scripts/approve_facts.py
"""
approve_facts.py — the human gate between what people TELL Luna and what
she KNOWS.

WHY: "remember that the lab closes at five" and corrections after "that's
wrong" are queued in PENDING_FACTS_PATH instead of going straight into
knowledge.txt — anyone at a booth can say anything to her, and a desk robot
that repeats whatever it was last told is a prank waiting to happen. A
person reviews the queue here; approved facts become ordinary knowledge.txt
lines ("question:answer"), which the running Luna hot-reloads.

Usage (from the project root):

    venv/bin/python scripts/approve_facts.py                       # list
    venv/bin/python scripts/approve_facts.py --approve 2 "when does the lab close"
    venv/bin/python scripts/approve_facts.py --approve 1   # uses the suggested
                                                           # question, if any
    venv/bin/python scripts/approve_facts.py --reject 3
"""

import argparse
import datetime as dt
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import learning  # noqa: E402


def _when(ts):
    try:
        return dt.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except Exception:                    # noqa: BLE001
        return "?"


def list_facts():
    items = learning.pending_facts()
    if not items:
        print("No facts waiting for approval.")
        return 0
    print(f"{len(items)} fact(s) waiting for approval "
          f"({learning._abs(learning._cfg['pending_path'])}):\n")
    for i, it in enumerate(items, 1):
        who = it.get("person") or "someone"
        print(f"  [{i}] {it.get('statement')}")
        print(f"      from {who}, {_when(it.get('ts'))}"
              + (f" — suggested question: \"{it['question']}\""
                 if it.get("question") else ""))
    print("\nApprove:  --approve N \"question it answers\"    Reject:  --reject N")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--approve", nargs="+", metavar=("N", "QUESTION"),
                   help="approve fact N; QUESTION is the knowledge.txt question")
    g.add_argument("--reject", type=int, metavar="N", help="drop fact N")
    ap.add_argument("--knowledge", default=None,
                    help="knowledge file (default KNOWLEDGE_PATH)")
    args = ap.parse_args(argv)

    if args.reject is not None:
        item = learning.reject_fact(args.reject)
        if item is None:
            print(f"No pending fact #{args.reject}.")
            return 1
        print(f"Rejected: {item.get('statement')}")
        return 0

    if args.approve:
        try:
            n = int(args.approve[0])
        except ValueError:
            print("--approve needs a number first", file=sys.stderr)
            return 2
        question = " ".join(args.approve[1:]) or None
        items = learning.pending_facts()
        if not 1 <= n <= len(items):
            print(f"No pending fact #{n}.")
            return 1
        if not question and not items[n - 1].get("question"):
            print("This fact has no suggested question — give one: "
                  f"--approve {n} \"the question it answers\"")
            return 1
        line = learning.approve_fact(n, question, knowledge_path=args.knowledge)
        if not line:
            print("Could not approve that fact (see the message above).")
            return 1
        print(f"Added to knowledge: {line}")
        print("A running Luna picks it up automatically (knowledge.txt "
              "hot-reloads). If you approved this ON THE PI, copy the line "
              "into the Mac's data/knowledge.txt too — the Mac is the source "
              "of truth and deploy.sh --go overwrites the Pi's copy.")
        return 0

    return list_facts()


if __name__ == "__main__":
    sys.exit(main())
