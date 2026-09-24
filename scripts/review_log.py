#!/usr/bin/env python3
# scripts/review_log.py
"""
review_log.py — read a day of Luna's conversation log as a transcript.

WHY: the JSONL log is complete but not readable at a glance. Tuning happens
by reading what she heard, why she answered (or ignored) it, which path
answered, how long it took and how people reacted — this prints exactly that.

Usage (from the project root):

    venv/bin/python scripts/review_log.py                 # today
    venv/bin/python scripts/review_log.py 2026-09-24 --ignored
    venv/bin/python scripts/review_log.py yesterday --rewards
    venv/bin/python scripts/review_log.py --sim           # data/logs_sim
    venv/bin/python scripts/review_log.py --json > day.json

  --ignored   also show utterances the addressing gate ignored (+ reason)
  --rewards   only turns that received a reward, plus a per-style summary
  --json      the merged records as JSON (updates folded into their turn)
  --dir DIR   read another log folder (default CONVO_LOG_DIR)
"""

import argparse
import datetime as dt
import json
import os
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import conversation_log  # noqa: E402

SIM_DIR = "data/logs_sim"


def _date_arg(s):
    if not s or s == "today":
        return dt.date.today().isoformat()
    if s == "yesterday":
        return (dt.date.today() - dt.timedelta(days=1)).isoformat()
    dt.date.fromisoformat(s)             # validate
    return s


def _clock(ts):
    try:
        return dt.datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
    except Exception:                    # noqa: BLE001
        return "??:??:??"


def _wrap(label, text, width=96):
    body = textwrap.fill(str(text), width=width - 13,
                         subsequent_indent=" " * 13)
    return f"    {label:<7}: {body}"


def _fmt_timings(t):
    t = t or {}
    ttft, total = t.get("ttft_s"), t.get("total_s")
    if total is None:
        return ""
    if ttft is None:
        return f"{total:.1f}s"
    return f"{ttft:.1f}s/{total:.1f}s"


def format_turn(rec):
    """A multi-line human rendering of one merged record."""
    ts = _clock(rec.get("ts"))
    path = rec.get("path") or "?"
    if path == "ignored":
        conf = rec.get("conf")
        loud = rec.get("loudness")
        bits = []
        if conf is not None:
            bits.append(f"conf {conf:.2f}")
        if loud is not None:
            bits.append(f"loud {loud:.2f}")
        head = f"{ts}  IGNORED  \"{rec.get('heard_clean') or ''}\""
        if bits:
            head += "  (" + ", ".join(bits) + ")"
        lines = [head]
        if rec.get("addr_reason"):
            lines.append(_wrap("why", rec["addr_reason"]))
        return "\n".join(lines)

    tim = _fmt_timings(rec.get("timings"))
    head = f"{ts}  [{path} {tim}]" if tim else f"{ts}  [{path}]"
    who = rec.get("person")
    if who:
        head += f"  {who}"
    meta = [m for m in (rec.get("context"), rec.get("style_arm")) if m]
    if meta:
        head += "  (" + ", ".join(meta) + ")"
    lines = [head, _wrap("heard", rec.get("heard_clean") or "")]
    if rec.get("heard_raw") and rec.get("heard_raw") != rec.get("heard_clean"):
        lines.append(_wrap("raw", rec["heard_raw"]))
    if rec.get("conf") is not None or rec.get("addr_reason"):
        s = f"conf {rec['conf']:.2f}" if rec.get("conf") is not None else ""
        if rec.get("addr_reason"):
            s = (s + "; " if s else "") + rec["addr_reason"]
        lines.append(_wrap("gate", s))
    if rec.get("reply"):
        lines.append(_wrap("luna", rec["reply"]))
    kb = rec.get("kb_matches") or []
    if kb:
        lines.append(_wrap("kb", ", ".join(
            f"{q} {s:.2f}" for q, s in kb[:3])))
    web = rec.get("web_matches") or []
    if web:
        lines.append(_wrap("web", ", ".join(f"{u} {s:.2f}" for u, s in web[:3])))
    if rec.get("reward") is not None:
        sig = ", ".join(f"{n} {v:+.2f}" for n, v in rec.get("reward_signals")
                        or [])
        lines.append(_wrap("reward", f"{rec['reward']:+.2f}"
                           + (f"  ({sig})" if sig else "")))
    return "\n".join(lines)


def summarize(records):
    """Counts per path, ignored, rewards — and per-arm mean reward."""
    by_path, arms = {}, {}
    rewarded = []
    for r in records:
        by_path[r.get("path")] = by_path.get(r.get("path"), 0) + 1
        if r.get("reward") is not None:
            rewarded.append(r["reward"])
            if r.get("style_arm"):
                s = arms.setdefault(r["style_arm"], [0.0, 0])
                s[0] += r["reward"]
                s[1] += 1
    lines = ["", "── summary " + "─" * 60,
             "turns by path: " + ", ".join(
                 f"{k}={v}" for k, v in sorted(by_path.items(),
                                                key=lambda kv: str(kv[0])))]
    if rewarded:
        lines.append(f"rewarded: {len(rewarded)}  mean reward "
                     f"{sum(rewarded) / len(rewarded):+.2f}")
    for arm, (tot, n) in sorted(arms.items(), key=lambda kv: -kv[1][0] / kv[1][1]):
        lines.append(f"  {arm:<32} n={n:<3} mean {tot / n:+.2f}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("date", nargs="?", default="today",
                    help="YYYY-MM-DD, 'today' (default) or 'yesterday'")
    ap.add_argument("--ignored", action="store_true",
                    help="include ignored utterances")
    ap.add_argument("--rewards", action="store_true",
                    help="only rewarded turns + per-style summary")
    ap.add_argument("--json", action="store_true", help="merged JSON records")
    ap.add_argument("--dir", default=None, help="log folder")
    ap.add_argument("--sim", action="store_true",
                    help=f"read simulated sessions ({SIM_DIR})")
    args = ap.parse_args(argv)

    if args.sim:
        conversation_log.configure(log_dir=SIM_DIR)
    elif args.dir:
        conversation_log.configure(log_dir=args.dir)
    try:
        date = _date_arg(args.date)
    except ValueError:
        print(f"bad date {args.date!r} (want YYYY-MM-DD)", file=sys.stderr)
        return 2

    records = conversation_log.iter_day(date)
    if not records:
        avail = conversation_log.days()
        print(f"No log for {date} in {conversation_log._dir()}."
              + (f" Days with logs: {', '.join(avail[-10:])}" if avail else ""))
        return 1

    shown = [r for r in records
             if (args.ignored or r.get("path") != "ignored")
             and (not args.rewards or r.get("reward") is not None)]
    if args.json:
        json.dump(shown, sys.stdout, ensure_ascii=False, indent=1)
        print()
        return 0

    print(f"Luna conversation log — {date} "
          f"({conversation_log.path_for(date)})\n")
    for r in shown:
        print(format_turn(r))
        print()
    print(summarize(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
