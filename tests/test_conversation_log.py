# tests/test_conversation_log.py
"""Unit tests for the conversation log (conversation_log.py) — temp dirs via
configure(), injected timestamps, nothing touches data/logs:

    python -m unittest tests.test_conversation_log -v
"""

import datetime as dt
import json
import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Point the face store at a throwaway dir BEFORE anything imports config.
_tmpdir = tempfile.mkdtemp(prefix="luna-test-faces-")
import config  # noqa: E402
config.FACE_DB_DIR = _tmpdir

import conversation_log as cl  # noqa: E402

FIELDS = ("turn_id", "ts", "mode", "heard_raw", "heard_clean", "conf",
          "loudness", "addressed", "addr_reason", "person", "path",
          "kb_matches", "web_matches", "reply", "timings", "style_arm",
          "context", "reward", "reward_signals")


class TestConversationLog(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="luna-test-convlog-")
        cl.configure(enabled=True, log_dir=self.dir, retention_days=45,
                     mode="mac")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _lines(self, ts=None):
        path = cl.path_for(ts if ts is not None else time.time())
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_new_turn_writes_every_field(self):
        tid = cl.new_turn(heard_clean="what is the vision", path="llm",
                          reply="To excel.", person="Alice",
                          kb_matches=[["vision", 0.8]],
                          timings={"ttft_s": 0.9, "total_s": 3.2},
                          style_arm="one_sentence|playful|ask",
                          context="known|kb|morning")
        self.assertTrue(tid)
        rec = self._lines()[0]
        for k in FIELDS:
            self.assertIn(k, rec)
        self.assertEqual(rec["turn_id"], tid)
        self.assertEqual(rec["mode"], "mac")
        self.assertTrue(rec["addressed"])            # default is True
        self.assertIsNone(rec["reward"])
        self.assertEqual(rec["reward_signals"], [])
        self.assertEqual(cl.last_turn()["reply"], "To excel.")

    def test_update_merges_in_iter_day(self):
        tid = cl.new_turn(heard_clean="hi", path="llm", reply="Hello!")
        cl.update_turn(tid, reward=0.5, reward_signals=[["thanks", 0.5]])
        raw = self._lines()
        self.assertEqual(len(raw), 2)                # append-only
        self.assertEqual(raw[1]["update"], tid)
        merged = cl.iter_day(dt.date.today())
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["reward"], 0.5)
        self.assertEqual(cl.last_turn()["reward"], 0.5)

    def test_ignored_logged_but_not_last_turn(self):
        cl.new_turn(heard_clean="first", path="kb", reply="A")
        cl.log_ignored("and then we went home", 0.62, 0.3,
                       "no signal; score 0.21")
        self.assertEqual(cl.last_turn()["heard_clean"], "first")
        recs = cl.iter_day(time.time())
        self.assertEqual(recs[-1]["path"], "ignored")
        self.assertFalse(recs[-1]["addressed"])
        self.assertEqual(recs[-1]["addr_reason"], "no signal; score 0.21")

    def test_note_heard_fills_gate_fields(self):
        cl.note_heard("luna what is the fee", conf=0.81, loudness=0.7,
                      reason="named")
        cl.new_turn(heard_clean="what is the fee", path="kb", reply="x")
        rec = cl.last_turn()
        self.assertEqual(rec["conf"], 0.81)
        self.assertEqual(rec["addr_reason"], "named")
        # consumed: the next turn does not inherit it
        cl.new_turn(heard_clean="what is the fee", path="kb", reply="x")
        self.assertIsNone(cl.last_turn()["conf"])

    def test_disabled_writes_nothing(self):
        cl.configure(enabled=False)
        cl.new_turn(heard_clean="hi", path="llm", reply="Hello")
        self.assertEqual(os.listdir(self.dir), [])
        self.assertEqual(cl.last_turn()["reply"], "Hello")

    def test_turn_ids_unique(self):
        ids = {cl.new_turn(path="llm", ts=1_700_000_000.0) for _ in range(5)}
        self.assertEqual(len(ids), 5)

    def test_corrupt_lines_skipped(self):
        cl.new_turn(heard_clean="ok", path="llm", reply="fine")
        with open(cl.path_for(time.time()), "a") as f:
            f.write("{not json\n")
        cl.new_turn(heard_clean="ok2", path="llm", reply="fine")
        self.assertEqual([r["heard_clean"] for r in cl.iter_day()],
                         ["ok", "ok2"])

    def test_prune_old_files(self):
        today = dt.date.today()
        old = (today - dt.timedelta(days=50)).isoformat()
        recent = (today - dt.timedelta(days=5)).isoformat()
        for d in (old, recent):
            open(cl.path_for(d), "w").close()
        self.assertEqual(cl.prune(45), 1)
        self.assertEqual(cl.days(), [recent])

    def test_never_raises_on_unwritable_dir(self):
        blocker = os.path.join(self.dir, "file")
        open(blocker, "w").close()
        cl.configure(log_dir=os.path.join(blocker, "sub"))
        tid = cl.new_turn(heard_clean="x", path="llm")   # prints once
        self.assertTrue(tid)
        cl.update_turn(tid, reward=1.0)
        self.assertEqual(cl.iter_day(), [])


if __name__ == "__main__":
    unittest.main()
