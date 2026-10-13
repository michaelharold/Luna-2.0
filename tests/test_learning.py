# tests/test_learning.py
"""Unit tests for the on-device learning loop (learning.py) and the persona
prompt it feeds. Temp dirs via configure(), a seeded RNG, a tiny bag-of-words
embedder and injected timestamps — no models, no sleeping:

    python -m unittest tests.test_learning -v
"""

import datetime as dt
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
import zlib
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Point the face store at a throwaway dir BEFORE anything imports config.
_tmpdir = tempfile.mkdtemp(prefix="luna-test-faces-")
import config  # noqa: E402
config.FACE_DB_DIR = _tmpdir

import conversation_log as cl  # noqa: E402
import learning as L           # noqa: E402
import persona                 # noqa: E402

T0 = dt.datetime(2026, 9, 24, 10, 0, 0).timestamp()   # a Thursday morning
ARM = "one_sentence|playful|ask"
CTX = "known|kb|morning"


def bow_embed(texts):
    """Deterministic bag-of-words embedding (hashed, L2-normalised)."""
    out = np.zeros((len(texts), 256), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z0-9']+", t.lower()):
            out[i, zlib.crc32(w.encode()) % 256] += 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-9)


class LearningTestBase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="luna-test-learn-")
        cl.configure(enabled=True, log_dir=os.path.join(self.dir, "logs"))
        L.configure(enabled=True, style_enabled=True,
                    dir=os.path.join(self.dir, "learning"),
                    pending_path=os.path.join(self.dir, "learning",
                                              "pending.txt"),
                    knowledge_path=os.path.join(self.dir, "knowledge.txt"),
                    diary_dir=os.path.join(self.dir, "diary"),
                    diary_enabled=True, diary_hour=22,
                    min_turns_to_steer=8, rng=1234)
        L.set_embedder(bow_embed)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def turn(self, q="what are the lab timings", reply="Labs run 9 to 5.",
             ts=T0, path="llm", person=None, arm=ARM, ctx=CTX, total=3.0,
             kb=None):
        cl.new_turn(ts=ts, heard_clean=q, reply=reply, path=path,
                    person=person, style_arm=arm, context=ctx,
                    kb_matches=kb or [], timings={"ttft_s": 1.0,
                                                  "total_s": total})
        return cl.last_turn()


class TestStyleBandit(LearningTestBase):
    def test_eight_distinct_arms_and_instructions(self):
        self.assertEqual(len(L.ARMS), 8)
        instr = {L.style_instruction(a) for a in L.ARMS}
        self.assertEqual(len(instr), 8)
        for s in instr:
            self.assertNotIn("\n", s)
            self.assertTrue(s.endswith("."))
        self.assertEqual(L.style_instruction("nope"), "")

    def test_context_key(self):
        self.assertEqual(L.context_key("Alice", "kb", T0), "known|kb|morning")
        self.assertEqual(L.context_key(None, "scope", T0 + 13 * 3600),
                         "stranger|scope|night")
        self.assertEqual(L.question_kind("what is the fee", 0.8), "kb")
        self.assertEqual(L.question_kind("what's the weather tomorrow", 0.1),
                         "scope")
        self.assertEqual(L.question_kind("what is photosynthesis", 0.1),
                         "general")

    def test_uniform_before_min_turns(self):
        rng = np.random.default_rng(0)
        counts = Counter(L.choose_style(CTX, rng=rng)[0] for _ in range(800))
        self.assertEqual(set(counts), set(L.ARMS))
        self.assertGreater(min(counts.values()), 60)     # ~100 each

    def test_update_math_and_persistence(self):
        L.update(ARM, CTX, 1.0)
        L.update(ARM, CTX, -1.0)
        L.update(ARM, CTX, 0.0)
        st = L.bandit_state()[CTX]
        self.assertEqual(st["_n"], 3)
        self.assertAlmostEqual(st[ARM][0], 1 + 1 + 0 + 0.5)
        self.assertAlmostEqual(st[ARM][1], 1 + 0 + 1 + 0.5)
        path = os.path.join(self.dir, "learning", "bandit.json")
        self.assertTrue(os.path.exists(path))
        self.assertFalse(os.path.exists(path + ".tmp"))
        L.configure(rng=5)                   # drop memory, reload from disk
        self.assertEqual(L.bandit_state()[CTX]["_n"], 3)

    def test_thompson_steers_to_rewarded_arm(self):
        good = "two_or_three|plain_warm|no_ask"
        for _ in range(12):
            L.update(good, CTX, 1.0)
            for a in L.ARMS:
                if a != good:
                    L.update(a, CTX, -1.0)
        rng = np.random.default_rng(7)
        picks = Counter(L.choose_style(CTX, rng=rng)[0] for _ in range(200))
        self.assertGreater(picks[good], 180)

    def test_style_off(self):
        L.configure(style_enabled=False)
        self.assertEqual(L.choose_style(CTX), (None, None))


class TestRewards(LearningTestBase):
    def test_explicit_positive(self):
        self.turn()
        r = L.on_utterance("good answer", now=T0 + 5)
        self.assertAlmostEqual(r, 1.0)               # 1.0 + engaged, clipped
        t = cl.last_turn()
        self.assertEqual(t["reward"], 1.0)
        self.assertIn("explicit", [s for s, _ in t["reward_signals"]])

    def test_thanks_plus_engaged(self):
        self.turn()
        r = L.on_utterance("oh thanks", now=T0 + 4)
        self.assertAlmostEqual(r, 0.75)

    def test_engaged_only(self):
        self.turn()
        self.assertAlmostEqual(
            L.on_utterance("and where is the library", now=T0 + 6), 0.25)

    def test_repeat_question_negative(self):
        self.turn(q="what are the lab timings")
        r = L.on_utterance("what are the lab timings", now=T0 + 6)
        self.assertAlmostEqual(r, -0.5)

    def test_correction_negative(self):
        self.turn()
        r = L.on_utterance("no that's wrong they close at four", now=T0 + 5)
        self.assertAlmostEqual(r, -1.0)

    def test_stale_or_other_person_ignored(self):
        self.turn(person="Alice")
        self.assertIsNone(L.on_utterance("thanks", person="Bob", now=T0 + 5))
        self.assertIsNone(L.on_utterance("thanks", person="Alice",
                                         now=T0 + L.REWARD_MAX_GAP + 1))

    def test_rewarded_only_once(self):
        self.turn()
        self.assertIsNotNone(L.on_utterance("thanks", now=T0 + 3))
        self.assertIsNone(L.on_utterance("thanks again", now=T0 + 6))

    def test_command_turn_not_scored(self):
        self.turn(path="command", reply=None, arm=None)
        self.assertIsNone(L.on_utterance("thanks", now=T0 + 3))

    def test_weak_positive_after_question_is_an_answer(self):
        self.turn(reply="Are you asking about the CS lab?")
        self.assertEqual(L.feedback_command("yes that's right"), 0)
        self.assertAlmostEqual(L.on_utterance("yes that's right",
                                              now=T0 + 4), 0.25)
        self.turn(reply="The CS lab opens at nine.", ts=T0 + 10)
        self.assertEqual(L.feedback_command("that's right"), 1)

    def test_feedback_command_short_only(self):
        self.turn()
        self.assertEqual(L.feedback_command("good answer"), 1)
        self.assertEqual(L.feedback_command("that's wrong"), -1)
        self.assertEqual(L.feedback_command(
            "good answer but can you also tell me where the canteen is"), 0)

    def test_feedback_command_needs_a_turn(self):
        self.assertEqual(L.feedback_command("good answer"), 0)

    def test_explicit_feedback(self):
        self.turn(ts=T0)
        self.assertTrue(L.explicit_feedback(-1, now=T0 + 5))
        self.assertEqual(cl.last_turn()["reward"], -1.0)
        self.assertFalse(L.explicit_feedback(+1, now=T0 + 6))  # already done

    def test_thumbs_up_pending(self):
        self.turn()
        L.note_gesture("thumbs_up", now=T0 + 2)
        r = L.on_utterance("cool", now=T0 + 5)
        self.assertAlmostEqual(r, 1.0)               # .5 + .25 + .6 clipped
        names = [s for s, _ in cl.last_turn()["reward_signals"]]
        self.assertIn("thumbs", names)

    def test_person_left_abandon_and_thumbs(self):
        self.turn()
        self.assertAlmostEqual(L.on_person_left("Alice", now=T0 + 4), -0.2)
        self.turn(ts=T0 + 30)
        L.note_gesture("thumbs_up", now=T0 + 31)
        self.assertAlmostEqual(L.on_person_left(None, now=T0 + 33), 0.6)
        self.turn(ts=T0 + 60)
        self.assertIsNone(L.on_person_left(None, now=T0 + 60 + 11))

    def test_departure_tick(self):
        # no camera: never penalised
        self.turn()
        for s in range(5):
            self.assertIsNone(L.departure_tick(0, None, now=T0 + s))
        # a face during the turn, then gone for DEPART_SECS
        L.configure()                        # fresh presence tracking
        self.turn(ts=T0 + 100)
        L.departure_tick(1, None, now=T0 + 99)
        self.assertIsNone(L.departure_tick(0, None, now=T0 + 101))
        r = L.departure_tick(0, None, now=T0 + 101 + L.DEPART_SECS)
        self.assertAlmostEqual(r, -0.2)

    def test_bandit_and_profile_updated_by_reward(self):
        self.turn(person="Alice")
        L.on_utterance("thanks", person="Alice", now=T0 + 3)
        self.assertEqual(L.bandit_state()[CTX]["_n"], 1)
        p = L.profiles()["Alice"]
        self.assertEqual(p["rewarded"], 1)
        self.assertAlmostEqual(p["mean_reward"], 0.75)

    def test_broken_embedder_never_raises(self):
        def boom(texts):
            raise RuntimeError("no model")
        L.set_embedder(boom)
        self.turn()
        self.assertAlmostEqual(L.on_utterance("thanks", now=T0 + 3), 0.75)
        self.assertEqual(L.exemplars_for("anything"), [])


class TestExemplars(LearningTestBase):
    def test_admit_and_retrieve(self):
        self.turn(q="what are the lab timings", reply="Nine to five!")
        L.on_utterance("thanks", now=T0 + 3)         # 0.75 >= 0.5
        ex = L.exemplars_for("what are the lab timings today")
        self.assertEqual(ex, [("what are the lab timings", "Nine to five!")])
        self.assertEqual(L.exemplars_for("tell me a joke about cats"), [])
        # survives a reload (json + npz)
        L.configure()
        L.set_embedder(bow_embed)
        self.assertEqual(len(L.exemplars_for("lab timings")), 1)

    def test_low_reward_and_kb_path_not_admitted(self):
        self.turn()
        L.on_utterance("and the library", now=T0 + 3)  # 0.25 < 0.5
        self.turn(path="kb", ts=T0 + 20)
        L.on_utterance("thanks", now=T0 + 23)
        self.assertEqual(L.all_exemplars(), [])

    def test_cap(self):
        old = L.EXEMPLAR_CAP
        L.EXEMPLAR_CAP = 3
        try:
            for i in range(5):
                ts = T0 + 100 * i
                self.turn(q=f"question number {i} about topic{i}x",
                          reply=f"answer {i}", ts=ts)
                L.on_utterance("thanks", now=ts + 3)
            self.assertEqual(len(L.all_exemplars()), 3)
        finally:
            L.EXEMPLAR_CAP = old


class TestProfiles(LearningTestBase):
    def test_visits_and_context(self):
        L.note_visit("Alice", now=T0)
        L.note_visit("Alice", now=T0 + 60)           # same visit
        self.assertEqual(L.profiles()["Alice"]["visits"], 1)
        self.assertIsNone(L.profile_context("Alice", now=T0 + 60))
        L.note_visit("Alice", now=T0 + 2 * 86400)
        for i in range(2):
            t = self.turn(person="Alice", ts=T0 + 2 * 86400 + 10 * i,
                          ctx="known|kb|morning",
                          kb=[["what are the lab timings", 0.8]])
            L.observe_turn(t)
        s = L.profile_context("Alice", now=T0 + 2 * 86400 + 30)
        self.assertIn("2 visits", s)
        self.assertIn("2 days ago", s)
        self.assertIn("lab timings", s)
        self.assertTrue(s.endswith("."))
        self.assertIsNone(L.profile_context("Nobody"))


class TestFacts(LearningTestBase):
    def test_teach_approve_reject(self):
        kp = os.path.join(self.dir, "knowledge.txt")
        with open(kp, "w") as f:
            f.write("what is the name of the college:TKM")   # no newline
        self.assertTrue(L.teach("the lab closes at five", "Alice",
                                question="when does the lab close"))
        self.assertTrue(L.teach("the canteen serves biryani on fridays"))
        self.assertFalse(L.teach("   "))
        items = L.pending_facts()
        self.assertEqual(len(items), 2)
        self.assertIsNone(L.approve_fact(2))         # no question given
        line = L.approve_fact(1)
        self.assertEqual(line, "when does the lab close:the lab closes at five")
        with open(kp) as f:
            self.assertEqual(f.read().splitlines(),
                             ["what is the name of the college:TKM", line])
        self.assertEqual(len(L.pending_facts()), 1)
        line = L.approve_fact(1, "what: does the canteen serve?")
        self.assertTrue(line.startswith("what  does the canteen serve:"))
        self.assertEqual(L.pending_facts(), [])
        L.teach("x is y")
        self.assertEqual(L.reject_fact(1)["statement"], "x is y")
        self.assertIsNone(L.reject_fact(1))


class TestDiary(LearningTestBase):
    def _day_of_turns(self, n):
        for i in range(n):
            self.turn(q=f"question {i} about the library", reply=f"answer {i}",
                      ts=T0 + 60 * i)

    def test_hour_turns_and_once(self):
        calls = []

        def fake_llm(messages):
            calls.append(messages)
            return ("Today many visitors asked about the library. I mixed up "
                    "one answer. Tomorrow I want to be clearer. Extra sentence.")
        self._day_of_turns(2)
        evening = T0 + 12 * 3600                       # 22:00
        self.assertFalse(L.maybe_write_diary(fake_llm, now=evening))  # 2 < 3
        self._day_of_turns(3)
        self.assertFalse(L.maybe_write_diary(fake_llm, now=T0))       # 10:00
        self.assertTrue(L.maybe_write_diary(fake_llm, now=evening))
        self.assertFalse(L.maybe_write_diary(fake_llm, now=evening + 60))
        self.assertEqual(len(calls), 1)
        self.assertIn("library", calls[0][1]["content"])
        with open(L.diary_path(evening)) as f:
            text = f.read().strip()
        self.assertEqual(len(re.split(r"(?<=[.!?])\s+", text)), 3)
        hit = L.diary_context("many visitors asked about the library")
        self.assertIn("library", hit)
        self.assertIsNone(L.diary_context("zebra quantum xylophone"))

    def test_llm_failure_is_harmless(self):
        self._day_of_turns(3)

        def broken(messages):
            raise RuntimeError("model gone")
        self.assertFalse(L.maybe_write_diary(broken,
                                             now=T0 + 12 * 3600))


class TestReplay(LearningTestBase):
    def test_replay_rebuilds_bandit(self):
        self.turn(person="Alice")
        L.on_utterance("thanks", person="Alice", now=T0 + 3)
        turns = cl.iter_day(T0)
        L.reset(delete_files=True)
        self.assertEqual(L.bandit_state(), {})
        self.assertEqual(L.replay_turns(turns), 1)
        self.assertEqual(L.bandit_state()[CTX]["_n"], 1)
        self.assertEqual(L.profiles()["Alice"]["turns"], 1)
        self.assertEqual(len(L.all_exemplars()), 1)


class TestPersonaLearningFields(unittest.TestCase):
    CTX = dict(kb_snippets=[("q1", "a1", 0.9)],
               web_snippets=[("site", "chunk", 0.5)],
               scene="a person smiling", person="Alice")

    def test_byte_identical_when_empty(self):
        expected = (
            "The person talking to you is Alice (you recognised their face)."
            "\n\nFacts from your knowledge base that may be relevant:\n- q1: a1"
            "\n\nExtracts from the college website that may be relevant:\n"
            "- chunk"
            "\n\nWhat your camera sees RIGHT NOW (answer from this, in your "
            "own warm words):\na person smiling"
            "\n\nAnswer only from the facts above. If they do not answer "
            "the question, say you do not have that information. Never make "
            "up numbers, names or dates.")
        self.assertEqual(persona.format_context(**self.CTX), expected)
        self.assertEqual(persona.format_context(
            **self.CTX, style=None, exemplars=(), profile=None, diary=None),
            expected)
        self.assertEqual(persona.format_context(), "")
        self.assertEqual(persona.build_user_turn("hi", style=None,
                                                 exemplars=()), "hi")

    def test_style_is_last_line(self):
        out = persona.format_context(
            **self.CTX, style="Answer in one short sentence.",
            exemplars=[("q", "a")], profile="You have met Alice twice.",
            diary="On 2026-09-23: I learned a lot.")
        self.assertEqual(out.splitlines()[-1],
                         "Style for this reply: Answer in one short sentence.")
        self.assertIn("You have met Alice twice.", out)
        self.assertIn("- Q: q\n  A: a", out)
        self.assertIn("From your diary: On 2026-09-23", out)
        msg = persona.build_user_turn("where is the lab",
                                      style="Be brief.")
        self.assertEqual(msg, "Style for this reply: Be brief.\n\n"
                              "Question: where is the lab")

    def test_turn_with_data_is_query_data_instructions(self):
        msg = persona.build_user_turn(
            "who is the hod", web_snippets=[("u", "HOD page: Dr. A", 0.6)],
            style="Be brief.")
        self.assertTrue(msg.startswith("Query: who is the hod\n\nData:\n"))
        self.assertIn("- HOD page: Dr. A", msg)
        data, instr = msg.split("\n\nInstructions: ")
        self.assertNotIn("Answer only from the facts above", data)
        self.assertTrue(instr.startswith('Answer the query "who is the hod"'))
        self.assertTrue(msg.endswith("Style for this reply: Be brief."))


if __name__ == "__main__":
    unittest.main()
