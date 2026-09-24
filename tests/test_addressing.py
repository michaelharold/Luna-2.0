# tests/test_addressing.py
"""The "only when she's being talked to" gate: clear speech alone never
passes, loudness and a visible face count as addressing signals, and every
decision can be explained term by term for the log. Pure logic."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import addressing  # noqa: E402
from config import (ADDR_THRESHOLD, ADDR_W_CONF, ADDR_W_LOUDNESS,  # noqa: E402
                    ADDR_W_QUESTION, LOUDNESS_NEAR_DB, LOUDNESS_FAR_DB)


class TestLoudness(unittest.TestCase):
    def test_mapping(self):
        f = addressing.loudness_score
        self.assertEqual(f(LOUDNESS_FAR_DB), 0.0)
        self.assertEqual(f(LOUDNESS_NEAR_DB), 1.0)
        mid = (LOUDNESS_NEAR_DB + LOUDNESS_FAR_DB) / 2
        self.assertAlmostEqual(f(mid), 0.5, places=6)
        self.assertEqual(f(LOUDNESS_FAR_DB - 30), 0.0)
        self.assertEqual(f(0.0), 1.0)

    def test_garbage_levels(self):
        for bad in (None, float("nan"), float("-inf"), "x"):
            self.assertEqual(addressing.loudness_score(bad), 0.0)


class TestGate(unittest.TestCase):
    CHATTER = "and then we went home after the movie"

    def test_crisp_chatter_rejected(self):
        ok, why = addressing.decide(self.CHATTER, conf=0.95)
        self.assertFalse(ok)
        self.assertIn("no addressing signal", why)

    def test_chatter_face_and_loud_accepted(self):
        self.assertTrue(addressing.is_addressed(
            self.CHATTER, conf=0.95, face_visible=True, loudness=1.0))

    def test_quiet_far_question_matches_arithmetic(self):
        text = "what time does the bus leave"
        score = addressing.address_score(text, conf=0.8, loudness=0.1)
        expected = ADDR_W_CONF * 0.8 + ADDR_W_LOUDNESS * 0.1 + ADDR_W_QUESTION
        self.assertAlmostEqual(score, expected, places=6)
        self.assertEqual(addressing.is_addressed(text, conf=0.8, loudness=0.1),
                         expected >= ADDR_THRESHOLD)

    def test_loud_counts_as_signal(self):
        self.assertTrue(addressing.has_signal(self.CHATTER, loudness=0.9))
        self.assertFalse(addressing.has_signal(self.CHATTER, loudness=0.2))

    def test_face_or_name_mode(self):
        old = addressing.ADDR_FACE_OR_NAME
        try:
            addressing.ADDR_FACE_OR_NAME = True
            q = "what is the pin code of the college"
            self.assertFalse(addressing.decide(q, conf=1.0, kb_score=0.9,
                                               loudness=1.0)[0])
            self.assertTrue(addressing.decide(q, conf=1.0, kb_score=0.9,
                                              face_visible=True)[0])
            self.assertTrue(addressing.decide("luna " + q, conf=1.0,
                                              kb_score=0.9)[0])
        finally:
            addressing.ADDR_FACE_OR_NAME = old

    def test_explain_sums_to_score(self):
        args = ("luna can you tell me where the canteen is", 0.7)
        kw = dict(kb_score=0.6, face_visible=True, awaiting_answer=False,
                  loudness=0.5)
        parts = addressing.explain(*args, **kw)
        self.assertEqual(set(parts), {"conf", "loudness", "name", "question",
                                      "second_person", "kb", "face",
                                      "continuity"})
        self.assertAlmostEqual(sum(parts.values()),
                               addressing.address_score(*args, **kw), places=9)
        self.assertIn("=", addressing.format_explain(parts))

    def test_leading_name_still_question(self):
        self.assertTrue(addressing.is_question("luna what courses are offered"))
        self.assertTrue(addressing.is_question("hey luna where is the lab"))


if __name__ == "__main__":
    unittest.main()
