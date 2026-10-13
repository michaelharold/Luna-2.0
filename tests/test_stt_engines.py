# tests/test_stt_engines.py
"""Unit tests for stt_engines.py — the engine table and text normalisation.
No model files needed:

    python -m unittest discover tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import stt_engines  # noqa: E402


class TestNormalize(unittest.TestCase):
    def test_vosk_style_text(self):
        self.assertEqual(stt_engines.normalize("What's the fee, Luna?"),
                         "what's the fee luna")

    def test_college_terms_spelt_like_the_knowledge_base(self):
        n = stt_engines.normalize
        self.assertEqual(n("Fee for the B.Tech program?"),
                         "fee for the b tech programme")
        self.assertEqual(n("btec programs"), "b tech programmes")
        self.assertEqual(n("the b take program"), "the b tech programme")
        self.assertEqual(n("M.Tech in AI"), "m tech in ai")

    def test_real_words_are_left_alone(self):
        n = stt_engines.normalize
        self.assertEqual(n("a tech talk"), "a tech talk")
        self.assertEqual(n("hostile facilities"), "hostile facilities")
        self.assertEqual(n("programmer"), "programmer")

    def test_non_speech_marks_are_dropped(self):
        n = stt_engines.normalize
        self.assertEqual(n("[BLANK_AUDIO]"), "")
        self.assertEqual(n("(laughs)"), "")
        self.assertEqual(n("*coughs* hello Luna"), "hello luna")


class TestEngineTable(unittest.TestCase):
    def test_numbers_and_names(self):
        self.assertEqual(sorted(stt_engines.ENGINES), [0, 1, 2, 3])
        self.assertEqual(stt_engines.name(0), "Vosk")
        self.assertEqual(stt_engines.name(9), "unknown")

    def test_vosk_and_unknown_never_load(self):
        self.assertFalse(stt_engines.available(0))
        self.assertFalse(stt_engines.available(9))
        self.assertIsNone(stt_engines.load(0))

    def test_config_picks_a_known_engine(self):
        import config
        self.assertIn(config.STT_ENGINE, stt_engines.ENGINES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
