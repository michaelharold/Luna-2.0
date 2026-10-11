# tests/test_logic.py
"""Unit tests for Luna's pure-logic modules — runnable on any machine with
numpy (no camera, mic, pygame or models needed):

    python -m unittest discover tests -v

Covers: addressing (was that meant for Luna?), the face store's enrollment /
matching / forgetting, the expression registry, and scene descriptions.
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Point the face store at a throwaway dir BEFORE anything imports config.
_tmpdir = tempfile.mkdtemp(prefix="luna-test-faces-")
import config  # noqa: E402
config.FACE_DB_DIR = _tmpdir


class TestAddressing(unittest.TestCase):
    def setUp(self):
        import addressing
        self.addr = addressing

    def test_name_mention(self):
        self.assertTrue(self.addr.mentions_name("luna what time is it"))
        self.assertTrue(self.addr.mentions_name("hey luna"))
        self.assertFalse(self.addr.mentions_name("the lunatic is on the grass"))
        self.assertFalse(self.addr.mentions_name("nothing relevant here"))

    def test_question_shape(self):
        self.assertTrue(self.addr.is_question("what is the department vision"))
        self.assertTrue(self.addr.is_question("can you wave"))
        self.assertFalse(self.addr.is_question("that movie was great"))

    def test_named_question_accepted_even_mumbled(self):
        # Clearly addressed (name + question) should survive low confidence.
        self.assertTrue(self.addr.is_addressed(
            "luna what courses are offered", conf=0.45))

    def test_bystander_chatter_rejected(self):
        # A crisp remark clearly NOT aimed at Luna: no name, no question,
        # no second person, no KB relevance, nobody in front of the camera.
        self.assertFalse(self.addr.is_addressed(
            "and then we went home", conf=0.70))

    def test_continuity_rescues_answer(self):
        # Same chatter-ish sentence, but Luna just asked a question.
        self.assertTrue(self.addr.is_addressed(
            "the second one please", conf=0.70, awaiting_answer=True))

    def test_min_words_floor(self):
        self.assertFalse(self.addr.is_addressed("luna", conf=1.0))
        # ...but when Luna just asked ("what's your name?") one word answers
        self.assertTrue(self.addr.is_addressed("michael", conf=1.0,
                                               awaiting_answer=True))


class TestFaceStore(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import face_store
        self.np = np
        self.store = face_store
        # start clean each test
        for n in list(self.store.known_names()):
            self.store.forget(n)

    def _vec(self, seed):
        rng = self.np.random.default_rng(seed)
        return rng.normal(size=128).astype("float32")

    def test_enroll_and_match(self):
        base = self._vec(1)
        samples = [base + self.np.random.default_rng(i).normal(
            scale=0.01, size=128).astype("float32") for i in range(5)]
        self.assertTrue(self.store.enroll("alice", samples))
        name, score = self.store.match(base)
        self.assertEqual(name, "Alice")
        self.assertGreater(score, 0.9)

    def test_stranger_stays_stranger(self):
        self.store.enroll("alice", [self._vec(1)])
        name, _ = self.store.match(self._vec(99))
        self.assertIsNone(name)

    def test_forget(self):
        self.store.enroll("bob", [self._vec(2)])
        self.assertIn("Bob", self.store.known_names())
        self.assertTrue(self.store.forget("bob"))
        self.assertNotIn("Bob", self.store.known_names())
        self.assertFalse(self.store.forget("bob"))

    def test_reenroll_replaces(self):
        self.store.enroll("cara", [self._vec(3)])
        self.store.enroll("cara", [self._vec(4)])
        self.assertEqual(self.store.known_names().count("Cara"), 1)
        name, _ = self.store.match(self._vec(4))
        self.assertEqual(name, "Cara")

    def test_persistence(self):
        self.store.enroll("dev", [self._vec(5)])
        self.store.load()          # simulate restart
        name, _ = self.store.match(self._vec(5))
        self.assertEqual(name, "Dev")


class TestExpressions(unittest.TestCase):
    def test_registry_complete(self):
        from expressions import EXPRESSIONS
        for name in ("neutral", "happy", "sad", "angry", "surprised",
                     "excited", "love", "shy", "sleeping",
                     "curious", "annoyed", "amazed",
                     "boop_left", "boop_right", "giggle"):
            self.assertIn(name, EXPRESSIONS)

    def test_unknown_falls_back_to_neutral(self):
        from expressions import get
        self.assertEqual(get("no-such-mood").name, "neutral")

    def test_boops_mirror(self):
        from expressions import get
        left, right = get("boop_left"), get("boop_right")
        self.assertEqual(left.asym, tuple(reversed(right.asym)))

    def test_curious_is_asymmetric(self):
        from expressions import get
        a = get("curious").asym
        self.assertNotEqual(a[0], a[1])


class TestSceneDescribe(unittest.TestCase):
    def setUp(self):
        import scene_context
        self.sc = scene_context

    def test_empty_room(self):
        d = self.sc.describe(0, None, "Neutral", 0.5)
        self.assertIn("don't see anyone", d)

    def test_recognised_person_with_emotion(self):
        d = self.sc.describe(1, "Alice", "Happy", 0.5)
        self.assertIn("Alice", d)
        self.assertIn("happy", d)

    def test_objects_listed_without_person_dupes(self):
        d = self.sc.describe(1, None, "Neutral", 0.5,
                             objects=[("person", 1), ("cup", 2), ("laptop", 1)])
        self.assertIn("2 cups", d)
        self.assertIn("a laptop", d)
        # YOLO's person entry must not double-report the face pipeline's human
        self.assertNotIn("a person", d)

    def test_vision_query_detection(self):
        self.assertTrue(self.sc.is_vision_query("hey what do you see there"))
        self.assertTrue(self.sc.is_vision_query("who am i"))
        self.assertFalse(self.sc.is_vision_query("what is the fee structure"))


def tearDownModule():
    shutil.rmtree(_tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
