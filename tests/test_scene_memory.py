# tests/test_scene_memory.py
"""Unit tests for scene memory (scene_memory.py) and the pure proactive
policy that uses it (proactive.choose_greeting / should_start_conversation).
Every timestamp is injected, so nothing here sleeps or needs a camera:

    python -m unittest tests.test_scene_memory -v
"""

import os
import random
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Point the face store at a throwaway dir BEFORE anything imports config.
_tmpdir = tempfile.mkdtemp(prefix="luna-test-faces-")
import config  # noqa: E402
config.FACE_DB_DIR = _tmpdir

import scene_memory as sm  # noqa: E402

T0 = 10_000.0


def feed(face_count, name, emotion, start, secs, step=0.5, objects=None):
    """Feed identical ticks from `start` for `secs` seconds; returns end time."""
    t = start
    while t <= start + secs + 1e-9:
        sm.observe(face_count, name, emotion, objects=objects, now=t)
        t += step
    return start + secs


def kinds():
    return [e["kind"] for e in sm.last_events()]


class TestSceneEvents(unittest.TestCase):
    def setUp(self):
        sm.reset()

    def test_arrival_is_one_event_not_one_per_frame(self):
        feed(1, "Alice", "Neutral", T0, 10)
        self.assertEqual(kinds(), ["arrived"])
        self.assertEqual(sm.last_events()[0]["who"], "Alice")

    def test_short_dropout_is_not_a_departure(self):
        feed(1, "Alice", "Neutral", T0, 5)
        sm.observe(0, None, "Neutral", now=T0 + 5.5)
        sm.observe(0, None, "Neutral", now=T0 + 6.5)      # 1.0 s gone
        feed(1, "Alice", "Neutral", T0 + 7, 3)
        self.assertEqual(kinds(), ["arrived"])
        self.assertIsNone(sm.seconds_since_left("Alice", now=T0 + 10))

    def test_departure_after_debounce(self):
        feed(1, "Alice", "Neutral", T0, 5)
        feed(0, None, "Neutral", T0 + 5.5, sm.LEAVE_DEBOUNCE_SECS + 0.5)   # gone long enough
        self.assertEqual(kinds(), ["arrived", "left"])
        left = sm.last_events()[-1]
        self.assertEqual(left["who"], "Alice")
        self.assertEqual(left["ts"], T0 + 5.5)             # when she vanished
        self.assertIsNone(sm.present_since(
            now=T0 + 6.5 + sm.LEAVE_DEBOUNCE_SECS))

    def test_recognition_flicker_is_ignored(self):
        t = T0
        for i in range(20):
            sm.observe(1, "Alice" if i % 3 else None, "Neutral", now=t)
            t += 0.5
        self.assertEqual(kinds(), ["arrived"])

    def test_anonymous_then_recognised_is_one_arrival(self):
        feed(1, None, "Sad", T0, 3)                        # mood while anonymous
        feed(1, "Alice", "Neutral", T0 + 3.5, 2)
        evs = sm.last_events()
        self.assertEqual([e["kind"] for e in evs], ["arrived", "mood"])
        self.assertTrue(all(e["who"] == "Alice" for e in evs))
        self.assertAlmostEqual(sm.present_since(now=T0 + 10), 10.0)

    def test_new_name_must_hold_before_it_replaces(self):
        feed(1, "Alice", "Neutral", T0, 5)
        sm.observe(2, "Bob", "Neutral", now=T0 + 5.5)       # one-frame flip
        feed(2, "Alice", "Neutral", T0 + 6, 3)
        self.assertEqual(kinds(), ["arrived"])
        feed(1, "Bob", "Neutral", T0 + 10, 3)               # Bob really took over
        self.assertEqual(kinds(), ["arrived", "left", "arrived"])
        self.assertEqual(sm.presence(now=T0 + 13)["name"], "Bob")
        self.assertIsNotNone(sm.seconds_since_left("Alice", now=T0 + 13))

    def test_mood_needs_to_be_held(self):
        feed(1, "Alice", "Neutral", T0, 2)
        feed(1, "Alice", "Happy", T0 + 2.5, 1.0)            # 1 s of smiling
        feed(1, "Alice", "Neutral", T0 + 4, 1)
        self.assertNotIn("mood", kinds())
        feed(1, "Alice", "Happy", T0 + 6, 3.0)              # 3 s of smiling
        self.assertEqual(kinds().count("mood"), 1)          # once, not per tick
        mood = [e for e in sm.last_events() if e["kind"] == "mood"][0]
        self.assertEqual((mood["who"], mood["detail"]), ("Alice", "Happy"))

    def test_neutral_is_never_a_mood(self):
        feed(1, "Alice", "Neutral", T0, 20)
        self.assertNotIn("mood", kinds())

    def test_objects_only_when_new(self):
        feed(1, "Alice", "Neutral", T0, 1, objects=["laptop", "cup"])
        feed(1, "Alice", "Neutral", T0 + 2, 1, objects=["laptop", "Book"])
        obj = [e["detail"] for e in sm.last_events() if e["kind"] == "objects"]
        self.assertEqual(obj, ["laptop, cup", "book"])

    def test_seconds_since_left(self):
        self.assertIsNone(sm.seconds_since_left("Alice", now=T0))
        self.assertIsNone(sm.seconds_since_left(None, now=T0))
        feed(1, "Alice", "Neutral", T0, 5)
        feed(0, None, "Neutral", T0 + 5.5, sm.LEAVE_DEBOUNCE_SECS + 0.5)
        self.assertAlmostEqual(sm.seconds_since_left("Alice", now=T0 + 65.5), 60.0)
        feed(1, "Alice", "Neutral", T0 + 70, 1)             # still answers when back
        self.assertAlmostEqual(sm.seconds_since_left("Alice", now=T0 + 71), 65.5)

    def test_event_cap(self):
        for i in range(sm.SCENE_MEMORY_EVENTS + 10):
            sm.note("asked", "Alice", f"q{i}", now=T0 + i)
        evs = sm.last_events()
        self.assertEqual(len(evs), sm.SCENE_MEMORY_EVENTS)
        self.assertEqual(evs[-1]["detail"], f"q{sm.SCENE_MEMORY_EVENTS + 9}")
        self.assertEqual(len(sm.last_events(3)), 3)
        self.assertEqual(sm.last_events(0), [])

    def test_max_age_prunes(self):
        sm.note("greeted", "Alice", "Hi", now=T0)
        sm.note("greeted", "Bob", "Hi", now=T0 + sm.SCENE_MEMORY_MAX_AGE)
        self.assertEqual(len(sm.last_events()), 2)
        sm.summary(now=T0 + sm.SCENE_MEMORY_MAX_AGE + 1)   # prunes on read
        self.assertEqual([e["who"] for e in sm.last_events()], ["Bob"])

    def test_observe_and_summary_never_raise(self):
        sm.observe("garbage", object(), 42, objects=[None, 3], now=T0)
        sm.observe(None, None, None, objects=5, now=T0 + 1)  # not iterable
        self.assertIsInstance(sm.summary(now=T0 + 2), str)

    def test_reset(self):
        feed(1, "Alice", "Happy", T0, 5)
        sm.reset()
        self.assertEqual(sm.last_events(), [])
        self.assertIsNone(sm.present_since(now=T0 + 6))
        self.assertEqual(sm.summary(now=T0 + 6), "")


class TestSceneSummary(unittest.TestCase):
    def setUp(self):
        sm.reset()

    def test_empty(self):
        self.assertEqual(sm.summary(now=T0), "")

    def test_current_person_with_duration_and_mood(self):
        feed(1, "Alice", "Happy", T0, 180)
        s = sm.summary(now=T0 + 180)
        self.assertIn("Alice has been in front of you for 3 minutes", s)
        self.assertIn("looks happy", s)

    def test_just_arrived(self):
        sm.observe(1, "Alice", "Neutral", now=T0)
        self.assertEqual(sm.summary(now=T0 + 1), "Alice just arrived.")

    def test_stranger(self):
        feed(1, None, "Neutral", T0, 3)
        self.assertEqual(sm.summary(now=T0 + 3),
                         "One person you have not met is here.")

    def test_somebody_left(self):
        feed(1, "Bob", "Neutral", T0, 30)
        feed(0, None, "Neutral", T0 + 30.5, sm.LEAVE_DEBOUNCE_SECS + 0.5)
        s = sm.summary(now=T0 + 30.5 + 300)
        self.assertEqual(s, "Nobody is in front of you right now. "
                            "Bob stopped by 5 minutes ago.")

    def test_earlier_mood_of_current_person(self):
        feed(1, "Alice", "Sad", T0, 5)
        feed(1, "Alice", "Neutral", T0 + 5.5, 120)
        s = sm.summary(now=T0 + 125.5)
        self.assertIn("Alice looked sad 2 minutes ago.", s)
        self.assertNotIn("looks sad", s)

    def test_luna_question_is_top_history(self):
        feed(1, "Bob", "Neutral", T0, 5)
        feed(0, None, "Neutral", T0 + 5.5, 2)
        feed(1, "Alice", "Neutral", T0 + 20, 30)
        sm.note("asked", "Alice", "How is your day going so far, Alice?",
                now=T0 + 50)
        s = sm.summary(now=T0 + 55)
        self.assertIn('You asked Alice just now: "How is your day going so '
                      'far, Alice?"', s)
        # … and after a couple of minutes other history is more useful again
        s = sm.summary(now=T0 + 50 + sm.ASKED_RELEVANT_SECS + 5)
        self.assertIn("Bob stopped by", s)

    def test_at_most_two_sentences(self):
        feed(1, "Bob", "Happy", T0, 10, objects=["laptop"])
        feed(0, None, "Neutral", T0 + 10.5, 2)
        feed(1, "Alice", "Sad", T0 + 20, 10)
        s = sm.summary(now=T0 + 30)
        self.assertLessEqual(s.count(". ") + 1, 2, s)

    def test_asked_recently(self):
        self.assertFalse(sm.asked_recently(now=T0))
        sm.note("asked", "Alice", "How was your day?", now=T0)
        self.assertTrue(sm.asked_recently(now=T0 + 30))
        self.assertFalse(sm.asked_recently(now=T0 + sm.ASKED_RELEVANT_SECS + 1))

    def test_human_durations(self):
        self.assertEqual(sm.human_duration(3), "a moment")
        self.assertEqual(sm.human_duration(40), "40 seconds")
        self.assertEqual(sm.human_duration(90), "a minute")
        self.assertEqual(sm.human_duration(600), "10 minutes")
        self.assertEqual(sm.human_duration(3700), "an hour")
        self.assertEqual(sm.human_ago(4), "just now")
        self.assertEqual(sm.human_ago(30), "a moment ago")
        self.assertEqual(sm.human_ago(300), "5 minutes ago")


class TestProactivePolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import proactive
        cls.p = proactive

    def test_greeting_vs_welcome_back(self):
        rng = random.Random(1)
        first = self.p.choose_greeting("Alice", None, rng)
        self.assertIn(first, [g.format(name="Alice") for g in config.GREET_PHRASES])
        back = self.p.choose_greeting("Alice", 120, rng)
        self.assertIn(back,
                      [g.format(name="Alice") for g in config.WELCOME_BACK_PHRASES])
        long_gone = self.p.choose_greeting("Alice", config.WELCOME_BACK_SECS + 1, rng)
        self.assertIn(long_gone,
                      [g.format(name="Alice") for g in config.GREET_PHRASES])

    def test_starter_fires_once_per_visit(self):
        visit, greeted = T0, T0 + 1
        started = set()
        gap = self.p.STARTER_AFTER_GREET_SECS
        fired = []
        t = T0
        while t < T0 + gap + 120:
            if self.p.should_start_conversation("Alice", t, visit, greeted, started):
                fired.append(t)
                started.add(visit)
            t += 0.5
        self.assertEqual(len(fired), 1)
        self.assertGreaterEqual(fired[0] - greeted, gap)
        self.assertGreaterEqual(fired[0] - visit, config.SCENE_LINGER_SECS)
        # a new visit (new person_since) may get one again
        self.assertTrue(self.p.should_start_conversation(
            "Alice", t + 200, t + 100, greeted, started))

    def test_starter_blockers(self):
        ok = dict(name="Alice", now=T0 + 500, person_since=T0,
                  greeted_at=T0 + 1, started_visits=set())
        self.assertTrue(self.p.should_start_conversation(**ok))
        for k, v in (("name", None), ("person_since", 0.0),
                     ("greeted_at", None), ("started_visits", {T0}),
                     ("conversation_active", True), ("speaking", True),
                     ("now", T0 + 2)):
            self.assertFalse(self.p.should_start_conversation(**{**ok, k: v}), k)

    def test_is_question(self):
        self.assertTrue(self.p.is_question("Hey Alice! How's it going? "))
        self.assertFalse(self.p.is_question("Welcome back, Alice!"))


def tearDownModule():
    shutil.rmtree(_tmpdir, ignore_errors=True)
    sm.reset()


if __name__ == "__main__":
    unittest.main()
