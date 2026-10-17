# tests/test_presence_context.py
"""Meaningful recognition, greetings and context — and a clean terminal:

    python -m unittest tests.test_presence_context -v

  * A face dropping out for a few seconds (looking down, turning away) is NOT
    leaving: vision keeps the name and the visit, scene memory records no
    departure, so nothing says "you're back again".
  * A known name is not dropped the moment one frame fails to match.
  * Luna greets a person once per visit — not again every two minutes while
    they sit there — and again only after they have really been away.
  * On an ordinary question the scene notes are background the model must
    not bring up, and they carry no guessed moods or objects; the live
    "what your camera sees" block is only for questions about the scene.
  * console: internals only print when LOG_VERBOSE is on; heard / source
    lines are short and readable.
"""

import io
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Point the face store at a throwaway dir BEFORE anything imports config.
_tmpdir = tempfile.mkdtemp(prefix="luna-test-faces-")
import config  # noqa: E402
config.FACE_DB_DIR = _tmpdir

import numpy as np  # noqa: E402

import console  # noqa: E402
import fetch  # noqa: E402
import proactive  # noqa: E402
import scene_memory as sm  # noqa: E402
from shared_state import state  # noqa: E402

T0 = 10_000.0
AWAY = config.PERSON_AWAY_SECS
RECOG_SW = config.RECOG_SWITCH_SECS


def feed(face_count, name, emotion, start, secs, step=0.5, objects=None):
    t = start
    while t <= start + secs + 1e-9:
        sm.observe(face_count, name, emotion, objects=objects, now=t)
        t += step
    return start + secs


# ══════════════════════════════════════════════════════════════════════════════
# Vision: who is in front of Luna
# ══════════════════════════════════════════════════════════════════════════════
class TestVisionPresence(unittest.TestCase):
    def setUp(self):
        try:
            import vision_module
        except Exception as e:                       # noqa: BLE001
            raise unittest.SkipTest(f"vision_module unavailable: {e}")
        self.vm = vision_module
        self.vm._clear_recognition()
        self.vm._face_run_start = self.vm._faceless_since = None
        self.result = ("Dee", 0.9)
        for p in (mock.patch.object(self.vm, "_embed",
                                    lambda f, r: np.ones(128)),
                  mock.patch.object(self.vm.face_store, "match",
                                    lambda e: self.result),
                  mock.patch.object(self.vm, "_log_sighting",
                                    lambda n, c: None),
                  mock.patch.object(self.vm, "_sight_moment", lambda n: None)):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.vm._clear_recognition)

    def recognise(self, now, n=2):
        for _ in range(n):
            self.vm._update_recognition(None, object(), now=now)

    def person(self):
        with state.lock:
            return state.person_name, state.person_since

    def test_brief_dropout_keeps_name_and_visit(self):
        self.recognise(T0)
        name, since = self.person()
        self.assertEqual(name, "Dee")
        t = T0
        while t < T0 + AWAY - 1:                       # face gone, not long
            self.vm._track_absence(False, t)
            t += 0.125
        self.vm._track_absence(True, t)
        self.recognise(t)
        self.assertEqual(self.person(), ("Dee", since))

    def test_real_absence_clears(self):
        self.recognise(T0)
        t = T0
        while t < T0 + AWAY + 1:
            self.vm._track_absence(False, t)
            t += 0.125
        self.assertEqual(self.person(), (None, 0.0))

    def test_one_failed_match_does_not_drop_the_name(self):
        self.recognise(T0)
        self.result = (None, 0.2)                      # bad angle
        self.recognise(T0 + 0.5, n=3)
        self.assertEqual(self.person()[0], "Dee")
        self.result = ("Dee", 0.9)
        self.recognise(T0 + 1.0)
        self.assertEqual(self.person()[0], "Dee")

    def test_looking_down_keeps_the_name(self):
        """Face never leaves the frame, but matches fail for 10 s (looking at
        the keyboard): still the same person — no drop, no re-greeting."""
        t = T0
        self.vm._track_absence(True, t)
        self.recognise(t)
        since = self.person()[1]
        self.result = (None, 0.2)
        while t < T0 + 10:
            t += 0.375
            self.vm._track_absence(True, t)
            self.vm._update_recognition(None, object(), now=t)
        self.assertEqual(self.person(), ("Dee", since))

    def test_after_a_gap_an_unknown_face_is_not_assumed(self):
        t = T0
        self.vm._track_absence(True, t)
        self.recognise(t)
        for k in range(1, 9):                       # 2 s with no face
            self.vm._track_absence(False, t + 0.25 * k)
        t += 2.5
        self.result = (None, 0.2)                   # someone unknown sits down
        while t < T0 + 2.5 + RECOG_SW + 1:
            self.vm._track_absence(True, t)
            self.vm._update_recognition(None, object(), now=t)
            t += 0.375
        self.assertIsNone(self.person()[0])

    def test_a_lasting_change_of_face_is_accepted(self):
        self.recognise(T0)
        self.result = ("Eve", 0.8)
        self.recognise(T0 + 0.5)
        self.assertEqual(self.person()[0], "Dee")      # not yet
        self.recognise(T0 + 0.5 + config.RECOG_SWITCH_SECS)
        self.assertEqual(self.person()[0], "Eve")

    def test_first_recognition_is_quick(self):
        self.recognise(T0)                             # two agreeing samples
        self.assertEqual(self.person()[0], "Dee")


# a match strong enough to learn from, yet new — between the configured
# bounds, whatever they are tuned to
GOOD = round((config.FACE_LEARN_MIN_SIM + config.FACE_LEARN_NOVEL_BELOW) / 2, 3)


class TestAutoLearn(unittest.TestCase):
    """Luna adds a new look for someone only when she is VERY sure."""

    def setUp(self):
        try:
            import vision_module
        except Exception as e:                       # noqa: BLE001
            raise unittest.SkipTest(f"vision_module unavailable: {e}")
        self.vm = vision_module
        self.ok = dict(name="Dee", conf=GOOD, published="Dee", since=T0,
                       faces=1, enrolling=False, runner_up=0.20,
                       last_learned=float("-inf"), now=T0 + 30)

    def test_precapture_while_the_name_is_asked(self):
        vm = self.vm
        vm._precapture.clear()
        with mock.patch.object(vm, "_embed", lambda f, d: np.ones(128)):
            with state.lock:
                state.enroll_precapture_until = time.time() + 5
                state.face_count = 1
            for _ in range(3):
                vm._update_precapture(None, object())
            with state.lock:
                state.face_count = 2                 # someone else in view
            vm._update_precapture(None, object())
        self.assertEqual(vm.take_precapture(), 3)
        self.assertEqual(len(vm._enroll_by_pose.get(0, [])), 3)
        vm._reset_enrollment()
        with state.lock:
            state.enroll_precapture_until = 0.0
            state.face_count = 0

    def test_all_conditions_hold(self):
        self.assertTrue(self.vm.should_learn(**self.ok))

    def test_each_condition_matters(self):
        for k, v in (("name", None),
                     ("published", "Eve"),                    # not the one shown
                     ("since", T0 + 25),                      # only 5 s steady
                     ("faces", 2),                            # someone else in view
                     ("enrolling", True),
                     ("conf", 0.50),                          # not sure enough
                     ("conf", 0.85),                          # look already known
                     ("runner_up", GOOD - config.FACE_LEARN_MARGIN / 2),                     # too close to another
                     ("last_learned", T0 + 30 - 60)):         # learned a minute ago
            with self.subTest(**{k: v}):
                self.assertFalse(self.vm.should_learn(**{**self.ok, k: v}))

    def test_five_steady_sightings_make_one_look(self):
        vm = self.vm
        added = []
        with mock.patch.object(vm.face_store, "add_template",
                               lambda n, e: added.append((n, e)) or True), \
                mock.patch.object(vm.face_store, "runner_up", lambda e, n: 0.1):
            with state.lock:
                state.person_name, state.person_since = "Dee", T0
                state.face_count, state.enroll_request = 1, None
            vm._learned_at.clear()
            for i in range(vm.FACE_LEARN_SAMPLES):
                learned = vm._maybe_learn("Dee", GOOD, np.full(128, i + 1.0),
                                          T0 + 30 + i)
            self.assertTrue(learned)
            self.assertEqual(len(added), 1)
            self.assertEqual(added[0][0], "Dee")
            np.testing.assert_allclose(added[0][1], np.full(128, 3.0))
            # ... and not again for FACE_LEARN_EVERY_SECS
            for i in range(10):
                vm._maybe_learn("Dee", GOOD, np.ones(128), T0 + 60 + i)
            self.assertEqual(len(added), 1)
        with state.lock:
            state.person_name, state.person_since, state.face_count = None, 0.0, 0

    def test_one_bad_sighting_restarts_the_count(self):
        vm = self.vm
        added = []
        with mock.patch.object(vm.face_store, "add_template",
                               lambda n, e: added.append(n) or True), \
                mock.patch.object(vm.face_store, "runner_up", lambda e, n: 0.1):
            with state.lock:
                state.person_name, state.person_since = "Dee", T0
                state.face_count, state.enroll_request = 1, None
            vm._learned_at.clear()
            for i in range(vm.FACE_LEARN_SAMPLES - 1):
                vm._maybe_learn("Dee", GOOD, np.ones(128), T0 + 30 + i)
            vm._maybe_learn("Dee", 0.40, np.ones(128), T0 + 40)     # unsure
            vm._maybe_learn("Dee", GOOD, np.ones(128), T0 + 41)
            self.assertEqual(added, [])
        with state.lock:
            state.person_name, state.person_since, state.face_count = None, 0.0, 0


# ══════════════════════════════════════════════════════════════════════════════
# Scene memory
# ══════════════════════════════════════════════════════════════════════════════
class TestScenePresence(unittest.TestCase):
    def setUp(self):
        sm.reset()

    def test_brief_dropout_is_not_a_departure(self):
        feed(1, "Michael", "Neutral", T0, 60)
        feed(0, None, "Neutral", T0 + 60.5, AWAY - 5)  # looked away
        feed(1, "Michael", "Neutral", T0 + 60.5 + AWAY - 4, 20)
        kinds = [e["kind"] for e in sm.last_events()]
        self.assertNotIn("left", kinds)
        s = sm.summary(now=T0 + 60.5 + AWAY + 20)
        self.assertNotIn("came back", s)
        self.assertNotIn("just arrived", s)

    def test_real_absence_is_a_departure(self):
        feed(1, "Michael", "Neutral", T0, 60)
        feed(0, None, "Neutral", T0 + 60.5, AWAY + 2)
        self.assertIn("left", [e["kind"] for e in sm.last_events()])
        self.assertIsNotNone(sm.seconds_since_left("Michael",
                                                   now=T0 + 60.5 + AWAY + 2))

    def test_people_only_summary(self):
        feed(1, "Alice", "Sad", T0, 5, objects=["laptop"])
        feed(1, "Alice", "Surprise", T0 + 5.5, 175)
        full = sm.summary(now=T0 + 180)
        self.assertIn("looks surprised", full)
        quiet = sm.summary(now=T0 + 180, moods=False, objects=False)
        self.assertIn("Alice has been in front of you for 3 minutes", quiet)
        for word in ("looks", "looked", "laptop", "noticed"):
            self.assertNotIn(word, quiet)


# ══════════════════════════════════════════════════════════════════════════════
# Greetings
# ══════════════════════════════════════════════════════════════════════════════
class TestGreetOncePerVisit(unittest.TestCase):
    def should(self, now, visit, greeted, last=None, seen=True):
        return proactive.should_greet("Michael", seen, visit, greeted,
                                      last, now)

    def test_greets_a_new_visit(self):
        self.assertTrue(self.should(T0, visit=T0, greeted={}))

    def test_never_twice_in_one_visit(self):
        greeted = {"Michael": T0}
        for later in (60, 121, 600, 3600):           # still sitting there
            self.assertFalse(self.should(T0 + later, T0, greeted, last=T0))

    def test_greets_again_after_a_real_absence(self):
        greeted = {"Michael": T0}
        back = T0 + 600
        self.assertTrue(self.should(back, visit=back, greeted=greeted,
                                    last=T0))

    def test_quick_return_after_greeting_is_quiet(self):
        greeted = {"Michael": T0}
        back = T0 + AWAY + 5
        self.assertFalse(self.should(back, visit=back, greeted=greeted,
                                     last=T0))

    def test_skipped_visit_is_not_greeted_later(self):
        """Back within the cooldown: no greeting — and none two minutes
        later either, while they are still sitting there."""
        greeted = {"Michael": T0}
        back = T0 + AWAY + 5
        self.assertFalse(self.should(back, back, greeted, last=T0))
        self.assertFalse(self.should(back + 300, back, greeted, last=T0))

    def test_not_while_out_of_sight(self):
        self.assertFalse(self.should(T0, visit=T0, greeted={}, seen=False))


# ══════════════════════════════════════════════════════════════════════════════
# What the model is told
# ══════════════════════════════════════════════════════════════════════════════
class TestBackgroundContext(unittest.TestCase):
    def test_scene_notes_are_marked_do_not_mention(self):
        out = fetch.route_message("hello",
                                  scene="Michael has been here a while.")
        self.assertIn("Michael has been here a while.", out)
        self.assertNotIn("right now", out.lower())
        self.assertIn("do not bring it up", out.lower())

    def test_a_camera_question_answers_from_the_live_scene(self):
        out = fetch.answer_message("what do you see", None,
                                   camera="a person smiling")
        self.assertIn("What your camera sees right now:\na person smiling", out)


# ══════════════════════════════════════════════════════════════════════════════
# Terminal output
# ══════════════════════════════════════════════════════════════════════════════
class TestConsole(unittest.TestCase):
    def out(self, fn, *a, verbose=False, **kw):
        buf = io.StringIO()
        with mock.patch.object(console, "LOG_VERBOSE", verbose), \
                redirect_stdout(buf):
            fn(*a, **kw)
        return buf.getvalue()

    def test_detail_only_when_verbose(self):
        self.assertEqual(self.out(console.detail, "[llm] 95 tok"), "")
        self.assertEqual(self.out(console.detail, "[llm] 95 tok",
                                  verbose=True), "[llm] 95 tok\n")

    def test_heard(self):
        self.assertEqual(self.out(console.heard, "what are the fees"),
                         "[heard] what are the fees\n")
        self.assertIn("ignored", self.out(console.heard, "india",
                                          ignored="not addressed to Luna"))

    def test_source_lists_pages_and_facts(self):
        line = self.out(console.source,
                        web=["https://tkmce.ac.in/achievements.php",
                             "https://tkmce.ac.in/achievements.php",
                             "https://tkmce.ac.in/awards.php#2024"],
                        kb=["what are the college achievements"])
        self.assertTrue(line.startswith("[source] "))
        self.assertIn("tkmce.ac.in/achievements.php", line)
        self.assertIn("tkmce.ac.in/awards.php", line)
        self.assertEqual(line.count("achievements.php"), 1)    # de-duplicated
        self.assertNotIn("https://", line)
        self.assertIn("what are the college achievements", line)

    def test_source_general_knowledge(self):
        self.assertIn("general knowledge", self.out(console.source))


if __name__ == "__main__":
    unittest.main()
