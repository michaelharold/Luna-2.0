# tests/test_enroll.py
"""Face registration options + vision performance helpers:

  - commands.py guided registration (poses in order, finish flag, success /
    failure announcements) with speak and vision mocked
  - "Did you say Alice?" confirmation: yes / no / a new name
  - start_enrollment_dialog()
  - vision_module's per-pose sample collection, scale_rows, person_since
  - scripts/enroll_faces.py with fake detector / embedder into a temp store

    python -m unittest tests.test_enroll -v
"""

import os
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# Point the face store at a throwaway dir BEFORE anything imports config.
_tmpdir = tempfile.mkdtemp(prefix="luna-test-enroll-")
import config  # noqa: E402
if "face_store" not in sys.modules:
    config.FACE_DB_DIR = _tmpdir

import commands  # noqa: E402
from shared_state import state  # noqa: E402


def _reset_state():
    with state.lock:
        state.enroll_request = None
        state.enroll_result = None
        state.enroll_finish = False
        state.enroll_pose = 0
        state.face_detected = True
        state.person_name = None
        state.person_since = 0.0
        state.awaiting_answer = False
        state.conversation_active = False
    commands._awaiting_name = False
    commands._awaiting_spelling = False
    commands._pending_name = None


class _CommandsTestBase(unittest.TestCase):
    """Speak is recorded (with the pose vision would see at that moment),
    face is a Mock, vision is a fake module with enroll_sample_count."""

    def setUp(self):
        _reset_state()
        self.spoken = []          # (text, enroll_pose, enroll_request)
        self.counts = {}          # pose -> sample count the fake reports

        def fake_speak(text):
            with state.lock:
                self.spoken.append((text, state.enroll_pose,
                                    state.enroll_request))

        fake_vision = types.ModuleType("vision_module")
        fake_vision.enroll_sample_count = \
            lambda pose=None: self.counts.get(pose, config.ENROLL_FRAMES_PER_POSE)

        self.face = mock.Mock(spec=["curious", "happy", "sad", "banner",
                                    "recognition_moment"])
        patches = [
            mock.patch.object(commands, "_speak", fake_speak),
            mock.patch.object(commands, "face", self.face),
            mock.patch.dict(sys.modules, {"vision_module": fake_vision}),
            mock.patch.object(commands, "ENROLL_POSE_SETTLE", 0.0),
            mock.patch.object(commands, "ENROLL_POSE_WAIT", 0.3),
            mock.patch.object(commands, "ENABLE_FACE_RECOGNITION", True),
            mock.patch.object(commands, "ENABLE_VOICE_COMMANDS", True),
            # the college's names: none, unless a test sets self.names
            mock.patch.object(commands, "_known_names", lambda: self.names),
        ]
        self.names = []
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(_reset_state)

    def texts(self):
        return [t for t, _, _ in self.spoken]

    def _responder(self, ok, seen_finish):
        """Plays vision: answers enroll_finish with (name, ok)."""
        def run():
            deadline = time.time() + 5
            while time.time() < deadline:
                with state.lock:
                    if state.enroll_finish and state.enroll_request:
                        seen_finish.append(state.enroll_request)
                        state.enroll_result = (state.enroll_request, ok)
                        return
                time.sleep(0.01)
        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t


class TestGuidedEnrollment(_CommandsTestBase):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(commands, "ENROLL_GUIDED_POSES", True)
        p.start()
        self.addCleanup(p.stop)

    def test_poses_spoken_in_order_and_success(self):
        seen_finish = []
        t = self._responder(True, seen_finish)
        commands._do_enroll("Alice")
        t.join(1)

        poses = list(config.ENROLL_POSES)
        # the greeting comes WITH the first pose prompt, not before it
        self.assertEqual(self.spoken[0][:2], (f"Okay Alice! {poses[0]}", 0))
        pose_lines = [(txt, pose) for txt, pose, _ in self.spoken
                      if txt in poses]
        self.assertEqual([txt for txt, _ in pose_lines], poses[1:])
        # vision saw pose i while prompt i was spoken
        self.assertEqual([pose for _, pose in pose_lines],
                         list(range(1, len(poses))))
        # the request was live during the intro and every prompt (the
        # closing "Nice to meet you" comes after cleanup)
        self.assertTrue(all(req == "Alice" for _, _, req in self.spoken[:-1]))
        self.assertIsNone(self.spoken[-1][2])
        self.assertEqual(seen_finish, ["Alice"])
        self.face.recognition_moment.assert_called_once_with("Alice")
        self.assertIn("Nice to meet you, Alice! I'll remember your face.",
                      self.texts())
        with state.lock:          # everything cleaned up
            self.assertIsNone(state.enroll_request)
            self.assertIsNone(state.enroll_result)
            self.assertFalse(state.enroll_finish)
            self.assertEqual(state.enroll_pose, 0)

    def test_straight_look_from_the_name_dialog_skips_that_pose(self):
        sys.modules["vision_module"].take_precapture = \
            lambda: config.ENROLL_FRAMES_PER_POSE
        seen_finish = []
        t = self._responder(True, seen_finish)
        commands._do_enroll("Alice")
        t.join(1)
        poses = list(config.ENROLL_POSES)
        self.assertEqual(self.spoken[0][:2], (f"Okay Alice! {poses[1]}", 1))
        self.assertNotIn(poses[0], " ".join(self.texts()))
        self.assertEqual(seen_finish, ["Alice"])

    def test_failure_apologises(self):
        seen_finish = []
        t = self._responder(False, seen_finish)
        commands._do_enroll("Bob")
        t.join(1)
        self.assertEqual(seen_finish, ["Bob"])
        self.face.sad.assert_called()
        self.face.recognition_moment.assert_not_called()
        self.assertTrue(any("couldn't get a clear look" in s
                            for s in self.texts()))

    def test_timeout_is_failure(self):
        with mock.patch.object(commands, "ENROLL_TIMEOUT", 0.3):
            commands._do_enroll("Carol")      # nobody answers
        self.assertTrue(any("couldn't get a clear look" in s
                            for s in self.texts()))
        with state.lock:
            self.assertIsNone(state.enroll_request)
            self.assertFalse(state.enroll_finish)

    def test_waits_for_pose_samples(self):
        # Pose 1 never fills up: Luna waits ENROLL_POSE_WAIT, then moves on.
        self.counts = {0: 99, 1: 0, 2: 99}
        seen_finish = []
        t = self._responder(True, seen_finish)
        t0 = time.time()
        commands._do_enroll("Dan")
        t.join(1)
        self.assertGreaterEqual(time.time() - t0, 0.25)
        self.assertEqual(seen_finish, ["Dan"])

    def test_fallback_without_recognition_moment(self):
        self.face = mock.Mock(spec=["curious", "happy", "sad", "banner"])
        with mock.patch.object(commands, "face", self.face):
            t = self._responder(True, [])
            commands._do_enroll("Eve")
            t.join(1)
        self.face.happy.assert_called()
        self.face.banner.assert_called_with("Hi Eve!")


class TestBurstEnrollment(_CommandsTestBase):
    def test_legacy_burst_never_sets_poses(self):
        with mock.patch.object(commands, "ENROLL_GUIDED_POSES", False):
            def vision():
                deadline = time.time() + 5
                while time.time() < deadline:
                    with state.lock:
                        if state.enroll_request:
                            state.enroll_result = (state.enroll_request, True)
                            return
                    time.sleep(0.01)
            t = threading.Thread(target=vision, daemon=True)
            t.start()
            commands._do_enroll("Fay")
            t.join(1)
        self.assertFalse(any(s in config.ENROLL_POSES for s in self.texts()))
        self.assertIn("Nice to meet you, Fay! I'll remember your face.",
                      self.texts())


class TestNameConfirmation(_CommandsTestBase):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(commands, "ENROLL_CONFIRM_NAME", True)
        p.start()
        self.addCleanup(p.stop)
        self.enrolled = []
        p2 = mock.patch.object(commands, "_do_enroll", self.enrolled.append)
        p2.start()
        self.addCleanup(p2.stop)

    def test_asks_then_yes_enrolls(self):
        self.assertTrue(commands.handle("remember me my name is alice"))
        self.assertEqual(self.texts()[-1], "Did you say Alice?")
        self.assertEqual(self.enrolled, [])
        with state.lock:
            self.assertTrue(state.awaiting_answer)
            self.assertTrue(state.conversation_active)
        self.assertTrue(commands.handle("yes"))
        self.assertEqual(self.enrolled, ["Alice"])

    def test_yes_variants(self):
        for reply in ("yeah", "yep", "that's correct", "right"):
            self.enrolled.clear()
            commands.handle("my name is alice")
            self.assertTrue(commands.handle(reply), reply)
            self.assertEqual(self.enrolled, ["Alice"], reply)

    def test_no_asks_to_spell_it(self):
        commands.handle("my name is alex")
        self.assertTrue(commands.handle("no"))
        self.assertEqual(self.texts()[-1], commands._SPELL_PROMPT)
        self.assertTrue(commands._awaiting_spelling)
        self.assertEqual(self.enrolled, [])
        # the spelled name goes through confirmation again
        self.assertTrue(commands.handle("n a v i a"))
        self.assertEqual(self.texts()[-1], "Did you say Navia?")
        self.assertTrue(commands.handle("yes"))
        self.assertEqual(self.enrolled, ["Navia"])

    def test_spelling_in_letter_words_or_the_name_itself(self):
        self.assertEqual(commands._spelled("en ay vee eye ay"), "Navia")
        self.assertEqual(commands._spelled("N-A-V-I-A".lower()), "Navia")
        self.assertIsNone(commands._spelled("navia"))
        commands.handle("my name is alex")
        commands.handle("no")
        commands.handle("alice")                 # said, not spelled
        self.assertEqual(self.texts()[-1], "Did you say Alice?")

    def test_nope_and_wrong(self):
        for reply in ("nope", "wrong", "that's not right"):
            commands.handle("my name is alex")
            commands.handle(reply)
            self.assertEqual(self.texts()[-1], commands._SPELL_PROMPT, reply)
            commands._awaiting_spelling = False
        self.assertEqual(self.enrolled, [])

    def test_a_close_college_name_is_offered_and_kept_on_yes(self):
        self.names = ["Navya Thomas", "Dr. Reshma Sheik"]
        commands.start_enrollment_dialog()
        commands.handle("navia")
        self.assertEqual(self.texts()[-1], "Did you say Navya?")
        with mock.patch.object(commands, "_do_enroll", self.enrolled.append):
            commands.handle("yes")
        self.assertEqual(self.enrolled, ["Navya"])

    def test_no_to_the_college_spelling_takes_what_was_heard(self):
        self.names = ["Navya Thomas"]
        commands.start_enrollment_dialog()
        commands.handle("navia")
        with mock.patch.object(commands, "_do_enroll", self.enrolled.append):
            commands.handle("no")
        self.assertEqual(self.enrolled, ["Navia"])

    def test_an_exact_college_name_is_not_asked_about(self):
        self.names = ["Reshma Sheik"]
        commands.start_enrollment_dialog()
        with mock.patch.object(commands, "_do_enroll", self.enrolled.append):
            commands.handle("reshma sheik")
        self.assertEqual(self.enrolled, ["Reshma Sheik"])
        self.assertNotIn("Did you say", self.texts()[-1])

    def test_whole_short_name_is_kept(self):
        commands.start_enrollment_dialog()
        commands.handle("navia thomas")
        self.assertEqual(self.texts()[-1], "Did you say Navia Thomas?")
        commands._pending_name = None
        self.assertEqual(commands._extract_name("my name is navia and i study cse"),
                         "Navia")
        self.assertEqual(commands._extract_name("my name is navia thomas"),
                         "Navia Thomas")

    def test_new_name_pattern_reasks(self):
        commands.handle("my name is alex")
        self.assertTrue(commands.handle("no my name is alice"))
        self.assertEqual(self.texts()[-1], "Did you say Alice?")
        commands.handle("yes")
        self.assertEqual(self.enrolled, ["Alice"])

    def test_unrelated_long_utterance_falls_through(self):
        commands.handle("my name is alex")
        self.assertFalse(commands.handle(
            "what courses does the department offer this year"))
        self.assertIsNone(commands._pending_name)
        self.assertEqual(self.enrolled, [])

    def test_confirmation_off_enrolls_directly(self):
        with mock.patch.object(commands, "ENROLL_CONFIRM_NAME", False):
            commands.handle("remember me my name is alice")
        self.assertEqual(self.enrolled, ["Alice"])

    def test_bare_remember_me_then_name_then_yes(self):
        commands.handle("remember me")
        self.assertEqual(self.texts()[-1], "Of course! What's your name?")
        commands.handle("bob")
        self.assertEqual(self.texts()[-1], "Did you say Bob?")
        commands.handle("yes")
        self.assertEqual(self.enrolled, ["Bob"])

    def test_start_enrollment_dialog(self):
        self.assertTrue(commands.start_enrollment_dialog())
        self.assertEqual(self.texts()[-1],
                         "Want me to remember you? Tell me your name.")
        self.assertTrue(commands._awaiting_name)
        with state.lock:
            self.assertTrue(state.awaiting_answer)
            self.assertTrue(state.conversation_active)
        commands.handle("my name is carol")
        commands.handle("yes")
        self.assertEqual(self.enrolled, ["Carol"])

    def test_start_dialog_from_another_thread(self):
        t = threading.Thread(target=commands.start_enrollment_dialog)
        t.start()
        t.join(2)
        self.assertTrue(commands._awaiting_name)

    def test_start_dialog_disabled(self):
        with mock.patch.object(commands, "ENABLE_FACE_RECOGNITION", False):
            self.assertFalse(commands.start_enrollment_dialog())
        self.assertFalse(commands._awaiting_name)

    def test_other_commands_still_work(self):
        with mock.patch.object(commands, "_cycle_style") as cyc:
            self.assertTrue(commands.handle("change your face"))
            cyc.assert_called_once()
        self.assertFalse(commands.handle("tell me a joke"))


# ── vision_module (imports torch + models — skipped if unavailable) ──────────

def _vision():
    try:
        import vision_module
        return vision_module
    except Exception as e:              # noqa: BLE001
        raise unittest.SkipTest(f"vision_module unavailable: {e}")


class TestScaleRows(unittest.TestCase):
    def test_scales_box_and_landmarks_not_score(self):
        vm = _vision()
        rows = np.arange(30, dtype=np.float32).reshape(2, 15)
        out = vm.scale_rows(rows, 2.0)
        np.testing.assert_allclose(out[:, :14], rows[:, :14] * 2.0)
        np.testing.assert_allclose(out[:, 14], rows[:, 14])
        np.testing.assert_allclose(rows[0, 0:3], [0, 1, 2])   # input untouched

    def test_none_and_empty(self):
        vm = _vision()
        self.assertIsNone(vm.scale_rows(None, 2.0))
        self.assertEqual(vm.scale_rows(np.zeros((0, 15)), 2.0).shape, (0, 15))

    def test_single_row(self):
        vm = _vision()
        out = vm.scale_rows(np.ones(15, dtype=np.float32), 0.5)
        self.assertAlmostEqual(float(out[13]), 0.5)
        self.assertAlmostEqual(float(out[14]), 1.0)

    def test_detect_scale_bounds(self):
        vm = _vision()
        self.assertEqual(vm._detect_scale(1.0), 1.0)
        self.assertEqual(vm._detect_scale(0.5), 0.5)
        self.assertEqual(vm._detect_scale(0.0), 1.0)
        self.assertEqual(vm._detect_scale("bad"), 1.0)


class TestVisionEnrollment(unittest.TestCase):
    def setUp(self):
        self.vm = _vision()
        _reset_state()
        self.vm._reset_enrollment()
        self.vm._enroll_done = False
        self.stored = []
        rng = np.random.default_rng(1)
        p = [
            mock.patch.object(self.vm, "_embed",
                              lambda f, r: rng.normal(size=128)),
            mock.patch.object(self.vm.face_store, "enroll",
                              lambda n, s: self.stored.append((n, len(s)))
                              or True),
        ]
        for x in p:
            x.start()
            self.addCleanup(x.stop)
        self.addCleanup(_reset_state)

    def _tick(self, n, row=object()):
        for _ in range(n):
            self.vm._update_enrollment(None, row)

    def test_guided_per_pose_and_finish(self):
        with mock.patch.object(self.vm, "ENROLL_GUIDED_POSES", True):
            with state.lock:
                state.enroll_request = "Alice"
            for pose in range(3):
                with state.lock:
                    state.enroll_pose = pose
                self._tick(3)
                self.assertEqual(self.vm.enroll_sample_count(pose), 3)
            self.assertEqual(self.vm.enroll_sample_count(), 9)
            with state.lock:
                self.assertIsNone(state.enroll_result)   # not before finish
                state.enroll_finish = True
            self._tick(1)
            with state.lock:
                self.assertEqual(state.enroll_result, ("Alice", True))
                self.assertEqual(state.person_name, "Alice")
                self.assertGreater(state.person_since, 0)
            self.assertEqual(self.stored, [("Alice", 9)])
            self.assertEqual(self.vm.enroll_sample_count(), 0)
            self._tick(3)                 # reported: ignore until cleared
            self.assertEqual(self.stored, [("Alice", 9)])
            with state.lock:
                state.enroll_request = None
            self._tick(1)
            self.assertFalse(self.vm._enroll_done)

    def test_guided_keeps_latest_per_pose(self):
        with mock.patch.object(self.vm, "ENROLL_GUIDED_POSES", True):
            with state.lock:
                state.enroll_request = "Alice"
            self._tick(self.vm.ENROLL_POSE_KEEP + 5)
            self.assertEqual(self.vm.enroll_sample_count(0),
                             self.vm.ENROLL_POSE_KEEP)

    def test_too_few_samples_fails(self):
        with mock.patch.object(self.vm, "ENROLL_GUIDED_POSES", True), \
                mock.patch.object(self.vm, "ENROLL_MIN_FRAMES", 4):
            with state.lock:
                state.enroll_request = "Bob"
            self._tick(2)
            self._tick(5, row=None)       # no face: no samples
            with state.lock:
                state.enroll_finish = True
            self._tick(1, row=None)
            with state.lock:
                self.assertEqual(state.enroll_result, ("Bob", False))
            self.assertEqual(self.stored, [])

    def test_legacy_burst_stops_at_enroll_frames(self):
        with mock.patch.object(self.vm, "ENROLL_GUIDED_POSES", False), \
                mock.patch.object(self.vm, "ENROLL_FRAMES", 5):
            with state.lock:
                state.enroll_request = "Cy"
            self._tick(4)
            with state.lock:
                self.assertIsNone(state.enroll_result)
            self._tick(1)
            with state.lock:
                self.assertEqual(state.enroll_result, ("Cy", True))
            self.assertEqual(self.stored, [("Cy", 5)])

    def test_person_since_on_recognition(self):
        vm = self.vm
        with mock.patch.object(vm, "_embed", lambda f, r: np.ones(128)), \
                mock.patch.object(vm.face_store, "match",
                                  lambda e: ("Dee", 0.9)), \
                mock.patch.object(vm, "_log_sighting", lambda n, c: None):
            vm._recog_last, vm._recog_streak = None, 0
            vm._update_recognition(None, object())
            vm._update_recognition(None, object())
            with state.lock:
                since = state.person_since
                self.assertEqual(state.person_name, "Dee")
            self.assertGreater(since, 0)
            vm._update_recognition(None, object())   # same person: unchanged
            with state.lock:
                self.assertEqual(state.person_since, since)
        vm._clear_recognition()
        with state.lock:
            self.assertEqual(state.person_since, 0.0)

    def test_scene_memory_feed_guarded(self):
        vm = self.vm
        calls = []
        fake = types.SimpleNamespace(
            observe=lambda **kw: calls.append(kw))
        with mock.patch.object(vm, "_scene_memory", fake), \
                mock.patch.object(vm, "_scene_memory_failed", False):
            vm._feed_scene_memory(2, "Happy")
            self.assertEqual(calls, [{"face_count": 2, "person_name": None,
                                      "emotion": "Happy"}])

        def boom(**kw):
            raise RuntimeError("nope")
        with mock.patch.object(vm, "_scene_memory",
                               types.SimpleNamespace(observe=boom)), \
                mock.patch.object(vm, "_scene_memory_failed", False):
            vm._feed_scene_memory(1, "Neutral")      # must not raise
            self.assertTrue(vm._scene_memory_failed)


# ── scripts/enroll_faces.py ──────────────────────────────────────────────────

class TestEnrollFacesScript(unittest.TestCase):
    def setUp(self):
        import enroll_faces
        import face_store
        self.ef = enroll_faces
        self.fs = face_store
        self.photos = tempfile.mkdtemp(prefix="luna-test-photos-")
        self.addCleanup(shutil.rmtree, self.photos, True)
        for person, files in (("alice", ["1.jpg", "2.jpg"]),
                              ("bob_smith", ["a.png", "noface.png"]),
                              ("empty", [])):
            os.makedirs(os.path.join(self.photos, person))
            for f in files:
                open(os.path.join(self.photos, person, f), "w").close()
        open(os.path.join(self.photos, "alice", "notes.txt"), "w").close()

        # fakes: "images" are paths; a filename with "noface" has no face;
        # embeddings are deterministic per person folder.
        self.loader = lambda p: p
        self.detect = lambda img: None if "noface" in img else ("row", img)
        self.embed = lambda img, row: np.full(
            128, 1.0 if "alice" in img else -1.0) + np.arange(128) * 0.01

    def tearDown(self):
        for n in ("Alice", "Bob Smith", "Zed"):
            self.fs.forget(n)

    def test_jobs_from_dir(self):
        jobs = dict(self.ef.jobs_from_dir(self.photos))
        self.assertEqual(sorted(jobs), ["Alice", "Bob Smith"])
        self.assertEqual(len(jobs["Alice"]), 2)          # .txt ignored

    def test_run_enrolls_and_warns(self):
        jobs = self.ef.jobs_from_dir(self.photos)
        results = self.ef.run(jobs, self.detect, self.embed, self.fs,
                              loader=self.loader)
        self.assertEqual(results, {"Alice": True, "Bob Smith": True})
        self.assertIn("Alice", self.fs.known_names())
        self.assertTrue(os.path.exists(self.fs._NAMES_PATH))
        name, conf = self.fs.match(self.embed("alice/x.jpg", None))
        self.assertEqual(name, "Alice")
        self.assertGreater(conf, 0.99)

    def test_no_usable_photo_not_enrolled(self):
        ok, used, skipped = self.ef.enroll_person(
            "Zed", ["noface1.jpg", "noface2.jpg"], self.detect, self.embed,
            self.fs, loader=self.loader)
        self.assertFalse(ok)
        self.assertEqual(used, 0)
        self.assertEqual(len(skipped), 2)
        self.assertNotIn("Zed", self.fs.known_names())

    def test_unreadable_image_skipped(self):
        ok, used, skipped = self.ef.enroll_person(
            "Zed", ["a.jpg", "b.jpg"], self.detect, self.embed, self.fs,
            loader=lambda p: None if p == "a.jpg" else p)
        self.assertTrue(ok)
        self.assertEqual((used, skipped), (1, ["a.jpg"]))

    def test_name_from_folder(self):
        self.assertEqual(self.ef.name_from_folder("/x/alice_smith/"),
                         "Alice Smith")


if __name__ == "__main__":
    unittest.main()
