# tests/test_people_intents.py
"""The people Luna can describe with certainty (people.py) and the plain-text
intents that trigger cards / context (intents.py):

    python -m unittest tests.test_people_intents -v
"""

import datetime
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import intents  # noqa: E402
import people  # noqa: E402

KB = """who is the principal: Dr. Sadiq A. is the principal of TKM College of Engineering.
who is the head of computer science department:dr Aneesh g nath is the head of computer science department
Who is Dr. Dimple A Shajahan?: Dr. Dimple A Shajahan is a Professor in the Department of Computer Science. She obtained her PhD from IIT Madras.
Who is Dr. Aneesh G Nath?: Dr. Aneesh G Nath is the head of  the Department of Computer Science. He obtained his PhD from IIT (BHU) Varanasi.
Who is B. Mohammed Siyad?: B. Mohammed Siyad is an Assistant Professor in the Department of Computer Science.
Who is Dr. Shameem Ansar?: Dr. Shameem Ansar is an Assistant Professor in the Department of Computer Science. He obtained his PhD from NIT Tiruchirappalli (NIT Trichy). His research focuses on the application of wireless networks.
Who is Dr. Reshma Sheik?: Dr. Reshma Sheik is an Assistant Professor in the Department of Computer Science.
what is the fee: The fee is set by the university.
"""

STAFF = [("Computer Science & Engineering — Regular teaching staff",
          "Regular teaching staff, 3 in total: Dr. DIMPLE A SHAJAHAN, "
          "Professor, Computer Science & Engineering; Dr. MANU J PILLAI, "
          "Associate Professor, COMPUTER SCIENCE AND ENGINEERING; Prof. NISA "
          "A K, Asst. Professor, Computer Science and Engg"),
         ("Computer Science & Engineering — Adhoc teaching staff",
          "Adhoc teaching staff, 2 in total: Mrs. FATHIMA ABIDA, Asst. "
          "Professor, Information Technology; Prof. MOHAMMED RAFI, Asst. "
          "Professor")]


class TestNames(unittest.TestCase):
    def test_display_name(self):
        d = people.display_name
        self.assertEqual(d("Dr. MANU J PILLAI"), "Dr. Manu J Pillai")
        self.assertEqual(d("dr Aneesh g nath"), "Dr. Aneesh G Nath")
        self.assertEqual(d("B. Mohammed Siyad"), "B. Mohammed Siyad")
        self.assertEqual(d("Prof. NISA A K"), "Prof. Nisa A K")

    def test_name_tokens_skip_titles_and_initials(self):
        self.assertEqual(people.name_tokens("Dr. Anantha Padmanabhan N K"),
                         {"anantha", "padmanabhan"})
        self.assertEqual(people.name_tokens("B. Mohammed Siyad"),
                         {"mohammed", "siyad"})


class TestDirectory(unittest.TestCase):
    def setUp(self):
        self.d = people.Directory.build(KB.splitlines(), STAFF)

    def test_kb_person_role_and_detail(self):
        p = self.d.lookup("Dimple Shajahan")
        self.assertEqual(p.name, "Dr. Dimple A Shajahan")
        self.assertEqual(p.role, "Professor · Computer Science")
        self.assertEqual(p.detail, "PhD from IIT Madras")
        self.assertEqual(p.source, "kb")

    def test_research_detail(self):
        p = self.d.lookup("Shameem Ansar")
        self.assertEqual(p.detail, "PhD from NIT Tiruchirappalli (NIT Trichy)")

    def test_head_role(self):
        self.assertEqual(self.d.lookup("Aneesh Nath").role,
                         "Head of Computer Science")

    def test_web_staff_fill_the_gaps(self):
        p = self.d.lookup("Manu Pillai")
        self.assertEqual(p.name, "Dr. Manu J Pillai")
        self.assertEqual(p.role, "Associate Professor · Computer Science and Engineering")
        self.assertEqual(p.source, "web")
        self.assertEqual(self.d.lookup("Nisa").role,
                         "Assistant Professor · Computer Science and Engineering")

    def test_kb_wins_over_web_for_the_same_person(self):
        self.assertEqual(self.d.lookup("Dimple").source, "kb")

    def test_role_question_names_the_person(self):
        p = self.d.for_kb_question("who is the principal")
        self.assertEqual(p.name, "Dr. Sadiq A.")
        self.assertEqual(p.role, "Principal of TKM College of Engineering")
        p = self.d.for_kb_question("who is the head of computer science department")
        self.assertEqual(p.name, "Dr. Aneesh G Nath")         # canonical form
        self.assertEqual(p.detail, "PhD from IIT (BHU) Varanasi")
        self.assertIsNone(self.d.for_kb_question("what is the fee"))

    def test_question_must_name_exactly_one_person(self):
        m = self.d.match_question
        self.assertEqual(m("who is reshma sheik").name, "Dr. Reshma Sheik")
        self.assertEqual(m("tell me about professor shameem ansar").name,
                         "Dr. Shameem Ansar")
        self.assertIsNone(m("who is mohammed"))     # Siyad and Rafi: ambiguous
        self.assertIsNone(m("who is reshma"))       # half a name is not certain
        self.assertEqual(m("who is mohammed siyad").name, "B. Mohammed Siyad")
        self.assertIsNone(m("who is the best teacher"))           # no name
        self.assertIsNone(m("is reshma in the lab today"))        # not "about"
        self.assertIsNone(m("who is reshma pillai"))   # names two people

    def test_one_word_names_need_only_that_word(self):
        d = people.Directory.build(
            ["Who is Dr. Thushara A?: Dr. Thushara A is a Professor in the "
             "Department of Computer Science."])
        self.assertEqual(d.match_question("who is thushara").name,
                         "Dr. Thushara A")

    def test_kb_entry_also_needs_the_name(self):
        """A knowledge-base match on a person entry ('who is b. mohammed
        siyad?') still needs the QUESTION to name them fully — role entries
        ('who is the principal') need no name."""
        f = self.d.for_kb_question
        self.assertIsNone(f("Who is B. Mohammed Siyad?", "who is mohammed"))
        self.assertEqual(f("Who is B. Mohammed Siyad?",
                           "who is mohammed siyad").name, "B. Mohammed Siyad")
        self.assertEqual(f("who is the principal", "who is the principal").name,
                         "Dr. Sadiq A.")

    def test_cut_chunks_drop_their_partial_edges(self):
        d = people.Directory.build([], [
            ("CSE — Adhoc teaching staff",
             "Adhoc teaching staff, 3 in total: Mrs. FATHIMA ABIDA, Asst. "
             "Professor; Prof. ANUMOD B, Asst. Professor; AMINA M"),
            ("CSE — Adhoc teaching staff",
             "ofessor, Information Technology; Mrs. VRINDHA S, Asst. "
             "Professor, Computer Science and Engineering")])
        names = sorted(p.name for p in d.people)
        self.assertEqual(names, ["Mrs. Fathima Abida", "Mrs. Vrindha S",
                                 "Prof. Anumod B"])

    def test_unknown_names(self):
        self.assertIsNone(self.d.lookup("Michael"))
        self.assertIsNone(self.d.lookup(""))


class TestIntents(unittest.TestCase):
    def test_date_and_time(self):
        dt = intents.date_time
        for q in ("what is the date today", "what's today's date",
                  "what day is it", "luna what is today's date please",
                  "what is today's day", "what is the day today",
                  "what day is today", "todays date", "what date is it",
                  "tell me the date", "what day it is", "what is the date"):
            self.assertEqual(dt(q), "date", q)
        for q in ("what time is it", "what's the time", "tell me the time",
                  "what is the time now"):
            self.assertEqual(dt(q), "time", q)
        for q in ("when do the semester exams start", "what is the fee",
                  "what is the timetable for third year",
                  "what are the dates of the exams",
                  "what is the last date for fee payment",
                  "what is the date of the series exam",
                  "what day is the exam", "when is the exam date"):
            self.assertIsNone(dt(q), q)

    def test_spoken_date_and_time(self):
        now = datetime.datetime(2026, 10, 12, 14, 7)
        self.assertEqual(intents.say_date(now),
                         "Today is Monday, the 12th of October.")
        self.assertEqual(intents.say_time(now), "It's 2:07 PM.")
        self.assertEqual(intents.say_date(datetime.datetime(2026, 10, 1)),
                         "Today is Thursday, the 1st of October.")
        self.assertEqual(intents.say_time(datetime.datetime(2026, 1, 1, 0, 5)),
                         "It's 12:05 AM.")

    def test_clock_is_set(self):
        self.assertTrue(intents.clock_is_set(datetime.datetime(2026, 10, 12)))
        self.assertFalse(intents.clock_is_set(datetime.datetime(1970, 1, 1)))

    def test_asks_about_person(self):
        a = intents.asks_about_person
        for q in ("who is reshma sheik", "tell me about dr aneesh",
                  "what are the achievements of shameem ansar",
                  "do you know professor siyad"):
            self.assertTrue(a(q), q)
        self.assertFalse(a("is reshma in the lab today"))

    def test_needs_scene_context(self):
        n = intents.needs_scene_context
        for q in ("who was here earlier", "did anyone come by",
                  "do you remember me", "what's my name", "who am i",
                  "have you seen alice today"):
            self.assertTrue(n(q), q)
        for q in ("what are the latest achievements of the college",
                  "who is the head of the computer science department",
                  "what is the fee for btech", "how are you doing today"):
            self.assertFalse(n(q), q)


if __name__ == "__main__":
    unittest.main()
