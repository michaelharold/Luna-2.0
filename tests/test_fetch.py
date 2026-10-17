# tests/test_fetch.py
"""Unit tests for fetch.py — Data Fetch(): the LIBRARY, the ROUTE / ANSWER
prompts, reading the route reply and choosing the page data. No model, no
network: a catalog and page files are written to a temp folder.

    python -m unittest tests.test_fetch -v
"""

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import fetch    # noqa: E402
import intents  # noqa: E402

PEOPLE = """# CSE people
Sources: https://h/faculty-cse.php
Updated: 2026-10-13

## Head of department HOD, 1 in total
- Dr. Mira Vance, Associate Professor

## Regular teaching staff, 3 in total
- Dr. Mira Vance, Associate Professor
- Dr. Oren Dale, Professor
- Prof. Saheer Kal, Assistant Professor
"""

STUDENTS = "# Students\n\n## Number of B.Tech students\n- 604 in 8 batches\n\n" \
           "## B.Tech student\n" + "\n".join(
               f"- Student{i} Name{i} (CSE 2K23, 2300{i:02d})" for i in range(300)) + \
           "\n- Pranav Prasanth (CSE 2K23A, 230099)\n"


def _write_site(d, kb_path):
    os.makedirs(os.path.join(d, "pages"))
    for name, text in (("cse_people", PEOPLE), ("students", STUDENTS)):
        with open(os.path.join(d, "pages", f"{name}.txt"), "w") as f:
            f.write(text)
    cat = {"version": 1, "built_at": 1.0, "pages": {
        "cse_people": {"title": "CSE people", "file": "cse_people.txt",
                       "description": "CSE faculty: HOD, professors",
                       "sources": ["https://h/faculty-cse.php"]},
        "students": {"title": "Students", "file": "students.txt",
                     "description": "Student lists", "list": True,
                     "sources": ["https://h/btech.php"]},
        "kb": {"title": "KB", "file": kb_path, "description": "Curated facts",
               "sources": [kb_path]}},
        "names": {"saheer": ["cse_people"], "vance": ["cse_people"],
                  "pranav": ["students"], "prasanth": ["students"]},
        "staff": [["CSE — Regular teaching staff", "Dr. Oren Dale, Professor"]]}
    with open(os.path.join(d, "catalog.json"), "w") as f:
        json.dump(cat, f)


def _deltas(text, size=3, closed=None):
    """A stream of text deltas like llm_ollama.stream(); `closed` (a list)
    records when the consumer closes it early."""
    try:
        for i in range(0, len(text), size):
            yield text[i:i + size]
    finally:
        if closed is not None:
            closed.append(True)


class _LibCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="luna-test-fetch-")
        self.kb = os.path.join(self.tmp, "knowledge.txt")
        with open(self.kb, "w") as f:
            f.write("who is the principal: Dr. Sadiq A.\n\n"
                    "when was the college founded: 1958\n")
        _write_site(self.tmp, self.kb)
        self.lib = fetch.Library(self.tmp)
        cfg = dict(fetch._DEFAULTS, KNOWLEDGE_PATH=self.kb, FETCH_MAX_PAGES=3,
                   FETCH_DATA_WORDS=120,
                   FETCH_SYSTEM_RULES="LIBRARY:\n{library}\nmax {max_pages}")
        p = mock.patch.object(fetch, "_cfg", lambda n: cfg[n])
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestLibrary(_LibCase):
    def test_pages_and_sections(self):
        self.assertEqual(self.lib.names(), ["cse_people", "students", "kb"])
        p = self.lib.page("cse_people")
        self.assertEqual(p.title, "CSE people")
        self.assertEqual([h for h, _ in p.sections],
                         ["Head of department HOD, 1 in total",
                          "Regular teaching staff, 3 in total"])
        self.assertNotIn("Sources", " ".join(sum((l for _, l in p.sections), [])))
        self.assertIsNone(self.lib.page("nope"))

    def test_kb_page_is_read_live(self):
        self.assertIn("- who is the principal: Dr. Sadiq A.",
                      self.lib.page("kb").sections[0][1])
        with open(self.kb, "a") as f:
            f.write("where is the canteen: near the library\n")
        self.assertIn("- where is the canteen: near the library",
                      self.lib.page("kb").sections[0][1])

    def test_name_hints(self):
        self.assertEqual(self.lib.name_hints("who is saheer"), ["cse_people"])
        self.assertEqual(self.lib.name_hints("is pranav prasanth a student"),
                         ["students"])
        self.assertEqual(self.lib.name_hints("what is the fee"), [])

    def test_staff_and_reload_when_the_catalog_changes(self):
        self.assertEqual(self.lib.staff(),
                         [("CSE — Regular teaching staff", "Dr. Oren Dale, Professor")])
        path = os.path.join(self.tmp, "catalog.json")
        with open(path) as f:
            cat = json.load(f)
        del cat["pages"]["students"]
        with open(path, "w") as f:
            json.dump(cat, f)
        os.utime(path, (time.time() + 5, time.time() + 5))
        self.assertEqual(self.lib.names(), ["cse_people", "kb"])

    def test_no_catalog_means_no_pages(self):
        empty = fetch.Library(os.path.join(self.tmp, "none"))
        self.assertEqual(empty.names(), [])
        self.assertFalse(empty.available())
        self.assertEqual(fetch.system_prompt("PERSONA", empty), "PERSONA")


class TestFasterPaths(_LibCase):
    def test_sub_headings_are_sections(self):
        _, secs = fetch.parse_page("# T\n## Admissions\n### B.Tech\n- June\n"
                                   "### M.Tech\n- July\n")
        self.assertEqual(secs, [("Admissions", []), ("B.Tech", ["- June"]),
                                ("M.Tech", ["- July"])])

    def test_title_match_only_when_one_page_is_named(self):
        self.assertEqual(self.lib.title_match("how many students study here"),
                         ["students"])
        self.assertEqual(self.lib.title_match("who is the hod"), [])

    def test_word_pages_for_an_overruled_reply(self):
        self.assertEqual(self.lib.word_pages("who are the professors")[0],
                         "cse_people")
        self.assertEqual(self.lib.word_pages("zzz"), [])

    def test_faq(self):
        path = os.path.join(self.tmp, "catalog.json")
        with open(path) as f:
            cat = json.load(f)
        cat["faq"] = [["Who is the HOD?", "Dr. Mira Vance.", "cse_people"]]
        with open(path, "w") as f:
            json.dump(cat, f)
        os.utime(path, (time.time() + 9, time.time() + 9))
        self.assertEqual(self.lib.faq(),
                         [("Who is the HOD?", "Dr. Mira Vance.", "cse_people")])


class TestDigestMode(_LibCase):
    def test_a_misheard_name_finds_its_line(self):
        got = self.lib.lookup_lines("tell me about saheer kall", limit=3)
        self.assertTrue(got and got[0].startswith("Prof. Saheer Kal"), got)
        got = self.lib.lookup_lines("is pranav prasanth a student", limit=2)
        self.assertTrue(got[0].startswith("Pranav Prasanth (CSE 2K23A"), got)
        self.assertEqual(self.lib.lookup_lines("hello there", limit=3), [])

    def test_fuzzy_name_hints(self):
        self.assertEqual(self.lib.name_hints("who is saheerr"), ["cse_people"])

    def test_digest_system_prompt_and_message(self):
        self.assertEqual(fetch.digest_system_prompt("P", self.lib), "P")
        with open(os.path.join(self.tmp, "digest.txt"), "w") as f:
            f.write("## CSE people\n- Dr. Mira Vance, HOD\n")
        self.assertEqual(fetch.digest_system_prompt("P", self.lib),
                         "P\n\nFACTS:\n## CSE people\n- Dr. Mira Vance, HOD")
        msg = fetch.ask_message("who is mira", ["Dr. Mira Vance (HOD)"],
                                person="Ann")
        self.assertIn('Query: "who is mira"', msg)
        self.assertIn("- Dr. Mira Vance (HOD)", msg)
        self.assertIn("The person asking is Ann", msg)
        self.assertNotIn("\n\n", msg)


class TestPrompts(_LibCase):
    def test_system_prompt_is_persona_library_rules_and_stable(self):
        a = fetch.system_prompt("PERSONA", self.lib)
        self.assertTrue(a.startswith("PERSONA\n\nLIBRARY:\n"))
        self.assertIn("cse_people: CSE faculty: HOD, professors", a)
        self.assertIn("kb: Curated facts", a)
        self.assertIn("max 3", a)
        self.assertEqual(a, fetch.system_prompt("PERSONA", self.lib))

    def test_route_message_carries_only_the_notes_given(self):
        self.assertEqual(fetch.route_message("hi"),
                         'TASK: ROUTE\nQuery: "hi"')
        msg = fetch.route_message("who is saheer", person="Ann", unsure=True,
                                  hints=["cse_people"], greeted=True)
        self.assertIn("The person asking is Ann", msg)
        self.assertIn("do not greet them again", msg)
        self.assertIn("Speech recognition was unsure", msg)
        self.assertIn("Names in the query appear in: cse_people.", msg)

    def test_answer_message(self):
        msg = fetch.answer_message("who is the hod", "=== CSE ===\n- Dr. A",
                                   exemplars=[("q", "a")], style="Be brief.")
        self.assertIn('Query: "who is the hod"', msg)
        self.assertIn("=== CSE ===\n- Dr. A", msg)
        self.assertIn("Style for a reply: Be brief.", msg)
        self.assertIn("- Q: q\n  A: a", msg)
        cam = fetch.answer_message("what do you see", None, camera="one person")
        self.assertIn("What your camera sees right now:\none person", cam)


class TestReadRoute(unittest.TestCase):
    NAMES = ["cse_people", "students", "kb", "admissions"]

    def test_pages_stops_the_stream(self):
        closed = []
        r = fetch.read_route(_deltas("PAGES: cse_people, students\nREPLY: x",
                                     closed=closed), self.NAMES)
        self.assertEqual((r.kind, r.pages), ("pages", ["cse_people", "students"]))
        self.assertEqual(closed, [True])

    def test_reply_streams_without_its_label(self):
        r = fetch.read_route(_deltas("REPLY: Hello there! I am Luna."),
                             self.NAMES)
        self.assertEqual(r.kind, "reply")
        self.assertEqual("".join(r.reply), "Hello there! I am Luna.")

    def test_again_ignore_and_zero(self):
        for text, kind in (("AGAIN", "again"), ("IGNORE", "ignore"),
                           ("0", "ignore"), ("  again.", "again")):
            self.assertEqual(fetch.read_route(_deltas(text), self.NAMES).kind,
                             kind, text)

    def test_an_answer_without_the_format_is_a_reply(self):
        r = fetch.read_route(_deltas("Sure! The sky is blue."), self.NAMES)
        self.assertEqual(r.kind, "reply")
        self.assertEqual("".join(r.reply), "Sure! The sky is blue.")

    def test_decorated_and_split_labels(self):
        r = fetch.read_route(_deltas("**PAGES**:\nadmissions\n", size=1),
                             self.NAMES)
        self.assertEqual((r.kind, r.pages), ("pages", ["admissions"]))
        r = fetch.read_route(iter(["PA", "GES: cse people, nope, kb.txt"]),
                             self.NAMES)
        self.assertEqual(r.pages, ["cse_people", "kb"])

    def test_empty_stream(self):
        r = fetch.read_route(iter(()), self.NAMES)
        self.assertEqual(r.kind, "reply")
        self.assertEqual(list(r.reply), [])

    def test_match_pages_limit_and_unknown(self):
        self.assertEqual(fetch.match_pages("admisions, kb, students, cse_people",
                                           self.NAMES, limit=3),
                         ["admissions", "kb", "students"])
        self.assertEqual(fetch.match_pages("library, fees", self.NAMES), [])


class TestPageData(_LibCase):
    def test_small_page_goes_whole(self):
        data, used = fetch.page_data(["cse_people"], "who is the hod", lib=self.lib)
        self.assertEqual(used, ["cse_people"])
        self.assertTrue(data.startswith("=== CSE people ==="))
        self.assertIn("Prof. Saheer Kal", data)
        self.assertNotIn("Sources:", data)

    def test_big_page_gives_the_lines_that_match(self):
        data, _ = fetch.page_data(["students"], "is pranav prasanth a student",
                                  lib=self.lib)
        self.assertIn("Pranav Prasanth (CSE 2K23A, 230099)", data)
        self.assertLess(len(data.split()), 160)

    def test_no_match_gives_the_top_of_the_page(self):
        data, _ = fetch.page_data(["students"], "how many students", lib=self.lib)
        self.assertIn("604 in 8 batches", data)

    def test_budget_is_shared(self):
        data, used = fetch.page_data(["students", "cse_people", "nope"],
                                     "pranav", lib=self.lib)
        self.assertEqual(used, ["students", "cse_people"])
        self.assertIn("Dr. Oren Dale", data)              # small page whole
        self.assertIn("Pranav Prasanth", data)
        self.assertLess(len(data.split()), 190)

    def test_sources(self):
        self.assertEqual(fetch.sources(["cse_people", "kb"], self.lib),
                         ["https://h/faculty-cse.php", self.kb])


class TestQueryWords(unittest.TestCase):
    def test_expand_abbreviations(self):
        self.assertEqual(intents.expand_abbreviations("who is the HOD of cse?"),
                         "who is the HOD head of department HOD of cse? "
                         "computer science and engineering")
        self.assertEqual(intents.expand_abbreviations("tell me about me"),
                         "tell me about me")     # 2-letter keys never expand

    def test_query_words(self):
        self.assertEqual(fetch.query_words("Who is the HOD of CSE?"),
                         {"hod", "head", "computer", "science", "engineering",
                          "cse"})


if __name__ == "__main__":
    unittest.main()
