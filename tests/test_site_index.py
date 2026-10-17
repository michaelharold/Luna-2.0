# tests/test_site_index.py
"""Unit tests for site_index.py — Indexing(): grouping the crawl into page
files, organizing them (a fake Llama), describing them, the catalog. No
network, no model.

    python -m unittest tests.test_site_index -v
"""

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import site_index  # noqa: E402

H = "https://c.edu/"
RULES = (
    ("students", "Student lists", ("#btech",)),
    ("cse_people", "CSE people", ("#staff", "#hod", "/faculty-cse")),
    ("cse_research", "CSE research", ("#publications", "/rs-cse")),
    ("news", "College news", ("c.edu/$", "/news.php")),
    ("admissions", "Admissions", ("admission", "let%20web")),
)


def _cfg(tmp, **over):
    base = dict(site_index._DEFAULTS, SITE_DIR=tmp, SITE_PAGES=RULES,
                SITE_SKIP_PAGES=("tenders",), SITE_SKIP_TITLES=("minutes",),
                SITE_OTHER_DEPT_DATA=("staff", "hod"), SITE_MIN_PAGE_WORDS=3,
                SITE_LLM_CHUNK_WORDS=60, SITE_PDF_MAX_WORDS=50,
                SITE_GROUP_MAX_WORDS=400, SITE_OTHER_DEPT_MAX_WORDS=400,
                SITE_ORGANIZE_PROMPT="ORGANIZE {title}\n{text}",
                SITE_DESCRIBE_PROMPT="DESCRIBE {title} | {headings}\n{text}",
                KNOWLEDGE_PATH=os.path.join(tmp, "knowledge.txt"))
    base.update(over)
    return lambda name: base[name]


def _crawl():
    cse = H + "computer-science-and-engineering"
    etlab = "https://e.lab/dept?id=2"
    docs = [
        {"url": H, "page": H, "title": "Home",
         "text": "Welcome to C College\nAdmissions open for B.Tech 2026 on 1 June"},
        {"url": H + "uploads/let%20web.pdf", "page": H,
         "title": "Home — let web", "text": "Lateral entry 2026 apply by 10 July"},
        {"url": H + "uploads/16-1186964-OBC-2021.pdf", "page": H,
         "title": "Home — 16-1186964-OBC-2021", "text": "G.O. text " * 10},
        {"url": H + "uploads/GB minutes.pdf", "page": H,
         "title": "Home — GB minutes", "text": "Minutes of the meeting " * 5},
        {"url": cse, "page": cse, "title": "CSE Department",
         "text": "The CSE department started in 1984 with 30 seats."},
        {"url": etlab + "#hod", "page": cse, "data": True,
         "title": "Computer Science (CSE) — Head of department HOD",
         "text": "Head of department HOD, 1 in total: Dr. MIRA VANCE, "
                 "Associate Professor."},
        {"url": etlab + "#regular_staff", "page": cse, "data": True,
         "title": "Computer Science (CSE) — Regular staff",
         "text": "Regular staff, 2 in total: Dr. MIRA VANCE, Associate "
                 "Professor; Dr. Oren Dale, Professor."},
        {"url": etlab + "#regular_teaching_staff", "page": cse, "data": True,
         "title": "Computer Science (CSE) — Regular teaching staff",
         "text": "Regular teaching staff, 2 in total: Dr. MIRA VANCE, "
                 "Associate Professor; Dr. Oren Dale, Professor."},
        {"url": etlab + "#publications", "page": cse, "data": True,
         "title": "Computer Science (CSE) — Publications",
         "text": "Publications, 2 in total: Dr. Oren Dale, Paper A, 2024; "
                 "Dr. Mira Vance, Paper B, 2025."},
        {"url": etlab + "#btech", "page": cse, "data": True,
         "title": "Computer Science (CSE) — B.Tech students",
         "text": "Number of B.Tech students: 2, in 1 batches (CSE 2K23, 2 students)."},
        {"url": etlab + "#btech/CSE-2K23/0", "page": cse, "topic": etlab + "#btech",
         "title": "Computer Science (CSE) — B.Tech student",
         "text": "Name: Pranav Prasanth; Batch: CSE 2K23; Admission No: 230011."},
        {"url": "https://e.lab/dept/1#regular_staff", "page": H, "data": True,
         "scope": "other", "title": "Mechanical Engineering — Regular staff",
         "text": "Regular staff, 1 in total: Dr. Gear Smith, Professor."},
        {"url": "https://e.lab/dept/1#publications", "page": H, "data": True,
         "scope": "other", "title": "Mechanical Engineering — Publications",
         "text": "Publications, 1 in total: Dr. Gear Smith, Gears, 2020."},
        {"url": H + "mechanical-engineering", "page": H + "mechanical-engineering",
         "title": "Mechanical", "text": "Mechanical engineering since 1958."},
        {"url": H + "tenders", "page": H + "tenders", "title": "Tenders",
         "text": "Tender for chairs 2025"},
        {"url": H + "hods#t0r0", "page": H + "hods", "title": "HODs",
         "text": "Name: Dr. A; Department: Civil."},
        {"url": H + "hods#t0r1", "page": H + "hods", "title": "HODs",
         "text": "Name: Dr. B; Department: Mechanical."},
        {"url": H + "hods", "page": H + "hods", "title": "HODs",
         "text": "Heads of departments\nName | Department\nDr. A | Civil"},
        {"url": "https://sister.edu/", "page": "https://sister.edu/",
         "title": "Sister", "text": "Sister institute founded 1990."},
    ]
    catalog = {H: {"scope": "general"}, cse: {"scope": "focus"},
               H + "mechanical-engineering": {"scope": "other"},
               H + "tenders": {"scope": "general"}, H + "hods": {"scope": "general"},
               "https://sister.edu/": {"scope": "affiliate"}}
    return {"url": H, "fetched_at": 1_790_000_000.0, "catalog": catalog,
            "docs": docs, "report": {H: "ok (12 words)"}, "loads": {}}


class _Case(unittest.TestCase):
    def setUp(self):
        out = mock.patch("sys.stdout", new_callable=io.StringIO)
        out.start()
        self.addCleanup(out.stop)
        self.tmp = tempfile.mkdtemp(prefix="luna-test-index-")
        with open(os.path.join(self.tmp, "knowledge.txt"), "w") as f:
            f.write("who is the principal: Dr. Sadiq A.\n")
        self.cfg = _cfg(self.tmp)
        p = mock.patch.object(site_index, "_cfg", lambda n: self.cfg(n))
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestGroup(_Case):
    def setUp(self):
        super().setUp()
        self.groups, self.report = site_index.group(_crawl())
        self.dropped = dict(self.report["dropped"])

    def urls(self, name):
        return [s["url"] for s in self.groups[name]["sources"]]

    def test_data_sections_go_by_their_own_name(self):
        self.assertIn("https://e.lab/dept?id=2#publications", self.urls("cse_research"))
        self.assertIn("https://e.lab/dept?id=2#hod", self.urls("cse_people"))
        # its own text matches no rule: a file of its own, by its URL
        self.assertEqual(self.urls("computer_science_and_engineering"),
                         [H + "computer-science-and-engineering"])

    def test_a_pdf_goes_by_its_own_url_and_the_home_page_by_its_end(self):
        self.assertIn(H + "uploads/let%20web.pdf", self.urls("admissions"))
        self.assertEqual(self.urls("news"), [H])

    def test_students_and_counts(self):
        st = self.groups["students"]["sources"]
        lines = sum((s.get("lines", []) for s in st), [])
        self.assertIn("Pranav Prasanth — B.Tech student, batch CSE 2K23, "
                      "admission no. 230011", lines)
        self.assertTrue(any("Number of B.Tech students: 2" in l for l in lines))

    def test_other_departments_and_sister_sites(self):
        self.assertIn("dept_mechanical", self.groups)
        mech = self.urls("dept_mechanical")
        self.assertIn("https://e.lab/dept/1#regular_staff", mech)
        self.assertIn(H + "mechanical-engineering", mech)
        self.assertIn("another department's data",
                      self.dropped["https://e.lab/dept/1#publications"])
        self.assertEqual(self.urls("tkm_institutions"), ["https://sister.edu/"])

    def test_skipped_and_junk(self):
        self.assertIn("SITE_SKIP_PAGES", self.dropped[H + "tenders"])
        self.assertIn("SITE_SKIP_TITLES", self.dropped[H + "uploads/GB minutes.pdf"])
        self.assertIn("number", self.dropped[H + "uploads/16-1186964-OBC-2021.pdf"])

    def test_table_rows_are_one_list_and_the_text_drops_the_table(self):
        hods = self.groups["hods"]["sources"]
        lists = [s for s in hods if s["kind"] == "list"]
        self.assertEqual(lists[0]["lines"], ["Name: Dr. A; Department: Civil",
                                             "Name: Dr. B; Department: Mechanical"])
        prose = [s for s in hods if s["kind"] == "prose"][0]
        self.assertNotIn("|", prose["text"])

    def test_pdf_title_check(self):
        ok = site_index._meaningful_pdf
        self.assertTrue(ok("TKM Academic Policy"))
        self.assertTrue(ok("let final 26-27"))
        self.assertTrue(ok("B TECH S3 S5 & S7"))
        self.assertFalse(ok("16-11869642-OBC-2021"))
        self.assertFalse(ok("01"))


class TestHelpers(unittest.TestCase):
    def test_data_lines(self):
        head, lines = site_index._data_lines(
            "Publications (papers), 2 in total: Dr. ORREN VASK, Paper A; B.")
        self.assertEqual(head, "Publications (papers), 2 in total")
        self.assertEqual(lines, ["Dr. Orren Vask, Paper A", "B"])

    def test_short_title_and_caps(self):
        self.assertEqual(site_index.short_title(
            "Computer Science (CSE) — Regular staff, 13 in total"),
            "Regular staff, 13 in total")
        self.assertEqual(site_index.short_title(
            "CSE Department > Vision Mission / Department Vision"),
            "Department Vision")
        self.assertEqual(site_index.short_title("Placement | TKM College"),
                         "Placement")
        self.assertEqual(site_index._tidy_caps(
            "Dr. ANEESH G NATH, COMPUTER SCIENCE AND ENGINEERING, IIT"),
            "Dr. Aneesh G Nath, Computer Science and Engineering, IIT")

    def test_fact_check(self):
        src = "Fee is 45,000 rupees. Founded 1958 by Musaliar in Kollam."
        ok, kept, inv = site_index.fact_check(
            src, "- Fee: 45000 rupees\n- Founded 1958 by Musaliar, Kollam", 0.8)
        self.assertTrue(ok, (kept, inv))
        ok, _, inv = site_index.fact_check(src, "- Fee: 50000\n- 1958", 0.8)
        self.assertFalse(ok)
        self.assertIn("50000", inv)

    def test_chunks_are_line_aligned(self):
        text = "\n".join(f"line {i} " + "w " * 8 for i in range(20))
        pieces = site_index._chunks(text, 30)
        self.assertTrue(all(len(p.split()) <= 30 for p in pieces))
        self.assertEqual("\n".join(pieces).split(), text.split())

    def test_faq_keeps_only_checked_pairs(self):
        class L:
            def ask(self, prompt, max_tokens):
                return ("Q: Who is the HOD?\nA: Dr. Mira Vance is the head.\n"
                        "2. Q: What is the fee?\nA: It is 99999 rupees a year.\n"
                        "Q: hi\nA: no")
        with mock.patch.object(site_index, "_cfg",
                               lambda n: 8 if n == "SITE_FAQ_PER_PAGE"
                               else site_index._DEFAULTS[n]):
            got = site_index.faq("cse_people", "CSE people",
                                 "- Dr. Mira Vance, HOD\n- Fee 45000", L())
        self.assertEqual(got, [["Who is the HOD?", "Dr. Mira Vance is the head.",
                                "cse_people"]])

    def test_digest_short_pages_as_they_are_long_ones_by_llama(self):
        class L:
            def ask(self, prompt, max_tokens):
                return "- Fee is 45000\n- Fee is 99999 (invented)\nNONE"
        texts = {"cse_people": "# CSE people\n## Staff\n- Dr. A, HOD\n",
                 "admissions": "# Admissions\n## B.Tech\n"
                               + "\n".join(f"- line {i} fee 45000" for i in range(40)),
                 "students": "# Students\n## B.Tech students\n"
                             "- Number of B.Tech students: 604, in 8 batches (x)\n"
                             "- Pranav (CSE, 1)\n"}
        entries = {n: {"title": t} for n, t in (("cse_people", "CSE people"),
                   ("admissions", "Admissions"), ("students", "Students"))}
        cfg = dict(site_index._DEFAULTS, SITE_PAGES=RULES,
                   SITE_DIGEST_WORDS={"": 30, "cse_": 50},
                   SITE_DIGEST_MAX_WORDS=100)
        with mock.patch.object(site_index, "_cfg", lambda n: cfg[n]):
            d = site_index.build_digest(texts, entries, L())
        self.assertIn("## CSE people\n- Dr. A, HOD", d)
        self.assertIn("- Fee is 45000", d)
        self.assertNotIn("99999", d)
        self.assertIn("- Number of B.Tech students: 604, in 8 batches", d)
        self.assertNotIn("Pranav", d)
        self.assertLess(d.index("Students"), d.index("CSE people"))  # SITE_PAGES order

    def test_name_index(self):
        idx = site_index.name_index({
            "cse_people": "- Dr. Mira Vance, Professor\n- Prof. Saheer Kal",
            "students": "- Pranav Prasanth — B.Tech student, batch CSE 2K23"})
        self.assertEqual(idx["vance"], ["cse_people"])
        self.assertEqual(idx["saheer"], ["cse_people"])
        self.assertEqual(idx["prasanth"], ["students"])
        self.assertNotIn("professor", idx)


class _FakeLlama:
    """Organizes by keeping every line (as "- " notes); describes with a
    fixed line. Counts its calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, prompt, max_tokens):
        self.calls += 1
        if prompt.startswith("DESCRIBE"):
            return ("Description: This file contains facts about "
                    + prompt.split("|")[0][9:].strip() + " with dates and notices")
        text = prompt.split("\n", 1)[1]
        if "Mechanical engineering since" in text:
            return "NONE"                    # 1 fact: really nothing
        if "Welcome to C College" in text:
            return "Here are the notes:\n- Admissions for B.Tech 2026 open on 1 June\nNONE"
        return "\n".join(f"- {l}" for l in text.split("\n") if l.strip())


class TestRun(_Case):
    def setUp(self):
        super().setUp()
        site_index._write_json(site_index.paths()["crawl"], _crawl())

    def test_run_writes_pages_catalog_and_reuses_the_cache(self):
        llm = _FakeLlama()
        cat = site_index.run(offline=True, client=llm)
        pages = cat["pages"]
        for name in ("cse_people", "cse_research", "students", "news",
                     "admissions", "kb", "tkm_institutions"):
            self.assertIn(name, pages)
        self.assertEqual(pages["kb"]["file"], self.cfg("KNOWLEDGE_PATH"))
        self.assertEqual(pages["news"]["description"],
                         "College news with dates and notices")
        self.assertTrue(pages["students"]["list"])
        self.assertEqual(cat["names"]["prasanth"], ["students"])
        self.assertTrue(cat["staff"])
        d = site_index.paths()["pages"]
        people = open(os.path.join(d, "cse_people.txt")).read()
        self.assertTrue(people.startswith("# CSE people\nSources: "))
        self.assertEqual(people.count("Dr. Mira Vance, Associate Professor"), 2,
                         people)                  # HOD + ONE staff list
        news = open(os.path.join(d, "news.txt")).read()
        self.assertIn("- Admissions for B.Tech 2026 open on 1 June", news)
        self.assertNotIn("NONE", news)
        self.assertNotIn("Here are the notes", news)
        # a second run asks Llama nothing new
        again = _FakeLlama()
        site_index.run(offline=True, client=again)
        self.assertEqual(again.calls, 0)

    def test_stale_page_files_are_removed(self):
        d = site_index.paths()["pages"]
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "gone.txt"), "w").close()
        site_index.run(offline=True, client=_FakeLlama())
        self.assertFalse(os.path.exists(os.path.join(d, "gone.txt")))

    def test_status_and_plan(self):
        self.assertIn("No index yet", site_index.status())
        site_index.run(offline=True, client=_FakeLlama())
        self.assertIn("cse_people", site_index.status())
        groups, _ = site_index.group(_crawl())
        self.assertIn("cse_research", site_index.plan(groups))

    def test_catalog_is_valid_json_for_fetch(self):
        site_index.run(offline=True, client=_FakeLlama())
        with open(site_index.paths()["catalog"]) as f:
            cat = json.load(f)
        import fetch
        lib = fetch.Library(self.tmp)
        self.assertEqual(sorted(lib.names()), sorted(cat["pages"]))
        self.assertTrue(lib.page("cse_people").sections)


if __name__ == "__main__":
    unittest.main()
