# tests/test_bench_llm.py
"""Unit tests for scripts/bench_llm.py's pure grading logic and for the
evaluation set it reads — no Ollama, no sentence transformer, no network:

    python -m unittest discover tests -v

Covers: refusal detection, must/must-not matching (plain + 're:'), spoken-
format checks, follow-up detection, per-category scoring, the run summary,
the run loop's timeout / Ctrl-C behaviour (fake client), and the shape of
tests/llm_eval_set.json.

llm_eval_set.json is a bare JSON list of items {id, category, question,
must_contain_any, must_not_contain, expect_refusal, expect_followup, note}.
Categories: kb (answerable from data/knowledge.txt, phrased unlike the KB
wording), general (small talk / world knowledge a 1-3B model should manage),
scope (the right answer is "I can't help with that / I don't have that
information" WITHOUT inventing facts). must_contain_any: case-insensitive
substrings, any one satisfies, 're:' prefix = regex. must_not_contain:
substrings / 're:' regexes that mark an invented fact. expect_followup:
true / false / null (null = don't care).
"""

import json
import os
import subprocess
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import bench_llm  # noqa: E402

EVAL_SET = os.path.join(_HERE, "llm_eval_set.json")


class TestRefusal(unittest.TestCase):
    def test_positives(self):
        for text in [
            "That's outside what I can help with, sorry!",
            "I don't have that information yet. Ask at the department office?",
            "I do not have that information, but I can tell you about the labs.",
            "Checking the weather isn't something I can do.",
            "I can't help with booking a cab, but I can point you to the store.",
            "I cannot help with that one.",
            "I'm not able to see the time.",
            "I'm unable to access live news.",
            "Hmm, I don't know the principal's name.",
            "I'm not sure about the hostel fee.",
            "I have no information about the mechanical lab.",
            "You could ask at the department office for the fee.",
            "That's beyond what I know.",
            "I can't check the stock market from here.",
            "I can't tell you the time, I have no clock!",
            "I don't have access to your grades.",
            "That’s outside what I can do right now.",   # curly apostrophe
            # natural small-model phrasings the first cut missed
            "I don't have information about the fee.",
            "Sorry, I have no idea about the principal.",
            "Hmm, no idea, sorry.",
            "I have no clue what the hostel costs.",
            "I couldn't tell you the time.",
            "That's not in my knowledge base.",
            "That's beyond me.",
            "I only know about the college, not the weather.",
            "I really don't know the principal's name.",
            "Not sure about that one, sorry!",
            "I'm honestly not quite sure about the fee.",
        ]:
            self.assertTrue(bench_llm.is_refusal(text), text)

    def test_negatives(self):
        for text in [
            "The college was established in 1958.",
            "Dr Aneesh G Nath heads the department. Want to know where his room is?",
            "Paris is the capital of France!",
            "The canteen is opposite Somanathan centre, near the back gate.",
            "",
            # chit-chat that merely contains 'not sure' / 'don't know' / 'no idea'
            "I'm doing great, thanks! Not sure what you're up to today?",
            "Don't know about you, but I love mornings!",
            "I don't know about you, but I love the campus!",
            "Great, I'm sure you'll love the campus!",
            "No idea? Ask me anything about the college!",
        ]:
            self.assertFalse(bench_llm.is_refusal(text), text)


class TestMatching(unittest.TestCase):
    def test_contains_any_plain(self):
        self.assertTrue(bench_llm.contains_any("Founded in 1958!", ["1958"]))
        self.assertTrue(bench_llm.contains_any("Dr ANEESH G Nath", ["aneesh"]))
        self.assertFalse(bench_llm.contains_any("no idea", ["1958", "aneesh"]))
        self.assertTrue(bench_llm.contains_any("anything", []))   # vacuous

    def test_contains_any_regex(self):
        self.assertTrue(bench_llm.contains_any("NAAC gave an A grade",
                                               ["re:\\ba\\W{0,3}grade\\b"]))
        self.assertFalse(bench_llm.contains_any("NAAC gave a B grade",
                                                ["re:\\ba\\W{0,3}grade\\b"]))

    def test_violates_plain(self):
        self.assertTrue(bench_llm.violates("The president is Joe Biden.",
                                           ["biden", "trump"]))
        self.assertFalse(bench_llm.violates("I can't check live news.",
                                            ["biden", "trump"]))
        self.assertFalse(bench_llm.violates("anything", []))

    def test_violates_regex(self):
        pat = ["re:\\bdr\\.?\\s+[a-z]"]
        self.assertTrue(bench_llm.violates("The principal is Dr. Ravi Kumar.", pat))
        self.assertTrue(bench_llm.violates("Principal: Dr Sunil.", pat))
        self.assertFalse(bench_llm.violates("I don't have the principal's name.", pat))
        self.assertTrue(bench_llm.violates("Call 9876543210.", ["re:\\d{6,}"]))
        self.assertFalse(bench_llm.violates("Room 204.", ["re:\\d{6,}"]))
        hits = bench_llm.must_not_hits("Call 9876543210 now", ["re:\\d{6,}", "later"])
        self.assertEqual(hits, ["re:\\d{6,}"])

    def test_bad_regex_does_not_raise(self):
        self.assertFalse(bench_llm.violates("text", ["re:("]))


class TestFormat(unittest.TestCase):
    def test_markdown_reply_is_flagged(self):
        reply = ("Here are the clubs:\n* Coding club\n* CSI\n- IEEE\n"
                 "**Note:** join early! `code` 😀")
        v = bench_llm.format_violations(reply)
        for label in ("list-marker", "bold", "backtick", "emoji"):
            self.assertIn(label, v)

    def test_numbered_list_is_flagged(self):
        v = bench_llm.format_violations("Clubs:\n1. Coding club\n2. CSI\n3. IEEE")
        self.assertIn("numbered-list", v)
        # a single "12. " style line is not a list (could be an answer)
        self.assertNotIn("numbered-list",
                         bench_llm.format_violations("12. That is twelve."))

    def test_clean_spoken_reply(self):
        reply = ("The college was established in 1958 by Janab A. Thangal "
                 "Kunju Musaliar. Want to know where it is?")
        self.assertEqual(bench_llm.format_violations(reply), [])

    def test_abbreviations_and_decimals_do_not_split(self):
        reply = ("Dr. Aneesh G Nath is the HOD. The average package is 5.03 LPA. "
                 "Anything else about placements?")
        self.assertEqual(len(bench_llm.split_sentences(reply)), 3)
        self.assertEqual(bench_llm.format_violations(reply), [])

    def test_initials_do_not_split(self):
        """Single-letter initials straight from knowledge.txt ("Janab A.
        Thangal", "Aneesh G. Nath") must not push a 4-sentence reply over
        the limit."""
        reply = ("It was founded by Janab A. Thangal Kunju Musaliar in 1958. "
                 "Neat, huh? What else? Curious about the labs?")
        self.assertEqual(len(bench_llm.split_sentences(reply)), 4)
        self.assertEqual(bench_llm.format_violations(reply), [])
        reply = ("The HOD is Dr. Aneesh G. Nath. I don't have a headcount. "
                 "Ask at the office. Anything else?")
        self.assertEqual(len(bench_llm.split_sentences(reply)), 4)
        self.assertNotIn("5-sentences", bench_llm.format_violations(reply))
        # a fifth real sentence is still counted
        self.assertEqual(len(bench_llm.split_sentences(
            "One. Two. Three. Four. Five.")), 5)

    def test_too_long(self):
        five = "One. Two. Three. Four. Five."
        self.assertIn("5-sentences", bench_llm.format_violations(five))
        long = " ".join(["word"] * 95) + "."
        self.assertTrue(any(v.endswith("-words")
                            for v in bench_llm.format_violations(long)))

    def test_empty(self):
        self.assertEqual(bench_llm.format_violations(""), ["empty"])
        self.assertEqual(bench_llm.format_violations("   \n"), ["empty"])

    def test_rupee_is_not_emoji(self):
        self.assertNotIn("emoji",
                         bench_llm.format_violations("The package is ₹5.03 LPA."))


class TestFollowup(unittest.TestCase):
    def test_ends_with_question(self):
        self.assertTrue(bench_llm.has_followup("It started in 1958. Curious about the founder?"))

    def test_question_in_middle(self):
        self.assertTrue(bench_llm.has_followup("Want the founder too? It was 1958."))

    def test_no_question(self):
        self.assertFalse(bench_llm.has_followup("Bye! Have a lovely day."))
        self.assertFalse(bench_llm.has_followup(""))


class TestScoring(unittest.TestCase):
    KB = {"id": "kb_x", "category": "kb", "must_contain_any": ["1958"],
          "must_not_contain": ["1985"], "expect_followup": None}
    GEN = {"id": "gen_x", "category": "general", "must_contain_any": ["paris"],
           "must_not_contain": [], "expect_followup": None}
    GEN_FREE = {"id": "gen_hi", "category": "general", "must_contain_any": [],
                "must_not_contain": [], "expect_followup": True}
    SCOPE = {"id": "scope_x", "category": "scope", "must_contain_any": [],
             "must_not_contain": ["re:\\d{6,}"], "expect_followup": None}

    def test_kb(self):
        g = bench_llm.score_item(self.KB, "It was founded in 1958. Want more?")
        self.assertEqual(g["score"], 1.0)
        self.assertEqual(g["symbol"], "✓")
        self.assertFalse(g["missed_kb"])
        # refusal on a kb item -> 0 and tallied as missed_kb
        g = bench_llm.score_item(self.KB, "I don't have that information.")
        self.assertEqual(g["score"], 0.0)
        self.assertTrue(g["missed_kb"])
        self.assertEqual(g["symbol"], "R")
        # wrong fact -> 0
        g = bench_llm.score_item(self.KB, "It was founded in 1960.")
        self.assertEqual(g["score"], 0.0)
        self.assertEqual(g["symbol"], "✗")
        # must-not hit -> 0 and hallucination
        g = bench_llm.score_item(self.KB, "Founded in 1958, or was it 1985?")
        self.assertEqual(g["score"], 0.0)
        self.assertTrue(g["halluc"])
        self.assertEqual(g["symbol"], "H")

    def test_general(self):
        self.assertEqual(bench_llm.score_item(self.GEN, "Paris!")["score"], 1.0)
        self.assertEqual(bench_llm.score_item(self.GEN, "London.")["score"], 0.0)
        self.assertEqual(bench_llm.score_item(self.GEN, "I'm not sure.")["score"], 0.0)
        # no must_contain_any: any non-refusal answer scores
        g = bench_llm.score_item(self.GEN_FREE, "I'm great! How is your day going?")
        self.assertEqual(g["score"], 1.0)
        self.assertTrue(g["followup"])
        self.assertTrue(g["followup_ok"])
        g = bench_llm.score_item(self.GEN_FREE, "I'm great.")
        self.assertEqual(g["score"], 1.0)
        self.assertFalse(g["followup_ok"])   # expected a follow-up

    def test_scope(self):
        clean = "I can't help with phone numbers, but the HOD's room is on the ground floor."
        g = bench_llm.score_item(self.SCOPE, clean)
        self.assertEqual(g["score"], 1.0)
        hedged = "The HOD is Dr Aneesh G Nath, he sits on the ground floor."
        g = bench_llm.score_item(self.SCOPE, hedged)
        self.assertEqual(g["score"], 0.5)
        self.assertEqual(g["symbol"], "~")
        invented = "Sure, call him on 9876543210."
        g = bench_llm.score_item(self.SCOPE, invented)
        self.assertEqual(g["score"], 0.0)
        self.assertTrue(g["halluc"])
        self.assertEqual(g["symbol"], "H")
        # invented AND refused still counts as invented
        g = bench_llm.score_item(self.SCOPE, "I don't know, maybe 9876543210?")
        self.assertEqual(g["score"], 0.0)

    def test_empty_reply_scores_zero_everywhere(self):
        for item in (self.KB, self.GEN, self.GEN_FREE, self.SCOPE):
            g = bench_llm.score_item(item, "")
            self.assertEqual(g["score"], 0.0, item["id"])
            self.assertIn("empty", g["format_violations"])

    def test_format_marks_symbol(self):
        g = bench_llm.score_item(self.GEN, "**Paris**")
        self.assertEqual(g["symbol"], "✓*")


class TestSummary(unittest.TestCase):
    def _run(self, answers):
        return {"model": "m", "variant": "rag", "size_gb": 1.0,
                "warmup": {"load_s": 0.5}, "answers": answers}

    def test_composite_and_medians(self):
        items = [
            ({"id": "k1", "category": "kb", "must_contain_any": ["1958"]}, "In 1958."),
            ({"id": "k2", "category": "kb", "must_contain_any": ["aneesh"]}, "No idea, I don't know."),
            ({"id": "g1", "category": "general", "must_contain_any": ["paris"]}, "Paris."),
            ({"id": "s1", "category": "scope", "must_not_contain": ["re:\\d{6,}"]},
             "I can't help with that."),
            ({"id": "s2", "category": "scope", "must_not_contain": ["re:\\d{6,}"]},
             "* 9876543210"),
        ]
        answers = []
        for n, (item, reply) in enumerate(items):
            g = bench_llm.score_item(item, reply)
            answers.append({"id": item["id"], "category": item["category"],
                            "ttft_s": 0.1 * (n + 1), "total_s": 1.0,
                            "gen_tok_s": 10.0 + n, "prefill_tok_s": 100.0,
                            "prompt_eval_count": 50, **g})
        s = bench_llm.summarize_run(self._run(answers))
        self.assertEqual(s["kb"], 0.5)
        self.assertEqual(s["general"], 1.0)
        self.assertEqual(s["scope"], 0.5)          # (1.0 + 0.0) / 2
        self.assertEqual(s["format"], 0.8)         # one list-marker reply
        self.assertEqual(s["halluc"], 1)
        self.assertEqual(s["missed_kb"], 1)
        expected = 0.40 * 0.5 + 0.30 * 0.5 + 0.15 * 1.0 + 0.15 * 0.8
        self.assertAlmostEqual(s["composite"], round(expected, 3))
        self.assertAlmostEqual(s["ttft_s"], 0.3)
        self.assertEqual(s["gen_tok_s"], 12.0)
        self.assertIsNone(s["cache_x"])            # warm-up had no prefill data

    def test_cache_ratio_from_warmup(self):
        item = {"id": "k1", "category": "kb", "must_contain_any": ["1958"]}
        g = bench_llm.score_item(item, "1958.")
        run = self._run([{"id": "k1", "category": "kb", "ttft_s": 0.2,
                          "total_s": 1.0, "gen_tok_s": 9.0,
                          "prefill_tok_s": 5000.0, "prompt_eval_count": 600,
                          **g}])
        # uncached warm-up prefilled 500 tokens in 1 s -> 500 tok/s; the
        # per-turn apparent rate of 5000 tok/s means the prefix cache hit
        run["warmup"] = {"load_s": 1.0, "prompt_eval_count": 500,
                         "prompt_eval_s": 1.0}
        s = bench_llm.summarize_run(run)
        self.assertEqual(s["warm_prefill_tok_s"], 500.0)
        self.assertEqual(s["cache_x"], 10.0)

    def test_markdown_render_lists_every_question(self):
        item = {"id": "k1", "category": "kb", "must_contain_any": ["1958"],
                "question": "when"}
        g = bench_llm.score_item(item, "1958.")
        run = self._run([{"id": "k1", "category": "kb", "ttft_s": 0.2,
                          "total_s": 1.0, "gen_tok_s": 9.0,
                          "prefill_tok_s": 90.0, "prompt_eval_count": 40, **g}])
        run["summary"] = bench_llm.summarize_run(run)
        md = bench_llm.render_summary_md([run], [item], {"platform": "x"},
                                         {"timestamp": "t", "options": {}})
        self.assertIn("| m | rag |", md)
        self.assertIn("| k1 | ✓ |", md)


class TestEvalSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = bench_llm.load_eval_set(EVAL_SET)
        cls.by_id = {i["id"]: i for i in cls.items}

    def _grade(self, item_id, reply):
        return bench_llm.score_item(self.by_id[item_id], reply)

    def test_file_is_a_bare_json_list(self):
        """Other consumers json.load the file as a list (the spec'd shape)."""
        with open(EVAL_SET, encoding="utf-8") as f:
            data = json.load(f)
        self.assertIsInstance(data, list)
        self.assertEqual(data[0]["id"], self.items[0]["id"])

    def test_scope_principal_refusal_is_not_a_hallucination(self):
        """A correct refusal that merely says 'the principal is ...' scores 1;
        an invented titled name is still an H."""
        for reply in [
            "I'm not sure who the principal is at TKM college of Engineering.  "
            "I'd need to check with the department office, though.  "
            "Would you like me to help you find the department office?",
            "I don't know who the principal is right now, sorry! Ask at the office?",
        ]:
            g = self._grade("scope_principal", reply)
            self.assertEqual(g["must_not_hits"], [], reply)
            self.assertEqual(g["score"], 1.0, reply)
            self.assertEqual(g["symbol"], "✓", reply)
        for reply in ["The principal is Dr. Ravi Kumar.",
                      "The principal is Mr. Kumar, I think.",
                      "Our principal is Prof Sunil."]:
            g = self._grade("scope_principal", reply)
            self.assertTrue(g["halluc"], reply)
            self.assertEqual(g["score"], 0.0, reply)

    def test_kb_naac_article_is_not_the_grade(self):
        """'a grade of B' is a wrong answer (H), not a match on the article."""
        for reply in ["NAAC gave the college a grade of B.",
                      "NAAC gave it a B grade.",
                      "NAAC gave it an A+ grade.",
                      "The NAAC grade is C, I think."]:
            g = self._grade("kb_naac", reply)
            self.assertTrue(g["halluc"], reply)
            self.assertEqual(g["score"], 0.0, reply)
        for reply in ["NAAC gave the college a grade of A.",
                      "The college is NAAC accredited with an A grade.",
                      "The college is NAAC accredited with A grade. Neat, huh?",
                      "It got grade A from NAAC.",
                      "NAAC graded it A.",
                      "NAAC grade: A."]:
            g = self._grade("kb_naac", reply)
            self.assertEqual(g["must_not_hits"], [], reply)
            self.assertEqual(g["score"], 1.0, reply)
        # no grade stated at all -> neither a match nor a hallucination
        g = self._grade("kb_naac", "NAAC gave it a grade. Want to know which?")
        self.assertEqual(g["score"], 0.0)
        self.assertFalse(g["halluc"])

    def test_gen_hi_closing_question_is_not_a_refusal(self):
        g = self._grade("gen_hi", "I'm doing great, thanks! Not sure what you're up to today?")
        self.assertFalse(g["refusal"])
        self.assertEqual(g["score"], 1.0)
        self.assertTrue(g["followup_ok"])

    def test_size_and_ids(self):
        self.assertGreaterEqual(len(self.items), 38)
        ids = [i["id"] for i in self.items]
        self.assertEqual(len(ids), len(set(ids)), "duplicate ids")

    def test_fields_and_categories(self):
        for it in self.items:
            self.assertIn(it["category"], bench_llm.VALID_CATEGORIES, it["id"])
            self.assertTrue(it["question"].strip(), it["id"])
            self.assertIsInstance(it.get("must_contain_any", []), list)
            self.assertIsInstance(it.get("must_not_contain", []), list)
            self.assertIn(it.get("expect_followup"), (True, False, None))
            self.assertIsInstance(it["expect_refusal"], bool)

    def test_category_counts(self):
        counts = {c: sum(1 for i in self.items if i["category"] == c)
                  for c in bench_llm.CATEGORIES}
        self.assertGreaterEqual(counts["kb"], 14)
        self.assertGreaterEqual(counts["general"], 8)
        self.assertGreaterEqual(counts["scope"], 12)

    def test_scope_items_expect_refusal(self):
        for it in self.items:
            if it["category"] == "scope":
                self.assertTrue(it["expect_refusal"], it["id"])
            else:
                self.assertFalse(it["expect_refusal"], it["id"])

    def test_kb_items_have_must_contain(self):
        for it in self.items:
            if it["category"] == "kb":
                self.assertTrue(it["must_contain_any"], it["id"])

    def test_regex_patterns_compile(self):
        import re
        for it in self.items:
            for p in it.get("must_contain_any", []) + it.get("must_not_contain", []):
                if p.startswith("re:"):
                    re.compile(p[3:])   # raises on a bad pattern

    def test_kb_facts_are_in_knowledge_txt(self):
        """Every kb item's expected substrings really are in knowledge.txt
        (guards the eval set against KB edits drifting away from it)."""
        qs, ans, raw = bench_llm.load_knowledge(
            os.path.join(_ROOT, "data", "knowledge.txt"))
        self.assertGreater(len(qs), 40)
        kb_text = raw.lower()
        for it in self.items:
            if it["category"] != "kb":
                continue
            plain = [s for s in it["must_contain_any"] if not s.startswith("re:")]
            if plain:
                self.assertTrue(any(s.lower() in kb_text for s in plain), it["id"])


class _FakeKB:
    """Enough of bench_llm.KnowledgeBase for build_messages: no retrieval."""
    text = "q:a"
    k, min_sim = 1, 0.0

    def top_k(self, query):
        return []


class _FakeClient:
    """Records the timeout every chat_stream call gets; optionally raises
    KeyboardInterrupt (or a timeout) on the n-th call."""

    def __init__(self, interrupt_at=None, timeout_at=None):
        self.timeouts = []
        self.interrupt_at = interrupt_at
        self.timeout_at = timeout_at

    def chat_stream(self, model, messages, options, keep_alive, think=None,
                    timeout=None):
        self.timeouts.append(timeout)
        n = len(self.timeouts)
        if n == self.interrupt_at:
            raise KeyboardInterrupt
        if n == self.timeout_at:
            raise TimeoutError("timed out")
        return {"reply": "It was 1958. Want more?", "ttft_s": 0.1,
                "total_s": 0.5, "eval_count": 8, "eval_duration_s": 0.2,
                "gen_tok_s": 40.0, "prompt_eval_count": 50,
                "prompt_eval_s": 0.1, "prefill_tok_s": 500.0, "load_s": 0.3,
                "ollama_total_s": 0.5, "done_reason": "stop"}


class TestRunLoop(unittest.TestCase):
    ITEMS = [{"id": f"k{n}", "category": "kb", "question": "when did it start",
              "must_contain_any": ["1958"]} for n in range(4)]

    def _run(self, client):
        return bench_llm.run_model_variant(client, "m", "rag", self.ITEMS,
                                           _FakeKB(), {"num_predict": 8}, None,
                                           1.0)

    def test_questions_get_the_bench_timeout_not_lunas(self):
        """Warm-up uses the load budget; every timed question uses
        BENCH_QUESTION_TIMEOUT (config.OLLAMA_TIMEOUT is far too short for
        an uncached prefill on the Pi)."""
        client = _FakeClient()
        run = self._run(client)
        self.assertEqual(client.timeouts[0], bench_llm.BENCH_LOAD_TIMEOUT)
        self.assertEqual(client.timeouts[1:],
                         [bench_llm.BENCH_QUESTION_TIMEOUT] * len(self.ITEMS))
        self.assertGreaterEqual(bench_llm.BENCH_QUESTION_TIMEOUT,
                                bench_llm.config.OLLAMA_TIMEOUT)
        self.assertFalse(run["interrupted"])
        self.assertEqual(run["summary"]["n"], 4)

    def test_timeout_is_recorded_as_too_slow(self):
        run = self._run(_FakeClient(timeout_at=3))   # warm-up + q1 ok, q2 stalls
        slow = run["answers"][1]
        self.assertEqual(slow["symbol"], "E")
        self.assertIn("too slow", slow["error"])
        self.assertIn(f"{bench_llm.BENCH_QUESTION_TIMEOUT:.0f}s", slow["error"])
        self.assertEqual(run["summary"]["errors"], 1)
        self.assertEqual(len(run["answers"]), 4)   # the run carries on

    def test_ctrl_c_keeps_the_answers_so_far(self):
        run = self._run(_FakeClient(interrupt_at=4))  # warm-up + 2 ok, 3rd hit
        self.assertTrue(run["interrupted"])
        self.assertEqual([a["id"] for a in run["answers"]], ["k0", "k1"])
        self.assertEqual(run["summary"]["n"], 2)
        self.assertEqual(run["summary"]["n_items"], 4)
        self.assertTrue(run["summary"]["interrupted"])
        self.assertEqual(run["summary"]["kb"], 1.0)
        md = bench_llm.render_summary_md([run], self.ITEMS, {"platform": "x"},
                                         {"timestamp": "t", "options": {}})
        self.assertIn("PARTIAL", md)
        self.assertIn("2 of 4", md)
        self.assertIn("| k3 | · |", md)     # unanswered question shown blank

    def test_ctrl_c_during_warmup_propagates(self):
        with self.assertRaises(KeyboardInterrupt):
            self._run(_FakeClient(interrupt_at=1))


class TestImportIsLazy(unittest.TestCase):
    def test_import_does_not_load_heavy_deps(self):
        """Importing the script must not touch Ollama or load the sentence
        transformer — checked in a fresh interpreter."""
        code = ("import sys; sys.path.insert(0, %r); import bench_llm; "
                "print('sentence_transformers' in sys.modules, "
                "'torch' in sys.modules)" % os.path.join(_ROOT, "scripts"))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, timeout=60, cwd=_ROOT)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "False False", out.stdout)


if __name__ == "__main__":
    unittest.main()
