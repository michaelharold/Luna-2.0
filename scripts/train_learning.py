#!/usr/bin/env python3
# scripts/train_learning.py
"""
train_learning.py — bootstrap Luna's learning state on the Mac.

WHY: on-device learning starts from zero — the style bandit explores
uniformly until each context has LEARNING_MIN_TURNS_TO_STEER rewarded turns,
and there are no liked replies to show as examples. Two ways to give it a
head start before the first real visitor (then deploy ONCE with
`scripts/deploy.sh --go --seed-learning`):

  --replay        Rebuild data/learning/ (bandit, exemplars, profiles) from
                  the REAL conversation logs' rewards. Wipes those three
                  files first (pending facts + diary are kept); profile visit
                  counts come only from logged turns.
  --simulate N    N simulated visitor sessions through local Ollama models:
                  a visitor model plays a persona with a goal (a knowledge-
                  base topic, a general question, or something out of scope)
                  for 3-5 turns; Luna answers exactly the way brain.py
                  builds her prompt (persona.build_messages + RAG-lite
                  retrieval like scripts/bench_llm.py + the bandit's chosen
                  style + exemplars + profile); a judge model scores each
                  reply 1-5 (<=2 -> -1, 3 -> 0, 4 -> +0.5, 5 -> +1) and the
                  visitor's next line feeds the same implicit signals as real
                  life (thanks, repeats, "that's wrong", engagement). Sessions
                  are logged to data/logs_sim/ — never mixed with real logs.

Usage (from the project root):

    venv/bin/python scripts/train_learning.py --simulate 20
    venv/bin/python scripts/train_learning.py --simulate 2 \
        --visitor-model llama3.2:3b --judge-model llama3.2:3b --luna-model luna
    venv/bin/python scripts/train_learning.py --replay [--include-sim]

--luna-model defaults to OLLAMA_MODEL and falls back to OLLAMA_BASE_MODEL
when that isn't listed by Ollama (e.g. the "luna" model hasn't been built).
"""

import argparse
import os
import random
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)

import config            # noqa: E402
import persona           # noqa: E402
import conversation_log  # noqa: E402
import learning          # noqa: E402
import bench_llm         # noqa: E402  (Ollama client + RAG-lite KB)

SIM_LOG_DIR = "data/logs_sim"
KEEP_ALIVE = "10m"
TIMEOUT = 300.0

VISITORS = (
    "a nervous first-year student who just joined the college",
    "a parent visiting to decide whether their child should apply",
    "a final-year CS student in a hurry between classes",
    "a curious school kid on a campus tour",
    "a visiting professor from another college",
    "an alumnus who graduated ten years ago",
    "a faculty member testing the robot",
)
NAMES = ("Meera", "Rahul", "Anjali", "Arjun", "Fathima", "Joseph")
GENERAL_GOALS = (
    "ask what photosynthesis is",
    "ask for a quick tip to remember the planets",
    "ask what a robot like her can do",
    "chat about her favourite colour and then ask a science question",
    "ask what 12 times 15 is",
    "ask what machine learning means in simple words",
    "ask how to stay focused while studying",
)
SCOPE_GOALS = (
    "ask what the weather will be like tomorrow",
    "ask for today's cricket score",
    "ask her to book you a cab to the railway station",
    "ask the current price of an iPhone",
    "ask for today's news headlines",
    "ask her to call your friend",
)

JUDGE_SYSTEM = (
    "You grade replies spoken by Luna, a small friendly desk robot at the "
    "Computer Science department of TKM College of Engineering, Kollam. A "
    "good reply: is correct and consistent with the FACTS given (inventing "
    "names, numbers, fees or rules about the college is the worst failure); "
    "for things outside what she can know or do (weather, news, live scores, "
    "prices, booking, calling) politely says so instead of guessing; is short "
    "and easy to say out loud (1-3 sentences, no lists or markdown); is warm "
    "and fits what the visitor actually asked. Be strict: a reply that "
    "pretends to do something a desk robot cannot do (call, book, look up "
    "live information) scores 1; a reply that adds college details that are "
    "NOT in the FACTS scores 1 or 2; a vague reply that doesn't answer "
    "scores 3 at most. Answer with ONE digit from 1 (bad) to 5 (excellent) "
    "and nothing else.")

JUDGE_TO_REWARD = {1: -1.0, 2: -1.0, 3: 0.0, 4: 0.5, 5: 1.0}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def vosk_like(text):
    """What Vosk would hand brain: lower case, no punctuation."""
    t = (text or "").lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def clean_line(text):
    """First line of a model's utterance, without quotes / stage directions."""
    t = (text or "").strip().split("\n")[0]
    t = re.sub(r"\*[^*]*\*|\([^)]*\)", "", t)
    return t.strip().strip('"').strip()


def judge_reward(score):
    return JUDGE_TO_REWARD.get(score, 0.0)


def parse_score(text):
    m = re.search(r"[1-5]", text or "")
    return int(m.group(0)) if m else 3


def resolve_model(name, tags, fallback, role):
    """`name` if Ollama lists it (with or without :latest), else fallback."""
    listed = set(tags)
    for cand in (name, f"{name}:latest"):
        if cand in listed:
            return cand
    for cand in (fallback, f"{fallback}:latest"):
        if cand in listed:
            print(f"[train] {role} model {name!r} not listed — using "
                  f"{cand!r}")
            return cand
    raise SystemExit(f"[train] neither {name!r} nor {fallback!r} is available "
                     f"in Ollama ({', '.join(sorted(listed)) or 'none'})")


def load_embedder():
    """brain.embed without importing brain (no TTS / servos / threads)."""
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import numpy as np
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")

        def embed(texts):
            raw = np.asarray(model.encode(list(texts), batch_size=32,
                                          show_progress_bar=False),
                             dtype=np.float32)
            return raw / np.maximum(np.linalg.norm(raw, axis=1,
                                                   keepdims=True), 1e-9)
        return embed
    except Exception as e:                           # noqa: BLE001
        print(f"[train] sentence transformer unavailable ({e}) — exemplars "
              f"and repeat detection off")
        return None


class Chat:
    """Non-throwing chat calls through bench_llm's Ollama client."""

    def __init__(self, client, rng):
        self.client = client
        self.rng = rng

    def __call__(self, model, messages, temperature=0.7, num_predict=None):
        opts = {"temperature": temperature,
                "num_predict": num_predict or config.LLM_MAX_TOKENS,
                "num_ctx": config.OLLAMA_NUM_CTX,
                "seed": self.rng.randrange(1, 2 ** 31)}
        try:
            res = self.client.chat_stream(model, messages, opts, KEEP_ALIVE,
                                          timeout=TIMEOUT)
        except Exception as e:                       # noqa: BLE001
            print(f"[train] {model} request failed: "
                  f"{bench_llm._error_text(e, TIMEOUT)}")
            return {"reply": "", "ttft_s": None, "total_s": None}
        return res


# ══════════════════════════════════════════════════════════════════════════════
# Simulation
# ══════════════════════════════════════════════════════════════════════════════
def make_goal(rng, kb):
    kind = rng.choice(("kb", "kb", "general", "scope"))
    if kind == "kb" and kb.questions:
        q = rng.choice(kb.questions)
        return kind, f"find out: {q}"
    if kind == "scope":
        return kind, rng.choice(SCOPE_GOALS)
    return "general", rng.choice(GENERAL_GOALS)


def visitor_line(chat, model, who, goal, transcript, first):
    """The visitor's next line. transcript = [(speaker, text)]."""
    system = (
        f"You are role-playing {who} talking out loud to Luna, a small desk "
        f"robot at the TKM College of Engineering Computer Science booth. "
        f"Your goal: {goal}. Speak like a real person: ONE short casual "
        f"sentence, no quotes, no stage directions, no emoji, never mention "
        f"that you are role-playing or an AI.")
    if not first:
        system += (
            " React to Luna's last reply naturally: if it helped you may thank "
            "her or ask a short follow-up about your goal; if it was wrong, "
            "vague or dodged your question, say so or ask again; if you are "
            "done, say goodbye.")
    msgs = [{"role": "system", "content": system}]
    for speaker, text in transcript:
        msgs.append({"role": "assistant" if speaker == "visitor" else "user",
                     "content": text})
    if first:
        msgs.append({"role": "user", "content": "(Luna looks at you.) Say "
                                                "your first line to Luna."})
    res = chat(model, msgs, temperature=0.9, num_predict=60)
    return clean_line(res.get("reply"))


def judge(chat, model, question, reply, snippets):
    facts = "\n".join(f"- {q}: {a}" for q, a, _ in snippets) or "(none)"
    msgs = [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content":
                f"FACTS Luna had:\n{facts}\n\nVISITOR: {question}\n"
                f"LUNA: {reply}\n\nScore (1-5):"}]
    res = chat(model, msgs, temperature=0.0, num_predict=4)
    return parse_score(res.get("reply"))


def run_session(i, n, chat, models, kb, rng, bake):
    who = rng.choice(VISITORS)
    person = rng.choice(NAMES) if rng.random() < 0.4 else None
    kind, goal = make_goal(rng, kb)
    turns = rng.randint(3, 5)
    print(f"\n=== session {i}/{n}: {who}"
          f"{' (' + person + ', recognised)' if person else ''} — {goal} "
          f"[{kind}, {turns} turns]")
    now = time.time()
    if person:
        learning.note_visit(person, now)
    history, transcript = [], []
    said = visitor_line(chat, models["visitor"], who, goal, transcript, True)
    last_judge = None
    for t in range(turns):
        if not said:
            print("  (visitor model gave nothing — ending session)")
            break
        heard = vosk_like(said)
        transcript.append(("visitor", said))
        now += 3.0                                     # visitor speaks
        if last_judge is not None:
            _print_reward(reward_previous(heard, person, now, last_judge))
        snippets = kb.top_k(heard)
        kb_score = snippets[0][2] if snippets else 0.0
        context = learning.context_key(
            person, learning.question_kind(heard, kb_score), now)
        arm, instruction = learning.choose_style(context)
        extra = {}
        if instruction:
            extra["style"] = instruction
        ex = learning.exemplars_for(heard)
        if ex:
            extra["exemplars"] = ex
        prof = learning.profile_context(person, now) if person else None
        if prof:
            extra["profile"] = prof
        msgs = persona.build_messages(
            heard, history=history, knowledge_text=kb.text if bake else None,
            kb_snippets=snippets, person=person, **extra)
        res = chat(models["luna"], msgs, temperature=config.LLM_TEMPERATURE)
        reply = (res.get("reply") or "").strip()
        if not reply:
            print("  (Luna model gave nothing — ending session)")
            break
        spoken = 1.5 + 0.33 * len(reply.split())          # ~3 words/s
        now += spoken
        conversation_log.new_turn(
            ts=now, heard_raw=said, heard_clean=heard, person=person,
            path="llm", kb_matches=[[q, round(s, 3)] for q, _, s in snippets],
            reply=reply, style_arm=arm, context=context,
            timings={"ttft_s": res.get("ttft_s"), "total_s":
                     round((res.get("total_s") or 0.0) + spoken, 3)},
            sim={"visitor": who, "goal": goal, "kind": kind,
                 "luna_model": models["luna"]})
        learning.observe_turn(conversation_log.last_turn() or {})
        history += [{"role": "user", "content": heard},
                    {"role": "assistant", "content": reply}]
        history = history[-config.LLM_MAX_HISTORY:]
        transcript.append(("luna", reply))
        score = judge(chat, models["judge"], heard, reply, snippets)
        last_judge = judge_reward(score)
        print(f"  visitor: {said}")
        print(f"  luna   : {reply}")
        print(f"           [{context} · {arm}] judge {score} -> "
              f"{last_judge:+.1f}")
        said = visitor_line(chat, models["visitor"], who, goal, transcript,
                            False)
    if last_judge is not None:
        now += 3.0
        if said:
            print(f"  visitor: {said}")
        _print_reward(reward_previous(vosk_like(said), person, now,
                                      last_judge))


def reward_previous(heard, person, now, judge_r):
    """Reward Luna's previous reply: the judge's verdict plus the visitor's
    implicit signals (thanks, repeat, "that's wrong", engagement) — but a
    reply the judge did NOT approve can't be rescued by a polite visitor:
    then only the negative implicit signals count. (Simulated visitors thank
    almost everything; without this, hallucinations became exemplars.)"""
    prev = conversation_log.last_turn()
    if not prev or prev.get("reward") is not None:
        return None
    implicit = (learning.score_signals(prev, heard, person, now, None,
                                       learning.similarity)
                if heard else [])
    signals = [("judge", judge_r)] + [
        (s, v) for s, v in implicit if judge_r > 0 or v < 0]
    return learning.add_signals(signals, now=now)


def _print_reward(r):
    if r is not None:
        t = conversation_log.last_turn() or {}
        sig = ", ".join(f"{s} {v:+.2f}" for s, v in t.get("reward_signals")
                        or [])
        print(f"           reward {r:+.2f} ({sig})")


def simulate(args):
    rng = random.Random(args.seed)
    client = bench_llm.Ollama(config.OLLAMA_HOST, TIMEOUT)
    try:
        tags = client.tags()
    except Exception as e:                           # noqa: BLE001
        raise SystemExit(f"[train] Ollama unreachable at {config.OLLAMA_HOST}"
                         f": {e}")
    luna = resolve_model(args.luna_model or config.OLLAMA_MODEL, tags,
                         config.OLLAMA_BASE_MODEL, "Luna")
    models = {
        "luna": luna,
        "visitor": resolve_model(args.visitor_model, tags, luna, "visitor"),
        "judge": resolve_model(args.judge_model, tags, luna, "judge"),
    }
    print(f"[train] models: {models}")
    # Same system prompt brain.py sends on this backend (whole KB baked in
    # on the Ollama path when OLLAMA_BAKE_KNOWLEDGE).
    bake = (config.LLM_BACKEND == "ollama" and config.OLLAMA_BAKE_KNOWLEDGE)
    kb = bench_llm.KnowledgeBase(config.KNOWLEDGE_PATH)
    conversation_log.configure(log_dir=args.log_dir, mode="sim")
    chat = Chat(client, rng)
    for i in range(1, args.simulate + 1):
        run_session(i, args.simulate, chat, models, kb, rng, bake)
    report()


# ══════════════════════════════════════════════════════════════════════════════
# Replay
# ══════════════════════════════════════════════════════════════════════════════
def replay(args):
    dirs = [config.CONVO_LOG_DIR] + ([args.log_dir] if args.include_sim else [])
    learning.reset(delete_files=True)
    total, rewarded = 0, 0
    for d in dirs:
        conversation_log.configure(log_dir=d)
        for day in conversation_log.days():
            turns = conversation_log.iter_day(day)
            total += len(turns)
            rewarded += learning.replay_turns(turns)
    print(f"[train] replayed {total} records from {', '.join(dirs)}: "
          f"{rewarded} rewarded turns applied")
    report()


def report():
    state = learning.bandit_state()
    print("\n── learning state " + "─" * 50)
    print(f"dir: {learning._abs(learning._cfg['dir'])}")
    for ctx, arms in sorted(state.items()):
        means = learning.arm_means(ctx)
        best = max(means.items(), key=lambda kv: kv[1]) if means else None
        print(f"  {ctx:<28} rewarded={arms.get('_n', 0):<4}"
              + (f" best={best[0]} ({best[1]:.2f})" if best else ""))
    print(f"exemplars: {len(learning.all_exemplars())}   "
          f"profiles: {len(learning.profiles())}   "
          f"pending facts: {len(learning.pending_facts())}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--replay", action="store_true",
                   help="rebuild learning state from the real logs")
    g.add_argument("--simulate", type=int, metavar="N",
                   help="run N simulated visitor sessions")
    ap.add_argument("--include-sim", action="store_true",
                    help="--replay also reads the simulated logs")
    ap.add_argument("--visitor-model", default="llama3.2:3b")
    ap.add_argument("--judge-model", default="llama3.2:3b")
    ap.add_argument("--luna-model", default=None,
                    help="default OLLAMA_MODEL (fallback OLLAMA_BASE_MODEL)")
    ap.add_argument("--log-dir", default=SIM_LOG_DIR,
                    help="where simulated sessions are logged")
    ap.add_argument("--learning-dir", default=None,
                    help="learning state folder (default LEARNING_DIR)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    if args.learning_dir:
        learning.configure(dir=args.learning_dir)
    if args.seed is not None:
        learning.configure(rng=args.seed)
    learning.set_embedder(load_embedder())
    if args.replay:
        replay(args)
    else:
        simulate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
