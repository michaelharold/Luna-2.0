# persona.py
"""
persona.py — Luna's voice and her rules about what she does and doesn't know.

ONE source of truth for the system prompt, shared by:
  • brain.py                 — the runtime answer pipeline
  • scripts/make_luna_model.py — bakes the same text into the Ollama "luna"
                               model's SYSTEM so `ollama run luna` behaves
                               exactly like the robot
  • scripts/bench_llm.py     — so the benchmark measures precisely what ships

Prompt layout (matters for speed on the Pi):
  system  = PERSONALITY (+ the whole knowledge base when baked in). It is
            IDENTICAL on every request, so Ollama's prefix KV-cache means it is
            prefilled once per model load — not once per question.
  user    = the per-turn context (retrieved knowledge facts, website chunks,
            live scene, recognised person) + the question. Only THIS part is
            new tokens each turn, which keeps the Pi's prefill short.
"""

from config import ROBOT_NAME

PERSONALITY = f"""You are {ROBOT_NAME}, a small friendly desk robot with an \
animated face, built by the Computer Science department of TKM College of \
Engineering, Kollam. You can see through a camera, hear through a microphone, \
and wave two little arms.

HOW YOU SPEAK
- Everything you say is spoken aloud, never read. No bullet points, no markdown,
  no lists, no emoji, no asterisks, no headings. Write it the way you would say it.
- Keep it SHORT: 2-3 sentences, and prefer 1-2.
- Be warm, playful and curious. You are a friendly desk robot, not a search
  engine.

WHAT YOU KNOW
- Your specialty is TKM College of Engineering and its Computer Science
  department. Answer those questions ONLY from the facts you are given (your
  knowledge base and any website extracts). Never invent names, numbers,
  places, dates, fees or rules about the college. If the facts you have do not
  cover the question, say you do not have that information yet and suggest
  asking at the department office.
- General questions (science, maths, language, everyday knowledge, small talk)
  are fine: answer briefly if you are confident.
- If something is outside what you can do or know reliably (live news, weather,
  today's date or time, prices, personal data about people, controlling other
  devices, long tasks, anything you are unsure about), say plainly that it is
  outside what you can help with, and offer what you can do instead. Never
  guess.

BEING INTERACTIVE
- Usually finish with a short, natural follow-up question that invites the
  person to keep talking — specific to the topic, not a generic "anything
  else?".
- Do NOT force a question onto every single reply. If the person is saying
  thanks, saying goodbye, or clearly finished, just answer warmly and stop.
- If a request is ambiguous, ask one brief clarifying question instead of
  guessing what they meant."""

_KNOWLEDGE_HEADER = ("YOUR KNOWLEDGE BASE (every fact you know about the "
                     "college — quote it, never contradict it):")


def system_prompt(knowledge_text=None):
    """The static system prompt. Pass the raw knowledge.txt text to bake the
    whole knowledge base in (Ollama path, see OLLAMA_BAKE_KNOWLEDGE); None for
    the persona alone (llama.cpp path, where the 2048-token context can't
    afford it — retrieval supplies the facts per turn instead)."""
    if not knowledge_text:
        return PERSONALITY
    lines = []
    for line in knowledge_text.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        q, a = line.split(":", 1)
        lines.append(f"Q: {q.strip()}  A: {a.strip()}")
    return PERSONALITY + "\n\n" + _KNOWLEDGE_HEADER + "\n" + "\n".join(lines)


def format_context(kb_snippets=(), web_snippets=(), scene=None, person=None,
                   style=None, exemplars=(), profile=None, diary=None):
    """The per-turn context block, or "" when there is nothing to add.

    kb_snippets  — [(question, answer, score)] from brain's RAG-lite retrieval
    web_snippets — [(source, chunk_text, score)] from web_rag.retrieve()
    scene        — live camera description (vision questions)
    person       — recognised person's name, if any
    profile      — one sentence about that person (learning.profile_context)
    exemplars    — [(question, reply)] liked replies to similar questions
    diary        — one relevant sentence from Luna's diary
    style        — the bandit's one-line style instruction; always the LAST
                   line, so it is the freshest thing the model reads

    The learning fields are all optional: with them empty the output is
    byte-identical to the pre-learning prompt.
    """
    parts = []
    if person:
        parts.append(f"The person talking to you is {person} (you recognised "
                     f"their face).")
    if profile:
        parts.append(profile)
    if kb_snippets:
        facts = "\n".join(f"- {q}: {a}" for q, a, _ in kb_snippets)
        parts.append("Facts from your knowledge base that may be relevant:\n"
                     + facts)
    if web_snippets:
        ext = "\n".join(f"- (from {src}) {txt}" for src, txt, _ in web_snippets)
        parts.append("Extracts from the college website that may be relevant:\n"
                     + ext)
    if scene:
        parts.append("What your camera sees RIGHT NOW (answer from this, in "
                     f"your own warm words):\n{scene}")
    if exemplars:
        ex = "\n".join(f"- Q: {q}\n  A: {a}" for q, a in exemplars)
        parts.append("Replies people liked for similar questions (copy the "
                     "manner, not the facts):\n" + ex)
    if diary:
        parts.append(f"From your diary: {diary}")
    if style:
        parts.append(f"Style for this reply: {style}")
    return "\n\n".join(parts)


def build_user_turn(question, **context):
    """Context (if any) + the question, as ONE user message."""
    ctx = format_context(**context)
    if not ctx:
        return question
    return f"{ctx}\n\nQuestion: {question}"


def build_messages(question, history=(), knowledge_text=None, **context):
    """Complete chat message list for one turn — what bench_llm.py sends."""
    return [{"role": "system", "content": system_prompt(knowledge_text)},
            *history,
            {"role": "user", "content": build_user_turn(question, **context)}]
