# intents.py
"""
intents.py — plain-text intents Luna decides BEFORE any model is asked.

    date_time(text)            "date" | "time" | "both" | None — answered from
                               the system clock, with a calendar / clock card
    say_date(dt), say_time(dt) how Luna says them
    clock_is_set(dt)           a Pi with no network and no RTC battery boots
                               in 1970 — then she says so instead of guessing
    asks_about_person(text)    "who is X", "tell me about X" — the questions a
                               person card may answer (people.py decides if
                               the person is certain)
    needs_scene_context(text)  questions about who is / was around — the only
                               ordinary questions that get scene memory

Pure functions on text; nothing here imports another Luna module.
"""

import re

_DATE = re.compile(
    r"\bwhat(?:'s|s| is)?\s+(?:the\s+)?today'?s\s+(?:date|day)\b"
    r"|\btoday'?s?\s+(?:date|day)\b"
    r"|\b(?:date|day)\s+(?:is\s+it\s+)?today\b"
    r"|\b(?:what|which)\s+(?:day|date)\s+(?:is\s+(?:it|today)|it\s+is)\b"
    r"|\bwhat(?:'s|s| is)?\s+the\s+(?:date|day)\s*(?:today|now|please|luna)?\s*\??$"
    r"|\bwhat\s+(?:is\s+)?today\s*\??$"
    r"|\b(?:tell|give)\s+me\s+(?:the\s+|today'?s\s+)?(?:date|day)\b"
    r"|\bdate\s+(?:today|now)\b")
_TIME = re.compile(
    r"\bwhat(?:'s|s| is)?\s+the\s+time\b"
    r"|\bwhat\s+time\s+is\s+it\b"
    r"|\btell\s+me\s+the\s+time\b"
    r"|\bcurrent\s+time\b"
    r"|\btime\s+(?:is\s+it\s+)?now\b")


def date_time(text):
    """'date', 'time', 'both' or None for a question about today / the time.
    "what are the dates of the exams" and "the timetable" are not."""
    t = (text or "").lower()
    d, h = bool(_DATE.search(t)), bool(_TIME.search(t))
    return "both" if d and h else "date" if d else "time" if h else None


def _ordinal(n):
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def say_date(dt):
    """'Today is Monday, the 12th of October.'"""
    return f"Today is {dt:%A}, the {_ordinal(dt.day)} of {dt:%B}."


def say_time(dt):
    """'It's 2:07 PM.'"""
    hour = dt.hour % 12 or 12
    return f"It's {hour}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}."


def clock_is_set(dt):
    """False for the epoch-ish dates an unsynchronised Pi boots with."""
    return dt.year >= 2025


_ABOUT_PERSON = re.compile(
    r"\bwho(?:\s+is|'s|s)\b"
    r"|\btell\s+me\s+(?:more\s+)?about\b"
    r"|\bdo\s+you\s+know\b"
    r"|\babout\s+(?:professor|prof|doctor|dr|mr|mrs|ms|miss|sir|madam)\b"
    r"|\b(?:achievements?|qualifications?|designation|research|profile|"
    r"details?|background|role)\s+(?:of|about|on)\b")


def asks_about_person(text):
    """Is this a 'who is / tell me about <someone>' question?"""
    return bool(_ABOUT_PERSON.search((text or "").lower()))


_SCENE = re.compile(
    r"\bwho\s+(?:was|were|is|are)\s+(?:here|around|there|with\s+me|in\s+front)\b"
    r"|\bany(?:one|body)\s+(?:come|came|here|else|around|visit)"
    r"|\bsome(?:one|body)\s+(?:came|come|was\s+here|is\s+here)\b"
    r"|\b(?:came|come)\s+by\b"
    r"|\bvisit(?:ed|or|ors)?\b"
    r"|\bearlier\b"
    r"|\bremember\s+me\b"
    r"|\bmy\s+name\b"
    r"|\bwho\s+am\s+i\b"
    r"|\bdo\s+you\s+know\s+me\b"
    r"|\brecogni[sz]e\s+me\b"
    r"|\b(?:have|did)\s+you\s+see(?:n)?\b"
    r"|\bstill\s+here\b"
    r"|\bbeen\s+here\b")


def needs_scene_context(text):
    """Does answering this need to know who is / was around? Only then does
    scene memory go into an ordinary question's prompt."""
    return bool(_SCENE.search((text or "").lower()))
