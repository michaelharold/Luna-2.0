# people.py
"""
people.py — the people Luna can describe with CERTAINTY.

A person card (face_cards.person_card: the name in big rounded bold, the role
under it, one highlight line) goes on screen only when Luna is sure WHO the
question is about and WHAT to show, and every word on it comes from stored
facts — never from the language model. This module is that certainty:

  * the knowledge base's own person entries — the department's curated facts
      "Who is Dr. Dimple A Shajahan?: Dr. Dimple A Shajahan is a Professor in
       the Department of Computer Science. She obtained her PhD from IIT Madras."
      -> Dr. Dimple A Shajahan / Professor · Computer Science / PhD from IIT Madras
  * role entries whose answer names one person
      "who is the principal: Dr. Sadiq A. is the principal of ..."
  * the college website's staff lists (data/site/catalog.json, written by
    site_index.py), for anyone the knowledge base leaves out
      "Dr. MANU J PILLAI, Associate Professor, COMPUTER SCIENCE AND ENGINEERING"

Directory.match_question() names a person only when the question asks about
someone ("who is", "tell me about"…) AND its name words point at exactly ONE
person — "who is mohammed" with two Mohammeds on the staff gets no card.

No pygame, no models: pure text, so it is unit-testable anywhere.
"""

import os
import re
from dataclasses import dataclass

import intents

TITLES = {"dr", "prof", "professor", "mr", "mrs", "ms", "miss", "smt", "sri",
          "shri", "sir"}
_TITLE_FORM = {"dr": "Dr.", "prof": "Prof.", "professor": "Prof.",
               "mr": "Mr.", "mrs": "Mrs.", "ms": "Ms.", "miss": "Miss",
               "smt": "Smt.", "sri": "Sri", "shri": "Shri"}
_WORD = re.compile(r"[A-Za-z]+")
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_ABBREV_END = re.compile(r"(?:^|\s)(?:Dr|Prof|Mr|Mrs|Ms|Smt|St|No|[A-Z])\.$")
_SMALL = {"and", "of", "the", "in", "for", "with"}


def closest_name(heard, names, cutoff=0.8):
    """(the known spelling, exact?) for a name speech recognition heard —
    "navia" against the college's students / staff / registered faces ->
    ("Navya", False) — or (None, False). A one-word name is compared with
    first names, a two-word one with first + second words, and so on."""
    import difflib
    h = (heard or "").lower().split()
    if not h:
        return None, False
    cands = {}
    for n in names:
        w = [x for x in (n or "").split()
             if x.lower().strip(".") not in TITLES and len(x.strip(".")) > 0]
        if len(w) < len(h):
            continue
        spelled = " ".join(w[:len(h)])
        cands.setdefault(spelled.lower(), spelled)
    key = " ".join(h)
    if key in cands:
        return cands[key], True
    m = difflib.get_close_matches(key, list(cands), n=1, cutoff=cutoff)
    return (cands[m[0]], False) if m else (None, False)


def name_tokens(name):
    """The words that identify a person: 3+ letters, no titles or initials."""
    return {w.lower() for w in _WORD.findall(name or "")
            if len(w) >= 3 and w.lower() not in TITLES}


def display_name(raw):
    """'Dr. MANU J PILLAI' / 'dr Aneesh g nath' -> 'Dr. Manu J Pillai'."""
    out = []
    for i, word in enumerate((raw or "").replace(",", " ").split()):
        core, dot = word.rstrip("."), "." if word.endswith(".") else ""
        if i == 0 and core.lower() in _TITLE_FORM:
            out.append(_TITLE_FORM[core.lower()])
            continue
        if len(core) == 1:
            out.append(core.upper() + dot)
            continue
        if core.isupper() or core.islower():
            core = core.capitalize()
        out.append(core + dot)
    return " ".join(out)


def _sentences(text):
    """Split without breaking 'Dr. Dimple' or 'Aneesh G. Nath'."""
    out = []
    for piece in _SPLIT.split((text or "").strip()):
        if out and _ABBREV_END.search(out[-1]):
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return [p.strip() for p in out if p.strip()]


def _cap(text):
    text = " ".join((text or "").split()).strip(" .")
    return text[:1].upper() + text[1:] if text else ""


def _tidy_dept(dept):
    """'COMPUTER SCIENCE AND ENGG' / 'Computer Science & Engineering' ->
    'Computer Science and Engineering'."""
    d = " ".join((dept or "").replace("&", " and ").split())
    d = re.sub(r"(?i)\bengg\b\.?", "Engineering", d)
    words = d.split()
    return " ".join(w.lower() if (i and w.lower() in _SMALL) else
                    (w.capitalize() if (w.isupper() or w.islower()) else w)
                    for i, w in enumerate(words))


def _tidy_designation(text):
    d = " ".join((text or "").split())
    d = re.sub(r"(?i)\basst\b\.?\s*", "Assistant ", d)    # not inside a word:
    d = re.sub(r"(?i)\bassoc\b\.?\s*", "Associate ", d)   # "Associate" stays
    d = re.sub(r"(?i)\s+adhoc\b", " (ad hoc)", d)
    return _cap(d)


def _tidy_role(raw):
    """'a Professor in the Department of Computer Science' -> 'Professor ·
    Computer Science'; 'head of the Department of Computer Science' -> 'Head
    of Computer Science'."""
    r = " ".join((raw or "").split()).strip(" .")
    r = re.sub(r"(?i)^(?:a|an|the)\s+", "", r)
    r = re.sub(r"(?i)\s+in\s+the\s+department\s+of\s+(.+)$", r" · \1", r)
    r = re.sub(r"(?i)\bof\s+the\s+department\s+of\s+", "of ", r)
    return _cap(r)


_ACRONYMS = {"ai": "AI", "cse": "CSE", "tkm": "TKM", "ece": "ECE",
             "eee": "EEE", "iqac": "IQAC", "it": "IT", "hod": "HOD"}


def _title_role(role):
    """A role typed all in lower case — 'Head of computer science department'
    — in title case, small words and acronyms kept right."""
    if not role or role[1:] != role[1:].lower():
        return role
    words = role.split()
    return " ".join(_ACRONYMS.get(w.lower(), w if (i and w.lower() in _SMALL)
                                  else w.capitalize())
                    for i, w in enumerate(words))


def _tidy_detail(sentence):
    """'She obtained her PhD from IIT Madras.' -> 'PhD from IIT Madras'."""
    s = " ".join((sentence or "").split()).strip(" .")
    s = re.sub(r"(?i)^(?:he|she|they)\s+(?:obtained|received|earned|completed|"
               r"got|holds|has|did)\s+(?:his|her|their|a|an)\s+", "", s)
    s = re.sub(r"(?i)^(?:his|her|their)\s+research\s+(?:focuses|is\s+focused|"
               r"is)\s+on\s+", "Research: ", s)
    return _cap(s)


@dataclass(frozen=True)
class Person:
    name: str
    role: str = ""
    detail: str = ""
    source: str = ""        # "kb" | "web"

    @property
    def tokens(self):
        return name_tokens(self.name)


def _from_kb(subject, answer):
    sents = _sentences(answer)
    role = ""
    if sents:
        m = re.search(r"(?i)\bis\s+(.+)$", sents[0])
        role = _tidy_role(m.group(1)) if m else ""
    detail = _tidy_detail(sents[1]) if len(sents) > 1 else ""
    return Person(display_name(subject), role, detail, "kb")


def _from_staff(title, text):
    """People from one website staff section: 'Regular teaching staff, 13 in
    total: Dr. A, Professor, Dept; Prof. B, Asst. Professor; ...'."""
    dept = _tidy_dept(title.split("—")[0]) if "—" in (title or "") else ""
    head = re.match(r"^[^:;]*\bin total:", text or "")
    body = text[head.end():] if head else (text or "")
    items = body.split(";")
    # The website index cuts long lists into overlapping chunks, so a chunk
    # may start or end in the middle of an entry ("AMINA M, A"). Those edge
    # entries are dropped — the overlap means they appear whole elsewhere.
    if not head and items:
        items = items[1:]
    if items and not (text or "").rstrip().endswith((".", ";")) and \
            len(items[-1].split(",")) < 2:
        items = items[:-1]
    out = []
    for item in items:
        parts = [p.strip() for p in item.strip().rstrip(".").split(",") if p.strip()]
        if not parts or not name_tokens(parts[0]):
            continue
        role = _tidy_designation(parts[1]) if len(parts) > 1 else ""
        where = _tidy_dept(parts[2]) if len(parts) > 2 else dept
        if role and where:
            role = f"{role} · {where}"
        out.append(Person(display_name(parts[0]), role or where, "", "web"))
    return out


def _norm(question):
    return " ".join(re.findall(r"[a-z0-9]+", (question or "").lower()))


class Directory:
    def __init__(self, persons=(), roles=None):
        self.people = list(persons)
        self.roles = dict(roles or {})       # normalised KB question -> Person
        self._known = set().union(*(p.tokens for p in self.people)) \
            if self.people else set()

    @classmethod
    def build(cls, kb_lines, staff=()):
        """From knowledge-base lines and website staff sections
        [(section title, text)]. The knowledge base wins over the website for
        the same person (it is the department's own wording)."""
        persons, role_lines = [], []
        for line in kb_lines or ():
            if ":" not in line:
                continue
            q, a = (x.strip() for x in line.split(":", 1))
            m = re.match(r"(?i)^who\s+is\s+(.+?)\s*\??$", q)
            if not m:
                continue
            if m.group(1).lower().startswith("the "):
                role_lines.append((q, a))
            elif name_tokens(m.group(1)):
                persons.append(_from_kb(m.group(1), a))
        for title, text in staff or ():
            for p in _from_staff(title, text):
                if not any(p.tokens <= q.tokens or q.tokens <= p.tokens
                           for q in persons):
                    persons.append(p)
        d = cls(persons)
        for q, a in role_lines:
            p = d._role_person(a)
            if p is not None:
                d.roles[_norm(q)] = p
        return d

    def _role_person(self, answer):
        """'Dr. Sadiq A. is the principal of …' -> that person, in the
        canonical form when they also have an entry of their own."""
        m = re.match(r"(?i)^\s*((?:dr|prof|professor|mr|mrs|ms)\.?\s+.+?)\s+is\s+"
                     r"(?:the|a|an)\s+(.+?)\s*\.?\s*$", answer or "")
        if not m:
            return None
        role = _title_role(_tidy_role(m.group(2)))
        known = self.lookup(m.group(1))
        if known is not None:
            return Person(known.name, role, known.detail, known.source)
        return Person(display_name(m.group(1)), role, "", "kb")

    def lookup(self, name):
        """The one person whose name contains all of `name`'s words."""
        toks = name_tokens(name)
        if not toks:
            return None
        hits = [p for p in self.people if toks <= p.tokens]
        return hits[0] if len(hits) == 1 else None

    def match_question(self, text):
        """The person a 'who is / tell me about' question names — only when
        its name words point at exactly one person AND name them well enough
        (named()): "who is mohammed" is not certain even when there is just
        one Mohammed Luna knows of."""
        if not intents.asks_about_person(text):
            return None
        asked = {w.lower() for w in _WORD.findall(text or "")} & self._known
        if not asked:
            return None
        hits = [p for p in self.people if asked <= p.tokens]
        if len(hits) == 1 and named(text, hits[0]):
            return hits[0]
        return None

    def for_kb_question(self, question, asked=None):
        """The person a knowledge-base entry is about: a role entry ('who is
        the principal' — certain by itself) or a person entry ('Who is Dr.
        X?'), which needs the `asked` question to name them too (named())."""
        p = self.roles.get(_norm(question))
        if p is not None:
            return p
        m = re.match(r"(?i)^who\s+is\s+(.+?)\s*\??$", (question or "").strip())
        if m and not m.group(1).lower().startswith("the "):
            p = self.lookup(m.group(1))
            if p is not None and (asked is None or named(asked, p)):
                return p
        return None


def named(text, person):
    """Does `text` name `person` well enough to be certain? Their whole
    name ("thushara" for Dr. Thushara A), or at least two of its words
    ("reshma sheik") — never half a name ("reshma", "mohammed")."""
    said = {w.lower() for w in _WORD.findall(text or "")} & person.tokens
    return bool(said) and (said == person.tokens or len(said) >= 2)


_cache = {"key": None, "dir": None}


def _kb_path():
    try:
        from config import KNOWLEDGE_PATH
    except Exception:                                       # noqa: BLE001
        KNOWLEDGE_PATH = "data/knowledge.txt"
    if os.path.isabs(KNOWLEDGE_PATH):
        return KNOWLEDGE_PATH
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        KNOWLEDGE_PATH)


def directory(staff=None):
    """The Directory from the knowledge base (+ website staff sections when
    given), rebuilt only when the knowledge file or the staff data change."""
    path = _kb_path()
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None
    staff = tuple(staff or ())
    key = (path, mtime, hash(staff))
    if _cache["key"] != key:
        lines = []
        if mtime is not None:
            with open(path, encoding="utf-8") as f:
                lines = f.read().splitlines()
        _cache["dir"] = Directory.build(lines, staff)
        _cache["key"] = key
    return _cache["dir"]
