# wiki_rag.py
"""
wiki_rag.py — live Wikipedia lookup for genuine general-knowledge questions.

web_rag.py answers from ONE configured, trusted site (the college's own pages)
and is safe to trust broadly. This module is the opposite shape: it can be
asked about ANYTHING, which makes it useful for real general knowledge
("what is quantum entanglement?") but dangerous for anything that merely
SOUNDS like a campus question — Wikipedia's keyword search returns *something*
for almost any query, including a made-up college or department, and a
confidently-worded real article about the wrong institution is exactly the
kind of invented-sounding answer persona.py exists to prevent (see
docs/UPDATES.txt or the git history around "P.K.M. college" for why this
matters in practice).

So brain.py must only call this when the knowledge base found NOTHING
relevant (empty kb_snippets) — that already rules out most campus-flavored
misses, because even a loose KB match on shared words like "college" or
"department" keeps a question on the KB/LLM path instead of reaching here.
As a second, independent guard, retrieve() itself refuses to look anything up
that mentions campus-domain vocabulary (see _DOMAIN_ADJACENT_TERMS) — belt and
suspenders, since one guard being wrong should not be enough to cause a false
answer.

No index, no cache, no embedder: each call is one live search + one summary
fetch (both official, free, no-key Wikipedia REST endpoints), so this is only
useful when the device happens to be online — same spirit as web_rag, but
without anything to fall back to offline. It only tries when connectivity.py
already knows the device is online, so an offline or half-up Pi pays nothing
here, not a timeout per question. Everything degrades to [] instead of
raising; a broken network must never break a turn.

    python wiki_rag.py --query "what is the capital of france"
    python wiki_rag.py --query "tell me about the chemistry department"  # []
"""

import re
import urllib.error
import urllib.parse
import urllib.request

import connectivity

try:
    from config import WIKI_RAG_ENABLED, WIKI_RAG_TIMEOUT, WIKI_RAG_LANG
except Exception:                       # standalone use without config.py
    WIKI_RAG_ENABLED, WIKI_RAG_TIMEOUT, WIKI_RAG_LANG = True, 5.0, "en"

# Wikipedia's API rejects requests with no descriptive User-Agent (403).
_HEADERS = {"User-Agent": "LunaBot/1.0 (+offline desk robot; general-knowledge fallback)"}


def _api_base():
    return f"https://{WIKI_RAG_LANG}.wikipedia.org"


# Wikipedia's search index ranks on keywords, not natural-language questions —
# leaving interrogative phrasing in measurably hurts relevance (a full-sentence
# "What is the capital of France?" has matched the unrelated article "Closed-
# ended question" ahead of "France" in testing). Stripping the question
# wrapper down to its keywords first fixes that.
_QUESTION_PREFIX_RE = re.compile(
    r"^(what('?s| is| are)|who('?s| is| are)|where('?s| is| are)|when('?s| is| are)|"
    r"why('?s| is| are)|how('?s| is| are)|tell me about|can you tell me)\s+(the\s+)?",
    re.IGNORECASE,
)

# If the question itself mentions campus-domain vocabulary, treat it as almost
# certainly about THIS campus, not real-world general knowledge — a real
# Wikipedia article that happens to share those words (a different college, a
# different department) is a false-confidence risk, not a genuine answer. This
# is deliberately broad; it only needs to rule OUT invoking a live web search,
# never to rule IN an answer.
_DOMAIN_ADJACENT_TERMS = re.compile(
    r"\b(department|college|campus|hod|advisor|professor|faculty|principal|"
    r"hostel|club|fest|placement|lab|laboratory|canteen|library|hall|office|"
    r"dean|lecturer|staff|luna)\b",
    re.IGNORECASE,
)


def _to_search_keywords(query):
    stripped = _QUESTION_PREFIX_RE.sub("", query).strip()
    return stripped.rstrip("?").strip() or query


def _get_json(url, params, timeout):
    full = url + "?" + urllib.parse.urlencode(params) if params else url
    req = urllib.request.Request(full, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        import json
        return json.loads(resp.read().decode("utf-8"))


def search_title(query, timeout=None):
    """The top matching article title for `query`'s keywords, or None. Split
    out from retrieve() so it's independently testable/mockable."""
    body = _get_json(f"{_api_base()}/w/api.php", {
        "action": "query", "list": "search", "srsearch": _to_search_keywords(query),
        "format": "json", "srlimit": 1,
    }, timeout or WIKI_RAG_TIMEOUT)
    hits = body.get("query", {}).get("search", [])
    return hits[0]["title"] if hits else None


def summary_for(title, timeout=None):
    """(text, url) for an article title's summary, or (None, None). Keeps only
    the first two sentences — this becomes spoken/prompt text, and a full
    Wikipedia lead paragraph is too long for either."""
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    body = _get_json(f"{_api_base()}/api/rest_v1/page/summary/{encoded}", {},
                     timeout or WIKI_RAG_TIMEOUT)
    extract = (body.get("extract") or "").strip()
    if not extract:
        return None, None
    sentences = re.split(r"(?<=[.!?])\s+", extract)
    text = " ".join(sentences[:2])
    url = (body.get("content_urls", {}).get("desktop", {}) or {}).get("page")
    return text, url


def retrieve(text, top_k=1):
    """[(source_url, chunk_text, score)] — at most one entry (there is no
    ranked index here, just "the" top article), [] when disabled, when the
    question looks campus-domain-specific, or on any failure (offline, no
    match, API error). score is always 1.0: there is nothing to rank against,
    it exists only so this return shape matches web_rag.retrieve()'s and both
    can be dropped into persona.format_context's wiki_snippets/web_snippets
    interchangeably."""
    if not WIKI_RAG_ENABLED or top_k <= 0 or not text or not text.strip():
        return []
    if _DOMAIN_ADJACENT_TERMS.search(text):
        return []
    if connectivity.state() is not True:
        return []
    try:
        title = search_title(text)
        if not title:
            return []
        chunk, url = summary_for(title)
        if not chunk:
            return []
        return [(url or title, chunk, 1.0)]
    except urllib.error.HTTPError as e:      # Wikipedia answered, just not OK
        print(f"[wiki_rag] lookup failed: HTTP {e.code}")
        return []
    except (urllib.error.URLError, OSError) as e:
        connectivity.mark_offline()
        print(f"[wiki_rag] lookup failed ({e}) — offline until the next probe")
        return []
    except (ValueError, KeyError) as e:
        print(f"[wiki_rag] unexpected response: {e}")
        return []


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Luna live Wikipedia fallback (wiki_rag)")
    ap.add_argument("--query", action="append", default=[], help="question (repeatable)")
    args = ap.parse_args()

    for q in args.query:
        hits = retrieve(q)
        print(f"\n[wiki_rag] query: {q!r} -> {len(hits)} hit(s)")
        for src, txt, score in hits:
            print(f"  {score:.2f}  {src}\n         {txt}")
