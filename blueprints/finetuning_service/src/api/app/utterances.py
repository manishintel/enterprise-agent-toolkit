"""
Mining router utterances out of the dataset a model was fine-tuned on.

A semantic route is defined by example queries. The dataset the model was trained
on is the best available source of them: it is, by construction, the distribution
the model is good at. This turns that dataset into a small, diverse, reviewable
set of utterances.

The pipeline is deliberately staged, and every stage reports what it dropped, so
the result is explainable rather than a black box: "1,204 user turns → 918 after
filtering → 412 unique → 30 selected" is something a reviewer can sanity-check.

Two decisions worth stating outright.

**Redaction is not optional.** Utterances end up in the gateway's database as
*configuration*, readable by anyone with gateway admin access, and they are drawn
from production traces. For a banking corpus that means card numbers, IBANs,
balances and names. Redaction runs before anything is returned, let alone
persisted, so this feature cannot become the path by which trace content leaks
out of the trace store.

**Selection is for coverage, not for volume.** semantic-router matches on nearest
neighbour, so thirty utterances spread across the space beat three hundred
clustered on one phrasing -- and every extra utterance is one more vector to
encode at setup. The selection is greedy farthest-point (k-center) over
embeddings, falling back to a lexical measure when no embedding model is
available.
"""

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .observability import get_logger

logger = get_logger(__name__)

# --- Redaction -------------------------------------------------------------
#
# Ordered: the longer, more specific patterns first, so a card number is not
# half-consumed by the generic digit-run rule.
_REDACTIONS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("[EMAIL]", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("[IBAN]", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")),
    # Separators only *between* digits, so the match cannot swallow the space
    # after the number and glue two words together.
    ("[CARD]", re.compile(r"\b\d(?:[ -]?\d){12,18}\b")),
    ("[IP]", re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")),
    # Plain digit runs first -- account numbers, sort codes, references. Before
    # the phone rule, or a 9-digit account gets labelled [PHONE], which redacts
    # correctly but tells a reviewer the wrong thing about their own data.
    ("[NUMBER]", re.compile(r"\b\d{6,}\b")),
    # Formatted numbers only: an international prefix, or digits broken up by
    # separators. Bare runs are already gone.
    ("[PHONE]", re.compile(r"(?<!\w)(?:\+\d[\d\s().-]{6,}\d|\d{2,4}(?:[\s().-]\d{2,4}){2,})(?!\w)")),
    ("[AMOUNT]", re.compile(r"(?<![\w.])(?:[$£€]\s?\d[\d,]*(?:\.\d+)?)")),
]

# Trailing punctuation and case should not make two identical questions look
# different; without this "...transaction" and "...transaction!" both take a slot.
_DEDUPE_STRIP = re.compile(r"[^a-z0-9 ]+")

_URL = re.compile(r"https?://\S+")
_CODE_FENCE = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_WHITESPACE = re.compile(r"\s+")

# Openers that carry no domain signal on their own. Matched whole, so "thanks for
# explaining how overdrafts work" survives while a bare "thanks" does not.
_PLEASANTRIES = {
    "hi", "hello", "hey", "thanks", "thank you", "thankyou", "ok", "okay", "yes", "no",
    "please", "sure", "got it", "good morning", "good afternoon", "good evening",
    "bye", "goodbye", "cheers", "great", "perfect", "nice", "cool", "test", "testing",
}


def redact(text: str) -> Tuple[str, List[str]]:
    """Replace identifiers and amounts with placeholders. Returns the kinds hit."""
    hits: List[str] = []
    for placeholder, pattern in _REDACTIONS:
        text, count = pattern.subn(placeholder, text)
        if count:
            hits.append(placeholder.strip("[]"))
    return text, hits


def normalise(text: str) -> str:
    """Collapse a turn to a single line of plain prose."""
    text = _CODE_FENCE.sub(" ", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _URL.sub("[URL]", text)
    return _WHITESPACE.sub(" ", text).strip()


def _content_to_text(content: Any) -> str:
    """A message's text, whether it is a string or multimodal content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") in (None, "text")
        ]
        return " ".join(p for p in parts if p)
    return ""


def collect_user_turns(rows: Iterable[Dict[str, Any]], first_turn_only: bool = True) -> List[str]:
    """
    The user's side of each conversation.

    Datasets here are chat-format (``{"messages": [...]}``). The first user turn
    is the query that opened the conversation, which is what a router has to match
    on; later turns are follow-ups that often make no sense out of context ("what
    about the second one?"), so they are excluded by default.
    """
    turns: List[str] = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list):
            # Fall back to the plain-text shapes some datasets use.
            for key in ("prompt", "input", "text", "question"):
                if isinstance(row.get(key), str):
                    turns.append(row[key])
                    break
            continue
        users = [_content_to_text(m.get("content")) for m in messages if m.get("role") == "user"]
        users = [u for u in users if u]
        if not users:
            continue
        turns.extend(users[:1] if first_turn_only else users)
    return turns


def _token_set(text: str) -> frozenset:
    return frozenset(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


# How semantic-router scores a route, verified against the live router: the mean
# of the route's ``top_k`` nearest utterances, not the best match. Two consequences
# that are not obvious and that the UI has to convey:
#
#   * a route with fewer than top_k utterances is scored on *all* of them, so a
#     small set is dragged down by its least relevant member -- 5 utterances gave
#     0.602 on a query where 12 gave 0.621;
#   * beyond top_k, adding diverse utterances does not dilute the score, because
#     only the nearest few ever count. More coverage is free.
ROUTE_TOP_K = 5


def route_score(query_vector: Sequence[float], utterance_vectors: Sequence[Sequence[float]]) -> float:
    """
    Score a query against one route the way the gateway's router will.

    Reproducing the aggregation matters: scoring on the best match instead reads
    much higher (0.857 where the router says 0.601), so a threshold calibrated
    against it would reject queries the UI promised would match.
    """
    if not utterance_vectors:
        return 0.0
    similarities = sorted((_cosine(query_vector, v) for v in utterance_vectors), reverse=True)
    window = similarities[:ROUTE_TOP_K]
    return sum(window) / len(window)


def select_diverse(
    candidates: List[str],
    limit: int,
    vectors: Optional[List[Sequence[float]]] = None,
) -> List[int]:
    """
    Indices of a spread-out subset, by greedy farthest-point selection.

    Starts from the most central candidate -- the one most typical of the dataset
    -- then repeatedly adds whichever remaining candidate is least similar to
    everything chosen so far. That covers the edges of the distribution instead of
    piling up near its middle, which is what a nearest-neighbour router needs.

    Without embeddings it does the same walk over token-set Jaccard similarity.
    Cruder, but it still avoids handing back thirty rephrasings of one question.
    """
    if limit >= len(candidates):
        return list(range(len(candidates)))
    if limit <= 0 or not candidates:
        return []

    if vectors is not None and len(vectors) == len(candidates):
        similarity = lambda i, j: _cosine(vectors[i], vectors[j])  # noqa: E731
    else:
        sets = [_token_set(c) for c in candidates]
        similarity = lambda i, j: _jaccard(sets[i], sets[j])  # noqa: E731

    n = len(candidates)
    # Most central first: highest mean similarity to everything else.
    mean_sim = [sum(similarity(i, j) for j in range(n) if j != i) / (n - 1) for i in range(n)]
    chosen = [max(range(n), key=lambda i: mean_sim[i])]

    while len(chosen) < limit:
        best, best_score = None, None
        for i in range(n):
            if i in chosen:
                continue
            # Distance to the *nearest* already-chosen item is what we maximise.
            worst = max(similarity(i, c) for c in chosen)
            if best_score is None or worst < best_score:
                best, best_score = i, worst
        if best is None:
            break
        chosen.append(best)
    return chosen


def extract(
    rows: List[Dict[str, Any]],
    *,
    limit: int = 30,
    min_words: int = 3,
    max_words: int = 40,
    first_turn_only: bool = True,
    redact_pii: bool = True,
    vectors_for: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Turn dataset rows into a reviewable utterance set, with a report of the funnel.

    ``vectors_for`` is an optional callable taking a list of strings and returning
    their embeddings; when absent, selection falls back to the lexical measure.
    It is a callable rather than a flag so this module stays free of any gateway
    or network dependency and can be tested on its own.
    """
    report: Dict[str, Any] = {
        "rows": len(rows),
        "user_turns": 0,
        "dropped": {"too_short": 0, "too_long": 0, "pleasantry": 0, "empty": 0, "duplicate": 0, "near_duplicate": 0},
        "redacted": {},
        "unique": 0,
        "selected": 0,
        "selection_basis": "lexical",
        "warnings": [],
    }

    turns = collect_user_turns(rows, first_turn_only=first_turn_only)
    report["user_turns"] = len(turns)

    kept: List[str] = []
    seen_exact = set()
    for raw in turns:
        text = normalise(raw)
        if not text:
            report["dropped"]["empty"] += 1
            continue
        if redact_pii:
            text, hits = redact(text)
            for hit in hits:
                report["redacted"][hit] = report["redacted"].get(hit, 0) + 1
        words = text.split()
        if text.lower().strip(" ?!.,") in _PLEASANTRIES:
            report["dropped"]["pleasantry"] += 1
            continue
        if len(words) < min_words:
            report["dropped"]["too_short"] += 1
            continue
        if len(words) > max_words:
            # Long turns are usually pasted context, not a query someone would
            # send at a router. Truncating them would invent an utterance.
            report["dropped"]["too_long"] += 1
            continue
        key = _WHITESPACE.sub(" ", _DEDUPE_STRIP.sub(" ", text.lower())).strip()
        if key in seen_exact:
            report["dropped"]["duplicate"] += 1
            continue
        seen_exact.add(key)
        kept.append(text)

    # Near-duplicates: same words, different order or filler. Keeping both wastes
    # a slot in the final set on a phrasing already covered.
    unique: List[str] = []
    unique_sets: List[frozenset] = []
    duplicates_of: Dict[int, int] = {}
    for text in kept:
        tokens = _token_set(text)
        match = next(
            (i for i, existing in enumerate(unique_sets) if _jaccard(tokens, existing) >= 0.85), None
        )
        if match is not None:
            report["dropped"]["near_duplicate"] += 1
            duplicates_of[match] = duplicates_of.get(match, 0) + 1
            continue
        unique.append(text)
        unique_sets.append(tokens)
    report["unique"] = len(unique)

    if not unique:
        report["warnings"].append(
            "No usable utterances came out of this dataset. Check that it is chat-format "
            "with user turns, and that they are longer than the minimum word count."
        )
        return {"utterances": [], "report": report}

    vectors = None
    if vectors_for is not None and len(unique) > limit:
        try:
            vectors = vectors_for(unique)
            report["selection_basis"] = "embeddings"
        except Exception as exc:  # embedding is an optimisation, not a requirement
            logger.warning(f"Falling back to lexical selection: {type(exc).__name__}: {exc}")
            report["warnings"].append(
                f"Could not embed candidates ({type(exc).__name__}), so selection used word overlap "
                f"instead of meaning. The set is still usable but may be less varied."
            )

    indices = select_diverse(unique, limit, vectors)
    selected = [
        {
            "text": unique[i],
            # How many near-duplicates this one stands in for: a rough measure of
            # how common the phrasing is in the dataset.
            "represents": 1 + duplicates_of.get(i, 0),
        }
        for i in indices
    ]
    report["selected"] = len(selected)

    if len(unique) < ROUTE_TOP_K:
        report["warnings"].append(
            f"Only {len(unique)} distinct utterances were found. The router scores a route on the "
            f"mean of its {ROUTE_TOP_K} nearest utterances, so a set smaller than that is scored on "
            f"all of them and is dragged down by its least relevant member — it will match less "
            f"readily than a larger set. Consider adding examples by hand."
        )

    return {"utterances": selected, "report": report}
