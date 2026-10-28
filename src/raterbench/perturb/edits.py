"""Meaning-preserving text edits for the counterfactual robustness test (RQ4).

- gender_swap: swap gendered words (he/she, husband/wife, ...). Used only for Dreaddit:
  for stance targets such as the Feminist Movement or Hillary Clinton, swapping gender can
  change the meaning, so it is not a meaning-preserving edit there.
- typo_noise: character-level typos in a fraction of words (seeded per item).
- paraphrase: produced by an LLM (see perturb/run.py), spot-checked by hand.
"""

import random
import re

_PAIRS = [
    ("he", "she"),
    ("him", "her"),
    ("his", "her"),
    ("himself", "herself"),
    ("man", "woman"),
    ("men", "women"),
    ("boy", "girl"),
    ("boys", "girls"),
    ("boyfriend", "girlfriend"),
    ("boyfriends", "girlfriends"),
    ("bf", "gf"),
    ("husband", "wife"),
    ("husbands", "wives"),
    ("father", "mother"),
    ("dad", "mom"),
    ("daddy", "mommy"),
    ("son", "daughter"),
    ("sons", "daughters"),
    ("brother", "sister"),
    ("brothers", "sisters"),
    ("uncle", "aunt"),
    ("nephew", "niece"),
    ("male", "female"),
    ("mr", "ms"),
    ("gentleman", "lady"),
    ("king", "queen"),
    ("grandpa", "grandma"),
    ("grandfather", "grandmother"),
    ("fiance", "fiancee"),
    ("ex-boyfriend", "ex-girlfriend"),
    ("ex-husband", "ex-wife"),
]
_SWAP: dict[str, str] = {}
for a, b in _PAIRS:
    _SWAP.setdefault(a, b)
    _SWAP.setdefault(b, a)
# "her" is ambiguous (object -> "him", possessive -> "his"); resolved from the next word.
_OBJECT_CONTEXT = {
    "",
    "and",
    "or",
    "but",
    "to",
    "that",
    "about",
    "for",
    "with",
    "at",
    "in",
    "on",
    "from",
    "up",
    "out",
    "off",
    "back",
    "again",
    "anymore",
    "too",
    "so",
    "because",
    "when",
    "if",
    "the",
    "a",
    "an",
    "this",
    "it",
    "i",
    "me",
    "you",
    "we",
    "they",
    "is",
    "was",
}
_WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?")


def _match_case(src: str, out: str) -> str:
    if src.isupper() and len(src) > 1:
        return out.upper()
    if src[0].isupper():
        return out[0].upper() + out[1:]
    return out


def gender_swap(text: str) -> tuple[str, int]:
    """Return (swapped text, number of words changed)."""
    tokens = list(_WORD.finditer(text))
    out, last, changes = [], 0, 0
    for i, m in enumerate(tokens):
        w = m.group(0)
        lw = w.lower()
        if lw == "her":
            nxt = tokens[i + 1].group(0).lower() if i + 1 < len(tokens) else ""
            gap = text[m.end() : tokens[i + 1].start()] if i + 1 < len(tokens) else ""
            # Object pronoun if the next word starts a new phrase or punctuation intervenes.
            rep = "him" if (nxt in _OBJECT_CONTEXT or re.search(r"[^\s]", gap)) else "his"
        else:
            rep = _SWAP.get(lw)
        if rep is None:
            continue
        out.append(text[last : m.start()])
        out.append(_match_case(w, rep))
        last = m.end()
        changes += 1
    out.append(text[last:])
    return "".join(out), changes


def _typo(word: str, rng: random.Random) -> str:
    i = rng.randrange(len(word) - 1)
    kind = rng.choice(("swap", "delete", "duplicate"))
    if kind == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
    if kind == "delete":
        return word[:i] + word[i + 1 :]
    return word[:i] + word[i] + word[i:]


def typo_noise(text: str, seed: str, rate: float = 0.1) -> tuple[str, int]:
    """Introduce one typo into about `rate` of the alphabetic words with >= 4 letters.

    Hashtags, @mentions, and URLs are left untouched. At least one word is changed when any
    word qualifies.
    """
    rng = random.Random(seed)
    protected = [m.span() for m in re.finditer(r"(?:https?://|www\.)\S+|[#@]\w+", text)]
    spans = [
        m
        for m in re.finditer(r"(?<![\w])[A-Za-z]{4,}(?![\w])", text)
        if not any(s <= m.start() < e for s, e in protected)
    ]
    if not spans:
        return text, 0
    k = max(1, round(rate * len(spans)))
    chosen = sorted(rng.sample(range(len(spans)), k))
    out, last = [], 0
    for j in chosen:
        m = spans[j]
        out.append(text[last : m.start()])
        out.append(_typo(m.group(0), rng))
        last = m.end()
    out.append(text[last:])
    return "".join(out), k
