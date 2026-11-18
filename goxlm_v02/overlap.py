"""Independent lexical overlap checks."""
import re
import unicodedata
from itertools import combinations, product

from .common import ROOT, digest


def terms(text):
    return re.findall(r"\w+|[^\w\s]", unicodedata.normalize("NFKC", text).casefold())


def fingerprint(text):
    return digest(" ".join(terms(text)).encode())


def features(row):
    w = terms(row["text"])
    return row, w, frozenset(zip(w, w[1:], w[2:], w[3:], w[4:]))


def findings(rows, other=None):
    left = [features(row) for row in rows]
    pairs = combinations(left, 2) if other is None else product(
        left, [features(row) for row in other])
    result = []
    for (a, words, grams), (b, other_words, other_grams) in pairs:
        reasons = [field for field in ("family_id", "source_id")
                   if a.get(field) and a[field] == b.get(field)]
        if words and words == other_words:
            reasons.append("normalized_exact")
        shared = len(grams & other_grams)
        if shared >= 6 and (shared / len(grams | other_grams) >= 0.35
                            or shared / min(len(grams), len(other_grams)) >= 0.75):
            reasons.append("shingle_overlap")
        if reasons:
            result.append({"left": a["document_id"], "right": b["document_id"],
                           "left_split": a.get("split"), "right_split": b.get("split"),
                           "reasons": reasons})
    return result


def demo_hits(rows):
    path = ROOT / "data/approved/science_technology_v01.txt"
    phrases = {tuple(terms(line)) for line in path.read_text(encoding="utf-8").splitlines()}
    sizes = {len(phrase) for phrase in phrases if len(phrase) >= 5}
    hits = []
    for row in rows:
        words = terms(row["text"])
        if any(tuple(words[i:i + size]) in phrases
               for size in sizes for i in range(len(words) - size + 1)):
            hits.append(row["document_id"])
    return hits
