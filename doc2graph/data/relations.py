"""Typed cross-page relations: CONTINUES, SAME_TABLE, REFERS_TO, SIMILAR_TO.

Stages (each can be used and evaluated on its own):

1. ``generate_candidates``  with an ``embed`` function every block is embedded once and
   each block's nearest neighbours on the other pages (cosine similarity) become
   candidates, whatever the page distance. Layout rules add structural candidates
   (page-boundary continuation, stacked tables, "Figure 4" references) and accept
   the unambiguous ones outright. Each candidate lists the relations its evidence allows.
2. ``decide``               a candidate is accepted by rule when its evidence is
   unambiguous, sent to an optional ``classifier`` (an LLM) when it is not, and
   without a classifier a semantic neighbour above ``accept_similarity`` becomes a
   ``SIMILAR_TO`` edge; everything else is dropped.
3. ``validate_decision``    deterministic checks on whatever the classifier said:
   known relation, one the rules allowed, confidence in [0, 1] above a threshold.

The classifier never creates a pair of its own: it only labels candidates, so it
cannot invent node ids, pages or relation types. ``embed`` is any callable
``list[str] -> array (n, d)``; ``classifier`` is any callable
``candidate -> {"relation", "confidence", "evidence"}``. LangChain adapters for both
live in ``doc2graph.langchain_models``; this module has no framework dependency. Self-reported LLM confidence is not a calibrated
probability; calibrate it on labelled pairs before reading it as one.
"""

import logging
import re

import numpy as np

log = logging.getLogger(__name__)

CONTINUES, SAME_TABLE, REFERS_TO, SIMILAR_TO, NONE = (
    "CONTINUES", "SAME_TABLE", "REFERS_TO", "SIMILAR_TO", "NONE",
)
RELATIONS = (CONTINUES, SAME_TABLE, REFERS_TO, SIMILAR_TO)

FLOW = ("Text", "List-item")
CAPTIONED = ("Caption", "Table", "Picture")
_REFERENCE = re.compile(r"\b(fig(?:ure)?|table)\.?\s*(\d+(?:\.\d+)*)", re.IGNORECASE)
_SENTENCE_END = (".", "!", "?", ":", ";", '"', "”", "。")


def _label(kind, number):
    return ("figure" if kind.lower().startswith("fig") else "table", number)


def _references(text):
    return {_label(k, n) for k, n in _REFERENCE.findall(text)}


def _caption_label(text):
    match = _REFERENCE.match(text.strip())
    return _label(*match.groups()) if match else None


def _x_overlap(a, b):
    inter = min(a[2], b[2]) - max(a[0], b[0])
    union = max(a[2], b[2]) - min(a[0], b[0])
    return max(inter, 0.0) / union if union > 0 else 0.0


def _cosine(embeddings):
    vectors = np.asarray(embeddings, dtype=float)
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return vectors


def _ref(page, node):
    return {"page": page, "node": node}


def _candidate(pages, i, a, j, b, allowed, rule, evidence):
    na, nb = pages[i]["nodes"][a], pages[j]["nodes"][b]
    return {
        "source": _ref(i, a),
        "target": _ref(j, b),
        "allowed": list(allowed),
        "rule": rule,  # relation the rules accept outright, or None if ambiguous
        "evidence": evidence,
        "source_text": na["text"],
        "target_text": nb["text"],
        "source_type": na.get("category"),
        "target_type": nb.get("category"),
        "similarity": None,
    }


def _continuation(pages, i, tail=2, head=2):
    """Last flow blocks of page i against the first flow blocks of page i + 1."""
    src = [n for n, x in enumerate(pages[i]["nodes"]) if x.get("category") in FLOW and x["text"].strip()]
    dst = [n for n, x in enumerate(pages[i + 1]["nodes"]) if x.get("category") in FLOW and x["text"].strip()]
    found = []
    for a in src[-tail:]:
        for b in dst[:head]:
            na, nb = pages[i]["nodes"][a], pages[i + 1]["nodes"][b]
            if na["box_norm"][3] < 0.7 or nb["box_norm"][1] > 0.3:
                continue  # a continuation runs from the page bottom to the next top
            open_end = not na["text"].rstrip().endswith(_SENTENCE_END)
            lower_start = nb["text"].lstrip()[:1].islower()
            if not (open_end or lower_start):
                continue
            tags = [t for t, ok in (("open_end", open_end), ("lowercase_start", lower_start)) if ok]
            rule = CONTINUES if open_end and lower_start else None
            found.append(_candidate(pages, i, a, i + 1, b, [CONTINUES], rule, "+".join(tags)))
    return found


def _tables(pages, i):
    found = []
    for a, na in enumerate(pages[i]["nodes"]):
        if na.get("category") != "Table" or na["box_norm"][3] < 0.75:
            continue
        for b, nb in enumerate(pages[i + 1]["nodes"]):
            if nb.get("category") != "Table" or nb["box_norm"][1] > 0.25:
                continue
            overlap = _x_overlap(na["box_norm"], nb["box_norm"])
            if overlap < 0.6:
                continue
            rule = SAME_TABLE if overlap >= 0.9 else None
            found.append(_candidate(pages, i, a, i + 1, b, [SAME_TABLE], rule, f"x_overlap={overlap:.2f}"))
    return found


def _reference_candidates(pages):
    """Flow text that names 'Figure 4' / 'Table 2' against the caption that carries it."""
    captions = {}
    for p, page in enumerate(pages):
        for n, node in enumerate(page["nodes"]):
            if node.get("category") in CAPTIONED:
                label = _caption_label(node["text"])
                if label:
                    captions.setdefault(label, []).append((p, n))
    found = []
    for p, page in enumerate(pages):
        for n, node in enumerate(page["nodes"]):
            if node.get("category") not in FLOW:
                continue
            for label in sorted(_references(node["text"])):
                for q, m in captions.get(label, []):
                    if q != p:  # same-page references are already near each other
                        found.append(_candidate(pages, p, n, q, m, [REFERS_TO], REFERS_TO, f"{label[0]} {label[1]}"))
    return found


def _embed_blocks(pages, embed, max_chars=512):
    """One embedding per non-trivial block: ``(unit vectors, [(page, node)])``."""
    keys, texts = [], []
    for p, page in enumerate(pages):
        for n, node in enumerate(page["nodes"]):
            text = node["text"].strip()
            if len(text) >= 3:
                keys.append((p, n))
                texts.append(text[:max_chars])
    return _cosine(embed(texts)), keys


def _semantic_allowed(pages, i, a, j, b):
    cats = pages[i]["nodes"][a].get("category"), pages[j]["nodes"][b].get("category")
    allowed = []
    if abs(j - i) == 1 and all(c in FLOW for c in cats):
        allowed.append(CONTINUES)
    if all(c == "Table" for c in cats):
        allowed.append(SAME_TABLE)
    if (cats[0] in FLOW and cats[1] in CAPTIONED) or (cats[1] in FLOW and cats[0] in CAPTIONED):
        allowed.append(REFERS_TO)
    return allowed + [SIMILAR_TO]


def _semantic_candidates(pages, vecs, keys, neighbours, min_similarity, seen):
    """Each block's nearest blocks on *other* pages, by cosine similarity."""
    page_of = np.array([k[0] for k in keys])
    sims = vecs @ vecs.T
    pairs = {}
    for r, (i, a) in enumerate(keys):
        row = np.where(page_of != i, sims[r], -1.0)
        for c in np.argsort(-row)[:neighbours]:
            if row[c] < min_similarity:
                break
            (i1, a1), (j1, b1) = sorted([keys[r], keys[c]])  # earlier page first
            pairs[(i1, a1, j1, b1)] = float(sims[r, c])
    out = []
    for (i, a, j, b), sim in sorted(pairs.items(), key=lambda kv: -kv[1]):
        if ((i, a), (j, b)) in seen:
            continue
        c = _candidate(pages, i, a, j, b, _semantic_allowed(pages, i, a, j, b), None, f"cosine={sim:.2f}")
        c["similarity"] = sim
        out.append(c)
    return out


def generate_candidates(pages, embed=None, top_k=3, neighbours=3, min_similarity=0.5):
    """Propose typed block pairs.

    Args:
        pages (list) : ordered pages as in ``build_document_graph``
        embed (callable) : optional ``list[str] -> array``; embeds every block, fills
            ``similarity`` on all candidates and adds nearest-neighbour candidates
            across all pages
        top_k (int) : structural continuation candidates kept per page boundary
        neighbours (int) : nearest blocks on other pages kept per block
        min_similarity (float) : cosine floor for a neighbour candidate
    """
    out = []
    for i in range(len(pages) - 1):
        cont = _continuation(pages, i)
        # A rule-accepted pair outranks an ambiguous one; similarity breaks ties below.
        out += cont[:top_k] if embed is None else cont
        out += _tables(pages, i)
    out += _reference_candidates(pages)
    if embed is None:
        return out

    vecs, keys = _embed_blocks(pages, embed)
    row = {k: r for r, k in enumerate(keys)}
    for c in out:
        a, b = (c["source"]["page"], c["source"]["node"]), (c["target"]["page"], c["target"]["node"])
        if a in row and b in row:
            c["similarity"] = float(vecs[row[a]] @ vecs[row[b]])
    structural = [c for c in out if c["rule"] is not None or c["allowed"] != [CONTINUES]]
    continuation = sorted(
        (c for c in out if c["rule"] is None and c["allowed"] == [CONTINUES]),
        key=lambda c: -(c["similarity"] or 0.0),
    )
    out = structural + continuation[: top_k * max(len(pages) - 1, 1)]
    seen = {((c["source"]["page"], c["source"]["node"]), (c["target"]["page"], c["target"]["node"])) for c in out}
    return out + _semantic_candidates(pages, vecs, keys, neighbours, min_similarity, seen)


def validate_decision(decision, candidate, min_confidence=0.5):
    """Return a clean ``{"relation", "confidence", "evidence"}`` or None to reject."""
    try:
        relation = str(decision["relation"]).upper()
        confidence = float(decision["confidence"])
    except (KeyError, TypeError, ValueError):
        return None
    if relation not in candidate["allowed"] or not 0.0 <= confidence <= 1.0:
        return None  # NONE, unknown, or a relation the evidence does not allow
    if confidence < min_confidence:
        return None
    return {"relation": relation, "confidence": confidence, "evidence": str(decision.get("evidence", ""))}


def decide(candidates, classifier=None, min_confidence=0.5, accept_similarity=0.8, max_classifier_calls=200):
    """Turn candidates into typed edges.

    Order of evidence: a rule accepts outright; otherwise the classifier labels the
    pair (at most ``max_classifier_calls`` pairs, most similar first); without a
    classifier a semantic neighbour with similarity >= ``accept_similarity`` becomes
    ``SIMILAR_TO``. ``accept_similarity`` depends on the embedding model: calibrate it.
    """
    edges, calls = [], 0
    for c in sorted(candidates, key=lambda c: (c["rule"] is None, -(c["similarity"] or 0.0))):
        if c["rule"] is not None:
            relation, score, origin, note = c["rule"], 1.0, "rule", c["evidence"]
        elif classifier is not None:
            if calls >= max_classifier_calls:
                continue
            calls += 1
            try:
                answer = classifier(c)
            except Exception as exc:  # a failed LLM call leaves the pair undecided
                log.warning("classifier failed for %s -> %s: %s", c["source"], c["target"], exc)
                continue
            clean = validate_decision(answer, c, min_confidence)
            if clean is None:
                continue
            relation, score, origin, note = clean["relation"], clean["confidence"], "llm", clean["evidence"]
        elif SIMILAR_TO in c["allowed"] and (c["similarity"] or 0.0) >= accept_similarity:
            relation, score, origin, note = SIMILAR_TO, c["similarity"], "embedding", c["evidence"]
        else:
            continue
        edges.append(
            {
                "source": c["source"],
                "target": c["target"],
                "relation": relation,
                "score": score,
                "origin": origin,
                "evidence": note,
                "similarity": c["similarity"],
            }
        )
    return edges


def build_typed_edges(
    pages, embed=None, classifier=None, top_k=3, min_confidence=0.5,
    neighbours=3, min_similarity=0.5, accept_similarity=0.8, max_classifier_calls=200,
):
    """Candidates -> decisions in one call. Without ``embed`` and ``classifier`` only rule edges remain."""
    candidates = generate_candidates(pages, embed, top_k, neighbours, min_similarity)
    return decide(candidates, classifier, min_confidence, accept_similarity, max_classifier_calls)
