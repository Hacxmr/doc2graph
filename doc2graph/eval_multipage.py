"""Evaluate cross-page edge construction on DocLayNet.

DocLayNet has no cross-page edge labels, so two checks are defined from what is
annotated:

1. Running-element recall/precision. A ground-truth edge joins two blocks of the
   same document, on different pages, that human annotators gave the same
   category in {Page-header, Page-footer} and whose boxes overlap (IoU >= 0.5 in
   page-normalised coordinates). Ground truth uses category and geometry, never
   the text. Predicted edges are scored only when both endpoints are
   header/footer blocks; other predicted edges are counted as unlabeled.
2. Cross-document control. For pages drawn from two different documents of the
   same category no edge should exist. The rate of predicted edges on such
   pairs, against the same-document pairs, measures false linking.

Page pairs are all unordered pairs of pages inside each multi-page document,
so results are per-pair; confidence intervals resample documents.

Usage: python -m doc2graph.eval_multipage --root <small_dataset dir> --out results.json
"""

import argparse
import itertools
import json
import random
from collections import Counter, defaultdict

from .data.doclaynet import HEADER_FOOTER, load_documents
from .data.edge_reduction import reduce_links
from .data.multipage import build_cross_page_links


def _iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _hf(node):
    return node["category"] in HEADER_FOOTER


def gold_edges(p, q, min_iou=0.5):  # see evaluate(min_iou=...)
    return {
        (i, j)
        for i, a in enumerate(p["nodes"])
        for j, b in enumerate(q["nodes"])
        if _hf(a)
        and a["category"] == b["category"]
        and _iou(a["box_norm"], b["box_norm"]) >= min_iou
    }


def _norm(text):
    return " ".join(text.casefold().split())


def predict(system, p, q, slot_tolerance=(0.05, 0.02)):
    """Return [(i, j, match_type)] linking page p nodes to page q nodes."""
    texts = [[n["text"] for n in p["nodes"]], [n["text"] for n in q["nodes"]]]
    if system == "single_page":  # original Doc2Graph: no edge leaves a page
        return []
    if system == "exact_text":
        by_text = defaultdict(list)
        for j, n in enumerate(q["nodes"]):
            if len(_norm(n["text"])) >= 3:
                by_text[_norm(n["text"])].append(j)
        return [
            (i, j, "exact_text")
            for i, n in enumerate(p["nodes"])
            for j in by_text.get(_norm(n["text"]), [])
            if len(_norm(n["text"])) >= 3
        ]
    boxes = (
        [[n["box_norm"] for n in p["nodes"]], [n["box_norm"] for n in q["nodes"]]]
        if system.startswith("proposed")
        else None
    )
    links = build_cross_page_links(
        texts,
        top_k=3,
        boxes_by_page=boxes,
        link_page_numbers=system == "proposed_page_numbers",
        slot_tolerance=slot_tolerance,
    )
    return [(k["source"]["node"], k["target"]["node"], k["match_type"]) for k in links]


SYSTEMS = ("single_page", "exact_text", "token_matcher", "proposed", "proposed_page_numbers")


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p == p and r == r and p + r else float("nan")
    return p, r, f


def _macro(rows):
    """Mean of per-document precision, recall and F1 (each document weighs the same)."""
    out = []
    for k in range(3):
        vals = [_prf(*r[:3])[k] for r in rows]
        if k == 2:  # a document with gold edges and no prediction scores F1 = 0
            vals = [0.0 if v != v and r[0] + r[2] > 0 else v for v, r in zip(vals, rows)]
        vals = [v for v in vals if v == v]
        out.append(sum(vals) / len(vals) if vals else float("nan"))
    return tuple(out)


def _ci(values):
    values = sorted(v for v in values if v == v)
    if not values:
        return (float("nan"), float("nan"))
    return values[int(0.025 * len(values))], values[min(int(0.975 * len(values)), len(values) - 1)]


def evaluate(docs, n_boot=1000, seed=0, min_iou=0.5):
    rng = random.Random(seed)
    per_doc = {s: {} for s in SYSTEMS}  # system -> doc -> [tp, fp, fn, unlabeled, pairs]
    by_category = {s: defaultdict(lambda: [0, 0, 0]) for s in SYSTEMS}
    same_pairs = []

    for name, pages in docs.items():
        for p, q in itertools.combinations(pages, 2):
            same_pairs.append((name, p, q))
            gold = gold_edges(p, q, min_iou)
            for s in SYSTEMS:
                pred = predict(s, p, q)
                scored = {(i, j) for i, j, _ in pred if _hf(p["nodes"][i]) and _hf(q["nodes"][j])}
                row = per_doc[s].setdefault(name, [0, 0, 0, 0, 0])
                row[0] += len(scored & gold)
                row[1] += len(scored - gold)
                row[2] += len(gold - scored)
                row[3] += len({(i, j) for i, j, _ in pred}) - len(scored)
                row[4] += 1
                cat = by_category[s][p["doc_category"]]
                cat[0] += len(scored & gold)
                cat[1] += len(scored - gold)
                cat[2] += len(gold - scored)

    results = {"pairs": len(same_pairs), "docs": len(docs), "systems": {}}
    results["gold_edges"] = sum(
        len(gold_edges(p, q, min_iou)) for _, p, q in same_pairs
    )
    names = list(docs)
    for s in SYSTEMS:
        tp, fp, fn, unl, npairs = (sum(per_doc[s][n][k] for n in names) for k in range(5))
        p_, r_, f_ = _prf(tp, fp, fn)
        boots = []
        for _ in range(n_boot):
            sample = [per_doc[s][rng.choice(names)] for _ in names]
            boots.append(_macro(sample))
        macro = _macro([per_doc[s][n] for n in names])
        results["systems"][s] = {
            "tp": tp, "fp": fp, "fn": fn,
            "micro_precision": p_, "micro_recall": r_, "micro_f1": f_,
            "precision": macro[0], "recall": macro[1], "f1": macro[2],
            "precision_ci": _ci([b[0] for b in boots]),
            "recall_ci": _ci([b[1] for b in boots]),
            "f1_ci": _ci([b[2] for b in boots]),
            "unlabeled_edges_per_pair": unl / npairs,
            "recall_by_category": {
                c: _prf(*v)[1] for c, v in sorted(by_category[s].items())
            },
        }

    # cross-document control: same number of pairs, same document category
    pool = defaultdict(list)
    for name, pages in docs.items():
        for page in pages:
            pool[page["doc_category"]].append(page)
    negatives = []
    while len(negatives) < len(same_pairs):
        _, p, _ = same_pairs[rng.randrange(len(same_pairs))]
        q = rng.choice(pool[p["doc_category"]])
        if q["name"] != p["name"]:
            negatives.append((p, q))
    results["cross_document_control"] = {}
    for s in SYSTEMS:
        same_rate, neg_rate, neg_hf, kinds = [], [], [], Counter()
        for _, p, q in same_pairs:
            same_rate.append(1 if predict(s, p, q) else 0)
        for p, q in negatives:
            pred = predict(s, p, q)
            neg_rate.append(1 if pred else 0)
            neg_hf.append(
                1 if any(_hf(p["nodes"][i]) and _hf(q["nodes"][j]) for i, j, _ in pred) else 0
            )
            kinds.update(t for _, _, t in pred)
        boots = [
            sum(rng.choice(neg_rate) for _ in neg_rate) / len(neg_rate) for _ in range(n_boot)
        ]
        results["cross_document_control"][s] = {
            "same_doc_pairs_with_edge": sum(same_rate) / len(same_rate),
            "cross_doc_pairs_with_edge": sum(neg_rate) / len(neg_rate),
            "cross_doc_pairs_with_edge_ci": _ci(boots),
            "cross_doc_pairs_with_hf_edge": sum(neg_hf) / len(neg_hf),
            "cross_doc_edge_types": dict(kinds),
        }
    return results


def _fmt(x, ci):
    return f"{x:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


GRID = [(dx, dy) for dx in (0.02, 0.05, 0.10) for dy in (0.01, 0.02, 0.04)]
DEFAULT_TOLERANCE = (0.05, 0.02)


def _doc_counts(pages, system, tolerance, min_iou):
    tp = fp = fn = 0
    for p, q in itertools.combinations(pages, 2):
        gold = gold_edges(p, q, min_iou)
        scored = {
            (i, j)
            for i, j, _ in predict(system, p, q, tolerance)
            if _hf(p["nodes"][i]) and _hf(q["nodes"][j])
        }
        tp, fp, fn = tp + len(scored & gold), fp + len(scored - gold), fn + len(gold - scored)
    return [tp, fp, fn]


def cross_validate(docs, folds=5, seed=0, min_iou=0.5, n_boot=1000, system="proposed_page_numbers"):
    """Choose the slot tolerances on the other folds, score each document once.

    Documents (not pages) are split, so no document is seen while tuning for it.
    """
    rng = random.Random(seed)
    names = sorted(docs)
    rng.shuffle(names)
    counts = {c: {n: _doc_counts(docs[n], system, c, min_iou) for n in names} for c in GRID}

    held, chosen = {}, []
    for f in range(folds):
        test = set(names[f::folds])
        train = [n for n in names if n not in test]
        best = max(GRID, key=lambda c: (_macro([counts[c][n] for n in train])[2], -GRID.index(c)))
        chosen.append(best)
        held.update({n: counts[best][n] for n in test})

    def summarise(rows):
        boots = [_macro([rng.choice(rows) for _ in rows]) for _ in range(n_boot)]
        p, r, f1 = _macro(rows)
        return {
            "precision": p, "recall": r, "f1": f1,
            "precision_ci": _ci([b[0] for b in boots]),
            "recall_ci": _ci([b[1] for b in boots]),
            "f1_ci": _ci([b[2] for b in boots]),
        }

    in_sample = max(GRID, key=lambda c: (_macro(list(counts[c].values()))[2], -GRID.index(c)))
    return {
        "folds": folds,
        "grid": GRID,
        "chosen_per_fold": chosen,
        "hand_set_all_documents": summarise(list(counts[DEFAULT_TOLERANCE].values())),
        "best_on_all_documents": {"tolerance": in_sample, **summarise(list(counts[in_sample].values()))},
        "held_out": summarise([held[n] for n in names]),
    }


def format_cv(cv):
    rows = [
        ("Hand-set tolerances (0.05, 0.02), all documents", cv["hand_set_all_documents"]),
        (f"Best tolerances {tuple(cv['best_on_all_documents']['tolerance'])} chosen on all documents (in-sample)",
         cv["best_on_all_documents"]),
        (f"Tolerances chosen on other folds, {cv['folds']}-fold held-out", cv["held_out"]),
    ]
    lines = ["", "| Protocol | Precision | Recall | F1 |", "|---|---|---|---|"]
    for label, v in rows:
        lines.append(
            f"| {label} | {_fmt(v['precision'], v['precision_ci'])} | {_fmt(v['recall'], v['recall_ci'])} "
            f"| {_fmt(v['f1'], v['f1_ci'])} |"
        )
    lines.append("")
    lines.append("Tolerances chosen per fold: " + ", ".join(str(tuple(c)) for c in cv["chosen_per_fold"]))
    return "\n".join(lines)


def _components(links):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for link in links:
        a = (link["source"]["page"], link["source"]["node"])
        b = (link["target"]["page"], link["target"]["node"])
        parent[find(a)] = find(b)
    groups = {}
    for node in list(parent):
        groups.setdefault(find(node), set()).add(node)
    return {frozenset(g) for g in groups.values()}


def evaluate_reduction(docs):
    """Edge counts before and after reduction, checking no entity is split."""
    totals = {"raw": 0, "unique": 0, "kept": 0, "entities": 0, "documents": 0}
    for pages in docs.values():
        links = build_cross_page_links(
            [[n["text"] for n in p["nodes"]] for p in pages],
            top_k=3,
            boxes_by_page=[[n["box_norm"] for n in p["nodes"]] for p in pages],
            link_page_numbers=True,
        )
        reduced = reduce_links(links)
        assert _components(links) == _components(reduced["links"]), "an entity was split"
        for key in ("raw", "unique", "kept", "entities"):
            totals[key] += reduced["stats"][key]
        totals["documents"] += 1
    return totals


def format_table(r):
    lines = [
        f"{r['docs']} documents, {r['pairs']} same-document page pairs, "
        f"{r['gold_edges']} gold running-element edges",
        "",
        "Macro-averaged over documents; 95% bootstrap CI over documents in brackets.",
        "",
        "| System | Precision | Recall | F1 | Unlabeled edges / pair |",
        "|---|---|---|---|---|",
    ]
    for s, v in r["systems"].items():
        lines.append(
            f"| {s} | {_fmt(v['precision'], v['precision_ci'])} | {_fmt(v['recall'], v['recall_ci'])} "
            f"| {_fmt(v['f1'], v['f1_ci'])} | {v['unlabeled_edges_per_pair']:.2f} |"
        )
    lines += [
        "",
        "| System | Same-doc pairs with an edge | Cross-doc pairs with an edge (control) | Cross-doc pairs with a header/footer edge |",
        "|---|---|---|---|",
    ]
    for s, v in r["cross_document_control"].items():
        ci = v["cross_doc_pairs_with_edge_ci"]
        lines.append(
            f"| {s} | {v['same_doc_pairs_with_edge']:.3f} | "
            f"{v['cross_doc_pairs_with_edge']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{v['cross_doc_pairs_with_hf_edge']:.3f} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, help="DocLayNet-small 'small_dataset' directory")
    parser.add_argument("--out", default=None, help="write the raw results as JSON")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cv", type=int, default=0, help="also run K-fold cross-validation by document")
    parser.add_argument("--min-iou", type=float, default=0.5, help="gold box overlap threshold")
    args = parser.parse_args()

    docs = load_documents(args.root, min_pages=2)
    results = evaluate(docs, n_boot=args.bootstrap, seed=args.seed, min_iou=args.min_iou)
    results["edge_reduction"] = evaluate_reduction(docs)
    if args.cv:
        results["cross_validation"] = cross_validate(
            docs, folds=args.cv, seed=args.seed, min_iou=args.min_iou, n_boot=args.bootstrap
        )
    print(format_table(results))
    if args.cv:
        print(format_cv(results["cross_validation"]))
    red = results["edge_reduction"]
    print(
        f"\nEdge reduction over {red['documents']} documents: {red['raw']} candidate links, "
        f"{red['unique']} unique pairs, {red['kept']} kept "
        f"({100 * (1 - red['kept'] / red['raw']):.1f}% fewer), {red['entities']} entities, "
        "no entity split."
    )
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
