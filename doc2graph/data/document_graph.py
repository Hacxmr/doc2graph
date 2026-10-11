"""Document-level graph: one Doc2Graph page graph per page plus cross-page edges."""

import dgl
import torch

from .edge_reduction import reduce_links
from .graph_builder import GraphBuilder
from .multipage import build_cross_page_links
from .relations import CONTINUES, REFERS_TO, SAME_TABLE, SIMILAR_TO, build_typed_edges

INTRA_PAGE, CROSS_PAGE = 0, 1
# edge types a relational GNN (e.g. R-GCN) can use directly
RELATION_ETYPE = {CONTINUES: 2, SAME_TABLE: 3, REFERS_TO: 4, SIMILAR_TO: 5}


def build_document_graph(
    pages,
    k=4,
    top_k=3,
    use_geometry=True,
    link_page_numbers=True,
    reduce=True,
    typed=False,
    embed=None,
    classifier=None,
    min_confidence=0.5,
    relation_options=None,
):
    """Join the pages of one document into a single graph.

    Args:
        pages (list) : ordered pages, each {"size": (w, h), "nodes": [{"text",
            "box" (pixels), "box_norm" (page-normalised)}, ...]}
        k (int) : nearest neighbours per node for intra-page edges (the same
            projection-based kNN Doc2Graph uses on a single page)
        top_k (int) : cross-page candidates kept per node and page pair
        use_geometry (bool) : also link running headers/footers by position
        reduce (bool) : drop duplicate and transitive cross-page links and order
            the rest by page (see edge_reduction); False keeps every candidate
        link_page_numbers (bool) : also link running page numbers; right inside
            one known document, but they carry no document identity
        typed (bool) : also add CONTINUES / SAME_TABLE / REFERS_TO / SIMILAR_TO edges (see
            relations.py); rules alone unless a classifier is given
        embed (callable) : optional ``list[str] -> array``; every block is embedded and its
            nearest neighbours on other pages become candidates (see relations.py)
        classifier (callable) : optional labeller for ambiguous candidates, e.g.
            ``langchain_classifier(model)``; its output is validated
        min_confidence (float) : classifier decisions below this are dropped
        relation_options (dict) : extra keyword arguments for ``build_typed_edges`` (neighbours,
            min_similarity, accept_similarity, max_classifier_calls)

    Returns:
        dict with the DGL graph (ndata ``page``, edata ``etype``/``score``), the
        global node index of every (page, node), the cross-page links used
        (``cross_page_links``), the groups of linked blocks (``entities``) and the
        reduction statistics.
    """
    builder = GraphBuilder()
    offsets, total = [], 0
    src, dst, etype, score = [], [], [], []

    for page in pages:
        offsets.append(total)
        boxes = [list(map(int, n["box"])) for n in page["nodes"]]
        if len(boxes) > 1:
            u, v = builder.knn_connection(page["size"], boxes, k=k)
            src += [total + a for a in u]
            dst += [total + b for b in v]
            etype += [INTRA_PAGE] * len(u)
            score += [1.0] * len(u)
        total += len(boxes)

    raw = build_cross_page_links(
        [[n["text"] for n in p["nodes"]] for p in pages],
        top_k=top_k,
        link_page_numbers=link_page_numbers,
        boxes_by_page=[[n["box_norm"] for n in p["nodes"]] for p in pages]
        if use_geometry
        else None,
    )
    reduced = reduce_links(raw)
    links = reduced["links"] if reduce else raw
    for link in links:
        a = offsets[link["source"]["page"]] + link["source"]["node"]
        b = offsets[link["target"]["page"]] + link["target"]["node"]
        src += [a, b]  # message passing both ways
        dst += [b, a]
        etype += [CROSS_PAGE] * 2
        score += [link["score"]] * 2

    typed_edges = []
    if typed:
        typed_edges = build_typed_edges(
            pages, embed=embed, classifier=classifier, min_confidence=min_confidence,
            **(relation_options or {})
        )
        for edge in typed_edges:
            a = offsets[edge["source"]["page"]] + edge["source"]["node"]
            b = offsets[edge["target"]["page"]] + edge["target"]["node"]
            src += [a, b]
            dst += [b, a]
            etype += [RELATION_ETYPE[edge["relation"]]] * 2
            score += [edge["score"]] * 2

    graph = dgl.graph(
        (torch.tensor(src, dtype=torch.int64), torch.tensor(dst, dtype=torch.int64)),
        num_nodes=total,
    )
    graph.ndata["page"] = torch.tensor(
        [i for i, p in enumerate(pages) for _ in p["nodes"]], dtype=torch.int64
    )
    graph.edata["etype"] = torch.tensor(etype, dtype=torch.int64)
    graph.edata["score"] = torch.tensor(score, dtype=torch.float32)
    return {
        "graph": graph,
        "offsets": offsets,
        "cross_page_links": links,
        "typed_edges": typed_edges,
        "entities": reduced["entities"],
        "reduction": reduced["stats"],
    }
