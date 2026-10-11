"""Draw the multi-page document graph of one DocLayNet document.

Top row: each page with translucent, numbered boxes (the text stays readable).
Bottom row: the same blocks as graph nodes at their page position, grey arrows
for reading order (NEXT) and coloured arcs for cross-page links. Output format
follows the file extension (png, pdf, svg). The kNN edges of the graph are
hidden unless --show-knn is given.

Usage:
    python -m doc2graph.visualize_document_graph --root <small_dataset dir> --list
    python -m doc2graph.visualize_document_graph --root <small_dataset dir> \
        --doc MongoDB-manual.pdf --pages 3 --out inference/doclaynet_graph.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch, Rectangle
from PIL import Image

from .data.doclaynet import load_documents
from .data.document_graph import INTRA_PAGE, build_document_graph

CATEGORY_COLORS = {
    "Text": "#4c78a8", "Title": "#e45756", "Section-header": "#f58518",
    "List-item": "#72b7b2", "Table": "#54a24b", "Picture": "#b279a2",
    "Caption": "#d4a017", "Formula": "#9d755d", "Footnote": "#8c8c8c",
    "Page-header": "#c0392b", "Page-footer": "#17a2b8",
}
LINK_COLORS = {
    "same_text": "#1b9e77", "shared_identifier": "#7570b3",
    "repeated_slot": "#e6194b", "shared_terms": "#e6ab02",
}
CHAIN = "#8a94a0"


def draw(pages, out_path, name, k=4, show_knn=False, dpi=220):
    """Two rows per page: the page with translucent boxes, and its graph.

    Top row: page image; every block is a translucent box (so the text stays
    readable) numbered outside its corner. Bottom row: the same blocks as nodes
    at their page position, grey arrows = reading order (NEXT), coloured arcs =
    cross-page links. Page numbers/links never cross the page text.
    """
    result = build_document_graph(pages, k=k)
    graph, links = result["graph"], result["cross_page_links"]
    n = len(pages)
    aspect = pages[0].get("aspect", 1.0)  # DocLayNet squares the page image; undo that
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 2 * 3.2 * aspect + 1.5), squeeze=False,
                             gridspec_kw={"hspace": 0.1, "wspace": 0.1})
    centers, present = {}, set()
    linked = {(k[e]["page"], k[e]["node"]): k["match_type"] for k in links for e in ("source", "target")}

    for i, page in enumerate(pages):
        w, h = page["size"]
        top, bottom = axes[0][i], axes[1][i]
        top.imshow(Image.open(page["image"]).convert("RGB"), extent=(0, w, h, 0))
        for ax in (top, bottom):
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.set_aspect(aspect)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#bbbbbb")
        top.set_title(f"page {page['page_no']}", fontsize=9, pad=3)
        bottom.set_xlabel(f"{len(page['nodes'])} nodes", fontsize=7, labelpad=2)

        prev = None
        for j, node in enumerate(page["nodes"]):
            x0, y0, x1, y1 = node["box"]
            colour = CATEGORY_COLORS.get(node["category"], "#000000")
            present.add(node["category"])
            ring = LINK_COLORS.get(linked.get((i, j)))
            top.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=colour, alpha=0.14, lw=0))
            top.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=colour, lw=0.4))
            if ring:  # a thin ring just outside the box marks nodes with cross-page links
                top.add_patch(Rectangle((x0 - 4, y0 - 4), x1 - x0 + 8, y1 - y0 + 8, fill=False,
                                        edgecolor=ring, lw=0.9))
            top.text(x0 - 4, y0, str(j + 1), ha="right", va="top", fontsize=4.8, color=colour,
                     fontweight="bold", clip_on=False)
            c = ((x0 + x1) / 2, (y0 + y1) / 2)
            centers[(i, j)] = c
            bottom.scatter([c[0]], [c[1]], s=16, color=colour, edgecolor="white", linewidth=0.4, zorder=3)
            bottom.text(c[0] + 12, c[1] - 6, str(j + 1), fontsize=4.2, color="#444444", zorder=4)
            if prev is not None:  # NEXT: reading order
                bottom.annotate("", xy=c, xytext=centers[(i, prev)],
                                arrowprops=dict(arrowstyle="-|>", color=CHAIN, lw=0.6, shrinkA=2.5,
                                                shrinkB=2.5, mutation_scale=5), zorder=2)
            prev = j

    if show_knn:
        where = [(i, j) for i, p in enumerate(pages) for j in range(len(p["nodes"]))]
        for u, v, et in zip(*(t.tolist() for t in graph.edges()), graph.edata["etype"].tolist()):
            if et == INTRA_PAGE:
                (pu, nu), (pv, nv) = where[u], where[v]
                axes[1][pu].plot(*zip(centers[(pu, nu)], centers[(pv, nv)]), color="#cccccc", lw=0.4, zorder=1)

    counts = {}
    for idx, link in enumerate(links):
        a, b = link["source"], link["target"]
        colour = LINK_COLORS.get(link["match_type"], "#000000")
        gap = b["page"] - a["page"]
        con = ConnectionPatch(
            centers[(a["page"], a["node"])], centers[(b["page"], b["node"])], "data", "data",
            axesA=axes[1][a["page"]], axesB=axes[1][b["page"]], arrowstyle="-|>", color=colour,
            lw=1.3, mutation_scale=7, connectionstyle=f"arc3,rad={0.12 * (1 if gap == 1 else 2.2)}",
            zorder=5)
        fig.add_artist(con)
        counts[link["match_type"]] = counts.get(link["match_type"], 0) + 1

    handles = [Patch(facecolor=CATEGORY_COLORS[c], alpha=0.5, edgecolor=CATEGORY_COLORS[c], label=c)
               for c in CATEGORY_COLORS if c in present]
    handles.append(Line2D([], [], color=CHAIN, lw=1, marker=">", markersize=3, label="NEXT (reading order)"))
    handles += [Line2D([], [], color=LINK_COLORS[t], lw=1.6, label=f"cross-page: {t} ({c})")
                for t, c in counts.items()]
    fig.subplots_adjust(top=0.9, bottom=0.1)
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    nexts = sum(max(len(p["nodes"]) - 1, 0) for p in pages)
    fig.suptitle(f"{name}: document graph, {graph.num_nodes()} nodes, {nexts} reading-order edges, "
                 f"{len(links)} cross-page links", fontsize=10, y=0.97)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, help="DocLayNet-small 'small_dataset' directory")
    parser.add_argument("--doc", default=None, help="document name, e.g. MongoDB-manual.pdf")
    parser.add_argument("--pages", type=int, default=4, help="how many pages to draw")
    parser.add_argument("--out", default="inference/doclaynet_graph.png")
    parser.add_argument("--show-knn", action="store_true", help="also draw the kNN edges")
    parser.add_argument("--list", action="store_true", help="list multi-page documents and exit")
    args = parser.parse_args()

    docs = load_documents(args.root, min_pages=2)
    if args.list:
        for name, pages in sorted(docs.items(), key=lambda kv: -len(kv[1])):
            print(f"{len(pages):3d} pages  {pages[0]['doc_category']:22s} {name}")
        return
    name = args.doc or max(docs, key=lambda n: len(docs[n]) if len(docs[n]) <= 12 else 0)
    if name not in docs:
        raise SystemExit(f"Unknown document '{name}'. Use --list to see the options.")
    pages = docs[name][: args.pages]
    draw(pages, args.out, name, show_knn=args.show_knn)
    print(f"Wrote {args.out} ({len(pages)} pages of {name})")


if __name__ == "__main__":
    main()
