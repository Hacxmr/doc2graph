"""Load DocLayNet pages and regroup them into documents.

DocLayNet annotates layout blocks page by page. Pages of one PDF share
``original_filename``, so documents are rebuilt by grouping on it and ordering
by ``page_no``. Each block becomes a node with its category, text and box.
"""

import json
from collections import defaultdict
from pathlib import Path

HEADER_FOOTER = ("Page-header", "Page-footer")


def _page(path, root, split):
    ann = json.loads(path.read_text())
    meta = ann["metadata"]
    width, height = meta["coco_width"], meta["coco_height"]

    blocks = defaultdict(list)
    for entry in ann["form"]:
        blocks[entry["id_box"]].append(entry)

    nodes = []
    for entries in blocks.values():
        entries.sort(key=lambda e: e["id_box_line"])
        # [x, y, w, h] in the pixels of the (coco_width x coco_height) page image
        x, y, w, h = entries[0]["box"]
        box = [
            max(int(x), 0),
            max(int(y), 0),
            min(int(x + w), width),
            min(int(y + h), height),
        ]
        nodes.append(
            {
                "text": " ".join(e["text"].strip() for e in entries).strip(),
                "category": entries[0]["category"],
                "box": box,
                "box_norm": [box[0] / width, box[1] / height, box[2] / width, box[3] / height],
            }
        )
    nodes.sort(key=lambda n: (n["box"][1], n["box"][0]))  # reading-order proxy
    return {
        "page_no": meta["page_no"],
        "num_pages": meta["num_pages"],
        "size": (meta["coco_width"], meta["coco_height"]),
        "aspect": meta["original_height"] / meta["original_width"],  # true page proportions
        "image": str(root / split / "images" / f"{meta['page_hash']}.png"),
        "nodes": nodes,
        "doc_category": meta["doc_category"],
        "split": split,
        "name": meta["original_filename"],
    }


def load_documents(root, min_pages=1, splits=("train", "val", "test")):
    """Return {document name: [page, ...]} ordered by page number."""
    root = Path(root)
    docs = defaultdict(list)
    for split in splits:
        for path in sorted((root / split / "annotations").glob("*.json")):
            if path.name.startswith("._"):  # macOS AppleDouble files
                continue
            page = _page(path, root, split)
            docs[page["name"]].append(page)
    return {
        name: sorted(pages, key=lambda p: p["page_no"])
        for name, pages in docs.items()
        if len(pages) >= min_pages
    }
