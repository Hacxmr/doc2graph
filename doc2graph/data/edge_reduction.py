"""Order cross-page links and remove redundant ones.

The matcher can link the same running header on pages 40, 214, 267 and 282 to
every other page (6 links). One chain (3 links) says the same thing. Links are
reduced to a maximum-score spanning forest, so every group of linked blocks
(an "entity") stays connected, no link is invented, and the survivors are
sorted by page so a chain reads in document order.
"""


def _key(end):
    return (end["page"], end["node"])


def reduce_links(links):
    """Return {"links", "entities", "stats"} for a list of cross-page links.

    * duplicates (a to b and b to a, or repeated candidates) keep the best score;
    * a cycle (a-b, b-c, a-c) loses its weakest link; ties keep neighbouring pages;
    * every link is oriented from the earlier to the later block;
    * links carry their entity id and their rank inside the entity, in page order.
    """
    best = {}
    for link in links:
        a, b = sorted((_key(link["source"]), _key(link["target"])))
        pair = (a, b)
        if pair not in best or link["score"] > best[pair]["score"]:
            best[pair] = {
                **link,
                "source": {"page": a[0], "node": a[1]},
                "target": {"page": b[0], "node": b[1]},
                "page_delta": b[0] - a[0],
            }

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    kept = []
    for pair in sorted(best, key=lambda p: (-best[p]["score"], p[1][0] - p[0][0], p)):
        ra, rb = find(pair[0]), find(pair[1])
        if ra != rb:
            parent[ra] = rb
            kept.append(best[pair])

    members = {}
    for node in list(parent):
        members.setdefault(find(node), []).append(node)
    groups = sorted(sorted(m) for m in members.values())
    entity_of = {node: i for i, group in enumerate(groups) for node in group}

    kept.sort(key=lambda k: (_key(k["source"]), _key(k["target"])))
    rank = {}
    for link in kept:
        eid = entity_of[_key(link["source"])]
        rank[eid] = rank.get(eid, 0) + 1
        link["entity"], link["rank"] = eid, rank[eid]
    kept.sort(key=lambda k: (k["entity"], k["rank"]))

    entities = [
        {"id": i, "members": [{"page": p, "node": n} for p, n in group]}
        for i, group in enumerate(groups)
    ]
    return {
        "links": kept,
        "entities": entities,
        "stats": {
            "raw": len(links),
            "unique": len(best),
            "kept": len(kept),
            "entities": len(groups),
        },
    }
