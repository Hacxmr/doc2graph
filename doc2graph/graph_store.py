"""Store document graphs in SQLite and query the relationships between blocks.

Tables
    documents(id, name, category, n_pages)
    pages(id, document_id, page_no, page_index, next_page)   next_page = page order
    blocks(id, document_id, page_id, node, category, text, x0, y0, x1, y1, entity)
    edges(id, document_id, src, dst, relation, match_type, score, rank, entity)

Containment (document > page > block) lives in foreign keys, not in edges.
Edge relations, each stored once, from the earlier to the later block:
    NEXT         the next block in reading order on the same page
    NEAR         spatial neighbour on the page (Doc2Graph kNN), without the pairs
                 that are already NEXT
    CONTINUES / SAME_TABLE / REFERS_TO
                 typed cross-page relations (data/relations.py); match_type holds
                 their origin, "rule" or "llm". Present when the document graph is
                 built with typed=True.
    SAME_ENTITY  cross-page link between blocks that repeat (same header, same
                 identifier); a spanning chain per entity, ordered by page

Usage:
    python -m doc2graph.graph_store build --root <small_dataset dir> --db graph.db
    python -m doc2graph.graph_store summary --db graph.db
    python -m doc2graph.graph_store find --db graph.db --text "Release 3.0.4"
    python -m doc2graph.graph_store chain --db graph.db --block "MongoDB-manual.pdf:p40:n0"
"""

import argparse
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
    id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, category TEXT, n_pages INTEGER);
CREATE TABLE IF NOT EXISTS pages(
    id TEXT PRIMARY KEY, document_id INTEGER NOT NULL REFERENCES documents(id),
    page_no INTEGER, page_index INTEGER, next_page TEXT);
CREATE TABLE IF NOT EXISTS blocks(
    id TEXT PRIMARY KEY, document_id INTEGER NOT NULL REFERENCES documents(id),
    page_id TEXT NOT NULL REFERENCES pages(id), node INTEGER, category TEXT, text TEXT,
    x0 REAL, y0 REAL, x1 REAL, y1 REAL, entity TEXT);
CREATE TABLE IF NOT EXISTS edges(
    id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL REFERENCES documents(id),
    src TEXT NOT NULL REFERENCES blocks(id), dst TEXT NOT NULL REFERENCES blocks(id),
    relation TEXT NOT NULL, match_type TEXT, score REAL, rank INTEGER, entity TEXT,
    rationale TEXT,
    UNIQUE(src, dst, relation));
CREATE INDEX IF NOT EXISTS edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS edges_dst ON edges(dst);
CREATE INDEX IF NOT EXISTS edges_doc_rel ON edges(document_id, relation);
CREATE INDEX IF NOT EXISTS blocks_entity ON blocks(entity);
"""


_WHY = {
    ("NEXT", None): "Next block in reading order on the same page",
    ("NEAR", None): "Spatial neighbour on the page (Doc2Graph k-nearest-neighbour edge)",
    ("SAME_ENTITY", "same_text"): "Identical text on both pages",
    ("SAME_ENTITY", "shared_identifier"): "Both blocks contain the same identifier (letters and digits)",
    ("SAME_ENTITY", "repeated_slot"): "Same page position, same text apart from a number (running header/footer)",
    ("SAME_ENTITY", "shared_terms"): "Blocks share most of their distinctive terms",
}


def _evidence(edge):
    sim = edge.get("similarity")
    note = edge.get("evidence") or ""
    if sim is None or "cosine=" in note:
        return note
    return f"{note} (embedding cosine {sim:.2f})".strip()


def rationale(relation, match_type=None, evidence=None):
    """Plain-language reason an edge exists (stored with the edge)."""
    if relation in ("CONTINUES", "SAME_TABLE", "REFERS_TO", "SIMILAR_TO"):  # match_type holds the origin
        return f"{match_type}: {evidence}" if evidence else str(match_type)
    return _WHY.get((relation, match_type), f"{relation} ({match_type})")


class GraphStore:
    def __init__(self, path):
        self.con = sqlite3.connect(path)
        # Network-style volumes (e.g. an external drive mounted over FUSE) do not
        # support SQLite's file locking and report a corrupt database. One
        # exclusive connection with an in-memory journal avoids that; the data is
        # derived and can be rebuilt, so weaker crash safety is acceptable.
        self.con.execute("PRAGMA locking_mode = EXCLUSIVE")
        self.con.execute("PRAGMA journal_mode = MEMORY")
        self.con.execute("PRAGMA synchronous = OFF")
        self.con.execute("PRAGMA foreign_keys = ON")
        self.con.executescript(SCHEMA)
        columns = [r[1] for r in self.con.execute("PRAGMA table_info(edges)")]
        if "rationale" not in columns:  # database written by an earlier version
            self.con.execute("ALTER TABLE edges ADD COLUMN rationale TEXT")
            for relation, match_type in self.con.execute(
                "SELECT DISTINCT relation, match_type FROM edges"
            ).fetchall():
                self.con.execute(
                    "UPDATE edges SET rationale = ? WHERE relation = ? AND match_type IS ?",
                    (rationale(relation, match_type), relation, match_type),
                )

    def close(self):
        self.con.commit()
        self.con.close()

    # ---- writing -------------------------------------------------------------
    def add_document(self, name, pages, result=None, category=None):
        """(Re)write one document. Returns the number of edges stored."""
        from .data.document_graph import INTRA_PAGE, build_document_graph

        result = result or build_document_graph(pages)
        con = self.con
        old = con.execute("SELECT id FROM documents WHERE name = ?", (name,)).fetchone()
        if old:  # idempotent: replace what was stored before
            for table in ("edges", "blocks", "pages"):
                con.execute(f"DELETE FROM {table} WHERE document_id = ?", (old[0],))
            con.execute("DELETE FROM documents WHERE id = ?", (old[0],))
        category = category or pages[0].get("doc_category")
        doc_id = con.execute(
            "INSERT INTO documents(name, category, n_pages) VALUES (?, ?, ?)",
            (name, category, len(pages)),
        ).lastrowid

        def page_no(i):
            return pages[i].get("page_no", i + 1)

        def page_id(i):
            return f"{name}:p{page_no(i)}"

        def block_id(i, j):
            return f"{name}:p{page_no(i)}:n{j}"

        entity_of = {
            (m["page"], m["node"]): f"{name}#{e['id']}"
            for e in result["entities"]
            for m in e["members"]
        }
        for i, page in enumerate(pages):
            nxt = page_id(i + 1) if i + 1 < len(pages) else None
            con.execute(
                "INSERT INTO pages VALUES (?, ?, ?, ?, ?)", (page_id(i), doc_id, page_no(i), i, nxt)
            )
            con.executemany(
                "INSERT INTO blocks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (block_id(i, j), doc_id, page_id(i), j, n.get("category"), n["text"],
                     *n["box"], entity_of.get((i, j)))
                    for j, n in enumerate(page["nodes"])
                ],
            )

        edges = []
        next_pairs = set()
        for i, page in enumerate(pages):
            for j in range(len(page["nodes"]) - 1):
                next_pairs.add((i, j, j + 1))
                edges.append((doc_id, block_id(i, j), block_id(i, j + 1), "NEXT", None, None, None, None, rationale("NEXT")))

        graph = result["graph"]
        where = [(i, j) for i, p in enumerate(pages) for j in range(len(p["nodes"]))]
        near = set()
        for u, v, et in zip(*(t.tolist() for t in graph.edges()), graph.edata["etype"].tolist()):
            if et != INTRA_PAGE:
                continue
            (pu, nu), (pv, nv) = where[u], where[v]
            a, b = sorted((nu, nv))
            if pu == pv and (pu, a, b) not in next_pairs and a != b:
                near.add((pu, a, b))  # each undirected pair once, never a NEXT pair
        for i, a, b in sorted(near):
            edges.append((doc_id, block_id(i, a), block_id(i, b), "NEAR", None, None, None, None, rationale("NEAR")))

        for link in result["cross_page_links"]:
            s, t = link["source"], link["target"]
            eid = entity_of.get((s["page"], s["node"]))
            edges.append(
                (doc_id, block_id(s["page"], s["node"]), block_id(t["page"], t["node"]),
                 "SAME_ENTITY", link["match_type"], link["score"], link.get("rank"), eid,
                 rationale("SAME_ENTITY", link["match_type"]))
            )
        for edge in result.get("typed_edges", []):
            s, t = edge["source"], edge["target"]
            edges.append(
                (doc_id, block_id(s["page"], s["node"]), block_id(t["page"], t["node"]),
                 edge["relation"], edge["origin"], edge["score"], None, None,
                 rationale(edge["relation"], edge["origin"], _evidence(edge)))
            )
        con.executemany(
            "INSERT OR IGNORE INTO edges(document_id, src, dst, relation, match_type, score, rank, entity,"
            " rationale) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            edges,
        )
        con.commit()
        return len(edges)

    # ---- reading -------------------------------------------------------------
    def counts(self):
        q = lambda t: self.con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # noqa: E731
        return {t: q(t) for t in ("documents", "pages", "blocks", "edges")}

    def relation_summary(self):
        return self.con.execute(
            "SELECT relation, COALESCE(match_type, '-'), COUNT(*) FROM edges "
            "GROUP BY 1, 2 ORDER BY 3 DESC"
        ).fetchall()

    def find(self, text, limit=10):
        return self.con.execute(
            "SELECT b.id, p.page_no, b.category, b.text FROM blocks b JOIN pages p ON p.id = b.page_id "
            "WHERE b.text LIKE ? LIMIT ?",
            (f"%{text}%", limit),
        ).fetchall()

    def chain(self, block):
        """All blocks of the entity of ``block``, in page order."""
        return self.con.execute(
            "SELECT b.id, p.page_no, b.category, b.text FROM blocks b JOIN pages p ON p.id = b.page_id "
            "WHERE b.entity = (SELECT entity FROM blocks WHERE id = ?) AND b.entity IS NOT NULL "
            "ORDER BY p.page_index, b.node",
            (block,),
        ).fetchall()

    def neighbors(self, block):
        return self.con.execute(
            "SELECT 'out' AS dir, e.relation, COALESCE(e.match_type, '-'), b.id, substr(b.text, 1, 50) "
            "FROM edges e JOIN blocks b ON b.id = e.dst WHERE e.src = ? "
            "UNION ALL SELECT 'in', e.relation, COALESCE(e.match_type, '-'), b.id, substr(b.text, 1, 50) "
            "FROM edges e JOIN blocks b ON b.id = e.src WHERE e.dst = ? "
            "ORDER BY 2, 1",
            (block, block),
        ).fetchall()

    def entities(self, document=None, limit=10):
        """Largest entities (blocks repeated across pages): id, pages, sample text."""
        rows = self.con.execute(
            "SELECT b.entity, COUNT(*) AS n, MIN(b.text) FROM blocks b JOIN documents d ON d.id = b.document_id "
            "WHERE b.entity IS NOT NULL AND (? IS NULL OR d.name = ?) GROUP BY b.entity "
            "ORDER BY n DESC LIMIT ?",
            (document, document, limit),
        ).fetchall()
        out = []
        for entity, n, sample in rows:
            pages = [r[0] for r in self.con.execute(
                "SELECT DISTINCT p.page_no FROM blocks b JOIN pages p ON p.id = b.page_id "
                "WHERE b.entity = ? ORDER BY p.page_index", (entity,))]
            out.append((entity, n, pages, sample))
        return out


def _build(args):
    from .data.doclaynet import load_documents

    docs = load_documents(args.root, min_pages=2)
    names = args.docs or list(docs)
    store = GraphStore(args.db)
    embed = classifier = None
    if args.local_embeddings and not args.hf:
        from .langchain_models import langchain_embedder, local_embeddings

        embed = langchain_embedder(local_embeddings(args.embedding_model))
    if args.hf:
        from .langchain_models import huggingface_models, langchain_classifier, langchain_embedder

        chat, embeddings = huggingface_models(args.llm, args.embedding_model)
        embed, classifier = langchain_embedder(embeddings), langchain_classifier(chat)
    vstore = None
    if args.vector_db:
        from .vector_store import VectorStore

        if embed is None:
            raise SystemExit("--vector-db needs --local-embeddings or --hf")
        vstore = VectorStore(args.vector_db, args.embedding_model)
        embed = vstore.embedder(embed)  # embeddings are read from / written to the vector database
    for name in names:
        result = None
        if args.typed or args.hf or args.local_embeddings:
            from .data.document_graph import build_document_graph

            result = build_document_graph(docs[name], typed=True, embed=embed, classifier=classifier)
        edges = store.add_document(name, docs[name], result)
        if vstore:
            vstore.index_document(name, docs[name], embed)
        print(f"{name}: {len(docs[name])} pages, {edges} edges")
    print("stored:", store.counts())
    store.close()
    if vstore:
        vstore.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="load DocLayNet documents into the database")
    b.add_argument("--root", required=True)
    b.add_argument("--db", required=True)
    b.add_argument("--docs", nargs="*", help="document names (default: all multi-page ones)")
    b.add_argument("--typed", action="store_true", help="add CONTINUES / SAME_TABLE / REFERS_TO (rules only)")
    b.add_argument("--local-embeddings", action="store_true",
                   help="embedding-based candidates with a local sentence-transformers model (no token)")
    b.add_argument("--vector-db", help="store / reuse block embeddings in this vector database")
    b.add_argument("--hf", action="store_true", help="also use Hugging Face models (implies --typed; needs HF_TOKEN)")
    b.add_argument("--llm", default="meta-llama/Llama-3.1-8B-Instruct")
    b.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    for name, extra in (("summary", []), ("entities", ["--doc", "--limit"]), ("find", ["--text"]),
                        ("chain", ["--block"]), ("neighbors", ["--block"])):
        p = sub.add_parser(name)
        p.add_argument("--db", required=True)
        for e in extra:
            p.add_argument(e, type=int if e == "--limit" else str, default=10 if e == "--limit" else None)
    args = parser.parse_args()

    if args.cmd == "build":
        return _build(args)
    store = GraphStore(args.db)
    if args.cmd == "summary":
        print("counts:", store.counts())
        print(f"{'relation':<12} {'match type':<18} edges")
        for relation, match, n in store.relation_summary():
            print(f"{relation:<12} {match:<18} {n}")
    elif args.cmd == "entities":
        for entity, n, pages, sample in store.entities(args.doc, args.limit):
            print(f"{entity:<28} {n:>3} blocks on pages {pages}  {sample[:50]!r}")
    elif args.cmd == "find":
        for row in store.find(args.text):
            print(row)
    elif args.cmd == "chain":
        for row in store.chain(args.block):
            print(row)
    elif args.cmd == "neighbors":
        for row in store.neighbors(args.block):
            print(row)
    store.close()


if __name__ == "__main__":
    main()
