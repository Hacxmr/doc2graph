"""An optional SQLite cache of block embeddings (no server, not needed by the default pipeline).

Every block's embedding is stored once per model, keyed by the hash of its text.
  * a cache: ``store.embedder(base_embed)`` is a drop-in ``embed`` for
    ``build_document_graph`` / ``graph_store build --vector-db`` that only embeds unseen text,
  * an exact search: nearest blocks to a query vector or to another block (numpy, brute force).

Embeddings themselves come from ``langchain_models.py``; the graph builds without this module.
"""

import argparse
import hashlib
import json
import sqlite3

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS vectors(
    model TEXT NOT NULL, hash TEXT NOT NULL, dim INTEGER NOT NULL, vec BLOB NOT NULL, text TEXT,
    PRIMARY KEY(model, hash));
CREATE TABLE IF NOT EXISTS members(
    block_id TEXT NOT NULL, model TEXT NOT NULL, document TEXT NOT NULL, page_no INTEGER, node INTEGER,
    category TEXT, hash TEXT NOT NULL,
    PRIMARY KEY(block_id, model));
CREATE INDEX IF NOT EXISTS members_doc ON members(document, model);
"""


def _hash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _unit(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def top_k(vecs, q, k):
    sims = vecs @ q
    k = min(k, len(sims))
    ids = np.argsort(-sims)[:k]
    return ids, sims[ids]


class VectorStore:
    def __init__(self, path, model="default"):
        self.model = model
        self.con = sqlite3.connect(path)
        self.con.execute("PRAGMA locking_mode = EXCLUSIVE")  # same reasoning as GraphStore
        self.con.execute("PRAGMA journal_mode = MEMORY")
        self.con.execute("PRAGMA synchronous = OFF")
        self.con.executescript(SCHEMA)

    def close(self):
        self.con.commit()
        self.con.close()

    # ---- writing -------------------------------------------------------------
    def embedder(self, embed, max_chars=512):
        """Cached ``embed``: looks texts up by hash, embeds and stores only the misses."""

        def cached(texts):
            clipped = [t.strip()[:max_chars] for t in texts]
            keys = [_hash(t) for t in clipped]
            found = {}
            for i in range(0, len(keys), 500):
                chunk = keys[i : i + 500]
                for h, dim, blob in self.con.execute(
                    f"SELECT hash, dim, vec FROM vectors WHERE model = ? AND hash IN ({','.join('?' * len(chunk))})",
                    [self.model, *chunk],
                ):
                    found[h] = np.frombuffer(blob, dtype=np.float32)
            missing = {h: t for h, t in zip(keys, clipped) if h not in found}
            if missing:
                new = _unit(embed(list(missing.values())))
                self.con.executemany(
                    "INSERT OR REPLACE INTO vectors(model, hash, dim, vec, text) VALUES (?, ?, ?, ?, ?)",
                    [(self.model, h, new.shape[1], v.tobytes(), t) for (h, t), v in zip(missing.items(), new)],
                )
                found.update(zip(missing, new))
            return np.stack([found[h] for h in keys]) if keys else np.zeros((0, 1), dtype=np.float32)

        return cached

    def index_document(self, name, pages, embed):
        """Embed every non-trivial block of a document and register it. Returns blocks indexed."""
        cached = self.embedder(embed)
        keys, texts = [], []
        for i, page in enumerate(pages):
            for n, node in enumerate(page["nodes"]):
                if len(node["text"].strip()) >= 3:
                    keys.append((page.get("page_no", i + 1), n, node.get("category")))  # as in GraphStore
                    texts.append(node["text"])
        cached(texts)
        self.con.execute("DELETE FROM members WHERE document = ? AND model = ?", (name, self.model))
        self.con.executemany(
            "INSERT INTO members(block_id, model, document, page_no, node, category, hash) VALUES (?,?,?,?,?,?,?)",
            [
                (f"{name}:p{p}:n{n}", self.model, name, p, n, c, _hash(t.strip()[:512]))
                for (p, n, c), t in zip(keys, texts)
            ],
        )
        self.con.commit()
        return len(keys)

    # ---- reading -------------------------------------------------------------
    def info(self):
        rows = self.con.execute(
            "SELECT model, COUNT(*), MIN(dim) FROM vectors GROUP BY model ORDER BY model"
        ).fetchall()
        docs = self.con.execute(
            "SELECT model, COUNT(DISTINCT document), COUNT(*) FROM members GROUP BY model ORDER BY model"
        ).fetchall()
        return {"vectors": rows, "documents": docs}

    def load(self, document=None):
        """(unit vectors, member rows) for this store's model, optionally one document."""
        rows = self.con.execute(
            "SELECT m.block_id, m.document, m.page_no, m.node, m.category, v.text, v.vec "
            "FROM members m JOIN vectors v ON v.model = m.model AND v.hash = m.hash "
            "WHERE m.model = ? AND (? IS NULL OR m.document = ?) ORDER BY m.document, m.page_no, m.node",
            (self.model, document, document),
        ).fetchall()
        if not rows:
            return np.zeros((0, 1), dtype=np.float32), []
        meta = [dict(zip(("id", "document", "page_no", "node", "category", "text"), r[:6])) for r in rows]
        return np.stack([np.frombuffer(r[6], dtype=np.float32) for r in rows]), meta

    def search(self, query_vec, k=5, document=None):
        vecs, meta = self.load(document)
        if not meta:
            return []
        q = _unit(np.asarray(query_vec, dtype=np.float32)[None])[0]
        ids, sims = top_k(vecs, q, k)
        return [(meta[i], float(x)) for i, x in zip(ids, sims)]

    def neighbors(self, block_id, k=5, other_pages_only=True):
        vecs, meta = self.load()
        index = {m["id"]: i for i, m in enumerate(meta)}
        if block_id not in index:
            raise KeyError(f"block not in the vector store: {block_id}")
        me = index[block_id]
        # ask for extra results: the block itself and (optionally) same-page blocks are dropped below
        ids, sims = top_k(vecs, vecs[me], len(meta))
        out = []
        for i, sim in zip(ids, sims):
            if i == me:
                continue
            same_page = meta[i]["document"] == meta[me]["document"] and meta[i]["page_no"] == meta[me]["page_no"]
            if other_pages_only and same_page:
                continue
            out.append((meta[i], float(sim)))
            if len(out) == k:
                break
        return out
