# Method and evaluation plan

## Research question

Doc2Graph constructs one graph per page. This extension combines ordered page
graphs into a document graph and proposes cross-page links. The next experiment
is to determine whether structural rules and embeddings produce correct typed
relations, rather than treating infrastructure or proxy scores as evidence of
relation accuracy.

## Data model

- A document is an ordered list of pages; a page is a list of text blocks with a pixel box and a
  page-normalised box.
- A block is identified by `<document>:p<page_no>:n<node>`, for example `MongoDB-manual.pdf:p40:n0`.
  Node IDs come from the page graph and do not change when pages are merged.
- Edges are stored once, from the earlier block to the later one, with a relation, an origin
  (`rule`, `embedding` or `llm`), a score and a short rationale.

| Relation | Scope | Source |
|---|---|---|
| `NEXT`, `NEAR` | within a page | reading order; Doc2Graph kNN neighbours |
| `SAME_ENTITY` | across pages | repeated text, identifiers, page slots, distinctive shared terms |
| `CONTINUES`, `SAME_TABLE`, `REFERS_TO` | across pages | layout rules, optional LLM for ambiguous candidates |
| `SIMILAR_TO` | across pages | embedding neighbour above a similarity threshold, when no stronger relation applies |

## Pipeline

1. **Page graphs:** the existing Doc2Graph page representation and intra-page structure.
2. **Document graph:** preserve page order, node IDs, page metadata and intra-page edges when joining
   pages (`data/document_graph.py`).
3. **Repeated entities:** generate `SAME_ENTITY` links (`data/multipage.py`) and reduce them to
   page-ordered chains (`data/edge_reduction.py`): duplicates collapse, the weakest link of any cycle is
   dropped, and no link is invented.
4. **Cross-page candidates:** layout rules and, optionally, embedding nearest neighbours on other pages
   (`data/relations.py`). Candidate-generation recall is measured separately from classification.
5. **Typed relations:** a candidate is accepted by rule when its evidence is unambiguous, sent to the
   optional classifier when it is not, and otherwise dropped. The classifier only labels existing
   candidates; its output is validated (known relation, one the rules allowed, confidence in range), so it
   cannot invent nodes, pages or relation types. Self-reported LLM confidence is not a calibrated
   probability.
6. **Storage and inspection:** the SQLite graph store (`graph_store.py`) and the drawing script.
7. **Evaluation:** compare predictions with human-labelled relationships, reporting per-relation
   precision, recall, F1, error analysis and ablations.

Embeddings are a core component; the SQLite embedding cache (`vector_store.py`) is an optional
convenience. Neo4j export, the 3D viewer, the LangChain retriever/RAG, invoice extraction, FAISS and the
embedding map were removed to keep the repository focused; they remain in git history and none of them
bears on cross-page relation quality.

## Existing proxy evaluation

DocLayNet-small provides no labels for `CONTINUES`, `SAME_TABLE`, or
`REFERS_TO`. The existing evaluation instead derives a proxy label for repeated
page headers and footers from category and geometry. Its reported page-number
system has macro F1 0.841 (95% document-bootstrap CI 0.778–0.895); it is a result
for repeated-element linking only. The pages are sampled and often nonadjacent,
so paragraph continuation and reading order are not evaluated.

### Baseline record

- Command: `python -m doc2graph.eval_multipage --root <small_dataset> --cv 5 --bootstrap 1000 --seed 0 --min-iou 0.5`.
- Headline row `proposed_page_numbers` (page numbers are opt-in), hand-set tolerances (0.05, 0.02):
  P 0.782, R 0.861, F1 0.841 [0.778, 0.895]. The 5-fold held-out F1 is 0.848 [0.786, 0.908].
  Without page numbers, `proposed` reaches F1 0.405; exact-text matching 0.204.
- A re-run on 2026-10-10 with unchanged code reproduced the saved results JSON exactly
  (parsed comparison). Environment: Python 3.10.19, torch 2.6.0, dgl 1.1.3, numpy 1.26.4.
- Dataset: DocLayNet-small (`pierreguillou/DocLayNet-small`), content hash
  `4ba594b7f63d76e57e5fd4e284bc025f3f3d2011a8f9b9a9893917d9afb2fbeb`
  (sha256 over sorted relative paths and per-file sha256, `._*` files excluded).
- Not recorded: the original run's exact arguments (`--cv 5` is inferred), the Hugging Face
  dataset revision, and a single source commit (the re-run used an uncommitted tree).
  Result files are kept locally, not in the repository.

## Benchmark protocol

Begin with 30–50 multi-page documents covering paragraph continuations, section
transitions, tables spanning pages, references, and repeated headers/footers.
Follow the [annotation guidelines](ANNOTATION_GUIDELINES.md). Double-label a
subset, report agreement before adjudication, and document the sampling method.
Treat this as a pilot, not as a sufficient final benchmark by default.

Run these comparisons on identical documents and comparable candidate sets:

- **A:** structural rules only.
- **B:** structural rules plus embeddings.
- **C:** structural rules plus embeddings plus optional LLM classification.

For each relation, report precision, recall, F1, support, and confidence
intervals. Report candidate-generation recall before the classifier. For the
LLM, also record model/version, prompt, latency, and cost. Inspect false
positives and false negatives and change one component based on the largest
failure category before rerunning the fixed evaluation.

Only after edge quality is established, compare ordinary text retrieval with
graph-aware retrieval on 20–30 questions requiring evidence across pages. Check
answer correctness and whether citations support each answer claim.

## Reproducibility record

Every evaluation run should record:

- Git commit hash and any local modifications.
- Python and dependency versions.
- Dataset source, version and document selection.
- Configuration, thresholds, embedding model and classifier settings.
- The exact command, complete metrics and where the output is stored.

Result files and generated databases are not committed (`results/` and `*.db` are gitignored). Keep them
locally or archive them with the run record. The baseline record above is the only result summary kept
in the repository, and it is a proxy result, not typed-relation accuracy.
