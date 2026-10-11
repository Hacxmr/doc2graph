# Method and evaluation limits

## Research question

Doc2Graph constructs one graph per page. This extension combines ordered page
graphs into a document graph and proposes cross-page links. The next experiment
is to determine whether structural rules and embeddings produce correct typed
relations, rather than treating infrastructure or proxy scores as evidence of
relation accuracy.

## Pipeline

1. **Page graphs:** use the existing Doc2Graph page representation and
   intra-page structure.
2. **Document graph:** preserve page order, node IDs, page metadata, and
   intra-page edges when joining pages.
3. **Repeated entities:** generate `SAME_ENTITY` links from repeated text,
   identifiers, repeated page slots, and distinctive shared terms. Reduce
   redundant links to page-ordered chains.
4. **Cross-page candidates:** use structural rules and optional embeddings.
   Measure candidate-generation recall independently of classification.
5. **Typed relations:** assign `CONTINUES`, `SAME_TABLE`, and `REFERS_TO`.
   Rules are the baseline; an LLM is an optional experimental classifier.
6. **Storage and inspection:** use the lightweight SQLite graph store and
   basic graph visualization.
7. **Evaluation:** compare predictions with human-labelled relationships,
   reporting per-relation precision, recall, F1, error analysis, and ablations.

The SQLite embedding cache is optional, and the LLM classifier is experimental. FAISS, the
embedding map and the vector-store CLI/benchmark were removed.
Neo4j export, the 3D viewer, the LangChain retriever/RAG, and invoice-specific
extraction were removed to keep the repository focused; they can be recovered
from git history. None of them establishes cross-page relation quality.

## Existing proxy evaluation

DocLayNet-small provides no labels for `CONTINUES`, `SAME_TABLE`, or
`REFERS_TO`. The existing evaluation instead derives a proxy label for repeated
page headers and footers from category and geometry. Its reported page-number
system has macro F1 0.841 (95% document-bootstrap CI 0.778–0.895); it is a result
for repeated-element linking only. The pages are sampled and often nonadjacent,
so paragraph continuation and reading order are not evaluated. See
[`results/doclaynet_eval.md`](../results/doclaynet_eval.md) for the complete
protocol and results.

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
- Dataset source, version, and document selection.
- Configuration, thresholds, embedding model, and classifier settings.
- The exact command, complete metrics, and output artifact location.

The checked-in DocLayNet result is an existing report, not a freshly reproduced
baseline for a new checkout. Re-run the suite and proxy evaluation before
claiming reproduction.
