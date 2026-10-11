# Doc2Graph multi-page extension

Doc2Graph builds a graph for each page. This project extends it to documents by
preserving page order and node IDs, linking repeated entities across pages, and
adding typed cross-page relation candidates. The research question is whether
structural rules and embeddings can identify useful cross-page relations.

The original Doc2Graph system is described in [the ECCV 2022 paper](https://arxiv.org/abs/2208.11168).

## Research status

The existing DocLayNet-small result evaluates a proxy task: linking repeated
headers and footers. With page numbers enabled, its reported macro F1 is 0.841
(95% document-bootstrap CI 0.778–0.895). This does **not** measure the accuracy
of `CONTINUES`, `SAME_TABLE`, or `REFERS_TO`. Those relations have not yet been
evaluated against human labels, and the repository has no retrieval benchmark.
See [method, baseline record and limits](docs/METHOD.md).

The next research milestone is a labelled pilot set of 30–50 multi-page
documents, followed by relation-wise evaluation, candidate-generation recall,
and controlled rules / embeddings / optional LLM ablations.

## Core pipeline

1. Build page-level graphs using the existing Doc2Graph representation.
2. Merge pages in document order, preserving node identity and page metadata.
3. Generate cross-page candidates with structural rules and, optionally,
   embeddings.
4. Assign `CONTINUES`, `SAME_TABLE`, and `REFERS_TO` with rules; an LLM can be
   enabled as an experimental classifier for ambiguous candidates.
5. Store and inspect the graph in SQLite; evaluate against human annotations.

`SAME_ENTITY` links and edge reduction remain available for repeated text,
identifiers, and running elements. SQLite is sufficient for the default graph
workflow. The SQLite embedding cache (`--vector-db`) is optional.

## Setup

Requires Python 3.9–3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Optional components are isolated in extras:

| Extra | Use |
|---|---|
| `local-embeddings` | Local sentence-transformer embeddings for cross-page retrieval |
| `huggingface` | Hugging Face embeddings and experimental LLM classification (`HF_TOKEN`) |
| `langchain` | LangChain adapters for embeddings and the LLM classifier |

Neo4j export, the 3D viewer, the LangChain retriever/RAG, and invoice
extraction were removed from the codebase (see git history); they are outside
the research pipeline.

## Reproduce the existing proxy baseline

Download and unpack the public [DocLayNet-small sample](https://huggingface.co/datasets/pierreguillou/DocLayNet-small):

```bash
mkdir -p data_external && cd data_external
curl -L -O https://huggingface.co/datasets/pierreguillou/DocLayNet-small/resolve/main/data/dataset_small.zip
unzip -q dataset_small.zip -d doclaynet_small_dl/extracted
cd ..
```

Run the existing test suite and proxy evaluation (the `--cv 5` fold check is part of
the reported result; the other values are the defaults):

```bash
uv run pytest tests
uv run python -m doc2graph.eval_multipage \
  --root data_external/doclaynet_small_dl/extracted/small_dataset \
  --cv 5 --bootstrap 1000 --seed 0 --min-iou 0.5 \
  --out results/doclaynet_eval_rerun.json
```

Record the source commit, Python/package environment, dataset source/version,
command/configuration, and complete output alongside every result. A re-run
reproduced the saved 0.841 proxy F1 exactly; see the baseline record in
[docs/METHOD.md](docs/METHOD.md) for what is and is not recorded (notably, no
Hugging Face dataset revision). Do not
describe it as typed-relation performance.

## Page-level inference (original Doc2Graph)

The pretrained checkpoints expect the same feature flags they were trained with;
without them the model has no input features and fails.

```bash
uv run python -m doc2graph.main \
  -addG -addT -addE -addV \
  --weights e2e-funsd-best.pt \
  --inference \
  --docs /path/to/your/image.png
```

Add `--multipage` and pass two or more page images to also write
`inference/document_graph.json` with cross-page text matches.

## Build and inspect a graph

```bash
DATA=data_external/doclaynet_small_dl/extracted/small_dataset
uv run python -m doc2graph.graph_store build \
  --root "$DATA" --db results/doclaynet_graph.db --typed
uv run python -m doc2graph.graph_store summary --db results/doclaynet_graph.db
```

For embedding-based candidate retrieval, install the optional local backend and
build the same documents with the same graph settings:

```bash
uv sync --extra local-embeddings
uv run python -m doc2graph.graph_store build \
  --root "$DATA" --db results/doclaynet_graph_embeddings.db --local-embeddings
```

For an experimental LLM ablation, use a small fixed document subset and record
model, prompt/version, latency, token/cost data, and the candidate set:

```bash
uv sync --extra huggingface
export HF_TOKEN=...
uv run python -m doc2graph.graph_store build \
  --root "$DATA" --db results/doclaynet_graph_llm.db --hf --docs <document.pdf>
```

## Evaluation plan

The current proxy score must remain separate from the new human-labelled
benchmark. Start with 30–50 varied multi-page documents covering paragraph
continuations, section transitions, tables spanning pages, references, and
repeated elements. Use [the annotation guidelines](docs/ANNOTATION_GUIDELINES.md)
to label `CONTINUES`, `SAME_TABLE`, and `REFERS_TO`; double-label a subset and
report agreement and adjudication.

Report precision, recall, and F1 per relation, plus candidate-generation recall
before classification. Compare rules only, rules plus embeddings, and rules plus
embeddings plus optional LLM under the same documents and candidate conditions.
Include latency and LLM cost, then inspect false positives and false negatives
before changing a component. Evaluate graph-aware retrieval against ordinary
text retrieval only after edge quality has been measured, using 20–30 questions
that require evidence from multiple pages and checking answer correctness and
citation support.

## Repository map

- `doc2graph/data/`: page graphs, document graphs, repeated-entity linking,
  edge reduction, and typed relation candidates.
- `doc2graph/eval_multipage.py`: existing DocLayNet proxy evaluation.
- `doc2graph/graph_store.py`: lightweight SQLite graph storage and CLI.
- `doc2graph/vector_store.py`, `doc2graph/langchain_models.py`: optional
  embedding cache and LLM/embedding adapters.
- `tests/`: unit and integration tests (kept local, not in the repository).
- `docs/METHOD.md`: method, assumptions, and current limits.
