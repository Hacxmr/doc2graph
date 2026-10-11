# Doc2Graph multi-page extension

Doc2Graph ([Gemelli et al., ECCV 2022 TiE workshop](https://arxiv.org/abs/2208.11168)) turns a
document *page* into a graph and applies a graph neural network to it. This project extends the
representation from pages to **documents**: it keeps page order and node identity, links blocks that
repeat across pages, and proposes **typed cross-page relations**.

| Relation | Meaning |
|---|---|
| `SAME_ENTITY` | the same running element or identifier on another page (header, footer, ID) |
| `CONTINUES` | the later block continues the sentence or paragraph of the earlier one |
| `SAME_TABLE` | the later block continues a table across the page break |
| `REFERS_TO` | one block explicitly refers to a figure or table carried by the other |
| `SIMILAR_TO` | semantically close blocks, when no stronger relation applies |

The research question is whether structural rules and embeddings (and, optionally, an LLM) produce
*correct* typed relations. See [docs/METHOD.md](docs/METHOD.md) for the method and the evaluation plan.

## Status

- **Implemented:** multipage graph construction, `SAME_ENTITY` linking with edge reduction, typed
  relation candidates (rules, embeddings, optional LLM), a SQLite graph store, and a proxy evaluation on
  DocLayNet-small.
- **Evaluated:** only a proxy task, linking repeated headers and footers. With page numbers enabled the
  macro F1 is 0.841 (95% document-bootstrap CI 0.778–0.895).
- **Not evaluated:** the accuracy of `CONTINUES`, `SAME_TABLE` and `REFERS_TO`. There are no human labels
  yet and no retrieval benchmark. The proxy score must not be read as typed-relation accuracy.
- **Next:** a labelled pilot set of 30–50 multi-page documents, per-relation precision/recall/F1,
  candidate-generation recall, and rules / +embeddings / +LLM ablations.

## Pipeline

1. Build page graphs with the original Doc2Graph representation.
2. Merge pages in document order, preserving node IDs and page metadata.
3. Generate cross-page candidates from structural rules and, optionally, embeddings.
4. Assign typed relations by rule; an optional LLM labels only the ambiguous candidates.
5. Store the graph in SQLite for inspection; evaluate against human annotations.

## Installation

Requires Python 3.9–3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Optional components are installed as extras:

| Extra | Adds |
|---|---|
| `local-embeddings` | local sentence-transformer embeddings (no token needed) |
| `huggingface` | Hugging Face embeddings and the experimental LLM classifier (`HF_TOKEN`) |
| `langchain` | the LangChain adapters only |

```bash
uv sync --extra local-embeddings
```

## Usage

### Page-level inference (original Doc2Graph)

The pretrained checkpoints need the feature flags they were trained with; without them the model has no
input features and fails.

```bash
uv run python -m doc2graph.main \
  -addG -addT -addE -addV \
  --weights e2e-funsd-best.pt \
  --inference \
  --docs page-1.png
```

Pass two or more page images with `--multipage` to also write `inference/document_graph.json`
(nodes, page-level links and cross-page text matches).

### Build and query a document graph

Download [DocLayNet-small](https://huggingface.co/datasets/pierreguillou/DocLayNet-small):

```bash
mkdir -p data_external && cd data_external
curl -L -O https://huggingface.co/datasets/pierreguillou/DocLayNet-small/resolve/main/data/dataset_small.zip
unzip -q dataset_small.zip -d doclaynet_small_dl/extracted
cd ..
```

Build the same documents under three conditions:

```bash
DATA=data_external/doclaynet_small_dl/extracted/small_dataset

# A: rules only
uv run python -m doc2graph.graph_store build --root "$DATA" --db graph_rules.db --typed

# B: rules + embeddings (local-embeddings extra); --vector-db <file> caches the embeddings
uv run python -m doc2graph.graph_store build --root "$DATA" --db graph_emb.db --local-embeddings

# C: rules + embeddings + LLM (experimental; huggingface extra and HF_TOKEN)
uv run python -m doc2graph.graph_store build --root "$DATA" --db graph_llm.db --hf --docs <document.pdf>
```

For an LLM run use a small fixed document subset and record the model, prompt, latency, token/cost data
and the candidate set.

Inspect a database:

```bash
uv run python -m doc2graph.graph_store summary   --db graph_rules.db
uv run python -m doc2graph.graph_store entities  --db graph_rules.db --doc <document.pdf>
uv run python -m doc2graph.graph_store find      --db graph_rules.db --text "Release 3.0.4"
uv run python -m doc2graph.graph_store chain     --db graph_rules.db --block "<document.pdf>:p40:n0"
uv run python -m doc2graph.graph_store neighbors --db graph_rules.db --block "<document.pdf>:p40:n0"
```

Draw one document's graph:

```bash
uv run python -m doc2graph.visualize_document_graph --root "$DATA" --list
uv run python -m doc2graph.visualize_document_graph --root "$DATA" --doc <document.pdf> --pages 3 --out graph.png
```

### Reproduce the proxy baseline

```bash
uv run python -m doc2graph.eval_multipage \
  --root "$DATA" --cv 5 --bootstrap 1000 --seed 0 --min-iou 0.5 \
  --out doclaynet_eval_rerun.json
```

A re-run reproduced the previously saved result exactly. The
[baseline record](docs/METHOD.md#baseline-record) lists what was *not* recorded: the original run's
arguments and the Hugging Face dataset revision.

## Repository layout

| Path | Content |
|---|---|
| `doc2graph/data/` | page graphs and features (original); document graph, repeated-entity links, edge reduction, typed relations |
| `doc2graph/models/`, `doc2graph/training/` | original Doc2Graph models and training |
| `doc2graph/inference.py`, `doc2graph/main.py` | original inference and CLI, plus `--multipage` |
| `doc2graph/eval_multipage.py` | DocLayNet proxy evaluation |
| `doc2graph/graph_store.py` | SQLite graph store and CLI |
| `doc2graph/langchain_models.py` | LangChain adapters for embeddings and the LLM classifier |
| `doc2graph/vector_store.py` | optional SQLite embedding cache |
| `doc2graph/visualize_document_graph.py` | document-graph drawing |
| `docs/` | [method and evaluation plan](docs/METHOD.md), [annotation guidelines](docs/ANNOTATION_GUIDELINES.md) |

Tests are kept in a local `tests/` folder that is not part of the repository.

## Citation and licence

Cite this software and the original Doc2Graph paper; see [CITATION.cff](CITATION.cff) and
[ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md). Released under the MIT licence ([LICENCE](LICENCE)).
