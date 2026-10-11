# Acknowledgements and prior work

This project extends **Doc2Graph** to multi-page documents. The single-page model, graph
construction and training code in this repository come from the original authors and remain
under their MIT licence (see `LICENCE`). The multi-page work listed in `docs/METHOD.md` is by
Mitali Raj, Dr. Sanket Biswas and Dr. Josep Lladós (the latter two are also authors of the original Doc2Graph paper).

## What is from prior work

| Component | Source |
|---|---|
| Doc2Graph model, per-page graph construction, GNN training (`doc2graph/models`, `doc2graph/data/graph_builder.py` and related) | Gemelli, Biswas, Civitelli, Lladós, Marinai. *Doc2Graph: a Task Agnostic Document Understanding Framework based on Graph Neural Networks.* ECCV 2022 TiE workshop. arXiv:2208.11168. Repository: github.com/andreagemelli/doc2graph |
| Evaluation data | Pfitzmann, Auer, Dolfi, Nassar, Staar. *DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis.* KDD 2022. Small subset: `pierreguillou/DocLayNet-small` on Hugging Face |
| Graph neural network library | DGL (Deep Graph Library) |
| LLM orchestration | LangChain (`langchain-core`, `langchain-huggingface`) |
| Models used through the Inference API | Whatever you choose; defaults are Llama 3.1 8B Instruct (Meta, gated licence) and `sentence-transformers/all-MiniLM-L6-v2` |

## What is new here

Cross-page entity links, typed relations with rationale, edge reduction, the SQLite store and the
multi-page evaluation. Details and
limits are in `docs/METHOD.md`.

If you use this work, cite both the original Doc2Graph paper and this repository (`CITATION.cff`).
