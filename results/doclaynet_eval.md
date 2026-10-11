# DocLayNet-small proxy evaluation: repeated header/footer linking

Command (unchanged evaluation code, defaults except `--cv`):

    python -m doc2graph.eval_multipage --root <small_dataset> --cv 5 --bootstrap 1000 --seed 0 --min-iou 0.5 --out <file>.json

Headline row: `proposed_page_numbers` (page numbers are opt-in), with hand-set position
tolerances (0.05, 0.02): F1 0.841 [0.778, 0.895]. The less optimistic figure is the 5-fold
held-out F1 of 0.848 [0.786, 0.908], where tolerances are chosen on other folds.

137 documents, 1986 same-document page pairs, 2051 gold running-element edges

Macro-averaged over documents; 95% bootstrap CI over documents in brackets.

| System | Precision | Recall | F1 | Unlabeled edges / pair |
|---|---|---|---|---|
| single_page | nan [nan, nan] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.00 |
| exact_text | 0.791 [0.650, 0.901] | 0.164 [0.109, 0.221] | 0.204 [0.140, 0.272] | 0.05 |
| token_matcher | 0.729 [0.619, 0.828] | 0.386 [0.298, 0.472] | 0.392 [0.305, 0.479] | 0.16 |
| proposed | 0.732 [0.622, 0.829] | 0.402 [0.321, 0.502] | 0.405 [0.324, 0.501] | 0.16 |
| proposed_page_numbers | 0.782 [0.708, 0.859] | 0.861 [0.797, 0.917] | 0.841 [0.778, 0.895] | 0.16 |

| System | Same-doc pairs with an edge | Cross-doc pairs with an edge (control) | Cross-doc pairs with a header/footer edge |
|---|---|---|---|
| single_page | 0.000 | 0.000 [0.000, 0.000] | 0.000 |
| exact_text | 0.119 | 0.000 [0.000, 0.000] | 0.000 |
| token_matcher | 0.171 | 0.005 [0.002, 0.008] | 0.001 |
| proposed | 0.173 | 0.005 [0.002, 0.008] | 0.001 |
| proposed_page_numbers | 0.582 | 0.030 [0.023, 0.037] | 0.027 |

| Protocol | Precision | Recall | F1 |
|---|---|---|---|
| Hand-set tolerances (0.05, 0.02), all documents | 0.782 [0.703, 0.853] | 0.861 [0.793, 0.919] | 0.841 [0.780, 0.898] |
| Best tolerances (0.02, 0.01) chosen on all documents (in-sample) | 0.809 [0.740, 0.873] | 0.861 [0.798, 0.917] | 0.848 [0.786, 0.899] |
| Tolerances chosen on other folds, 5-fold held-out | 0.809 [0.736, 0.877] | 0.861 [0.797, 0.919] | 0.848 [0.786, 0.908] |

Tolerances chosen per fold: (0.02, 0.01), (0.02, 0.01), (0.02, 0.01), (0.02, 0.01), (0.02, 0.01)

Edge reduction over 137 documents: 525 candidate links, 525 unique pairs, 358 kept (31.8% fewer), 213 entities, no entity split.

This is the originally saved result. A re-run on 2026-10-10 with the unchanged
evaluation code (`--cv 5 --bootstrap 1000 --seed 0 --min-iou 0.5`) produced JSON
identical (parsed comparison) to `results/doclaynet_eval.json`: proposed_page_numbers F1 = 0.841
(95% document-bootstrap CI 0.778-0.895). The re-run
copy was not kept; its sha256 is recorded in
[reproduction/PROVENANCE.md](reproduction/PROVENANCE.md).

Provenance limits: the exact arguments of the original run and the Hugging Face
revision of DocLayNet-small were not recorded. The `--cv 5` argument is inferred
from the 5-fold table above, and the dataset is identified only by the content
hash in the provenance file. The re-run was made on an uncommitted working tree,
so it is not tied to a single source commit. This is a proxy result for repeated
header/footer linking; it says nothing about `CONTINUES`, `SAME_TABLE` or
`REFERS_TO`.
