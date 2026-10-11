# DocLayNet-small proxy evaluation: re-run record (2026-10-10)

Task: repeated header/footer linking (proxy). It does not measure CONTINUES, SAME_TABLE or REFERS_TO.

## Git state
- HEAD: f602316962d81cb887ca4a39923bf03e3c0d01f4 (branch master)
- Working tree is **dirty and uncommitted**: the multipage code, tests and docs are untracked, and
  several upstream files are modified. A commit hash alone does not identify this code.
- uv.lock sha256: b973271a71760f6056828a8da2305c5878d881d096d1c413a54292c3cb6f70d1

## Environment
Python 3.10.19, torch 2.6.0, dgl 1.1.3, numpy 1.26.4 (uv environment on /Volumes/d2g-dev).

## Dataset
- Path: data_external/doclaynet_small_dl/extracted/small_dataset (train 2073 / val 192 / test 147 files, 2412 total)
- Content hash (sha256 over sorted relative path + per-file sha256, `._*` files excluded):
  4ba594b7f63d76e57e5fd4e284bc025f3f3d2011a8f9b9a9893917d9afb2fbeb
- Source: pierreguillou/DocLayNet-small on Hugging Face. **The upstream revision/commit was never recorded**
  for the earlier run or this one, so the dataset version is identified only by the hash above.
- The earlier README pointed at data_external/doclaynet_small/small_dataset; the data actually present is
  under doclaynet_small_dl/extracted/.

## Command (defaults of the unchanged evaluation code; no methodology edits)
    uv run --no-sync python -m doc2graph.eval_multipage \
      --root data_external/doclaynet_small_dl/extracted/small_dataset \
      --cv 5 --bootstrap 1000 --seed 0 --min-iou 0.5 \
      --out <file>.json
The original run's arguments were not stored; `--cv 5` was inferred from the 5-fold table in
results/doclaynet_eval.md and the defaults were assumed for the rest.

## Result
Headline proposed_page_numbers: P 0.782, R 0.861, F1 0.841 [0.778, 0.895].
The re-run's full output parsed to JSON identical to results/doclaynet_eval.json (parsed
comparison, not a byte comparison; sha256 of the re-run file 91fa91d228b4bfd6326a792a785442167806d9dd4b7579800db6898affc84ca4).
The re-run copies were deleted as redundant duplicates; results/doclaynet_eval.json is the record.
