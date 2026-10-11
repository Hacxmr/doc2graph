# Pilot annotation guidelines

Annotate relationships between text blocks from different pages of the same
document. Use the document as context. Do not infer a relationship from
embedding similarity alone. Label each directed source/target pair once, with
source on the earlier page and target on the later page.

## Labels

- **`CONTINUES`**: the later block is the direct textual or structural
  continuation of the earlier block. This includes a sentence or paragraph
  split across pages. A new paragraph on the same topic is not automatically a
  continuation.
- **`SAME_TABLE`**: blocks are parts of the same table continued across pages.
  Consider repeated column headers, table numbering/caption, row sequence, and
  visual structure together. Similar tables elsewhere in the document do not
  qualify.
- **`REFERS_TO`**: the source block explicitly refers to the target block or
  object, such as a figure, table, section, or page content. Record the
  direction of the reference. A coincidental shared number is insufficient.
- **`NONE`**: none of the defined relationships applies.
- **`UNCERTAIN`**: available document context is insufficient to decide. Keep
  this separate from `NONE`; adjudicate or exclude uncertain pairs from the
  primary metric and report their count.

Relations may co-occur when the evidence supports each one independently.
Repeated headers, footers, identifiers, and repeated text can also receive a
separate `SAME_ENTITY` annotation; do not treat that label as a substitute for
one of the three typed relations.

## Annotation procedure

1. Review all pages of the document in order, not only the candidate pair.
2. Label plausible cross-page pairs and a sample of generated candidates.
   Preserve the system candidate set and candidate scores separately from
   labels so candidate recall can be calculated.
3. Record a short evidence note and the relevant page/block IDs for every
   positive label. For `NONE`, record the most plausible alternative when
   helpful.
4. Have a second annotator independently label a subset before discussion.
   Report agreement per relation (and on `NONE`/`UNCERTAIN`) before
   adjudication; describe how disagreements were resolved.
5. Keep document-level splits intact for any tuning and held-out evaluation.

## Suggested annotation fields

Use one row per source/target pair and relation label:

`document_id, source_page, source_block_id, target_page, target_block_id, relation, evidence, annotator, confidence, adjudication`

Use stable block IDs from the extracted document representation. Record the
dataset version and extraction version because block IDs can change if parsing
changes. Sample documents to cover the relation types and document layouts;
report the sampling criteria and counts rather than presenting the pilot as a
representative corpus without evidence.
