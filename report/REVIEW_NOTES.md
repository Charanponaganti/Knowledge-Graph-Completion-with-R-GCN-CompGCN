# Report revision and verification

The final report uses the supplied `acl2023.docx` page geometry, two-column layout, Times New Roman typography, title block and page-number footer. Review line numbering is omitted. The original reference was not modified. Captions use the template's 10-point typography in normal text flow because its floating caption styles displaced captions from their tables. The final PDF is exported by Microsoft Word; the packaged LibreOffice renderer could not run because this Windows runtime has no bundled LibreOffice.

## Deliverables and sources

- `Knowledge_Graph_Completion_Report.docx`: editable report based on the supplied Word template.
- `Knowledge_Graph_Completion_Report.pdf`: visually reviewed 12-page PDF, including references.
- `report_content.md`: revised narrative.
- `build_report.py`: generates the Word report and alternative LaTeX source, with result tables populated from the root `results.json`.
- `latex/main.pdf`: same final Word-exported PDF at the repository's existing report location.
- `latex/main.tex`: equivalent narrative and tables in the existing ACL LaTeX style. A local TeX compiler was unavailable, so LaTeX compilation and its pagination are not certified; the 12-page requirement is verified for the Word-exported PDF.

## Updated repository review on 7 November 2026

Reviewed the merged repository at `3ba4528`, including the long-run export introduced in `533fba6`. Comparing the executable notebook cells with the preceding version yields identical Python ASTs: the notebook restructuring changes presentation, not model/training/evaluation logic. The `src` modules and requirements are unchanged. The notebook has no saved execution outputs, and its checked-in `TIME_BUDGET` remains `fast`, while the exported run records `long`. The report explicitly instructs reproduction with `long`.

The revised results use 12 epochs × 600 steps, 15,000 local test triples, 2,000 validation triples, and 6 × 300-step ablations. CompGCN-mult now has local MRR 0.6307 and OGB MRR 0.6201. One layer is the best recorded ablation. The degree diagnostic peaks in the third group rather than the highest-degree group; updated interval boundaries are read directly from the JSON. Correlation's near-tie-level output is described as a diagnostic clue, not a confirmed score-collapse diagnosis. Updated figures and all result exports are used together.

## Substantive corrections

1. Local evaluation calls the type-aware corruption helper; it is not an unconstrained candidate test with the recorded flags.
2. Both test protocols use 500 sampled negatives, but have different test populations, negative lists, filtering and tie handling.
3. Equal configured training updates are not equal compute or equal wall time.
4. A configured epoch in the saved long run is 600 sampled batches, not one pass through all training triples.
5. Ablations allow 1,800 updates, one quarter of the main 7,200-update maximum.
6. The error-analysis code computes reciprocal mean directional rank, not standard bidirectional MRR. Its saved values are relabelled, not recomputed or fabricated.
7. All seven models have the same MRR ordering across the two protocols in the new long-run export; DistMult now leads the shallow baselines.
8. CompGCN multiplication also has a different relation initialisation, so operator choice is not the only experimental difference.
9. The active self-adversarial loss does not use the label-smoothing constant.
10. Validation in the current notebook uses 100 negatives and tail prediction. The saved curve's 500-candidate label is stale and that chart is not reproduced in the revised report.
11. Spark computes degree counts, but degree summary statistics are calculated after collecting the column into pandas.
12. The two Spark timing sweeps measure different operations and do not measure total ETL or cluster scalability.
13. Unsupported peak-memory, CPU-saturation, universal model-ranking and embedding-type-cluster claims are removed.
14. The report records the negative replacement type-boundary issue, resume-tag limitations and missing per-query ranks as limitations; model code and saved results are unchanged.

## Originality and AI-text review

The prose was rewritten specifically from the repository, with explanations and attributed background material. No external paper prose was pasted into the report. Review covered formulaic language, unsupported claims, citations, numerical consistency and layout.

No institutional plagiarism database or AI detector was run. No similarity percentage or AI percentage has been measured. This is an AI-assisted revision; it cannot truthfully be certified as containing no AI-generated content or guaranteed to score below 10 percent on a detector. A formal similarity check requires the institution's approved service, and any authorship declaration must reflect the assistance actually used.

## Verification scope

The saved main and OGB CSV metrics, ablation JSON and Spark CSV are reconciled against `results.json`. Final tables are generated from that JSON. All 12 rendered pages are visually reviewed for readable tables, caption placement, page numbers and overflow. The original template hash is retained in the task-local QA record and its unchanged package parts are checked against the output. Training was not rerun; saved outputs are internally reconciled, not independently reproduced.
