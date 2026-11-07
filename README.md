# Knowledge Graph Completion with R-GCN / CompGCN

Link prediction on large KGs with a **PySpark** ETL stage and **PyTorch Geometric** encoders (R-GCN, CompGCN). Written for a Big Data course: the notebook is meant to run on a **Kaggle T4 (16 GB)** and also on a local **RTX 4050 (typically 6 GB)**.

Scores in the report come only from `outputs/results.json` of an executed run. Run the notebook, then update the report from that file.

## Dataset choice

| Switch `DATASET =` | Role | Why |
|---|---|---|
| `biokg` **(default)** | Main OGB benchmark | Largest OGB KG that is honestly trainable with neighbor sampling on a T4. Entity table is tens of MB, not a gigabyte. Official evaluator exists. |
| `wikidata5m_subset` | Large-scale Spark job | Full Wikidata5M / ogbl-wikikg2 **will not** fit full-batch R-GCN on a T4 or a 4050. Spark degree + k-core filtering builds a subgraph that does. |
| `fb15k237` | Smoke test | Minutes, not hours. Use this first. |

`ogbl-wikikg2` is discussed in the report table and **not** trained end-to-end.

## Hardware presets (`MACHINE = "auto"`)

| Preset | GPU | VRAM | Batch | Dim | Fanout | Spark driver |
|---|---|---|---|---|---|---|
| `kaggle_t4` | NVIDIA T4 | 16 GB | 512 | 128 | [10, 10] | 12g |
| `rtx4050` | GeForce RTX 4050 | ~6 GB | 256 | 64 | [8, 8] | 8g |
| `cpu` | — | — | 128 | 32 | [5, 5] | 8g |

`MACHINE = "auto"` reads the live GPU name and VRAM. A 4050 is treated as memory-tight: smaller embeddings, fewer negatives, smaller eval candidate sets.

**All five models run on every machine:** DistMult, R-GCN, CompGCN-sub, CompGCN-mult, CompGCN-corr. CompGCN composition (including the corr FFT) runs in float32 with autocast off. Each model is isolated: an OOM retries once with half batch and fanout, and any other failure is recorded while the next model still runs.

Set `QUICK_RUN = True` first (~15–25 min, all five models for one epoch). Then set it `False` for the report tables.

Default `TIME_BUDGET = 'fast'` takes **about 3–5 hours on a T4**, including R-GCN ablations and the official OGB evaluation of every model. `'long'` trains longer. New work stops at `TIME_LIMIT_H = 11` regardless.

### What the training and evaluation do

- Every step trains **tail and head prediction**. GNN query embeddings add the base entity vector, so they share a space with the candidate table.
- `MATCHED_DISTMULT = True`: DistMult uses the same epochs / steps / negatives as the GNNs (fairer comparison).
- `ADV_NEG = True`: RotatE-style self-adversarial negative weighting.
- `TYPE_AWARE_NEG = True` on `ogbl-biokg`: training negatives stay inside the same biomedical entity type.
- After the five main models, optional shallow decoder ablations: TransE and ComplEx (`EXTRA_DECODERS`).
- Filtered ranking uses Spark-built int64 keys of every true triple, searched on the GPU. Every model sees the same candidates (fixed seed).
- The official `ogbl-biokg` evaluator runs for every model, with head and tail negatives, on the **original OGB test order**. Earlier versions paired OGB negatives with Spark-reordered test rows; do not reuse those OGB numbers.

### Kaggle restarts

Results are saved after every model (`outputs/results_partial.json`, `outputs/<model>.pt`, `outputs/ablation.json`). If a session dies, keep `RESUME = True` and Run All again: finished models are reloaded, not retrained.

## Files

```
notebooks/final.ipynb                       # executed long-run Kaggle copy
notebooks/kg_completion_rgcn_compgcn.ipynb   # general-purpose self-contained notebook
src/spark_etl.py  models.py  train.py  evaluate.py   # mirrors for reading / viva; notebook does not import them
requirements.txt
report/latex/                   # ACL 2023 LaTeX report (main.tex, custom.bib, compile.bat, main.pdf)
report/figures/                 # PNGs from the Kaggle run
```

**Report format:** ACL 2023 LaTeX (`report/latex`): A4, two-column, Times, author–year citations, mandatory Limitations section. Build with `report/latex/compile.bat` or Overleaf.

## Run on Kaggle (assignment)

1. New notebook, **GPU T4**, persistence off or on as you like.
2. **Internet ON** for the first session (pip + OGB download).
3. Copy `notebooks/kg_completion_rgcn_compgcn.ipynb` into the editor (or upload it).
4. Optional datasets under `/kaggle/input`:
   - OGB cache containing `ogbl_biokg` (so later sessions can use internet **OFF**)
   - Wikidata5M transductive `train.txt` / `valid.txt` / `test.txt` (or the KEPLER filenames)
   - FB15k-237 `train.txt` `valid.txt` `test.txt`
5. Cell 2: `QUICK_RUN = True`, `DATASET = "fb15k237"` or `"biokg"`. Run all.
6. If that works: `QUICK_RUN = False`, `DATASET = "biokg"`, full run. Spark **stops itself** before training so RAM is free.
7. Download `/kaggle/working/kgc_report_artifacts.zip` (last cell): `results.json`, `main_results.csv`, `ogb_results.csv`, `ablation.json`, figures. The run is complete when the payload cell prints `ALL MODELS OK`.

Expected wall-clock (order of magnitude, not a promise):

| Mode | Dataset | Machine | Time |
|---|---|---|---|
| `QUICK_RUN=True` | FB15k-237 / biokg slice | T4 or 4050 | ~15–25 min |
| `TIME_BUDGET='fast'` (default) | ogbl-biokg | T4 16 GB | **~3–5 hours** |
| `TIME_BUDGET='fast'` | ogbl-biokg | RTX 4050 6 GB | **~4–6 hours** |
| `TIME_BUDGET='long'` | ogbl-biokg | T4 | ~6–9 hours, capped by `TIME_LIMIT_H = 11` |

## Run locally (RTX 4050)

You need **Java 11+** (Spark), a CUDA PyTorch that matches your driver, then:

```text
pip install -r requirements.txt
```

Open `notebooks/kg_completion_rgcn_compgcn.ipynb`, keep `MACHINE = "auto"`. Data lands in `./outputs` and figures in `report/figures/`.

Windows + Spark sometimes wants `JAVA_HOME` and can complain about Hadoop native libs. If Spark fails locally, run the Spark+train notebook on Kaggle; the 4050 is still useful for `fb15k237` / `QUICK_RUN` model debug.

## Build the report

The revised report is available in `report/Knowledge_Graph_Completion_Report.docx`
and `report/Knowledge_Graph_Completion_Report.pdf` (12 pages including references),
using the supplied ACL Word format. See `report/REVIEW_NOTES.md` for the metric
and methodology corrections. Its tables are generated from the saved root
`results.json`; the experiments were not rerun for this editorial revision.
The current report covers the saved **long** T4 run (12 × 600 steps; 15,000
local test triples). The executed, long-run Kaggle notebook is
`notebooks/final.ipynb`; `notebooks/kg_completion_rgcn_compgcn.ipynb` remains
the general-purpose notebook with its `fast` default.

To regenerate the editable report and alternative LaTeX source:

```text
python report/build_report.py --template "C:\Users\Charan\Downloads\acl2023.docx"
```

The builder requires `python-docx`. Export the DOCX through Word to update the
PDF and check pagination after content changes. The existing `report/latex/main.pdf`
is a copy of the final Word export; the alternative LaTeX build has not been
pagination-verified locally.

For a new experiment, replace the saved result exports and copy its PNGs into
`report/figures/`, then regenerate the report and review the narrative against
the new run. Do not type made-up MRR/Hits numbers or assume old conclusions
still apply. `report/latex/compile.bat` builds the alternative LaTeX version.

## OOM / troubleshooting

- **CUDA OOM:** the train loop halves batch and fanout on OOM, and each model gets one retry at half size. If a model still fails, the others keep running and the failure is listed at the end.
- **CPU RAM OOM during Spark:** lower `SPARK_DRIVER_MEMORY` in the preset, call `spark.stop()` (already at the end of ETL), delete `spark-tmp`.
- **OGB download / no internet:** attach files under `/kaggle/input`; loaders search that tree before calling the downloader.
- **No Java:** Spark will not start. Kaggle images usually have it; locally install a JDK and set `JAVA_HOME`.
- **torch-scatter install fails:** the notebook continues; the custom CSR sampler does not require it. `RGCNConv` still needs a working `torch-geometric`.
- **Synthetic data warning:** if you see it, no real KG was found. Do not put those scores in the report.

## Viva map

- Why not Wikidata5M full-batch? Embedding + 20M-edge GNN > 16 GB, and far above 6 GB.
- Where is Spark? Ingest, clean, stats, k-core, dense ids, inverse relations, filtered lookup Parquet, scaling plot.
- Difference R-GCN vs CompGCN? Relation-specific `W_r` vs composition `φ(h, r)` then `W_λ(r)`.
- DistMult? Decoder `⟨h,r,t⟩` and the no-GNN baseline.
- 4050 vs T4? Same code, different preset; 4050 is not a second method.
