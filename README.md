# Knowledge Graph Completion with R-GCN / CompGCN

Link prediction on large KGs with a **PySpark** ETL stage and **PyTorch Geometric** encoders (R-GCN, CompGCN). Written for a Big Data course: the notebook is meant to run on a **Kaggle T4 (16 GB)** and also on a local **RTX 4050 (typically 6 GB)**.

This repository does **not** contain experimental scores. Run the notebook, then fill the report from `outputs/results.json`.

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
| `kaggle_t4` | NVIDIA T4 | 16 GB | 1024 | 64 | [15, 10] | 12g |
| `rtx4050` | GeForce RTX 4050 | ~6 GB | 256 | 32 | [8, 8] | 8g |
| `cpu` | — | — | 128 | 32 | [5, 5] | 8g |

`MACHINE = "auto"` reads the live GPU name and VRAM. A 4050 is treated as memory-tight: smaller embeddings, fewer negatives, smaller eval candidate sets. **CompGCN-corr (FFT) is the first thing to OOM** on 6 GB; the notebook skips it rather than crash.

Set `QUICK_RUN = True` until the pipeline finishes. Only then set it `False` for the tables in the report.

## Files

```
notebooks/kg_completion_rgcn_compgcn.ipynb   # upload this to Kaggle (self-contained)
src/spark_etl.py  models.py  train.py  evaluate.py   # mirrors for reading / viva; notebook does not import them
requirements.txt
report/FORMAT_NOTES.md          # extracted ACL 2023 Word rules
report/RESULTS_TO_FILL.md       # checklist after the run
report/build_report.py          # writes report/kg_completion_report.docx (you run it)
report/figures/                 # copy PNGs here from Kaggle
acl2023.docx                    # uploaded course template (unchanged)
```

**Report format:** the template is the ACL 2023 **Word** file, so the report is generated as `.docx` (not LaTeX). Geometry: A4, two-column, 2.5 cm margins, Times New Roman, author–year citations, mandatory Limitations section. See `report/FORMAT_NOTES.md`.

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
7. Download `/kaggle/working/outputs/results.json` and `/kaggle/working/figures/*.png`.

Expected wall-clock (order of magnitude, not a promise):

| Mode | Dataset | Machine | Time |
|---|---|---|---|
| `QUICK_RUN=True` | FB15k-237 / synthetic | T4 or 4050 | 5–20 min |
| Full | ogbl-biokg | T4 16 GB | several hours (stay under 12 h; 8–15 epochs) |
| Full | ogbl-biokg | RTX 4050 6 GB | slower than T4; fewer epochs (preset uses 8) |
| Full | Wikidata5M subset | T4 | Spark minutes + train hours |

## Run locally (RTX 4050)

You need **Java 11+** (Spark), a CUDA PyTorch that matches your driver, then:

```text
pip install -r requirements.txt
```

Open `notebooks/kg_completion_rgcn_compgcn.ipynb`, keep `MACHINE = "auto"`. Data lands in `./outputs` and figures in `report/figures/`.

Windows + Spark sometimes wants `JAVA_HOME` and can complain about Hadoop native libs. If Spark fails locally, run the Spark+train notebook on Kaggle; the 4050 is still useful for `fb15k237` / `QUICK_RUN` model debug.

## Build the Word report (you run this)

```text
pip install python-docx
# put Kaggle PNGs in report/figures/ and results.json in outputs/
python report/build_report.py
```

Edit `AUTHOR`, `AFFILIATION`, `EMAIL` at the top of `report/build_report.py`. Replace any leftover `[[INSERT RESULT: ...]]` using `report/RESULTS_TO_FILL.md`. Do not type made-up MRR/Hits numbers.

## OOM / troubleshooting

- **CUDA OOM:** the train loop already falls back to batch 64 / fanout [4] / dim 16. If it still dies, set `DATASET = "fb15k237"`, `QUICK_RUN = True`, or `COMPGCN_OPS = ["mult"]` (drop `corr`).
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
