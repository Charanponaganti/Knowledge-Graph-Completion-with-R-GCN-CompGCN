# Knowledge Graph Completion with R-GCN and CompGCN

Big Data course project, IIT Guwahati.

Charan Ponaganti (240150024), M Charan (240150018), Lalit Deshmane (240150044)

We do link prediction on the `ogbl-biokg` knowledge graph. PySpark is used to clean and
index the data, and then we train DistMult, TransE, ComplEx, R-GCN and CompGCN
(sub, mult, corr) and compare them with the official OGB evaluator.

The code is in one notebook: `notebooks/final.ipynb`. It already has the outputs of our
final run saved in it.

## How to run (Kaggle)

1. Create a new Kaggle notebook and upload `notebooks/final.ipynb`.
2. Settings: Accelerator = **GPU T4**, Internet = **On** (needed to install packages and
   download the dataset).
3. In the config cell you can change:
   - `TIME_BUDGET = 'long'` (our final run, about 1.8 hours on a T4) or `'fast'` (shorter)
   - `QUICK_RUN = True` for a quick test run first
4. Click **Run All**.
5. Results, tables and plots are printed in the notebook. Everything is also saved in
   `/kaggle/working/outputs/`, and the last cell makes a zip file you can download.

If the Kaggle session restarts, keep `RESUME = True` and run all again. Models that
already finished are loaded instead of trained again.

## How to run locally

You need Python 3.10+, Java 11+ (for Spark) and PyTorch with CUDA.

```
pip install -r requirements.txt
jupyter notebook notebooks/final.ipynb
```

Then run all cells. A GPU is strongly recommended.
