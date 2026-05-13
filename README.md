# CS 1090B Milestone 4 - Group #99

**Sentiment Analysis and Change-Point Detection on Amazon Cell Phone Reviews**

Team (alphabetical, not contribution order): Xuekun Huang, Ghazal Jafari Naeimi, Angkai Li, Bruce Meng

`cs1090b_ms4_main_group99.ipynb` is the main notebook. `src/` holds the implementation code it imports. `cache/`, `models/`, `outputs/`, and `figures/` contain all precomputed results.

---

## How to run it

**No GPU needed.** Everything is pre-cached, the notebook loads cached results and skips all training. A cached re-run takes ~5-15 minutes.

Because the full project is several GB (raw data, HuggingFace base models, intermediate caches), it cannot fit in a 50 MB zip. We offer two delivery options.

### Option A — download the full pre-built folder (easiest)

Download the complete `1090b-ms4/` folder from Google Drive (the same folder linked in our Video Presentation submission):

[(https://drive.google.com/drive/folders/1zMSiCMCAOibAd4NY6aczk9bpJC2dnISO)]


Unzip and run the notebook, everything is already in place.

We acknowledge this is not the most-recommended path per the course rubric, since cloud links raise concerns around access permissions and post-deadline modification. **We guarantee that the Google Drive folder remains public and that no file in it will be modified, replaced, or removed after the submission deadline.**

### Option B — submitted code + supplementary download (course-compliant)

The submitted archive contains the source code, notebook, README, requirements, and the essential final model weights needed for grading. To re-run end-to-end, download the data.zip and cache.zip from Google Drive:

[(https://drive.google.com/drive/folders/1zMSiCMCAOibAd4NY6aczk9bpJC2dnISO)]

Place each item according to the **Structure** section below, then run the notebook.

### Troubleshooting

**`UnpicklingError: Weights only load failed`**: the `_SUCCESS` marker got touched during extraction. Re-extract a fresh copy.

**HuggingFace tries to download base models**: check that `cache/hf_hub` exists.

---

## Structure

```
1090b-ms4/
|-- cs1090b_ms4_main_group99.ipynb                 main notebook
|-- README.md                                      this file
|-- requirements.txt                               pinned dependencies
|
|-- src/
|   |-- config/
|   |   |-- env.py                                 HF env vars, GPU env setup
|   |   |-- hyperparams.py                         all hyperparameters (single source of truth)
|   |   |-- paths.py                               all file paths (single source of truth)
|   |   +-- runtime.py                             device detection (CUDA/BF16/FP16), random seeds
|   |
|   |-- utils/
|   |   |-- io.py                                  atomic_to_csv, atomic_to_pic helpers
|   |   |-- format.py                              table + text formatting
|   |   |-- losses.py                              FocalLoss (used by R3 TST + R4 MOMENT)
|   |   |-- metrics.py                             RMSE, weighted mean, correlation report
|   |   +-- training.py                            BestStateTracker, early stopping
|   |
|   |-- data/
|   |   |-- load.py                                load raw 5-core JSON
|   |   |-- clean.py                               clean + feature-engineer review table
|   |   |-- eda.py                                 EDA plots and summary tables
|   |   |-- timeseries.py                          daily time series construction
|   |   |-- panels.py                              monthly + weekly DeBERTa panels (Task 2 input)
|   |   +-- nlp_balanced.py                        balanced NLP subset for Task 1 training
|   |
|   |-- task1/
|   |   |-- splits.py                              shared 70/10/20 train/val/test split
|   |   |-- vader.py                               VADER rule-based baseline
|   |   |-- tfidf_ridge.py                         TF-IDF n-grams + Ridge regression baseline
|   |   |-- deberta.py                             DeBERTa-v3-base + LoRA training + inference
|   |   |-- deberta_internals.py                   LoRA config, training loop, checkpoint logic
|   |   |-- compare.py                             model comparison plots
|   |   +-- summary.py                             Task 1 output file summary
|   |
|   |-- task2/
|   |   |-- synth.py                               synthetic AR(1) window generation (R2/R3 training data)
|   |   |-- records.py                             monthly/weekly record construction
|   |   |-- pelt.py                                Route 1: PELT + CROPS-elbow + ClaSP
|   |   |-- clasp.py                               ClaSP wrapper (parameter-free CPD baseline)
|   |   |-- autocpd.py                             Route 2: 2-layer MLP + product embedding
|   |   |-- tst.py                                 Route 3: BERT-style transformer + masked pretraining
|   |   |-- moment.py                              Route 4: MOMENT-1-large + LoRA boundary head
|   |   |-- hybrid.py                              hybrid gamma-sweep decoding (PELT cost + neural priors)
|   |   |-- evaluate.py                            F1 vs PELT pseudo-GT, phase profiling, k-means
|   |   |-- legacy.py                              multivariate ruptures.Pelt ablation baseline
|   |   +-- weekly.py                              weekly pipeline mirror (Sec 9.W)
|   |
|   +-- validation/
|       |-- events.py                              EVENT_ANCHORS (5 phones) + KNOWN_EVENTS (21 events)
|       |-- hits.py                                primary single-anchor hit detection
|       |-- extended.py                            extended tolerance sweep + edge-effect filter
|       |-- fdr.py                                 BH-FDR + Bonferroni-Holm correction
|       |-- magnitude.py                           Hedges' g effect size + 95% CI
|       |-- multi_anchor.py                        21-event multi-anchor bootstrap (10,000 rounds)
|       |-- reverse.py                             reverse-direction match vs KNOWN_EVENTS
|       +-- plots.py                               4 validation plots (overlay, bootstrap, forest, bar)
|
|-- data/                                          raw 5-core JSON
|-- cache/                                         cleaned reviews, panels, splits, HF base models
|-- models/                                        Ridge, DeBERTa+LoRA, R2/R3/R4 weights
|-- outputs/                                       predictions, change-points, eval tables
+-- figures/                                       all plots
```

---

## Results

### Task 1: per-review sentiment regression (225,533 held-out test reviews)

| Model | MAE | RMSE |
|---|---|---|
| **DeBERTa-v3-base + LoRA** | **0.1053** | **0.2118** |
| TF-IDF + Ridge | 0.2146 | 0.3300 |
| VADER (rule-based) | 0.3765 | 0.5255 |

DeBERTa halves the error of TF-IDF + Ridge. Expected, but the gap is bigger than we thought.

### Task 2: product change-point detection (4 routes)

F1 vs PELT pseudo-GT at +/-4 tolerance:

| Route | Monthly F1 | Weekly F1 |
|---|---|---|
| R2 AutoCPD | **0.448** | **0.386** |
| R3 TST | 0.430 | 0.365 |
| R4 MOMENT + LoRA | 0.343 | 0.156 |

A tiny 8K-parameter MLP (R2) beats a 6.3M-trainable foundation model (R4). The 2-layer MLP also ties the 3-layer Transformer (R3). Bigger is not always better.

### Real-world event validation

5 documented smartphone events as external ground truth. 3 of 4 evaluable anchors show significant sentiment shifts (Hedges' g 95% CI excludes zero):

- Galaxy S5 dropped after Note 7's discontinuation (g = -0.77)
- Moto G2 dropped after Moto G3 launched (g = -1.34)
- Galaxy S3 Mini rose after Galaxy S5 launched (g = +1.47)



