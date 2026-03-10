# Cross-Platform Emotion Network Generalization (CSE 261)

**Anish Kallepalli and Akshar Tumu — UC San Diego**

This repository implements the full pipeline for the project: *Cross-Platform Emotion Network Generalization Using Social Media Text*. We train emotion classifiers on Reddit (GoEmotions), evaluate cross-platform transfer to Twitter, and build emotion co-occurrence networks to explain generalization gaps.

---

## Table of contents

- [Quick start](#quick-start)
- [What is implemented](#what-is-implemented)
- [Pipeline steps (run_all.py)](#pipeline-steps-run_allpy)
- [Data and label mapping](#data-and-label-mapping)
- [Outputs](#outputs)
- [Config](#config)
- [Word clouds (per emotion)](#word-clouds-per-emotion)
- [Report and figures](#report-and-figures)

---

## Quick start

```powershell
cd Code
pip install -r requirements.txt
python download_data.py
python run_all.py
```

Results and figures: `outputs/results/`, `outputs/figures/`.

For a fast test run (small data), set in `config.py`: `USE_SAMPLE_LIMIT = True`, `SAMPLE_LIMIT = 100`, then run `python run_all.py`.

---

## What is implemented

### Data and preprocessing

- **download_data.py** — One-time download of GoEmotions (Reddit) and dair-ai/emotion (Twitter) from HuggingFace into `data/`. Builds `data_cache.json` with 6-class aligned labels so the rest of the pipeline runs offline.
- **data_load.py** — Loads both datasets; maps GoEmotions (27 + neutral) → 6 labels (sadness, joy, love, anger, fear, surprise); optional sample limiting for quick runs. Exposes `get_cached()` for other scripts to read from `data_cache.json`.

### Models (trained on Reddit unless noted)

| Component | Script | Description |
|-----------|--------|-------------|
| **TF-IDF + Logistic Regression** | train_baselines.py | Classical baseline on Reddit; also trains in-domain Twitter TF-IDF+LR for upper bound. |
| **TF-IDF + SGD** | train_baselines.py | Alternative classical classifier (TF-IDF + SGDClassifier). |
| **BERT embeddings + LR/SGD** | train_bert_embeds.py | Frozen BERT (bert-base-uncased) → pooled output → LogisticRegression or SGDClassifier. |
| **Fine-tuned BERT** | train_bert.py | Full fine-tuning of BERT on Reddit (HuggingFace Trainer, multi-GPU via accelerate). |
| **LLM baseline** | eval_llm.py | Instruction-tuned LLM used as a training-free classifier (evaluated on Reddit val and Twitter test). |
| **Zero-shot Qwen** | eval_qwen_zero_shot.py | Qwen2.5-1.5B-Instruct zero-shot emotion classification (Reddit val, Twitter test). |

### Evaluation and analysis

- **evaluate.py** — Runs all saved models on Reddit test and Twitter test (ground truth labels); writes accuracy, macro-F1, micro-F1, per-class F1, confusion matrices, and predictions to `outputs/results/evaluation_results.json`.
- **build_networks.py** — Builds emotion co-occurrence graphs: Reddit from gold multi-labels; Twitter from predicted multi-labels (BERT or TF-IDF). Edge weights use PMI. Saves NetworkX graphs and edge lists.
- **analyze_networks.py** — Computes density, degree centrality, PageRank, modularity, communities; identifies top edges and confused emotion pairs on Twitter. Saves `network_analysis.json` and `network_*_edges.json`.
- **save_figures.py** — Saves confusion matrices (Reddit/Twitter) and network visualizations to `outputs/figures/`.

### Word clouds per emotion

- **wordcloud_emotions.py** — Generates one word cloud per dataset (Reddit and Twitter) to show overall vocabulary / emotion mix per platform. Saves to `outputs/figures/wordclouds/`. See [Word clouds](#word-clouds).

---

## Pipeline steps (run_all.py)

The full pipeline runs these scripts in order:

1. **data_load.py** — Load data and label alignment.
2. **train_baselines.py** — Train TF-IDF+LR and TF-IDF+SGD (Reddit); Twitter in-domain TF-IDF+LR.
3. **train_bert_embeds.py** — BERT embeddings + LR/SGD baselines.
4. **train_bert.py** — Fine-tune BERT on Reddit (via `accelerate launch` if multiple GPUs).
5. **eval_llm.py** — LLM/keyword baseline.
6. **eval_qwen_zero_shot.py** — Zero-shot Qwen emotion classification.
7. **evaluate.py** — Evaluate all models on Reddit & Twitter test sets.
8. **build_networks.py** — Build emotion networks.
9. **analyze_networks.py** — Network analysis (centrality, communities, confused pairs).
10. **save_figures.py** — Save confusion matrices and network figures.

Run a single step from `Code/`:

```powershell
python data_load.py
python train_baselines.py
# ... etc
python save_figures.py
```

---

## Data and label mapping

- **Reddit**: GoEmotions (27 emotions + neutral). We map to **6 classes** (same as dair-ai/emotion): **sadness, joy, love, anger, fear, surprise**. Mapping is in `config.GO_EMOTIONS_TO_SIX` (e.g. admiration/amusement/approval → joy; grief/remorse/sadness → sadness).
- **Twitter**: dair-ai/emotion with the same 6 labels; no mapping needed.

Splits: both datasets provide train / validation / test. We use Reddit for training and validation; Twitter for in-domain training (upper bound) and for cross-domain evaluation (Reddit-trained models on Twitter test).

---

## Outputs

| Path | Contents |
|------|----------|
| `data/` | GoEmotions cache, emotion (Twitter) cache, `data_cache.json`. |
| `outputs/models/` | TF-IDF+LR, TF-IDF+SGD, BERT fine-tuned, BERT-embed pipelines; Twitter TF-IDF+LR; BERT checkpoints (e.g. `bert_reddit_run/checkpoint-*`). |
| `outputs/results/` | `evaluation_results.json`, `network_analysis.json`, `llm_eval.json`, `qwen_zero_shot.json`, `network_reddit_edges.json`, `network_twitter_edges.json`. |
| `outputs/figures/` | `confusion_matrix_reddit.png`, `confusion_matrix_twitter.png`, `network_reddit.png`, `network_twitter.png`; optional `wordclouds/` (see below). |
| `outputs/network_*.pkl` | NetworkX graphs (Reddit, Twitter). |

---

## Config

In **config.py**:

- **USE_SAMPLE_LIMIT** — `True` = limit samples (fast test), `False` = full data.
- **SAMPLE_LIMIT** — Max samples per split when `USE_SAMPLE_LIMIT` is True (e.g. 100 or 10).
- **BERT_EPOCHS**, **BATCH_SIZE**, **BERT_LR** — BERT training.
- **SIX_LABELS**, **GO_EMOTIONS_TO_SIX** — Label set and Reddit→6-class mapping.
- **QWEN_MODEL_NAME** — Zero-shot model (e.g. Qwen2.5-1.5B-Instruct).

---

## Word clouds

We generate **one word cloud per dataset** (Reddit and Twitter): all texts from each dataset are combined, tokenized (with stopword removal), and turned into a single word cloud so you can compare the overall vocabulary and emotion-related language between the two platforms.

### Setup

Install the wordcloud package:

```powershell
pip install wordcloud
```

### Run

From `Code/` (after at least one run of `data_load.py` or `run_all.py` so `data_cache.json` exists):

```powershell
python wordcloud_emotions.py
```

- **Input**: Uses cached data from `data_load.get_cached()` (all splits per dataset).
- **Output**: Saves under `outputs/figures/wordclouds/`:
  - `wordcloud_reddit.png` — full Reddit dataset
  - `wordcloud_twitter.png` — full Twitter dataset

Words are lowercased and filtered by a simple stopword list; you can change max words, colormap, and stopwords in the script.

---

## Report and figures

- **Paper**: Source in `../Paper/`. Compile with `pdflatex` / `bibtex` in `Paper/`. See `Paper/README.md`.
- **Numbers**: Fill tables from `outputs/results/evaluation_results.json` and `outputs/results/network_analysis.json`.
- **Slides**: Use the same results and network/confusion figures; word clouds can be used to illustrate per-emotion language per platform.

---

## Git

Repo is under `Code/`. Typical `.gitignore` includes `data/`, `outputs/` (large files and re-downloadable data). After cloning:

```powershell
cd Code
pip install -r requirements.txt
python download_data.py
python run_all.py
```
