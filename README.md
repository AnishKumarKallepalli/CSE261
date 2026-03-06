# Cross-Platform Emotion Network Generalization (CSE 261)

Pipeline: data + label alignment → train Reddit models (TF-IDF+LR, TF-IDF+SGD, BERT) → LLM baseline → evaluate → emotion networks → figures.

---

## Quick start

```powershell
cd Code
pip install -r requirements.txt
python download_data.py
python run_all.py
```

Figures and results: `outputs/figures/`, `outputs/results/`.

---

## Commands (in order)

### 1. Setup

```powershell
cd Code
pip install -r requirements.txt
```

### 2. Get data (one time, needs internet)

```powershell
python download_data.py
```

Stores data under `data/` (GoEmotions, Twitter emotion, `data_cache.json`). After this, the rest can run offline.

### 3. Fast test run (small data)

In `config.py` set:

- `USE_SAMPLE_LIMIT = True`
- `SAMPLE_LIMIT = 100`  (or `10` for a very quick test)

Then:

```powershell
python run_all.py
```

### 4. Full run (all data)

In `config.py` set:

- `USE_SAMPLE_LIMIT = False`

Then:

```powershell
python run_all.py
```

### 5. Run a single step

From `Code/`:

```powershell
python data_load.py
python train_baselines.py
python train_bert.py
python eval_llm.py
python evaluate.py
python build_networks.py
python analyze_networks.py
python save_figures.py
```

### 6. Compile the report (Paper folder)

From project root:

```powershell
cd Paper
pdflatex research_paper
bibtex research_paper
pdflatex research_paper
pdflatex research_paper
```

Open `Paper/research_paper.pdf`. See `Paper/README.md` for other options (VS Code, Overleaf).

### 7. Git

Repo is under `Code/`. `data/` and `outputs/` are in `.gitignore` (long paths and large files; data is re-downloaded with `download_data.py`).

```powershell
cd Code
git init
git add .
git status
git commit -m "CSE 261 pipeline and report"
```

To push to a remote later:

```powershell
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

---

## What gets created

| Path | Contents |
|------|----------|
| `data/` | Datasets and `data_cache.json` (after `download_data.py`) |
| `outputs/models/` | TF-IDF+LR, TF-IDF+SGD, BERT, Twitter baseline |
| `outputs/results/` | `evaluation_results.json`, `network_analysis.json`, `llm_eval.json`, edge lists |
| `outputs/figures/` | `confusion_matrix.png`, `network_reddit.png`, `network_twitter.png` |
| `outputs/network_*.pkl` | NetworkX graphs |

---

## Config (`config.py`)

- **`USE_SAMPLE_LIMIT`** — `True` = limit samples (fast test), `False` = full data.
- **`SAMPLE_LIMIT`** — Max samples per split when `USE_SAMPLE_LIMIT` is True (e.g. `100` or `10`).
- **`BERT_EPOCHS`**, **`BATCH_SIZE`** — BERT training; reduce for faster runs.

---

## Label mapping

GoEmotions (27 + neutral) → 6 classes: **sadness, joy, love, anger, fear, surprise** (same as dair-ai/emotion). Defined in `config.GO_EMOTIONS_TO_SIX`. See **DATASETS.md** for dataset details.

---

## Report

Report source: **`../Paper/research_paper.tex`**. Compile in `Paper/` (see step 6). Fill tables using numbers from `outputs/results/evaluation_results.json` and `outputs/results/network_analysis.json`.
