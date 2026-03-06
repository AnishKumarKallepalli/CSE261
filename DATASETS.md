# Datasets: local and which ones we use

## Which Twitter dataset we use (the best one)

We use **[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)** (CARER dataset, Saravia et al. 2018). It is the standard for 6-class emotion on Twitter:

- **6 labels:** sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5)
- **Splits:** 16k train, 2k validation, 2k test (config `split`); well-defined and widely used in papers
- **Citation:** CARER paper (ACL/EMNLP); same dataset used in many Hugging Face emotion models

The Kaggle options you mentioned are alternatives; we stick with dair-ai/emotion so the code and report match the standard benchmark.

**Reddit:** We use **GoEmotions** (Google, 27 emotions + neutral) from [google-research-datasets/go_emotions](https://huggingface.co/datasets/google-research-datasets/go_emotions).

---

## Run fully locally (download once, then offline)

1. **One-time download (with internet):**
   ```powershell
   cd "C:\Users\anish\Desktop\UCSD\Masters\Winter 2026\CSE 291\Project\Code"
   python download_data.py
   ```
   This downloads both datasets into **`Code/data/`** and builds **`Code/data/data_cache.json`**.

2. **After that, everything is local.** Run the pipeline with no network:
   ```powershell
   python run_all.py
   ```

---

## Where things are stored

| What | Where |
|------|--------|
| Raw GoEmotions (HF cache) | `Code/data/go_emotions/` |
| Raw Twitter emotion (HF cache) | `Code/data/emotion/` |
| Processed data (used by all scripts) | `Code/data/data_cache.json` |
| Models, results, figures | `Code/outputs/` |

So **all dataset-related files** are under **`Code/data/`**. Copy or back up that folder to keep the datasets locally elsewhere.

---

## If you already have `outputs/data_cache.json`

The code still finds it: `get_cached()` looks in `data/data_cache.json` first, then `outputs/data_cache.json`. To move to the new layout, run `python download_data.py` once; it will write the cache to `data/` and fill `data/go_emotions/` and `data/emotion/` for future use.
