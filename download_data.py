"""
One-time download: fetch both datasets into Code/data/ so the rest of the pipeline runs locally (offline).

Run once with internet:
    python download_data.py

After this, Code/data/ will contain:
  - data/go_emotions/   (GoEmotions Reddit, from Hugging Face)
  - data/emotion/      (Twitter emotion, dair-ai/emotion - CARER dataset)
  - data/data_cache.json (processed texts + 6-class labels for the pipeline)

Then run_all.py (or any script) uses only these local files; no network needed.
"""

import os
import sys

# Ensure we run from Code directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import config
from data_load import load_all

if __name__ == "__main__":
    print("Downloading datasets into:", config.DATA_DIR)
    print("  GoEmotions ->", config.GO_EMOTIONS_CACHE)
    print("  Twitter emotion (dair-ai/emotion) ->", config.EMOTION_CACHE)
    print("(Requires internet. After this, the pipeline runs locally.)\n")
    reddit, twitter = load_all()
    print("\nDone. Data is in Code/data/. You can run run_all.py offline.")
