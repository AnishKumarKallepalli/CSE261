"""
Config for CSE 261: Cross-Platform Emotion Network Generalization.
Paths, dataset names, label mapping (GoEmotions 27+neutral -> 6 labels), seed.
"""

import os

# Reproducibility
RANDOM_SEED = 42

# Fast test run: limit samples per split so the pipeline runs in minutes. Set to False for full run.
USE_SAMPLE_LIMIT = True
SAMPLE_LIMIT = 100  # max samples per split (train/val/test) when USE_SAMPLE_LIMIT is True

# Paths (relative to Code folder)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# All raw and processed data under DATA_DIR so the project runs fully locally
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Local dataset cache (HF downloads go here; then we use data_cache.json for the pipeline)
GO_EMOTIONS_CACHE = os.path.join(DATA_DIR, "go_emotions")
EMOTION_CACHE = os.path.join(DATA_DIR, "emotion")

# Datasets (HuggingFace names; we use dair-ai/emotion as the standard Twitter emotion dataset)
GO_EMOTIONS_NAME = "google-research-datasets/go_emotions"
TWITTER_EMOTION_NAME = "dair-ai/emotion"
# Twitter emotion split: "split" gives train/val/test; "unsplit" is one big split
TWITTER_EMOTION_CONFIG = "split"

# 6-label space (same order as dair-ai/emotion: sadness, joy, love, anger, fear, surprise)
SIX_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
SIX_LABEL2ID = {l: i for i, l in enumerate(SIX_LABELS)}

# GoEmotions 27 emotion names (order from dataset)
GO_EMOTIONS_27 = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise"
]
# neutral is index 28 in GoEmotions (we skip or map to a default)

# Mapping: each GoEmotions (27) label -> 6-label class
# Used for training Reddit models in 6-class mode and evaluating on Twitter.
GO_EMOTIONS_TO_SIX = {
    "admiration": "joy",
    "amusement": "joy",
    "anger": "anger",
    "annoyance": "anger",
    "approval": "joy",
    "caring": "love",
    "confusion": "surprise",
    "curiosity": "surprise",
    "desire": "love",
    "disappointment": "sadness",
    "disapproval": "anger",
    "disgust": "anger",
    "embarrassment": "sadness",
    "excitement": "joy",
    "fear": "fear",
    "gratitude": "joy",
    "grief": "sadness",
    "joy": "joy",
    "love": "love",
    "nervousness": "fear",
    "optimism": "joy",
    "pride": "joy",
    "realization": "surprise",
    "relief": "joy",
    "remorse": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
}

# BERT / RoBERTa
BERT_MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 32
BERT_EPOCHS = 3
BERT_LR = 2e-5

# Baseline embedding (for embed+NB/SGD): use a small pre-trained or simple average
# We use sklearn's TfidfVectorizer for TF-IDF; for "embed" we use same TF-IDF as 300-dim then SGD (simpler, no gensim)
USE_EMBED_MODEL = "tfidf"  # "tfidf" = no extra embed; set "glove" if you add GloVe later

# Zero-shot Qwen (emotion classification)
QWEN_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # Use 7B or larger for better quality if GPU allows
