"""
Step 1: Load GoEmotions (Reddit) and Twitter emotion datasets; apply label mapping to 6-class.
Returns (texts, labels, splits) for training and evaluation.
"""

import os
import json
import numpy as np
from datasets import load_dataset
import config

# GoEmotions has 28 labels: 27 emotions + neutral (index 28)
NEUTRAL_IDX_GO = 27


def _go_emotion_ids_to_six(emotion_ids, emotion_names_27):
    """Map list of GoEmotions indices to single 6-class label. Uses first non-neutral emotion."""
    for eid in emotion_ids:
        if eid == NEUTRAL_IDX_GO or eid >= len(emotion_names_27):
            continue
        name = emotion_names_27[eid]
        six_name = config.GO_EMOTIONS_TO_SIX.get(name)
        if six_name is not None:
            return config.SIX_LABEL2ID[six_name]
    return config.SIX_LABEL2ID["sadness"]  # default if only neutral


def _go_emotion_ids_to_multilabel_six(emotion_ids, emotion_names_27):
    """Return set of 6-class indices that appear in this multi-label (for co-occurrence)."""
    out = set()
    for eid in emotion_ids:
        if eid == NEUTRAL_IDX_GO or eid >= len(emotion_names_27):
            continue
        name = emotion_names_27[eid]
        six_name = config.GO_EMOTIONS_TO_SIX.get(name)
        if six_name is not None:
            out.add(config.SIX_LABEL2ID[six_name])
    return list(out) if out else [config.SIX_LABEL2ID["sadness"]]


def load_go_emotions():
    """Load GoEmotions from local cache (Code/data/go_emotions) or download once. Returns train/val/test."""
    try:
        ds = load_dataset(config.GO_EMOTIONS_NAME, "simplified", cache_dir=config.GO_EMOTIONS_CACHE)
    except Exception:
        ds = load_dataset(config.GO_EMOTIONS_NAME, cache_dir=config.GO_EMOTIONS_CACHE)
    emotion_names = config.GO_EMOTIONS_27

    def _process(split_key):
        split = ds[split_key]
        texts = [x["text"] for x in split]
        label_list = split["labels"]
        single_labels = []
        multi_labels = []
        for labs in label_list:
            if isinstance(labs, (list, tuple)):
                ids = list(labs) if labs else [NEUTRAL_IDX_GO]
            else:
                ids = [int(labs)] if labs is not None else [NEUTRAL_IDX_GO]
            single_labels.append(_go_emotion_ids_to_six(ids, emotion_names))
            multi_labels.append(_go_emotion_ids_to_multilabel_six(ids, emotion_names))
        return texts, np.array(single_labels), multi_labels

    out = {}
    for key in ["train", "validation", "test"]:
        if key not in ds:
            continue
        texts, y_single, y_multi = _process(key)
        out[key] = {"texts": texts, "labels": y_single, "multi_labels": y_multi}
    return out


def load_twitter_emotion():
    """Load dair-ai/emotion (6 labels) from local cache (Code/data/emotion) or download once."""
    ds = load_dataset(
        config.TWITTER_EMOTION_NAME,
        config.TWITTER_EMOTION_CONFIG,
        cache_dir=config.EMOTION_CACHE,
    )
    out = {}
    for key in ["train", "validation", "test"]:
        if key not in ds:
            continue
        split = ds[key]
        texts = [x["text"] for x in split]
        labels = np.array([x["label"] for x in split])
        out[key] = {"texts": texts, "labels": labels, "multi_labels": [[l] for l in labels]}
    return out


def _apply_sample_limit(data_dict):
    """If config.USE_SAMPLE_LIMIT, truncate each split to config.SAMPLE_LIMIT samples."""
    if not getattr(config, "USE_SAMPLE_LIMIT", False):
        return data_dict
    n = getattr(config, "SAMPLE_LIMIT", None)
    if n is None:
        return data_dict
    out = {}
    for split_name, v in data_dict.items():
        out[split_name] = {
            "texts": v["texts"][:n],
            "labels": np.asarray(v["labels"])[:n],
            "multi_labels": v["multi_labels"][:n],
        }
    return out


def load_all():
    """Load both datasets; save processed splits to outputs for other scripts."""
    np.random.seed(config.RANDOM_SEED)
    reddit = load_go_emotions()
    twitter = load_twitter_emotion()
    reddit = _apply_sample_limit(reddit)
    twitter = _apply_sample_limit(twitter)

    # Save so other scripts can load without HuggingFace each time (optional)
    def to_json_safe(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [to_json_safe(x) for x in obj]
        return obj

    cache_path = os.path.join(config.DATA_DIR, "data_cache.json")
    cache = {
        "reddit": {
            k: {"texts": v["texts"], "labels": v["labels"].tolist(), "multi_labels": to_json_safe(v["multi_labels"])}
            for k, v in reddit.items()
        },
        "twitter": {
            k: {"texts": v["texts"], "labels": v["labels"].tolist(), "multi_labels": to_json_safe(v["multi_labels"])}
            for k, v in twitter.items()
        },
    }
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=0)
    print("Cached:", os.path.basename(os.path.dirname(cache_path)) + "/" + os.path.basename(cache_path))
    return reddit, twitter


def get_cached():
    """Load from local cache. Prefer Code/data/data_cache.json; fallback to outputs/data_cache.json."""
    cache_path = os.path.join(config.DATA_DIR, "data_cache.json")
    if not os.path.exists(cache_path):
        cache_path = os.path.join(config.OUTPUT_DIR, "data_cache.json")
    if not os.path.exists(cache_path):
        return load_all()
    with open(cache_path) as f:
        cache = json.load(f)
    reddit = {
        k: {
            "texts": v["texts"],
            "labels": np.array(v["labels"]),
            "multi_labels": v["multi_labels"],
        }
        for k, v in cache["reddit"].items()
    }
    twitter = {
        k: {
            "texts": v["texts"],
            "labels": np.array(v["labels"]),
            "multi_labels": v["multi_labels"],
        }
        for k, v in cache["twitter"].items()
    }
    reddit = _apply_sample_limit(reddit)
    twitter = _apply_sample_limit(twitter)
    return reddit, twitter


if __name__ == "__main__":
    reddit, twitter = load_all()
    for name, data in [("Reddit", reddit), ("Twitter", twitter)]:
        sizes = ", ".join(f"{s}: {len(data[s]['texts'])}" for s in data)
        print(f"{name}: {sizes}")
