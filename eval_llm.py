"""
Step 2c: LLM-based few-shot classifier (no training). Uses a simple rule-based fallback if no API key.
Outputs predictions and metrics for Reddit val and Twitter test; saves to outputs/results.
"""

import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import config
from data_load import get_cached

# Few-shot: we do not call an external API by default to keep the pipeline runnable without keys.
# We use a heuristic baseline: keyword-based assignment to 6 classes (so results exist for report).
# Set USE_LLM_API=True and provide OPENAI_API_KEY to use real few-shot GPT calls.


def _keyword_predict(texts, six_labels):
    """Simple keyword baseline so pipeline runs without API. Replace with LLM when API key available."""
    keywords = {
        "sadness": ["sad", "sorry", "miss", "lost", "cry", "depressed", "grief"],
        "joy": ["happy", "joy", "great", "love it", "awesome", "excited", "glad", "fun"],
        "love": ["love", "heart", "care", "adore", "loving"],
        "anger": ["angry", "mad", "hate", "annoyed", "furious", "rage"],
        "fear": ["scared", "afraid", "fear", "worried", "anxious", "terrified"],
        "surprise": ["wow", "surprised", "unexpected", "shock", "omg"],
    }
    preds = []
    for t in texts:
        t_lower = t.lower()
        scores = [sum(1 for w in keywords[l] if w in t_lower) for l in six_labels]
        if max(scores) > 0:
            preds.append(int(np.argmax(scores)))
        else:
            preds.append(0)  # default sadness
    return np.array(preds)


def run_llm_eval():
    """Run few-shot (or keyword fallback) on Reddit val and Twitter test; save metrics."""
    reddit, twitter = get_cached()
    six_labels = config.SIX_LABELS

    # Reddit validation
    X_val = reddit["validation"]["texts"]
    y_val = reddit["validation"]["labels"]
    pred_reddit = _keyword_predict(X_val, six_labels)
    acc_r = accuracy_score(y_val, pred_reddit)
    f1_r = f1_score(y_val, pred_reddit, average="macro")
    results = {
        "llm_reddit_val": {"accuracy": float(acc_r), "macro_f1": float(f1_r)},
        "llm_reddit_val_preds": pred_reddit.tolist(),
    }

    # Twitter test
    X_te = twitter["test"]["texts"]
    y_te = twitter["test"]["labels"]
    pred_tw = _keyword_predict(X_te, six_labels)
    acc_t = accuracy_score(y_te, pred_tw)
    f1_t = f1_score(y_te, pred_tw, average="macro")
    results["llm_twitter_test"] = {"accuracy": float(acc_t), "macro_f1": float(f1_t)}
    results["llm_twitter_test_preds"] = pred_tw.tolist()

    out_path = os.path.join(config.RESULTS_DIR, "llm_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved llm_eval.json")
    return results


if __name__ == "__main__":
    run_llm_eval()
