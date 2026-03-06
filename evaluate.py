"""
Step 3: Evaluate all models (Reddit-trained and Twitter in-domain) on Reddit test and Twitter test.
Saves Accuracy, macro-F1, micro-F1, per-emotion F1 to outputs/results.
"""

import os
import json
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import config
from data_load import get_cached


def eval_sklearn(pipe, texts, labels):
    preds = pipe.predict(texts)
    acc = accuracy_score(labels, preds)
    macro = f1_score(labels, preds, average="macro")
    micro = f1_score(labels, preds, average="micro")
    per_class = f1_score(labels, preds, average=None)
    cm = confusion_matrix(labels, preds)
    return {"accuracy": float(acc), "macro_f1": float(macro), "micro_f1": float(micro), "per_class_f1": per_class.tolist(), "confusion_matrix": cm.tolist()}


def eval_bert(model_dir, texts, labels, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    preds = []
    bs = config.BATCH_SIZE
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        enc = tokenizer(batch, truncation=True, max_length=config.MAX_LENGTH, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        preds.extend(out.logits.argmax(1).cpu().numpy().tolist())
    preds = np.array(preds)
    acc = accuracy_score(labels, preds)
    macro = f1_score(labels, preds, average="macro")
    micro = f1_score(labels, preds, average="micro")
    per_class = f1_score(labels, preds, average=None)
    cm = confusion_matrix(labels, preds)
    return {"accuracy": float(acc), "macro_f1": float(macro), "micro_f1": float(micro), "per_class_f1": per_class.tolist(), "confusion_matrix": cm.tolist()}


def main():
    reddit, twitter = get_cached()
    results = {}

    # Reddit test
    X_reddit_test = reddit["test"]["texts"]
    y_reddit_test = reddit["test"]["labels"]

    # Twitter test
    X_tw_test = twitter["test"]["texts"]
    y_tw_test = twitter["test"]["labels"]

    # TF-IDF LR Reddit -> Reddit test (in-domain)
    path_lr = os.path.join(config.MODELS_DIR, "tfidf_lr_reddit.pkl")
    if os.path.exists(path_lr):
        with open(path_lr, "rb") as f:
            pipe = pickle.load(f)
        results["tfidf_lr_reddit_on_reddit_test"] = eval_sklearn(pipe, X_reddit_test, y_reddit_test)
        results["tfidf_lr_reddit_on_twitter_test"] = eval_sklearn(pipe, X_tw_test, y_tw_test)

    # TF-IDF SGD Reddit -> both
    path_sgd = os.path.join(config.MODELS_DIR, "tfidf_sgd_reddit.pkl")
    if os.path.exists(path_sgd):
        with open(path_sgd, "rb") as f:
            pipe = pickle.load(f)
        results["tfidf_sgd_reddit_on_reddit_test"] = eval_sklearn(pipe, X_reddit_test, y_reddit_test)
        results["tfidf_sgd_reddit_on_twitter_test"] = eval_sklearn(pipe, X_tw_test, y_tw_test)

    # BERT Reddit -> both
    bert_dir = os.path.join(config.MODELS_DIR, "bert_reddit")
    if os.path.exists(bert_dir):
        results["bert_reddit_on_reddit_test"] = eval_bert(bert_dir, X_reddit_test, y_reddit_test)
        results["bert_reddit_on_twitter_test"] = eval_bert(bert_dir, X_tw_test, y_tw_test)

    # Twitter in-domain (upper bound)
    path_tw = os.path.join(config.MODELS_DIR, "tfidf_lr_twitter.pkl")
    if os.path.exists(path_tw):
        with open(path_tw, "rb") as f:
            pipe = pickle.load(f)
        results["tfidf_lr_twitter_on_twitter_test"] = eval_sklearn(pipe, X_tw_test, y_tw_test)

    # Add label names for report
    results["label_names"] = config.SIX_LABELS
    out_path = os.path.join(config.RESULTS_DIR, "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved evaluation_results.json")
    return results


if __name__ == "__main__":
    main()
