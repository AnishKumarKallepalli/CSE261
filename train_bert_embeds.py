"""
Train BERT-embedding baselines on Reddit (GoEmotions mapped to 6 labels).

We freeze a BERT encoder, extract CLS embeddings for Reddit train texts, and
train two sklearn classifiers on top:
- LogisticRegression
- SGDClassifier

Models are saved under MODELS_DIR as:
- bert_embed_lr_reddit.pkl
- bert_embed_sgd_reddit.pkl
"""

import os
import pickle

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, SGDClassifier
from transformers import AutoTokenizer, AutoModel

import config
from data_load import get_cached


def _bert_embed(texts, tokenizer, model, device, batch_size=None):
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=config.MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        # CLS token embedding
        cls_emb = out.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_emb)
    if not embeddings:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return np.vstack(embeddings)


def main():
    reddit, _ = get_cached()

    X_train = reddit["train"]["texts"]
    y_train = reddit["train"]["labels"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    model = AutoModel.from_pretrained(config.BERT_MODEL_NAME).to(device)
    model.eval()

    print("Computing BERT CLS embeddings for Reddit train...")
    X_train_emb = _bert_embed(X_train, tokenizer, model, device)
    print(f"Train embeddings shape: {X_train_emb.shape}")

    # Logistic Regression baseline (multiclass handled automatically)
    lr = LogisticRegression(max_iter=1000, n_jobs=-1)
    lr.fit(X_train_emb, y_train)

    # SGD baseline (logistic loss)
    sgd = SGDClassifier(loss="log_loss")
    sgd.fit(X_train_emb, y_train)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    path_lr = os.path.join(config.MODELS_DIR, "bert_embed_lr_reddit.pkl")
    path_sgd = os.path.join(config.MODELS_DIR, "bert_embed_sgd_reddit.pkl")

    with open(path_lr, "wb") as f:
        pickle.dump(lr, f)
    with open(path_sgd, "wb") as f:
        pickle.dump(sgd, f)

    print("Saved:", path_lr)
    print("Saved:", path_sgd)


if __name__ == "__main__":
    main()

