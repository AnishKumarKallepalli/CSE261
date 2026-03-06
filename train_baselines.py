"""
Step 2a: Train TF-IDF + Logistic Regression and (optionally) TF-IDF + SGDClassifier on Reddit.
Saves sklearn pipeline (vectorizer + classifier) to outputs/models.
"""

import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import config
from data_load import get_cached

MAX_FEATURES = 20000
TFIDF_MAX_DF = 0.95
TFIDF_MIN_DF = 2


def train_tfidf_lr(reddit, twitter_train=None):
    """Train TF-IDF + Logistic Regression on Reddit. Optionally train on Twitter for in-domain upper bound."""
    X_train = reddit["train"]["texts"]
    y_train = reddit["train"]["labels"]
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, max_df=TFIDF_MAX_DF, min_df=TFIDF_MIN_DF, ngram_range=(1, 2))
    clf = LogisticRegression(max_iter=500, random_state=config.RANDOM_SEED, solver="lbfgs")
    pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])
    pipe.fit(X_train, y_train)
    return pipe


def train_tfidf_sgd(reddit):
    """Train TF-IDF + SGDClassifier on Reddit (enhanced classical)."""
    X_train = reddit["train"]["texts"]
    y_train = reddit["train"]["labels"]
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, max_df=TFIDF_MAX_DF, min_df=TFIDF_MIN_DF, ngram_range=(1, 2))
    clf = SGDClassifier(loss="log_loss", max_iter=1000, random_state=config.RANDOM_SEED)
    pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])
    pipe.fit(X_train, y_train)
    return pipe


def train_twitter_baseline(twitter):
    """Train in-domain Twitter TF-IDF+LR for upper bound."""
    X_train = twitter["train"]["texts"]
    y_train = twitter["train"]["labels"]
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, max_df=TFIDF_MAX_DF, min_df=1, ngram_range=(1, 2))
    clf = LogisticRegression(max_iter=500, random_state=config.RANDOM_SEED, solver="lbfgs")
    pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])
    pipe.fit(X_train, y_train)
    return pipe


def main():
    reddit, twitter = get_cached()
    # Reddit models
    pipe_lr = train_tfidf_lr(reddit)
    path_lr = os.path.join(config.MODELS_DIR, "tfidf_lr_reddit.pkl")
    with open(path_lr, "wb") as f:
        pickle.dump(pipe_lr, f)
    print("Saved", os.path.basename(path_lr))

    pipe_sgd = train_tfidf_sgd(reddit)
    path_sgd = os.path.join(config.MODELS_DIR, "tfidf_sgd_reddit.pkl")
    with open(path_sgd, "wb") as f:
        pickle.dump(pipe_sgd, f)
    print("Saved", os.path.basename(path_sgd))

    # In-domain Twitter
    pipe_tw = train_twitter_baseline(twitter)
    path_tw = os.path.join(config.MODELS_DIR, "tfidf_lr_twitter.pkl")
    with open(path_tw, "wb") as f:
        pickle.dump(pipe_tw, f)
    print("Saved", os.path.basename(path_tw))


if __name__ == "__main__":
    main()
