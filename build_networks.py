"""
Step 4: Build emotion co-occurrence graphs for Reddit (gold multi-label) and Twitter (predicted multi-label from model).
Edge weights: PMI or normalized co-occurrence. Save graphs to outputs.
"""

import os
import json
import pickle
import numpy as np
import networkx as nx
from collections import defaultdict
import config
from data_load import get_cached

# Use Reddit gold multi-labels for Reddit network. For Twitter we need predicted multi-label:
# run BERT (or TF-IDF LR) and take top-2 or threshold to get multiple labels per tweet, then co-occurrence.


def _pmi(counts_ij, count_i, count_j, n_docs):
    """PMI(i,j) = log( P(i,j) / (P(i)P(j)) ). Use co-occurrence counts."""
    if counts_ij == 0 or count_i == 0 or count_j == 0:
        return 0.0
    p_ij = counts_ij / n_docs
    p_i = count_i / n_docs
    p_j = count_j / n_docs
    return np.log2((p_ij + 1e-10) / (p_i * p_j + 1e-10))


def build_cooccurrence_network(multi_labels_list, label_names):
    """Build weighted graph from multi-label list. Each item is list of label indices (0-5)."""
    n = len(config.SIX_LABELS)
    cooccur = defaultdict(float)
    single_count = defaultdict(float)
    for labels in multi_labels_list:
        labels = [int(x) for x in labels if 0 <= x < n]
        if not labels:
            continue
        for i in labels:
            single_count[i] += 1
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = min(labels[i], labels[j]), max(labels[i], labels[j])
                cooccur[(a, b)] += 1
    # Also add (i,i) as single? No, edges are between different emotions. So (i,j) i<j only.
    N = len(multi_labels_list)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, name=label_names[i])
    for (i, j), c in cooccur.items():
        pmi = _pmi(c, single_count[i], single_count[j], N)
        G.add_edge(i, j, weight=pmi, count=c)
    return G


def get_predicted_multilabel(texts, model_type="bert", top_k=2):
    """Get predicted multi-label (top-k or threshold) for Twitter texts using Reddit-trained model."""
    from data_load import get_cached
    reddit, _ = get_cached()
    label_names = config.SIX_LABELS
    n = len(label_names)
    if model_type == "bert":
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        model_dir = os.path.join(config.MODELS_DIR, "bert_reddit")
        if not os.path.exists(model_dir):
            model_type = "tfidf_lr"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
            model.eval()
            preds_multilabel = []
            bs = config.BATCH_SIZE
            for i in range(0, len(texts), bs):
                batch = texts[i : i + bs]
                enc = tokenizer(batch, truncation=True, max_length=config.MAX_LENGTH, padding=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = model(**enc).logits
                # top-k labels per sample
                for k in range(logits.size(0)):
                    probs = torch.softmax(logits[k], 0).cpu().numpy()
                    top = np.argsort(probs)[-top_k:]
                    preds_multilabel.append(top.tolist())
            return preds_multilabel
    # Fallback: TF-IDF LR single label -> duplicate as "multi-label" of length 1
    import pickle
    path_lr = os.path.join(config.MODELS_DIR, "tfidf_lr_reddit.pkl")
    with open(path_lr, "rb") as f:
        pipe = pickle.load(f)
    single = pipe.predict(texts)
    return [[int(s)] for s in single]


def main():
    reddit, twitter = get_cached()
    label_names = config.SIX_LABELS

    # Reddit network from gold multi-labels
    reddit_multi = reddit["train"]["multi_labels"] + reddit["validation"]["multi_labels"] + reddit["test"]["multi_labels"]
    G_reddit = build_cooccurrence_network(reddit_multi, label_names)
    path_reddit = os.path.join(config.OUTPUT_DIR, "network_reddit.pkl")
    with open(path_reddit, "wb") as f:
        pickle.dump(G_reddit, f)
    print("Saved network_reddit.pkl")

    # Twitter: predicted multi-label from BERT (or TF-IDF)
    twitter_texts = twitter["train"]["texts"] + twitter["validation"]["texts"] + twitter["test"]["texts"]
    twitter_multi_pred = get_predicted_multilabel(twitter_texts, model_type="bert", top_k=2)
    G_twitter = build_cooccurrence_network(twitter_multi_pred, label_names)
    path_twitter = os.path.join(config.OUTPUT_DIR, "network_twitter.pkl")
    with open(path_twitter, "wb") as f:
        pickle.dump(G_twitter, f)
    print("Saved network_twitter.pkl")

    # Save edge lists for report
    def edges_for_json(G):
        return [(G.nodes[i].get("name", i), G.nodes[j].get("name", j), G[i][j].get("weight", 0)) for i, j in G.edges()]
    with open(os.path.join(config.RESULTS_DIR, "network_reddit_edges.json"), "w") as f:
        json.dump(edges_for_json(G_reddit), f, indent=2)
    with open(os.path.join(config.RESULTS_DIR, "network_twitter_edges.json"), "w") as f:
        json.dump(edges_for_json(G_twitter), f, indent=2)


if __name__ == "__main__":
    main()
