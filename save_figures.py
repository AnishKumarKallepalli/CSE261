"""
Save figures to outputs/figures/: confusion matrices (Reddit model on Reddit test + on Twitter)
and emotion network graphs. Run after evaluate.py and build_networks.py (or at end of run_all.py).
"""
import os
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

import config


def _save_one_confusion_matrix(data, result_key, out_basename, title):
    """Save a single confusion matrix figure if result_key exists in data."""
    if result_key not in data or "confusion_matrix" not in data[result_key]:
        return False
    labels = data.get("label_names", config.SIX_LABELS)
    cm = np.array(data[result_key]["confusion_matrix"])
    n = cm.shape[0]
    tick_labels = labels[:n] if len(labels) >= n else [str(i) for i in range(n)]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    cm_max = cm.max() if cm.size else 1
    for i in range(n):
        for j in range(min(n, cm.shape[1])):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="black" if cm[i, j] < cm_max / 2 else "white")
    plt.colorbar(im, ax=ax, label="Count")
    ax.set_title(title)
    plt.tight_layout()
    out = os.path.join(config.FIGURES_DIR, out_basename)
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved", out_basename)
    return True


def save_confusion_matrices():
    """Save confusion matrix for Reddit model on Reddit test (GT) and on Twitter (cross-domain)."""
    path = os.path.join(config.RESULTS_DIR, "evaluation_results.json")
    if not os.path.exists(path):
        print("No evaluation_results.json found; skipping confusion matrices")
        return
    with open(path) as f:
        data = json.load(f)
    # Reddit model on Reddit test (in-domain, ground truth Reddit)
    for key in ("bert_reddit_on_reddit_test", "tfidf_lr_reddit_on_reddit_test"):
        if _save_one_confusion_matrix(
            data, key,
            "confusion_matrix_reddit.png",
            "Confusion matrix (Reddit model on Reddit test)\n" + key,
        ):
            break
    else:
        print("No Reddit-on-Reddit confusion matrix found in results")
    # Reddit model on Twitter (cross-domain)
    for key in ("bert_reddit_on_twitter_test", "tfidf_lr_reddit_on_twitter_test"):
        if _save_one_confusion_matrix(
            data, key,
            "confusion_matrix_twitter.png",
            "Confusion matrix (Reddit model on Twitter test)\n" + key,
        ):
            break
    else:
        print("No Reddit-on-Twitter confusion matrix found in results")


def save_network_graph(G, out_path, title):
    if G is None or G.number_of_nodes() == 0:
        return
    labels = config.SIX_LABELS
    pos = nx.spring_layout(G, seed=config.RANDOM_SEED, k=1.5)
    node_labels = {i: labels[i] for i in G.nodes()}
    edges = G.edges()
    weights = [G[u][v].get("weight", 0) for u, v in edges]
    w_min, w_max = min(weights) if weights else 0, max(weights) if weights else 1
    edge_widths = [1 + 2 * (w - w_min) / (w_max - w_min + 1e-10) for w in weights]
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=800, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, ax=ax)
    nx.draw_networkx_labels(G, pos, node_labels, font_size=10, ax=ax)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved", os.path.basename(out_path))


def main():
    save_confusion_matrices()
    path_r = os.path.join(config.OUTPUT_DIR, "network_reddit.pkl")
    path_t = os.path.join(config.OUTPUT_DIR, "network_twitter.pkl")
    if os.path.exists(path_r):
        with open(path_r, "rb") as f:
            G_r = pickle.load(f)
        save_network_graph(G_r, os.path.join(config.FIGURES_DIR, "network_reddit.png"), "Reddit emotion co-occurrence")
    if os.path.exists(path_t):
        with open(path_t, "rb") as f:
            G_t = pickle.load(f)
        save_network_graph(G_t, os.path.join(config.FIGURES_DIR, "network_twitter.png"), "Twitter emotion co-occurrence (predicted)")
    print("Figures in:", config.FIGURES_DIR)


if __name__ == "__main__":
    main()
