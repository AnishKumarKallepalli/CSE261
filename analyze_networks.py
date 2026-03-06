"""
Step 5: Network comparison and analysis. Graph stats, centrality, community detection,
edge comparison, and link to transfer errors (confused pairs vs network difference).
"""
import os
import json
import pickle
import warnings

import numpy as np
import networkx as nx
import config

warnings.filterwarnings("ignore", category=RuntimeWarning, module="networkx")

HAS_LOUVAIN = False
HAS_GREEDY = False
try:
    from networkx.algorithms.community import louvain_communities
    HAS_LOUVAIN = True
except ImportError:
    pass
if not HAS_LOUVAIN:
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        HAS_GREEDY = True
    except ImportError:
        pass


def load_graphs():
    G_reddit = None
    G_twitter = None
    path_r = os.path.join(config.OUTPUT_DIR, "network_reddit.pkl")
    path_t = os.path.join(config.OUTPUT_DIR, "network_twitter.pkl")
    if os.path.exists(path_r):
        with open(path_r, "rb") as f:
            G_reddit = pickle.load(f)
    if os.path.exists(path_t):
        with open(path_t, "rb") as f:
            G_twitter = pickle.load(f)
    return G_reddit, G_twitter


def graph_stats(G, name):
    if G is None or G.number_of_nodes() == 0:
        return {}
    density = nx.density(G)
    degrees = [d for _, d in G.degree()]
    avg_degree = np.mean(degrees) if degrees else 0
    degree_cent = nx.degree_centrality(G)
    pagerank = nx.pagerank(G, weight="weight")
    out = {
        "density": density,
        "avg_degree": avg_degree,
        "degree_centrality": {str(k): v for k, v in degree_cent.items()},
        "pagerank": {str(k): v for k, v in pagerank.items()},
    }
    try:
        if HAS_LOUVAIN:
            communities = louvain_communities(G, weight="weight")
            m = G.number_of_edges()
            out["modularity"] = float(nx.community.modularity(G, communities, weight="weight")) if m > 0 else None
            out["communities"] = [[G.nodes[n].get("name", n) for n in c] for c in communities]
        elif HAS_GREEDY:
            communities = list(greedy_modularity_communities(G, weight="weight"))
            m = G.number_of_edges()
            out["modularity"] = float(nx.community.modularity(G, communities, weight="weight")) if m > 0 else None
            out["communities"] = [[G.nodes[n].get("name", n) for n in c] for c in communities]
        else:
            out["modularity"] = None
            out["communities"] = []
    except (ZeroDivisionError, FloatingPointError, ValueError):
        out["modularity"] = None
        out["communities"] = []
    return out


def top_edges(G, k=10):
    if G is None:
        return []
    edges = [(u, v, G[u][v].get("weight", 0)) for u, v in G.edges()]
    edges.sort(key=lambda x: -x[2])
    names = config.SIX_LABELS
    return [(names[u], names[v], w) for u, v, w in edges[:k]]


def main():
    G_reddit, G_twitter = load_graphs()
    results = {}

    if G_reddit is not None:
        results["reddit_stats"] = graph_stats(G_reddit, "reddit")
        results["reddit_top_edges"] = top_edges(G_reddit, 10)
    if G_twitter is not None:
        results["twitter_stats"] = graph_stats(G_twitter, "twitter")
        results["twitter_top_edges"] = top_edges(G_twitter, 10)

    # Load evaluation results for confusion vs network
    eval_path = os.path.join(config.RESULTS_DIR, "evaluation_results.json")
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_res = json.load(f)
        # Confusion: which pairs (true, pred) are most confused on Twitter for Reddit model?
        key = "bert_reddit_on_twitter_test"
        if key not in eval_res:
            key = "tfidf_lr_reddit_on_twitter_test"
        if key in eval_res and "confusion_matrix" in eval_res[key]:
            cm = np.array(eval_res[key]["confusion_matrix"])
            labels = config.SIX_LABELS
            confused_pairs = []
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    if i != j and cm[i, j] > 0:
                        confused_pairs.append((labels[i], labels[j], int(cm[i, j])))
            confused_pairs.sort(key=lambda x: -x[2])
            results["confused_pairs_twitter"] = confused_pairs[:15]
    results["label_names"] = config.SIX_LABELS
    out_path = os.path.join(config.RESULTS_DIR, "network_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved network_analysis.json")
    return results


if __name__ == "__main__":
    main()
