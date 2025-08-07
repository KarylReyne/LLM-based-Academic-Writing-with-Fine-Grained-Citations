import numpy as np


def minmax_normalization(x, eps=1e-6):
    x_min = min(x)
    x_max = max(x)
    x_norm = []
    for e in x:
        x_norm.append((e-x_min)/(x_max-x_min+eps))
    return x_norm


def recall_at_k(rankings, gold_ids, k=1):
    assert len(rankings) == len(gold_ids), f"{len(rankings)}, {len(gold_ids)}"
    hits = 0
    for i in range(len(rankings)):
        if gold_ids[i] in rankings[i][:k]:
            hits += 1 
    return float(hits/len(rankings))


def single_recall_at_k(rankings, gold, k=1):
    return gold in rankings[:k]

