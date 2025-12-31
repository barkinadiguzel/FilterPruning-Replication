from src.pruning.filter_score import l1_filter_score

def independent_pruning(conv, ratio):
    scores = l1_filter_score(conv)
    k = int(len(scores) * ratio)
    prune_idx = scores.argsort()[:k].tolist()
    return prune_idx
