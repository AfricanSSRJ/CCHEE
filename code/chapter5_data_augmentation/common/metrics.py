from collections import defaultdict


def prf(n_pred, n_gold, n_correct):
    p = n_correct / n_pred if n_pred else 0.0
    r = n_correct / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def micro_prf(pred_sets, gold_sets):
    n_pred = n_gold = n_correct = 0
    for ps, gs in zip(pred_sets, gold_sets):
        n_pred += len(ps)
        n_gold += len(gs)
        n_correct += len(ps & gs)
    return prf(n_pred, n_gold, n_correct)


def macro_f1(pred_sets, gold_sets, label_of, labels):
    pc = defaultdict(int)
    gc = defaultdict(int)
    cc = defaultdict(int)
    for ps, gs in zip(pred_sets, gold_sets):
        for item in ps:
            pc[label_of(item)] += 1
        for item in gs:
            gc[label_of(item)] += 1
        for item in ps & gs:
            cc[label_of(item)] += 1
    scores = []
    for lab in labels:
        _, _, f = prf(pc[lab], gc[lab], cc[lab])
        scores.append(f)
    return sum(scores) / len(scores) if scores else 0.0


def grouped_prf(pred_sets, gold_sets, group_of, groups):
    pc = defaultdict(int)
    gc = defaultdict(int)
    cc = defaultdict(int)
    for ps, gs in zip(pred_sets, gold_sets):
        for item in ps:
            pc[group_of(item)] += 1
        for item in gs:
            gc[group_of(item)] += 1
        for item in ps & gs:
            cc[group_of(item)] += 1
    result = {}
    for g in groups:
        result[g] = prf(pc[g], gc[g], cc[g])
    return result
