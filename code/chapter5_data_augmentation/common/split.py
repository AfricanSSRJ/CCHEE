import random
from collections import defaultdict


def _primary_coarse(sample, type_system):
    coarses = [
        type_system.coarse_of(e["label"])
        for e in sample.get("events", [])
        if type_system.known_type(e.get("label", ""))
    ]
    if not coarses:
        return "NONE"
    return max(set(coarses), key=coarses.count)


def _fine_types_in(samples, type_system):
    found = set()
    for x in samples:
        for e in x.get("events", []):
            if type_system.known_type(e.get("label", "")):
                found.add(e["label"])
    return found


def _ensure_rare_in_train(train, dev, test, type_system):
    train_types = _fine_types_in(train, type_system)
    for pool in (dev, test):
        keep = []
        for x in pool:
            types = {
                e["label"]
                for e in x.get("events", [])
                if type_system.known_type(e.get("label", ""))
            }
            if types - train_types:
                train.append(x)
                train_types |= types
            else:
                keep.append(x)
        pool[:] = keep


def split_dataset(data, type_system, ratios=(0.8, 0.1, 0.1), seed=42):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for s in data:
        buckets[_primary_coarse(s, type_system)].append(s)
    train, dev, test = [], [], []
    for _, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * ratios[0])
        n_dev = int(n * ratios[1])
        train += items[:n_train]
        dev += items[n_train:n_train + n_dev]
        test += items[n_train + n_dev:]
    _ensure_rare_in_train(train, dev, test, type_system)
    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)
    return train, dev, test
