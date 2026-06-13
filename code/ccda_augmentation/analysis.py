from collections import Counter


def _split_by_cumulative(counter, ratio=0.7):
    total = sum(counter.values())
    common = set()
    acc = 0
    for key, count in counter.most_common():
        common.add(key)
        acc += count
        if acc >= total * ratio:
            break
    rare = set(counter) - common
    return common, rare


def event_type_counts(samples, type_system):
    counter = Counter()
    for s in samples:
        for e in s.get("events", []):
            if type_system.known_type(e.get("label", "")):
                counter[e["label"]] += 1
    return counter


def role_combo_counts(samples, type_system):
    counter = Counter()
    for s in samples:
        for e in s.get("events", []):
            if not type_system.known_type(e.get("label", "")):
                continue
            for a in e.get("arguments", []):
                counter[(e["label"], a["role"])] += 1
    return counter


class LongTailSplit:
    def __init__(self, samples, type_system, ratio=0.7):
        self.type_system = type_system
        et = event_type_counts(samples, type_system)
        rc = role_combo_counts(samples, type_system)
        self.cets, self.rets = _split_by_cumulative(et, ratio)
        self.cars, self.rars = _split_by_cumulative(rc, ratio)

    def is_rare_type(self, full):
        return full in self.rets

    def is_rare_role(self, full, role):
        return (full, role) in self.rars

    def detection_group(self, full):
        return "RETs" if full in self.rets else "CETs"

    def argument_group(self, full, role):
        return "RARs" if (full, role) in self.rars else "CARs"
