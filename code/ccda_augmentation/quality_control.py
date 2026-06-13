MODERN_MARKERS = ["的", "了", "吗", "呢", "着", "很", "非常", "我们", "你们", "因为", "所以"]


def event_level_filter(samples, type_system):
    kept = []
    for s in samples:
        valid = True
        for e in s.get("events", []):
            full = e.get("label")
            if not type_system.known_type(full):
                valid = False
                break
            trigger = e.get("trigger", "")
            if not trigger or trigger not in s["text"] or len(trigger) > 4:
                valid = False
                break
        if valid:
            kept.append(s)
    return kept


def argument_level_filter(samples, type_system):
    kept = []
    for s in samples:
        text = s["text"]
        valid = True
        for e in s.get("events", []):
            roles = type_system.roles_of(e["label"])
            for a in e.get("arguments", []):
                span = text[a["start_offset"]:a["end_offset"]]
                if a["role"] not in roles or span != a.get("text", span):
                    valid = False
                    break
                if any(m in span for m in MODERN_MARKERS):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            kept.append(s)
    return kept


def batch_level_filter(base_train, augmented, train_eval_fn, batch_size=200, patience=2):
    kept = []
    history = {"detection_f1": [], "argument_f1": [], "full_f1": []}
    for i in range(0, len(augmented), batch_size):
        batch = augmented[i:i + batch_size]
        scores = train_eval_fn(base_train + kept + batch)
        drop = False
        for metric in history:
            seq = history[metric] + [scores[metric]]
            if len(seq) > patience and all(
                seq[-1 - j] < seq[-2 - j] for j in range(patience)
            ):
                drop = True
        if drop:
            continue
        kept.extend(batch)
        for metric in history:
            history[metric].append(scores[metric])
    return kept


def quality_control(base_train, augmented, type_system, train_eval_fn=None, batch_size=200):
    filtered = event_level_filter(augmented, type_system)
    filtered = argument_level_filter(filtered, type_system)
    if train_eval_fn is None:
        return filtered
    return batch_level_filter(base_train, filtered, train_eval_fn, batch_size)
