from collections import Counter

from .common.metrics import prf


def gold_full_events(samples, type_system):
    result = []
    for s in samples:
        events = []
        for e in s.get("events", []):
            if not type_system.known_type(e.get("label", "")):
                continue
            args = frozenset(
                (
                    a.get("start_offset", a.get("start")),
                    a.get("end_offset", a.get("end")),
                    a["role"],
                )
                for a in e.get("arguments", [])
                if a.get("start_offset", a.get("start")) is not None
                and a.get("end_offset", a.get("end")) is not None
            )
            events.append(
                {
                    "sen_id": s["sen_id"],
                    "trigger": (e["start_offset"], e["end_offset"]),
                    "type": e["label"],
                    "arguments": args,
                }
            )
        result.append(events)
    return result


def _detection_set(events):
    return {(e["sen_id"], e["trigger"][0], e["trigger"][1], e["type"]) for e in events}


def _argument_set(events):
    out = set()
    for e in events:
        key = (e["sen_id"], e["trigger"][0], e["trigger"][1])
        for s, t, role in e["arguments"]:
            out.add(key + (s, t, role))
    return out


def _full_set(events):
    return {
        (e["sen_id"], e["trigger"][0], e["trigger"][1], e["type"], e["arguments"])
        for e in events
    }


def _micro(pred_per_sent, gold_per_sent, extractor):
    n_pred = n_gold = n_correct = 0
    for preds, golds in zip(pred_per_sent, gold_per_sent):
        ps = extractor(preds)
        gs = extractor(golds)
        n_pred += len(ps)
        n_gold += len(gs)
        n_correct += len(ps & gs)
    return prf(n_pred, n_gold, n_correct)


def evaluate_full(pred_per_sent, gold_per_sent):
    det = _micro(pred_per_sent, gold_per_sent, _detection_set)
    arg = _micro(pred_per_sent, gold_per_sent, _argument_set)
    full = _micro(pred_per_sent, gold_per_sent, _full_set)
    return {"detection_f1": det[2], "argument_f1": arg[2], "full_f1": full[2]}


def grouped_detection_f1(pred_per_sent, gold_per_sent, group_fn):
    pc = Counter()
    gc = Counter()
    cc = Counter()
    for preds, golds in zip(pred_per_sent, gold_per_sent):
        ps = _detection_set(preds)
        gs = _detection_set(golds)
        for item in ps:
            pc[group_fn(item[3])] += 1
        for item in gs:
            gc[group_fn(item[3])] += 1
        for item in ps & gs:
            cc[group_fn(item[3])] += 1
    return {g: prf(pc[g], gc[g], cc[g])[2] for g in set(list(pc) + list(gc))}


def grouped_argument_f1(pred_per_sent, gold_per_sent, type_of, group_fn):
    pc = Counter()
    gc = Counter()
    cc = Counter()
    for preds, golds in zip(pred_per_sent, gold_per_sent):
        ptypes = {(e["sen_id"], e["trigger"][0], e["trigger"][1]): e["type"] for e in preds}
        gtypes = {(e["sen_id"], e["trigger"][0], e["trigger"][1]): e["type"] for e in golds}
        ps = _argument_set(preds)
        gs = _argument_set(golds)
        for item in ps:
            full = ptypes.get(item[:3])
            if full:
                pc[group_fn(full, item[-1])] += 1
        for item in gs:
            full = gtypes.get(item[:3])
            if full:
                gc[group_fn(full, item[-1])] += 1
        for item in ps & gs:
            full = gtypes.get(item[:3])
            if full:
                cc[group_fn(full, item[-1])] += 1
    return {g: prf(pc[g], gc[g], cc[g])[2] for g in set(list(pc) + list(gc))}
