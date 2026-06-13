import json

from .common.alignment import align_with_used
from .common.metrics import micro_prf
from .nn_models.generative import HFGenerator
from .dataset import gold_argument_set, load_argument_examples
from .prompt import build_messages, select_fewshot


def _parse_output(raw):
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return {"events": []}
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return {"events": []}


def _to_predictions(sample, parsed):
    text = sample["text"]
    trigger_anchor = {e["id"]: e["start_offset"] for e in sample["events"]}
    preds = {}
    for ev in parsed.get("events", []):
        eid = ev.get("id")
        anchor = trigger_anchor.get(eid, 0)
        used = set()
        for arg in ev.get("arguments", []):
            role = arg.get("role")
            for span_text in arg.get("spans", []):
                pos = align_with_used(text, span_text, anchor, used)
                if pos:
                    preds.setdefault(eid, set()).add((sample["sen_id"], eid, pos[0], pos[1], role))
    return preds


def run_generative(model_name, test_samples, type_system, shots=0, train_pool=None, max_new_tokens=512):
    generator = HFGenerator(model_name)
    examples = load_argument_examples(test_samples, type_system)
    gold = gold_argument_set(examples)
    pred_by_event = {}
    for sample in test_samples:
        fewshot = select_fewshot(sample["text"], train_pool, shots) if shots and train_pool else None
        messages = build_messages(sample, type_system, fewshot)
        raw = generator.generate(messages, max_new_tokens=max_new_tokens, temperature=0.0)
        parsed = _parse_output(raw)
        for eid, preds in _to_predictions(sample, parsed).items():
            pred_by_event[(sample["sen_id"], eid)] = preds
    pred = [pred_by_event.get((ex["sen_id"], ex["event_id"]), set()) for ex in examples]
    return micro_prf(pred, gold)
