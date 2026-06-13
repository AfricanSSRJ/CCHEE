import json

from ..common.alignment import align_with_used
from ..type_resources import definitions_block
from ..nn_models.generative import HFGenerator, train_sft


SYSTEM = (
    "你是古汉语历史事件抽取助手。给定古汉语原文text与事件类型定义type_definitions，"
    "请从原文中抽取所有事件，包含触发词、事件类型及其论元角色。触发词与论元文本均须来自原文连续字符。"
    "输出严格JSON数组，每个元素包含trigger、type_coarse、type_fine、arguments字段，"
    "arguments为[{role, spans}]列表。若无事件则输出空数组[]。"
)


def _build_user(text, type_block):
    return json.dumps({"text": text, "type_definitions": type_block}, ensure_ascii=False)


def _build_target(sample, type_system):
    out = []
    for e in sample.get("events", []):
        if not type_system.known_type(e.get("label", "")):
            continue
        roles = {}
        for a in e.get("arguments", []):
            roles.setdefault(a["role"], []).append(
                a.get("text", sample["text"][a["start_offset"]:a["end_offset"]])
            )
        out.append(
            {
                "trigger": e["trigger"],
                "label": e["label"],
                "type_coarse": type_system.coarse_of(e["label"]),
                "type_fine": type_system.fine_of(e["label"]),
                "arguments": [{"role": r, "spans": v} for r, v in roles.items()],
            }
        )
    return json.dumps(out, ensure_ascii=False)


def build_sft_samples(samples, type_system, definitions):
    type_block = definitions_block(type_system, definitions)
    sft = []
    for s in samples:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": _build_user(s["text"], type_block)},
        ]
        sft.append({"messages": messages, "output": _build_target(s, type_system)})
    return sft


def _parse(raw):
    raw = raw.strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return []


def _to_events(sample, parsed, type_system):
    text = sample["text"]
    events = []
    used_triggers = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        trigger = item.get("trigger", "")
        full = item.get("label")
        if not type_system.known_type(full):
            full = type_system.to_full(item.get("type_coarse", ""), item.get("type_fine", ""))
        if not trigger or not full or trigger not in text:
            continue
        tpos = align_with_used(text, trigger, 0, used_triggers)
        if tpos is None:
            continue
        args = set()
        used_args = set()
        for arg in item.get("arguments", []):
            role = arg.get("role")
            if role not in type_system.roles_of(full):
                continue
            for span in arg.get("spans", []):
                pos = align_with_used(text, span, tpos[0], used_args)
                if pos:
                    args.add((pos[0], pos[1], role))
        events.append(
            {"sen_id": sample["sen_id"], "trigger": tpos, "type": full, "arguments": frozenset(args)}
        )
    return events


class GenerativeFull:
    def __init__(self, model_name, type_system, definitions, sft_cfg, output_dir):
        self.model_name = model_name
        self.type_system = type_system
        self.definitions = definitions
        self.sft_cfg = sft_cfg
        self.output_dir = output_dir
        self.lora_path = None

    def fit(self, train_samples):
        sft = build_sft_samples(train_samples, self.type_system, self.definitions)
        self.lora_path = train_sft(self.model_name, sft, self.output_dir, self.sft_cfg)
        return self

    def predict(self, samples):
        generator = HFGenerator(self.model_name, lora_path=self.lora_path)
        type_block = definitions_block(self.type_system, self.definitions)
        predictions = []
        for s in samples:
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": _build_user(s["text"], type_block)},
            ]
            raw = generator.generate(messages, max_new_tokens=512, temperature=0.0)
            predictions.append(_to_events(s, _parse(raw), self.type_system))
        return predictions
