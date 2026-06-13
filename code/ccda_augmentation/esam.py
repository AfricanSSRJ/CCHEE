import json

from .common.alignment import align_nearest, align_with_used


MODERN_MARKERS = ["的", "了", "吗", "呢", "着", "很", "非常", "我们", "你们", "这个", "那个", "因为", "所以"]


def _complete_examples(samples, type_system, full, limit=3):
    examples = []
    for s in samples:
        for e in s.get("events", []):
            if e.get("label") != full:
                continue
            roles = type_system.roles_of(full)
            covered = {a["role"] for a in e.get("arguments", [])}
            if len(covered & set(roles)) >= max(1, len(roles) - 1):
                examples.append((s["text"], e))
        if len(examples) >= limit:
            break
    return examples[:limit]


def _build_prompt(type_system, full, examples):
    roles = type_system.roles_of(full)
    coarse = type_system.coarse_of(full)
    fine = type_system.fine_of(full)
    shots = []
    for text, e in examples:
        args = [{"role": a["role"], "span": a.get("text", text[a["start_offset"]:a["end_offset"]])}
                for a in e.get("arguments", [])]
        shots.append({"text": text, "trigger": e["trigger"], "type_coarse": coarse,
                      "type_fine": fine, "arguments": args})
    system = (
        "你是古汉语史籍写作与事件标注助手。请仿照《二十四史》的书面文言风格，"
        "围绕给定事件类型生成一条全新的古汉语句子，并标注其触发词与核心论元。"
        "触发词必须是句中真实出现的连续字符，且能独立支持事件类型判定；"
        "各论元文本必须在句中显式出现。输出严格JSON对象，包含text、trigger、type_coarse、"
        "type_fine、arguments字段，arguments为[{role, span}]列表。不得使用现代汉语词汇或口语表达。"
    )
    user = json.dumps(
        {"type_coarse": coarse, "type_fine": fine, "roles": roles, "examples": shots},
        ensure_ascii=False,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse(raw):
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None


def _has_modern(text):
    return any(m in text for m in MODERN_MARKERS)


def _to_sample(obj, type_system, sen_id, target_full=None):
    text = obj.get("text", "")
    trigger = obj.get("trigger", "")
    full = obj.get("label")
    if not type_system.known_type(full):
        full = type_system.to_full(obj.get("type_coarse", ""), obj.get("type_fine", ""))
    # 目标事件类型由提示词固定，优先采用调用方指定的 target_full，
    # 避免 (粗粒度,细粒度) 反查在同名末段类型上产生歧义。
    if target_full is not None and type_system.known_type(target_full):
        full = target_full
    if not text or not trigger or not full or trigger not in text:
        return None
    if _has_modern(text):
        return None
    tpos = align_nearest(text, trigger, 0)
    if tpos is None:
        return None
    used = {tpos}
    arguments = []
    for a in obj.get("arguments", []):
        role = a.get("role")
        span = a.get("span", "")
        if role not in type_system.roles_of(full) or span not in text:
            continue
        pos = align_with_used(text, span, tpos[0], used)
        if pos:
            arguments.append({"role": role, "text": span, "start_offset": pos[0], "end_offset": pos[1]})
    return {
        "sen_id": sen_id,
        "text": text,
        "events": [
            {
                "id": 1,
                "trigger": trigger,
                "label": full,
                "start_offset": tpos[0],
                "end_offset": tpos[1],
                "arguments": arguments,
            }
        ],
        "source": "ESAM",
    }


class ESAM:
    def __init__(self, client, type_system, candidates_per_type=30):
        self.client = client
        self.type_system = type_system
        self.candidates_per_type = candidates_per_type

    def _repair_trigger(self, obj):
        messages = [
            {"role": "system", "content": "请修正以下JSON，使trigger为text中真实出现的连续字符，仅输出JSON。"},
            {"role": "user", "content": json.dumps(obj, ensure_ascii=False)},
        ]
        fixed = _parse(self.client.chat(messages, temperature=0.0))
        return fixed if fixed else obj

    def generate_for_type(self, samples, full, start_id=0):
        examples = _complete_examples(samples, self.type_system, full)
        prompt = _build_prompt(self.type_system, full, examples)
        produced = []
        for i in range(self.candidates_per_type):
            raw = self.client.chat(prompt)
            obj = _parse(raw)
            if obj is None:
                continue
            if obj.get("trigger") and obj.get("text") and obj["trigger"] not in obj["text"]:
                obj = self._repair_trigger(obj)
            sample = _to_sample(obj, self.type_system, start_id + i, target_full=full)
            if sample is not None:
                produced.append(sample)
        return produced

    def run(self, train_samples, rare_types):
        augmented = []
        sid = 10 ** 7
        for full in rare_types:
            produced = self.generate_for_type(train_samples, full, sid)
            augmented.extend(produced)
            sid += self.candidates_per_type
        return augmented
