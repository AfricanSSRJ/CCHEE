import json
from collections import Counter


def _char_ngrams(text, n=2):
    grams = Counter()
    for i in range(len(text) - n + 1):
        grams[text[i:i + n]] += 1
    for ch in text:
        grams[ch] += 1
    return grams


def _cosine(a, b):
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def select_fewshot(query_text, pool, k=2):
    qg = _char_ngrams(query_text)
    scored = [(s, _cosine(qg, _char_ngrams(s["text"]))) for s in pool]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:k]]


SYSTEM = (
    "你是古汉语事件抽取助手。给定古汉语原文text以及多个待处理事件events"
    "（每个事件包含label、trigger、roles），请为每个事件抽取对应论元。"
    "请仅输出来自原文的片段文本span，不要输出offset、解释或额外内容。"
    "若某个role没有对应论元，则不要输出该role；若同一role对应多个论元，则输出多个span。"
    "输出必须严格遵循JSON格式。"
)


def build_user_content(text, events, type_system):
    payload = {
        "text": text,
        "events": [
            {
                "id": e["id"],
                "label": e["label"],
                "trigger": {"text": e["trigger"], "start": e["start_offset"], "end": e["end_offset"]},
                "roles": type_system.roles_of(e["label"]),
            }
            for e in events
        ],
        "output_json_schema": {
            "events": [{"id": "{EVENT_ID}", "arguments": [{"role": "{ROLE}", "spans": ["{SPAN}"]}]}]
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def build_assistant_target(text, events):
    out = {"events": []}
    for e in events:
        roles = {}
        for a in e.get("arguments", []):
            roles.setdefault(a["role"], []).append(a.get("text", text[a["start_offset"]:a["end_offset"]]))
        out["events"].append(
            {"id": e["id"], "arguments": [{"role": r, "spans": v} for r, v in roles.items()]}
        )
    return json.dumps(out, ensure_ascii=False)


def build_messages(sample, type_system, fewshot=None):
    messages = [{"role": "system", "content": SYSTEM}]
    for ex in fewshot or []:
        messages.append({"role": "user", "content": build_user_content(ex["text"], ex["events"], type_system)})
        messages.append({"role": "assistant", "content": build_assistant_target(ex["text"], ex["events"])})
    messages.append({"role": "user", "content": build_user_content(sample["text"], sample["events"], type_system)})
    return messages
