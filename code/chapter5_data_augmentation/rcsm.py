import copy
import json

from .common.alignment import align_with_used


def _missing_roles(event, type_system):
    defined = type_system.roles_of(event["label"])
    covered = {a["role"] for a in event.get("arguments", [])}
    return [r for r in defined if r not in covered]


def _build_prompt(text, full, missing, type_system):
    coarse = type_system.coarse_of(full)
    fine = type_system.fine_of(full)
    system = (
        "你是古汉语史籍续写助手。请围绕给定原句，补充一段不超过15字的古汉语片段，"
        "为指定的缺失论元角色提供内容，使补充片段在人物、时间、地点与叙事逻辑上与原句保持一致，"
        "并保持文言书面风格。输出严格JSON对象，包含supplement字段以及arguments字段"
        "（[{role, span}]，span须出现在supplement中）。"
    )
    user = json.dumps(
        {"text": text, "type_coarse": coarse, "type_fine": fine, "missing_roles": missing},
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


class RCSM:
    def __init__(self, client, type_system, max_supplement_len=15):
        self.client = client
        self.type_system = type_system
        self.max_supplement_len = max_supplement_len

    def supplement_sample(self, sample, common_types, rare_role_fn):
        new_sample = copy.deepcopy(sample)
        text = new_sample["text"]
        changed = False
        for event in new_sample["events"]:
            full = event.get("label")
            if not self.type_system.known_type(full) or full not in common_types:
                continue
            missing = [r for r in _missing_roles(event, self.type_system) if rare_role_fn(full, r)]
            if not missing:
                continue
            prompt = _build_prompt(text, full, missing, self.type_system)
            obj = _parse(self.client.chat(prompt))
            if obj is None:
                continue
            supplement = obj.get("supplement", "")
            if not supplement or len(supplement) > self.max_supplement_len:
                continue
            offset = len(text)
            text = text + supplement
            used = set()
            for a in obj.get("arguments", []):
                role = a.get("role")
                span = a.get("span", "")
                if role not in missing or span not in supplement:
                    continue
                pos = align_with_used(supplement, span, 0, used)
                if pos:
                    event.setdefault("arguments", []).append(
                        {
                            "role": role,
                            "text": span,
                            "start_offset": offset + pos[0],
                            "end_offset": offset + pos[1],
                        }
                    )
                    changed = True
        new_sample["text"] = text
        new_sample["source"] = "RCSM"
        return new_sample if changed else None

    def run(self, train_samples, common_types, rare_role_fn):
        augmented = []
        for sample in train_samples:
            has_common = any(
                self.type_system.known_type(e.get("label", "")) and e["label"] in common_types
                for e in sample.get("events", [])
            )
            if not has_common:
                continue
            result = self.supplement_sample(sample, common_types, rare_role_fn)
            if result is not None:
                augmented.append(result)
        return augmented
