import os

from .common.io_utils import load_json, load_jsonl


SEED_DEFINITIONS = {
    "军事-作战-攻击-征伐": "一方主动出兵攻击另一方的军事行为，通常由一国之君或将领发起。",
    "人生-出生": "一个人降生于世的事件，通常记录其出生时间、地点及父母信息。",
    "交流-国家交流-出使": "某人受命前往他国或他地执行外交使命的事件，通常代表本国进行沟通交涉。",
}


def build_definitions(type_system, override_path=None):
    definitions = {}
    if override_path and os.path.exists(override_path):
        definitions.update(load_json(override_path))
    for full in type_system.fine_types:
        if full in definitions:
            continue
        if full in SEED_DEFINITIONS:
            definitions[full] = SEED_DEFINITIONS[full]
        else:
            coarse = type_system.coarse_of(full)
            fine = type_system.fine_of(full)
            roles = "、".join(type_system.roles_of(full))
            definitions[full] = f"属于{coarse}类的{fine}事件，核心论元角色包括{roles}。"
    return definitions


def definitions_block(type_system, definitions):
    lines = []
    for full in type_system.fine_types:
        coarse = type_system.coarse_of(full)
        fine = type_system.fine_of(full)
        lines.append(f"{coarse}-{fine}：{definitions[full]}")
    return "\n".join(lines)


def load_translations(path):
    if not path or not os.path.exists(path):
        return {}
    return {item["sen_id"]: item["translation"] for item in load_jsonl(path)}
