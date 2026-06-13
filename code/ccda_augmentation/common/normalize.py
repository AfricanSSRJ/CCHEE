"""数据规范化工具。

数据集中“论元”的位置字段在不同来源下可能写作 start_offset/end_offset
或 start/end（例如 GPT-4o 自动标注产出的 all_arguments.jsonl 使用 start/end，
而仓库样例文件使用 start_offset/end_offset）。事件触发词字段统一为
start_offset/end_offset。本模块在数据载入后将所有论元统一规范为
start_offset/end_offset，并补全/校验 text 字段，使下游所有模块只需处理一种约定。
"""


def _find_nearest(text, span_text, anchor):
    if not span_text:
        return None
    positions = []
    start = text.find(span_text)
    while start != -1:
        positions.append((start, start + len(span_text)))
        start = text.find(span_text, start + 1)
    if not positions:
        return None
    return min(positions, key=lambda p: min(abs(p[0] - anchor), abs(p[1] - anchor)))


def normalize_event_arguments(event, text):
    """将单个事件的 arguments 规范为 [{role, text, start_offset, end_offset}]。

    处理逻辑：
    - 偏移字段优先取 start_offset/end_offset，缺失时回退到 start/end；
    - 若同时提供 text 且与偏移切片不一致，则以 text 为准，按“就近触发词”重新定位；
    - 若仅有 text 而无有效偏移，则在原句中按就近原则定位；
    - 无法解析的论元被丢弃（保持与原标注流程“宁缺勿错”的口径一致）。
    """
    anchor = event.get("start_offset", 0) or 0
    normalized = []
    for a in event.get("arguments", []):
        role = a.get("role")
        if role is None:
            continue
        so = a.get("start_offset", a.get("start"))
        eo = a.get("end_offset", a.get("end"))
        txt = a.get("text")
        valid = (
            isinstance(so, int)
            and isinstance(eo, int)
            and 0 <= so < eo <= len(text)
        )
        if valid and txt is not None and text[so:eo] != txt:
            valid = False
        if not valid:
            pos = _find_nearest(text, txt, anchor) if txt else None
            if pos is None:
                continue
            so, eo = pos
        if txt is None:
            txt = text[so:eo]
        normalized.append(
            {"role": role, "text": txt, "start_offset": so, "end_offset": eo}
        )
    event["arguments"] = normalized
    return event


def normalize_dataset(data):
    """就地规范化整个数据集（每行一个句子级样本）。"""
    for sample in data:
        text = sample.get("text", "")
        for event in sample.get("events", []):
            normalize_event_arguments(event, text)
    return data
