import copy
import random

from .analysis import LongTailSplit
from .common.alignment import align_with_used
from .esam import ESAM
from .quality_control import quality_control
from .rcsm import RCSM


def random_oversampling(train, type_system, longtail, factor=2):
    extra = []
    for s in train:
        if any(
            type_system.known_type(e.get("label", "")) and longtail.is_rare_type(e["label"])
            for e in s.get("events", [])
        ):
            for _ in range(factor - 1):
                extra.append(copy.deepcopy(s))
    for i, s in enumerate(extra):
        s["sen_id"] = 3 * 10 ** 7 + i
    return train + extra


def _eda_perturb(text, rng):
    chars = list(text)
    if len(chars) < 4:
        return text
    op = rng.choice(["swap", "delete", "duplicate"])
    i = rng.randrange(len(chars) - 1)
    if op == "swap":
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    elif op == "delete":
        chars.pop(i)
    else:
        chars.insert(i, chars[i])
    return "".join(chars)


def _realign_event(event, old_text, new_text, used):
    trigger = event.get("trigger", "")
    if not trigger:
        return None
    tpos = align_with_used(new_text, trigger, event.get("start_offset", 0), used)
    if tpos is None:
        return None
    event["start_offset"], event["end_offset"] = tpos
    new_args = []
    for a in event.get("arguments", []):
        span = a.get("text", old_text[a["start_offset"]:a["end_offset"]])
        pos = align_with_used(new_text, span, tpos[0], used)
        if pos is None:
            continue
        a["text"] = span
        a["start_offset"], a["end_offset"] = pos
        new_args.append(a)
    event["arguments"] = new_args
    return event


def eda(train, type_system, longtail, seed=42):
    rng = random.Random(seed)
    extra = []
    for s in train:
        if not any(
            type_system.known_type(e.get("label", "")) and longtail.is_rare_type(e["label"])
            for e in s.get("events", [])
        ):
            continue
        new_text = _eda_perturb(s["text"], rng)
        if new_text == s["text"]:
            continue
        clone = copy.deepcopy(s)
        used = set()
        kept = []
        for e in clone["events"]:
            realigned = _realign_event(e, s["text"], new_text, used)
            if realigned is not None:
                kept.append(realigned)
        clone["text"] = new_text
        clone["events"] = kept
        clone["source"] = "EDA"
        if clone["events"]:
            extra.append(clone)
    for i, s in enumerate(extra):
        s["sen_id"] = 4 * 10 ** 7 + i
    return train + extra


def _reassign_ids(samples, base):
    """为增强样本重排唯一 sen_id，避免与原始训练集或彼此发生主键冲突。"""
    for i, s in enumerate(samples):
        s["sen_id"] = base + i
    return samples


def prepare_augmentation(client, train, type_system, ratio=0.7):
    """一次性生成 ESAM 与 RCSM 增强样本（未经质量控制）并缓存。

    主实验各对比设置（ESAM only / RCSM only / Full）与消融实验复用同一批
    生成样本，既避免重复调用大模型，也保证各设置之间增强来源一致，
    与论文「同一套增强样本」的口径相符。
    """
    longtail = LongTailSplit(train, type_system, ratio)
    esam_aug = ESAM(client, type_system).run(train, longtail.rets) if client else []
    rcsm_aug = (
        RCSM(client, type_system).run(train, longtail.cets, longtail.is_rare_role)
        if client
        else []
    )
    _reassign_ids(esam_aug, base=10 ** 7)
    _reassign_ids(rcsm_aug, base=2 * 10 ** 7)
    return longtail, esam_aug, rcsm_aug


def compose_ccda(base_train, esam_aug, rcsm_aug, type_system,
                 use_esam=True, use_rcsm=True, use_qc=True, train_eval_fn=None):
    """基于缓存的增强样本组合出某一设置的训练集。"""
    augmented = []
    if use_esam:
        augmented += [copy.deepcopy(s) for s in esam_aug]
    if use_rcsm:
        augmented += [copy.deepcopy(s) for s in rcsm_aug]
    if use_qc:
        # 三级质量控制：事件层 + 论元层结构校验，叠加模型层批次级筛选
        # （仅当传入 train_eval_fn 时第三级才会执行；否则只做前两级结构校验）。
        augmented = quality_control(base_train, augmented, type_system, train_eval_fn)
    # use_qc=False 时跳过全部筛选，直接返回原始增强样本，
    # 以忠实复现消融实验中的 “w/o 质量控制” 设置。
    return base_train + augmented


def run_ccda(client, train, type_system, use_esam=True, use_rcsm=True, use_qc=True,
             train_eval_fn=None, ratio=0.7):
    """便捷入口：生成 + 组合一步到位（每次调用都会重新生成，成本较高，
    批量实验请改用 prepare_augmentation + compose_ccda 以复用生成结果）。"""
    longtail, esam_aug, rcsm_aug = prepare_augmentation(client, train, type_system, ratio)
    composed = compose_ccda(train, esam_aug, rcsm_aug, type_system,
                            use_esam, use_rcsm, use_qc, train_eval_fn)
    return composed, longtail
