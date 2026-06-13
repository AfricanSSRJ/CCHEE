import argparse
import os

from .arg_dataset import load_argument_examples
from .common.io_utils import load_jsonl
from .common.normalize import normalize_dataset
from .common.schema import TypeSystem
from .common.seed import set_seed
from .common.split import split_dataset
from .analysis import LongTailSplit
from .augment import compose_ccda, eda, prepare_augmentation, random_oversampling
from .evaluate import (evaluate_full, gold_full_events, grouped_argument_f1,
                       grouped_detection_f1)
from .ext_models.generative_full import GenerativeFull
from .ext_models.joint_bio import JointBIO
from .ext_models.pipeline_span import PipelineSpan
from .resources_cfg import ARG_CFG, BIO_CFG, SFT_CFG, TRIGGER_CFG


def make_model(model_type, args, type_system, definitions, out_dir):
    if model_type == "pipeline":
        return PipelineSpan(args.guwenbert, type_system, TRIGGER_CFG, ARG_CFG)
    if model_type == "joint":
        return JointBIO(args.guwenbert, type_system, BIO_CFG)
    return GenerativeFull(args.xunzi, type_system, definitions, SFT_CFG, out_dir)


def fit_predict(model, model_type, train, test, type_system):
    if model_type == "pipeline":
        model.fit(train, load_argument_examples(train, type_system))
    else:
        model.fit(train)
    return model.predict(test)


def make_qc_eval_fn(args, type_system, dev):
    # 第三级（模型层）质量控制所需的批次级评估回调：
    # 在候选训练集上训练一个代表性模型（Pipeline-Span，与论文 5.3.4 口径一致），
    # 并在验证集上返回事件检测 / 论元抽取 / 完整事件抽取三项 F1。
    # 为控制批次级筛选的总开销，此处使用减少 epoch 的轻量探针配置；
    # 该配置仅用于“筛样本”的相对比较，不影响最终模型的训练超参数。
    dev_gold = gold_full_events(dev, type_system)
    probe_trigger = dict(TRIGGER_CFG, epochs=max(1, TRIGGER_CFG["epochs"] // 5))
    probe_arg = dict(ARG_CFG, epochs=max(1, ARG_CFG["epochs"] // 5))

    def eval_fn(candidate_train):
        model = PipelineSpan(args.guwenbert, type_system, probe_trigger, probe_arg)
        model.fit(candidate_train, load_argument_examples(candidate_train, type_system))
        pred = model.predict(dev)
        return evaluate_full(pred, dev_gold)

    return eval_fn


class CCDAPool:
    """缓存一次性生成的 ESAM/RCSM 增强样本，并按 (use_esam,use_rcsm,use_qc)
    记忆化组合后的训练集，保证主实验与消融实验复用同一批生成样本，
    且相同设置只组合（含第三级质量控制）一次。"""

    def __init__(self, client, train, type_system, ratio, qc_eval_fn):
        self.train = train
        self.type_system = type_system
        self.qc_eval_fn = qc_eval_fn
        self.longtail, self.esam_aug, self.rcsm_aug = prepare_augmentation(
            client, train, type_system, ratio
        )
        self._cache = {}

    def get(self, use_esam, use_rcsm, use_qc):
        key = (use_esam, use_rcsm, use_qc)
        if key not in self._cache:
            self._cache[key] = compose_ccda(
                self.train, self.esam_aug, self.rcsm_aug, self.type_system,
                use_esam=use_esam, use_rcsm=use_rcsm, use_qc=use_qc,
                train_eval_fn=self.qc_eval_fn,
            )
        return self._cache[key]


def build_variants(train, type_system, pool=None):
    longtail = pool.longtail if pool is not None else LongTailSplit(train, type_system)
    variants = [
        ("Baseline", train, longtail),
        ("Random Oversampling", random_oversampling(train, type_system, longtail), longtail),
        ("EDA", eda(train, type_system, longtail), longtail),
    ]
    if pool is not None:
        variants.append(("CCDA(ESAM only)", pool.get(True, False, True), longtail))
        variants.append(("CCDA(RCSM only)", pool.get(False, True, True), longtail))
        variants.append(("CCDA(Full)", pool.get(True, True, True), longtail))
    return variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--definitions", default=None)
    parser.add_argument("--guwenbert", default="ethanyt/guwenbert-base")
    parser.add_argument("--xunzi", default="Xunzillm4cc/Xunzi-Qwen2-7B")
    parser.add_argument("--out", default="outputs/chapter5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--longtail_ratio", type=float, default=0.7,
                        help="长尾划分的累计贡献阈值（二八原则，默认 0.7）。")
    parser.add_argument("--use_gpt4o", action="store_true")
    parser.add_argument("--fast_qc", action="store_true",
                        help="跳过第三级（模型层）批次质量控制，仅保留事件层/论元层结构校验以加速；"
                             "默认执行完整三级质量控制。")
    parser.add_argument("--models", default="pipeline,joint,generative")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    type_system = TypeSystem(args.schema)
    from .resources_cfg import build_definitions_safe

    definitions = build_definitions_safe(type_system, args.definitions)
    data = normalize_dataset(load_jsonl(args.data))
    train, dev, test = split_dataset(data, type_system, seed=args.seed)
    test_gold = gold_full_events(test, type_system)

    client = None
    if args.use_gpt4o:
        from .llm.client import GPT4oClient

        client = GPT4oClient(temperature=0.7, max_tokens=512)

    # 默认启用三级质量控制（含模型层批次筛选）；--fast_qc 时仅做前两级结构校验。
    qc_eval_fn = None if args.fast_qc else make_qc_eval_fn(args, type_system, dev)
    pool = None
    if client is not None:
        # 一次性生成 ESAM/RCSM 增强样本，主实验与消融实验全程复用。
        pool = CCDAPool(client, train, type_system, args.longtail_ratio, qc_eval_fn)
    variants = build_variants(train, type_system, pool)
    model_types = args.models.split(",")

    for model_type in model_types:
        print(f"== {model_type} 主实验（事件检测F1 / 论元抽取F1 / 完整事件抽取F1）==")
        for name, variant_train, _ in variants:
            model = make_model(model_type, args, type_system, definitions, os.path.join(args.out, f"{model_type}_{name}"))
            pred = fit_predict(model, model_type, variant_train, test, type_system)
            scores = evaluate_full(pred, test_gold)
            print(f"{name:22} | {scores['detection_f1']*100:6.2f} | {scores['argument_f1']*100:6.2f} | {scores['full_f1']*100:6.2f}")

    if pool is not None:
        _run_ablation(args, type_system, definitions, pool, test, test_gold)
    _longtail_report(args, type_system, definitions, train, test, test_gold, variants, model_types)


def _run_ablation(args, type_system, definitions, pool, test, test_gold):
    # 复现论文表 5-5 / 5-6 的消融实验（Pipeline-Span 与 Generative），
    # 复用 CCDAPool 中已生成并经质量控制的训练集。
    print("== CCDA 消融实验（事件检测F1 / 论元抽取F1 / 完整事件抽取F1）==")
    settings = [
        ("CCDA(Full)", pool.get(True, True, True)),
        ("w/o ESAM", pool.get(False, True, True)),
        ("w/o RCSM", pool.get(True, False, True)),
        ("w/o 质量控制", pool.get(True, True, False)),
        ("仅保留原始样本", pool.train),
    ]
    for model_type in ("pipeline", "generative"):
        print(f"-- {model_type} --")
        for name, variant_train in settings:
            model = make_model(model_type, args, type_system, definitions,
                               os.path.join(args.out, f"abl_{model_type}_{name}"))
            pred = fit_predict(model, model_type, variant_train, test, type_system)
            scores = evaluate_full(pred, test_gold)
            print(f"{name:18} | {scores['detection_f1']*100:6.2f} | {scores['argument_f1']*100:6.2f} | {scores['full_f1']*100:6.2f}")


def _longtail_report(args, type_system, definitions, train, test, test_gold, variants, model_types):
    ccda = [v for v in variants if v[0] == "CCDA(Full)"]
    baseline = [v for v in variants if v[0] == "Baseline"][0]
    if not ccda:
        return
    print("== 长尾部分增益分析 ==")
    for model_type in model_types:
        if model_type == "joint":
            continue
        longtail = ccda[0][2]
        base_model = make_model(model_type, args, type_system, definitions, os.path.join(args.out, f"{model_type}_lt_base"))
        base_pred = fit_predict(base_model, model_type, baseline[1], test, type_system)
        aug_model = make_model(model_type, args, type_system, definitions, os.path.join(args.out, f"{model_type}_lt_aug"))
        aug_pred = fit_predict(aug_model, model_type, ccda[0][1], test, type_system)
        det_before = grouped_detection_f1(base_pred, test_gold, longtail.detection_group)
        det_after = grouped_detection_f1(aug_pred, test_gold, longtail.detection_group)
        arg_before = grouped_argument_f1(base_pred, test_gold, None, longtail.argument_group)
        arg_after = grouped_argument_f1(aug_pred, test_gold, None, longtail.argument_group)
        for group in ("CETs", "RETs"):
            print(f"{model_type:10} | {group}(检测) | {det_before.get(group,0)*100:6.2f} -> {det_after.get(group,0)*100:6.2f}")
        for group in ("CARs", "RARs"):
            print(f"{model_type:10} | {group}(论元) | {arg_before.get(group,0)*100:6.2f} -> {arg_after.get(group,0)*100:6.2f}")


if __name__ == "__main__":
    main()
