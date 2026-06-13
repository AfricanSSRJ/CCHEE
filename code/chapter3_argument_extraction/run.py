import argparse

from .common.io_utils import load_jsonl
from .common.normalize import normalize_dataset
from .common.schema import TypeSystem
from .common.seed import set_seed
from .common.split import split_dataset
from .dataset import load_argument_examples
from .generative_baseline import run_generative
from .train_bio import train_bio
from .train_span import train_span


DISC_CFG = {
    "lr": 2e-5,
    "batch_size": 16,
    "max_len": 256,
    "max_span_len": 10,
    "neg_ratio": 3,
    "epochs": 15,
}


def _print_row(paradigm, model, mode, p, r, f):
    print(f"{paradigm:6} | {model:32} | {mode:10} | {p*100:6.2f} | {r*100:6.2f} | {f*100:6.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--guwenbert", default="ethanyt/guwenbert-base")
    parser.add_argument("--bert", default="bert-base-chinese")
    parser.add_argument("--qwen", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--xunzi", default="Xunzillm4cc/Xunzi-Qwen2-7B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_generative", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    type_system = TypeSystem(args.schema)
    data = normalize_dataset(load_jsonl(args.data))
    train, dev, test = split_dataset(data, type_system, seed=args.seed)
    train_ex = load_argument_examples(train, type_system)
    dev_ex = load_argument_examples(dev, type_system)
    test_ex = load_argument_examples(test, type_system)

    print("paradigm | model | mode | P | R | Micro-F1")

    for name, encoder in [("BERT-base-Chinese", args.bert), ("GuwenBERT", args.guwenbert)]:
        p, r, f = train_span(DISC_CFG, train_ex, dev_ex, test_ex, type_system, encoder)
        _print_row("判别式", name + "+Span", "监督微调", p, r, f)
        p, r, f = train_bio(DISC_CFG, train_ex, dev_ex, test_ex, type_system, encoder)
        _print_row("判别式", name + "+BIO", "监督微调", p, r, f)

    if not args.skip_generative:
        for name, model in [("Qwen2.5-7B-Instruct", args.qwen), ("Xunzi-Qwen2-7B", args.xunzi)]:
            p, r, f = run_generative(model, test, type_system, shots=0)
            _print_row("生成式", name, "Zero-shot", p, r, f)
            p, r, f = run_generative(model, test, type_system, shots=2, train_pool=train)
            _print_row("生成式", name, "2-shot", p, r, f)


if __name__ == "__main__":
    main()
