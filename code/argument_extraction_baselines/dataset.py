import random

import torch
from torch.utils.data import Dataset

from .nn_models.bio_tagger import build_bio_labels
from .nn_models.encoding import char_encode, pad_batch


def load_argument_examples(samples, type_system):
    examples = []
    for s in samples:
        text = s["text"]
        for e in s.get("events", []):
            if not type_system.known_type(e.get("label", "")):
                continue
            gold = []
            for a in e.get("arguments", []):
                so = a.get("start_offset", a.get("start"))
                eo = a.get("end_offset", a.get("end"))
                if so is not None and eo is not None:
                    gold.append((so, eo, a["role"]))
            examples.append(
                {
                    "sen_id": s["sen_id"],
                    "event_id": e["id"],
                    "text": text,
                    "trigger_span": (e["start_offset"], e["end_offset"]),
                    "event_type": e["label"],
                    "arguments": gold,
                }
            )
    return examples


def gold_argument_set(examples):
    sets = []
    for ex in examples:
        sets.append({(ex["sen_id"], ex["event_id"], s, e, r) for s, e, r in ex["arguments"]})
    return sets


class SpanArgDataset(Dataset):
    def __init__(self, examples, tokenizer, type_system, max_len=256, max_span_len=10, neg_ratio=3, train=True):
        self.examples = examples
        self.tokenizer = tokenizer
        self.type_system = type_system
        self.max_len = max_len
        self.max_span_len = max_span_len
        self.neg_ratio = neg_ratio
        self.train = train
        self.role2id = type_system.role2id

    def __len__(self):
        return len(self.examples)

    def _candidates(self, n_chars, n_valid):
        spans = []
        for s in range(n_valid):
            for e in range(s + 1, min(s + self.max_span_len, n_valid) + 1):
                spans.append((s, e))
        return spans

    def __getitem__(self, idx):
        ex = self.examples[idx]
        chars = list(ex["text"])
        input_ids, char2tok = char_encode(self.tokenizer, chars, ex["trigger_span"], self.max_len)
        n_valid = len(char2tok)
        gold_map = {(s, e): r for s, e, r in ex["arguments"] if e <= n_valid}
        spans = self._candidates(len(chars), n_valid)
        labels = [self.role2id.get(gold_map.get(sp, "O"), 0) for sp in spans]
        if self.train:
            pos = [i for i, l in enumerate(labels) if l != 0]
            neg = [i for i, l in enumerate(labels) if l == 0]
            random.shuffle(neg)
            keep = set(pos) | set(neg[: max(1, len(pos) * self.neg_ratio)])
            spans = [spans[i] for i in sorted(keep)]
            labels = [labels[i] for i in sorted(keep)]
        starts = [char2tok[s] for s, e in spans]
        ends = [char2tok[e - 1] for s, e in spans]
        return {
            "input_ids": input_ids,
            "spans": spans,
            "starts": starts,
            "ends": ends,
            "labels": labels,
            "meta": (ex["sen_id"], ex["event_id"]),
        }


def span_collate(batch, pad_id):
    seqs = [b["input_ids"] for b in batch]
    padded, mask = pad_batch(seqs, pad_id)
    max_spans = max(len(b["spans"]) for b in batch)
    starts, ends, labels, span_mask = [], [], [], []
    for b in batch:
        pad = max_spans - len(b["spans"])
        starts.append(b["starts"] + [0] * pad)
        ends.append(b["ends"] + [0] * pad)
        labels.append(b["labels"] + [-100] * pad)
        span_mask.append([1] * len(b["spans"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(padded),
        "attention_mask": torch.tensor(mask),
        "span_starts": torch.tensor(starts),
        "span_ends": torch.tensor(ends),
        "labels": torch.tensor(labels),
        "span_mask": torch.tensor(span_mask),
        "spans": [b["spans"] for b in batch],
        "meta": [b["meta"] for b in batch],
    }


class BIOArgDataset(Dataset):
    def __init__(self, examples, tokenizer, type_system, max_len=256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id, self.id2label = build_bio_labels(type_system.roles)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        chars = list(ex["text"])
        input_ids, char2tok = char_encode(self.tokenizer, chars, ex["trigger_span"], self.max_len)
        n_valid = len(char2tok)
        labels = [-100] * len(input_ids)
        for tok in char2tok:
            labels[tok] = self.label2id["O"]
        for s, e, role in ex["arguments"]:
            if e > n_valid:
                continue
            labels[char2tok[s]] = self.label2id["B-" + role]
            for c in range(s + 1, e):
                labels[char2tok[c]] = self.label2id["I-" + role]
        return {"input_ids": input_ids, "labels": labels, "char2tok": char2tok, "meta": (ex["sen_id"], ex["event_id"])}


def bio_collate(batch, pad_id):
    seqs = [b["input_ids"] for b in batch]
    padded, mask = pad_batch(seqs, pad_id)
    max_len = len(padded[0])
    labels = [b["labels"] + [-100] * (max_len - len(b["labels"])) for b in batch]
    return {
        "input_ids": torch.tensor(padded),
        "attention_mask": torch.tensor(mask),
        "labels": torch.tensor(labels),
        "char2tok": [b["char2tok"] for b in batch],
        "meta": [b["meta"] for b in batch],
    }
