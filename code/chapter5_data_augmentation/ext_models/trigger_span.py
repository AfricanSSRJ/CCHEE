import random
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from ..nn_models.encoding import char_encode, pad_batch
from ..nn_models.span_classifier import SpanClassifier


class TriggerSpanDataset(Dataset):
    def __init__(self, samples, tokenizer, type_system, max_len=256, max_trigger_len=6, neg_ratio=5, train=True):
        self.samples = samples
        self.tokenizer = tokenizer
        self.type_system = type_system
        self.max_len = max_len
        self.max_trigger_len = max_trigger_len
        self.neg_ratio = neg_ratio
        self.train = train
        self.type2id = {"O": 0}
        for t in type_system.fine_types:
            self.type2id[t] = len(self.type2id)
        self.id2type = {v: k for k, v in self.type2id.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        chars = list(s["text"])
        input_ids, char2tok = char_encode(self.tokenizer, chars, None, self.max_len)
        n_valid = len(char2tok)
        gold = {}
        for e in s.get("events", []):
            if self.type_system.known_type(e.get("label", "")) and e["end_offset"] <= n_valid:
                gold[(e["start_offset"], e["end_offset"])] = e["label"]
        spans = []
        for a in range(n_valid):
            for b in range(a + 1, min(a + self.max_trigger_len, n_valid) + 1):
                spans.append((a, b))
        labels = [self.type2id.get(gold.get(sp, "O"), 0) for sp in spans]
        if self.train:
            pos = [i for i, l in enumerate(labels) if l != 0]
            neg = [i for i, l in enumerate(labels) if l == 0]
            random.shuffle(neg)
            keep = sorted(set(pos) | set(neg[: max(1, len(pos) * self.neg_ratio)]))
            spans = [spans[i] for i in keep]
            labels = [labels[i] for i in keep]
        starts = [char2tok[a] for a, b in spans]
        ends = [char2tok[b - 1] for a, b in spans]
        return {"input_ids": input_ids, "spans": spans, "starts": starts, "ends": ends, "labels": labels, "sen_id": s["sen_id"]}


def _collate(batch, pad_id):
    seqs = [b["input_ids"] for b in batch]
    padded, mask = pad_batch(seqs, pad_id)
    max_spans = max(len(b["spans"]) for b in batch)
    starts, ends, labels = [], [], []
    for b in batch:
        pad = max_spans - len(b["spans"])
        starts.append(b["starts"] + [0] * pad)
        ends.append(b["ends"] + [0] * pad)
        labels.append(b["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(padded),
        "attention_mask": torch.tensor(mask),
        "span_starts": torch.tensor(starts),
        "span_ends": torch.tensor(ends),
        "labels": torch.tensor(labels),
        "spans": [b["spans"] for b in batch],
        "sen_id": [b["sen_id"] for b in batch],
    }


class TriggerDetector:
    def __init__(self, encoder_name, type_system, cfg):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.type_system = type_system
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.type2id = {"O": 0}
        for t in type_system.fine_types:
            self.type2id[t] = len(self.type2id)
        self.id2type = {v: k for k, v in self.type2id.items()}
        self.model = SpanClassifier(encoder_name, len(self.type2id), cfg["max_span_len"])
        self.model.to(self.device)

    def fit(self, train_samples):
        ds = TriggerSpanDataset(train_samples, self.tokenizer, self.type_system, self.cfg["max_len"], train=True)
        loader = DataLoader(ds, batch_size=self.cfg["batch_size"], shuffle=True,
                            collate_fn=partial(_collate, pad_id=self.tokenizer.pad_token_id))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg["lr"])
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        for _ in range(self.cfg["epochs"]):
            self.model.train()
            for batch in loader:
                optimizer.zero_grad()
                logits = self.model(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                    batch["span_starts"].to(self.device),
                    batch["span_ends"].to(self.device),
                )
                loss = loss_fn(logits.view(-1, len(self.type2id)), batch["labels"].view(-1).to(self.device))
                loss.backward()
                optimizer.step()
        return self

    @torch.no_grad()
    def predict(self, samples):
        self.model.eval()
        ds = TriggerSpanDataset(samples, self.tokenizer, self.type_system, self.cfg["max_len"], train=False)
        loader = DataLoader(ds, batch_size=self.cfg["batch_size"],
                            collate_fn=partial(_collate, pad_id=self.tokenizer.pad_token_id))
        result = {}
        for batch in loader:
            logits = self.model(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
                batch["span_starts"].to(self.device),
                batch["span_ends"].to(self.device),
            )
            preds = logits.argmax(-1).cpu().tolist()
            for i, sen_id in enumerate(batch["sen_id"]):
                triggers = []
                for j, sp in enumerate(batch["spans"][i]):
                    label = preds[i][j]
                    if label != 0:
                        triggers.append((sp[0], sp[1], self.id2type[label]))
                result[sen_id] = triggers
        return result
