from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from ..nn_models.bio_tagger import BIOTagger, decode_bio
from ..nn_models.encoding import char_encode, pad_batch


def build_joint_labels(type_system):
    labels = ["O"]
    for t in type_system.fine_types:
        labels += ["B-TRG-" + t, "I-TRG-" + t]
    for r in type_system.roles:
        labels += ["B-ARG-" + r, "I-ARG-" + r]
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


class JointBIODataset(Dataset):
    def __init__(self, samples, tokenizer, type_system, label2id, max_len=256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.type_system = type_system
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        chars = list(s["text"])
        input_ids, char2tok = char_encode(self.tokenizer, chars, None, self.max_len)
        n_valid = len(char2tok)
        labels = [-100] * len(input_ids)
        for tok in char2tok:
            labels[tok] = self.label2id["O"]
        for e in s.get("events", []):
            if not self.type_system.known_type(e.get("label", "")):
                continue
            full = e["label"]
            if e["end_offset"] <= n_valid:
                labels[char2tok[e["start_offset"]]] = self.label2id["B-TRG-" + full]
                for c in range(e["start_offset"] + 1, e["end_offset"]):
                    labels[char2tok[c]] = self.label2id["I-TRG-" + full]
            for a in e.get("arguments", []):
                a_start = a.get("start_offset", a.get("start"))
                a_end = a.get("end_offset", a.get("end"))
                if a_start is None or a_end is None or a_end > n_valid:
                    continue
                b_tag = "B-ARG-" + a["role"]
                if b_tag not in self.label2id:
                    continue
                labels[char2tok[a_start]] = self.label2id[b_tag]
                for c in range(a_start + 1, a_end):
                    labels[char2tok[c]] = self.label2id["I-ARG-" + a["role"]]
        return {"input_ids": input_ids, "labels": labels, "char2tok": char2tok, "text": s["text"], "sen_id": s["sen_id"]}


def _collate(batch, pad_id):
    seqs = [b["input_ids"] for b in batch]
    padded, mask = pad_batch(seqs, pad_id)
    max_len = len(padded[0])
    labels = [b["labels"] + [-100] * (max_len - len(b["labels"])) for b in batch]
    return {
        "input_ids": torch.tensor(padded),
        "attention_mask": torch.tensor(mask),
        "labels": torch.tensor(labels),
        "char2tok": [b["char2tok"] for b in batch],
        "text": [b["text"] for b in batch],
        "sen_id": [b["sen_id"] for b in batch],
    }


def _reconstruct(triggers, arguments, type_system):
    events = []
    for ts, te, full in triggers:
        events.append({"trigger": (ts, te), "type": full, "arguments": set()})
    for as_, ae, role in arguments:
        candidates = [
            e for e in events if role in type_system.roles_of(e["type"])
        ]
        if not candidates:
            continue
        anchor = (as_ + ae) / 2
        target = min(candidates, key=lambda e: abs((e["trigger"][0] + e["trigger"][1]) / 2 - anchor))
        target["arguments"].add((as_, ae, role))
    return events


class JointBIO:
    def __init__(self, encoder_name, type_system, cfg):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.type_system = type_system
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.label2id, self.id2label = build_joint_labels(type_system)
        self.model = BIOTagger(encoder_name, len(self.label2id))
        self.model.to(self.device)

    def fit(self, train_samples):
        ds = JointBIODataset(train_samples, self.tokenizer, self.type_system, self.label2id, self.cfg["max_len"])
        loader = DataLoader(ds, batch_size=self.cfg["batch_size"], shuffle=True,
                            collate_fn=partial(_collate, pad_id=self.tokenizer.pad_token_id))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg["lr"])
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        for _ in range(self.cfg["epochs"]):
            self.model.train()
            for batch in loader:
                optimizer.zero_grad()
                logits = self.model(batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device))
                loss = loss_fn(logits.view(-1, len(self.label2id)), batch["labels"].view(-1).to(self.device))
                loss.backward()
                optimizer.step()
        return self

    @torch.no_grad()
    def predict(self, samples):
        self.model.eval()
        ds = JointBIODataset(samples, self.tokenizer, self.type_system, self.label2id, self.cfg["max_len"])
        loader = DataLoader(ds, batch_size=self.cfg["batch_size"],
                            collate_fn=partial(_collate, pad_id=self.tokenizer.pad_token_id))
        per_sentence = {}
        for batch in loader:
            logits = self.model(batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device))
            preds = logits.argmax(-1).cpu().tolist()
            for i, sen_id in enumerate(batch["sen_id"]):
                char_tags = [self.id2label[preds[i][t]] for t in batch["char2tok"][i]]
                triggers, arguments = [], []
                for s, e, lab in decode_bio(char_tags):
                    if lab.startswith("TRG-"):
                        triggers.append((s, e, lab[4:]))
                    elif lab.startswith("ARG-"):
                        arguments.append((s, e, lab[4:]))
                events = _reconstruct(triggers, arguments, self.type_system)
                per_sentence[sen_id] = [
                    {"sen_id": sen_id, "trigger": e["trigger"], "type": e["type"],
                     "arguments": frozenset(e["arguments"])}
                    for e in events
                ]
        return [per_sentence.get(s["sen_id"], []) for s in samples]
