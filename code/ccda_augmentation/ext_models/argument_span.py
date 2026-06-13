from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ..arg_dataset import SpanArgDataset, span_collate
from ..nn_models.encoding import add_trigger_markers
from ..nn_models.span_classifier import SpanClassifier


class ArgumentSpanModel:
    def __init__(self, encoder_name, type_system, cfg):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.type_system = type_system
        self.cfg = cfg
        self.tokenizer = add_trigger_markers(AutoTokenizer.from_pretrained(encoder_name))
        self.num_labels = len(type_system.role2id)
        self.model = SpanClassifier(encoder_name, self.num_labels, cfg["max_span_len"])
        self.model.resize(len(self.tokenizer))
        self.model.to(self.device)

    def fit(self, train_examples):
        ds = SpanArgDataset(train_examples, self.tokenizer, self.type_system, self.cfg["max_len"],
                            self.cfg["max_span_len"], self.cfg["neg_ratio"], True)
        loader = DataLoader(ds, batch_size=self.cfg["batch_size"], shuffle=True,
                            collate_fn=partial(span_collate, pad_id=self.tokenizer.pad_token_id))
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
                loss = loss_fn(logits.view(-1, self.num_labels), batch["labels"].view(-1).to(self.device))
                loss.backward()
                optimizer.step()
        return self

    @torch.no_grad()
    def predict(self, probe_examples):
        self.model.eval()
        ds = SpanArgDataset(probe_examples, self.tokenizer, self.type_system, self.cfg["max_len"],
                            self.cfg["max_span_len"], train=False)
        loader = DataLoader(ds, batch_size=self.cfg["batch_size"],
                            collate_fn=partial(span_collate, pad_id=self.tokenizer.pad_token_id))
        id2role = self.type_system.id2role
        result = {}
        for batch in loader:
            logits = self.model(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
                batch["span_starts"].to(self.device),
                batch["span_ends"].to(self.device),
            )
            preds = logits.argmax(-1).cpu().tolist()
            for i, meta in enumerate(batch["meta"]):
                args = set()
                for j, sp in enumerate(batch["spans"][i]):
                    label = preds[i][j]
                    if label != 0:
                        args.add((sp[0], sp[1], id2role[label]))
                result[meta] = args
        return result
