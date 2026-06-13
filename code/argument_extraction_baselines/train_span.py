from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .common.metrics import micro_prf
from .nn_models.encoding import add_trigger_markers
from .nn_models.span_classifier import SpanClassifier
from .dataset import SpanArgDataset, gold_argument_set, span_collate


def _predict(model, loader, device, id2role):
    model.eval()
    pred_by_event = {}
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["span_starts"].to(device),
                batch["span_ends"].to(device),
            )
            preds = logits.argmax(-1).cpu().tolist()
            for i, meta in enumerate(batch["meta"]):
                spans = batch["spans"][i]
                pred_by_event.setdefault(meta, set())
                for j, sp in enumerate(spans):
                    label = preds[i][j]
                    if label != 0:
                        pred_by_event[meta].add((meta[0], meta[1], sp[0], sp[1], id2role[label]))
    return pred_by_event


def evaluate(model, examples, loader, device, id2role):
    pred_by_event = _predict(model, loader, device, id2role)
    gold = gold_argument_set(examples)
    pred = []
    for ex in examples:
        pred.append(pred_by_event.get((ex["sen_id"], ex["event_id"]), set()))
    return micro_prf(pred, gold)


def train_span(cfg, train_ex, dev_ex, test_ex, type_system, encoder_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = add_trigger_markers(AutoTokenizer.from_pretrained(encoder_name))
    num_labels = len(type_system.role2id)
    model = SpanClassifier(encoder_name, num_labels, max_span_len=cfg["max_span_len"])
    model.resize(len(tokenizer))
    model.to(device)

    collate = partial(span_collate, pad_id=tokenizer.pad_token_id)
    train_ds = SpanArgDataset(train_ex, tokenizer, type_system, cfg["max_len"], cfg["max_span_len"], cfg["neg_ratio"], True)
    dev_ds = SpanArgDataset(dev_ex, tokenizer, type_system, cfg["max_len"], cfg["max_span_len"], train=False)
    test_ds = SpanArgDataset(test_ex, tokenizer, type_system, cfg["max_len"], cfg["max_span_len"], train=False)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=cfg["batch_size"], collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    id2role = type_system.id2role

    best_f1 = -1.0
    best_state = None
    for _ in range(cfg["epochs"]):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["span_starts"].to(device),
                batch["span_ends"].to(device),
            )
            loss = loss_fn(logits.view(-1, num_labels), batch["labels"].view(-1).to(device))
            loss.backward()
            optimizer.step()
        _, _, f1 = evaluate(model, dev_ex, dev_loader, device, id2role)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return evaluate(model, test_ex, test_loader, device, id2role)
