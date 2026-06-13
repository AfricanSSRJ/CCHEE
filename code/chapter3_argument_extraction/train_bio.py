from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .common.metrics import micro_prf
from .nn_models.bio_tagger import BIOTagger, decode_bio
from .nn_models.encoding import add_trigger_markers
from .dataset import BIOArgDataset, bio_collate, gold_argument_set


def _predict(model, dataset, loader, device, id2label):
    model.eval()
    pred_by_event = {}
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            preds = logits.argmax(-1).cpu().tolist()
            for i, meta in enumerate(batch["meta"]):
                char2tok = batch["char2tok"][i]
                tok2char = {t: c for c, t in enumerate(char2tok)}
                tags = []
                for tok in char2tok:
                    tags.append(id2label[preds[i][tok]])
                spans = decode_bio(tags)
                pred_by_event.setdefault(meta, set())
                for s, e, role in spans:
                    pred_by_event[meta].add((meta[0], meta[1], s, e, role))
    return pred_by_event


def evaluate(model, examples, dataset, loader, device, id2label):
    pred_by_event = _predict(model, dataset, loader, device, id2label)
    gold = gold_argument_set(examples)
    pred = [pred_by_event.get((ex["sen_id"], ex["event_id"]), set()) for ex in examples]
    return micro_prf(pred, gold)


def train_bio(cfg, train_ex, dev_ex, test_ex, type_system, encoder_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = add_trigger_markers(AutoTokenizer.from_pretrained(encoder_name))
    train_ds = BIOArgDataset(train_ex, tokenizer, type_system, cfg["max_len"])
    dev_ds = BIOArgDataset(dev_ex, tokenizer, type_system, cfg["max_len"])
    test_ds = BIOArgDataset(test_ex, tokenizer, type_system, cfg["max_len"])
    num_labels = len(train_ds.label2id)
    id2label = train_ds.id2label

    model = BIOTagger(encoder_name, num_labels)
    model.resize(len(tokenizer))
    model.to(device)

    collate = partial(bio_collate, pad_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=cfg["batch_size"], collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    best_f1 = -1.0
    best_state = None
    for _ in range(cfg["epochs"]):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = loss_fn(logits.view(-1, num_labels), batch["labels"].view(-1).to(device))
            loss.backward()
            optimizer.step()
        _, _, f1 = evaluate(model, dev_ex, dev_ds, dev_loader, device, id2label)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return evaluate(model, test_ex, test_ds, test_loader, device, id2label)
