import torch
import torch.nn as nn
from transformers import AutoModel


class BIOTagger(nn.Module):
    def __init__(self, encoder_name, num_labels, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def resize(self, vocab_size):
        self.encoder.resize_token_embeddings(vocab_size)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.classifier(self.dropout(hidden))


def build_bio_labels(types):
    labels = ["O"]
    for t in types:
        labels.append("B-" + t)
        labels.append("I-" + t)
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


def decode_bio(tags):
    spans = []
    start = None
    cur = None
    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("B-"):
            if start is not None:
                spans.append((start, i, cur))
            start = i
            cur = tag[2:]
        elif tag.startswith("I-") and cur == tag[2:] and start is not None:
            continue
        else:
            if start is not None:
                spans.append((start, i, cur))
            start = None
            cur = None
    return spans
