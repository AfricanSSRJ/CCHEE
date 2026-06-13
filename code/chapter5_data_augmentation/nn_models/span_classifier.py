import torch
import torch.nn as nn
from transformers import AutoModel


class SpanClassifier(nn.Module):
    def __init__(self, encoder_name, num_labels, max_span_len=10, width_dim=64, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.max_span_len = max_span_len
        self.width_emb = nn.Embedding(max_span_len + 2, width_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2 + width_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def resize(self, vocab_size):
        self.encoder.resize_token_embeddings(vocab_size)

    def forward(self, input_ids, attention_mask, span_starts, span_ends):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        batch = input_ids.size(0)
        rows = torch.arange(batch, device=input_ids.device).unsqueeze(1)
        start_h = hidden[rows, span_starts]
        end_h = hidden[rows, span_ends]
        width = (span_ends - span_starts).clamp(min=0, max=self.max_span_len + 1)
        feat = torch.cat([start_h, end_h, self.width_emb(width)], dim=-1)
        return self.classifier(feat)
