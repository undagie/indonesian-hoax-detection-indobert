"""Transformer encoder wrapper (IndoBERT / XLM-R / mBERT) + tokenised dataset."""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


def build_tokenizer(model_name: str):
    """Load a Hugging Face fast tokenizer for ``model_name``."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


class TextDataset(Dataset):
    """A simple tokenised text dataset returning ``input_ids``, ``attention_mask``, ``labels``."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer,
        max_len: int = 256,
    ) -> None:
        assert len(texts) == len(labels), "texts and labels must align"
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


class TransformerClassifier(nn.Module):
    """Encoder plus linear head; ``forward`` returns logits for an external loss."""

    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel

        cfg = AutoConfig.from_pretrained(model_name)
        cfg.hidden_dropout_prob = dropout
        cfg.attention_probs_dropout_prob = dropout
        self.backbone = AutoModel.from_pretrained(model_name, config=cfg)
        hidden = cfg.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)
        self.num_classes = num_classes
        self.model_name = model_name

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        out = self.backbone(**kw)


        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            hs = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float() if attention_mask is not None else None
            if mask is not None:
                pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            else:
                pooled = hs.mean(dim=1)
        return self.classifier(self.dropout(pooled))


@torch.no_grad()
def extract_embeddings(
    model: TransformerClassifier,
    dataset: TextDataset,
    *,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Return mean-pooled backbone embeddings (N x hidden) for downstream SMOTE / t-SNE."""
    device = device or next(model.parameters()).device
    model.eval()
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    out: List[np.ndarray] = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        hs = model.backbone(input_ids=ids, attention_mask=attn).last_hidden_state
        mask = attn.unsqueeze(-1).float()
        pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        out.append(pooled.cpu().numpy())
    return np.concatenate(out, axis=0)
