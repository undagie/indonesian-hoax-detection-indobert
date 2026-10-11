"""Contextual word substitution via IndoBERT masked language modelling."""

from __future__ import annotations

import logging
from typing import List, Optional

from src.augment.base import Augmenter

log = logging.getLogger(__name__)


class ContextualAugmenter(Augmenter):
    name = "contextual"

    def __init__(
        self,
        model_name: str = "indolem/indobert-base-uncased",
        prob: float = 0.15,
        top_k: int = 5,
        device: Optional[str] = None,
        max_length: int = 256,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(seed=seed)
        self.model_name = model_name
        self.prob = float(prob)
        self.top_k = int(top_k)
        self.max_length = max_length
        try:
            import torch
        except ImportError as e:
            raise ImportError("PyTorch is required for contextual augmentation") from e
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = None
        self._model = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        log.info("Loading contextual MLM %s on %s", self.model_name, self.device)
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(self.device).eval()

    def augment(self, text: str) -> str:
        import torch

        self._lazy_load()
        enc = self._tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        ids = enc["input_ids"][0].clone()
        attn = enc["attention_mask"][0]
        special_ids = set(self._tok.all_special_ids)
        candidate_positions = [
            i for i, t in enumerate(ids.tolist())
            if t not in special_ids and attn[i].item() == 1
        ]
        if not candidate_positions:
            return text
        n_to_mask = max(1, int(round(self.prob * len(candidate_positions))))
        positions = self.rng.sample(candidate_positions, k=min(n_to_mask, len(candidate_positions)))

        for pos in positions:
            masked = ids.clone()
            original = masked[pos].item()
            masked[pos] = self._tok.mask_token_id
            with torch.no_grad():
                logits = self._model(masked.unsqueeze(0)).logits[0, pos]
            topk = torch.topk(logits, k=self.top_k + 1).indices.tolist()
            candidates = [t for t in topk if t != original and t not in special_ids]
            if not candidates:
                continue
            ids[pos] = candidates[self.rng.randint(0, len(candidates) - 1)]

        return self._tok.decode(ids, skip_special_tokens=True).strip()
