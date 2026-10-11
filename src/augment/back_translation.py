"""Back-translation via Helsinki-NLP Opus-MT (id -> pivot -> id). Models load lazily."""

from __future__ import annotations

import logging
from typing import List, Optional

from src.augment.base import Augmenter

log = logging.getLogger(__name__)


class BackTranslationAugmenter(Augmenter):
    """Translate text id -> pivot -> id using two pre-trained models."""

    name = "back_translation"

    def __init__(
        self,
        pivot: str = "en",
        model_id2pivot: str = "Helsinki-NLP/opus-mt-id-en",
        model_pivot2id: str = "Helsinki-NLP/opus-mt-en-id",
        device: Optional[str] = None,
        batch_size: int = 8,
        max_length: int = 256,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(seed=seed)
        self.pivot = pivot
        self.model_id2pivot = model_id2pivot
        self.model_pivot2id = model_pivot2id
        self.batch_size = batch_size
        self.max_length = max_length

        try:
            import torch
        except ImportError as e:
            raise ImportError("PyTorch is required for back-translation") from e
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._fwd = None
        self._bwd = None
        self._fwd_tok = None
        self._bwd_tok = None

    def _lazy_load(self) -> None:
        if self._fwd is not None:
            return
        from transformers import MarianMTModel, MarianTokenizer

        log.info("Loading back-translation models %s / %s on %s",
                 self.model_id2pivot, self.model_pivot2id, self.device)
        self._fwd_tok = MarianTokenizer.from_pretrained(self.model_id2pivot)
        self._fwd = MarianMTModel.from_pretrained(self.model_id2pivot).to(self.device).eval()
        self._bwd_tok = MarianTokenizer.from_pretrained(self.model_pivot2id)
        self._bwd = MarianMTModel.from_pretrained(self.model_pivot2id).to(self.device).eval()

    def _translate(self, texts: List[str], tok, model) -> List[str]:
        import torch

        outputs: List[str] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=self.max_length).to(self.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_length=self.max_length, num_beams=2)
            outputs.extend(tok.batch_decode(gen, skip_special_tokens=True))
        return outputs

    def augment(self, text: str) -> str:
        return self.augment_batch([text])[0]

    def augment_batch(self, texts: List[str]) -> List[str]:
        self._lazy_load()
        pivots = self._translate(texts, self._fwd_tok, self._fwd)
        return self._translate(pivots, self._bwd_tok, self._bwd)
