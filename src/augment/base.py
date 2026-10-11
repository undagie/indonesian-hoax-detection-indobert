"""Augmenter interface: ``augment(text) -> str``."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List, Optional


class Augmenter(ABC):
    """Abstract augmenter. Implementations must be deterministic given a seed."""

    name: str = "base"

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    @abstractmethod
    def augment(self, text: str) -> str:
        """Return a single augmented version of ``text``."""

    def augment_batch(self, texts: List[str]) -> List[str]:
        return [self.augment(t) for t in texts]


class NoOpAugmenter(Augmenter):
    """Identity augmenter used when augmentation is disabled."""

    name = "none"

    def augment(self, text: str) -> str:
        return text
