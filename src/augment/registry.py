"""Instantiate augmenters by name."""

from __future__ import annotations

from typing import Optional

from src.augment.base import Augmenter, NoOpAugmenter

AVAILABLE_AUGMENTERS = [
    "none",
    "synonym",
    "swap",
    "delete",
    "insert",
    "back_translation",
    "contextual",
]


def get_augmenter(name: str, *, seed: Optional[int] = None, **kwargs) -> Augmenter:
    """Instantiate an augmenter by name. Heavy ones import lazily."""
    name = (name or "none").lower()
    if name == "none":
        return NoOpAugmenter(seed=seed)
    if name in {"synonym", "swap", "delete", "insert"}:
        from src.augment.eda import EDAAugmenter

        return EDAAugmenter(op=name, alpha=kwargs.get("alpha", 0.15), seed=seed)
    if name == "back_translation":
        from src.augment.back_translation import BackTranslationAugmenter

        return BackTranslationAugmenter(seed=seed, **kwargs)
    if name == "contextual":
        from src.augment.contextual import ContextualAugmenter

        return ContextualAugmenter(seed=seed, **kwargs)
    raise ValueError(f"Unknown augmenter '{name}'. Available: {AVAILABLE_AUGMENTERS}")
