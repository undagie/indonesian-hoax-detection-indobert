"""Transformer training loop with AMP, warmup, early stopping, and checkpoints."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.eval.metrics import compute_all_metrics

log = logging.getLogger(__name__)




def _make_grad_scaler(enabled: bool):
    """CUDA GradScaler compatible with torch 2.2+ and 2.4+ APIs."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_ctx(enabled: bool):
    """CUDA autocast context compatible with torch 2.2+ and 2.4+ APIs."""
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


@dataclass
class TrainResult:
    best_epoch: int
    best_metric: float
    best_metric_name: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    test_metrics: Optional[Dict[str, Any]] = None
    y_true: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    y_proba: Optional[np.ndarray] = None
    run_dir: Optional[str] = None

    def to_jsonable(self) -> Dict[str, Any]:
        d = asdict(self)
        for key in ("y_true", "y_pred", "y_proba"):
            v = getattr(self, key)
            if v is not None:
                d[key] = np.asarray(v).tolist()
        return d


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        *,
        loss_fn: nn.Module,
        device: torch.device,
        run_dir: str,
        batch_size: int = 16,
        eval_batch_size: int = 32,
        epochs: int = 10,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        grad_accum_steps: int = 1,
        early_stopping_patience: int = 3,
        early_stopping_metric: str = "f1_macro",
        early_stopping_mode: str = "max",
        use_amp: bool = True,
        num_workers: int = 2,
        sampler: Optional[torch.utils.data.Sampler] = None,
        log_every: int = 50,
        save_best_model: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss_fn = loss_fn.to(device) if hasattr(loss_fn, "to") else loss_fn
        self.device = device
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)

        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_mode = early_stopping_mode
        self.use_amp = use_amp and device.type == "cuda"
        self.log_every = log_every
        self.save_best_model = save_best_model

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        self.optimizer = torch.optim.AdamW(
            self._param_groups(), lr=lr, weight_decay=weight_decay
        )
        total_steps = max(1, math.ceil(len(self.train_loader) / self.grad_accum_steps)) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = _linear_warmup_decay(self.optimizer, warmup_steps, total_steps)
        self.scaler = _make_grad_scaler(self.use_amp)

        self._best_state: Optional[Dict[str, torch.Tensor]] = None

    def _param_groups(self) -> List[Dict[str, Any]]:
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        decay_p, no_decay_p = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay_p if any(nd in n for nd in no_decay) else decay_p).append(p)
        return [
            {"params": decay_p, "weight_decay": self.weight_decay},
            {"params": no_decay_p, "weight_decay": 0.0},
        ]

    def _step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        labels = batch.pop("labels")
        with _autocast_ctx(self.use_amp):
            logits = self.model(**batch)
            loss = self.loss_fn(logits, labels)
            loss = loss / self.grad_accum_steps
        return loss

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        all_logits, all_labels = [], []
        for batch in loader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            labels = batch.pop("labels")
            with _autocast_ctx(self.use_amp):
                logits = self.model(**batch)
            all_logits.append(logits.float().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        logits = np.concatenate(all_logits, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        proba = _softmax(logits)
        preds = proba.argmax(axis=1)
        metrics = compute_all_metrics(labels, preds, proba)
        return {
            "metrics": metrics,
            "logits": logits,
            "proba": proba,
            "preds": preds,
            "labels": labels,
        }

    def _improved(self, current: float, best: float) -> bool:
        if math.isnan(current):
            return False
        return current > best if self.early_stopping_mode == "max" else current < best

    def fit(self) -> TrainResult:
        best_val = -math.inf if self.early_stopping_mode == "max" else math.inf
        best_epoch = -1
        patience_left = self.early_stopping_patience
        history: List[Dict[str, Any]] = []

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            running, n_batches = 0.0, 0
            for step, batch in enumerate(self.train_loader, start=1):
                loss = self._step(batch)
                self.scaler.scale(loss).backward()
                if step % self.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                running += float(loss.item()) * self.grad_accum_steps
                n_batches += 1
                if step % self.log_every == 0:
                    log.info(
                        "epoch %d step %d/%d loss=%.4f lr=%.2e",
                        epoch, step, len(self.train_loader),
                        running / max(n_batches, 1),
                        self.optimizer.param_groups[0]["lr"],
                    )

            eval_out = self._evaluate(self.val_loader)
            val_metrics = eval_out["metrics"]
            row = {
                "epoch": epoch,
                "train_loss": running / max(n_batches, 1),
                **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))},
            }
            history.append(row)
            log.info("epoch %d | train_loss=%.4f | val_%s=%.4f",
                     epoch, row["train_loss"], self.early_stopping_metric,
                     val_metrics.get(self.early_stopping_metric, float("nan")))


            current = float(val_metrics.get(self.early_stopping_metric, float("nan")))
            if self._improved(current, best_val):
                best_val = current
                best_epoch = epoch
                patience_left = self.early_stopping_patience
                self._best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                if self.save_best_model:
                    torch.save(self._best_state, os.path.join(self.run_dir, "best_model.pt"))
            else:
                patience_left -= 1
                if patience_left <= 0:
                    log.info("Early stopping triggered at epoch %d (best=%d).", epoch, best_epoch)
                    break


        with open(os.path.join(self.run_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        _write_history_csv(history, os.path.join(self.run_dir, "history.csv"))

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)

        return TrainResult(
            best_epoch=best_epoch,
            best_metric=best_val,
            best_metric_name=self.early_stopping_metric,
            history=history,
            run_dir=self.run_dir,
        )

    @torch.no_grad()
    def predict(self, dataset: Dataset, batch_size: int = 32) -> Dict[str, np.ndarray]:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        out = self._evaluate(loader)
        return {"preds": out["preds"], "proba": out["proba"], "labels": out["labels"]}




def _linear_warmup_decay(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _write_history_csv(history: List[Dict[str, Any]], path: str) -> None:
    import csv

    if not history:
        return
    keys = list(history[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k, "") for k in keys})
