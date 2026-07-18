"""Training loop for ScalpTCN: BCEWithLogits(pos_weight) + AdamW, early
stopping on validation average precision (the right metric for a rare
positive rate — accuracy/loss alone reward always-predict-negative)."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss, n_batches = 0.0, 0
    probs, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb_f = yb.to(device).float()
        logits = model(xb)
        total_loss += criterion(logits, yb_f).item()
        n_batches += 1
        probs.append(torch.sigmoid(logits).cpu().numpy())
        trues.append(np.asarray(yb))
    val_loss = total_loss / max(n_batches, 1)
    if n_batches == 0:
        return float("nan"), 0.0
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(trues)
    # average_precision_score needs both classes present; an all-negative
    # (or all-positive) val batch set has no meaningful AP -> 0.0
    val_ap = (float(average_precision_score(y_true, y_prob))
              if 0 < y_true.sum() < len(y_true) else 0.0)
    return val_loss, val_ap


def train(
    model: nn.Module,
    loaders: tuple[DataLoader, DataLoader],
    epochs: int,
    lr: float,
    pos_weight: float,
    device: torch.device,
    patience: int,
    weight_decay: float = 1e-4,
    grad_clip: float = 5.0,
    verbose: bool = True,
) -> tuple[list[dict], dict | None]:
    """Train `model` in place; returns (history, best_state_dict).

    `loaders` = (train_loader, val_loader), each yielding (X, y) batches
    with X float32 [B, n_features, window_s] and y in {0, 1}. Val is used
    ONLY for early stopping (average precision) — never for gradient steps.
    `model` ends up with the best-val-AP weights loaded (mirrors the LOB
    port's train_model early-stop-restore behavior); `best_state_dict` is a
    CPU deep copy suitable for torch.save, or None if val never produced a
    usable AP (e.g. an empty/degenerate val loader).
    """
    train_loader, val_loader = loaders
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(pos_weight), device=device)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: list[dict] = []
    best_ap, best_state, bad_epochs = -1.0, None, 0

    for epoch in range(epochs):
        # train-only augmentation hook (scalp.deep.dataset.JitterDataset):
        # redraw this epoch's noise. val_loader.dataset is never touched
        # here, so val batches stay bit-identical regardless of jitter.
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        model.train()
        t0 = time.time()
        total_loss, n_batches = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb_f = yb.to(device).float()
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb_f)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        val_loss, val_ap = _evaluate(model, val_loader, device, criterion)
        rec = {
            "epoch": epoch,
            "train_loss": total_loss / max(n_batches, 1),
            "val_loss": val_loss,
            "val_ap": val_ap,
            "seconds": time.time() - t0,
        }
        history.append(rec)
        if verbose:
            print(f"  epoch {epoch:2d}  train_loss {rec['train_loss']:.4f}  "
                  f"val_loss {val_loss:.4f}  val_AP {val_ap:.4f}  "
                  f"({rec['seconds']:.1f}s)")

        if val_ap > best_ap:
            best_ap, bad_epochs = val_ap, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"  early stop at epoch {epoch} "
                          f"(best val AP {best_ap:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_state
