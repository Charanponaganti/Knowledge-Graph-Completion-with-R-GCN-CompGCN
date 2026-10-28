"""Training loop with AMP, clipping, early stopping, OOM fallback.

Mirrored in the Kaggle notebook. Not imported by the notebook.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_neg: int = 32
    grad_clip: float = 1.0
    patience: int = 5
    use_amp: bool = True
    seed: int = 42
    log_every: int = 50


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def sample_negatives(pos_t: torch.Tensor, n_ent: int, k: int) -> torch.Tensor:
    return torch.randint(0, n_ent, (pos_t.size(0), k), device=pos_t.device)


def bce_loss(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """pos: [B], neg: [B, K]."""
    labels_pos = torch.ones_like(pos)
    labels_neg = torch.zeros_like(neg)
    loss_pos = F.binary_cross_entropy_with_logits(pos, labels_pos)
    loss_neg = F.binary_cross_entropy_with_logits(neg, labels_neg)
    return loss_pos + loss_neg


def train_epochs(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_iter: Callable,
    n_ent: int,
    cfg: TrainConfig,
    device: torch.device,
    eval_fn: Optional[Callable] = None,
) -> Dict[str, List[float]]:
    history = {"loss": [], "val_mrr": [], "epoch_s": []}
    scaler = GradScaler(enabled=cfg.use_amp and device.type == "cuda")
    best_mrr, bad = -1.0, 0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running, n_steps = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(batch_iter(), start=1):
            try:
                with autocast(enabled=cfg.use_amp and device.type == "cuda"):
                    pos_s, neg_s = batch["pos_score"], batch["neg_score"]
                    loss = bce_loss(pos_s, neg_s)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                running += float(loss.detach())
                n_steps += 1
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[OOM] step {step}: emptying cache; skip batch. Lower BATCH_SIZE / FANOUT / NUM_NEG.")
                optimizer.zero_grad(set_to_none=True)
                continue
            if step % cfg.log_every == 0:
                print(f"  epoch {epoch} step {step} loss={running / max(n_steps, 1):.4f}")
        dt = time.time() - t0
        avg = running / max(n_steps, 1)
        history["loss"].append(avg)
        history["epoch_s"].append(dt)
        val_mrr = None
        if eval_fn is not None:
            val_mrr = eval_fn(model)
            history["val_mrr"].append(val_mrr)
            print(f"epoch {epoch:03d}  loss={avg:.4f}  val_MRR={val_mrr:.4f}  time={dt:.1f}s")
            if val_mrr > best_mrr:
                best_mrr, bad = val_mrr, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    print(f"early stopping at epoch {epoch}")
                    break
        else:
            print(f"epoch {epoch:03d}  loss={avg:.4f}  time={dt:.1f}s")

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
