"""Filtered link-prediction metrics and OGB evaluator wrapper.

Mirrored in the Kaggle notebook. Not imported by the notebook.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch


def ranks_from_scores(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """pos [B], neg [B, N] -> 1-based rank (optimistic on ties)."""
    return (neg >= pos.unsqueeze(1)).sum(dim=1) + 1


def metrics_from_ranks(ranks: torch.Tensor) -> Dict[str, float]:
    r = ranks.float()
    out = {
        "MRR": float(torch.mean(1.0 / r).item()),
        "MR": float(torch.mean(r).item()),
        "Hits@1": float((r <= 1).float().mean().item()),
        "Hits@3": float((r <= 3).float().mean().item()),
        "Hits@10": float((r <= 10).float().mean().item()),
    }
    return out


@torch.no_grad()
def filtered_tail_ranks(
    score_fn,
    triples: torch.Tensor,
    n_ent: int,
    true_tails: Dict[Tuple[int, int], set],
    device: torch.device,
    chunk: int = 2048,
) -> torch.Tensor:
    """Full filtered tail ranking. Feasible for FB15k-237; chunked over entities."""
    ranks = []
    for i in range(triples.size(0)):
        h, r, t = [int(x) for x in triples[i].tolist()]
        filt = true_tails.get((h, r), set())
        best = None
        pos = None
        for start in range(0, n_ent, chunk):
            ents = torch.arange(start, min(start + chunk, n_ent), device=device)
            s = score_fn(
                torch.full((ents.numel(),), h, device=device, dtype=torch.long),
                torch.full((ents.numel(),), r, device=device, dtype=torch.long),
                ents,
            )
            if pos is None and t >= start and t < start + chunk:
                pos = s[t - start]
            mask = torch.zeros_like(s, dtype=torch.bool)
            for e in filt:
                if start <= e < start + chunk and e != t:
                    mask[e - start] = True
            s = s.masked_fill(mask, -1e9)
            mx = s.max()
            best = mx if best is None else torch.maximum(best, mx)
        rank = 1
        # recompute properly: count how many survive above pos
        better = 0
        for start in range(0, n_ent, chunk):
            ents = torch.arange(start, min(start + chunk, n_ent), device=device)
            s = score_fn(
                torch.full((ents.numel(),), h, device=device, dtype=torch.long),
                torch.full((ents.numel(),), r, device=device, dtype=torch.long),
                ents,
            )
            mask = torch.zeros_like(s, dtype=torch.bool)
            for e in filt:
                if start <= e < start + chunk and e != t:
                    mask[e - start] = True
            s = s.masked_fill(mask, -1e9)
            if t >= start and t < start + chunk:
                pos = s[t - start]
            better += int((s > pos).sum().item())
        ranks.append(better + 1)
    return torch.tensor(ranks, dtype=torch.float32)


@torch.no_grad()
def candidate_ranks(
    score_fn,
    triples: torch.Tensor,
    neg_tails: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """OGB-style: rank the true tail among provided negative tails."""
    h, r, t = triples[:, 0].to(device), triples[:, 1].to(device), triples[:, 2].to(device)
    pos = score_fn(h, r, t)
    b, k = neg_tails.size()
    neg = score_fn(
        h.unsqueeze(1).expand(-1, k).reshape(-1),
        r.unsqueeze(1).expand(-1, k).reshape(-1),
        neg_tails.to(device).reshape(-1),
    ).view(b, k)
    return ranks_from_scores(pos, neg)


def ogb_eval(name: str, y_pred_pos: np.ndarray, y_pred_neg: np.ndarray) -> Dict[str, float]:
    from ogb.linkproppred import Evaluator

    ev = Evaluator(name=name)
    return ev.eval({"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg})
