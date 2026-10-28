"""DistMult, R-GCN (basis), and CompGCN encoders.

Mirrored in the Kaggle notebook. Not imported by the notebook.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


CompOp = Literal["sub", "mult", "corr"]


def ccorr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Circular correlation via FFT (Nickel / CompGCN)."""
    n = a.size(-1)
    return torch.fft.irfft(torch.conj(torch.fft.rfft(a, dim=-1)) * torch.fft.rfft(b, dim=-1), n=n, dim=-1)


class DistMult(nn.Module):
    """No-GNN baseline: score(h, r, t) = <h, r, t>."""

    def __init__(self, n_ent: int, n_rel: int, dim: int, p_norm: float = 1.0):
        super().__init__()
        self.ent = nn.Embedding(n_ent, dim)
        self.rel = nn.Embedding(n_rel, dim)
        self.p_norm = p_norm
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)

    def encode(self, *args, **kwargs) -> torch.Tensor:
        return self.ent.weight

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (self.ent(h) * self.rel(r) * self.ent(t)).sum(dim=-1)

    def score_with_emb(self, h_emb: torch.Tensor, r: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return (h_emb * self.rel(r) * t_emb).sum(dim=-1)


class RGCNEncoder(nn.Module):
    """Stack of basis-decomposition R-GCN layers (Schlichtkrull et al., 2018)."""

    def __init__(self, n_ent: int, n_rel: int, dim: int, n_layers: int = 2,
                 n_bases: int = 8, dropout: float = 0.2):
        super().__init__()
        from torch_geometric.nn import RGCNConv

        self.ent = nn.Embedding(n_ent, dim)
        self.layers = nn.ModuleList()
        self.n_rel = n_rel
        for _ in range(n_layers):
            self.layers.append(
                RGCNConv(dim, dim, num_relations=n_rel, num_bases=n_bases, self_loops=True)
            )
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.ent.weight)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor,
               n_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.layers):
            h = conv(h, edge_index, edge_type)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


class CompGCNLayer(nn.Module):
    """One CompGCN layer with a composition operator (Vashishth et al., 2020)."""

    def __init__(self, dim: int, n_rel: int, op: CompOp = "mult"):
        super().__init__()
        self.op = op
        self.w_in = nn.Linear(dim, dim, bias=False)
        self.w_out = nn.Linear(dim, dim, bias=False)
        self.w_loop = nn.Linear(dim, dim, bias=False)
        self.w_rel = nn.Linear(dim, dim, bias=False)

    def compose(self, ent: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        if self.op == "sub":
            return ent - rel
        if self.op == "mult":
            return ent * rel
        if self.op == "corr":
            return ccorr(ent, rel)
        raise ValueError(self.op)

    def forward(self, x: torch.Tensor, rel: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        n = x.size(0)
        dim = x.size(1)
        n_orig_rel = rel.size(0) // 2 if rel.size(0) % 2 == 0 else rel.size(0)
        is_inv = edge_type >= n_orig_rel
        rel_id = torch.remainder(edge_type, max(n_orig_rel, 1))
        composed = self.compose(x[src], rel[rel_id])
        msg = torch.where(is_inv.unsqueeze(-1), self.w_out(composed), self.w_in(composed))
        agg = torch.zeros(n, dim, device=x.device, dtype=x.dtype)
        agg.index_add_(0, dst, msg)
        deg = torch.zeros(n, device=x.device, dtype=x.dtype).index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        deg = deg.clamp(min=1.0).unsqueeze(-1)
        out = self.w_loop(x) + agg / deg
        rel_out = self.w_rel(rel)
        return out, rel_out


class CompGCNEncoder(nn.Module):
    def __init__(self, n_ent: int, n_rel: int, dim: int, n_layers: int = 2,
                 op: CompOp = "mult", dropout: float = 0.2):
        super().__init__()
        self.ent = nn.Embedding(n_ent, dim)
        self.rel = nn.Embedding(n_rel, dim)
        self.layers = nn.ModuleList([CompGCNLayer(dim, n_rel, op) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor,
               rel_emb: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        h = x
        r = rel_emb if rel_emb is not None else self.rel.weight
        for i, layer in enumerate(self.layers):
            h, r = layer(h, r, edge_index, edge_type)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h, r


class KGModel(nn.Module):
    """Encoder + DistMult decoder. Optional TransE decoder for ablations."""

    def __init__(self, encoder: nn.Module, n_rel: int, dim: int, decoder: str = "distmult"):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.rel_dec = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.rel_dec.weight)

    def score_emb(self, h: torch.Tensor, r_id: torch.Tensor, t: torch.Tensor,
                  rel_mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        r = rel_mat[r_id] if rel_mat is not None else self.rel_dec(r_id)
        if self.decoder == "transe":
            return -torch.norm(h + r - t, p=1, dim=-1)
        return (h * r * t).sum(dim=-1)
