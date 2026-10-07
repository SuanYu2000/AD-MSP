"""Adaptive view readout and classification-driven view selection.

Every candidate attends over stage1+stage2+stage3 tokens, so where a view looks
is decided by attention. Views kept for the class prototype are the top-k whose
support prototypes separate the episode's classes.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveViewReadout(nn.Module):
    """One query per view, shared over the concatenated multi-scale token sequence."""

    def __init__(self, scale_dims, out_dim, n_views):
        super().__init__()
        self.n_views = n_views
        self.out_dim = out_dim
        self.scale_proj = nn.ModuleList([nn.Linear(dim, out_dim) for dim in scale_dims])
        self.scale_embed = nn.Parameter(torch.zeros(len(scale_dims), out_dim))
        nn.init.normal_(self.scale_embed, std=0.02)
        self.query_proj = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
        self.ctx_proj = nn.Linear(512, out_dim)
        self.wq = nn.Linear(out_dim, out_dim, bias=False)
        self.wk = nn.Linear(out_dim, out_dim, bias=False)
        self.wv = nn.Linear(out_dim, out_dim, bias=False)
        self.score_bias = nn.Parameter(torch.zeros(n_views))
        self.scale = out_dim ** -0.5

    def set_view_text(self, view_text):
        text = view_text.detach().float()
        if 'view_text' in self._buffers:
            self.view_text = text
        else:
            self.register_buffer('view_text', text)

    def view_queries(self, ctx, n_ctx_list):
        pooled = torch.stack([ctx[p, :n_ctx].mean(dim=0) for p, n_ctx in enumerate(n_ctx_list)], dim=0)
        return self.query_proj(self.view_text) + self.ctx_proj(pooled)

    def forward(self, maps, ctx, n_ctx_list):
        """
        maps: (stage1, stage2, stage3)
        returns features (B, P, D), attention (B, P, M), token counts per stage
        """
        pieces = []
        lengths = []
        for scale, feat_map in enumerate(maps):
            b, _, h, w = feat_map.shape
            tok = feat_map.flatten(2).transpose(1, 2)
            tok = self.scale_proj[scale](tok) + self.scale_embed[scale]
            pieces.append(tok)
            lengths.append(h * w)
        tokens = torch.cat(pieces, dim=1)
        queries = self.view_queries(ctx, n_ctx_list)
        logits = torch.einsum('pd,bmd->bpm', self.wq(queries), self.wk(tokens)) * self.scale
        alpha = logits.softmax(dim=-1)
        features = torch.einsum('bpm,bmd->bpd', alpha, self.wv(tokens))
        return features, alpha, lengths


def class_separation(proto):
    """Mean off-diagonal cosine distance between class prototypes. proto: (way, P, D)."""
    way, n_views, _ = proto.shape
    if way < 2:
        return proto.new_zeros(n_views)
    sim = torch.einsum('apd,bpd->pab', proto, proto)
    eye = torch.eye(way, device=proto.device, dtype=torch.bool)
    off = sim.masked_select(~eye.view(1, way, way)).view(n_views, -1)
    return (1.0 - off).mean(dim=-1)


def select_view_weights(scores, top_k):
    """Top-k by score. Softmax weights are kept only on the selected views."""
    k = min(int(top_k), scores.numel())
    soft = torch.softmax(scores, dim=0)
    index = torch.topk(scores, k).indices
    mask = torch.zeros_like(scores).scatter(0, index, 1.0)
    weights = soft * mask.detach()
    weights = weights / weights.sum().clamp(min=1e-8)
    return weights, index


def attention_overlap(alpha):
    """Mean off-diagonal cosine of attention maps. alpha: (B, P, M)."""
    normed = F.normalize(alpha, dim=-1)
    sim = torch.einsum('bpm,bqm->bpq', normed, normed)
    n = sim.shape[-1]
    eye = torch.eye(n, device=sim.device, dtype=torch.bool)
    return sim.masked_select(~eye.unsqueeze(0)).mean()


def attention_floor_penalty(alpha, floor_ratio=0.25):
    """Penalize views whose attention entropy falls below floor_ratio * log(M)."""
    ent = -(alpha * alpha.clamp(min=1e-8).log()).sum(dim=-1).mean(dim=0)
    floor = math.log(alpha.shape[-1]) * floor_ratio
    return F.relu(floor - ent).mean(), ent


def stage_attention_mass(alpha, lengths):
    """Fraction of attention on each stage. Returns (P, n_stages), averaged over the batch."""
    chunks = alpha.split(lengths, dim=-1)
    return torch.stack([chunk.sum(dim=-1).mean(dim=0) for chunk in chunks], dim=-1)
