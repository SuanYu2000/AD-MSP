"""View-conditioned readout: each semantic view pools a different feature map.

Class-agnostic view text (no class name) is the query. Shared projections
attend over Visformer patch tokens, so the N views no longer share one global vector.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def neutral_sentence(template):
    """Template with the class slot replaced by a neutral noun."""
    text = ' '.join(template.strip().split())
    if text.endswith(' of a'):
        text = text[:-5] + ' of an object'
    elif ' of a ' in text:
        text = text.replace(' of a ', ' of an object ', 1)
    else:
        text = text + ' object'
    if not text.endswith('.'):
        text += '.'
    return text


class ViewConditionedReadout(nn.Module):
    """Cross-attention from a view query onto one of the backbone feature maps."""

    def __init__(self, scale_dims, out_dim, n_views, view_scales):
        super().__init__()
        if len(view_scales) != n_views:
            raise ValueError(f'expected {n_views} view scales, got {len(view_scales)}')
        self.out_dim = out_dim
        self.n_views = n_views
        self.scale_proj = nn.ModuleList([
            nn.Linear(dim, out_dim) for dim in scale_dims
        ])
        self.query_proj = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
        self.ctx_proj = nn.Linear(512, out_dim)
        self.wq = nn.Linear(out_dim, out_dim, bias=False)
        self.wk = nn.Linear(out_dim, out_dim, bias=False)
        self.wv = nn.Linear(out_dim, out_dim, bias=False)
        self.scale = out_dim ** -0.5
        self.register_buffer('view_scales', torch.tensor(list(view_scales), dtype=torch.long))

    def set_view_text(self, view_text):
        """Fixed CLIP embedding of the class-agnostic view sentences, shape (P, 512)."""
        if 'view_text' in self._buffers:
            self.view_text = view_text.detach().float()
        else:
            self.register_buffer('view_text', view_text.detach().float())

    def view_queries(self, ctx, n_ctx_list):
        if not hasattr(self, 'view_text'):
            raise RuntimeError('call set_view_text() before forward')
        pooled = []
        for p, n_ctx in enumerate(n_ctx_list):
            pooled.append(ctx[p, :n_ctx].mean(dim=0))
        ctx_vec = torch.stack(pooled, dim=0)
        return self.query_proj(self.view_text) + self.ctx_proj(ctx_vec)

    def forward(self, maps, ctx, n_ctx_list):
        """
        maps: (stage1, stage2, stage3), each (B, C, H, W)
        ctx: prompt context (P, max_n_ctx, 512)
        returns features (B, P, D) and a list of attention weights (B, H*W)
        """
        queries = self.view_queries(ctx, n_ctx_list)
        features = []
        attentions = []
        for p in range(self.n_views):
            scale = int(self.view_scales[p])
            tokens = maps[scale]
            b, _, h, w = tokens.shape
            tok = tokens.flatten(2).transpose(1, 2)
            tok = self.scale_proj[scale](tok)
            q = self.wq(queries[p])
            k = self.wk(tok)
            v = self.wv(tok)
            logits = torch.einsum('d,bmd->bm', q, k) * self.scale
            alpha = logits.softmax(dim=-1)
            feat = torch.einsum('bm,bmd->bd', alpha, v)
            features.append(feat)
            attentions.append((alpha, (h, w)))
        return torch.stack(features, dim=1), attentions


def _entropy(alpha):
    return -(alpha * alpha.clamp(min=1e-8).log()).sum(dim=-1).mean()


def attention_entropy_loss(attentions, ratios):
    """Loose |H(alpha) - ratio * log(M)|. ratio None skips that view."""
    losses = []
    entropies = []
    for (alpha, _), ratio in zip(attentions, ratios):
        ent = _entropy(alpha)
        entropies.append(ent)
        if ratio is None:
            continue
        target = math.log(alpha.shape[-1]) * float(ratio)
        losses.append((ent - ent.new_tensor(target)).abs())
    if not losses:
        zero = attentions[0][0].new_zeros(())
        return zero, entropies
    return torch.stack(losses).mean(), entropies


def attention_cosine_penalty(attentions, common=28):
    """Penalize overlapping attention after resampling every map to a common grid."""
    grids = []
    for alpha, (h, w) in attentions:
        b = alpha.shape[0]
        spatial = alpha.view(b, 1, h, w)
        spatial = F.interpolate(spatial, size=(common, common), mode='bilinear', align_corners=False)
        spatial = spatial.flatten(1)
        spatial = spatial / spatial.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        grids.append(F.normalize(spatial, dim=-1))
    stacked = torch.stack(grids, dim=1)
    sim = torch.einsum('bph,bqh->bpq', stacked, stacked)
    n = sim.shape[-1]
    eye = torch.eye(n, device=sim.device, dtype=torch.bool)
    return sim.masked_select(~eye.unsqueeze(0)).mean()
