"""Cross-routing: map N text prompts to Visformer attention head scales."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadGuidance(nn.Module):
    """
    Input:  text_features (B, N, text_dim)
    Output: head_scale (B, H), prompt_weight (B, N), route_matrix (B, N, H)
    """

    def __init__(self, text_dim, num_heads, num_prompts):
        super().__init__()
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.num_prompts = num_prompts

        self.head_ref = nn.Parameter(torch.randn(num_heads, text_dim) * 0.02)
        self.q_proj = nn.Linear(text_dim, text_dim, bias=False)
        self.k_proj = nn.Linear(text_dim, text_dim, bias=False)
        self.scale = text_dim ** -0.5

    def forward(self, text_features):
        # text_features: (B, N, D)
        B, N, D = text_features.shape
        q = self.q_proj(text_features)
        k = self.k_proj(self.head_ref)
        route_logits = torch.einsum('bnd,hd->bnh', q, k) * self.scale
        route_matrix = F.softmax(route_logits, dim=-1)

        head_scale = route_matrix.sum(dim=1)
        head_scale = head_scale * (self.num_heads / (head_scale.sum(dim=-1, keepdim=True) + 1e-8))

        prompt_weight = route_matrix.sum(dim=-1)
        prompt_weight = prompt_weight / (prompt_weight.sum(dim=-1, keepdim=True) + 1e-8)

        return head_scale, prompt_weight, route_matrix


def prompt_diversity_loss(text_features):
    """Penalize cosine similarity between different prompts."""
    B, N, _ = text_features.shape
    if N <= 1:
        return text_features.new_tensor(0.)
    t = F.normalize(text_features, dim=-1)
    sim = torch.bmm(t, t.transpose(1, 2))
    eye = torch.eye(N, device=sim.device, dtype=torch.bool).unsqueeze(0)
    off_diag = sim.masked_select(~eye).view(B, N, N - 1)
    return (off_diag ** 2).mean()


def pool_prompt_features(text_features, prompt_weight):
    """Weighted pool N prompt embeddings -> (B, D)."""
    return (text_features * prompt_weight.unsqueeze(-1)).sum(dim=1)
