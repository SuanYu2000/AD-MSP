"""Episode routing and support-reliability fusion for multi-view prompts.

Branch weights come from the support visual prototypes of the current episode,
not a global softmax. Fusion alpha grows with visual-text agreement and log(shot):
more reliable support (typical of 5-shot) trusts the visual prototype more.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EpisodeRouter(nn.Module):
    """Map the mean support prototype to N prompt-branch logits."""

    def __init__(self, feat_dim, n_prompts):
        super().__init__()
        hidden = max(feat_dim // 4, n_prompts)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_prompts),
        )
        # Zero logits => uniform branch weights at initialization.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, im_proto):
        # im_proto: (way, D) -> logits (N,)
        return self.mlp(im_proto.mean(dim=0))


class SupportFusion(nn.Module):
    """alpha = sigmoid(w_agree * agreement + w_shot * log(shot) + b)."""

    def __init__(self):
        super().__init__()
        self.w_agree = nn.Parameter(torch.zeros(1))
        # Mild shot bias before training: 1-shot leans on text, 5-shot leans on vision.
        # agreement / shot / bias are all updated by the adaptive classification loss.
        self.w_shot = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.tensor([-0.8]))

    def forward(self, im_proto, proto_sem, shot):
        # im_proto: (way, D), proto_sem: (way, N, D), both L2-normalized on the last dim.
        text_proto = F.normalize(proto_sem.mean(dim=1), dim=-1)
        agreement = (im_proto * text_proto).sum(dim=-1).mean()
        log_shot = im_proto.new_tensor(math.log(float(shot)))
        alpha = torch.sigmoid(self.w_agree * agreement + self.w_shot * log_shot + self.bias)
        return alpha.squeeze(), agreement


def routing_weights(logits):
    return F.softmax(logits, dim=0)


def routing_entropy(weights):
    return -(weights * weights.clamp(min=1e-8).log()).sum()


def routing_entropy_penalty(weights, n_prompts, floor_ratio=0.5):
    """Penalize branch weights whose entropy falls below log(N) * floor_ratio."""
    entropy = routing_entropy(weights)
    floor = math.log(n_prompts) * floor_ratio
    return F.relu(entropy.new_tensor(floor) - entropy), entropy


def aggregate_with_weights(sim_stack, weights):
    """sim_stack: (query, way, N), weights: (N,) -> (query, way)."""
    return (sim_stack * weights.view(1, 1, -1)).sum(dim=-1)
