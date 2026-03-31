# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Extended for Part-Listing I-JEPA: Slot Attention module
# Based on "Object-Centric Learning with Slot Attention" (Locatello et al., NeurIPS 2020)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    """
    Slot Attention module for part-centric object decomposition.

    Maps unstructured input features into K discrete, exchangeable slot
    representations via iterative competitive attention. Each slot ideally
    captures one semantic part of the image.

    Key mechanism: softmax is applied over the SLOT dimension (not the input
    dimension), so slots compete to "claim" input features. This encourages
    each slot to specialize in a distinct part/region.

    Args:
        num_slots: number of slot vectors (K). Should match expected number
                   of parts (e.g., 8 for bird parts).
        dim: feature dimension of input and slots
        iters: number of iterative refinement steps (typically 3)
        hidden_dim: hidden dimension for the slot MLP refinement
        eps: epsilon for numerical stability in normalization
    """
    def __init__(self, num_slots, dim, iters=3, hidden_dim=128, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps

        # Learnable slot initialization parameters
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_mu)

        # Attention projections
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        # Recurrent update (GRU)
        self.gru = nn.GRUCell(dim, dim)

        # MLP refinement after GRU update
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

        self.scale = dim ** -0.5

    def _init_slots(self, batch_size, device, dtype):
        """Initialize slots by sampling from learned Gaussian."""
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(batch_size, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)
        return slots

    def forward(self, inputs, num_slots=None):
        """
        Args:
            inputs: feature map [B, N, D] (e.g., from predictor norm output)
            num_slots: optional override for number of slots

        Returns:
            slots: refined slot representations [B, K, D]
            attn_weights: final attention map [B, K, N] showing which inputs
                          each slot claims (useful for visualization)
        """
        B, N, D = inputs.shape
        K = num_slots or self.num_slots

        # Precompute keys and values from inputs (they don't change across iters)
        inputs_normed = self.norm_input(inputs)
        k = self.k_proj(inputs_normed)  # [B, N, D]
        v = self.v_proj(inputs_normed)  # [B, N, D]

        # Initialize slots
        slots = self._init_slots(B, inputs.device, inputs.dtype)  # [B, K, D]

        # Iterative competitive attention
        attn_weights = None
        for _ in range(self.iters):
            slots_prev = slots
            slots_normed = self.norm_slots(slots)

            # Compute attention: queries from slots, keys from inputs
            q = self.q_proj(slots_normed)  # [B, K, D]

            # Attention: softmax over SLOTS dimension (competition)
            # attn_logits: [B, K, N]
            attn_logits = torch.einsum('bkd,bnd->bkn', q, k) * self.scale
            attn_weights = F.softmax(attn_logits, dim=1)  # softmax over K (slots compete)

            # Weighted mean of values per slot
            # Normalize attention weights per slot (so they sum to 1 over N)
            attn_normalized = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + self.eps)
            updates = torch.einsum('bkn,bnd->bkd', attn_normalized, v)  # [B, K, D]

            # GRU update
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            ).reshape(B, K, D)

            # MLP refinement with residual
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn_weights
