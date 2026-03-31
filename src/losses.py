# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Extended for Part-Listing I-JEPA: Loss functions for part-listing

import torch
import torch.nn as nn
import torch.nn.functional as F


def slot_part_assignment_loss(slots, text_embeds, temperature=0.1):
    """
    Encourages each slot to correspond to a unique part label.

    Computes cosine similarity between slot representations and text
    embeddings, then uses a contrastive-style loss to encourage one-to-one
    matching between slots and part labels.

    Uses Hungarian matching (optimal assignment) to find the best
    slot-to-part mapping, then computes cross-entropy loss on that mapping.

    Args:
        slots: slot representations [B, K_slots, D]
        text_embeds: text part-label embeddings [B, K_parts, D]
        temperature: softmax temperature for similarity

    Returns:
        loss: scalar assignment loss
    """
    # Normalize for cosine similarity
    slots_norm = F.normalize(slots, dim=-1)
    text_norm = F.normalize(text_embeds, dim=-1)

    # Cosine similarity matrix: [B, K_slots, K_parts]
    sim_matrix = torch.einsum('bsd,bpd->bsp', slots_norm, text_norm)
    sim_matrix = sim_matrix / temperature

    B, K_slots, K_parts = sim_matrix.shape

    # Soft assignment loss: encourage each slot to match one part
    # and each part to be matched by one slot

    # Slot → Part assignment (each slot picks its best part)
    slot_to_part_log = F.log_softmax(sim_matrix, dim=-1)  # [B, K_slots, K_parts]
    # Part → Slot assignment (each part picks its best slot)
    part_to_slot_log = F.log_softmax(sim_matrix, dim=-2)  # [B, K_slots, K_parts]

    # Target: uniform assignment if K_slots == K_parts (one-to-one)
    # Otherwise: encourage spreading
    if K_slots == K_parts:
        # Use identity-based target with Hungarian matching
        target = _hungarian_matching(sim_matrix.detach())
    else:
        # Uniform target: each part should be equally represented
        target = torch.ones(B, K_slots, K_parts, device=slots.device)
        target = target / K_parts

    # Cross-entropy loss for assignment
    loss_s2p = -(target * slot_to_part_log).sum(dim=-1).mean()
    loss_p2s = -(target.transpose(-2, -1) * part_to_slot_log.transpose(-2, -1)).sum(dim=-1).mean()

    loss = (loss_s2p + loss_p2s) / 2.0

    return loss


def _hungarian_matching(sim_matrix):
    """
    Compute optimal one-to-one assignment using Hungarian algorithm.

    Args:
        sim_matrix: similarity scores [B, K_slots, K_parts]

    Returns:
        target: one-hot assignment matrix [B, K_slots, K_parts]
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        # Fallback: use argmax (greedy, not optimal but works)
        return _greedy_matching(sim_matrix)

    B, K_slots, K_parts = sim_matrix.shape
    target = torch.zeros_like(sim_matrix)

    # Convert to numpy for scipy
    cost = -sim_matrix.detach().cpu().numpy()

    for b in range(B):
        row_ind, col_ind = linear_sum_assignment(cost[b])
        target[b, row_ind, col_ind] = 1.0

    return target.to(sim_matrix.device)


def _greedy_matching(sim_matrix):
    """Greedy fallback for Hungarian matching."""
    B, K_slots, K_parts = sim_matrix.shape
    target = torch.zeros_like(sim_matrix)

    for b in range(B):
        used_parts = set()
        # Sort slots by max similarity (descending)
        slot_max_sim, _ = sim_matrix[b].max(dim=-1)
        slot_order = slot_max_sim.argsort(descending=True)

        for s in slot_order:
            best_part = -1
            best_sim = -float('inf')
            for p in range(K_parts):
                if p not in used_parts and sim_matrix[b, s, p] > best_sim:
                    best_sim = sim_matrix[b, s, p].item()
                    best_part = p
            if best_part >= 0:
                target[b, s, best_part] = 1.0
                used_parts.add(best_part)

    return target


def diversity_loss(slots, eps=1e-8):
    """
    Prevents slot collapse by penalizing high similarity between slot pairs.

    Computes cosine similarity between all pairs of slots and penalizes
    values close to 1 (identical slots).

    Args:
        slots: slot representations [B, K, D]
        eps: numerical stability

    Returns:
        loss: scalar diversity loss (lower = more diverse slots)
    """
    # Normalize slots
    slots_norm = F.normalize(slots, dim=-1)  # [B, K, D]

    # Pairwise cosine similarity: [B, K, K]
    sim_matrix = torch.bmm(slots_norm, slots_norm.transpose(1, 2))

    # Mask diagonal (self-similarity = 1, not penalized)
    K = slots.shape[1]
    eye = torch.eye(K, device=slots.device).unsqueeze(0)  # [1, K, K]
    sim_matrix = sim_matrix * (1 - eye)  # zero out diagonal

    # Penalize high off-diagonal similarities
    # Mean of squared similarities (pushes toward 0)
    loss = (sim_matrix ** 2).sum(dim=(-2, -1)) / (K * (K - 1) + eps)
    loss = loss.mean()

    return loss


class PartListingLoss(nn.Module):
    """
    Combined loss for Part-Listing I-JEPA.

    Components:
    1. JEPA loss: Smooth L1 between predicted and target representations
    2. Slot-Part assignment loss: encourages slot-to-part correspondence
    3. Diversity loss: prevents slot collapse

    Args:
        slot_loss_weight: weight for slot-part assignment loss (λ_slot)
        diversity_loss_weight: weight for diversity loss (λ_div)
        use_slot_loss: whether to compute slot-related losses
        temperature: temperature for slot assignment similarity
    """
    def __init__(self, slot_loss_weight=0.1, diversity_loss_weight=0.05,
                 use_slot_loss=False, temperature=0.1):
        super().__init__()
        self.slot_loss_weight = slot_loss_weight
        self.diversity_loss_weight = diversity_loss_weight
        self.use_slot_loss = use_slot_loss
        self.temperature = temperature

    def forward(self, pred, target, slot_out=None, text_embeds=None):
        """
        Compute total loss.

        Args:
            pred: predicted representations [B, N_tgt, D]
            target: target representations [B, N_tgt, D]
            slot_out: slot representations [B, K, D] or None
            text_embeds: text embeddings [B, K_parts, D] or None

        Returns:
            total_loss: scalar total loss
            loss_dict: dictionary of individual loss components
        """
        # 1. JEPA loss (same as original I-JEPA)
        jepa_loss = F.smooth_l1_loss(pred, target)

        loss_dict = {'jepa_loss': jepa_loss.item()}
        total_loss = jepa_loss

        # 2. Slot-Part assignment loss (if slot attention is active)
        if self.use_slot_loss and slot_out is not None and text_embeds is not None:
            slot_assign_loss = slot_part_assignment_loss(
                slot_out, text_embeds, temperature=self.temperature)
            div_loss = diversity_loss(slot_out)

            total_loss = total_loss + \
                self.slot_loss_weight * slot_assign_loss + \
                self.diversity_loss_weight * div_loss

            loss_dict['slot_assign_loss'] = slot_assign_loss.item()
            loss_dict['diversity_loss'] = div_loss.item()

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict
