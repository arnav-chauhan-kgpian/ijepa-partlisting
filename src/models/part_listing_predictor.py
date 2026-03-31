# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Extended for Part-Listing I-JEPA: Predictor with cross-attention + slot attention

import math
from functools import partial

import torch
import torch.nn as nn

from src.models.vision_transformer import (
    Block,
    get_2d_sincos_pos_embed,
)
from src.models.cross_attention import CrossAttentionBlock
from src.models.slot_attention import SlotAttention
from src.utils.tensors import trunc_normal_, repeat_interleave_batch
from src.masks.utils import apply_masks


class PartListingPredictor(nn.Module):
    """
    Extended Vision Transformer Predictor with text cross-attention
    and optional slot attention for part-listing.

    Architecture:
    1. Project encoder output (D_enc) → predictor dim (D_pred)
    2. Add positional embeddings + concatenate mask tokens
    3. Cross-attention blocks: Q=image tokens, KV=text part-label embeddings
    4. Self-attention predictor blocks (standard transformer)
    5. (Optional) Slot Attention branch for part-centric grouping
    6. Extract mask-token predictions, project back to D_enc

    Args:
        num_patches: total number of patches in the image grid
        embed_dim: encoder output dimension (D_enc)
        predictor_embed_dim: internal predictor dimension (D_pred)
        depth: number of self-attention predictor blocks
        num_heads: number of attention heads
        num_cross_attn_blocks: number of cross-attention blocks
        text_embed_dim: dimension of text embeddings (input to cross-attn KV)
        use_slot_attention: whether to use Slot Attention branch
        num_slots: number of slots for Slot Attention
        slot_iters: number of Slot Attention iterations
        slot_hidden_dim: hidden dim for Slot Attention MLP
        mlp_ratio: MLP expansion ratio
        qkv_bias: whether to use bias in QKV projections
        qk_scale: custom scale for QK attention
        drop_rate: dropout rate
        attn_drop_rate: attention dropout rate
        drop_path_rate: stochastic depth rate
        norm_layer: normalization layer class
        init_std: standard deviation for weight initialization
    """
    def __init__(
        self,
        num_patches,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        num_cross_attn_blocks=2,
        text_embed_dim=384,
        use_slot_attention=False,
        num_slots=8,
        slot_iters=3,
        slot_hidden_dim=128,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        self.predictor_embed_dim = predictor_embed_dim
        self.num_cross_attn_blocks = num_cross_attn_blocks

        # -- Project encoder dim → predictor dim
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # -- Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        # -- Sinusoidal positional embedding (fixed)
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim),
            requires_grad=False)
        predictor_pos_embed = get_2d_sincos_pos_embed(
            self.predictor_pos_embed.shape[-1],
            int(num_patches ** .5),
            cls_token=False)
        self.predictor_pos_embed.data.copy_(
            torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))

        # -- Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # -- NEW: Cross-attention blocks (text → image conditioning)
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                kv_dim=text_embed_dim,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i] if i < len(dpr) else 0.,
                norm_layer=norm_layer,
            )
            for i in range(num_cross_attn_blocks)
        ])

        # -- Self-attention predictor blocks (standard, same as original)
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        # -- Predictor output
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # -- NEW: Optional Slot Attention for part-centric grouping
        self.use_slot_attention = use_slot_attention
        self.slot_attention = None
        if use_slot_attention:
            self.slot_attention = SlotAttention(
                num_slots=num_slots,
                dim=predictor_embed_dim,
                iters=slot_iters,
                hidden_dim=slot_hidden_dim,
            )

        # -- Weight initialization
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks_x, masks, text_embeds=None):
        """
        Forward pass of the Part-Listing Predictor.

        Args:
            x: encoder output [B*num_enc_masks, N_ctx, D_enc]
            masks_x: list of context mask index tensors
            masks: list of target mask index tensors
            text_embeds: text part-label embeddings [B, K, D_text] or None

        Returns:
            pred: predicted target representations [B*..., N_tgt, D_enc]
            slot_out: slot representations [B*..., num_slots, D_enc] or None
            cross_attn_maps: list of cross-attention maps or None
        """
        assert (masks is not None) and (masks_x is not None), \
            'Cannot run predictor without mask indices'

        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks, list):
            masks = [masks]

        # -- Batch size (original, before mask-based expansion)
        B = len(x) // len(masks_x)

        # -- Project encoder dim → predictor dim
        x = self.predictor_embed(x)

        # -- Add positional embedding to context tokens
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_x)

        _, N_ctxt, D = x.shape

        # -- Concatenate mask tokens with positional embeddings
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        pred_tokens += pos_embs
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # -- NEW: Cross-attention with text embeddings (TI-JEPA style)
        cross_attn_maps = []
        if text_embeds is not None:
            # Expand text_embeds to match the repeated batch dimension
            B_expanded = x.shape[0]
            if text_embeds.shape[0] != B_expanded:
                repeat_factor = B_expanded // text_embeds.shape[0]
                text_embeds_expanded = text_embeds.repeat(repeat_factor, 1, 1)
            else:
                text_embeds_expanded = text_embeds

            for cross_blk in self.cross_attn_blocks:
                x, attn_map = cross_blk(x, text_embeds_expanded,
                                        return_attention=True)
                cross_attn_maps.append(attn_map)

        # -- Self-attention predictor blocks (original I-JEPA path)
        for blk in self.predictor_blocks:
            x = blk(x)

        x = self.predictor_norm(x)

        # -- NEW: Optional Slot Attention branch
        slot_out = None
        if self.use_slot_attention and self.slot_attention is not None:
            # Apply slot attention to full feature map
            # Slots are in predictor_embed_dim space, matching text_embeds
            slot_out, slot_attn = self.slot_attention(x)

        # -- Extract predictions for mask tokens only (original I-JEPA)
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return x, slot_out, cross_attn_maps if cross_attn_maps else None


def part_listing_predictor(**kwargs):
    """Factory function for PartListingPredictor."""
    model = PartListingPredictor(
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model
