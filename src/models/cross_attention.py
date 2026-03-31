# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Extended for Part-Listing I-JEPA: Cross-Attention module
# Inspired by TI-JEPA (arXiv:2503.06380) text-to-image cross-attention

import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """
    Text-to-Image Cross-Attention (TI-JEPA style).

    Queries come from image tokens, Keys/Values come from text tokens.
    This allows image representations to be conditioned on textual
    part-label information.

    Args:
        dim: dimension of query (image) tokens
        num_heads: number of attention heads
        kv_dim: dimension of key/value (text) tokens. If None, uses dim.
        attn_drop: dropout rate for attention weights
        proj_drop: dropout rate for output projection
    """
    def __init__(self, dim, num_heads=8, kv_dim=None, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        kv_dim = kv_dim or dim

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(kv_dim, dim)
        self.v_proj = nn.Linear(kv_dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        """
        Args:
            x: image tokens [B, N, D] (queries)
            context: text tokens [B, K, D_text] (keys/values)
        Returns:
            output: text-conditioned image tokens [B, N, D]
            attn_weights: attention map [B, H, N, K]
        """
        B, N, C = x.shape
        K = context.shape[1]
        H = self.num_heads

        q = self.q_proj(x).reshape(B, N, H, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(context).reshape(B, K, H, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(B, K, H, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention: Q·K^T / √d_k
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, K]
        attn = attn.softmax(dim=-1)
        attn_weights = attn
        attn = self.attn_drop(attn)

        # Weighted sum of values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, D]
        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out, attn_weights


class CrossAttentionBlock(nn.Module):
    """
    Full cross-attention block with LayerNorm, residual connections, and MLP.

    Architecture (per block):
        x = x + CrossAttn(LN(x), LN(context))
        x = x + MLP(LN(x))

    This follows the TI-JEPA t2i cross-attention block design:
    self-norm → cross-attention → residual → norm → MLP → residual.

    Args:
        dim: dimension of image tokens
        num_heads: number of attention heads
        kv_dim: dimension of text tokens (keys/values)
        mlp_ratio: ratio of MLP hidden dim to input dim
        drop: dropout rate for MLP
        attn_drop: dropout rate for attention weights
        drop_path: stochastic depth rate
        norm_layer: normalization layer class
    """
    def __init__(self, dim, num_heads, kv_dim=None, mlp_ratio=4.0,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_q = norm_layer(dim)
        self.norm_kv = norm_layer(kv_dim or dim)
        self.cross_attn = CrossAttention(
            dim=dim, num_heads=num_heads, kv_dim=kv_dim,
            attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

        # Stochastic depth (reuse from vision_transformer if available)
        from src.models.vision_transformer import DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, context, return_attention=False):
        """
        Args:
            x: image tokens [B, N, D]
            context: text tokens [B, K, D_text]
            return_attention: if True, return attention weights
        Returns:
            x: text-conditioned image tokens [B, N, D]
            attn: (optional) attention weights [B, H, N, K]
        """
        # Cross-attention with residual
        residual = x
        cross_out, attn = self.cross_attn(self.norm_q(x), self.norm_kv(context))
        x = residual + self.drop_path(cross_out)

        # MLP with residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if return_attention:
            return x, attn
        return x
