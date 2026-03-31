# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Extended for Part-Listing I-JEPA: Text Encoder for part labels
# Uses frozen CLIP text encoder by default, with learned encoder as fallback

import torch
import torch.nn as nn

from src.utils.tensors import trunc_normal_


class PartLabelEncoder(nn.Module):
    """
    Encodes part-label text strings into dense embeddings.

    Supports two modes:
    - 'clip': Frozen CLIP text encoder + learned linear projection
    - 'learned': Lightweight embedding + small transformer (low-memory)

    For PartImageNet, part labels are strings like:
    ["head", "body", "wing", "tail", "leg", "beak", ...]

    Args:
        embed_dim: output embedding dimension (matches predictor dim)
        encoder_type: 'clip' or 'learned'
        freeze_encoder: whether to freeze the text encoder weights
        clip_model_name: CLIP model variant to use
        vocab_size: vocabulary size for learned encoder
        max_num_parts: maximum number of part labels per image
        learned_depth: number of transformer layers for learned encoder
    """
    def __init__(
        self,
        embed_dim=384,
        encoder_type='clip',
        freeze_encoder=True,
        clip_model_name='openai/clip-vit-base-patch16',
        vocab_size=1000,
        max_num_parts=40,
        learned_depth=2,
        init_std=0.02,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.embed_dim = embed_dim
        self.max_num_parts = max_num_parts

        if encoder_type == 'clip':
            self._init_clip_encoder(clip_model_name, embed_dim, freeze_encoder)
        elif encoder_type == 'learned':
            self._init_learned_encoder(vocab_size, embed_dim, max_num_parts,
                                       learned_depth, init_std)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Use 'clip' or 'learned'.")

    def _init_clip_encoder(self, model_name, embed_dim, freeze):
        """Initialize with frozen CLIP text encoder."""
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError:
            raise ImportError(
                "CLIP text encoder requires `transformers` package. "
                "Install with: pip install transformers"
            )

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.clip_text = CLIPTextModel.from_pretrained(model_name)
        clip_dim = self.clip_text.config.hidden_size  # typically 512 for base

        if freeze:
            for param in self.clip_text.parameters():
                param.requires_grad = False
            self.clip_text.eval()

        self.freeze_encoder = freeze
        # Project CLIP dim → predictor embed dim
        self.projector = nn.Linear(clip_dim, embed_dim)

        # Cache for tokenized part labels (avoid re-tokenizing each iteration)
        self._cached_tokens = None
        self._cached_labels = None

    def _init_learned_encoder(self, vocab_size, embed_dim, max_num_parts,
                               depth, init_std):
        """Initialize lightweight learned text encoder."""
        self.freeze_encoder = False
        self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_num_parts, embed_dim))
        trunc_normal_(self.pos_embed, std=init_std)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=max(1, embed_dim // 64),
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.projector = nn.Identity()  # already in target dim

        # Simple tokenizer: maps part-label strings to token IDs
        # This is populated during training via register_vocabulary()
        self._vocab = {}
        self._next_id = 1  # 0 is padding

    def register_vocabulary(self, part_labels):
        """
        Register part-label strings for the learned encoder.
        Call this once before training with all unique part labels.

        Args:
            part_labels: list of unique part-label strings
        """
        assert self.encoder_type == 'learned', "Only for learned encoder"
        for label in part_labels:
            label_lower = label.lower().strip()
            if label_lower not in self._vocab:
                self._vocab[label_lower] = self._next_id
                self._next_id += 1

    def _tokenize_learned(self, part_labels, device):
        """Convert part-label strings to token IDs for learned encoder."""
        batch_size = len(part_labels)
        max_k = max(len(labels) for labels in part_labels)
        token_ids = torch.zeros(batch_size, max_k, dtype=torch.long, device=device)

        for i, labels in enumerate(part_labels):
            for j, label in enumerate(labels):
                label_lower = label.lower().strip()
                tid = self._vocab.get(label_lower, 1)  # fallback to 1 (unknown)
                token_ids[i, j] = tid

        return token_ids

    def forward(self, part_labels, device=None):
        """
        Encode part labels into embeddings.

        Args:
            part_labels: list of lists of part-label strings
                         e.g., [["head", "wing", "tail"], ["head", "wing", "tail"]]
                         OR pre-tokenized tensor [B, K] for learned encoder
            device: target device (inferred from parameters if None)

        Returns:
            text_embeds: [B, K, embed_dim] tensor of part-label embeddings
        """
        if device is None:
            device = next(self.parameters()).device

        if self.encoder_type == 'clip':
            return self._forward_clip(part_labels, device)
        else:
            return self._forward_learned(part_labels, device)

    def _forward_clip(self, part_labels, device):
        """Forward pass with frozen CLIP text encoder."""
        # Flatten all part labels for batch tokenization
        # part_labels: list of K strings (same labels for all images in batch)
        if isinstance(part_labels[0], list):
            # Assume same labels across batch; use first sample
            labels = part_labels[0]
            batch_size = len(part_labels)
        else:
            labels = part_labels
            batch_size = 1

        # Check cache to avoid re-tokenizing the same labels
        if self._cached_labels != labels:
            tokens = self.tokenizer(
                labels,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors='pt'
            ).to(device)
            self._cached_tokens = tokens
            self._cached_labels = labels
        else:
            tokens = {k: v.to(device) for k, v in self._cached_tokens.items()}

        # Get CLIP text embeddings
        with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
            clip_out = self.clip_text(**tokens)
            # Use pooled output (CLS token) per label: [K, clip_dim]
            text_features = clip_out.last_hidden_state[:, 0, :]  # CLS token

        # Project to predictor dimension: [K, embed_dim]
        text_embeds = self.projector(text_features)

        # Expand for batch: [B, K, embed_dim]
        text_embeds = text_embeds.unsqueeze(0).expand(batch_size, -1, -1)

        return text_embeds

    def _forward_learned(self, part_labels, device):
        """Forward pass with learned text encoder."""
        if isinstance(part_labels, torch.Tensor):
            token_ids = part_labels.to(device)
        else:
            token_ids = self._tokenize_learned(part_labels, device)

        B, K = token_ids.shape

        # Embed tokens
        x = self.word_embedding(token_ids)  # [B, K, embed_dim]

        # Add positional encoding
        x = x + self.pos_embed[:, :K, :]

        # Transformer encoding
        padding_mask = (token_ids == 0)  # mask padding tokens
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)
        x = self.projector(x)

        return x

    def train(self, mode=True):
        """Override to keep CLIP encoder frozen even during training."""
        super().train(mode)
        if self.encoder_type == 'clip' and self.freeze_encoder:
            self.clip_text.eval()
        return self
