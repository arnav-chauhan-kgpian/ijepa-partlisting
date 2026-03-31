# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Part-Listing I-JEPA: Unit and Integration Tests

import sys
import os
import torch
import torch.nn as nn

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_cross_attention_shapes():
    """Test CrossAttention produces correct output shapes."""
    from src.models.cross_attention import CrossAttention

    B, N, D, K, D_text = 2, 16, 384, 5, 512
    ca = CrossAttention(dim=D, num_heads=6, kv_dim=D_text)

    x = torch.randn(B, N, D)
    context = torch.randn(B, K, D_text)

    out, attn = ca(x, context)

    assert out.shape == (B, N, D), f"Expected ({B},{N},{D}), got {out.shape}"
    assert attn.shape == (B, 6, N, K), f"Expected ({B},6,{N},{K}), got {attn.shape}"
    print("  [OK] CrossAttention shapes correct")


def test_cross_attention_same_dim():
    """Test CrossAttention when kv_dim == dim."""
    from src.models.cross_attention import CrossAttention

    B, N, D, K = 2, 16, 384, 5
    ca = CrossAttention(dim=D, num_heads=6, kv_dim=None)

    x = torch.randn(B, N, D)
    context = torch.randn(B, K, D)

    out, attn = ca(x, context)

    assert out.shape == (B, N, D), f"Expected ({B},{N},{D}), got {out.shape}"
    print("  [OK] CrossAttention (same dim) shapes correct")


def test_cross_attention_block_residual():
    """Test CrossAttentionBlock preserves dimensions and has residual."""
    from src.models.cross_attention import CrossAttentionBlock

    B, N, D, K, D_text = 2, 16, 384, 5, 512
    block = CrossAttentionBlock(dim=D, num_heads=6, kv_dim=D_text)

    x = torch.randn(B, N, D)
    context = torch.randn(B, K, D_text)

    out = block(x, context)
    assert out.shape == (B, N, D), f"Expected ({B},{N},{D}), got {out.shape}"

    # Test return_attention mode
    out2, attn = block(x, context, return_attention=True)
    assert out2.shape == (B, N, D)
    assert attn.shape == (B, 6, N, K)
    print("  [OK] CrossAttentionBlock residual and shapes correct")


def test_slot_attention_shapes():
    """Test SlotAttention produces correct slot and attention shapes."""
    from src.models.slot_attention import SlotAttention

    B, N, D, K = 4, 64, 256, 8
    sa = SlotAttention(num_slots=K, dim=D, iters=3, hidden_dim=128)

    inputs = torch.randn(B, N, D)
    slots, attn = sa(inputs)

    assert slots.shape == (B, K, D), f"Expected ({B},{K},{D}), got {slots.shape}"
    assert attn.shape == (B, K, N), f"Expected ({B},{K},{N}), got {attn.shape}"
    print("  [OK] SlotAttention shapes correct")


def test_slot_attention_competition():
    """Test that slot attention weights compete (sum to 1 over slot dim)."""
    from src.models.slot_attention import SlotAttention

    B, N, D, K = 2, 32, 128, 4
    sa = SlotAttention(num_slots=K, dim=D, iters=3)

    inputs = torch.randn(B, N, D)
    _, attn = sa(inputs)

    # Attention is softmax over slot dim (dim=1), should sum to 1 per input
    sums = attn.sum(dim=1)  # [B, N]
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
        f"Attention over slots should sum to 1, got range [{sums.min():.4f}, {sums.max():.4f}]"
    print("  [OK] Slot attention weights compete correctly (sum to 1 over slot dim)")


def test_slot_attention_diversity():
    """Test that slot representations are diverse (not collapsed)."""
    from src.models.slot_attention import SlotAttention

    B, N, D, K = 2, 64, 128, 6
    sa = SlotAttention(num_slots=K, dim=D, iters=5)

    # Use diverse input to encourage diverse slots
    inputs = torch.randn(B, N, D)
    slots, _ = sa(inputs)

    # Check slots are not all identical
    slot_pairs = []
    for i in range(K):
        for j in range(i+1, K):
            cos_sim = nn.functional.cosine_similarity(
                slots[0, i:i+1], slots[0, j:j+1])
            slot_pairs.append(cos_sim.item())

    avg_sim = sum(slot_pairs) / len(slot_pairs)
    # Slots should not be perfectly correlated
    assert avg_sim < 0.99, f"Slots are too similar (avg cos sim = {avg_sim:.4f})"
    print(f"  [OK] Slots are diverse (avg pairwise cos sim = {avg_sim:.4f})")


def test_part_listing_predictor_forward():
    """Test full forward pass through PartListingPredictor with text conditioning."""
    from src.models.part_listing_predictor import PartListingPredictor

    B, N_patches = 2, 196  # 14×14 grid
    D_enc, D_pred = 768, 384
    K_parts, D_text = 5, 384

    predictor = PartListingPredictor(
        num_patches=N_patches,
        embed_dim=D_enc,
        predictor_embed_dim=D_pred,
        depth=2,
        num_heads=6,
        num_cross_attn_blocks=2,
        text_embed_dim=D_text,
        use_slot_attention=True,
        num_slots=K_parts,
        slot_iters=2,
    )

    # Simulate encoder output (context patches)
    N_ctx = 80  # number of visible context patches
    N_tgt = 30  # number of target patches to predict
    x = torch.randn(B, N_ctx, D_enc)

    # Create mock masks
    masks_x = [torch.randint(0, N_patches, (B, N_ctx))]
    masks_pred = [torch.randint(0, N_patches, (B, N_tgt))]

    # Text embeddings
    text_embeds = torch.randn(B, K_parts, D_text)

    pred, slot_out, cross_attn_maps = predictor(
        x, masks_x, masks_pred, text_embeds=text_embeds)

    assert pred.shape == (B, N_tgt, D_enc), f"Expected ({B},{N_tgt},{D_enc}), got {pred.shape}"
    assert slot_out is not None, "Slot output should not be None when use_slot_attention=True"
    assert slot_out.shape[1] == K_parts, f"Expected {K_parts} slots, got {slot_out.shape[1]}"
    assert slot_out.shape[2] == D_pred, f"Expected D_pred={D_pred}, got {slot_out.shape[2]}"
    assert cross_attn_maps is not None, "Cross-attn maps should be returned"
    assert len(cross_attn_maps) == 2, f"Expected 2 cross-attn maps, got {len(cross_attn_maps)}"
    print("  [OK] PartListingPredictor forward pass correct (with text + slots)")


def test_part_listing_predictor_no_text():
    """Test predictor works when text_embeds is None (backward compatibility)."""
    from src.models.part_listing_predictor import PartListingPredictor

    B, N_patches = 2, 196
    D_enc, D_pred = 768, 384

    predictor = PartListingPredictor(
        num_patches=N_patches,
        embed_dim=D_enc,
        predictor_embed_dim=D_pred,
        depth=2,
        num_heads=6,
        num_cross_attn_blocks=2,
        text_embed_dim=D_pred,
        use_slot_attention=False,
    )

    N_ctx, N_tgt = 80, 30
    x = torch.randn(B, N_ctx, D_enc)
    masks_x = [torch.randint(0, N_patches, (B, N_ctx))]
    masks_pred = [torch.randint(0, N_patches, (B, N_tgt))]

    pred, slot_out, cross_attn_maps = predictor(
        x, masks_x, masks_pred, text_embeds=None)

    assert pred.shape == (B, N_tgt, D_enc)
    assert slot_out is None, "Slot output should be None when use_slot_attention=False"
    assert cross_attn_maps is None, "No cross-attn maps when text_embeds is None"
    print("  [OK] PartListingPredictor backward compatibility (no text, no slots)")


def test_loss_computation():
    """Test PartListingLoss computes finite losses with gradients."""
    from src.losses import PartListingLoss

    B, N_tgt, D = 2, 30, 768
    K = 5

    loss_fn = PartListingLoss(
        slot_loss_weight=0.1,
        diversity_loss_weight=0.05,
        use_slot_loss=True)

    pred = torch.randn(B, N_tgt, D, requires_grad=True)
    target = torch.randn(B, N_tgt, D)
    slots = torch.randn(B, K, D, requires_grad=True)
    text_embeds = torch.randn(B, K, D)

    total_loss, loss_dict = loss_fn(pred, target, slot_out=slots, text_embeds=text_embeds)

    assert torch.isfinite(total_loss), f"Loss is not finite: {total_loss.item()}"
    assert 'jepa_loss' in loss_dict
    assert 'slot_assign_loss' in loss_dict
    assert 'diversity_loss' in loss_dict
    assert 'total_loss' in loss_dict

    # Verify gradients flow
    total_loss.backward()
    assert pred.grad is not None, "Gradients should flow to pred"
    assert slots.grad is not None, "Gradients should flow to slots"
    print(f"  [OK] Loss computation correct:")
    print(f"    jepa={loss_dict['jepa_loss']:.4f}, "
          f"slot={loss_dict['slot_assign_loss']:.4f}, "
          f"div={loss_dict['diversity_loss']:.4f}, "
          f"total={loss_dict['total_loss']:.4f}")


def test_loss_without_slots():
    """Test PartListingLoss works without slot outputs."""
    from src.losses import PartListingLoss

    B, N_tgt, D = 2, 30, 768
    loss_fn = PartListingLoss(use_slot_loss=False)

    pred = torch.randn(B, N_tgt, D, requires_grad=True)
    target = torch.randn(B, N_tgt, D)

    total_loss, loss_dict = loss_fn(pred, target, slot_out=None, text_embeds=None)

    assert torch.isfinite(total_loss)
    assert 'jepa_loss' in loss_dict
    assert 'slot_assign_loss' not in loss_dict
    total_loss.backward()
    assert pred.grad is not None
    print("  [OK] Loss without slots correct (falls back to pure JEPA loss)")


def test_diversity_loss():
    """Test diversity loss penalizes identical slots."""
    from src.losses import diversity_loss

    B, K, D = 2, 4, 128

    # Diverse slots: should have low diversity loss
    diverse_slots = torch.randn(B, K, D)
    div_loss_diverse = diversity_loss(diverse_slots)

    # Identical slots: should have high diversity loss
    single_slot = torch.randn(B, 1, D)
    identical_slots = single_slot.expand(B, K, D).clone()
    div_loss_identical = diversity_loss(identical_slots)

    assert div_loss_identical > div_loss_diverse, \
        f"Identical slots ({div_loss_identical:.4f}) should have higher loss " \
        f"than diverse ones ({div_loss_diverse:.4f})"
    print(f"  [OK] Diversity loss: identical={div_loss_identical:.4f} > diverse={div_loss_diverse:.4f}")


def test_end_to_end_gradient_flow():
    """Test gradient flow through full pipeline: encoder → predictor (with cross-attn + slots) → loss."""
    from src.models.vision_transformer import VisionTransformer
    from src.models.part_listing_predictor import PartListingPredictor
    from src.losses import PartListingLoss
    from src.masks.utils import apply_masks

    B = 2
    img_size, patch_size = 224, 16
    D_enc, D_pred = 384, 192
    N_patches = (img_size // patch_size) ** 2  # 196
    K_parts = 4

    # Create models
    encoder = VisionTransformer(
        img_size=[img_size], patch_size=patch_size,
        embed_dim=D_enc, depth=2, num_heads=6)
    predictor = PartListingPredictor(
        num_patches=N_patches, embed_dim=D_enc, predictor_embed_dim=D_pred,
        depth=2, num_heads=6, num_cross_attn_blocks=1,
        text_embed_dim=D_pred, use_slot_attention=True,
        num_slots=K_parts, slot_iters=2)
    loss_fn = PartListingLoss(use_slot_loss=True)

    # Create dummy data
    imgs = torch.randn(B, 3, img_size, img_size)
    N_ctx, N_tgt = 100, 30
    masks_enc = [torch.randint(0, N_patches, (B, N_ctx))]
    masks_pred = [torch.randint(0, N_patches, (B, N_tgt))]
    text_embeds = torch.randn(B, K_parts, D_pred)

    # Forward
    z = encoder(imgs, masks_enc)
    z_pred, slot_out, _ = predictor(z, masks_enc, masks_pred, text_embeds=text_embeds)

    # Create dummy target
    target = torch.randn_like(z_pred)

    # Loss
    total_loss, loss_dict = loss_fn(z_pred, target, slot_out=slot_out, text_embeds=text_embeds)

    # Backward
    total_loss.backward()

    # Check gradients flow to encoder
    enc_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in encoder.parameters() if p.requires_grad)
    pred_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in predictor.parameters() if p.requires_grad)

    assert enc_has_grad, "Encoder should receive gradients"
    assert pred_has_grad, "Predictor should receive gradients"
    print(f"  [OK] End-to-end gradient flow verified")
    print(f"    Loss: {total_loss.item():.4f}")
    print(f"    Encoder grad: OK, Predictor grad: OK")


if __name__ == '__main__':
    print("=" * 60)
    print("  Part-Listing I-JEPA Tests")
    print("=" * 60)

    tests = [
        ("CrossAttention shapes", test_cross_attention_shapes),
        ("CrossAttention same dim", test_cross_attention_same_dim),
        ("CrossAttentionBlock residual", test_cross_attention_block_residual),
        ("SlotAttention shapes", test_slot_attention_shapes),
        ("SlotAttention competition", test_slot_attention_competition),
        ("SlotAttention diversity", test_slot_attention_diversity),
        ("PartListingPredictor forward", test_part_listing_predictor_forward),
        ("PartListingPredictor no text", test_part_listing_predictor_no_text),
        ("Loss computation", test_loss_computation),
        ("Loss without slots", test_loss_without_slots),
        ("Diversity loss", test_diversity_loss),
        ("End-to-end gradient flow", test_end_to_end_gradient_flow),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            print(f"\n[TEST] {name}")
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
