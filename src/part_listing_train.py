# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Part-Listing I-JEPA Training Script
# Extends the I-JEPA training loop with text cross-attention and slot attention

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.part_listing_dataset import make_partimagenet
from src.losses import PartListingLoss

from src.helper import (
    load_checkpoint,
    init_part_listing_model,
    init_part_listing_opt)
from src.transforms import make_transforms

# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- PART-LISTING PARAMS
    use_part_listing = args['meta'].get('use_part_listing', True)
    text_encoder_type = args['meta'].get('text_encoder_type', 'clip')
    num_cross_attn_blocks = args['meta'].get('num_cross_attn_blocks', 2)
    use_slot_attention = args['meta'].get('use_slot_attention', False)
    num_slots = args['meta'].get('num_slots', 8)
    slot_iters = args['meta'].get('slot_iters', 3)
    slot_loss_weight = args['meta'].get('slot_loss_weight', 0.1)
    diversity_loss_weight = args['meta'].get('diversity_loss_weight', 0.05)
    clip_model_name = args['meta'].get(
        'clip_model_name', 'openai/clip-vit-base-patch16')

    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    # -- Part-listing data params
    annotation_file = args['data'].get('annotation_file', None)

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']
    patch_size = args['mask']['patch_size']
    num_enc_masks = args['mask']['num_enc_masks']
    min_keep = args['mask']['min_keep']
    enc_mask_scale = args['mask']['enc_mask_scale']
    num_pred_masks = args['mask']['num_pred_masks']
    pred_mask_scale = args['mask']['pred_mask_scale']
    aspect_ratio = args['mask']['aspect_ratio']

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    dump = os.path.join(folder, 'params-part-listing-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger with additional part-listing fields
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'jepa_loss'),
                           ('%.5f', 'slot_loss'),
                           ('%.5f', 'div_loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    # -- init model (with part-listing extensions)
    encoder, predictor, text_encoder = init_part_listing_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        text_encoder_type=text_encoder_type,
        num_cross_attn_blocks=num_cross_attn_blocks,
        use_slot_attention=use_slot_attention,
        num_slots=num_slots,
        slot_iters=slot_iters,
        clip_model_name=clip_model_name)
    target_encoder = copy.deepcopy(encoder)

    # -- init loss function
    loss_fn = PartListingLoss(
        slot_loss_weight=slot_loss_weight,
        diversity_loss_weight=diversity_loss_weight,
        use_slot_loss=use_slot_attention,
    )

    # -- make data transforms
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep)

    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter)

    # -- init data-loaders/samplers (PartImageNet)
    _, unsupervised_loader, unsupervised_sampler = make_partimagenet(
            transform=transform,
            batch_size=batch_size,
            mask_collator=mask_collator,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            annotation_file=annotation_file,
            drop_last=True)
    ipe = len(unsupervised_loader)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_part_listing_opt(
        encoder=encoder,
        predictor=predictor,
        text_encoder=text_encoder,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)

    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # Note: text_encoder is NOT wrapped in DDP if frozen (CLIP)
    # If using learned encoder, wrap it
    if text_encoder_type == 'learned':
        text_encoder = DistributedDataParallel(text_encoder, static_graph=True)

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'text_encoder': text_encoder.state_dict() if hasattr(text_encoder, 'state_dict') else None,
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        jepa_loss_meter = AverageMeter()
        slot_loss_meter = AverageMeter()
        div_loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred, part_labels) in enumerate(unsupervised_loader):

            def load_imgs():
                # -- unsupervised imgs
                imgs = udata[0].to(device, non_blocking=True)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, masks_1, masks_2)
            imgs, masks_enc, masks_pred = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))
                        B = len(h)
                        h = apply_masks(h, masks_pred)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                        return h

                def forward_context():
                    # Encode visible context patches
                    z = encoder(imgs, masks_enc)

                    # Encode part labels (text)
                    text_embeds = text_encoder(part_labels, device=device)

                    # Predict target representations with text conditioning
                    z_pred, slot_out, cross_attn_maps = predictor(
                        z, masks_enc, masks_pred, text_embeds=text_embeds)

                    return z_pred, slot_out, text_embeds

                def compute_loss(z_pred, h, slot_out, text_embeds):
                    total_loss, loss_dict = loss_fn(
                        z_pred, h,
                        slot_out=slot_out,
                        text_embeds=text_embeds)
                    total_loss = AllReduce.apply(total_loss)
                    return total_loss, loss_dict

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = forward_target()
                    z_pred, slot_out, text_embeds = forward_context()
                    loss, loss_dict = compute_loss(z_pred, h, slot_out, text_embeds)

                # Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Step 3. Momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (float(loss), loss_dict, _new_lr, _new_wd, grad_stats)

            (loss, loss_dict, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            jepa_loss_meter.update(loss_dict.get('jepa_loss', 0.))
            slot_loss_meter.update(loss_dict.get('slot_assign_loss', 0.))
            div_loss_meter.update(loss_dict.get('diversity_loss', 0.))
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                csv_logger.log(
                    epoch + 1, itr, loss,
                    loss_dict.get('jepa_loss', 0.),
                    loss_dict.get('slot_assign_loss', 0.),
                    loss_dict.get('diversity_loss', 0.),
                    maskA_meter.val, maskB_meter.val, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        '[%d, %5d] loss: %.3f (jepa: %.3f, slot: %.3f, div: %.3f) '
                        'masks: %.1f %.1f '
                        '[wd: %.2e] [lr: %.2e] '
                        '[mem: %.2e] '
                        '(%.1f ms)'
                        % (epoch + 1, itr,
                           loss_meter.avg,
                           jepa_loss_meter.avg,
                           slot_loss_meter.avg,
                           div_loss_meter.avg,
                           maskA_meter.avg,
                           maskB_meter.avg,
                           _new_wd,
                           _new_lr,
                           torch.cuda.max_memory_allocated() / 1024.**2,
                           time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()

            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f (jepa: %.3f, slot: %.3f, div: %.3f)'
                     % (loss_meter.avg, jepa_loss_meter.avg,
                        slot_loss_meter.avg, div_loss_meter.avg))
        save_checkpoint(epoch+1)


if __name__ == "__main__":
    main()
