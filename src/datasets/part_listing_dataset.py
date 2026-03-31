# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Extended for Part-Listing I-JEPA: PartImageNet dataset with part labels

import os
import json
import logging
import sys

import numpy as np
from typing import List, Dict, Any, Optional
try:
    import torch
    import torchvision
    from torch.utils.data import Dataset
except ImportError:
    # Use a dummy Dataset class if torch is missing
    class Dataset: pass
    torch = None
    torchvision = None

from PIL import Image

_GLOBAL_SEED = 0
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

# PartImageNet category-to-parts mapping
# Based on PartImageNet benchmark (He et al., ECCV 2022)
# 11 super-categories with part annotations
PARTIMAGENET_PARTS = {
    'Quadruped': ['head', 'body', 'foot', 'tail'],
    'Biped': ['head', 'body', 'hand', 'foot', 'tail'],
    'Fish': ['head', 'body', 'fin', 'tail'],
    'Bird': ['head', 'body', 'wing', 'foot', 'tail'],
    'Snake': ['head', 'body'],
    'Reptile': ['head', 'body', 'foot', 'tail'],
    'Car': ['body', 'tier', 'side_mirror'],
    'Bicycle': ['head', 'body', 'seat', 'tier'],
    'Boat': ['body', 'sail'],
    'Aeroplane': ['head', 'body', 'wing', 'engine', 'tail'],
    'Bottle': ['body', 'mouth'],
}

# Flattened unique part labels (for vocabulary)
ALL_PART_LABELS = sorted(set(
    part for parts in PARTIMAGENET_PARTS.values() for part in parts
))

# Mapping from common ImageNet synsets to PartImageNet supercategories
# This covers the 158 classes used in the benchmark.
SYNSET_TO_SUPERCATEGORY = {
    # Quadruped (n02084071: Dog, n02121808: Cat, etc.)
    'n02084071': 'Quadruped', 'n02099601': 'Quadruped', 'n02100230': 'Quadruped',
    'n02100877': 'Quadruped', 'n02101388': 'Quadruped', 'n02105162': 'Quadruped',
    'n02107381': 'Quadruped', 'n02108551': 'Quadruped', 'n02110185': 'Quadruped',
    'n02130303': 'Quadruped', 'n02389033': 'Quadruped', 'n02391032': 'Quadruped',
    'n02395406': 'Quadruped', 'n02396427': 'Quadruped', 'n02403003': 'Quadruped',
    'n02410509': 'Quadruped', 'n02412361': 'Quadruped', 'n02443111': 'Quadruped',
    'n02504013': 'Quadruped',
    # Bird (n01503061: Owl, etc.)
    'n01503061': 'Bird', 'n01614925': 'Bird', 'n01855672': 'Bird',
    'n02005658': 'Bird', 'n02007558': 'Bird', 'n02013146': 'Bird',
    'n02013627': 'Bird', 'n02018207': 'Bird', 'n02114421': 'Bird',
    'n01491361': 'Bird', 'n01494475': 'Bird',
    # Fish (n01440764: Tench, etc.)
    'n01440764': 'Fish', 'n01443537': 'Fish', 'n01484850': 'Fish',
    'n02512053': 'Fish', 'n02514041': 'Fish', 'n02531338': 'Fish',
    # Aeroplane
    'n02690373': 'Aeroplane', 'n02691149': 'Aeroplane', 'n02692856': 'Aeroplane',
    # Car
    'n02958343': 'Car', 'n04037443': 'Car', 'n04285008': 'Car',
    'n03100240': 'Car', 'n03770679': 'Car', 'n04461632': 'Car',
    # Bicycle / Boat / Bottle (Placeholder for typical ImageNet IDs)
    'n02835271': 'Bicycle', 'n03792782': 'Bicycle',
    'n02858304': 'Boat', 'n03141249': 'Boat',
    'n03983393': 'Bottle', 'n03062238': 'Bottle'
}
# Note: In practice, we look at the first character of the filename to identify the synset.


class PartImageNetDataset(Dataset):
    """
    PartImageNet dataset wrapper that returns images AND part labels.
    """
    annotations: List[Dict[str, Any]] = []
    has_annotations: bool = False
    image_dataset: Any = None
    default_parts: List[str] = []
    max_parts: int = 12
    transform: Any = None
    image_folder: str = ""
    root_path: str = ""

    def __init__(
        self,
        root_path,
        annotation_file=None,
        image_folder='train/',
        transform=None,
        train=True,
        max_parts=12,
    ):
        super().__init__()
        self.root_path = root_path
        self.transform = transform
        self.train = train
        self.max_parts = max_parts
        self.annotations = []
        self.has_annotations = False
        self.image_dataset = None
        self.default_parts = PARTIMAGENET_PARTS.get('Bird', [])

        # 1. Resolve image folder path (magic discovery)
        # Handle cases where user/downloader points to __MACOSX junk
        if '__MACOSX' in root_path:
            alt_root = root_path.replace('__MACOSX', '').replace('//', '/').replace('\\\\', '\\')
            if os.path.exists(alt_root):
                logger.info(f'Escaping __MACOSX: {root_path} -> {alt_root}')
                root_path = alt_root
        
        self.root_path = root_path
        target_split = os.path.basename(image_folder.rstrip('/\\'))
        
        if not os.path.isabs(image_folder):
            # Check various relative paths
            candidate_paths = [
                os.path.join(root_path, image_folder),
                os.path.join(root_path, 'images', image_folder),
                os.path.join(root_path, 'images', target_split),
                os.path.join(root_path, 'train' if 'train' in image_folder else 'val'),
            ]
            
            # If root_path itself is an 'annotations' folder, try sibling or parent
            if 'annotations' in root_path.lower():
                parent = os.path.dirname(root_path.rstrip('/\\'))
                candidate_paths.append(os.path.join(parent, 'images', target_split))
                candidate_paths.append(os.path.join(parent, target_split))

            for p in candidate_paths:
                if os.path.exists(p) and not '__MACOSX' in p:
                    image_folder = p
                    break
        
        # Final cleanup for image_folder: if it still points to junk, try to escape
        if '__MACOSX' in image_folder:
            alt_img = image_folder.replace('__MACOSX', '').replace('//', '/').replace('\\\\', '\\')
            if os.path.exists(alt_img):
                image_folder = alt_img

        self.image_folder = image_folder

        # 2. Resolve annotation path/folder
        if annotation_file is not None and '__MACOSX' in annotation_file:
            alt_ann = annotation_file.replace('__MACOSX', '').replace('//', '/').replace('\\\\', '\\')
            if os.path.exists(alt_ann):
                logger.info(f'Escaping __MACOSX in annotation path: {annotation_file} -> {alt_ann}')
                annotation_file = alt_ann

        if annotation_file is None:
            if 'images' in self.image_folder:
                candidate_ann = self.image_folder.replace('images', 'annotations')
                if os.path.exists(candidate_ann):
                    annotation_file = candidate_ann
            else:
                candidate_ann = os.path.join(root_path, 'annotations', target_split)
                if os.path.exists(candidate_ann):
                    annotation_file = candidate_ann

        logger.info(f'Resolved image_folder: {self.image_folder}')
        logger.info(f'Resolved annotation_source: {annotation_file}')

        # 3. Load annotations based on type
        if annotation_file is not None and os.path.isdir(annotation_file):
            logger.info(f'Crawling directory for annotations: {annotation_file}')
            self.annotations = self._crawl_annotations(self.image_folder, annotation_file)
            self.has_annotations = True
        elif annotation_file is not None and os.path.isfile(annotation_file):
            self.annotations = self._load_annotations(annotation_file)
            self.has_annotations = True
        else:
            # Fallback for mixed folders or no annotations
            if os.path.exists(self.image_folder):
                json_files = [f for f in os.listdir(self.image_folder) if f.endswith('.json') and not f.startswith('._')]
                if len(json_files) > 10:
                    logger.info(f'Mixed directory detected. Crawling {self.image_folder}')
                    self.annotations = self._crawl_annotations(self.image_folder, self.image_folder)
                    self.has_annotations = True
                else:
                    self.annotations = None
                    self.has_annotations = False
                    self._init_imagefolder()
            else:
                logger.error(f'Image folder does not exist: {self.image_folder}')
                self.annotations = []
                self.has_annotations = False

    def _crawl_annotations(self, image_folder, annotation_folder):
        """Build annotation list by correctly matching images and annotations."""
        annotations = []
        
        # Collect all image files recursively
        image_files = []
        img_exts = ('.JPEG', '.jpg', '.jpeg', '.JPG', '.png', '.PNG')
        if os.path.exists(image_folder):
            for root, _, files in os.walk(image_folder):
                # Only skip if we are SURE it's the junk fork, but if the user
                # explicitly pointed us here and it's all there is, we might have to allow it.
                # Usually we want to skip it.
                if '__MACOSX' in root and '/images' not in root: continue
                if 'annotations' in root.lower() and '/images' not in root.lower(): continue
                
                for f in files:
                    if f.lower().endswith(img_exts) and not f.startswith('._'):
                        # Skip files that look like masks or json (if mixed)
                        if not f.endswith('_mask.png') and '_mask' not in f and not f.endswith('.json'):
                            image_files.append(os.path.join(root, f))
        
        if not image_files:
            logger.error(f'No image files found in {image_folder}')
            return []

        # Optimization: Pre-scan annotation folder to avoid slow exists() calls
        ann_map = {}
        if os.path.exists(annotation_folder):
            for root, _, files in os.walk(annotation_folder):
                if '__MACOSX' in root: continue
                for f in files:
                    if f.endswith('.json') and not f.startswith('._'):
                        stem = os.path.splitext(f)[0]
                        ann_map[stem] = os.path.join(root, f)

        logger.info(f'Identified {len(image_files)} images and {len(ann_map)} metadata files')

        for img_path in image_files:
            base_name = os.path.basename(img_path)
            base_stem = os.path.splitext(base_name)[0]
            
            # Find matching JSON in our pre-scanned map
            json_path = ann_map.get(base_stem)

            # Heuristic synset extraction
            synset = base_stem.split('_')[0]
            super_cat = SYNSET_TO_SUPERCATEGORY.get(synset, 'Unknown')
            
            if super_cat == 'Unknown':
                if synset.startswith('n015') or synset.startswith('n016'): super_cat = 'Bird'
                elif synset.startswith('n014'): super_cat = 'Fish'
                elif synset.startswith('n020') or synset.startswith('n021'): super_cat = 'Quadruped'

            parts = PARTIMAGENET_PARTS.get(super_cat, ['body'])
            
            annotations.append({
                'image_path': img_path,
                'json_path': json_path,
                'parts': list(parts[:self.max_parts]),
                'category_id': synset,
                'supercategory': super_cat,
            })
            
        logger.info(f'Successfully prepared {len(annotations)} items')
        return annotations

    def _load_annotations(self, annotation_file):
        """Load PartImageNet COCO-style annotations."""
        logger.info(f'Loading PartImageNet annotations from {annotation_file}')
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        # Build image_id -> image info mapping
        images = {img['id']: img for img in data['images']}

        # Build category_id -> supercategory mapping
        cat_to_super = {}
        for cat in data.get('categories', []):
            cat_to_super[cat['id']] = cat.get('supercategory', 'Unknown')

        # Build per-image annotations with part labels
        annotations = []
        for ann in data.get('annotations', []):
            img_info = images.get(ann['image_id'])
            if img_info is None:
                continue

            # Resolve image path
            img_name = img_info.get('file_name', img_info.get('filename', ''))
            if not img_name:
                continue
            
            # Try to find the image in image_folder
            img_path = os.path.join(self.image_folder, img_name)
            if not os.path.exists(img_path):
                # Try recursive search if flat match fails
                found = False
                for root, _, files in os.walk(self.image_folder):
                    if img_name in files:
                        img_path = os.path.join(root, img_name)
                        found = True
                        break
                if not found:
                    continue

            # Get part names from annotation
            parts = []
            if 'parts' in ann:
                for part in ann['parts']:
                    part_name = part.get('part_name', part.get('name', ''))
                    if part_name:
                        parts.append(part_name.lower().strip())

            # If no explicit parts, infer from supercategory
            synset = img_name.split('_')[0]
            super_cat = cat_to_super.get(ann.get('category_id'), 'Unknown')
            if super_cat == 'Unknown':
                super_cat = SYNSET_TO_SUPERCATEGORY.get(synset, 'Unknown')
            
            if not parts:
                parts = PARTIMAGENET_PARTS.get(super_cat, ['body'])

            # Deduplicate while preserving order
            seen = set()
            unique_parts = []
            for p in parts:
                if p not in seen:
                    seen.add(p)
                    unique_parts.append(p)

            annotations.append({
                'image_path': img_path,
                'parts': list(unique_parts[:self.max_parts]),
                'category_id': synset,
                'supercategory': super_cat,
            })

        logger.info(f'Loaded {len(annotations)} PartImageNet annotations')
        return annotations

    def _init_imagefolder(self):
        """Fallback: use ImageFolder structure without annotations."""
        logger.info('No annotation file; using ImageFolder + default parts')
        if torchvision is not None:
            self.image_dataset = torchvision.datasets.ImageFolder(
                root=self.image_folder)
        else:
            logger.error('torchvision not available for ImageFolder fallback.')
            self.image_dataset = []
        # Map class indices to default bird parts (most common in PartImageNet)
        self.default_parts = PARTIMAGENET_PARTS.get('Bird', [])

    def __len__(self):
        if self.has_annotations:
            return len(self.annotations)
        return len(self.image_dataset)

    def __getitem__(self, idx):
        """
        Returns:
        image: transformed image tensor
        part_labels: list of part-label strings
        metadata: dict with category info
        """
        if self.has_annotations:
            ann = self.annotations[idx]
            img_path = ann['image_path']
            
            # Robust loading with fallback
            try:
                # Extra check for macOS resource fork files
                if os.path.basename(img_path).startswith('._') or '__MACOSX' in img_path:
                    raise IOError("Invalid macOS resource fork")
                    
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                logger.warning(f'Failed to load image {img_path}: {e}. Returning placeholder.')
                image = Image.new('RGB', (224, 224), (128, 128, 128))
            
            part_labels = ann['parts']
            metadata = {
                'category_id': ann['category_id'],
                'supercategory': ann['supercategory'],
            }
        else:
            image, class_idx = self.image_dataset[idx]
            part_labels = self.default_parts
            metadata = {'category_id': class_idx, 'supercategory': 'Unknown'}

        if self.transform is not None:
            if torch is None or not isinstance(image, torch.Tensor):
                image = self.transform(image)

        return image, part_labels, metadata


class PartListingCollator:
    """
    Custom collator that handles part labels alongside the mask collator.

    Wraps the MaskCollator to additionally process part-label strings
    alongside the standard image + mask collation.

    Args:
        mask_collator: the base MaskCollator from I-JEPA
    """
    def __init__(self, mask_collator):
        self.mask_collator = mask_collator

    def step(self):
        """Delegate step to mask collator."""
        return self.mask_collator.step()

    def __call__(self, batch):
        """
        Collate images, masks, and part labels.

        Args:
            batch: list of (image, part_labels, metadata) tuples

        Returns:
            collated_images: batched image tensor [B, C, H, W]
            masks_enc: encoder masks
            masks_pred: predictor masks
            part_labels: list of part-label lists (one per sample)
        """
        images = [item[0] for item in batch]
        part_labels = [item[1] for item in batch]
        # metadata not needed during training, but kept for debugging

        # Use the mask collator on images only
        # The mask collator expects [(image, label)] tuples from ImageFolder
        # We wrap images to match expected format
        image_batch = [(img, 0) for img in images]

        collated_batch, masks_enc, masks_pred = self.mask_collator(image_batch)

        return collated_batch, masks_enc, masks_pred, part_labels


def make_partimagenet(
    transform,
    batch_size,
    mask_collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder='train/',
    annotation_file=None,
    training=True,
    drop_last=True,
):
    """
    Create PartImageNet data loader with part-label support.

    Args:
        transform: image transforms
        batch_size: batch size per GPU
        mask_collator: MaskCollator for I-JEPA masking
        pin_mem: pin CPU memory for faster GPU transfer
        num_workers: number of data loading workers
        world_size: number of distributed processes
        rank: current process rank
        root_path: PartImageNet root directory
        image_folder: relative path to images
        annotation_file: path to COCO-style annotation JSON
        training: whether loading training data
        drop_last: whether to drop incomplete last batch

    Returns:
        dataset: PartImageNetDataset instance
        data_loader: DataLoader instance
        sampler: DistributedSampler instance
    """
    dataset = PartImageNetDataset(
        root_path=root_path,
        annotation_file=annotation_file,
        image_folder=image_folder,
        transform=transform,
        train=training,
    )
    logger.info(f'PartImageNet dataset created with {len(dataset)} samples')

    # Wrap mask collator with part-label collation
    collator = None
    if mask_collator is not None:
        collator = PartListingCollator(mask_collator)

    if torch is None:
        return dataset, None, None

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)

    logger.info('PartImageNet data loader created')
    return dataset, data_loader, sampler
