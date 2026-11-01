"""
CelebA Dataset Adapter
======================
Same interface as CUBDataset, so the whole pipeline (perception, predicates,
certified trees, evaluation) runs unchanged.

- Concepts: the 40 per-image binary attributes (+/-1 in list_attr_celeba.txt),
  bound to landmark-derived part regions (left_eye / right_eye / nose / mouth)
  or the whole-face region for global attributes (hair, age, accessories...).
- Parts: boxes centered on the 5 aligned landmarks (mouth pair merged);
  'face' = full image, so global attributes ground like any other region.
- Labels: a held-out TARGET attribute (default 'Male') becomes the class
  label (0/1) and is EXCLUDED from the concept vocabulary.
- Splits: official list_eval_partition.txt (0=train, 1=val, 2=test), with
  configurable subset sizes for fast local runs.
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

from .cub import CUBSample, get_cub_transforms

CELEBA_PARTS: List[str] = ["left_eye", "right_eye", "nose", "mouth", "face"]

#: attribute -> grounding part; everything not listed binds to 'face'.
CELEBA_ATTR_PART: Dict[str, str] = {
    "Arched_Eyebrows": "left_eye", "Bushy_Eyebrows": "left_eye",
    "Bags_Under_Eyes": "left_eye", "Narrow_Eyes": "left_eye",
    "Eyeglasses": "left_eye",
    "Big_Nose": "nose", "Pointy_Nose": "nose",
    "Big_Lips": "mouth", "Mouth_Slightly_Open": "mouth", "Smiling": "mouth",
    "Wearing_Lipstick": "mouth",
    "5_o_Clock_Shadow": "face", "Attractive": "face", "Bald": "face",
    "Bangs": "face", "Black_Hair": "face", "Blond_Hair": "face", "Blurry": "face",
    "Brown_Hair": "face", "Chubby": "face", "Double_Chin": "face",
    "Goatee": "face", "Gray_Hair": "face", "Heavy_Makeup": "face",
    "High_Cheekbones": "face", "Mustache": "face", "No_Beard": "face",
    "Oval_Face": "face", "Pale_Skin": "face", "Receding_Hairline": "face",
    "Rosy_Cheeks": "face", "Sideburns": "face", "Straight_Hair": "face",
    "Wavy_Hair": "face", "Wearing_Earrings": "face", "Wearing_Hat": "face",
    "Wearing_Necklace": "face", "Wearing_Necktie": "face", "Young": "face",
    "Male": "face",
}

# Landmark boxes are centered on the annotated point with these normalized sizes
_PART_SIZE: Dict[str, Tuple[float, float]] = {  # (h, w)
    "left_eye": (0.10, 0.13), "right_eye": (0.10, 0.13),
    "nose": (0.15, 0.13), "mouth": (0.10, 0.22),
}


class CelebADataset(Dataset):
    """CelebA with CUBDataset-compatible interface (lazy image decode)."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: int = 224,
        use_mock_if_missing: bool = True,
        num_mock_samples: int = 120,
        seed: int = 42,
        target_attribute: str = "Male",
        max_samples: int = 0,
    ):
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.transform = get_cub_transforms(image_size, is_train=False)
        self.target_attribute = target_attribute
        self.part_names = CELEBA_PARTS
        self.num_classes = 2
        self.species_list = ["no", "yes"]

        img_root = os.path.join(data_dir, "img_align_celeba")
        attr_file = os.path.join(data_dir, "list_attr_celeba.txt")
        if not (os.path.isdir(img_root) and os.path.exists(attr_file)):
            raise FileNotFoundError(
                f"CelebA not found at '{data_dir}' (need img_align_celeba/ and "
                f"list_attr_celeba.txt)."
            )

        with open(attr_file) as f:
            f.readline()  # count line
            self.attr_names = f.readline().split()
            self._attr_by_id: Dict[str, np.ndarray] = {}
            self._ids: List[str] = []
            for line in f:
                parts = line.split()
                if len(parts) != len(self.attr_names) + 1:
                    continue
                self._ids.append(parts[0])
                self._attr_by_id[parts[0]] = (np.array([int(v) for v in parts[1:]]) > 0).astype(np.float32)

        part_file = os.path.join(data_dir, "list_landmarks_align_celeba.txt")
        self._landmarks: Dict[str, np.ndarray] = {}
        if os.path.exists(part_file):
            with open(part_file) as f:
                f.readline(); f.readline()
                for line in f:
                    v = line.split()
                    if len(v) == 11:
                        self._landmarks[v[0]] = np.array([float(x) for x in v[1:]])
        self._img_w, self._img_h = 178, 218  # aligned CelebA format

        split_file = os.path.join(data_dir, "list_eval_partition.txt")
        want = {"train": 0, "val": 1, "test": 2}[split]
        self._partition: Dict[str, int] = {}
        if os.path.exists(split_file):
            with open(split_file) as f:
                for line in f:
                    a, b = line.split()
                    self._partition[a] = int(b)

        # Concept vocabulary: all attributes except the target, vocab order fixed
        self.concept_names = [a for a in self.attr_names if a != target_attribute]
        self.attributes = self.concept_names
        self.num_attributes = len(self.concept_names)
        t_idx = self.attr_names.index(target_attribute)

        self.samples: List[dict] = []
        for fname in self._ids:
            if self._partition.get(fname, 0) != want:
                continue
            self.samples.append({
                "sample_id": int(fname.split(".")[0]),
                "fname": fname,
                "label": int(self._attr_by_id[fname][t_idx]),
                "concepts": self._attr_by_id[fname][[self.attr_names.index(c) for c in self.concept_names]],
            })
        if max_samples and len(self.samples) > max_samples:
            rng = np.random.RandomState(seed)
            keep = rng.choice(len(self.samples), size=max_samples, replace=False)
            self.samples = [self.samples[i] for i in sorted(keep)]

    # ---- CUBDataset-compatible accessors -----------------------------------
    def meta(self, i: int):
        """(sample_id, part_boxes, part_visibilities, concept_labels) — no decode."""
        rec = self.samples[i]
        boxes, vis = self.part_boxes_for(rec["fname"])
        return rec["sample_id"], boxes, vis, rec["concepts"]

    def part_boxes_for(self, fname: str):
        boxes: Dict[str, Tuple[float, float, float, float]] = {}
        vis: Dict[str, float] = {}
        lm = self._landmarks.get(fname)  # lefteye_x,y righteye_x,y nose_x,y lmouth_x,y rmouth_x,y
        if lm is not None:
            def center(x, y, part):
                h, w = _PART_SIZE[part]
                return (max(0.0, y / self._img_h - h / 2), max(0.0, x / self._img_w - w / 2),
                        min(1.0, y / self._img_h + h / 2), min(1.0, x / self._img_w + w / 2))
            boxes["left_eye"] = center(lm[0], lm[1], "left_eye")
            boxes["right_eye"] = center(lm[2], lm[3], "right_eye")
            boxes["nose"] = center(lm[4], lm[5], "nose")
            mx = (lm[6] + lm[8]) / 2 / self._img_w
            my = (lm[7] + lm[9]) / 2 / self._img_h
            h, w = _PART_SIZE["mouth"]
            boxes["mouth"] = (max(0.0, my - h / 2), max(0.0, mx - w / 2),
                              min(1.0, my + h / 2), min(1.0, mx + w / 2))
            for p in ("left_eye", "right_eye", "nose", "mouth"):
                vis[p] = 1.0
        else:
            for p in ("left_eye", "right_eye", "nose", "mouth"):
                boxes[p] = (0.4, 0.2, 0.5, 0.3)
                vis[p] = 1.0
        boxes["face"] = (0.0, 0.0, 1.0, 1.0)
        vis["face"] = 1.0
        return boxes, vis

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> CUBSample:
        rec = self.samples[idx]
        path = os.path.join(self.data_dir, "img_align_celeba", rec["fname"])
        pil = Image.open(path).convert("RGB")
        tensor = self.transform(pil)
        boxes, vis = self.part_boxes_for(rec["fname"])
        return CUBSample(
            sample_id=rec["sample_id"], image=pil, image_tensor=tensor,
            label=rec["label"], species_name=self.species_list[rec["label"]],
            part_boxes=boxes, part_visibilities=vis,
            attribute_labels=rec["concepts"], metadata={"path": path},
        )
