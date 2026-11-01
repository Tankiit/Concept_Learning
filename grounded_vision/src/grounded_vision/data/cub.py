"""
CUB-200-2011 Dataset Loader and Grounding Schema
================================================
Provides dataset interfaces for Caltech-UCSD Birds-200-2011, including:
- 15 canonical anatomical part definitions
- Part hierarchy (head, body, wings, tail)
- Attribute metadata with part binding
- Preprocessing transforms and mock generation for hermetic testing.
"""

from __future__ import annotations
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

# Canonical 15 CUB parts (1-indexed in raw dataset, converted to 0-indexed here)
CUB_PARTS: List[str] = [
    "back",
    "beak",
    "belly",
    "breast",
    "crown",
    "forehead",
    "left_eye",
    "left_leg",
    "left_wing",
    "nape",
    "right_eye",
    "right_leg",
    "right_wing",
    "tail",
    "throat",
]

# Anatomical hierarchy mapping macro-regions to constituent micro-parts
CUB_PART_HIERARCHY: Dict[str, List[str]] = {
    "head": ["beak", "crown", "forehead", "left_eye", "right_eye", "throat", "nape"],
    "torso": ["back", "breast", "belly"],
    "appendages": ["left_wing", "right_wing", "left_leg", "right_leg", "tail"],
}

# Inverse lookup: micro-part to macro-region
PART_TO_MACRO: Dict[str, str] = {}
for macro, parts in CUB_PART_HIERARCHY.items():
    for p in parts:
        PART_TO_MACRO[p] = macro

# Representative subset of 28 key attributes bound to specific parts for clean modeling
# Format: (attribute_id, part_name, attribute_name, value)
CANONICAL_BOUND_ATTRIBUTES: List[Tuple[int, str, str, str]] = [
    (0, "beak", "bill_shape", "curved"),
    (1, "beak", "bill_shape", "dagger"),
    (2, "beak", "bill_shape", "hooked"),
    (3, "beak", "bill_shape", "needle"),
    (4, "beak", "bill_shape", "cone"),
    (5, "beak", "bill_color", "black"),
    (6, "beak", "bill_color", "yellow"),
    (7, "crown", "crown_color", "black"),
    (8, "crown", "crown_color", "yellow"),
    (9, "crown", "crown_color", "red"),
    (10, "crown", "crown_color", "white"),
    (11, "forehead", "forehead_color", "black"),
    (12, "forehead", "forehead_color", "yellow"),
    (13, "breast", "breast_color", "yellow"),
    (14, "breast", "breast_color", "white"),
    (15, "breast", "breast_color", "black"),
    (16, "breast", "breast_color", "red"),
    (17, "breast", "breast_pattern", "striped"),
    (18, "belly", "belly_color", "yellow"),
    (19, "belly", "belly_color", "white"),
    (20, "belly", "belly_color", "black"),
    (21, "left_wing", "wing_color", "black"),
    (22, "left_wing", "wing_color", "brown"),
    (23, "left_wing", "wing_pattern", "striped"),
    (24, "right_wing", "wing_color", "black"),
    (25, "right_wing", "wing_color", "brown"),
    (26, "tail", "tail_shape", "forked"),
    (27, "tail", "tail_color", "black"),
]

# Total canonical attributes count
CUB_ATTRIBUTES: List[str] = [
    f"{part}:{attr}={val}" for _, part, attr, val in CANONICAL_BOUND_ATTRIBUTES
]

# Example Bird Species names (subset for demonstration and evaluation)
DEFAULT_SPECIES: List[str] = [
    "001.Black_footed_Albatross",
    "002.Laysan_Albatross",
    "003.Sooty_Albatross",
    "004.Groove_billed_Ani",
    "005.Crested_Auklet",
    "006.Least_Auklet",
    "007.Parakeet_Auklet",
    "008.Rhinoceros_Auklet",
    "009.Brewer_Blackbird",
    "010.Red_winged_Blackbird",
    "011.Rusty_Blackbird",
    "012.Yellow_headed_Blackbird",
    "013.Bobolink",
    "014.Indigo_Bunting",
    "015.Lazuli_Bunting",
    "016.Painted_Bunting",
    "017.Cardinal",
    "018.Spotted_Catbird",
    "019.Gray_Catbird",
    "020.Yellow_breasted_Chat",
]


@dataclass
class CUBSample:
    """A single structured CUB sample."""
    sample_id: int
    image: Image.Image
    image_tensor: torch.Tensor
    label: int
    species_name: str
    part_boxes: Dict[str, Tuple[float, float, float, float]]  # part -> (ymin, xmin, ymax, xmax) normalized [0, 1]
    part_visibilities: Dict[str, float]  # part -> visibility in [0, 1]
    attribute_labels: np.ndarray  # binary vector [num_attributes]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _DiskRecord:
    """Lightweight per-image metadata for real CUB data; the PIL image and
    tensor are loaded lazily in __getitem__ to keep full-split memory flat."""
    sample_id: int
    image_path: str
    label: int
    species_name: str
    # part -> (x_norm, y_norm, visible); entries only for annotated parts
    part_points: Dict[str, Tuple[float, float, int]]
    attribute_labels: np.ndarray  # binary vector [num canonical attributes]


def _canonical_attribute_cub_ids(data_dir: str) -> List[Optional[int]]:
    """
    Maps each canonical bound attribute to its 1-based CUB attribute id by name,
    e.g. ("beak", "bill_shape", "curved") -> id of "has_bill_shape::curved_(up_or_down)".
    Falls back to a prefix match; None when CUB has no such attribute.
    """
    name_to_id: Dict[str, int] = {}
    with open(os.path.join(data_dir, "attributes.txt"), "r") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                name_to_id[parts[1]] = int(parts[0])

    mapping: List[Optional[int]] = []
    for _, part, attr_name, val in CANONICAL_BOUND_ATTRIBUTES:
        exact = f"has_{attr_name}::{val}"
        if exact in name_to_id:
            mapping.append(name_to_id[exact])
            continue
        prefix_hits = [aid for name, aid in name_to_id.items() if name.startswith(exact)]
        mapping.append(prefix_hits[0] if len(prefix_hits) == 1 else None)
    return mapping


def _parse_cub_annotations(data_dir: str):
    """
    Parses the raw CUB-200-2011 metadata once per data_dir.
    Returns (class_names, id_to_record) where id_to_record maps image id to a
    dict with path, label, split, part points, and canonical attribute labels.
    """
    class_names: List[str] = []
    with open(os.path.join(data_dir, "classes.txt"), "r") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                name = parts[1].split(".", 1)[1] if parts[1][:3].isdigit() else parts[1]
                class_names.append(f"{int(parts[0]):03d}.{name}")

    id_to_path: Dict[int, str] = {}
    with open(os.path.join(data_dir, "images.txt"), "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                id_to_path[int(parts[0])] = parts[1]

    id_to_split: Dict[int, int] = {}
    with open(os.path.join(data_dir, "train_test_split.txt"), "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                id_to_split[int(parts[0])] = int(parts[1])

    id_to_label: Dict[int, int] = {}
    with open(os.path.join(data_dir, "image_class_labels.txt"), "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                id_to_label[int(parts[0])] = int(parts[1]) - 1  # 0-indexed

    part_id_to_name: Dict[int, str] = {}
    parts_file = os.path.join(data_dir, "parts", "parts.txt")
    if os.path.exists(parts_file):
        with open(parts_file, "r") as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    part_id_to_name[int(parts[0])] = parts[1]

    # Image sizes for normalizing part points
    id_to_parts: Dict[int, Dict[str, Tuple[float, float, int]]] = {}
    part_locs = os.path.join(data_dir, "parts", "part_locs.txt")
    if os.path.exists(part_locs):
        from PIL import Image as _PILImage
        size_cache: Dict[int, Tuple[int, int]] = {}
        with open(part_locs, "r") as f:
            for line in f:
                fields = line.strip().split()
                if len(fields) != 5:
                    continue
                img_id, part_id, x, y, visible = (
                    int(fields[0]), int(fields[1]), float(fields[2]), float(fields[3]), int(fields[4]),
                )
                part_name = part_id_to_name.get(part_id)
                if part_name is None:
                    continue
                if img_id not in size_cache:
                    with _PILImage.open(os.path.join(data_dir, "images", id_to_path[img_id])) as im:
                        size_cache[img_id] = im.size  # (W, H)
                w, h = size_cache[img_id]
                id_to_parts.setdefault(img_id, {})[part_name] = (x / w, y / h, visible)

    # Full 312-dim binary attribute labels per image (vocabularies slice this)
    num_attrs = 312
    attrs_txt = os.path.join(data_dir, "attributes.txt")
    if os.path.exists(attrs_txt):
        with open(attrs_txt) as f:
            num_attrs = sum(1 for line in f if line.strip())
    id_to_attrs: Dict[int, np.ndarray] = {}
    attr_labels_file = os.path.join(data_dir, "attributes", "image_attribute_labels.txt")
    if os.path.exists(attr_labels_file):
        with open(attr_labels_file, "r") as f:
            for line in f:
                fields = line.strip().split()
                if len(fields) < 3:
                    continue
                img_id, attr_id, present = int(fields[0]), int(fields[1]), int(fields[2])
                vec = id_to_attrs.get(img_id)
                if vec is None:
                    vec = np.zeros(num_attrs, dtype=np.float32)
                    id_to_attrs[img_id] = vec
                vec[attr_id - 1] = 1.0 if present else 0.0

    records: Dict[int, Dict[str, Any]] = {}
    for img_id, rel_path in id_to_path.items():
        full_path = os.path.join(data_dir, "images", rel_path)
        if not os.path.exists(full_path):
            continue
        label = id_to_label.get(img_id, 0)
        records[img_id] = {
            "path": full_path,
            "label": label,
            "split": id_to_split.get(img_id, 0),
            "parts": id_to_parts.get(img_id, {}),
            "attrs": id_to_attrs.get(img_id, np.zeros(num_attrs, dtype=np.float32)),
        }
    return class_names, records


def get_cub_transforms(image_size: int = 224, is_train: bool = False) -> T.Compose:
    """Returns standard ImageNet preprocessing transforms for ViT/DINO."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if is_train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


class CUBDataset(Dataset):
    """
    CUB-200-2011 Dataset loader supporting both physical datasets on disk
    and self-contained deterministic synthetic mocking for hermetic workflows.
    """
    def __init__(
        self,
        data_dir: str = "data/CUB_200_2011",
        split: str = "train",
        image_size: int = 224,
        use_mock_if_missing: bool = True,
        num_mock_samples: int = 120,
        seed: int = 42,
        bound_attributes: Optional[List[Any]] = None,
        attribute_cub_ids: Optional[List[int]] = None,
    ):
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.transform = get_cub_transforms(image_size, is_train=(split == "train"))
        self.species_list = DEFAULT_SPECIES
        self.num_classes = len(self.species_list)

        # Active bound-attribute vocabulary (defaults to the canonical 28)
        from .attributes import get_bound_attributes, vocabulary_cub_ids
        has_attrs_txt = os.path.exists(os.path.join(data_dir, "attributes.txt"))
        if bound_attributes is not None:
            self.bound_attributes = list(bound_attributes)
            self.attribute_cub_ids = list(attribute_cub_ids) if attribute_cub_ids else None
        else:
            self.bound_attributes = get_bound_attributes("canonical", data_dir)
            self.attribute_cub_ids = (
                vocabulary_cub_ids("canonical", data_dir) if has_attrs_txt else None
            )
        self.attributes = [f"{p}:{a}={v}" for _, p, a, v in self.bound_attributes]
        self.num_attributes = len(self.attributes)
        self.part_names = CUB_PARTS

        self.samples: List[CUBSample] = []
        has_real_data = os.path.exists(os.path.join(data_dir, "images.txt"))

        if has_real_data:
            self._load_from_disk()
        elif use_mock_if_missing:
            self._generate_synthetic_data(num_mock_samples, seed)
        else:
            raise FileNotFoundError(
                f"CUB dataset not found at '{data_dir}' and use_mock_if_missing is False."
            )

    def _load_from_disk(self) -> None:
        """Parses raw CUB-200-2011 annotations into lightweight records.

        Real part points, visibilities, canonical attribute labels, and class
        names are loaded; images are decoded lazily in __getitem__.
        """
        class_names, records = _parse_cub_annotations_cached(self.data_dir)
        self.species_list = class_names or DEFAULT_SPECIES
        self.num_classes = len(self.species_list)

        split_mask = 1 if self.split == "train" else 0
        template = self._default_part_boxes()
        self._template_sizes = {
            part: (box[2] - box[0], box[3] - box[1]) for part, box in template.items()
        }

        for img_id, rec in records.items():
            if rec["split"] != split_mask:
                continue
            if self.attribute_cub_ids is not None:
                attrs = rec["attrs"][np.asarray(self.attribute_cub_ids, dtype=int) - 1]
            else:
                attrs = np.zeros(self.num_attributes, dtype=np.float32)
            self.samples.append(
                _DiskRecord(
                    sample_id=img_id,
                    image_path=rec["path"],
                    label=rec["label"],
                    species_name=self.species_list[rec["label"] % len(self.species_list)],
                    part_points=rec["parts"],
                    attribute_labels=attrs,
                )
            )

    def meta(self, i: int):
        """(sample_id, part_boxes, part_visibilities, attribute_labels) without
        decoding the image (shared interface with CelebADataset)."""
        rec = self.samples[i]
        if isinstance(rec, CUBSample):
            return rec.sample_id, rec.part_boxes, rec.part_visibilities, rec.attribute_labels
        boxes, vis = self._part_boxes_from_points(rec.part_points)
        return rec.sample_id, boxes, vis, rec.attribute_labels

    def _part_boxes_from_points(
        self, part_points: Dict[str, Tuple[float, float, int]]
    ) -> Tuple[Dict[str, Tuple[float, float, float, float]], Dict[str, float]]:
        """Centers a fixed-size template box on each annotated part point."""
        boxes: Dict[str, Tuple[float, float, float, float]] = {}
        visibilities: Dict[str, float] = {}
        template = self._default_part_boxes()
        for part in CUB_PARTS:
            h, w = self._template_sizes[part]
            point = part_points.get(part)
            if point is not None:
                x, y, visible = point
                ymin = float(np.clip(y - h / 2.0, 0.0, 1.0 - h))
                xmin = float(np.clip(x - w / 2.0, 0.0, 1.0 - w))
                boxes[part] = (ymin, xmin, ymin + h, xmin + w)
                visibilities[part] = 1.0 if visible else 0.0
            else:
                boxes[part] = template[part]
                visibilities[part] = 0.0
        return boxes, visibilities

    def _default_part_boxes(self) -> Dict[str, Tuple[float, float, float, float]]:
        """Provides default normalized anatomical locations."""
        return {
            "crown": (0.05, 0.35, 0.25, 0.65),
            "forehead": (0.10, 0.40, 0.25, 0.60),
            "beak": (0.15, 0.55, 0.30, 0.85),
            "left_eye": (0.12, 0.42, 0.20, 0.50),
            "right_eye": (0.12, 0.50, 0.20, 0.58),
            "throat": (0.22, 0.42, 0.35, 0.65),
            "nape": (0.15, 0.30, 0.30, 0.45),
            "breast": (0.30, 0.35, 0.55, 0.65),
            "back": (0.28, 0.20, 0.60, 0.45),
            "belly": (0.50, 0.35, 0.75, 0.65),
            "left_wing": (0.35, 0.15, 0.70, 0.45),
            "right_wing": (0.35, 0.55, 0.70, 0.85),
            "left_leg": (0.72, 0.38, 0.92, 0.48),
            "right_leg": (0.72, 0.52, 0.92, 0.62),
            "tail": (0.65, 0.10, 0.95, 0.30),
        }

    def _generate_synthetic_data(self, num_samples: int, seed: int) -> None:
        """Generates synthetic bird images and ground truth annotations for self-contained testing."""
        rng = np.random.RandomState(seed + (0 if self.split == "train" else 1000))
        base_boxes = self._default_part_boxes()

        for idx in range(num_samples):
            # Select species
            label = int(rng.choice(len(self.species_list)))
            species_name = self.species_list[label]

            # Generate synthetic image (RGB with colored shapes at part locations)
            img_arr = np.full((self.image_size, self.image_size, 3), 220, dtype=np.uint8)
            # Add subtle background texture
            noise = rng.randint(-15, 15, size=img_arr.shape).astype(np.int16)
            img_arr = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Assign species-characteristic attributes (canonical 28 vocabulary only;
            # the synthetic rules use hard-coded canonical indices)
            attr_vector = np.zeros(self.num_attributes, dtype=np.float32)
            if self.num_attributes == 28:
                # Species rule: e.g. even species have yellow belly & cone bill; odd have black crown & curved bill
                if label % 2 == 0:
                    attr_vector[4] = 1.0   # cone bill
                    attr_vector[6] = 1.0   # yellow bill
                    attr_vector[18] = 1.0  # yellow belly
                    attr_vector[13] = 1.0  # yellow breast
                else:
                    attr_vector[0] = 1.0   # curved bill
                    attr_vector[5] = 1.0   # black bill
                    attr_vector[7] = 1.0   # black crown
                    attr_vector[21] = 1.0  # black wing

            # Jitter part boxes slightly
            sample_boxes: Dict[str, Tuple[float, float, float, float]] = {}
            visibilities: Dict[str, float] = {}
            for part, (ymin, xmin, ymax, xmax) in base_boxes.items():
                dy = rng.uniform(-0.02, 0.02)
                dx = rng.uniform(-0.02, 0.02)
                ny0 = float(np.clip(ymin + dy, 0.0, 0.9))
                nx0 = float(np.clip(xmin + dx, 0.0, 0.9))
                ny1 = float(np.clip(ymax + dy, ny0 + 0.05, 1.0))
                nx1 = float(np.clip(xmax + dx, nx0 + 0.05, 1.0))
                sample_boxes[part] = (ny0, nx0, ny1, nx1)
                visibilities[part] = 1.0 if rng.uniform() > 0.05 else 0.0

            pil_img = Image.fromarray(img_arr)
            tensor_img = self.transform(pil_img)

            self.samples.append(
                CUBSample(
                    sample_id=idx,
                    image=pil_img,
                    image_tensor=tensor_img,
                    label=label,
                    species_name=species_name,
                    part_boxes=sample_boxes,
                    part_visibilities=visibilities,
                    attribute_labels=attr_vector,
                    metadata={"synthetic": True},
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> CUBSample:
        rec = self.samples[idx]
        if isinstance(rec, CUBSample):  # synthetic mock path
            return rec
        pil_img = Image.open(rec.image_path).convert("RGB")
        tensor_img = self.transform(pil_img)
        boxes, vis = self._part_boxes_from_points(rec.part_points)
        return CUBSample(
            sample_id=rec.sample_id,
            image=pil_img,
            image_tensor=tensor_img,
            label=rec.label,
            species_name=rec.species_name,
            part_boxes=boxes,
            part_visibilities=vis,
            attribute_labels=rec.attribute_labels,
            metadata={"path": rec.image_path},
        )


def create_synthetic_cub_batch(batch_size: int = 4, image_size: int = 224, seed: int = 42) -> List[CUBSample]:
    """Helper to quickly obtain a small verified batch of CUBSamples."""
    dataset = CUBDataset(
        use_mock_if_missing=True,
        num_mock_samples=batch_size,
        image_size=image_size,
        seed=seed,
    )
    return [dataset[i] for i in range(batch_size)]


@lru_cache(maxsize=4)
def _parse_cub_annotations_cached(data_dir: str):
    return _parse_cub_annotations(data_dir)
