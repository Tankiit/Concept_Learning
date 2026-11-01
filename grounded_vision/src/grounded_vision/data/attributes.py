"""
Bound Attribute Vocabulary Registry
===================================
Makes the attribute vocabulary a first-class, config-driven dimension of the
pipeline. Two vocabularies exist:

- "canonical": the 28 hand-picked part-bound attributes (the historical
  default; used by tests and the synthetic mock).
- "full": every CUB-200-2011 attribute (312) bound to a grounding region by
  attribute-name prefix. Wing and leg/foot attributes bind to BOTH left and
  right instances (matching the canonical convention); attributes without a
  natural part (size, shape, plumage, head, eye, upperparts, underparts, ...)
  bind to the whole-bird entity "bird", grounded on a full-image region.

Vocabulary entries are `(attr_id, part, attr_name, value)` tuples (local ids,
dense from 0); `vocabulary_cub_ids` returns the aligned raw CUB attribute ids
for label supervision.
"""

from __future__ import annotations
import os
from functools import lru_cache
from typing import List, NamedTuple, Optional

from .cub import CANONICAL_BOUND_ATTRIBUTES, CUB_PARTS

#: Prefix rules mapping CUB attribute-name stems to grounding regions.
#: Wing and leg/foot map to a PAIR of parts. Everything else -> "bird".
PREFIX_PARTS: List = [
    ("back_of_neck", ("back",)),
    ("upper_tail", ("tail",)),
    ("lower_tail", ("tail",)),
    ("bill", ("beak",)),
    ("crown", ("crown",)),
    ("forehead", ("forehead",)),
    ("nape", ("nape",)),
    ("throat", ("throat",)),
    ("breast", ("breast",)),
    ("belly", ("belly",)),
    ("back", ("back",)),
    ("tail", ("tail",)),
    ("wing", ("left_wing", "right_wing")),
    ("leg", ("left_leg", "right_leg")),
    ("foot", ("left_leg", "right_leg")),
]


class BoundAttribute(NamedTuple):
    attr_id: int
    part: str
    attr_name: str
    value: str


def _parse_attributes_txt(data_dir: str):
    """[(cub_id, stem, value)] from attributes.txt, e.g. 'has_bill_shape::curved'."""
    entries = []
    with open(os.path.join(data_dir, "attributes.txt")) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) != 2 or "::" not in parts[1]:
                continue
            stem, value = parts[1].removeprefix("has_").split("::", 1)
            entries.append((int(parts[0]), stem, value))
    return sorted(entries)


def _bind(stem: str):
    for prefix, regions in PREFIX_PARTS:
        if stem.startswith(prefix):
            return regions
    return ("bird",)


@lru_cache(maxsize=4)
def _full_vocabulary_and_cub_ids(data_dir: str):
    """Cached (vocabulary, aligned cub ids) for the full 312-attribute set."""
    vocab: List[BoundAttribute] = []
    cub_ids: List[int] = []
    for cub_id, stem, value in _parse_attributes_txt(data_dir):
        for part in _bind(stem):
            vocab.append(BoundAttribute(len(vocab), part, stem, value))
            cub_ids.append(cub_id)
    return tuple(vocab), tuple(cub_ids)


@lru_cache(maxsize=4)
def _canonical_vocabulary_and_cub_ids(data_dir: str):
    """The 28 canonical attributes (local ids 0..27 preserved) with raw CUB ids."""
    raw = {cub_id: (stem, value) for cub_id, stem, value in _parse_attributes_txt(data_dir)}
    vocab, cub_ids = [], []
    for attr_id, part, attr_name, value in CANONICAL_BOUND_ATTRIBUTES:
        vocab.append(BoundAttribute(attr_id, part, attr_name, value))
        cub_ids.append(_match_cub_id(raw, part, attr_name, value))
    return tuple(vocab), tuple(cub_ids)


#: Canonical attribute names that don't exist verbatim in CUB. attr 27
#: ("tail_color") was silently unsupervised in the old cub.py matcher — CUB
#: only has upper/lower_tail_color; bind to the upper (usually visible) one.
CANONICAL_ALIASES = {"tail_color": "upper_tail_color"}


def _match_cub_id(raw, part: str, attr_name: str, value: str) -> int:
    """Finds the CUB id for a canonical entry by exact then unique-prefix match."""
    attr_name = CANONICAL_ALIASES.get(attr_name, attr_name)
    exact = f"{attr_name}::{value}"
    hits = [cid for cid, (stem, val) in raw.items() if f"{stem}::{val}" == exact]
    if hits:
        return hits[0]
    prefix = [cid for cid, (stem, val) in raw.items()
              if f"{stem}::{val}".startswith(exact)]
    if len(prefix) == 1:
        return prefix[0]
    raise KeyError(f"No CUB attribute for canonical ({part}, {attr_name}, {value})")


def get_bound_attributes(attribute_set: str, data_dir: str) -> List[BoundAttribute]:
    if attribute_set == "canonical":
        if not os.path.exists(os.path.join(data_dir, "attributes.txt")):
            # mock/hermetic path: no raw CUB annotations needed
            return [BoundAttribute(a, p, n, v) for a, p, n, v in CANONICAL_BOUND_ATTRIBUTES]
        return list(_canonical_vocabulary_and_cub_ids(data_dir)[0])
    if attribute_set == "full":
        return list(_full_vocabulary_and_cub_ids(data_dir)[0])
    raise ValueError(f"Unknown attribute_set {attribute_set!r} (canonical|full)")


def vocabulary_cub_ids(attribute_set: str, data_dir: str) -> List[int]:
    if attribute_set == "canonical":
        return list(_canonical_vocabulary_and_cub_ids(data_dir)[1])
    return list(_full_vocabulary_and_cub_ids(data_dir)[1])


def active_vocabulary(cfg: dict) -> List[BoundAttribute]:
    """Convenience: resolve the config's vocabulary (defaults to canonical)."""
    data = cfg["data"]
    data_dir = data["data_dir"]
    if data.get("dataset") == "celeba":
        return celeba_vocabulary(data_dir, data.get("target_attribute", "Male"))
    return get_bound_attributes(data.get("attribute_set", "canonical"), data_dir)


@lru_cache(maxsize=4)
def _celeba_vocabulary(data_dir: str, target: str):
    """(attr_id, part, attr_name, 'yes') for every CelebA attribute except the
    target; order matches CelebADataset.concept_names."""
    from .celeba import CELEBA_ATTR_PART
    with open(os.path.join(data_dir, "list_attr_celeba.txt")) as f:
        f.readline()  # count line
        names = f.readline().split()
    vocab = []
    for i, name in enumerate(n for n in names if n != target):
        vocab.append(BoundAttribute(i, CELEBA_ATTR_PART.get(name, "face"), name, "yes"))
    return tuple(vocab)


def celeba_vocabulary(data_dir: str, target: str = "Male") -> List[BoundAttribute]:
    return list(_celeba_vocabulary(data_dir, target))


def valid_parts(vocabulary) -> set:
    return set(CUB_PARTS) | {"bird"}
