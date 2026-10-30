"""
Region-Exemplar Prototype Similarity Bank
=========================================
Computes similarity between pooled anatomical region features and a bank of
representative visual prototypes (exemplars) for each attribute and part category.
Provides interpretable provenance metrics (e.g., "beak has 0.88 similarity to curved-bill prototype #3").
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F

from ..data.cub import CANONICAL_BOUND_ATTRIBUTES


@dataclass
class PrototypeMatch:
    """Represents a match between a region and a prototype exemplar."""
    prototype_id: str
    attribute_id: int
    part_name: str
    attribute_name: str
    value: str
    similarity: float
    exemplar_metadata: Dict[str, Any] = field(default_factory=dict)


class AttributePrototypeBank:
    """
    Maintains a set of exemplar prototype vectors for each bound attribute:
    (part_name, attribute_name, value).
    """
    def __init__(
        self,
        embedding_dim: int = 384,
        num_prototypes_per_attr: int = 3,
        temperature: float = 0.1,
        attributes=None,
    ):
        from ..data.cub import CANONICAL_BOUND_ATTRIBUTES
        self.embedding_dim = embedding_dim
        self.num_prototypes_per_attr = num_prototypes_per_attr
        self.temperature = temperature
        self.attributes = list(attributes) if attributes is not None else list(CANONICAL_BOUND_ATTRIBUTES)

        # Mapping: attr_id -> Tensor of shape [num_prototypes, embedding_dim]
        self.prototypes: Dict[int, torch.Tensor] = {}
        # Prototype metadata registry
        self.metadata: Dict[int, List[Dict[str, Any]]] = {}

        self._initialize_prototypes()

    def _initialize_prototypes(self) -> None:
        """Initializes normalized prototype vectors for each bound attribute."""
        for attr_id, part, attr_name, val in self.attributes:
            # Deterministic initialization based on attr_id seed
            rng = np.random.RandomState(42 + attr_id)
            rand_vecs = rng.randn(self.num_prototypes_per_attr, self.embedding_dim).astype(np.float32)
            tensor_vecs = torch.tensor(rand_vecs)
            tensor_vecs = F.normalize(tensor_vecs, p=2, dim=1)
            self.prototypes[attr_id] = tensor_vecs
            self.metadata[attr_id] = [
                {
                    "prototype_id": f"proto_{attr_id}_{i}",
                    "part": part,
                    "attr": attr_name,
                    "val": val,
                    "index": i,
                }
                for i in range(self.num_prototypes_per_attr)
            ]

    def update_prototype_exemplars(
        self,
        attr_id: int,
        exemplar_features: torch.Tensor,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Updates the prototypes for a given attribute with real pooled region features.

        Args:
            attr_id: Canonical attribute index
            exemplar_features: Tensor of shape [K, embedding_dim]
            metadata_list: Optional metadata for each exemplar
        """
        assert exemplar_features.ndim == 2
        assert exemplar_features.shape[1] == self.embedding_dim
        normed = F.normalize(exemplar_features, p=2, dim=1)
        self.prototypes[attr_id] = normed
        if metadata_list is not None:
            self.metadata[attr_id] = metadata_list

    def match_region(
        self, region_feature: torch.Tensor, part_name: Optional[str] = None
    ) -> Dict[int, PrototypeMatch]:
        """
        Computes cosine similarities between a pooled region feature [D] and all prototypes.

        Args:
            region_feature: Tensor of shape [D]
            part_name: Optional filter to restrict matches only to prototypes of this part

        Returns:
            Dict mapping attr_id -> PrototypeMatch for the top-matching exemplar
        """
        feat = F.normalize(region_feature.unsqueeze(0), p=2, dim=1)  # [1, D]
        matches: Dict[int, PrototypeMatch] = {}

        for attr_id, part, attr_name, val in self.attributes:
            if part_name is not None and part != part_name:
                continue
            
            protos = self.prototypes[attr_id].to(feat.device)  # [K, D]
            sims = torch.matmul(feat, protos.t()).squeeze(0)  # [K]
            best_idx = int(torch.argmax(sims).item())
            best_sim = float(sims[best_idx].item())

            meta = self.metadata[attr_id][best_idx] if best_idx < len(self.metadata[attr_id]) else {}
            matches[attr_id] = PrototypeMatch(
                prototype_id=f"proto_{attr_id}_{best_idx}",
                attribute_id=attr_id,
                part_name=part,
                attribute_name=attr_name,
                value=val,
                similarity=best_sim,
                exemplar_metadata=meta,
            )

        return matches
