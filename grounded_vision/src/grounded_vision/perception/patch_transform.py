from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import torch


class PatchTransform:
    """Train-fitted patch transform applied before region pooling.

    rotate: x -> (x - mean) @ V
    whiten: x -> (x - mean) @ V @ diag(1 / sqrt(lambda + eps_rel * lambda_max))

    V: full-rank eigenbasis of the train patch covariance, by decreasing variance.
    """

    def __init__(self, mean: Optional[torch.Tensor] = None,
                 components: Optional[torch.Tensor] = None, n_patches: int = 0,
                 eigenvalues: Optional[torch.Tensor] = None,
                 whiten: bool = False, eps_rel: float = 1e-5):
        self.mean_ = None if mean is None else torch.as_tensor(mean).float()
        self.components_ = None if components is None else torch.as_tensor(components).float()
        self.eigenvalues_ = None if eigenvalues is None else torch.as_tensor(eigenvalues).float()
        self.n_patches = int(n_patches)
        self.whiten = bool(whiten)
        self.eps_rel = float(eps_rel)
        if self.whiten and self.components_ is not None and self.eigenvalues_ is None:
            raise ValueError("whiten=True needs the eigenvalues")

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None and self.components_ is not None

    @property
    def dim(self) -> int:
        self._check()
        return int(self.mean_.shape[0])

    @property
    def mode(self) -> str:
        return "whiten" if self.whiten else "rotate"

    @property
    def scale_(self) -> Optional[torch.Tensor]:
        """Per-axis whitening scale 1/sqrt(lambda + eps_rel * lambda_max), or None."""
        if not self.whiten:
            return None
        lam = self.eigenvalues_.double().clamp_min(0.0)
        return (1.0 / torch.sqrt(lam + self.eps_rel * lam.max())).float()

    def fit_feature_maps(self, feature_maps: np.ndarray, batch_size: int = 256) -> "PatchTransform":
        """Mean and covariance eigenbasis over every patch of a [N, D, H, W] array."""
        if feature_maps.ndim != 4:
            raise ValueError(f"Expected [N, D, H, W] feature maps, got shape {feature_maps.shape}")
        D = feature_maps.shape[1]
        total = torch.zeros(D, dtype=torch.float64)
        count = 0
        for start in range(0, feature_maps.shape[0], batch_size):
            chunk = torch.from_numpy(np.asarray(feature_maps[start:start + batch_size]))
            total += chunk.to(torch.float64).sum(dim=(0, 2, 3))
            count += chunk.shape[0] * chunk.shape[2] * chunk.shape[3]
        if count == 0:
            raise ValueError("cannot fit a patch transform on an empty feature cache")
        mean64 = total / count
        cov = torch.zeros(D, D, dtype=torch.float64)
        for start in range(0, feature_maps.shape[0], batch_size):
            chunk = torch.from_numpy(np.asarray(feature_maps[start:start + batch_size]))
            x = chunk.to(torch.float64).permute(0, 2, 3, 1).reshape(-1, D) - mean64
            cov += x.T @ x
        cov /= max(1, count - 1)
        evals, evecs = torch.linalg.eigh(cov)
        order = torch.argsort(evals, descending=True)
        self.mean_ = mean64.float()
        self.components_ = evecs[:, order].float()
        self.eigenvalues_ = evals[order].float()
        self.n_patches = count
        return self

    def _check(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("PatchTransform is not fitted.")

    def transform_feature_map(self, fmap: torch.Tensor) -> torch.Tensor:
        """[B, D, H, W] -> [B, D, H, W] in the centred, decorrelated (optionally whitened) basis."""
        self._check()
        if fmap.ndim != 4 or fmap.shape[1] != self.dim:
            raise ValueError(f"Expected [B, {self.dim}, H, W], got {tuple(fmap.shape)}")
        x = fmap - self.mean_.to(fmap.device, fmap.dtype).view(1, -1, 1, 1)
        W = self.components_
        if self.whiten:
            W = W * self.scale_.view(1, -1)
        return torch.einsum("bdhw,dk->bkhw", x, W.to(fmap.device, fmap.dtype))

    def state_dict(self) -> Dict[str, object]:
        self._check()
        return {"mean": self.mean_.cpu(), "components": self.components_.cpu(),
                "eigenvalues": None if self.eigenvalues_ is None else self.eigenvalues_.cpu(),
                "n_patches": self.n_patches, "whiten": self.whiten, "eps_rel": self.eps_rel}

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "PatchTransform":
        return cls(mean=state["mean"], components=state["components"],
                   n_patches=int(state.get("n_patches", 0)),
                   eigenvalues=state.get("eigenvalues"),
                   whiten=bool(state.get("whiten", False)),
                   eps_rel=float(state.get("eps_rel", 1e-5)))
