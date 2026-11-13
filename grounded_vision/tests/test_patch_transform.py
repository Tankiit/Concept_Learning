import os

import numpy as np
import torch

from grounded_vision.perception.patch_transform import PatchTransform
from grounded_vision.perception.regions import Region, pool_region_features
from grounded_vision.pipeline import FeatureProvider, PERCEPTION_CHECKPOINT


def _fmaps(n=20, d=16, h=4, w=4, seed=0):
    rng = np.random.RandomState(seed)
    mix = rng.randn(d, d)  # correlated channels
    x = rng.randn(n, h, w, d) @ mix + 3.0  # large shared offset
    return x.transpose(0, 3, 1, 2).astype(np.float32)


def _flat(t):
    return t.permute(0, 2, 3, 1).reshape(-1, t.shape[1]).double()


def test_mean_matches_flat_mean_and_batching():
    fm = _fmaps()
    flat = fm.transpose(0, 2, 3, 1).reshape(-1, fm.shape[1])
    r = PatchTransform().fit_feature_maps(fm, batch_size=7)
    np.testing.assert_allclose(r.mean_.numpy(), flat.mean(axis=0), atol=1e-4)
    assert r.n_patches == flat.shape[0]


def test_rotation_is_orthogonal_decorrelates_and_keeps_shape():
    fm = _fmaps()
    r = PatchTransform().fit_feature_maps(fm, batch_size=7)
    V = r.components_.double()
    np.testing.assert_allclose((V.T @ V).numpy(), np.eye(16), atol=1e-5)
    out_t = r.transform_feature_map(torch.from_numpy(fm))
    assert out_t.shape == fm.shape
    out = _flat(out_t)
    np.testing.assert_allclose(out.mean(dim=0).numpy(), 0.0, atol=1e-4)
    cov = (out.T @ out / (out.shape[0] - 1)).numpy()
    np.testing.assert_allclose(cov - np.diag(np.diag(cov)), 0.0, atol=1e-3)
    assert np.all(np.diff(np.diag(cov)) <= 1e-6)  # sorted by decreasing variance
    pooled = pool_region_features(out_t[:1], [Region("r", "bird", (0, 0, 1, 1), 1.0)])
    assert pooled["r"].shape == (16,)


def test_rotation_preserves_centred_distances():
    fm = _fmaps()
    r = PatchTransform().fit_feature_maps(fm)
    out = _flat(r.transform_feature_map(torch.from_numpy(fm)))
    cen = _flat(torch.from_numpy(fm)) - r.mean_.double()
    np.testing.assert_allclose(out.norm(dim=1).numpy(), cen.norm(dim=1).numpy(), rtol=1e-4)


def test_state_dict_roundtrip():
    fm = _fmaps()
    r = PatchTransform().fit_feature_maps(fm)
    r2 = PatchTransform.from_state_dict(r.state_dict())
    x = torch.from_numpy(fm)
    np.testing.assert_allclose(r2.transform_feature_map(x).numpy(),
                               r.transform_feature_map(x).numpy(), atol=1e-6)


def test_provider_applies_checkpoint_transform(tmp_path):
    fm = _fmaps(n=6)
    r = PatchTransform().fit_feature_maps(fm)
    np.savez(tmp_path / "test_features.npz", cls_tokens=np.ones((6, 16), np.float32),
             feature_maps=fm, labels=np.zeros(6), sample_ids=np.arange(6))
    assert FeatureProvider(str(tmp_path), "test").transform is None

    torch.save({"patch_transform": r.state_dict()}, os.path.join(tmp_path, PERCEPTION_CHECKPOINT))
    cls, fmap = FeatureProvider(str(tmp_path), "test").get(3)
    np.testing.assert_allclose(cls.numpy(), 1.0)  # CLS untouched
    expected = r.transform_feature_map(torch.from_numpy(fm[3:4]))
    np.testing.assert_allclose(fmap.numpy(), expected.numpy(), atol=1e-6)
    raw = FeatureProvider(str(tmp_path), "test", transform=False).get(3)[1]
    np.testing.assert_allclose(raw[0].numpy(), fm[3])


def test_whitening_gives_identity_covariance():
    fm = _fmaps()
    r = PatchTransform(whiten=True, eps_rel=1e-9).fit_feature_maps(fm, batch_size=7)
    out = _flat(r.transform_feature_map(torch.from_numpy(fm)))
    np.testing.assert_allclose(out.mean(dim=0).numpy(), 0.0, atol=1e-4)
    cov = (out.T @ out / (out.shape[0] - 1)).numpy()
    np.testing.assert_allclose(cov, np.eye(16), atol=1e-3)


def test_whitening_eps_caps_null_directions():
    fm = _fmaps()
    fm[:, -1] = fm[:, 0]  # duplicate channel -> one zero-variance direction
    r = PatchTransform(whiten=True, eps_rel=1e-5).fit_feature_maps(fm)
    scale = r.scale_.double()
    lam_max = r.eigenvalues_.double().max()
    assert torch.isfinite(scale).all()
    assert scale.max() <= 1.0 / np.sqrt(1e-5 * lam_max) + 1e-6


def test_whitening_state_roundtrip_and_legacy_rotation_state():
    fm = _fmaps()
    x = torch.from_numpy(fm)
    w = PatchTransform(whiten=True).fit_feature_maps(fm)
    w2 = PatchTransform.from_state_dict(w.state_dict())
    assert w2.whiten
    np.testing.assert_allclose(w2.transform_feature_map(x).numpy(),
                               w.transform_feature_map(x).numpy(), atol=1e-5)
    r = PatchTransform().fit_feature_maps(fm)
    legacy = {k: v for k, v in r.state_dict().items() if k in ("mean", "components", "n_patches")}
    r2 = PatchTransform.from_state_dict(legacy)
    assert not r2.whiten
    np.testing.assert_allclose(r2.transform_feature_map(x).numpy(),
                               r.transform_feature_map(x).numpy(), atol=1e-6)


def test_provider_reads_pre_rename_rotation_key(tmp_path):
    fm = _fmaps(n=2)
    r = PatchTransform().fit_feature_maps(fm)
    np.savez(tmp_path / "test_features.npz", cls_tokens=np.ones((2, 16), np.float32),
             feature_maps=fm, labels=np.zeros(2), sample_ids=np.arange(2))
    torch.save({"rotation": r.state_dict()}, os.path.join(tmp_path, PERCEPTION_CHECKPOINT))
    fmap = FeatureProvider(str(tmp_path), "test").get(1)[1]
    np.testing.assert_allclose(fmap.numpy(), r.transform_feature_map(torch.from_numpy(fm[1:2])).numpy(),
                               atol=1e-6)
