"""Strict cached-feature contract; no fabricated CUB annotations or splits."""
from pathlib import Path
import numpy as np

SPLITS = ('train', 'tune', 'calibration', 'test')


def validate(d):
    required = {'x','c','observed','y','ids','groups','split','concept_names'}
    if required - d.keys():
        raise ValueError(f'Missing keys: {required-d.keys()}')
    x, c = d['x'], d['c']
    if x.ndim not in (2,4) or c.ndim != 2 or c.shape[1] < 2:
        raise ValueError('x must be [N,D] or [N,C,H,W]; c must be [N,M], M>=2')
    n, m = c.shape
    if any(len(d[k]) != n for k in ['x','observed','y','ids','groups','split']):
        raise ValueError('Row count mismatch')
    if d['observed'].shape != c.shape or not np.isin(d['observed'],[0,1]).all():
        raise ValueError('Invalid observation mask')
    if d['y'].shape != (n,) or not np.issubdtype(d['y'].dtype,np.integer):
        raise ValueError('y must be one-dimensional integer class IDs')
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite features/images')
    obs = d['observed'].astype(bool)
    if not np.isin(c[obs],[0,1]).all():
        raise ValueError('Observed concepts must be 0/1')
    if len(set(d['ids'].tolist())) != n:
        raise ValueError('ids must be unique strings')
    if any(d[k].dtype.kind not in 'US' for k in ['ids','groups','split','concept_names']):
        raise ValueError('Metadata must be non-object Unicode/string arrays')
    if d['concept_names'].shape != (m,) or len(set(d['concept_names'])) != m:
        raise ValueError('Unique concept names required')
    if set(d['split']) != set(SPLITS):
        raise ValueError(f'All four splits must exist, named {SPLITS}')
    seen = {}
    for group, split in zip(d['groups'],d['split']):
        if group in seen and seen[group] != split:
            raise ValueError(f'Original-image group crosses splits: {group}')
        seen[group] = split
    known_classes = np.unique(d['y'][d['split']=='train'])
    if len(known_classes)<2 or not np.array_equal(known_classes,np.arange(len(known_classes))):
        raise ValueError('Training labels must include contiguous classes 0..K-1')
    if not np.isin(d['y'],known_classes).all():
        raise ValueError('Class outside training vocabulary')
    # This implementation calibrates the complete annotated concept vector.
    # Partial-vector calibration would have a different target and is not silently substituted.
    final = np.isin(d['split'], ['calibration','test'])
    if not obs[final].all():
        raise ValueError('Joint calibration/test require complete concept labels. Narrow the vocabulary/target or obtain annotations; do not impute unknowns as negatives.')
    if not obs[d['split']=='train'].any() or not obs[d['split']=='tune'].any():
        raise ValueError('Training/tuning need observed concepts')
    d = dict(d)
    d['x'] = x.astype('float32')
    d['c'] = np.where(obs,c,0).astype('float32')
    d['observed'] = obs
    return d


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return validate({k:z[k] for k in z.files})


def demo(seed=7, counts=(1200,300,400,200), features=False):
    rng = np.random.default_rng(seed)
    n = sum(counts)
    c = rng.integers(0,2,(n,3))
    visible = rng.random((n,3)) > .12
    x = rng.uniform(0,.15,(n,3,16,16)).astype('float32')
    boxes = [(2,7,1,6),(1,6,7,12),(9,14,9,14)]
    for i in range(n):
        for j,(a,b,u,v) in enumerate(boxes):
            if visible[i,j]: x[i,j,a:b,u:v] += c[i,j]*rng.uniform(.55,.85)
            else: x[i,:,a:b,u:v] = .4
    x = np.clip(x+rng.normal(0,.04,x.shape),0,1).astype('float32')
    if features:
        x = np.stack([x[:,j,a:b,u:v].mean((1,2)) for j,(a,b,u,v) in enumerate(boxes)],1)
    ids = np.array([f'demo-{i:05d}' for i in range(n)])
    return validate({'x':x,'c':c,'observed':np.ones_like(c,dtype=bool),
        'y':c[:,0] & (c[:,1]|c[:,2]), 'ids':ids,'groups':ids.copy(),
        'split':np.repeat(np.array(SPLITS),counts), 'visible':visible,
        'concept_names':np.array(['red_left','green_top','blue_right'])})


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(description='Write a synthetic cached-feature NPZ for testing the adapter')
    p.add_argument('--out',type=Path,default=Path('demo_features.npz'))
    a=p.parse_args(); a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,**demo(features=True))
    print(a.out)
