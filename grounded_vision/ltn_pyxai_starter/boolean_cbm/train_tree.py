"""Fit a deployable tree to frozen predicate scores and task labels.

NPZ fields: q[N,M], y[N] (integer or Unicode task labels), split[N],
ids[N], groups[N] (original image identity), concept_names[M].
Head fitting uses train rows only. Test labels are used only for evaluation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .logic import fit_tree, threshold_scores


def train(path, output, max_depth=6, min_samples_leaf=5):
    with np.load(path, allow_pickle=False) as archive:
        d = dict(archive)
    required = {'q', 'y', 'split', 'ids', 'groups', 'concept_names'}
    if not required.issubset(d):
        raise ValueError(f'Missing fields: {required - set(d)}')
    q, y, split = d['q'], d['y'], d['split']
    if q.ndim != 2 or q.shape[1] != len(d['concept_names']):
        raise ValueError('Expected q[N,M] and concept_names[M]')
    for key in ('y', 'split', 'ids', 'groups'):
        if d[key].shape != (len(q),):
            raise ValueError(f'Invalid {key} shape')
    if y.dtype.kind not in 'iuUS':
        raise ValueError('Task labels must be integers or strings')
    for key in ('split', 'ids', 'groups', 'concept_names'):
        if d[key].dtype.kind not in 'US':
            raise ValueError(f'{key} must contain strings')
    if len(set(d['ids'])) != len(q):
        raise ValueError('IDs must be unique')
    if not set(split).issubset({'train', 'tune', 'calibration', 'test'}):
        raise ValueError('Unknown split')
    groups = {}
    for group, part in zip(d['groups'], split):
        if group in groups and groups[group] != part:
            raise ValueError('An original image appears across multiple splits')
        groups[group] = part
    bits = threshold_scores(q)
    train_rows, test_rows = split == 'train', split == 'test'
    if not train_rows.any() or not test_rows.any():
        raise ValueError('Need nonempty train and held-out test splits')
    if not set(map(str, y[test_rows])).issubset(set(map(str, y[train_rows]))):
        raise ValueError('Test has task classes absent from training')
    head = fit_tree(bits[train_rows], y[train_rows], d['concept_names'].tolist(),
                    max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=True)
    head.save(dest / 'head.json')
    records = []
    for index in np.flatnonzero(test_rows):
        prediction = head.class_names[head.predict(bits[index])]
        reason = head.reasons(bits[index])[0]
        records.append({'id': str(d['ids'][index]), 'prediction': prediction,
            'target': str(y[index]), 'reason': list(reason),
            'named_literals': head.describe(reason), 'exact_sufficiency_checked': True,
            'semantic_correctness': 'not_evaluated', 'spatial_certificate': 'not_available_from_scores'})
    (dest / 'explanations.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    summary = {'test_n': len(records),
        'test_accuracy': float(np.mean([r['prediction'] == r['target'] for r in records])),
        'mean_reason_length': float(np.mean([len(r['reason']) for r in records])),
        'threshold': 0.5, 'head': 'trained_tree_is_actual_predictor',
        'reason_policy': 'one_subset_minimal_reason_not_necessarily_shortest',
        'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf,
        'reader_provenance': 'user_must_verify_reader_was_frozen_before_test_access'}
    (dest / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--min-samples-leaf', type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.out, args.max_depth, args.min_samples_leaf), indent=2))
