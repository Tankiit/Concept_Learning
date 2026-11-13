"""Synthetic visual integration test; not a benchmark or novelty result.

LTNtorch trains predicates and two gate choices. The deployed Boolean head
is compiled exactly by exhaustive Shannon expansion, NOT fitted to samples.
"""
import argparse
import copy
import csv
import importlib.metadata
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
import ltn

NAMES = ['red_left', 'green_top', 'blue_right']
WORLDS = np.array(list(itertools.product([0, 1], repeat=3)), dtype=np.int64)


def make_data(n, seed, occlusion=0.12):
    rng = np.random.default_rng(seed)
    c = rng.integers(0, 2, (n, 3))
    x = rng.uniform(0, 0.15, (n, 3, 16, 16)).astype('float32')
    boxes = [(2, 7, 1, 6), (1, 6, 7, 12), (9, 14, 9, 14)]
    visible = rng.random((n, 3)) > occlusion
    for i in range(n):
        for j, (a, b, d, e) in enumerate(boxes):
            if visible[i, j]:
                x[i, j, a:b, d:e] += c[i, j] * rng.uniform(0.55, 0.85)
            else:
                x[i, :, a:b, d:e] = 0.4
    x = np.clip(x + rng.normal(0, 0.04, x.shape), 0, 1).astype('float32')
    y = c[:, 0] & (c[:, 1] | c[:, 2])
    return torch.from_numpy(x), torch.from_numpy(c).float(), torch.from_numpy(y).float(), visible


class VisualReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                                 nn.AvgPool2d(2), nn.Flatten(),
                                 nn.Linear(8 * 8 * 8, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, x):
        return self.net(x).sigmoid()


class LogicHead(nn.Module):
    """Fixed topology g_outer(c0, g_inner(c1,c2)); learned AND/OR at each gate.

    This restricted structure search is custom code, not an LTNtorch feature.
    Concept identity/topology is fixed; arbitrary rule discovery is not claimed.
    """
    def __init__(self):
        super().__init__()
        self.gate_logits = nn.Parameter(torch.randn(2, 2) * 0.02)
        self.and_op = ltn.fuzzy_ops.AndProd(stable=False)
        self.or_op = ltn.fuzzy_ops.OrProbSum(stable=False)

    def soft_gate(self, a, b, index, temperature):
        weights = (self.gate_logits[index] / temperature).softmax(-1)
        return weights[0] * self.and_op(a, b) + weights[1] * self.or_op(a, b)

    def forward(self, c, temperature=1.0):
        inner = self.soft_gate(c[:, 1], c[:, 2], 0, temperature)
        return self.soft_gate(c[:, 0], inner, 1, temperature)

    def hard(self, c):
        c = np.asarray(c, dtype=np.int64)
        choices = self.gate_logits.detach().argmax(-1).cpu().numpy()
        inner = c[:, 1] & c[:, 2] if choices[0] == 0 else c[:, 1] | c[:, 2]
        return c[:, 0] & inner if choices[1] == 0 else c[:, 0] | inner

    def formula(self):
        ops = ['AND', 'OR']
        inner, outer = self.gate_logits.detach().argmax(-1).tolist()
        return f'({NAMES[0]} {ops[outer]} ({NAMES[1]} {ops[inner]} {NAMES[2]}))'


def ltn_loss(scores, labels):
    # One variable name means corresponding rows, not a Cartesian product.
    # Convex gate mixtures can overshoot [0,1] by floating-point roundoff.
    scores = scores.clamp(0.0, 1.0)
    truth = labels * scores + (1 - labels) * (1 - scores)
    v = ltn.Variable('example', truth)
    forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier='f')
    predicates = [ltn.Predicate(func=lambda t, j=j: t[:, j]) for j in range(4)]
    satisfaction = ltn.fuzzy_ops.SatAgg()(*[forall(v, p(v)) for p in predicates])
    return 1 - satisfaction


def train(args, train_data, tune_data):
    torch.manual_seed(args.seed)
    reader, head = VisualReader(), LogicHead()
    optimizer = torch.optim.Adam(list(reader.parameters()) + list(head.parameters()), lr=0.008)
    x, c, y, _ = train_data
    best, checkpoint, history, first_grad = -1, None, [], None
    for epoch in range(args.epochs):
        reader.train()
        temp = max(0.2, 1 - epoch / max(1, args.epochs - 1) * 0.8)
        for ids in torch.randperm(len(x)).split(128):
            q = reader(x[ids])
            p = head(q, temp)
            scores = torch.cat([q, p[:, None]], dim=1)
            labels = torch.cat([c[ids], y[ids, None]], dim=1)
            loss = ltn_loss(scores, labels) if args.objective == 'ltn' else nn.functional.binary_cross_entropy(scores.clamp(1e-6, 1-1e-6), labels)
            optimizer.zero_grad()
            loss.backward()
            if first_grad is None:
                first_grad = {'reader_gradient_norm': float(reader.net[0].weight.grad.norm()),
                              'gate_gradient_norm': float(head.gate_logits.grad.norm())}
                assert min(first_grad.values()) > 0, 'Broken gradient path'
            optimizer.step()
        reader.eval()
        with torch.no_grad():
            qt = reader(tune_data[0]).numpy()
        ca = float(((qt >= 0.5) == tune_data[1].numpy()).mean())
        ya = float((head.hard(qt >= 0.5) == tune_data[2].numpy()).mean())
        history.append({'epoch': epoch, 'loss_last_batch': float(loss.detach()),
                        'tune_concept_accuracy': ca, 'tune_hard_task_accuracy': ya})
        if ca + ya > best:
            best = ca + ya
            checkpoint = (copy.deepcopy(reader.state_dict()), copy.deepcopy(head.state_dict()))
        if epoch % 20 == 0:
            print(f'epoch={epoch} tune_concept={ca:.3f} tune_task={ya:.3f}', flush=True)
    reader.load_state_dict(checkpoint[0])
    head.load_state_dict(checkpoint[1])
    reader.eval()
    return reader, head, history, first_grad


def compile_pyxai(table):
    from pyxai import Builder, Explaining, Tools
    Tools.set_verbose(0)
    def expand(prefix):
        mask = np.all(WORLDS[:, :len(prefix)] == prefix, axis=1)
        values = np.unique(table[mask])
        if len(values) == 1:
            # DecisionNode wraps numeric children in LeafNode internally.
            return int(values[0])
        return Builder.DecisionNode(len(prefix) + 1, operator=Builder.GE, threshold=0.5,
                                    left=expand(prefix + [0]), right=expand(prefix + [1]))
    tree = Builder.DecisionTree(3, expand([]), n_classes=2,
                                feature_names=NAMES, force_features_equal_to_binaries=True)
    exp = Explaining.initialize(tree, instance=WORLDS[0].tolist(), features_type={'binary': NAMES})
    predicted = np.array([exp.predict(w.tolist()) for w in WORLDS])
    assert np.array_equal(predicted, table), 'Export changed the deployed Boolean function'
    return exp


def completion_mask(reason):
    mask = np.ones(len(WORLDS), dtype=bool)
    for lit in reason:
        mask &= WORLDS[:, abs(lit)-1] == (lit > 0)
    return mask


def all_minimal_reasons(bits, table):
    target = table[np.all(WORLDS == bits, axis=1)][0]
    reasons = []
    for k in range(4):
        for inds in itertools.combinations(range(3), k):
            reason = tuple((j+1) if bits[j] else -(j+1) for j in inds)
            if any(set(r).issubset(reason) for r in reasons):
                continue
            if np.all(table[completion_mask(reason)] == target):
                reasons.append(reason)
    return sorted(reasons, key=lambda r: (len(r), r))


def verify_pyxai(exp, table):
    from pyxai import Explaining
    checked = 0
    bank = {}
    for w in WORLDS:
        exp.set_instance(w.tolist())
        # force_features_equal_to_binaries makes literal j mean concept j.
        assert tuple(exp.binary_representation) == tuple((i+1) if v else -(i+1) for i, v in enumerate(w))
        target = int(exp.predict(w.tolist()))
        reasons = [tuple(sorted(map(int, r), key=abs)) for r in exp.sufficient_reason(n=Explaining.ALL)]
        for reason in reasons:
            assert completion_mask(reason).any()
            assert np.all(table[completion_mask(reason)] == target)
            assert all(w[abs(lit)-1] == (lit > 0) for lit in reason)
            for lit in reason:
                smaller = tuple(v for v in reason if v != lit)
                assert np.any(table[completion_mask(smaller)] != target), 'Not subset-minimal'
            checked += 1
        assert set(reasons) == set(all_minimal_reasons(w, table)), 'Enumeration differs from exhaustive checker'
        bank[tuple(w)] = sorted(reasons, key=lambda r: (len(r), r))
    return bank, checked


def calibrate(q, c, alpha):
    # Maximum over coordinates calibrates the full vector, without independence.
    scores = np.where(c == 1, 1-q, q).max(axis=1)
    k = math.ceil((len(scores) + 1) * (1-alpha))
    # Score is bounded by one; this represents the all-worlds set when k>n.
    return float(np.sort(scores)[k-1]) if k <= len(scores) else 1.0


def semantically_true(reason, c):
    return bool(all(c[abs(lit)-1] == (lit > 0) for lit in reason))


def evaluate(q, c, y, visible, soft_pred, table, bank, threshold):
    records = []
    for i, probs in enumerate(q):
        bits = (probs >= 0.5).astype(int)
        reasons = bank[tuple(bits)]
        ordinary = reasons[0]
        plausible = np.where(WORLDS == 1, 1-probs, probs).max(axis=1) <= threshold
        viable = [r for r in reasons if plausible.any() and np.all(completion_mask(r)[plausible])]
        robust = viable[0] if viable else None
        hard_pred = int(table[np.all(WORLDS == bits, axis=1)][0])
        row = {'id': f'test-{i:05d}', 'target': int(y[i]), 'concept_true': c[i].astype(int).tolist(),
               'concept_score': probs.tolist(), 'concept_bits': bits.tolist(), 'visible': visible[i].tolist(),
               'hard_prediction': hard_pred, 'soft_prediction': int(soft_pred[i]),
               'ordinary_reason': ordinary, 'ordinary_semantic_correct': semantically_true(ordinary, c[i]),
               'accepted': robust is not None, 'robust_reason': robust,
               'robust_semantic_correct': semantically_true(robust, c[i]) if robust is not None else None,
               'length_matched': robust is not None and len(ordinary) == len(robust),
               'concept_set_size': int(plausible.sum()),
               'concept_set_covers_truth': bool(np.where(c[i] == 1, 1-probs, probs).max() <= threshold)}
        records.append(row)
    accepted = [r for r in records if r['accepted']]
    matched = [r for r in accepted if r['length_matched']]
    def mean(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else None
    metrics = {
        'test_count': len(records), 'accepted_count': len(accepted), 'matched_count': len(matched),
        'task_accuracy': float(np.mean([r['target'] == r['hard_prediction'] for r in records])),
        'concept_accuracy': float(((q >= 0.5) == c).mean()),
        'soft_hard_disagreement': float(np.mean([r['soft_prediction'] != r['hard_prediction'] for r in records])),
        'joint_concept_coverage': mean(records, 'concept_set_covers_truth'),
        'explanation_coverage': len(accepted)/len(records),
        'ordinary_semantic_correct_all': mean(records, 'ordinary_semantic_correct'),
        'ordinary_semantic_correct_same_accepted': mean(accepted, 'ordinary_semantic_correct'),
        'robust_semantic_correct_accepted': mean(accepted, 'robust_semantic_correct'),
        'issued_false_reason_rate_all_test': sum(not r['robust_semantic_correct'] for r in accepted)/len(records),
        'paired_same_image_same_length_delta': float(np.mean([int(r['robust_semantic_correct'])-int(r['ordinary_semantic_correct']) for r in matched])) if matched else None,
    }
    # Logical event inclusion, not a claim about conditional accepted-set risk.
    assert all(r['robust_semantic_correct'] or not r['concept_set_covers_truth'] for r in accepted)
    return records, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--objective', choices=['ltn', 'bce'], default='ltn')
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--out', type=Path, default=Path('results/ltn_seed7'))
    args = p.parse_args()
    if not 0 < args.alpha < 1 or args.epochs < 1:
        p.error('Require 0 < alpha < 1 and epochs >= 1')
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    split_names = ['train', 'tune', 'calibration', 'test']
    data = [make_data(n, args.seed*100 + k) for k, n in enumerate([1600, 400, 800, 800])]
    reader, head, history, grad = train(args, data[0], data[1])
    with torch.no_grad():
        qcal = reader(data[2][0]).numpy()
        qtest = reader(data[3][0]).numpy()
        soft_pred = (head(torch.tensor(qtest), temperature=0.2).numpy() >= 0.5).astype(int)
    threshold = calibrate(qcal, data[2][1].numpy(), args.alpha)
    table = head.hard(WORLDS)
    exp = compile_pyxai(table)
    bank, checked = verify_pyxai(exp, table)
    records, metrics = evaluate(qtest, data[3][1].numpy(), data[3][2].numpy(), data[3][3], soft_pred, table, bank, threshold)
    target_table = WORLDS[:, 0] & (WORLDS[:, 1] | WORLDS[:, 2])
    metrics.update(grad)
    metrics.update({'objective': args.objective, 'seed': args.seed, 'formula': head.formula(),
                    'alpha': args.alpha, 'calibration_threshold': threshold,
                    'export_equivalence_worlds_checked': len(WORLDS), 'pyxai_reasons_independently_checked': checked,
                    'target_truth_table_agreement': float(np.mean(table == target_table)),
                    'status': 'synthetic integration test only; not evidence of a research advantage'})
    (args.out/'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False)+'\n')
    (args.out/'per_example.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (args.out/'training.json').write_text(json.dumps(history, indent=2)+'\n')
    versions = {name: importlib.metadata.version(name) for name in ['torch', 'LTNtorch', 'pyxai', 'numpy']}
    config = vars(args).copy(); config['out'] = str(args.out)
    (args.out/'provenance.json').write_text(json.dumps({'config': config, 'versions': versions,
        'split_seeds': dict(zip(split_names, [args.seed*100+k for k in range(4)])),
        'upstream_ltn_commit_inspected': 'd1bd98169cc2121f8cdd25ff99901e4589923c95',
        'upstream_pyxai_commit_inspected': '2eb18994832e67b7ce082e357f18b5f2d12631a6',
        'installed_packages': 'PyPI releases; not GitHub HEAD builds'}, indent=2)+'\n')
    torch.save({'reader': reader.state_dict(), 'head': head.state_dict()}, args.out/'model.pt')
    np.savez_compressed(args.out/'calibration.npz', q=qcal, concepts=data[2][1].numpy())
    with (args.out/'truth_table.csv').open('w') as f:
        writer = csv.writer(f); writer.writerow(NAMES+['learned_output', 'target_output'])
        writer.writerows([list(w)+[int(a),int(b)] for w,a,b in zip(WORLDS,table,target_table)])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(9, 5))
    for ax,i in zip(axes.flat, range(8)):
        ax.imshow(data[3][0][i].permute(1,2,0).numpy())
        ax.set_title(f"c={data[3][1][i].int().tolist()}\ny={int(data[3][2][i])}", fontsize=9)
        ax.axis('off')
    fig.suptitle('Synthetic images: hidden attributes remain labelled under occlusion')
    fig.tight_layout(); fig.savefig(args.out/'sample_images.png', dpi=130); plt.close(fig)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
