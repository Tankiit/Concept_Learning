import itertools
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from .logic import BooleanHead, compile_boolean_function, fit_tree, threshold_scores
from .predicates import CropPredicateReader, masked_concept_loss, region_union, replace_outside
from .demo import run
from .train_tree import train


def brute_reasons(head, x):
    result = set()
    literals = [j+1 if value else -(j+1) for j, value in enumerate(x)]
    worlds = list(itertools.product((0, 1), repeat=len(x)))
    for size in range(len(x)+1):
        for r in itertools.combinations(literals, size):
            if any(set(previous).issubset(r) for previous in result):
                continue
            compatible = [w for w in worlds if all(w[abs(l)-1] == int(l > 0) for l in r)]
            if all(head.predict(w) == head.predict(x) for w in compatible):
                result.add(r)
    return result


class LogicTests(unittest.TestCase):
    def test_pyxai_all_reasons_agree_with_independent_enumeration(self):
        functions = [lambda c: c[0] and (c[1] or not c[2]),
                     lambda c: sum(c) % 2,
                     lambda c: (c[0] + c[1] + c[2]) % 3]
        for f in functions:
            h = compile_boolean_function(f, ['a', 'b', 'c'], ['0', '1', '2'])
            for x in itertools.product((0, 1), repeat=3):
                self.assertEqual(set(h.reasons(x, all_reasons=True)), brute_reasons(h, x))

    def test_constant_and_negative_reasons(self):
        h = compile_boolean_function(lambda c: 1, ['a', 'b'])
        self.assertEqual(h.reasons([0, 1]), [()])
        h = compile_boolean_function(lambda c: not c[1], ['a', 'b'])
        self.assertEqual(h.reasons([1, 0]), [(-2,)])
        self.assertFalse(h.verify((-2,), [1, 1]))
        with self.assertRaises(ValueError):
            h.verify((2, -2), [1, 0])

    def test_repeated_feature_consistency(self):
        # The class-1 leaf is unreachable, since it requires a=0 AND a=1.
        h = BooleanHead(['a'], {'feature': 1, 'false': {'feature': 1, 'false': 0, 'true': 1}, 'true': 0})
        self.assertIsNone(h.counterexample((), 0))
        self.assertEqual(h.reasons([0]), [()])

    def test_learned_tree_exact_export_and_roundtrip(self):
        x = np.array(list(itertools.product((0, 1), repeat=4)))
        y = np.array([(int(row[0]) + int(row[2])) % 3 for row in x])
        h = fit_tree(x, y, ['a', 'b', 'c', 'd'], min_samples_leaf=1)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'head.json'
            h.save(p, [.5]*4)
            restored, t = BooleanHead.load(p)
            exp = restored.pyxai_explainer()
            for row, label in zip(x, y):
                self.assertEqual(int(restored.class_names[restored.predict(row)]), label)
                self.assertEqual(exp.predict(row.tolist()), restored.predict(row))
                self.assertEqual(set(restored.reasons(row, all_reasons=True)), brute_reasons(restored, row))

    def test_unknown_is_not_false(self):
        for q in [[.9, np.nan], [.5, np.inf], [.1, 2.0]]:
            with self.assertRaises(ValueError):
                threshold_scores(q)
        np.testing.assert_array_equal(threshold_scores([.5, .49]), [1, 0])
        logits = torch.tensor([[0., 2.]], requires_grad=True)
        loss = masked_concept_loss(logits, torch.tensor([[1., float('nan')]]),
                                   torch.tensor([[True, False]]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(logits.grad[0, 1]), 0.)

    def test_pos_weight_reweights_only_positives(self):
        # Rare-positive concept: unweighted loss is dominated by the negative;
        # pos_weight=8 must move the optimum toward predicting positive.
        labels = torch.tensor([[1., 0.], [0., 0.]])
        observed = torch.ones((2, 2), dtype=torch.bool)
        w = torch.tensor([8., 1.])
        for raw in (0., 1., 2.):
            logits = torch.full((2, 2), float(raw), requires_grad=True)
            base = masked_concept_loss(logits, labels, observed)
            weighted = masked_concept_loss(logits, labels, observed, pos_weight=w)
            self.assertTrue(torch.isfinite(weighted))
            self.assertGreater(float(weighted), float(base))
        with self.assertRaises(ValueError):
            masked_concept_loss(torch.zeros(2, 2), labels, observed, pos_weight=torch.zeros(2))
        with self.assertRaises(ValueError):
            masked_concept_loss(torch.zeros(2, 2), labels, observed, pos_weight=torch.ones(3))

    def test_crop_dependency_with_learned_modules(self):
        torch.manual_seed(7)
        net = CropPredicateReader(3).eval()
        boxes = np.array([[0, 5, 0, 5], [2, 7, 2, 7], [6, 8, 6, 8]])
        x = torch.rand(3, 8, 8, requires_grad=True)
        q = net(x[None], boxes[None])[0]
        for j in range(3):
            support = region_union(boxes, (j+1,), (8, 8))
            gradient = torch.autograd.grad(q[j], x, retain_graph=True)[0]
            self.assertEqual(int(torch.count_nonzero(gradient[:, ~support])), 0)
            for seed in range(3):
                edited = replace_outside(x.detach(), support, seed)
                after = net(edited[None], boxes[None])[0, j]
                torch.testing.assert_close(after, q[j], rtol=0, atol=0)

    def test_union_counts_overlap_and_negative_literals(self):
        boxes = np.array([[0, 4, 0, 4], [2, 6, 2, 6]])
        mask = region_union(boxes, (1, -2), (8, 8))
        self.assertEqual(int(mask.sum()), 16+16-4)
        self.assertEqual(int(region_union(boxes, (), (8, 8)).sum()), 0)

    def test_spatial_demo(self):
        result = run()
        self.assertEqual({tuple(r['reason']) for r in result['certificates']}, {(1, 2), (1, -3)})
        self.assertTrue(all(r['area_fraction'] == .5 for r in result['certificates']))

    def test_real_data_interface_and_group_leakage_rejection(self):
        worlds = np.array(list(itertools.product((0, 1), repeat=3)))
        x = np.tile(worlds, (3, 1))
        n = len(x)
        d = dict(q=.1+.8*x, y=x[:, 0] & (x[:, 1] | (1-x[:, 2])),
                 concept_names=np.array(['a', 'b', 'c']),
                 split=np.array(['train']*16+['test']*8),
                 ids=np.array([f'i{i}' for i in range(n)]), groups=np.array([f'g{i}' for i in range(n)]))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'data.npz'
            np.savez(p, **d)
            result = train(p, Path(tmp) / 'model', min_samples_leaf=1)
            self.assertEqual(result['test_accuracy'], 1.)
            d['groups'][-1] = d['groups'][0]
            np.savez(p, **d)
            with self.assertRaisesRegex(ValueError, 'multiple splits'):
                train(p, Path(tmp) / 'bad')


if __name__ == '__main__':
    unittest.main()
