"""The stored tree is the deployed classifier, not a surrogate for another head.

All features are Boolean. Signed literals use one-based feature IDs. No theory
of species co-occurrence or concept exclusivity restricts the completion space.
"""
import itertools
import json
from pathlib import Path

import numpy as np


def threshold_scores(scores, thresholds=0.5):
    q = np.asarray(scores, dtype=float)
    t = np.asarray(thresholds, dtype=float)
    if q.ndim not in (1, 2) or q.shape[-1] == 0:
        raise ValueError('Expected [M] or [N,M] scores')
    if t.ndim > 1 or (t.ndim == 1 and t.shape != (q.shape[-1],)):
        raise ValueError('Use a scalar threshold or one threshold per predicate')
    if not np.isfinite(q).all() or not np.isfinite(t).all():
        raise ValueError('Unknown/NaN scores cannot silently become false predicates')
    if not ((q >= 0) & (q <= 1)).all() or not ((t > 0) & (t < 1)).all():
        raise ValueError('Scores must be in [0,1], thresholds in (0,1)')
    return (q >= t).astype(np.int8)


def _bits(bits, m):
    x = np.asarray(bits)
    if x.shape != (m,) or not np.isin(x, [0, 1]).all():
        raise ValueError(f'Expected {m} Boolean values; unknown is not false')
    return x.astype(np.int8)


class BooleanHead:
    def __init__(self, names, root, class_names=('negative', 'positive')):
        self.names = tuple(names)
        self.class_names = tuple(class_names)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError('Predicate names must be nonempty and unique')
        if not all(isinstance(n, str) and n for n in self.names):
            raise ValueError('Use nonempty string names')
        if not self.class_names or not all(isinstance(n, str) for n in self.class_names):
            raise ValueError('Use string class names')
        self._validate(root)
        # Own the deployed function; later caller mutations cannot change it.
        self.root = json.loads(json.dumps(root))
        self._explainer = None

    def _validate(self, node):
        if type(node) is int:
            if not 0 <= node < len(self.class_names):
                raise ValueError('Leaf class index out of range')
            return
        if not isinstance(node, dict) or set(node) != {'feature', 'false', 'true'}:
            raise ValueError('Node requires feature, false, true')
        j = node['feature']
        if type(j) is not int or not 1 <= j <= len(self.names):
            raise ValueError('Node feature IDs are one-based')
        self._validate(node['false'])
        self._validate(node['true'])

    def predict(self, bits):
        x = _bits(bits, len(self.names))
        node = self.root
        while isinstance(node, dict):
            node = node['true' if x[node['feature'] - 1] else 'false']
        return node

    def _assignment(self, reason):
        assignments = {}
        for lit in reason:
            if not isinstance(lit, (int, np.integer)) or isinstance(lit, bool):
                raise ValueError('Reason literals must be signed integer IDs')
            if not 1 <= abs(lit) <= len(self.names) or abs(lit) in assignments:
                raise ValueError('Invalid, duplicate, or contradictory literal')
            assignments[abs(lit)] = int(lit > 0)
        return assignments

    def counterexample(self, reason, target):
        """Exact search of tree leaves under a partial assignment, no sampling.

        Returns a complete Boolean input changing the target, or None if UNSAT.
        Repeated tests of a feature must share the same value along the path.
        """
        if not 0 <= target < len(self.class_names):
            raise ValueError('Invalid target')
        def search(node, assignment):
            if type(node) is int:
                if node == target:
                    return None
                return tuple(assignment.get(j, 0) for j in range(1, len(self.names) + 1))
            j = node['feature']
            values = (assignment[j],) if j in assignment else (0, 1)
            for value in values:
                found = search(node['true' if value else 'false'], {**assignment, j: value})
                if found is not None:
                    return found
            return None
        return search(self.root, self._assignment(reason))

    def verify(self, reason, bits, require_subset_minimal=True):
        x = _bits(bits, len(self.names))
        assignment = self._assignment(reason)
        if any(x[j-1] != v for j, v in assignment.items()):
            return False
        target = self.predict(x)
        if self.counterexample(reason, target) is not None:
            return False
        if require_subset_minimal:
            r = tuple(reason)
            for i in range(len(r)):
                if self.counterexample(r[:i] + r[i+1:], target) is None:
                    return False
        return True

    def pyxai_explainer(self):
        if self._explainer is None:
            from pyxai import Builder, Explaining, Tools
            Tools.set_verbose(0)
            def convert(node):
                if type(node) is int:
                    return node
                return Builder.DecisionNode(node['feature'], operator=Builder.GE, threshold=0.5,
                    left=convert(node['false']), right=convert(node['true']))
            root = convert(self.root)
            if type(root) is int:
                root = Builder.LeafNode(root)
            tree = Builder.DecisionTree(len(self.names), root,
                n_classes=len(self.class_names), feature_names=list(self.names),
                force_features_equal_to_binaries=True)
            self._explainer = Explaining.initialize(tree, instance=[0] * len(self.names))
        return self._explainer

    def reasons(self, bits, all_reasons=False):
        """PyXAI subset-minimal reasons, then independent exact verification.

        `all_reasons=True` can be exponential. Default returns one reason, not
        necessarily a minimum-cardinality or minimum-area reason.
        """
        x = _bits(bits, len(self.names))
        target = self.predict(x)
        # PyXAI 2.0.1 enumeration fails for a leaf-only constant tree.
        if self.counterexample((), target) is None:
            return [()]
        from pyxai import Explaining
        exp = self.pyxai_explainer()
        exp.set_instance(x.tolist())
        if exp.target_prediction != target:
            raise RuntimeError('PyXAI export disagrees with deployed tree')
        raw = exp.sufficient_reason(n=Explaining.ALL if all_reasons else 1)
        raw = raw if all_reasons else [raw]
        result = [tuple(sorted(map(int, r), key=abs)) for r in raw]
        if not result:
            raise RuntimeError('No reason returned; no certificate issued')
        for reason in result:
            if not self.verify(reason, x):
                raise RuntimeError('Reason failed exact sufficiency/minimality check')
        return sorted(set(result), key=lambda r: (len(r), r))

    def describe(self, reason):
        self._assignment(reason)
        return [self.names[abs(l)-1] if l > 0 else 'NOT ' + self.names[abs(l)-1] for l in reason]

    def save(self, path, thresholds=0.5):
        threshold_scores(np.zeros(len(self.names)), thresholds)
        Path(path).write_text(json.dumps({'schema': 1, 'predicate_names': self.names,
            'class_names': self.class_names, 'thresholds': np.asarray(thresholds).tolist(),
            'root': self.root}, indent=2) + '\n')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data['schema'] != 1:
            raise ValueError('Unsupported schema')
        threshold_scores(np.zeros(len(data['predicate_names'])), data['thresholds'])
        return cls(data['predicate_names'], data['root'], data['class_names']), data['thresholds']


def compile_boolean_function(function, names, class_names=('negative', 'positive'), max_concepts=10):
    """Exact full-cube expansion of a small hard function; never distillation."""
    m = len(names)
    if not 1 <= m <= max_concepts:
        raise ValueError('Truth-table export is limited to small heads; use circuit SAT otherwise')
    worlds = list(itertools.product((0, 1), repeat=m))
    labels = [function(w) for w in worlds]
    if any(not isinstance(y, (int, np.integer, bool)) or not 0 <= y < len(class_names) for y in labels):
        raise ValueError('The hard function must return a valid class index')
    def expand(values, depth):
        if len(set(values)) == 1:
            return int(values[0])
        half = len(values) // 2
        return {'feature': depth + 1, 'false': expand(values[:half], depth+1),
                'true': expand(values[half:], depth+1)}
    head = BooleanHead(names, expand(labels, 0), class_names)
    exp = head.pyxai_explainer()
    for x, y in zip(worlds, labels):
        if head.predict(x) != y or exp.predict(list(x)) != y:
            raise RuntimeError('Full-cube export equivalence failed')
    return head


def fit_tree(bits, labels, names, max_depth=6, min_samples_leaf=5, seed=0):
    """Train the actual downstream head on frozen reader bits, not teacher labels.

    Exact structural export scales with tree size, not 2**number_of_concepts.
    """
    from sklearn.tree import DecisionTreeClassifier
    x, y = np.asarray(bits), np.asarray(labels)
    if x.ndim != 2 or x.shape[1] != len(names) or len(x) == 0 or not np.isin(x, [0, 1]).all():
        raise ValueError('Expected nonempty binary [N,M] training data')
    if y.shape != (len(x),):
        raise ValueError('Expected one task label per row')
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        random_state=seed).fit(x, y)
    t = model.tree_
    def convert(i):
        if t.children_left[i] == t.children_right[i]:
            return int(t.value[i].argmax())
        if not 0 <= t.threshold[i] < 1:
            raise RuntimeError('A learned split does not separate Boolean values')
        return {'feature': int(t.feature[i]+1), 'false': convert(t.children_left[i]),
                'true': convert(t.children_right[i])}
    head = BooleanHead(names, convert(0), tuple(map(str, model.classes_)))
    if not np.array_equal([head.class_names[head.predict(row)] for row in x],
                          model.predict(x).astype(str)):
        raise RuntimeError('Structural export check failed')
    return head
