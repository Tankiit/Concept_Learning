"""Exact gate CNF + cardinality class comparisons, with bounded reason search.

Reason search may be incomplete because of its length/query budget. Every issued
reason is checked by an exact solver. Unknown solver results never issue a reason.
"""
import itertools
import math
import numpy as np
from pysat.card import CardEnc, EncType
from pysat.solvers import Solver


def predict(circuit, bits):
    x = np.atleast_2d(bits).astype(np.int64)
    for layer in circuit['layers']:
        a, b = x[:, layer['a']], x[:, layer['b']]
        x = (np.asarray(layer['gate'])[None, :] >> (3 - (2*a+b))) & 1
    return x.reshape(len(x), circuit['classes'], -1).sum(-1).argmax(-1)


def calibrate(q, concepts, alpha=.1):
    q, concepts = np.asarray(q), np.asarray(concepts)
    if not 0 < alpha < 1 or len(q) == 0 or q.shape != concepts.shape:
        raise ValueError('Invalid calibration shapes/alpha')
    if not np.isin(concepts, [0, 1]).all() or not np.isfinite(q).all() or not ((q>=0)&(q<=1)).all():
        raise ValueError('Calibration requires complete binary labels and finite probabilities')
    scores = np.where(concepts == 1, 1-q, q).max(1)
    k = math.ceil((len(q)+1)*(1-alpha))
    return float(np.sort(scores)[k-1]) if k <= len(q) else 1.


def allowed_values(q, threshold):
    q = np.asarray(q)
    return np.stack([q <= threshold, 1-q <= threshold], axis=-1)


def semantic_truth(reason, c):
    return all(c[abs(l)-1] == int(l > 0) for l in reason)


class Explainer:
    def __init__(self, circuit, conflict_budget=100000):
        if circuit['tie_break'] != 'lowest_class_index' or circuit['aggregation'] != 'equal_group_sum':
            raise ValueError('Unsupported output semantics')
        self.circuit = circuit
        self.m = len(circuit['concept_names'])
        self.conflict_budget = conflict_budget
        clauses = []
        nodes = list(range(1, self.m+1)); top = self.m
        for layer in circuit['layers']:
            new = []
            for ia, ib, gate in zip(layer['a'], layer['b'], layer['gate']):
                a, b = nodes[ia], nodes[ib]
                top += 1; z = top; new.append(z)
                for av, bv in itertools.product([0, 1], repeat=2):
                    out = (gate >> (3-(2*av+bv))) & 1
                    clauses.append([(-a if av else a), (-b if bv else b), z if out else -z])
            nodes = new
        self.clauses, self.top = clauses, top
        self.groups = np.array(nodes).reshape(circuit['classes'], -1).tolist()
        self.solvers = {}

    def _solver(self, target, challenger):
        key = (target, challenger)
        if key not in self.solvers:
            # Challenger beats target if score_j >= score_y (+1 if j>y).
            # sum(j) + sum(not y) >= group_size + tie_adjustment.
            lits = self.groups[challenger] + [-v for v in self.groups[target]]
            bound = len(self.groups[target]) + int(challenger > target)
            cnf = CardEnc.atleast(lits=lits, bound=bound, top_id=self.top,
                                 encoding=EncType.seqcounter)
            self.solvers[key] = Solver(name='g3', bootstrap_with=self.clauses + cnf.clauses)
        return self.solvers[key]

    def sufficient(self, reason, target):
        if any(abs(l) < 1 or abs(l) > self.m for l in reason) or any(-l in reason for l in reason):
            raise ValueError('Invalid or contradictory reason')
        for challenger in range(len(self.groups)):
            if challenger == target:
                continue
            solver = self._solver(target, challenger)
            solver.conf_budget(self.conflict_budget)
            result = solver.solve_limited(assumptions=list(reason))
            if result is None:
                return None
            if result:
                return False
        return True

    def shortest(self, bits, allowed=None, max_length=4, max_candidates=5000):
        bits = np.asarray(bits, dtype=int)
        target = int(predict(self.circuit, bits)[0])
        if allowed is None:
            indices = list(range(self.m))
        else:
            allowed = np.asarray(allowed, dtype=bool)
            if allowed.shape != (self.m, 2):
                raise ValueError('Expected allowed values with shape (concepts,2)')
            if not allowed.any(1).all():
                return None, 'empty_concept_set'
            indices = [j for j in range(self.m) if allowed[j].sum() == 1 and allowed[j,bits[j]]]
        queried = 0
        for k in range(min(max_length, len(indices))+1):
            for inds in itertools.combinations(indices, k):
                if queried >= max_candidates:
                    return None, 'candidate_budget'
                queried += 1
                reason = tuple(j+1 if bits[j] else -(j+1) for j in inds)
                valid = self.sufficient(reason, target)
                if valid is None:
                    return None, 'solver_budget'
                if valid:
                    return reason, 'verified'
        return None, 'no_reason_within_length'

    def close(self):
        for solver in self.solvers.values():
            solver.delete()
        self.solvers.clear()


def crosscheck_pyxai(circuit, max_concepts=10):
    """Exhaustive exact tree export only for tiny heads; never distillation."""
    m = len(circuit['concept_names'])
    if m > max_concepts:
        raise ValueError('PyXAI truth-table cross-check is restricted to small heads')
    from pyxai import Builder, Explaining, Tools
    Tools.set_verbose(0)
    worlds = np.array(list(itertools.product([0,1], repeat=m)))
    labels = predict(circuit, worlds)
    def build(prefix):
        vals = labels[np.all(worlds[:,:len(prefix)] == prefix, axis=1)]
        if len(np.unique(vals)) == 1:
            return int(vals[0])
        return Builder.DecisionNode(len(prefix)+1, operator=Builder.GE, threshold=.5,
                                   left=build(prefix+[0]), right=build(prefix+[1]))
    names = circuit['concept_names']
    root = build([])
    if isinstance(root, int):
        root = Builder.LeafNode(root)
    tree = Builder.DecisionTree(m, root, n_classes=circuit['classes'],
        feature_names=names, force_features_equal_to_binaries=True)
    exp = Explaining.initialize(tree, instance=worlds[0].tolist(), features_type={'binary':names})
    checker = Explainer(circuit)
    checked = 0
    constant_checks = 0
    try:
        for w, label in zip(worlds, labels):
            if exp.predict(w.tolist()) != label:
                raise RuntimeError('PyXAI export mismatch')
            exp.set_instance(w.tolist())
            # PyXAI 2.0.1's sufficient_reason errors on a leaf-only tree.
            # Check its predictions above; verify the empty reason with SAT.
            if len(np.unique(labels)) == 1:
                if checker.sufficient((),int(label)) is not True:
                    raise RuntimeError('Constant circuit must have an empty sufficient reason')
                constant_checks += 1
                continue
            for r in exp.sufficient_reason(n=Explaining.ALL):
                if checker.sufficient(tuple(map(int,r)),int(label)) is not True:
                    raise RuntimeError('PyXAI / circuit sufficiency mismatch')
                checked += 1
    finally:
        checker.close()
    return {'worlds':len(worlds), 'pyxai_reasons_checked':checked,
            'constant_empty_reasons_checked_by_sat':constant_checks}
