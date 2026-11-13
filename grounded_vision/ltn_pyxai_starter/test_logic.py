"""Exact small-domain checks; these test semantics, not learned performance."""
import itertools
import unittest
import numpy as np
from pilot import WORLDS, all_minimal_reasons, compile_pyxai, completion_mask, verify_pyxai, calibrate


class LogicChecks(unittest.TestCase):
    def test_all_four_learnable_formulas(self):
        for inner_or, outer_or in itertools.product([False, True], repeat=2):
            inner = WORLDS[:, 1] | WORLDS[:, 2] if inner_or else WORLDS[:, 1] & WORLDS[:, 2]
            table = WORLDS[:, 0] | inner if outer_or else WORLDS[:, 0] & inner
            _, count = verify_pyxai(compile_pyxai(table), table)
            self.assertGreaterEqual(count, 8)

    def test_unknown_completion_semantics(self):
        table = WORLDS[:, 0] & (WORLDS[:, 1] | WORLDS[:, 2])
        # For every partial observation, a supported sufficient reason exists
        # iff the decision is constant across its (nonempty) completions.
        for partial in itertools.product([0, 1, None], repeat=3):
            observed = tuple((j+1) if v else -(j+1) for j,v in enumerate(partial) if v is not None)
            mask = completion_mask(observed)
            bits = WORLDS[mask][0]
            candidates = all_minimal_reasons(bits, table)
            supported = any(np.all(completion_mask(r)[mask]) for r in candidates)
            self.assertEqual(supported, len(np.unique(table[mask])) == 1)

    def test_joint_calibration_uses_max_not_average(self):
        q = np.array([[.9,.8,.7],[.1,.2,.9],[.6,.4,.2]])
        c = np.array([[1,1,1],[0,0,0],[1,1,0]])
        self.assertAlmostEqual(calibrate(q,c,0.5), 0.6)
        self.assertEqual(calibrate(q,c,0.01), 1.0)


if __name__ == '__main__':
    unittest.main()
