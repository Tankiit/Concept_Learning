"""Run: python -m boolean_cbm.demo

Deterministic synthetic example; it is not a CUB accuracy experiment.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from .logic import compile_boolean_function, threshold_scores
from .predicates import region_union, replace_outside


NAMES = ('red_in_patch_1', 'green_in_patch_2', 'blue_in_patch_3')
BOXES = np.array([[0, 4, 0, 4], [0, 4, 4, 8], [4, 8, 0, 4]])


def rule(c):
    return int(c[0] and (c[1] or not c[2]))


def read_scores(image):
    # A transparent toy reader. A learned CropPredicateReader has the same
    # spatial dependency pattern; its semantic quality must be measured.
    return np.array([float(image[j, y0:y1, x0:x1].mean())
                     for j, (y0, y1, x0, x1) in enumerate(BOXES)])


def run(output=None):
    head = compile_boolean_function(rule, NAMES)
    image = torch.zeros(3, 8, 8)
    image[0, :4, :4] = 0.9
    image[1, :4, 4:] = 0.8
    image[2, 4:, :4] = 0.1
    bits = threshold_scores(read_scores(image))
    reasons = head.reasons(bits, all_reasons=True)
    records = []
    for reason in reasons:
        keep = region_union(BOXES, reason, image.shape[-2:])
        trials = []
        for seed in range(25):
            edited = replace_outside(image, keep, seed)
            changed_bits = threshold_scores(read_scores(edited))
            if not torch.equal(edited[:, keep], image[:, keep]):
                raise RuntimeError('Edit touched protected pixels')
            if head.predict(changed_bits) != head.predict(bits):
                raise RuntimeError('Spatial composition invariant failed')
            trials.append(changed_bits.tolist())
        records.append({'reason': list(reason), 'named_literals': head.describe(reason),
            'area_fraction': float(keep.mean()), 'outside_edits_checked': len(trials)})
    checked = 0
    for world in itertools.product((0, 1), repeat=3):
        for reason in head.reasons(world, all_reasons=True):
            for completion in itertools.product((0, 1), repeat=3):
                if all(completion[abs(l)-1] == int(l > 0) for l in reason):
                    if rule(completion) != rule(world):
                        raise RuntimeError('Exhaustive sufficiency check failed')
            checked += 1
    result = {'synthetic_only': True, 'formula': 'c1 AND (c2 OR NOT c3)',
        'bits': bits.tolist(), 'prediction': head.predict(bits), 'certificates': records,
        'boolean_worlds_checked': 8, 'reasons_checked': checked,
        'scope': 'fixed boxes and deterministic crop readers; no claim of semantic correctness'}
    if output:
        Path(output).write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
