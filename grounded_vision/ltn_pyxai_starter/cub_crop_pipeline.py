"""CUB-200-2011 (official split): crop readers -> Boolean bits -> tree head -> PyXAI reasons.

Real-data execution of the boolean_cbm four-step pipeline. No synthetic data.

Steps (this file):
1. CropPredicateReader: one small CNN per canonical bound attribute, trained on
   part crops sliced from the raw image before any learned operation. The crop
   is resized to a fixed square after slicing (resize mixes pixels inside the
   crop only, so the region dependency contract is preserved). Deployment score
   of attribute j = sigmoid(reader.networks[j](resize(crop(image, box_j)))).
2. threshold_scores at a fixed 0.5 turns probabilities into deployed bits.
3. fit_tree: a CART over frozen bits and task labels IS the deployed classifier.
   Depth/leaf selected on a validation sub-split carved from official TRAIN
   only, then refit on all official TRAIN images.
4. head.reasons(): PyXAI sufficient reason per official-test image, then the
   independent exact sufficiency/minimality checker in boolean_cbm.logic.

Data contract (grounded_vision repository conventions):
- OFFICIAL CUB train/test split only (train_test_split.txt); test labels are
  read once at evaluation, never for fitting.
- Vocabulary: the repository's 28 canonical bound attributes. One documented
  substitution: canonical tail:tail_color=black binds CUB attribute
  has_upper_tail_color::black (the file has no has_tail_color::* names).
- Part boxes: the repository's fixed template sizes per part, centred on the
  CUB part point (pixels); the default template box is used when a point is
  missing. A point marked invisible still yields a box and a finite score:
  a finite score is a model prediction, not verified evidence.
- Unknown is not false: reader training masks annotations with CUB certainty
  level 1 (worker guesses; in this file every such row is an absent label) and
  crops of invisible parts. Semantic evaluation of issued literals uses the
  annotation present-flag as ground truth and reports certainty separately.

This is not a CUB accuracy benchmark and makes no grounding claim: crop
readers trained from scratch on ~6k images are expected to be the bottleneck
(repository milestone v2). The readout of interest is the truth rate of issued
reason literals under the official split.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from boolean_cbm.logic import fit_tree, threshold_scores
from boolean_cbm.predicates import CropPredicateReader, masked_concept_loss

CUB_ROOT = '/Users/tanmoy/research/data/CUB_200_2011/CUB_200_2011'

# Canonical 28 bound attributes (part, attribute, value) from the repository's
# src/grounded_vision/data/cub.py, with the tail-color CUB id substitution.
CANONICAL = [
    ('beak', 'bill_shape', 'curved'), ('beak', 'bill_shape', 'dagger'),
    ('beak', 'bill_shape', 'hooked'), ('beak', 'bill_shape', 'needle'),
    ('beak', 'bill_shape', 'cone'), ('beak', 'bill_color', 'black'),
    ('beak', 'bill_color', 'yellow'), ('crown', 'crown_color', 'black'),
    ('crown', 'crown_color', 'yellow'), ('crown', 'crown_color', 'red'),
    ('crown', 'crown_color', 'white'), ('forehead', 'forehead_color', 'black'),
    ('forehead', 'forehead_color', 'yellow'), ('breast', 'breast_color', 'yellow'),
    ('breast', 'breast_color', 'white'), ('breast', 'breast_color', 'black'),
    ('breast', 'breast_color', 'red'), ('breast', 'breast_pattern', 'striped'),
    ('belly', 'belly_color', 'yellow'), ('belly', 'belly_color', 'white'),
    ('belly', 'belly_color', 'black'), ('left_wing', 'wing_color', 'black'),
    ('left_wing', 'wing_color', 'brown'), ('left_wing', 'wing_pattern', 'striped'),
    ('right_wing', 'wing_color', 'black'), ('right_wing', 'wing_color', 'brown'),
    ('tail', 'tail_shape', 'forked'), ('tail', 'tail_color', 'black'),
]
TAIL_COLOR_OVERRIDE = 'has_upper_tail_color::black'
# Template part boxes in normalised (ymin, xmin, ymax, xmax); repository cub.py.
TEMPLATE = {
    'crown': (0.05, 0.35, 0.25, 0.65), 'forehead': (0.10, 0.40, 0.25, 0.60),
    'beak': (0.15, 0.55, 0.30, 0.85), 'left_eye': (0.12, 0.42, 0.20, 0.50),
    'right_eye': (0.12, 0.50, 0.20, 0.58), 'throat': (0.22, 0.42, 0.35, 0.65),
    'nape': (0.15, 0.30, 0.30, 0.45), 'breast': (0.30, 0.35, 0.55, 0.65),
    'back': (0.28, 0.20, 0.60, 0.45), 'belly': (0.50, 0.35, 0.75, 0.65),
    'left_wing': (0.35, 0.15, 0.70, 0.45), 'right_wing': (0.35, 0.55, 0.70, 0.85),
    'left_leg': (0.72, 0.38, 0.92, 0.48), 'right_leg': (0.72, 0.52, 0.92, 0.62),
    'tail': (0.65, 0.10, 0.95, 0.30),
}
CUB_PART_IDS = {  # 1-based part ids from parts.txt
    'back': 1, 'beak': 2, 'belly': 3, 'breast': 4, 'crown': 5, 'forehead': 6,
    'left_eye': 7, 'left_leg': 8, 'left_wing': 9, 'nape': 10, 'right_eye': 11,
    'right_leg': 12, 'right_wing': 13, 'tail': 14, 'throat': 15,
}
VOCAB_PARTS = sorted({p for p, _, _ in CANONICAL})


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_attribute_ids(cub_dir):
    name_to_id = {}
    for line in open(Path(cub_dir) / 'attributes.txt'):
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            name_to_id[parts[1]] = int(parts[0])
    ids = []
    for part, attr, value in CANONICAL:
        exact = f'has_{attr}::{value}'
        if exact in name_to_id:
            ids.append((name_to_id[exact], exact))
            continue
        if (part, attr, value) == ('tail', 'tail_color', 'black'):
            ids.append((name_to_id[TAIL_COLOR_OVERRIDE], TAIL_COLOR_OVERRIDE))
            continue
        hits = [aid for name, aid in name_to_id.items() if name.startswith(exact)]
        if len(hits) != 1:
            raise RuntimeError(f'Canonical attribute {exact} does not resolve uniquely')
        ids.append((hits[0], next(n for n, a in name_to_id.items() if a == hits[0] and n.startswith(exact))))
    return ids


def load_records(cub_dir, class_limit):
    cub_dir = Path(cub_dir)
    ids = resolve_attribute_ids(cub_dir)
    attr_ids = [aid for aid, _ in ids]

    id_to_path, id_to_label, id_to_split = {}, {}, {}
    for line in open(cub_dir / 'images.txt'):
        i, rel = line.split()
        id_to_path[int(i)] = rel
    for line in open(cub_dir / 'image_class_labels.txt'):
        i, c = line.split()
        id_to_label[int(i)] = int(c)
    for line in open(cub_dir / 'train_test_split.txt'):
        i, s = line.split()
        id_to_split[int(i)] = int(s)

    points = {}
    for line in open(cub_dir / 'parts' / 'part_locs.txt'):
        f = line.split()
        if len(f) != 5:
            continue
        img, pid, x, y, vis = int(f[0]), int(f[1]), float(f[2]), float(f[3]), int(f[4])
        points[(img, pid)] = (x, y, vis)

    want = set(attr_ids)
    annotations = {}
    for line in open(cub_dir / 'attributes' / 'image_attribute_labels.txt'):
        f = line.split()
        img, aid, present, certainty = int(f[0]), int(f[1]), int(f[2]), int(f[3])
        if aid in want:
            annotations[(img, aid)] = (present, certainty)

    records = []
    for img in sorted(id_to_path):
        label = id_to_label[img] - 1
        if class_limit and label >= class_limit:
            continue
        part_rows = {}
        for part in VOCAB_PARTS:
            part_rows[part] = points.get((img, CUB_PART_IDS[part]))
        attr_rows = []
        for aid in attr_ids:
            if (img, aid) not in annotations:
                raise RuntimeError(f'Missing annotation for image {img} attribute {aid}')
            attr_rows.append(annotations[(img, aid)])
        records.append({
            'id': img,
            'path': str(cub_dir / 'images' / id_to_path[img]),
            'species': id_to_path[img].split('/')[0],
            'label': label,
            'train': id_to_split[img] == 1,
            'parts': part_rows,
            'present': np.array([p for p, _ in attr_rows], dtype=np.int8),
            'certainty': np.array([c for _, c in attr_rows], dtype=np.int8),
        })
    return records, ids


def pixel_box(part, point, width, height):
    """Template-sized integer box (y0, y1, x0, x1), upper bounds exclusive."""
    ymin, xmin, ymax, xmax = TEMPLATE[part]
    h, w = (ymax - ymin) * height, (xmax - xmin) * width
    if point is None:
        y0, x0 = ymin * height, xmin * width
    else:
        x, y, _ = point
        y0 = float(np.clip(y - h / 2.0, 0.0, max(0.0, height - h)))
        x0 = float(np.clip(x - w / 2.0, 0.0, max(0.0, width - w)))
    return (int(math.floor(y0)), int(min(height, math.ceil(y0 + h))),
            int(math.floor(x0)), int(min(width, math.ceil(x0 + w))))


def extract_crops(records, crop_size, progress_every=1000):
    """Slice each part crop from the raw image, then resize to a fixed square.

    Slicing happens before any learned operation; the resize mixes pixels
    inside the crop only. Returned uint8 arrays halve the cache footprint.
    """
    cache = {part: [] for part in VOCAB_PARTS}
    meta = []
    for n, rec in enumerate(records):
        image = Image.open(rec['path']).convert('RGB')
        width, height = image.size
        for part in VOCAB_PARTS:
            y0, y1, x0, x1 = pixel_box(part, rec['parts'][part], width, height)
            crop = image.crop((x0, y0, x1, y1)).resize((crop_size, crop_size), Image.BILINEAR)
            cache[part].append(np.asarray(crop, dtype=np.uint8).transpose(2, 0, 1))
        meta.append((rec['id'], width, height))
        if progress_every and (n + 1) % progress_every == 0:
            print(f'  crops: {n + 1}/{len(records)} images', flush=True)
    return {part: np.stack(rows) for part, rows in cache.items()}, meta


def part_attribute_columns():
    cols = {part: [] for part in VOCAB_PARTS}
    for j, (part, _, _) in enumerate(CANONICAL):
        cols[part].append(j)
    return cols


@torch.no_grad()
def reader_scores(reader, cache, rows, cols, batch=512):
    """Vectorised scores for global row indices; exact equivalent of calling
    reader.networks[j] on each resized crop (asserted by contract_spot_check)."""
    reader.eval()
    q = np.zeros((len(rows), len(CANONICAL)), dtype=np.float64)
    for part, col_js in cols.items():
        X = cache[part][rows]
        for start in range(0, len(X), batch):
            chunk = torch.from_numpy(X[start:start + batch]).float() / 255.0
            for j in col_js:
                logits = reader.networks[j](chunk).reshape(-1)
                q[start + np.arange(len(chunk)), j] = torch.sigmoid(logits).numpy()
    return q


@torch.no_grad()
def contract_spot_check(reader, cache, cols, seed, n_checks=8):
    """Assert networks[j](crop) equals CropPredicateReader.forward(crop, boxes)
    with predicate j's box set to the full frame: same slicing + same module."""
    crop_size = cache[VOCAB_PARTS[0]].shape[-1]
    full = np.array([0, crop_size, 0, crop_size], dtype=int)
    picks = np.random.default_rng(seed).choice(len(cache[VOCAB_PARTS[0]]),
                                               size=min(n_checks, len(cache[VOCAB_PARTS[0]])), replace=False)
    worst = 0.0
    for k in picks:
        for part, col_js in cols.items():
            image = torch.from_numpy(cache[part][k:k + 1]).float() / 255.0
            boxes = np.tile(full, (1, len(CANONICAL), 1))
            for j in col_js:
                via_forward = float(reader(image, boxes).reshape(-1)[j])
                direct = float(reader.networks[j](image).reshape(-1)[0])
                worst = max(worst, abs(via_forward - direct))
    if worst > 1e-4:
        raise RuntimeError(f'CropPredicateReader.forward disagrees with direct net call ({worst})')
    return {'images_checked': int(len(picks)), 'max_abs_logit_difference': worst}


def train_readers(reader, cache, fit_rows, records, cols, args):
    """Masked BCE per part-batch. Mask = certainty >= 2 AND part point visible."""
    opt = torch.optim.Adam(reader.parameters(), lr=args.lr)
    part_data = {}
    for part, col_js in cols.items():
        col_index = np.array(col_js)
        present = np.stack([records[k]['present'] for k in fit_rows])[:, col_index]
        certainty = np.stack([records[k]['certainty'] for k in fit_rows])[:, col_index]
        visible = np.array([1 if records[k]['parts'][part] and records[k]['parts'][part][2] == 1 else 0
                            for k in fit_rows], dtype=bool)
        observed = visible[:, None] & (certainty >= args.min_certainty)
        pos = (present == 1) & observed
        neg = (present == 0) & observed
        pw = (neg.sum(0) / np.clip(pos.sum(0), 1, None)).clip(max=50.0)
        part_data[part] = {
            'X': cache[part][fit_rows], 'y': torch.from_numpy(present.astype(np.float32)),
            'obs': torch.from_numpy(observed), 'pw': torch.from_numpy(pw.astype(np.float32)),
            'cols': col_js,
        }
    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(args.epochs):
        reader.train()
        total, seen = 0.0, 0
        for part, d in part_data.items():
            order = rng.permutation(len(d['X']))
            for start in range(0, len(order), args.batch_size):
                sel = order[start:start + args.batch_size]
                chunk = torch.from_numpy(d['X'][sel]).float() / 255.0
                logits = torch.zeros(len(sel), len(d['cols']))
                for c, j in enumerate(d['cols']):
                    logits[:, c] = reader.networks[j](chunk).reshape(-1)
                loss = masked_concept_loss(logits, d['y'][sel], d['obs'][sel], pos_weight=d['pw'])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(sel)
                seen += int(d['obs'][sel].sum())
        history.append({'epoch': epoch, 'mean_masked_loss': total / max(1, seen)})
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f'  reader epoch {epoch}: masked loss {history[-1]["mean_masked_loss"]:.4f}', flush=True)
    return history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cub-dir', default=CUB_ROOT)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--classes', type=int, default=0, help='first K species (0 = all 200)')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--crop-size', type=int, default=48)
    p.add_argument('--val-fraction', type=float, default=0.15)
    p.add_argument('--min-certainty', type=int, default=2, help='mask CUB certainty-1 guesses in training')
    p.add_argument('--depths', default='4,6,8,12')
    p.add_argument('--min-leaves', default='2,4,8')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.epochs < 1 or not 0 < args.val_fraction < 0.5 or args.crop_size < 8:
        p.error('Invalid epochs, val fraction, or crop size')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('--> loading CUB metadata', flush=True)
    records, attr_ids = load_records(args.cub_dir, args.classes)
    train_idx = [k for k, r in enumerate(records) if r['train']]
    test_idx = [k for k, r in enumerate(records) if not r['train']]
    if not train_idx or not test_idx:
        p.error('Need both official train and test images')
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(train_idx))
    n_val = int(args.val_fraction * len(train_idx))
    val_pos = sorted(perm[:n_val].tolist())
    fit_pos = sorted(perm[n_val:].tolist())
    val_idx = [train_idx[k] for k in val_pos]
    fit_idx = [train_idx[k] for k in fit_pos]
    labels_train = np.array([records[k]['label'] for k in train_idx])
    print(f'--> {len(records)} images (official train {len(train_idx)} = fit {len(fit_idx)} + val {len(val_idx)}, test {len(test_idx)}), '
          f'{len(records[test_idx[0]]["present"])} concepts, {len(set(labels_train))} classes', flush=True)

    print('--> extracting part crops (slice raw image, then resize)', flush=True)
    cache, _ = extract_crops(records, args.crop_size)
    cols = part_attribute_columns()

    print('--> step 1: training CropPredicateReader on fit-split crops', flush=True)
    reader = CropPredicateReader(n_predicates=len(CANONICAL))
    history = train_readers(reader, cache, np.array(fit_idx), records, cols, args)
    check = contract_spot_check(reader, cache, cols, args.seed)

    print('--> scoring all images with the frozen reader', flush=True)
    all_rows = np.arange(len(records))
    q = reader_scores(reader, cache, all_rows, cols)
    bits = threshold_scores(q)

    print('--> step 3: fitting/selecting the deployed tree head', flush=True)
    names = [f'{part}:{attr}={value}' for part, attr, value in CANONICAL]
    y_train = [str(records[k]['label']) for k in train_idx]
    grid = [(d, l) for d in (int(x) for x in args.depths.split(','))
            for l in (int(x) for x in args.min_leaves.split(','))]
    table = []
    best = None
    for depth, leaf in grid:
        head = fit_tree(bits[train_idx], np.array(y_train), names, max_depth=depth, min_samples_leaf=leaf)
        acc = float(np.mean([head.class_names[head.predict(bits[k])] == str(records[k]['label']) for k in val_idx]))
        table.append({'max_depth': depth, 'min_samples_leaf': leaf, 'val_accuracy': acc})
        if best is None or acc > best[0]:
            best = (acc, depth, leaf)
    val_acc, depth, leaf = best
    head = fit_tree(bits[train_idx], np.array(y_train), names, max_depth=depth, min_samples_leaf=leaf)
    print(f'--> selected depth={depth} leaf={leaf} (val acc {val_acc:.3f}), refit on all official train', flush=True)

    print('--> step 4: PyXAI sufficient reasons + independent exact check on official test', flush=True)
    rows = []
    for k in test_idx:
        rec = records[k]
        reason = head.reasons(bits[k])[0]
        prediction = head.class_names[head.predict(bits[k])]
        literal_truth, literal_cert, literal_vis = [], [], []
        for lit in reason:
            j = abs(lit) - 1
            truth = int(rec['present'][j]) == int(lit > 0)
            literal_truth.append(bool(truth))
            literal_cert.append(int(rec['certainty'][j]))
            point = rec['parts'][CANONICAL[j][0]]
            literal_vis.append(bool(point and point[2] == 1))
        rows.append({
            'id': rec['id'], 'species': rec['species'], 'target': rec['label'],
            'prediction': prediction, 'correct': prediction == str(rec['label']),
            'reason': list(reason), 'named_literals': head.describe(reason),
            'literal_truth': literal_truth,
            'all_literals_true': all(literal_truth) if literal_truth else True,
            'literal_certainty': literal_cert, 'literal_parts_visible': literal_vis,
            'scores': q[k].round(4).tolist(), 'bits': bits[k].astype(int).tolist(),
            'exact_sufficiency_checked': True,
        })

    present = np.stack([records[k]['present'] for k in test_idx]).astype(int)
    certainty = np.stack([records[k]['certainty'] for k in test_idx])
    q_te, bits_te = q[test_idx], bits[test_idx]
    literal_all = [t for r in rows for t in r['literal_truth']]
    metrics = {
        'protocol': 'official CUB train/test split; readers+tree fit on train only; '
                    'hyperparameters selected on a train-carved validation sub-split',
        'test_count': len(rows),
        'classes': args.classes or 200,
        'task_accuracy': float(np.mean([r['correct'] for r in rows])),
        'majority_baseline': float(np.bincount([r['target'] for r in rows]).max() / len(rows)),
        'concept_accuracy_all': float((bits_te == present).mean()),
        'concept_accuracy_certainty_ge2': float((bits_te == present)[certainty >= 2].mean()),
        'reason_mean_length': float(np.mean([len(r['reason']) for r in rows])),
        'reason_empty_count': sum(not r['reason'] for r in rows),
        'issued_literal_count': len(literal_all),
        'literal_truth_rate': float(np.mean(literal_all)) if literal_all else None,
        'all_literals_true_rate': float(np.mean([r['all_literals_true'] for r in rows])),
        'all_literals_true_rate_parts_visible': float(np.mean(
            [r['all_literals_true'] for r in rows if r['reason'] and all(r['literal_parts_visible'])])) if any(
            r['reason'] and all(r['literal_parts_visible']) for r in rows) else None,
        'literal_certainty1_count': sum(c == 1 for r in rows for c in r['literal_certainty']),
        'reader_val_selection': {'grid': table, 'selected': {'max_depth': depth, 'min_samples_leaf': leaf,
                                                              'val_accuracy': val_acc}},
        'crop_contract_check': check,
        'scope': 'crop readers from scratch are the known bottleneck; no grounding, '
                 'calibration, or causal claim; reason truth is measured against CUB '
                 'attribute annotations on the official test split only',
    }
    per_attr = []
    for j, (part, attr, value) in enumerate(CANONICAL):
        y_true, c = present[:, j], certainty[:, j]
        per_attr.append({
            'name': names[j], 'cub_attribute_id': attr_ids[j][0], 'cub_name': attr_ids[j][1],
            'test_accuracy': float((bits_te[:, j] == y_true).mean()),
            'test_accuracy_certainty_ge2': float((bits_te[:, j] == y_true)[c >= 2].mean()),
            'positive_rate': float(y_true.mean()), 'n_positive': int(y_true.sum()),
        })

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    head.save(out / 'head.json')
    torch.save({'state_dict': reader.state_dict(), 'crop_size': args.crop_size,
                'concept_names': names}, out / 'reader.pt')
    (out / 'explanations.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    (out / 'per_attribute.json').write_text(json.dumps(per_attr, indent=2) + '\n')
    versions = {n: importlib.metadata.version(n) for n in ['torch', 'numpy', 'pyxai', 'scikit-learn', 'pillow']}
    provenance = {
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'versions': versions,
        'vocabulary': [{'name': n, 'cub_attribute_id': a, 'cub_name': m} for n, (a, m) in zip(names, attr_ids)],
        'vocabulary_note': 'canonical 28 from grounded_vision cub.py; tail:tail_color=black bound to '
                           f'{TAIL_COLOR_OVERRIDE} (documented substitution; file has no has_tail_color::*)',
        'data_sha256': {f: sha256(Path(args.cub_dir) / f) for f in
                        ['images.txt', 'image_class_labels.txt', 'train_test_split.txt',
                         'parts/part_locs.txt', 'attributes.txt', 'attributes/image_attribute_labels.txt']},
        'template_boxes': TEMPLATE,
        'training_mask_rule': f'certainty >= {args.min_certainty} AND part point visible',
        'reader_deployment_contract': 'score_j = sigmoid(networks[j](resize48(crop(image, box_j))/255)); '
                                      'verified equal to CropPredicateReader.forward on the spot-check sample',
        'train_history': history,
    }
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({k: v for k, v in metrics.items() if k not in ('reader_val_selection',)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
