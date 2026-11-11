#!/usr/bin/env python3
"""Q2 (independence gap): is P(all premises supported) just compounding of
per-literal marginals, or are premise errors correlated?

Reads the committed per-instance audit records (reason literals + annotation
support per literal) and compares, per block (setting, K):
  observed_i  = all literals of instance i's reason annotation-supported
  predicted_i = prod_l p_l, with p_l the marginal support rate of literal l
                over its occurrences in reasons across the test set
Cluster bootstrap over instances (marginals re-estimated per resample).
Also reports the predict-the-prior baselines for present/absent clause
precision (Q4): P(gt_bit==1) among present-literal occurrences, P(gt_bit==0)
among absent-literal occurrences.

Inputs : results/cbm_cem_audit.json (K=4/10, T and D), results/cbm_cem_audit_K200.json
Output : results/independence_gap_cub.json
"""
import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_FILES = [os.path.join(ROOT, "results", "cbm_cem_audit.json"),
            os.path.join(ROOT, "results", "cbm_cem_audit_K200.json")]
OUT = os.path.join(ROOT, "results", "independence_gap_cub.json")
N_BOOT = 10000
RNG = np.random.default_rng(42)
LEN_BINS = [(1, 3), (4, 6), (7, 10), (11, 10**9)]


def load_blocks():
    blocks = []
    for f in IN_FILES:
        d = json.load(open(f))
        for b in d["blocks"]:
            blocks.append(b)
    return blocks


def block_arrays(block):
    """Return per-instance literal-support boolean rows and reason lengths."""
    rows, lens = [], []
    for rec in block["per_instance"]:
        lits = rec["literals"]
        rows.append([bool(l["supported"]) for l in lits])
        lens.append(len(lits))
    return rows, np.array(lens)


def analyse_block(block):
    pi = block["per_instance"]
    ids_per_inst, sup_per_inst, gt_bits, bits = [], [], [], []
    obs = []
    for rec in pi:
        lits = rec["literals"]
        ids = [f"{l['feature']}|{l['bit']}" for l in lits]
        sups = [bool(l["supported"]) for l in lits]
        ids_per_inst.append(ids)
        sup_per_inst.append(sups)
        obs.append(all(sups))
        for l in lits:
            bits.append(l["bit"])
            gt_bits.append(l["gt_bit"])
    n = len(pi)
    obs = np.array(obs, dtype=float)

    # marginal support rate per literal id over its occurrences
    occ = {}
    for ids, sups in zip(ids_per_inst, sup_per_inst):
        for i, s in zip(ids, sups):
            occ.setdefault(i, []).append(s)
    p_lit = {i: float(np.mean(v)) for i, v in occ.items()}

    pred = np.array([np.prod([p_lit[i] for i in ids]) if ids else 1.0
                     for ids in ids_per_inst], dtype=float)

    lens = np.array([len(i) for i in ids_per_inst])

    # cluster bootstrap: resample instances, re-estimate marginals, recompute both means
    boots = np.empty((N_BOOT, 2))
    for b in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        bo = obs[idx]
        acc = {}
        for k in idx:
            for i, s in zip(ids_per_inst[k], sup_per_inst[k]):
                acc.setdefault(i, []).append(s)
        pb = {i: float(np.mean(v)) for i, v in acc.items()}
        bp = np.array([np.prod([pb[i] for i in ids_per_inst[k]]) if ids_per_inst[k] else 1.0
                       for k in idx])
        boots[b] = (bo.mean(), bp.mean())

    ratio = obs.mean() / pred.mean()
    ratio_boot = boots[:, 0] / boots[:, 1]

    # Q4 baselines: predict-the-prior over literal occurrences. The meaningful
    # comparison is against the concept's occurrence base rate (P(gt=1) over ALL
    # occurrences of the concept in reasons, regardless of asserted polarity),
    # not P(gt=1 | asserted present), which IS the present-precision itself.
    bits_a = np.array(bits)
    gt_a = np.array(gt_bits)
    present = bits_a == 1
    absent = bits_a == 0
    feat_a = np.array([i.rsplit("|", 1)[0] for i in
                       [l for ids in ids_per_inst for l in ids]])
    feat_pos_rate = {}
    for f_, g_ in zip(feat_a, gt_a):
        feat_pos_rate.setdefault(f_, []).append(g_)
    feat_pos_rate = {f_: float(np.mean(v)) for f_, v in feat_pos_rate.items()}
    pres_base = np.mean([feat_pos_rate[ids.rsplit("|", 1)[0]]
                         for ids, b_ in zip([l for ids in ids_per_inst for l in ids], bits_a)
                         if b_ == 1])
    abs_base = np.mean([1.0 - feat_pos_rate[ids.rsplit("|", 1)[0]]
                        for ids, b_ in zip([l for ids in ids_per_inst for l in ids], bits_a)
                        if b_ == 0])

    # per-literal marginal histogram: is support per-concept deterministic?
    p_vals = np.array(list(p_lit.values()))
    hist = {"n_distinct_literals": int(len(p_vals)),
            "at_0": float((p_vals == 0).mean()), "at_1": float((p_vals == 1).mean()),
            "strictly_between": float(((p_vals > 0) & (p_vals < 1)).mean())}

    # Q8 filter: restrict to instances whose reason uses only literals with
    # marginal support >= t (oracle-flavoured: marginals from the same test set;
    # upper bound on what a holdout-calibrated exclusion filter could achieve)
    filt = {}
    for t in [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]:
        keep = np.array([bool(ids) and min(p_lit[i] for i in ids) >= t
                         for ids in ids_per_inst])
        cov = float(keep.mean())
        filt[str(t)] = {
            "coverage": cov,
            "p_all_given_covered": float(obs[keep].mean()) if keep.any() else None,
            "n_covered": int(keep.sum()),
        }

    # cross-fitted versions: marginals estimated on one half of the instances,
    # evaluated out-of-sample on the other (unseen literals fall back to the
    # pooled half marginal). Removes the oracle flavour of both ratio and filter.
    perm = RNG.permutation(n)
    halves = [perm[: n // 2], perm[n // 2:]]
    pred_x = np.empty(n)
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
    covered_x = {t: np.zeros(n, dtype=bool) for t in thresholds}
    for est, ev in ((halves[0], halves[1]), (halves[1], halves[0])):
        acc = {}
        for k in est:
            for i, s in zip(ids_per_inst[k], sup_per_inst[k]):
                acc.setdefault(i, []).append(s)
        p_est = {i: float(np.mean(v)) for i, v in acc.items()}
        pool = float(np.mean([s for v in acc.values() for s in v]))
        for k in ev:
            ps = [p_est.get(i, pool) for i in ids_per_inst[k]]
            pred_x[k] = np.prod(ps) if ps else 1.0
            for t in thresholds:
                covered_x[t][k] = bool(ps) and min(ps) >= t
    filt_x = {}
    for t in thresholds:
        cv = covered_x[t]
        filt_x[str(t)] = {
            "coverage": float(cv.mean()),
            "p_all_given_covered": float(obs[cv].mean()) if cv.any() else None,
            "n_covered": int(cv.sum()),
        }

    out = {
        "setting": block["setting"], "K": block["K"],
        "reader": block.get("reader", "linear"),
        "n_instances": n,
        "n_literal_occurrences": int(len(bits_a)),
        "observed_joint": float(obs.mean()),
        "independence_predicted_joint": float(pred.mean()),
        "ratio_obs_over_pred": float(ratio),
        "ratio_boot_median": float(np.median(ratio_boot)),
        "ratio_boot_pct": [float(np.percentile(ratio_boot, 2.5)),
                           float(np.percentile(ratio_boot, 97.5))],
        "obs_boot_pct": [float(np.percentile(boots[:, 0], 2.5)),
                         float(np.percentile(boots[:, 0], 97.5))],
        "pred_boot_pct": [float(np.percentile(boots[:, 1], 2.5)),
                          float(np.percentile(boots[:, 1], 97.5))],
        "mean_reason_len": float(lens.mean()),
        "by_length": {},
        "literal_marginal_hist": hist,
        "ratio_crossfit": float(obs.mean() / pred_x.mean()) if pred_x.mean() > 0 else None,
        "q8_literal_filter": filt,
        "q8_literal_filter_crossfit": filt_x,
        "q4_baselines": {
            "n_present_occurrences": int(present.sum()),
            "n_absent_occurrences": int(absent.sum()),
            "present_base_rate": float(pres_base),        # P(gt=1) over occurrences of those concepts
            "absent_base_rate": float(abs_base),          # P(gt=0) over occurrences of those concepts
            "present_precision_observed": block["audit"]["present_clause_precision"],
            "absent_precision_observed": block["audit"]["absent_clause_precision"],
        },
    }
    for lo, hi in LEN_BINS:
        m = (lens >= lo) & (lens <= hi)
        if m.sum() == 0:
            continue
        out["by_length"][f"L{lo}-{min(hi, int(lens.max()))}"] = {
            "n": int(m.sum()),
            "observed": float(obs[m].mean()),
            "predicted": float(pred[m].mean()),
            "ratio": float(obs[m].mean() / pred[m].mean()) if pred[m].mean() > 0 else None,
        }
    return out


def main():
    blocks = load_blocks()
    results = [analyse_block(b) for b in blocks]
    payload = {
        "protocol": {
            "question": "Is P(all premises supported) explained by independent compounding of per-literal marginals (Q2), and do present/absent clause precisions beat predict-the-prior baselines (Q4)?",
            "observed": "fraction of instances whose reason has all literals annotation-supported (committed per_instance records)",
            "predicted": "prod of per-literal marginal support rates (estimated over literal occurrences in reasons, same test set)",
            "bootstrap": f"cluster bootstrap over instances, {N_BOOT} resamples, marginals re-estimated per resample",
            "inputs": [os.path.relpath(f, ROOT) for f in IN_FILES],
            "note": "ratio > 1 means premise errors cluster (joint support better than independence predicts); ratio < 1 means errors anti-correlate",
        },
        "blocks": results,
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=1)
    # console table
    print(f"{'block':<22}{'n':>5}{'len':>6}{'obs':>8}{'pred':>8}{'ratio':>8}  ratio 95% CI")
    for r in results:
        ci = r["ratio_boot_pct"]
        print(f"{r['setting']}/K{r['K']}/{r['reader']:<6}{r['n_instances']:>5}"
              f"{r['mean_reason_len']:>6.2f}{r['observed_joint']:>8.4f}"
              f"{r['independence_predicted_joint']:>8.4f}{r['ratio_obs_over_pred']:>8.3f}"
              f"  [{ci[0]:.2f}, {ci[1]:.2f}]")
    print()
    print("by length (obs vs pred):")
    for r in results:
        parts = [f"{k}:{v['observed']:.3f}/{v['predicted']:.3f}(n={v['n']})"
                 for k, v in r["by_length"].items()]
        print(f"  {r['setting']}/K{r['K']}: " + "  ".join(parts))
    print()
    print("Q4 baselines (observed precision vs concept occurrence base rate):")
    for r in results:
        q = r["q4_baselines"]
        print(f"  {r['setting']}/K{r['K']}: present {q['present_precision_observed']:.3f} vs base "
              f"{q['present_base_rate']:.3f} (n={q['n_present_occurrences']}) | "
              f"absent {q['absent_precision_observed']:.3f} vs base {q['absent_base_rate']:.3f} "
              f"(n={q['n_absent_occurrences']})")
    print()
    print("literal marginal histogram (distinct literals used by reasons):")
    for r in results:
        h = r["literal_marginal_hist"]
        print(f"  {r['setting']}/K{r['K']}: n={h['n_distinct_literals']}, p=0: {h['at_0']:.2f}, "
              f"p=1: {h['at_1']:.2f}, strictly between: {h['strictly_between']:.2f}")
    print()
    print("Q8 filter (keep instances whose reason uses only literals with marginal support >= t):")
    for r in results:
        parts = [f"t={t}: cov={v['coverage']:.3f}, P(all|cov)={v['p_all_given_covered']}"
                 for t, v in r["q8_literal_filter"].items()]
        print(f"  {r['setting']}/K{r['K']}: " + "; ".join(parts))
    print()
    print("Q8 filter, cross-fitted marginals (out-of-sample):")
    for r in results:
        parts = [f"t={t}: cov={v['coverage']:.3f}, P(all|cov)={v['p_all_given_covered']}"
                 for t, v in r["q8_literal_filter_crossfit"].items()]
        print(f"  {r['setting']}/K{r['K']} (xfit ratio {r['ratio_crossfit']:.2f}): " + "; ".join(parts))


if __name__ == "__main__":
    main()
