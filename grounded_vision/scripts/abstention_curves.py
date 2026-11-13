#!/usr/bin/env python3
"""Abstention curves: literal-reliability filtering vs calibrated/raw reader
confidence vs random, on joint premise support (review Q1 + Q2, 2026-11-11).

Joins the X2 per-instance dump (min calibrated / raw confidence over the ORD
reason's literals; val-isotonic calibrator for the calibrated arm) with the
CBM-audit per-instance dump (per-literal annotation support) by sid.

Scores per test instance (higher = keep):
  A_ir : min over reason literals of in-reason marginal annotation support
         (test-estimated, tree-conditioned -- oracle-flavoured upper bound)
  A_cq : min over reason literals of per-concept reader accuracy
         (test-estimated, polarity-blind -- coarse secondary)
  B    : min calibrated reader confidence over reason literals
         (isotonic fit on the val carve-out -- deployment-honest)
  C    : min raw reader confidence over reason literals
  D    : random

Outcome: all reason literals annotation-supported.
Reported: joint support at fixed coverages with cluster-bootstrap CIs
(instances resampled; D resamples scores), and the exact n / n_correct at the
cited 13.6% operating point (D/K=10, t=0.5).

Inputs : results/x2_uncertainty_reasons.json, results/cbm_cem_audit.json
Output : results/abstention_curves.json
"""
import argparse
import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_X2 = os.path.join(ROOT, "results", "x2_uncertainty_reasons.json")
DEFAULT_AUD = os.path.join(ROOT, "results", "cbm_cem_audit.json")
DEFAULT_OUT = os.path.join(ROOT, "results", "abstention_curves.json")
N_BOOT = 10000
RNG = np.random.default_rng(42)
COVERAGES = [0.05, 0.10, 0.14, 0.20, 0.30, 0.50]


def parse_lit(s):
    return s[:-1], 1 if s.endswith("+") else 0


def analyse(xb, ab, cq):
    audit = {r["sid"]: r for r in ab["per_instance"]}
    rows = []
    mismatch = 0
    for rec in xb["per_instance"]:
        ar = audit.get(rec["sid"])
        if ar is None:
            continue
        # reason-identity sanity check between the two dumps
        x2_set = sorted(parse_lit(s) for s in rec.get("lits_ord", []))
        aud_set = sorted((l["feature"], l["bit"]) for l in ar["literals"])
        if x2_set != aud_set:
            mismatch += 1
            continue
        if not ar["literals"]:
            continue
        # in-reason marginal support for this block (test-estimated)
        rows.append({
            "sid": rec["sid"],
            "y": float(all(l["supported"] for l in ar["literals"])),
            "B": rec.get("min_conf_ord_cal"),
            "C": rec.get("min_conf_ord_raw"),
            "lits": [(l["feature"], l["bit"]) for l in ar["literals"]],
        })
    if mismatch:
        print(f"  WARNING: {mismatch} sid reason mismatches skipped")
    # A_ir: per-literal marginal annotation support from this block's occurrences
    occ = {}
    for rec in xb["per_instance"]:
        ar = audit.get(rec["sid"])
        if ar is None or not ar["literals"]:
            continue
        for l in ar["literals"]:
            occ.setdefault((l["feature"], l["bit"]), []).append(bool(l["supported"]))
    p_lit = {k: float(np.mean(v)) for k, v in occ.items()}

    y = np.array([r["y"] for r in rows])
    scores = {
        "A_ir": np.array([min(p_lit[lit] for lit in r["lits"]) for r in rows]),
        "A_cq": np.array([min(cq[f"{f}"]["accuracy"] if f in cq else 0.5
                              for f, b in r["lits"]) for r in rows]),
        "B": np.array([r["B"] for r in rows]),
        "C": np.array([r["C"] for r in rows]),
    }
    n = len(rows)

    def at_cov(y_, s_, c):
        k = max(1, int(round(c * len(y_))))
        idx = np.argsort(-s_)[:k]
        return y_[idx].mean(), k

    out = {"setting": xb["setting"], "K": xb["K"], "n": n,
           "raw_joint": float(y.mean()), "n_all_supported": int(y.sum()),
           "at_coverage": {}}
    for c in COVERAGES:
        cell = {}
        for name, s in scores.items():
            v, k = at_cov(y, s, c)
            boots = []
            for _ in range(N_BOOT):
                bi = RNG.integers(0, n, n)
                bv, _ = at_cov(y[bi], s[bi], c)
                boots.append(bv)
            cell[name] = {"joint": float(v), "k": k, "n_correct": float(v * k),
                          "boot_pct": [float(np.percentile(boots, 2.5)),
                                       float(np.percentile(boots, 97.5))]}
        # D: random order, bootstrap over instance resamples (score irrelevant)
        boots = []
        for _ in range(N_BOOT):
            bi = RNG.integers(0, n, n)
            k = max(1, int(round(c * n)))
            boots.append(y[bi][:k].mean())
        cell["D_random"] = {"joint": float(y.mean()), "k": max(1, int(round(c * n))),
                            "boot_pct": [float(np.percentile(boots, 2.5)),
                                         float(np.percentile(boots, 97.5))]}
        out["at_coverage"][f"{c:.3f}"] = cell
    return out


def main():
    global N_BOOT, COVERAGES
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--x2", default=DEFAULT_X2, help="X2 reasons JSON (per_instance)")
    ap.add_argument("--audit", default=DEFAULT_AUD, help="CBM audit JSON (per_instance)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output JSON (refuses to overwrite)")
    ap.add_argument("--coverages", default=",".join(str(c) for c in COVERAGES))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--force", action="store_true", help="allow overwriting --out")
    args = ap.parse_args()
    COVERAGES = [float(c) for c in args.coverages.split(",")]
    N_BOOT = args.n_boot
    if os.path.exists(args.out) and not args.force:
        raise SystemExit(f"REFUSING to overwrite existing {args.out} (pass --force)")
    x2 = json.load(open(args.x2))
    aud = json.load(open(args.audit))
    results = []
    for xb in x2["blocks"]:
        if xb.get("seed") != 42:
            continue
        ab = next((b for b in aud["blocks"]
                   if b["setting"] == xb["setting"] and b["K"] == xb["K"]
                   and b.get("reader", "linear") == "linear"), None)
        if ab is None:
            continue
        print(f"block {xb['setting']}/K{xb['K']}")
        results.append(analyse(xb, ab, aud.get("concept_quality", {})))
    payload = {
        "protocol": {
            "question": "Does abstention by literal reliability (A) beat calibrated (B) / raw (C) reader confidence and random (D) at matched coverage on joint premise support?",
            "joins": f"{os.path.basename(args.x2)} per_instance (min_conf_ord_cal/raw, lits_ord) x {os.path.basename(args.audit)} per_instance (annotation support) by sid, seed 42, reason-identity checked",
            "scores": {"A_ir": "min in-reason marginal annotation support (test-estimated, tree-conditioned; oracle-flavoured)",
                       "A_cq": "min per-concept reader accuracy (test-estimated, polarity-blind)",
                       "B": "min val-isotonic calibrated confidence over reason literals (deployment-honest)",
                       "C": "min raw confidence",
                       "D_random": "random order (expect the raw rate at every coverage)"},
            "bootstrap": f"cluster over instances, {N_BOOT}",
            "coverages": COVERAGES,
        },
        "blocks": results,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    for r in results:
        print(f"\n{r['setting']}/K{r['K']}: n={r['n']}, raw joint support "
              f"{r['raw_joint']:.3f} ({r['n_all_supported']}/{r['n']})")
        print(f"{'cov':>6} {'A_ir':>18} {'A_cq':>18} {'B_cal':>18} {'C_raw':>18} {'D_rand':>18}")
        for c, cell in r["at_coverage"].items():
            def fmt(name):
                v = cell[name]
                return f"{v['joint']:.3f} [{v['boot_pct'][0]:.2f},{v['boot_pct'][1]:.2f}] k={v['k']}"
            print(f"{c:>6} {fmt('A_ir'):>18} {fmt('A_cq'):>18} {fmt('B'):>18} "
                  f"{fmt('C'):>18} {fmt('D_random'):>18}")


if __name__ == "__main__":
    main()
