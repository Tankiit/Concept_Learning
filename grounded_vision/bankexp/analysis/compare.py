"""
compare.py — PyXAI vs other explainers across feature banks.
=====================================================================
analysis/compare.py (stdlib only).  Assembles one table per feature
bank (semantic predicates vs cls-token PCA vs mean-patch PCA ...),
one row per bank, one column per explainer/metric:

    PyXAI tree AXp   mean size, verified, teacher-sufficient (exact)
    erasure (conaxp) mean size (random + contribution orders), exact
    MHS set cover    minimum cover size (A exact / B random / C hybrid)
    context          teacher acc, tree acc, agreement

Usage
-----
    python analysis/compare.py --banks ""=intermediate_results \
        _cls=intermediate_results/bank_cls \
        _meanpatch=intermediate_results/bank_meanpatch --classes 4

Output: results/comparison_banks_K{K}.md + .json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"


def load_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.is_file() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", nargs="+", required=True,
                    help="tag=bank_dir pairs (tag '' for the predicate bank)")
    ap.add_argument("--classes", type=int, default=4)
    args = ap.parse_args()
    K = args.classes

    rows = []
    for spec in args.banks:
        tag, _, bdir = spec.partition("=")
        bank = Path(bdir)
        r = RESULTS
        teacher = load_json(bank / f"teacher_K{K}_metrics.json")
        tree = load_json(bank / f"tree_K{K}_metrics.json")
        m1 = load_json(r / f"metric1_margin_sufficiency_K{K}{tag}.summary.json")
        m2 = load_json(r / f"metric2_coverage_K{K}{tag}.summary.json")
        m3 = load_json(r / f"metric3_robustness_K{K}{tag}.summary.json")
        m4r = load_json(r / f"metric4_erasure_conaxp_K{K}{tag}_random.summary.json")
        m4c = load_json(
            r / f"metric4_erasure_conaxp_K{K}{tag}_contribution.summary.json")
        m6 = load_json(r / f"metric6_max_coverage_K{K}{tag}.summary.json")
        meta = load_json(bank / "meta.json").get("protocol", {})

        n = mean_size = verified = sets_match = None
        reasons_p = r / f"reasons_K{K}{tag}.jsonl"
        if reasons_p.is_file():
            recs = [json.loads(l) for l in reasons_p.read_text().splitlines()
                    if l.strip()]
            ok = [x for x in recs if x.get("status") == "ok"]
            n = len(ok)
            mean_size = round(sum(x["native_size"] for x in ok) / max(1, len(ok)), 3)
            verified = round(sum(bool(x.get("native_verified")) for x in ok)
                             / max(1, len(ok)), 4)
            sm = [x for x in ok if "sets_match" in x]
            sets_match = (round(sum(x["sets_match"] for x in sm) / len(sm), 4)
                          if sm else None)

        rows.append({
            "bank": tag or "predicates",
            "features": meta.get("n_features"),
            "semantic": meta.get("semantic", True),
            "n_reasons": n,
            "teacher_acc": teacher.get("test_acc"),
            "tree_acc": tree.get("test_acc"),
            "tree_teacher_agreement": tree.get("teacher_agreement"),
            "pyxai_axp_size": mean_size,
            "pyxai_axp_verified": verified,
            "pyxai_vs_native_sets_match": sets_match,
            "m1_frac_teacher_sufficient": m1.get("frac_sufficient_exact_teacher"),
            "m2_inreason_agreement": m2.get("mean_teacher_agreement"),
            "m3_tree_necessity": m3.get("frac_tree_necessity_min1"),
            "m3_teacher_necessity": m3.get("frac_teacher_necessity_min1"),
            "erasure_size_random": m4r.get("mean_size"),
            "erasure_size_contribution": m4c.get("mean_size"),
            "mhs_cover_exact": m6.get("A_exact"),
            "mhs_cover_random": m6.get("B_random"),
            "mhs_cover_hybrid": m6.get("C_hybrid"),
            "mhs_cover_greedy": m6.get("greedy_freq"),
        })

    cols = list(rows[0].keys()) if rows else []
    lines = ["# PyXAI vs erasure vs MHS across feature banks "
             f"(K={K}, official split)", "",
             "| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(
            str(v) if (v := row.get(c)) is not None else "—"
            for c in cols) + " |")
    md = "\n".join(lines) + "\n"
    out_md = RESULTS / f"comparison_banks_K{K}.md"
    out_md.write_text(md)
    (RESULTS / f"comparison_banks_K{K}.json").write_text(
        json.dumps(rows, indent=2))
    print(md)
    print(f"saved: {out_md}")


if __name__ == "__main__":
    main()
