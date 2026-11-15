#!/usr/bin/env python3
"""Strict premise support across K (convention-free scoring).

Convention: negated literals on invisible parts count supported.
Strict: a reason counts only if every literal sits on a VISIBLE part
and is supported (asserted literals on invisible parts are refuted
under both, so only unresolved negations differ).
Sources: cbm_cem_audit.json (K=4/10), task1_cbcem_K20_50.json (K=50),
ablation_depth_K20_perinstance.json (K=20, visibility included),
cbm_cem_audit_K200.json (K=200) + CUB part visibility by sid.
Output: results/strict_support_cub.json (refuses to overwrite).
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from grounded_vision.data.cub import CUBDataset  # noqa: E402

DATA = "/Users/tanmoy/research/data/CUB_200_2011/CUB_200_2011"
OUT = os.path.join(REPO, "results", "strict_support_cub.json")


def vis_by_sid():
    ds = CUBDataset(data_dir=DATA, split="test", use_mock_if_missing=False)
    return {ds.meta(i)[0]: ds.meta(i)[2] for i in range(len(ds))}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING to overwrite {OUT}")
    vis = vis_by_sid()

    def rate(records):
        n = strict = 0
        for r in records:
            lits = r["literals"]
            if not lits:
                continue
            n += 1
            if r["all_supported"] and all(
                    vis.get(l["f"].split(":")[0], 0.0) > 0.5
                    if "f" in l else l.get("part_visible", True)
                    for l in lits):
                strict += 1
        return n, strict / n

    out = {"protocol": "strict: all literals on visible parts AND supported; "
                       "convention values from the same dumps",
           "blocks": {}}
    jobs = {
        "4": json.load(open(f"{REPO}/results/cbm_cem_audit.json")),
        "10": json.load(open(f"{REPO}/results/cbm_cem_audit.json")),
    }
    for K, d in jobs.items():
        b = next(x for x in d["blocks"] if x["setting"] == "T_tree_head" and x["K"] == int(K))
        n, p = rate(b["per_instance"])
        out["blocks"][f"K{K}"] = {"n": n, "strict": round(p, 4),
                                  "convention": round(b["audit"]["p_all_supported"], 4)}
        print(f"K={K}: n={n} strict={p:.3f} convention={b['audit']['p_all_supported']:.3f}")
    pi = json.load(open(f"{REPO}/results/ablation_depth_K20_perinstance.json"))["instances"]
    n = strict = 0
    for r in pi:
        cell = r["by_depth"]["12"]
        if not cell["literals"]:
            continue
        n += 1
        if cell["all_supported"] and all(l["part_visible"] for l in cell["literals"]):
            strict += 1
    out["blocks"]["K20"] = {"n": n, "strict": round(strict / n, 4), "convention": 0.0699}
    print(f"K=20: n={n} strict={strict/n:.3f} convention=0.070")
    for K, src in (("50", "task1_cbcem_K20_50.json"), ("200", "cbm_cem_audit_K200.json")):
        d = json.load(open(f"{REPO}/results/{src}"))
        b = next(x for x in d["blocks"] if x["setting"] == "T_tree_head" and x["K"] == int(K))
        n, p = rate(b["per_instance"])
        conv = b["audit"]["p_all_supported"]
        out["blocks"][f"K{K}"] = {"n": n, "strict": round(p, 4), "convention": round(conv, 4)}
        print(f"K={K}: n={n} strict={p:.3f} convention={conv:.3f}")
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
