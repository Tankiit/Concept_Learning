"""Faithful template NLX for certified, spatially grounded activation reasons.

Superseded for new runs by the NLX built into run_activation_track_traintest.py
(which also names visible parts inside the certified tile); this script covers
the original per-tile wording over a run directory of your choice.
"""
import argparse
import json
from pathlib import Path

DEFAULT_RUN = Path(
    "/Users/cril/tanmoy/research/data/grounded_vision_runs/cub_bound_real_full"
)


def phrase(g):
    c = g["condition"]
    t = g["tile"]
    return (
        f"DINO activation channel {g['activation_channel']} in spatial tile "
        f"(row {t['row'] + 1}, column {t['col'] + 1}) is {c['operator']} "
        f"{c['threshold']:.3f} (observed {c['observed_value']:.3f})"
    )


def build_nlx(record):
    """Attaches a template NLX to one explanation record.

    Records without a (non-empty) 'grounding' list get explicit wording
    instead of a crash or a dangling 'because .'. No faithfulness or
    abstention claims are attached here — those are measured by the
    auditor, not asserted by the generator.
    """
    grounding = record.get("grounding") or []
    clauses = [phrase(g) for g in grounding]
    if clauses:
        evidence = "; and ".join(clauses)
        text = (
            f"The activation tree predicts class {record['prediction']} "
            f"because {evidence}."
        )
    else:
        text = (
            f"The activation tree predicts class {record['prediction']} "
            f"(no grounded conditions recorded)."
        )
    record["nlx"] = {"text": text, "style": "activation_grounded_template"}
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN,
                    help="run directory holding the source JSONL")
    ap.add_argument("--source", default="activation_tree_grounded_explanations.jsonl")
    ap.add_argument("--out", default="activation_grounded_nlx.jsonl")
    args = ap.parse_args()

    source = args.run_dir / args.source
    out = args.run_dir / args.out
    n_written, n_skipped = 0, 0
    with source.open() as fin, out.open("w") as fout:
        for line in fin:
            r = json.loads(line)
            if "grounding" not in r:
                n_skipped += 1
                continue
            fout.write(json.dumps(build_nlx(r)) + "\n")
            n_written += 1
    print(f"wrote {n_written} records to {out}; skipped {n_skipped} without grounding")


if __name__ == "__main__":
    main()
