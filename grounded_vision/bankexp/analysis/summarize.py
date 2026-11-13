"""
Summarize — shared CSV schema aggregation over results/.
=====================================================================
analysis/summarize.py (numpy/stdlib only; no model loading).  Reads
every results/*.csv, groups by (method, order, K, source), and reports
n, mean size, and the mean of each numeric metric column.  Writes
results/summary.md + results/summary.json.

Schema contract (all metric CSVs):
    REQUIRED: method (str), K (int), sample_id (int)
    size (int) when the row carries one explanation
    everything else: per-method numeric metric columns
"""

from __future__ import annotations
import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

REQUIRED = ["method", "K", "sample_id"]


def is_num(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def summarize(csv_paths):
    groups = OrderedDict()
    for path in csv_paths:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                if any(c not in row for c in REQUIRED):
                    continue
                key = (row["method"], row.get("order", ""),
                       int(row["K"]), Path(path).stem)
                g = groups.setdefault(key, {"n": 0, "size_sum": 0.0,
                                            "size_n": 0})
                g["n"] += 1
                if is_num(row.get("size")) and row["size"] != "":
                    g["size_sum"] += float(row["size"])
                    g["size_n"] += 1
                for col, val in row.items():
                    if col in REQUIRED or col in ("size", "reason", "order",
                                                  "wilson_lo", "wilson_hi"):
                        continue
                    if is_num(val):
                        g.setdefault(col + "_sum", 0.0)
                        g[col + "_sum"] += float(val)
                        g.setdefault(col + "_n", 0)
                        g[col + "_n"] += 1
    table = []
    for (method, order, K, src), g in groups.items():
        entry = OrderedDict([
            ("method", method), ("order", order or None), ("K", K),
            ("source", src), ("n", g["n"]),
            ("mean_size", round(g["size_sum"] / g["size_n"], 4)
             if g["size_n"] else None),
        ])
        for k, v in g.items():
            if k.endswith("_sum"):
                col = k[:-4]
                entry["mean_" + col] = round(v / max(1, g[col + "_n"]), 4)
        table.append(entry)
    return table


def to_markdown(table):
    if not table:
        return "_(no CSV rows found)_\n"
    cols = []
    for entry in table:
        for c in entry:
            if c not in cols:
                cols.append(c)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for entry in table:
        lines.append("| " + " | ".join(
            str(v) if (v := entry.get(c)) is not None else "—"
            for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(RESULTS_DIR))
    ap.add_argument("--csvs", nargs="*", default=None)
    args = ap.parse_args()
    paths = [Path(p) for p in args.csvs] if args.csvs else \
        sorted(Path(args.dir).glob("*.csv"))
    table = summarize(paths)
    md = ("# bankexp summary\n\nSources: "
          + ", ".join(p.name for p in paths) + "\n\n" + to_markdown(table))
    Path(args.dir).mkdir(exist_ok=True, parents=True)
    Path(args.dir).joinpath("summary.md").write_text(md)
    Path(args.dir).joinpath("summary.json").write_text(json.dumps(table, indent=2))
    print(md)
    print(f"saved: {Path(args.dir)/'summary.md'}")


if __name__ == "__main__":
    main()
