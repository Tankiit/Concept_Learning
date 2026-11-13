# Results

All metric CSVs share the schema: `method, K, sample_id` (+ `size` when
a row carries one explanation, + per-metric numeric columns).  Aggregate
view: `summary.md` / `summary.json` (written by `analysis/summarize.py`).

| file | metric | writer |
|---|---|---|
| `reasons_K{K}.jsonl` | PyXAI reasons + mutual verifier audits (per instance) | `src/explain.py` |
| `metric1_margin_sufficiency_K{K}.csv` | 1 — exact box-LP sufficiency of each reason for the linear teacher | `analysis/margin_sufficiency.py` |
| `metric2_coverage_K{K}.csv` | 2 — tree-vs-teacher agreement inside each reason (Wilson CI) | `analysis/coverage.py` |
| `metric3_robustness_K{K}.csv` | 3 — necessity + predicate-flip survival (tree exact, teacher instance-level) | `analysis/robustness.py` |
| `metric4_erasure_conaxp_K{K}_{order}.csv` | 4 — greedy erasure baseline for the teacher (random / contribution orders) | `src/conaxp.py` |
| `metric5_ale_K{K}.csv` | 5 — ALE baseline via beyond_l2 (external repo; see `src/ale.py`) | `src/ale.py` |

Models and banks (teacher npz, tree json, bank npz, meta) live in
`../intermediate_results/` and are not tracked — rebuild with
`src/bank.py` + `src/teacher.py`.
