# BankExp — Concept Bank → Teacher → Student → Certificates

Pipeline for measuring what a formally-certified concept explanation is
actually a certificate OF: sufficiency for the deployed Boolean student,
exact (box-LP) sufficiency for the linear teacher, measured transfer to
the teacher on-distribution, and the size price of teacher-scoped
certificates.  Structure follows
[CSU-TrustLab/behavior-explainer](https://github.com/CSU-TrustLab/behavior-explainer).

## Repository Structure

```
bankexp/
├── src/                          # Core pipeline (stages run in order)
│   ├── bank.py                   # DINO tokens -> region predicates -> binary bank
│   ├── altbank.py                # Alternative banks: cls-token / mean-patch PCA (non-semantic)
│   ├── teacher.py                # Design B teacher: linear CBM head over the bank
│   ├── distill.py                # CART student on teacher predictions -> tree_K{K}.json
│   ├── explain.py                # PyXAI reasons + independent numpy verifier (--native-only supported)
│   ├── conaxp.py                 # Erasure-based baseline on the bank (greedy, exact oracle)
│   ├── ale.py                    # Thin wrapper calling beyond_l2's run_formal_exp.py
│   └── run_experiment.py         # Parametrized runner (--bank-dir / --tag for bank comparison)
├── analysis/                     # Metric computation (numpy only, no model loading)
│   ├── margin_sufficiency.py     # Metric 1 — exact box-LP sufficiency vs the linear teacher
│   ├── coverage.py               # Metric 2 — tree-vs-teacher agreement inside a reason
│   ├── robustness.py             # Metric 3 — necessity + predicate-flip survival
│   ├── max_coverage.py           # Metric 6 — MHS set cover over per-image reasons (A/B/C solvers)
│   ├── compare.py                # Cross-bank comparison table (PyXAI vs erasure vs MHS)
│   └── summarize.py              # Aggregate all results/*.csv -> summary.md/json
├── utils/                        # Shared utilities
│   ├── tree_ops.py               # numpy routing + exact sufficiency DFS + minimal reasons
│   ├── box_lp.py                 # Exact box-LP sufficiency for a linear head
│   └── mhs.py                    # Minimal hitting set solvers (port of behavior-explainer's utils/mhs.py)
├── datasets/                     # Data (not tracked — see per-folder READMEs)
│   └── CUB/README.md
├── vocabs/                       # Concept vocabulary (generated from the bank meta)
├── intermediate_results/         # Bank npz, teachers, trees (not tracked)
├── results/                      # Metric CSVs, reasons, summaries
├── notebooks/                    # (reserved for the reviewer notebook)
└── tests/                        # Unit tests (19 passing; tree_ops, box_lp, mhs ported from
                                  #  behavior-explainer, run_experiment-pattern integration)
```

## Environments

Two interpreters (mirrors the environment split; see also
`src/run_experiment.py --conda-python/--pyxai-python`):

| env | interpreter | used by | needs |
|---|---|---|---|
| PyC | conda `torch-multimodal` | `src/bank.py`, `src/teacher.py` | torch, sklearn, `grounded_vision` src |
| PyXAI | `ltn_pyxai_starter/.venv` | `src/distill.py`, `src/explain.py` | pyxai 2.0.1, sklearn, numpy |
| any | either | `src/conaxp.py`, `src/ale.py`, `analysis/*`, `tests/*` | numpy |

Known pyxai 2.0.1 defects honoured here: `minimal_sufficient_reason`
crashes on Linux and **deadlocks on arm64 macOS** for trees that use a
subset of declared features — never called.  Tree-model calls
(`sufficient_reason`, `is_sufficient_reason`, predictions) run fine
everywhere.

## Reproducing Results

### 1. Fit the concept bank (once, K-agnostic, ~4 min)

```bash
python src/bank.py \
  --cache-dir /Users/tanmoy/research/data/grounded_vision_runs/cub_local_et \
  --data-dir  /Users/tanmoy/research/data/CUB_200_2011/CUB_200_2011
```

Frozen protocol (identical to the committed audit runs): cached DINOv3
feature maps, oracle GT part boxes (labelled limitation), logistic
readers on pooled region features, val-carved isotonic calibration.
Subsequent runs load the cached bank; pass `--rebuild` to refit.

### 2. Run the full experiment for one K

```bash
python src/run_experiment.py --classes 4 \
  --cache-dir <dino cache> --data-dir <CUB root>
```

Runs teacher -> distill -> explain -> metrics 1-4 -> summary, invoking
the right interpreter per stage.  All artifacts land in
`intermediate_results/` (models) and `results/` (CSVs + reasons).

### 3. Baselines

```bash
python src/conaxp.py --classes 4 --order random        # metric 4
python src/conaxp.py --classes 4 --order contribution
python src/ale.py --check --repo /path/to/beyond_l2    # metric 5 (external)
```

### 4. Tests

```bash
python -m pytest tests/ -q
```

## Results (first run: K=4, official split, seed 42)

| stage | result |
|---|---|
| teacher(bits) test acc | 0.627 (majority 0.254) |
| distilled student | 0.661, teacher agreement 0.627, depth 4 / 9 leaves |
| PyXAI vs independent verifier | 118/118 prediction match, 118/118 mutual sufficiency audits, sizes equal, mean reason 2.89 literals |
| metric 1 (exact, box-LP) | **0/118 tree reasons sufficient for the linear teacher** (worst competing margin +7.09 logits) |
| metric 2 (measured) | in-reason teacher agreement 0.612 |
| metric 3 | tree necessity 1.0 vs teacher necessity 0.33; r=1 survival tree 1.00 / teacher 0.66 |
| metric 4 (conaxp) | teacher-exact erasure reasons 16.94 (random) / 14.5 (contribution) literals |

## PyXAI vs other explainers across feature banks (K=4)

`results/comparison_banks_K4.md` (via `analysis/compare.py`).  Three
banks from the SAME cached DINOv3 features — semantic region predicates
(36), cls-token PCA-32, mean-patch PCA-32 — same protocol, same
explainers:

| bank | teacher acc | PyXAI AXp size | tree reasons teacher-sufficient (exact) | in-reason agreement | erasure size (rand/contrib) | MHS cover |
|---|---|---|---|---|---|---|
| predicates (36, semantic) | 0.627 | 2.89 | **0/118** | 0.612 | 16.94 / 14.5 | 1 |
| cls PCA-32 | 0.805 | 3.00 | **0/118** | 0.596 | 16.38 / 13.77 | 1 |
| meanpatch PCA-32 | 0.737 | 3.00 | **0/118** | 0.720 | 16.36 / 13.68 | 2 |

The qualitative story is FEATURE-VECTOR-INVARIANT: ~3-literal exact tree
certificates that never transfer to the teacher's box (0/118 on every
bank), a ~5x literal price for teacher-scoped certificates, and
degenerate 1-2 concept MHS covers at K=4 (the distilled trees lean on
one dominant split).  PyXAI vs the independent numpy verifier agree on
sizes 118/118 everywhere (sets equal 84-97%; the rest are alternative
same-size minimal reasons).  Tests: `python -m pytest tests/ -q` —
19 passing, including `test_mhs.py` ported from behavior-explainer and
a `test_run_experiment.py` integration test (synthetic bank, pyxai-free
via `explain.py --native-only`).
