# CVPR 2027 hand-off — the three runs (assignment brief)

**Issued:** 2026-11-11. Runner-facing assignment as written by the lead. Execution status: all three tasks have ready scripts at [`grounded_vision/scripts/tasks/`](../../grounded_vision/scripts/tasks/) (see that README for exact commands and the environment note); the scientific criteria were pre-registered BEFORE any run in [`CVPR2027_PREREG_TASK_RUNS.md`](CVPR2027_PREREG_TASK_RUNS.md) — read it first. Writing, abstract decisions, and the reconciliation of the two trust results stay with the lead; the runner delivers numbers.

---

## Delivery status 2026-11-14 — the three runs EXECUTED (by the lead, locally)

**All three tasks ran on 2026-11-14 and mechanical-PASSed** (executed by the lead on the local Mac, conda `torch-multimodal`, local flip-free caches, pre-registered seeds/paths; numbers and verdict tables in [`CVPR2027_PREREG_TASK_RUNS.md`](CVPR2027_PREREG_TASK_RUNS.md) § execution results):

- **Task 1** — K=20 **GRACEFUL** (B@0.14 = 0.361 [0.24,0.47], n=72, vs 2×R = 0.140), K=50 **GRACEFUL** (0.143 [0.07,0.23], n=84, vs 0.070) → **pre-registered rule 1: abstract unchanged, body gains the scaling curve**; K=200 GRACEFUL on its face (0.095 [0.04,0.15] vs 0.033) → **rule 3 fired, prescribed bug-check executed and clean** (join exact, zero structural-only reasons, marginals 95% diffuse — the oracle filter dies, calibration still separates), decision reserved to the lead.
- **Task 2** — **SUBSET-SPECIFIC (0/5 draws dominated)**: P(all) per draw 0.267–0.431 (no draw > 50%); failing reasons concentrate on one literal everywhere (shares 0.62–0.94) but via **negated** clauses with **high** marginals (0.90–0.98) — not the pre-registered ≤0.10-marginal profile. Abstract sentence stays; body must state subset-dependence.
- **Task 3** — cov_forcing ≡ cov_ranking and support_kept ≡ support_minimal ≡ support_ranking at every δ, both K; K=4 is a step function (30/118 = the pipit reasons, support 1.000). **The `guarantee_scaling` 0.07–0.10 is the raw-confidence selection event, not support** — the pre-registered "coverage event, not support" branch (E1/E2 rejected, E3 equality).

Artifacts: `results/task1_{cbcem_K20_50, x2_K20_50_200, abstention_K20_50, abstention_K200, verdicts}.json`, `results/figures/task1_abstention_scaling.{png,pdf}`, `results/task2_draw{0–4}_*.json` + `task2_subset_sensitivity.json`, `results/task3_forcing_sweep.json`. Ground rules verified: all files new, per-instance dumps present, every percentage beside its n.

**Two committed-script bugs found and fixed during the runs** (default behavior unchanged bit-for-bit): (1) `run_x2_uncertainty.py` crashed after full compute on any `--classes` without K=4 (verdict block hardcoded a K=4 lookup before the JSON dump — one full X2 run lost to it); (2) `run_cbm_cem_audit.py --class-ids` never remapped labels to 0..K−1 (crash for draws with ids ≥ K, silent wrong-problem training otherwise). Also: `scripts/tasks/` plumbing now runs under conda `torch-multimodal` (the env of every committed number) instead of base anaconda, and the Task 1 driver is resumable across interrupted stages.

### Prior intake, same day — branch `distillationAndTransforms` (RSMoon, 2026-11-13)

**What the branch does contain** (a perception + reasoning upgrade line, reviewable on its own merits):

- `src/grounded_vision/perception/patch_transform.py` — train-fitted full-rank **patch rotation / optional whitening** before region pooling (eigenbasis of the train patch covariance; fitted on train patches only). Stored inside `perception_checkpoint.pt` so readers and features cannot disagree on the embedding space; the CLS token (PartDetector) stays raw; a legacy `rotation` key is still read.
- `scripts/fit_perception.py` — `--rotate/--whiten/--no-*` flags, and **seeded initialisation** (`experiment.seed`, default 42) — previously unseeded, so identical invocations trained different readers. Genuine reproducibility fix.
- `scripts/train_head.py` — optional **tree distillation** (`reasoning.distillation` in config): RandomForest teacher over concept bits → 10× bit-flip augmentation (1–3 attribute-bit flips, part bits fixed) relabelled by the teacher → student `CertifiableTreeHead` at depth 20; train fidelity + distillation config persisted in `tree_head.pkl`.
- `scripts/compare_patch_transform.py` — end-to-end **raw vs rotated vs whitened** harness over one symlinked feature cache (no duplicated GBs), multi-seed with mean ± std, part-IoU-constant sanity check baked in, and a comparison table that already includes the paper-spine metrics (sufficiency rate, clause correctness, P(all clauses correct), NL faithfulness, reason size) — i.e. it measures the effect on our claims, not just accuracy.
- `configs/cub_bound_local.yaml` (local protocol: distillation + whitening on, seed 42, no mock fallback), `tests/test_patch_transform.py` (9 tests), `tqdm` dep.

**Verification (lead, 2026-11-14):** 9/9 new tests pass; full suite 54 passed, 1 skipped; the distillation flip-column selection checked against `FeatureCompiler` BOUND vocabulary — it flips attribute bits only, `has_part(...)` bits excluded. Correct.

**Flags for the lead before merging:**

1. `configs/cub_bound.yaml` (canonical) defaults were flipped to `rotate_patches: true, whiten_patches: true`. **Every committed number was produced on raw patches** — merging as-is silently changes the canonical config's embedding space. Revert the default to raw (variants stay opt-in via flags / the local config), or record the switch as a deliberate protocol break.
2. Distillation fidelity is reported on train only inside `train_head.py`; the test-side effect is only visible through `evaluate.py` metrics (the comparison table covers this — fine, just don't cite train fidelity as if it were the Setting-D headline).
3. The bit-flip augmentation creates off-manifold rows by design (independent flips can break one-hot value groups); consistent with the distillation intent, but its interaction with `enforce_domain_constraints: true` is untested.
4. Local-run note: `cub_bound_local.yaml` expects relative `data/CUB_200_2011` and a fresh cache; to run `compare_patch_transform.py` on this Mac, point `data.cache_dir` at the flip-free eval-transform cache `/Users/tanmoy/research/data/grounded_vision_runs/cub_local_et/` (it has the required `{train,test}_features.npz`) and `data_dir` at the nested local CUB root.

---

## Task 1 (first, due in 1.5 days): calibrated abstention at more class counts

- Run the existing audit and uncertainty-aware reason pipeline at **K=20 and K=50**, and add the **calibrated arm at K=200**, which has never been measured.
- Then run `scripts/abstention_curves.py` on each.
- **Deliverable:** one table per K with coverage, joint premise support, number of instances and bootstrap CI for the four scores (**calibrated, raw, oracle marginal, random**), plus one figure of support against coverage with a curve per K.
- This decides whether the paper's claim is the tradeoff or the collapse, so it goes first.

Runner: `task1_calibrated_scaling.py` (pipelines → curves → figure → pre-registered GRACEFUL/CLIFF verdict lines).

## Task 2 (due day 3): sensitivity of the 4-class result

- Five random 4-class draws with fixed seeds, same pipeline.
- **Deliverable:** premise support per draw, and for each draw whether a single literal accounts for most failures and which one.
- This tells you whether "one literal carries every failure" is a property of your subset or a pattern.

Runner: `task2_subset_sensitivity.py` (draws fixed at seeds 1000–1004 via the new `--class-ids` pipeline option; dominance criterion pre-registered: literal in ≥80% of failing reasons AND marginal support ≤0.10; verdict PATTERN ≥3/5 draws).

## Task 3 (due day 4): the forcing sweep

- For each threshold, check whether the trusted literals **force** the prediction, **minimise to a reason**, and score support on both the minimal reason and the whole trusted set.
- **Deliverable:** threshold, coverage and both support numbers, **compared against the ranking rule at the same threshold**.
- This both answers the exact-query question and should explain why `guarantee_scaling` reports 0.07–0.10.

Runner: `task3_forcing_sweep.py` (δ ∈ {0.5…0.9}, K ∈ {4,10}; greedy-irreducible minimisation, documented; prints the side-by-side with the committed `guarantee_scaling` trusted-only numbers; interpretation reserved to the lead).

## Ground rules to hand over with the tasks

1. **Write new result files and never overwrite committed ones.** (Enforced in code: output paths refuse to overwrite.)
2. **Dump per-instance records, not only aggregates.**
3. **Fit calibrators on validation only.**
4. **Report the number of instances beside every percentage.**
5. **Send Task 1 numbers as soon as they exist, even if they look bad.**

## If finished early

The next job is the comparison table against the **reasoning-shortcuts (Marconato et al.)** and **Teso et al.** papers, since citations are due on the 15th. Ask the lead for the brief first — the related-work notes start from zero (nothing is in the repo).

## The cliff decision (made before the runs, per the lead's rule)

Recorded in [`CVPR2027_PREREG_TASK_RUNS.md`](CVPR2027_PREREG_TASK_RUNS.md) before Task 1 executes: **CLIFF at K** = upper 95% bootstrap bound of calibrated-arm support at 14% coverage falls below **2× the unconditional rate R_K**; **GRACEFUL** = lower bound ≥ 2×R_K; ambiguous CIs decided by the curve average over 5–30% coverage. Abstract-change rules committed with it (including: a GRACEFUL K=200 triggers a bug-check before celebration, since the oracle filter is dead at K=200).
