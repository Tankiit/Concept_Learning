# CVPR 2027 submission plan — the eight-day cut

**Drafted:** 2026-11-08. Calendar verified against the official CVPR 2027 pages (same day); every experimental claim checked against the committed result JSONs at repo HEAD `d4e66b5`. Source: an external draft plan (its §1–§5 are reproduced here, corrected and annotated — see §2 "What changed and why").

---

## 1. Calendar — verified 2026-11-08

All from the official CVPR 2027 pages, not third-party trackers (one tracker disagrees; the official page wins):

| Milestone | Date (11:59 PM AOE) |
| --- | --- |
| Paper registration / enrollment (title, abstract, authors lock) | **Tue Nov 10, 2026** |
| Full paper submission | **Mon Nov 16, 2026** |
| Supplementary materials | Mon Nov 23, 2026 |
| Reviews released / final decisions | Jan 25 / Feb 25, 2027 |
| Conference (Seattle) | June 20–25, 2027 |

The call for papers states: "*Date is fixed, no extension will be given*." Sources: <https://cvpr.thecvf.com/Conferences/2027/Dates>, <https://cvpr.thecvf.com/Conferences/2027/CallForPapers>.

Arithmetic from Nov 8: **two days** to lock title/abstract/authors, **eight days** to a PDF. That leaves no room for the full P1–P11 program; this plan is the cut that still supports a paper.

## 2. What changed from the source plan, and why

The source plan was checked claim-by-claim against the repo on 2026-11-08. Five corrections are folded into this version (full ledger in the Appendix):

1. **Two of the three "must" experiments are already executed.** The vocabulary-ceiling ladder and the conditional-fidelity + margin-intervention suite were run and committed on the morning of 2026-11-08 ([`design_b_v2.json`](../../grounded_vision/results/design_b_v2.json), [`conditional_fidelity.json`](../../grounded_vision/results/conditional_fidelity.json), [`design_b.json`](../../grounded_vision/results/design_b.json)). The Nov 10–12 run window is really for the uncertainty-aware comparison only.
2. **"Setting A/B" naming is new and collides.** No A/B setting strings exist in the repo, and the letters are already taken: *mechanism A/B* = certified minimal reason vs certified forcing set ([`run_certified_literals_cub.py`](../../grounded_vision/scripts/run_certified_literals_cub.py)); *arm A/B* = PyXAI exclusion arms. This plan uses **Setting T (native tree head)** and **Setting D (distilled, = Design B)**.
3. **Experiment numbering collides with wiki usage.** E0 (uniqueness counting, `run_pyxai_modal.py`) and E1 (output-match vs certificate-event, 2.6% vs 2.4%, RESULTS §7) are taken. This plan renames the source plan's E1–E4 to **X1–X4**.
4. **The flip condition is not pre-registered anywhere.** The source plan said "your notes have one candidate (some attribute-positive literal certifying at τ≥0.8)". The existing pre-registrations state the *opposite* expectation (see §8). If that flip condition is wanted, it must be written down fresh, before anything runs.
5. **The related-work notes it cites are not in this repo.** No Teso / Canizales / Soria / Huang anywhere (grepped, including PDFs). The Nov 15 related-work session starts from zero; the wiki's PyXAI engine-verification section is the only adjacent material ([`REVIEW_VALIDATION_ROADMAP.md`](REVIEW_VALIDATION_ROADMAP.md)).

## 3. The paper in one paragraph

**Working claim:** *A formal sufficiency certificate is a statement about a tree, not about the image or the neural model. We measure how far it transfers, and show which concept-level claims survive uncertainty and counterexample testing.*

This is a **measured-limit paper with a method inside it**: uncertainty-aware reasons, a vocabulary-ceiling diagnostic, and conditional fidelity with finite-sample bounds. Do **not** claim novelty for concept-level formal explanations or partial-input explanations — prior work (Canizales, Soria, Teso; probabilistic abduction) occupies that ground. *(Citations to be pulled fresh on Nov 15 — not currently in the repo.)*

**Contributions to write against:**

1. **Claim ladder:** proof → robustness → fidelity → alignment → semantics → image truth, each rung labelled exact or measured. (Partially exists as the rung table: [`predicate_rung_table.json`](../../grounded_vision/results/predicate_rung_table.json), n = 5,794.)
2. **Vocabulary-ceiling diagnostic:** the best fidelity any function of φ can reach; separates "vocabulary can't express the head" from "tree too shallow". **Executed** — see §5, X1.
3. **Conditional fidelity with Wilson bounds:** P(neural = c | φ(x) satisfies R), per reason. **Executed** — see §5, X3.
4. **Uncertainty-aware reasons (Setting T):** sufficiency over per-image completions, vs ordinary reasons at matched length and coverage. **Not started** — the only genuinely new "must". See §5, X2.

## 4. Scope: must, should, cut — with execution status

| Tier | Item | Status at 2026-11-08 HEAD `d4e66b5` | Remaining work |
| --- | --- | --- | --- |
| Must | **X1** ceiling on CUB (bits vs continuous scores) | **Executed.** Ladder in `design_b_v2.json`: bits36 0.627 < region384 0.712 < scores36 0.746 < gtbits36 0.805; heads: patch 0.898, gt 0.678. Older concept-vocabulary ceiling in `concept_ceilings_gt_regions_v2.json` (tree-on-GT 15.6%) | Interpret; decide the bits-vs-scores story; write |
| Must | **X2** Setting T vs D: ordinary vs uncertainty-aware reasons, matched length + coverage | **Not started.** No "completions" machinery anywhere in the repo | Build it. Known risk: `design_b.json` has explained fraction 0.0 (`no_trusted_reason` 118/118 at both K) — the matched-coverage leg currently has almost nothing to match on; the trusted-reason δ sweep tops out at 28.8% coverage (`trusted_reasons_*.json`) |
| Must | **X3** conditional fidelity + continuous margin interventions (keep-R / delete-R vs size-matched random) | **Executed.** `conditional_fidelity.json`: overall agreement 0.814 (K=4) / 0.613 (K=10); pipit pattern {bill_color=yellow=0} n=30, fidelity 1.0 / 0.8. `design_b_v2.json` margins: sufficiency retention −6.71 (R) vs −6.25 (random); necessity drop −0.39 vs −0.55 (R ≈ random, n=118). The 0.25-floor discrete intervention is documented in `design_b.json`: donor 29/118 = 0.246 vs random 0.280; meanpatch 0.254 vs 0.263 — exactly the 4-class chance floor | Write-up; multiplicity / min-n choices; decide how the two intervention variants are presented |
| Should | **X4** CLEVR positive control | Existing CLEVR artifacts are **not** a certified-literal control: `clevr_relational_results.json` (relational track, acc 0.698, necessity 1.0) and the PyXAI exclusion/E0 check. A certified-literal run on CLEVR = new pipeline work | Recommendation: **cut to limitations now** — the GT-concept arm (gtbits36 0.805) is already an executed positive control on CUB |
| Should | Baseline-adjusted literal certification (lift over always-absent rate) | **Partially exists:** `certified_literals_frontier*.json` carries τ ∈ [0.5, 0.95] × K ∈ {4, 10, 20} with Bonferroni / lift / class-MI | Decide whether lift enters the main table or appendix |
| Cut | LTN / DeepProbLog baselines | — | Revisit only if the paper claims to learn predicates (joint training) |
| Cut | Straight-through / joint training | — | Future work; frozen predicates first |
| Cut | Forests / boosted trees, predicted parts, P2, P8–P10 | — | Appendix or follow-up |

**Scope statement to carry into the paper:** the guarantee applies to the final discrete model and its declared predicate domain, however the predicates were learned. Box centres are labelled **oracle localisation** in the main results (and in the abstract).

## 5. Experiment key (renumbered)

- **X1** ceiling ladder — executed 2026-11-08.
- **X2** T-vs-D matched comparison — new build; the headline.
- **X3** conditional fidelity + interventions — executed 2026-11-08; write-up remains.
- **X4** CLEVR certified-literal positive control — recommended cut.

## 6. Eight-day schedule (Sun Nov 8 → Mon Nov 16) — corrected

- **Nov 8–9 (today/tomorrow):** freeze the Setting T/D protocol, the matching rule, and the flip condition (**to be written from scratch** — see §8; nothing is pre-registered under that name). Write the abstract (skeleton in §9).
- **Nov 10:** register title, abstract, authors. Author list locks here.
- **Nov 10–12:** build and run **X2** (the only unexecuted "must"). X1/X3 are re-analysis of committed JSONs, not new runs. Do not start X4.
- **Nov 13–14:** figures and tables; draft method + results. **Known backlog:** wiki content stops at 2026-11-06; the eleven Nov 8 JSONs (rounds 2–7: certified literals, frontier, rung table, shift, conditional fidelity, localization noise, certified-only retrain, design B, design B v2) are not yet written up anywhere.
- **Nov 15:** related work **from zero** (citations not in repo), limitations, reviewer-facing caveats, anonymisation.
- **Nov 16:** submit, with portal buffer.

## 7. Reviewer defences (evidence-mapped)

- **Teso et al.** — say what is shared and where we differ: we add uncertainty-aware completions, measured fidelity to a neural teacher, and the ceiling diagnostic. *(Citation must be pulled; not in repo.)*
- **Oracle parts** — label in setup and abstract: oracle localisation.
- **Small samples** — "30 images, one species at K=4" is real and committed: all 30 all-correct single-clause reasons are American Pipit, other three species 0/88 (`certified_literals_cub.json`). Present as feasibility evidence with Wilson intervals shown.
- **PyXAI correctness** — state version 2.0.1 and the literal-based vs feature-based minimality choice; each reason verified independently by the native verifier. Known engine caveats documented in [`REVIEW_VALIDATION_ROADMAP.md`](REVIEW_VALIDATION_ROADMAP.md) (four pyxai 2.0.1 bugs, sklearn node-id gotcha).
- **Multiplying guarantees** — report the joint event directly: P(all clauses correct) 25.4% (K=4) → 1.7% (K=200) on 5,794 test images (`guarantee_scaling_cub.json`); a marginal conformal guarantee does not certify each explanation.

## 8. Decisions to make today (2026-11-08)

1. **Flip condition.** What X1/X2 result would reframe the paper from "method" to "measured limit"? The candidate in the source plan — *an attribute-positive literal certifying at τ ≥ 0.8* — is **not** in any note, and the existing pre-registrations predict the opposite: `run_certified_literals_cub.py` P2 ("bill_color=yellow, pos, fails at every τ; most present-literals fail at τ = 0.8") and `build_predicate_rung_table.py` PR1 ("≥ 20 predicates reach rung 3 via a NEGATIVE literal at τ = 0.8; zero via a positive"). Given `yellow_bill_audit.json` (reader fires 88/118 where GT is 0/118 — a polarity inversion), positive-literal certification at high τ would be genuinely surprising. Write the condition down, either direction, **before** X2 runs.
2. **CLEVR.** Nothing certified exists on CLEVR; X4 is new pipeline work in the same window as X2. Recommendation on the evidence: move to limitations today and lean on the executed GT-concept control.
3. **Fallback.** If X1's ceiling reads low and X2 shows no matched gain: submit the measured-limit version to CVPR anyway, or hold for a workshop / ICML? Evidence to weigh: P(all) at K=200 is already 1.7%, explained fraction on the distilled arm is 0.0 — the committed numbers already point at "measured limit". Decide before running, not after.

## 9. Title and abstract (filled 2026-11-11 from executed results + the claim audit)

Skeleton history: the original skeleton (Nov 8) left TODOs for X2 and the ceilings; all fills below are grounded in committed JSONs, with the 2026-11-11 claim-audit corrections folded in ([`CVPR2027_CLAIM_AUDIT.md`](CVPR2027_CLAIM_AUDIT.md) — in particular: certificate-exactness is a unit test and is no longer claimed as a finding; the graded bank metrics replace the saturated 0/118 count).

**Title (recommended, final cut 2026-11-11):** *Sufficient for the Tree, Not for the Bird: Auditing What Concept Explanations Actually Certify*

Alternates: *The Premise Gap: Calibration-Gated Trust for Certified Concept Explanations* (leads with the constructive half) · *What Is a Sufficient Reason Sufficient For? Premise Support and Calibrated Abstention in Concept Bottleneck Models* (question form, names both halves) · *Certificates Inherit Premise Failure: Auditing Concept Explanations from Proof to Image* (finding-led). Rationale for the recommendation: it states the paper's thesis in one line (the certificate covers the symbolic model, not the image), fits the CUB domain without naming it, survives reviewers who dislike cute titles because the subtitle is a plain description, and does not overclaim repair — the calibration half arrives as the fourth contribution, not the title.

**Abstract (draft):**

> Concept bottleneck models (CBMs) explain a prediction in terms of human-readable concepts, but without guarantees that the cited concepts are enough to determine it. Formal explanations supply that guarantee: a minimal set of concept literals that provably entails the prediction of the Boolean concept model. The proof covers that model — it does not establish that the literals are true of the image, or that the neural model being summarised agrees. We make this gap measurable on CUB fine-grained recognition with oracle part localisation, with four contributions. (1) A claim ladder separating what is proved from what is measured to transfer, so users of certified explanations know which guarantee they actually hold. (2) A vocabulary-ceiling diagnostic separating an inexpressive concept vocabulary from a symbolic model that lacks capacity, so repair effort targets the right component. (3) A premise-support audit showing that verified reasons agree only partially with the neural teacher, that their premises are rarely all true of the image and become rarer as classes are added, and that failures concentrate in positive literals — a single literal carries all of them in our smallest setting — so repair targets specific (concept, polarity) slots rather than the model at large. (4) A calibration-gated abstention rule: raw concept confidence does not identify unreliable reasons — it is no better than chance — whereas per-concept, polarity-aware calibration does, recovering high premise support at low coverage, so deployments can tell when a certified reason may be trusted. Formal certificates inherit premise failure; we measure it, localise it, and show what calibration-gated abstention can and cannot repair.

Number-free by user choice (2026-11-11); all quantities live in the body. Provenance for the body numbers: fidelity `conditional_fidelity.json`; premise-support scaling `cbm_cem_audit*.json` / `guarantee_scaling_cub.json`; concentration + polarity asymmetry `independence_gap_cub.json` (+ `yellow_bill_audit.json`); abstention curves with matched random / raw-confidence baselines `abstention_curves.json`. Deliberately omitted (body-only): independence-clustering ratio 1.52 (single block, hedged); no-selection-escapes (oracle gain 0, UAR≡ORD). The "cannot repair" clause is anchored to the coverage cost (unconditional rate unchanged), NOT to a K=200 calibrated-filter collapse — that remains unmeasured (no X2 K=200 block). See [`CVPR2027_CLAIM_AUDIT.md`](CVPR2027_CLAIM_AUDIT.md) keep list.

## 10. Run-config skeleton (X2 build)

```python
CONFIG = dict(
    backbone="dinov3", K=[4, 10, 20], parts="oracle",   # predicted parts = stretch
    settings=["T_tree_head", "D_distilled"],            # renamed; A/B letters taken
    depths=[2, 3, 4, 5],
    match_on=["mean_reason_length", "coverage"],        # TODO: choose the matching rule
    delta=0.05, seeds=[0, 1, 2],                        # seeds 0/1 already exist in frontier runs
)

def main(cfg):
    # X1, X3 already executed — read results/design_b_v2.json, results/conditional_fidelity.json
    res  = run_setting_t_vs_d(...)                      # X2, the new build
    # TODO: one results.json per run; every table script reads from it
```

---

## Appendix: verification ledger (2026-11-08, HEAD `d4e66b5`)

| Source-plan claim | Verdict | Evidence |
| --- | --- | --- |
| CVPR dates Nov 10 / 16 / 23, no extensions | **True** (official pages) | Dates + CallForPapers pages, quoted in §1 |
| Vocabulary-ceiling diagnostic, cheap, decides the paper | **True, already executed** | `design_b_v2.json`; also `concept_ceilings_gt_regions_v2.json` |
| Setting A/B (tree head vs distilled) as established protocol | **False as named** — new vocabulary, collides with mechanism A/B, arm A/B | grep across repo; `run_design_b.py`, `run_certified_literals_cub.py` |
| Conditional fidelity needs running | **Already executed** | `conditional_fidelity.json` (round 6, 2026-11-08) |
| Intervention test "sitting at its 0.25 floor" | **True** | `design_b.json`: 0.246 vs 0.280 (donor), 0.254 vs 0.263 (meanpatch); K=10: 0.16–0.20; pre-registered as PB4 |
| Continuous margin intervention fixes it | **Already executed too** | `design_b_v2.json`: retention −6.71 vs −6.25; necessity −0.39 vs −0.55 |
| Uncertainty-aware reasons / per-image completions | **Not found anywhere** — genuinely new | grep; nearest: isotonic trust in `guarantee_scaling_cub.json` |
| CLEVR positive control, "should" tier | **New work if kept** — existing CLEVR artifacts are relational-track + PyXAI exclusion, not certified literals | `clevr_relational_results.json`, `pyxai_exclusion_check_clevr_arm{A,B}.json` |
| "30 images, one species at K=4" | **True** | `certified_literals_cub.json`: Pipit n=30, all-correct 30; others 0/88 |
| Flip condition already in notes (positive literal, τ≥0.8) | **Not found; pre-regs predict the opposite** | P2 in `run_certified_literals_cub.py`; PR1 in `build_predicate_rung_table.py` |
| Notes list Canizales / Soria / Teso / Huang | **Not in this repo** (may live in an external note store) | case-insensitive grep of all files incl. PDFs |
| K = [4, 10, 20] supported | **True** (4/10/20 in frontier; 4–200 in scaling sweep) | `certified_literals_frontier*.json`, `guarantee_scaling_cub.json` |

---

## Status addendum 2026-11-14 (three days out; supersedes §6 schedule for the remaining items)

**Paper drafts COMPLETE (wiki, not yet .tex):** intro v3 (number-free, `PAPER_INTRO_DRAFT.md`), body v2 (`PAPER_BODY_DRAFT.md`: §2 related work incl. the Canizales mirror-image paragraph, §3 formal method, §4 CVPR-style experiments, §5–6, appendix stubs incl. optional SUN block), abstention figure script (`scripts/plot_abstention_curves.py` + draft PNG). Abstract §9 unchanged and consistent.

**Off-CUB replications EXECUTED** (keep-list addendum in `CVPR2027_CLAIM_AUDIT.md`; ledger in `SECOND_DATASETS.md`): CelebA 0.693 [0.682,0.704] n=6,654; BDD-OIA (rsbench) 0.534 [0.522,0.546] n=6,433; SUN 0.833/0.704 (K=4/10; carved split — protocol deviation, appendix placement recommended). Cross-dataset findings folded into intro v2/v3: polarity direction domain-specific; calibration-specific superiority CUB-only; discrete channel fails everywhere; reader-strength gradient.

**Remaining to submission:** (1) commit replication artifacts (`results/{celeba_audit_smiling,bdd_oia_audit,sun_audit}.json` + 5 scripts + figure script) — required by the committed-JSON rule before the paper cites them; (2) `fig:pipeline` method figure; (3) Nov 15 citation pass (placeholder keys listed in both drafts; canizales resolved); (4) user decision on SUN placement (appendix recommended); (5) optional subset-draws run (upgrades the K=4 hedge); (6) assemble .tex from the wiki drafts.

## Branch intake 2026-11-14 (`distillationAndTransforms`) — code, no numbers

Collaborator branch (RSMoon, Nov 13) reviewed on intake; full delivery-status, verification and merge flags in [`CVPR2027_HANDOFF_TASKS.md`](CVPR2027_HANDOFF_TASKS.md). Net effect on this plan: (a) reviewed, tested code for patch rotation/whitening (perception) and tree distillation (Setting-D machinery) now exists with a raw/rotated/whitened comparison harness — but **zero result files**, so §9 and the keep list are unchanged; (b) **the three hand-off runs are still unexecuted anywhere** — Task 1 (calibrated arm at K=20/50/200) remains the only pending item that can still change the abstract's "cannot repair" clause before Nov 16; (c) merge only after the canonical `cub_bound.yaml` default is reverted to raw patches (flag 1 in the hand-off page) — all committed numbers are raw-patch.
