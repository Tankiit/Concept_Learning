# CVPR 2027 — Introduction draft

**Drafted:** 2026-11-12. Built from the keep list in [`CVPR2027_CLAIM_AUDIT.md`](CVPR2027_CLAIM_AUDIT.md) (final cut 2026-11-11) and the §9 abstract of [`CVPR2027_SUBMISSION_PLAN.md`](CVPR2027_SUBMISSION_PLAN.md). Contributions bullets mirror the abstract verbatim in structure (number-free, "helps with" clauses). Title: *Sufficient for the Tree, Not for the Bird: Auditing What Concept Explanations Actually Certify*.

---

## LaTeX source

```latex
% NOTE 2026-11-12: citation keys are PLACEHOLDERS (related-work pass
% scheduled Nov 15). Planned anchors: CBMs koh2020concept; post-hoc
% concept explanations yuksekgonul2023posthoc; CEM zarlenga2022cem;
% abductive explanations ignatiev2019abduction; PyXAI pyxai;
% concept-level formal / partial-input explanations teso, soria, huang
% (these occupy the novelty ground -- do NOT claim first formal concept
% explanations); reasoning shortcuts marconato2024 (comparison lives in
% the audit section, not here).
% RESOLVED 2026-11-12: canizales = Canizales, Gopinath, Pasareanu, Mangal,
% "Concept-Based Abductive and Contrastive Explanations for Behaviors of
% Vision Models", arXiv:2605.06640 -- the CSU-TrustLab/behavior-explainer
% repo. Their machinery is already ported (grounded_vision/bankexp/) and
% run on our K=4 setup; see bankexp/results/comparison_banks_K4.md and the
% keep-list rung 10 (graded bank metrics).
\section{Introduction}
\label{sec:intro}

Concept-based neural networks promise predictions that can be justified in
a human-readable vocabulary: \emph{this bird is an American Pipit because
its bill is not yellow and its bill shape is not curved}
\cite{koh2020concept,yuksekgonul2023posthoc,zarlenga2022cem}. Formal
explanation methods make that promise a guarantee: an \emph{abductive}
(sufficient-reason) certificate is a minimal set of concept literals that
provably entails the prediction of a Boolean concept model---remove any
literal and the entailment breaks
\cite{ignatiev2019abduction,pyxai,teso,canizales,soria,huang}. The
certificate is exact, cheap to compute, and free of extraction heuristics,
which is precisely why it is attractive as the interface between a neural
model and the person relying on its explanation.

The guarantee, however, is a statement about the symbolic model---not about
the image, and not about the network it summarises. It rests on two
premises that the proof does not establish: that the cited literals are
\emph{true of the input}, and that the symbolic model \emph{agrees with the
neural model} being explained. Both premises can fail while the certificate
remains perfectly exact, and the certificate silently inherits the failure:
nothing in the verified object signals which premise has broken. It is
sufficient for the tree, not for the bird. What is missing is a measurement
of how far such certificates actually transfer, and of what---if
anything---restores them.

We make this \emph{premise gap} measurable. On CUB fine-grained recognition
with oracle part localisation, we read a fixed concept vocabulary with
trained concept readers, attach a Boolean head in two ways---trained
natively on predicted concepts (Setting~T) and distilled from the neural
teacher (Setting~D)---and certify minimal sufficient reasons with an exact
engine, independently re-verified, while sweeping the task from a small
class subset to the full benchmark. For every certified reason we measure
\emph{premise support} (are all cited literals true of the image, against
held-out part annotations) and \emph{conditional fidelity} (does the neural
teacher issue the same prediction whenever the cited literals hold), and we
organise the results on a \emph{claim ladder} from what is proved to what
merely transfers. We then replicate the premise-support measurement on two
further datasets with per-image concept ground truth---CelebA face
attributes and the BDD-OIA driving-action benchmark---keeping the ladder
and the audit fixed while the domain, features, and task type vary.

Four findings survive the full audit---across both heads, three scales,
and three datasets.
\textbf{Premise support is the binding constraint, and it collapses
with scale.} The fraction of certified reasons whose literals are all true
of the image falls steeply as classes are added, and lies below the
predictive accuracy of the model being explained at every scale: a
certified explanation is less likely to be fully correct than the
prediction it certifies. The same ordering---premise support well below
accuracy---holds on both replication datasets. Conditional fidelity is
partial but is not the binding premise.
\textbf{The failure is concentrated, not diffuse.} On the smallest
class subset, a single literal---yellow bill colour---occurs in every
failing reason with no support in the part annotations, while every
passing reason cites only the \emph{absence} of concepts. The pattern
replicates: on the driving benchmark the failing slots are positive
traffic literals (green light, red light), on CelebA the negations of two
face attributes---so failure targets identifiable (concept, polarity) slots
rather than the model at large, with the polarity direction set by the
domain rather than universal.
\textbf{The failure is identifiable, and abstention is the lever.}
Abstaining whenever the minimum \emph{calibrated}, polarity-aware
per-concept confidence of a reason's literals is low---the calibrator fit
on a small validation split---recovers high premise support at partial
coverage on every dataset we test, on the main audit statistically
indistinguishable from an oracle filter that knows each literal's true
support, whereas discrete per-concept reliability never separates failure
from success. Where the readers are demonstrably miscalibrated, raw
confidence selects at or below chance and calibration is load-bearing;
where readers are already calibrated, raw confidence tracks the calibrated
rule---so we recommend the calibrated rule, and locate the miscalibration
diagnosis in the settings that exhibit it. \textbf{No selection rule we tested escapes the premise
failure.} Oracle choice of concepts or readers gains nothing, a
substantially stronger reader raises accuracy and leaves premise support
unchanged, and retraining only on certified examples changes nothing;
calibration-gated abstention is the only lever that moves premise support,
and it does so at the price of coverage.

In contributions form:
\begin{itemize}
\item \textbf{A claim ladder} separating what a certified explanation
proves (entailment by the symbolic model) from what must be measured to
transfer (fidelity, premise support), so users of certified explanations
know which guarantee they actually hold (\S\ref{sec:ladder}).
\item \textbf{A vocabulary-ceiling diagnostic} separating an inexpressive
concept vocabulary from a symbolic model that lacks capacity, so repair
effort targets the right component (\S\ref{sec:ceiling}).
\item \textbf{A premise-support audit} across two heads, three scales, and
three datasets, showing that verified reasons agree only partially with the
neural teacher, that their premises are rarely all true of the image and
become rarer as classes are added, and that failures concentrate in
identifiable (concept, polarity) slots, so repair targets those slots
rather than the model at large (\S\ref{sec:audit}).
\item \textbf{A calibration-gated abstention rule} under which raw concept
confidence does not identify unreliable reasons---it is no better than
chance---whereas per-concept, polarity-aware calibration does, recovering
high premise support at low coverage, so deployments can tell when a
certified reason may be trusted (\S\ref{sec:abstention}).
\end{itemize}

We are equally deliberate about what we do not claim. That the
certification machinery is exact is a property of abductive certificates by
construction; we verify it with an independent checker and treat it as a
sanity check, not a finding. Our main audit is CUB with oracle localisation
and decision-tree heads, and its CUB-specific quantities---scaling with
class count, interventions, vocabulary ceilings---remain single-bed; the
premise-support measurement itself is replicated across three datasets,
two feature pipelines, and task types from binary attributes to driving
actions. And calibration repairs trust conditionally, paying in coverage:
we report what it can and cannot restore.
```

---

## Quantity → artifact map (for the BODY — the intro draft itself is number-free, all quantities live in the body)

| Sentence | Quantity | Artifact |
| --- | --- | --- |
| Finding 1 | 25.4% / 11.5% / 1.7%, joint 0.8%, always < accuracy (0.60/0.44/0.19) | `cbm_cem_audit.json`, `cbm_cem_audit_K200.json`, `guarantee_scaling_cub.json` |
| Finding 1 | fidelity 0.81 / 0.61 | `conditional_fidelity.json` |
| Finding 2 | one literal in all 88 K=4 failures (0/88 support), 30 passes absent-only, polarity beyond base rates | `independence_gap_cub.json`, `yellow_bill_audit.json` (worked example, not standalone claim) |
| Finding 3 | 0.697 [0.52, 0.85] at 13.6% coverage (n=33, 23 correct); oracle 0.727; random 0.123; raw 0.030 | `abstention_curves.json` |
| Finding 4 | oracle gain 0 (mlp 5/200); stronger reader 0.602→0.678 acc, P(sup) 25.4→24.6%; certified-only retrain unchanged | `cbm_cem_audit*.json`, `certified_only_cub.json` |

## Drop-list compliance check

- No "100% verified" as a finding (scope paragraph calls it by-construction sanity). ✔
- No cross-dataset-invariant phrasing (explicitly disclaimed in scope paragraph). ✔
- No CLEVR premise claims. ✔
- No 0/118 count, no "6×", no "calibrated filter fails at K=200" (coverage-cost anchoring only). ✔
- Yellow-bill appears only as the concentration example. ✔
- K=4 hedged as "the four-class subset" in both places it carries weight. ✔

## Flags before this freezes

1. **Subset sensitivity (keep-list open item):** finding 2's "on the four-class subset" hedge must stay until the ≥5 random 4-class draws run; if it never runs, the hedge stays permanently.
2. **Trust-reconciliation (audit re-audit §open):** `guarantee_scaling_cub.json` per-instance isotonic trust (P(kept-true|covered) ≈ 0.07–0.1) vs `abstention_curves.json` B — body must state the fixed-δ trusted-sufficient query vs ranking distinction, or the scaling readout gets rerun as a ranking. The intro sentence is consistent with either resolution but the body text depends on it.
3. **Numbers in intro vs number-free style:** RESOLVED v3 (2026-11-13) — user instruction; intro is now fully number-free (no measured quantities, no structural counts, findings de-numbered to bold titles). Only LaTeX comments and bib keys carry digits.
4. **Subset sensitivity hedge, number-free form:** finding 2 now says "the smallest class subset" — the four-class specificity lives in the body; hedge still required there until the ≥5 random subset draws run.
4. **Trim targets if the page runs long:** the fidelity parenthetical in finding 1, and the scope paragraph can compress to two sentences.

---

## Changelog 2026-11-12 (late) — v2, after the CelebA + BDD-OIA replications

Grounded in `results/celeba_audit_smiling.json` and `results/bdd_oia_audit.json` (see the keep-list addendum in [`CVPR2027_CLAIM_AUDIT.md`](CVPR2027_CLAIM_AUDIT.md)). Paragraphs 1–2 (promise, premise gap), the setup paragraph's CUB core, contributions bullets 1/2/4, and the ladder framing are unchanged.

1. **Setup paragraph:** + one sentence announcing the two replications (CelebA attributes, BDD-OIA driving actions) with the ladder/audit held fixed.
2. **Finding (1):** + clause — the ~20-point support-vs-accuracy gap persists on both replications (0.69 / 0.53 vs 0.88 / 0.72–0.75).
3. **Finding (2):** concentration now stated as replicated, with the honest polarity nuance — CUB and BDD-OIA fail on *positive* literals, CelebA on *negated* attributes; the transferable claim is identifiable (concept, polarity) slots, polarity direction is domain-set. (The old "positive literals near zero, negatives near one, beyond base rates" universal phrasing is gone — CelebA falsified it.)
4. **Finding (3):** retitled "identifiable, and abstention is the lever". Calibration-specific superiority now scoped to miscalibrated readers (CUB); the recommended rule (calibrated min-confidence) is universal; the discrete reliability channel fails everywhere. This replaces the old universal "raw confidence is at or below chance".
5. **Contribution bullet 3:** "two heads and three scales" → "two heads, three scales, and three datasets".
6. **Scope paragraph:** rewritten — "one dataset, stated as scope" superseded for the premise-support measurement (three datasets, two feature pipelines); CUB-specific quantities (scaling, interventions, ceilings) remain single-bed. Still disclaims exactness-by-construction and the coverage cost.

**Flags before this freezes (v2):**
- The two replication JSONs and scripts are **uncommitted** — commit before the paper cites them (user's committed-JSON rule).
- Replications are single seed (42), single depth-4 tree, no multiplicity handling — label as replication evidence in the body; CUB remains the audited main result.
- Wording is count-agnostic where cheap ("both replication datasets") so a SUN row, if the tarball lands in time, slots in with one edit.
- Decide main-paper table vs rebuttal placement for the replication rows (recommendation: main paper, one compact table — it removes the top structural objection). If held for rebuttal, revert items 1–3 and 6 via this changelog.

---

## Changelog 2026-11-13 — v3: number-free pass (user instruction)

Intro now carries **zero numerals in rendered text**, matching the abstract's convention. Changes: setup sentence "four-class subset → full 200 classes" → "a small class subset to the full benchmark"; finding 1 stripped of all quantities (25.4/11.5/1.7/0.8, the ~20-point gap clause, 0.69/0.53/0.88/0.72–0.75, fidelity 0.81/0.61) — now "falls steeply as classes are added… lies below the predictive accuracy… the same ordering holds on both replication datasets"; finding 2 "four-class subset" → "smallest class subset", "one literal… zero support" → "a single literal… no support"; finding 3 stripped of 70%/14%/0.96/0.72, "(CUB)" moved out of the miscalibration clause; findings de-numbered (bold titles, no "(1)–(4)"). Verified by scan: only LaTeX comments and bib keys contain digits. The quantity→artifact map above is now the body's shopping list.
