# behavior-explainer comparison — code, paper numbers, and what a dataset-level comparison can measure

**Drafted:** 2026-11-12, after cloning [CSU-TrustLab/behavior-explainer](https://github.com/CSU-TrustLab/behavior-explainer) to `grounded_vision/tmp/behavior-explainer` (HEAD `a3a9583`, 87 files). Companion to [`CVPR2027_CLAIM_AUDIT.md`](CVPR2027_CLAIM_AUDIT.md) (rung 10 = our ported-machinery comparison on CUB) and [`CVPR2027_SUBMISSION_PLAN.md`](CVPR2027_SUBMISSION_PLAN.md) (§7 reviewer defences). The repo is the official code for **Canizales, Gopinath, Pasareanu, Mangal, "Concept-Based Abductive and Contrastive Explanations for Behaviors of Vision Models", arXiv:2605.06640** — the `canizales` prior-work anchor.

---

## 1. What their certificate actually is (verified from code + paper)

Pipeline (`src/`): finetune ResNet18/VGG19 on RIVAL10 or EuroSAT → train bidirectional linear aligners penultimate↔CLIP (`train_aligner.py`) → concept vocabulary = CLIP text vectors, ~150–300 words, MS-COCO-derived, energy-ordered (`build_vocab.py`, shipped in `vocabs/MCS_90_NA_*.txt`) → **concepts are manipulated, not read**: presence is operationalised as cosine strength `st(z)_i = cos(z, c_i)`, and "setting a concept false" means *erasing its direction from the embedding* (Ortho projection / SpLiCE ADMM / LEACE — `utils/concept_eraser.py`).

The certificate (`src/explain.py`): an AXp is a concept subset S such that erasing everything **outside** S preserves the network's prediction — the oracle `CheckAXpFast` calls `pred_fn`, which is a forward pass of the *actual finetuned head* through `align_inv`. Three consequences:

1. **Fidelity to the neural model holds by construction** — the oracle *is* the network. Our Setting T/D tree certificates have a measured fidelity premise (0.814 / 0.613); theirs has none to measure. This is their design's genuine advantage and the paper should say so.
2. **Exactness is conditional on a monotonicity assumption.** Their WeakConAXp/CXp reduction to a single forward pass assumes the head is monotone in each concept strength; they validate empirically — holding rates O_A/O_C range **52.5%–99.1%** depending on behavior (their Table 3). Where monotonicity fails, the weak certificate is not a certificate; full enumeration is the fallback.
3. **Premise support (image-truth of concepts) is undefined in their vocabulary.** No dataset they use has concept annotations; "validity" is graded by an LLM judge (Gemini 3 Pro). Their own examples of spurious certificates: **{Fairy(+)}** for AnnualCrop and **{Smoke(+) ∧ Rat(+) ∧ Duck(+)}** for SeaLake→Forest. That is the premise gap arriving in their pipeline — visible, but unmeasurable without ground truth, so they filter post-hoc with an LLM instead of calibrating against annotations.

Multiplicity: certificates are nowhere near unique in their setting — their Table 2 reports, per image, e.g. Ortho+XpEnum 42.5 AXps (RIVAL10/ResNet18, B(Car)), XpSatEnum 547.7 AXps + 1601.7 CXps; EuroSAT/VGG19 B(Lake,Pasture): 194.9 AXps (XpEnum), **10,233.8** (XpSatEnum). NaiveEnum is depth-bounded at |S|≤2. Compute: hours per behavior-config (1h default timeout in code; their README: full suite ≈ a week on GPU).

Contrast with ours: one effectively unique minimal reason per instance (mean 2.89 literals; UAR≡ORD 722/722), exact by tree construction, milliseconds to certify — but fidelity is a *measured* premise (0.814/0.613) and premise support is the binding, scaling failure (25.4% → 1.7%).

## 2. Side-by-side

| dimension | behavior-explainer (theirs) | grounded_vision (ours) |
| --- | --- | --- |
| certified object | finetuned neural head, queried through erasure ops in aligned CLIP space | Boolean tree head (Setting T native / D distilled) over read concept bits |
| exactness | conditional (WeakConAXp; head monotonicity holds 52.5–99.1%) | exact by construction (native DFS + PyXAI, cross-verified) |
| fidelity to neural model | by construction (oracle = the head) | measured premise: 0.814 (K=4) / 0.613 (K=10) |
| premise support (literals true of image) | not measurable — no concept GT in RIVAL10/EuroSAT; LLM-judged "validity" instead | the headline: 25.4% → 11.5% → 1.7%, concentrated in (concept, polarity) slots |
| explanations per image | 10¹–10⁴ (enumerable families) | ~1 minimal reason (2.89 literals mean) |
| concepts | CLIP text directions, sign = polarity, generic vocab (same word list on birds and satellites) | trained part-grounded readers, CUB attribute vocabulary, annotation-checkable |
| cost | forward-pass oracle × ~10³–10⁵ checks/image-config; ~week GPU for full suite | instant tree certification; all committed runs fit a laptop |
| what the audit finds in their own results | spurious concepts (Fairy, Smoke, Rat, Duck) visible to their LLM judge | the same failure class, measured, localised, and calibration-recoverable (0.697 at 13.6% coverage) |

The one-sentence framing for related work: *their approach certifies against the network by construction and leaves image-truth to an LLM judge; ours certifies against a symbolic head and measures image-truth against annotations — the two premises are complementary, and the binding one (premise support) is only measurable where concept ground truth exists.*

## 3. Dataset verdicts

| dataset | classes / n | concept GT | fidelity measurable | premise support measurable |
| --- | --- | --- | --- | --- |
| CUB (ours) | 4/10/200 of 200; 5,794 test | per-image attributes + part boxes | yes (committed) | yes (committed) |
| RIVAL10 (theirs) | 10 CIFAR-like superclasses; 21,178 / 5,306 (URL split, shipped meta) | class labels + object/background segmentation; the CVPR'22 source paper studies visual attributes (attr ablation script) — per-image attribute labels plausibly exist in the official Box zip, **unverified** | yes (if we run a model) | only if attribute labels are per-image and mappable to a vocabulary — needs zip inspection; the one candidate for an off-CUB premise-support audit |
| EuroSAT (theirs) | 10; 27,000 × 64×64 | none | yes | **no — impossible**, no annotation of "forest"/"river" presence per patch beyond the label itself |

## 4. Can we run their code here? (feasibility, 2026-11-12)

Verdict: **not before Nov 16, and unnecessary for the paper's comparison.** Code-level blockers, all verified in the clone:

1. `Pickler.read("clip")` expects a legacy `t2c` module from a non-public sibling repo (`run_experiment.py` l.45–48) — we'd have to fabricate the pickled CLIP wrapper ourselves.
2. The engine is float64 + CUDA-oriented; MPS does not support float64 matrix ops well, and the oracle is a per-check single-image forward pass — on CPU, one behavior-config (185 images × 250 NINA iters × ~200-concept OneAXp reductions) is 10⁶-scale forwards = hours-to-days.
3. ResNet×EuroSAT needs an external SSL4EO MoCo checkpoint not shipped; the shipped `results/*.csv` are from an integration test (their own `results/README.md`), not paper numbers.

What IS feasible, in cost order:

- **Done (0 h):** paper-numbers comparison above — their Table 2/3 numbers vs our committed JSONs. This is what the related-work section needs; no rerun of their suite adds to it.
- **Optional pre-deadline (~0.5–1 day):** EuroSAT through **our** bankexp port (altbank path — cls/meanpatch PCA needs no parts; DINO features via Modal, ~minutes of GPU). Buys a second-dataset row for the decomposition: certificate sizes, metric1–3, MHS on EuroSAT. Honest scope: premise support *not measurable there* — which itself supports the paper's point that annotation support is the scarce resource. Decide against the keep-list runs (subset draws, mid-K abstention) if time conflicts — those defend actual claims.
- **Post-deadline / rebuttal (~2–3 days):** download the RIVAL10 Box zip, inspect per-image attribute labels; if they exist, replant the CUB reader recipe (attribute readers → tree head → certificates → premise support against attribute GT). This is the only route to an off-CUB premise-support number. Do not promise it in the submitted paper; it is the natural "second dataset" response to reviewers.

EuroSAT RGB zip download started 2026-11-12 to `grounded_vision/tmp/datasets/` (DFKI host, slow) — for the optional run above only.

## 5. Ready related-work paragraph (for the Nov 15 citation pass)

> Closest to our audit is the behavior-explainer line of Canizales et al. [arXiv:2605.06640], which computes abductive and contrastive explanations for *behaviors* of vision models by querying the network head through concept-erasure operators in aligned CLIP space. Because their oracle is the network itself, fidelity holds by construction — but exactness is conditional on an empirical head-monotonicity assumption (holding rates 52.5–99.1% in their own validation), explanations come in families of hundreds per image, and the truth of the cited concepts is graded by an LLM judge rather than measured, with spurious certificates such as {Fairy(+)} visible in their own analysis. Our audit is complementary: we certify against a symbolic head where exactness is unconditional and reasons are essentially unique, and we spend the saved effort measuring the premise their setting cannot express — whether the cited literals are true of the image — finding it binding, structured, and calibration-recoverable only at coverage cost.

Numbers used (all checkable): their Tables 2–3 (arXiv:2605.06640v1); ours per the keep list ([`CVPR2027_CLAIM_AUDIT.md`](CVPR2027_CLAIM_AUDIT.md) §Keep list).
