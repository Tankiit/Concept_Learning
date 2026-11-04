# The research story & talk track

**Use:** spoken narrative for group meetings, talks, or the paper's introduction. Every number below is verified against the committed result files (see [`RESULTS.md`](RESULTS.md); cross-checked 2026-11-04). Vocabulary follows the agreed register — explanations are **machine-checked proofs**; "certificate" appears only where it is the formal-methods term of art. Three versions: one-liner, 5-minute arc, full story.

---

## The one-liner

> Instead of asking an LLM *why* and hoping it doesn't lie, we make every explanation a **machine-checked proof over grounded concepts** — and after an adversarial review forced us to fix real bugs and regenerate every number, one finding survived everywhere we looked: **the proof layer is never the bottleneck; perception and feature compilation are.**

## The 5-minute arc

**1. The two failure modes (the hook).**
Concept bottleneck models reduce an image to an attribute bag with no connection to image evidence. LLM-generated explanations sound right and hallucinate freely. Attribution methods like SHAP *rank* features but prove nothing — "these features mattered" is not "these features *suffice*."

**2. The move (the idea).**
Keep the classifier simple enough to be *checked*. A decision tree over grounded predicates — with an independent verifier, not the training procedure, deciding what is a valid explanation. A sufficient reason is a proof object: a minimal set of conditions that provably entails the prediction, re-checkable by an independent solver. Sufficiency is decidable by exhaustive leaf analysis; minimality and necessity are audited, not assumed.

**3. The stress test (why the numbers are believable).**
An external review (2026-11-01) attacked the pipeline and confirmed five defects. Three were real bugs, fixed the same week with regression tests: a verifier class-mapping flaw, negated relational templates that verbalized *absence* as *presence*, and a tile-pooling bug sitting in *four* scripts that had silently invalidated the original activation artifacts. Two became roadmap items (three-valued predicates, binding-based audit). Then everything was regenerated (2026-11-02) under an honest protocol — official train/test split, tree fit on train only, held-out evaluation. What survived: the proof guarantees. What got honest: the accuracy (the historical "95%" was train-side; held-out is 56.8%).

**4. The evidence (three pillars).**
- **Proofs hold at scale.** On real CUB, 100% of *held-out* explanations are verified sufficient at a mean of 3.79 conditions (4-species protocol; 5.98 at 200-class scale) — while the depth-6 tree itself scores 56.8% and 4.3% respectively. The accuracy-vs-verifiability trade, displayed honestly, at both scales.
- **Grounding is measured, not asserted.** Perturbation milestone on real CUB with matched controls: reader–annotation agreement 82.8%; beak and belly predicates are specific under *all three* control families; the crown/breast "failure" was mostly adjacent-control contamination (with farthest controls, crown passes 4/4, breast 5/5); the true residual bottleneck is the deployed reader heads themselves (0.691 vs 0.824 probe agreement).
- **Relational generality.** On CLEVR-style relational QA, 100% of certified reasons cite relational predicates and every condition is necessary — accuracy is an honest 69.8% because the *feature compilation* approximates the question semantics; the proof layer stays exact.

**5. The honesty (the claim ladder).**
Say it as four rungs: *proved* (sufficiency, minimality, necessity, contrastives — exact, at every scale tested); *measured* (grounding specificity with matched controls); *sensitivity but not correction* (mask/blur shows the model relies on the region; "changing yellow to black changes the prediction" is future work); *open* (channel→attribute semantics, semantic image edits).

**6. The trade and where it goes.**
We trade end-to-end accuracy for checkability — and every track localizes its residual error *somewhere other than the proof layer*: tree capacity (activation), reader quality (milestone v2), feature compilation (CLEVR). Next work follows the measurements: reader-head training first, then relational expansion on the validated part families.

---

## The full story

### Act I — the problem with "why"

Explainability has a trust problem on both ends. Downstream, LLM verbalizers generate fluent rationales with no binding to the model: they can say "yellow belly" when the network never looked at the belly. Upstream, the dominant quantitative tools — saliency maps, SHAP — are *attributions*: continuous scores that rank influence. None of them can answer the only question a user ultimately cares about: *is this enough?* If the model predicts *warbler* because of conditions C, is C **sufficient** for that prediction — provably, not plausibly?

Concept bottleneck models (CBMs) promised interpretable intermediate concepts but, as usually built, the concepts float: "has_attribute(belly_color=yellow)" with no region, no exemplar, no probability, no audit trail. The concept vector is as opaque as the pixels it replaced.

### Act II — the design principle

The framework's design principle is **provenance all the way down, proof all the way up**:

- **Down:** every predicate fact in the database carries a `Provenance` record — which region (bbox, detector confidence), which reader probability, which prototype match, which patch indices. A verbalized sentence can be walked backward: sentence → logical conditions → facts → regions → pixels.
- **Up:** the classifier is a decision tree *precisely so that* sufficiency is decidable by exhaustive leaf analysis. The `TreeSufficiencyVerifier` is independent of training: a condition set is sufficient iff at least one leaf is reachable under it and *zero* reachable leaves predict any other class. Minimal sufficient reasons are computed as prime implicants by backward elimination; necessity is audited by leave-one-condition-out; contrastives name the exact boundary condition that would move the prediction. An independent d-DNNF checker (PyXAI on Linux/Modal — it cannot run on macOS) re-proves sufficiency and minimality on sampled payloads, so "who proved it" is an explicit field of every certificate. (Those audit numbers are server artifacts pending copy into the repo — [`RESULTS.md`](RESULTS.md) §5.)
- **Constrained:** a domain theory enforces mutual exclusions (a bill cannot be both hooked and needle) and prerequisites (an attribute requires its part) before anything is compiled.

The language layer is deliberately deterministic — template verbalization with polarity-aware phrasing — and an auditor re-reads every sentence against the proof's own conditions, measuring omitted conditions and unsupported claims. Faithfulness is a *measured* number, not a vibe: 100% in the current benchmark, with hallucination rate 0%.

### Act III — the stress test (the story beat that buys credibility)

An external review (2026-11-01) went claim-by-claim through the pipeline and confirmed five defects. Three were implementation bugs, fixed the same week and locked with regression tests (`tests/test_review_fixes.py`):

1. the verifier compared leaf argmax *positions* against class *labels* — silently correct only for contiguous training labels;
2. negated relational templates verbalized a certified **absence** as **presence**;
3. the activation track's tile pooling averaged over the wrong axes — in **four** scripts, not one — so the original "tiles" were full-width row strips and every tile-derived artifact was invalid.

Two confirmed issues became design work rather than bugs: unobserved parts defaulting to negative facts (no UNKNOWN state — Requirement 1), and the NL audit matching tokens instead of entity–attribute bindings (Requirement 3).

Then everything was regenerated (2026-11-02) under an honest protocol: official CUB train/test split, tree fit on train only, evaluation on held-out test. The result is the track's central separation: **the proof layer is scale-stable** — 100% of held-out explanations remain sufficient at ~4–6 conditions — **while the classifier's honest accuracy is 56.8%** (4-species; the historical "95%" was in-cache, train-side) **and 4.3%** at full 200-class scale. Tell it as: *we attacked our own claims harder than a reviewer would, and the accounting survived.*

### Act IV — what the experiments say

Four results, each making a different point:

1. **The activation track (proofs at scale).** On real CUB under the official split, with a DINO-activation tree: 100% of *held-out* explanations verified sufficient at a mean of 3.79 conditions (4-species protocol; 5.98 at 200-class scale). Grounding: each literal is tied to a channel, a 4×4 tile, its patch range, and coordinates — and CUB part names appear **only** where a visible part annotation lies inside the tile; otherwise the explanation says "an unannotated image region." No guessing.
2. **The grounding milestone (grounding measured, not asserted).** For the 16 recurring part–attribute predicates (of 19 in scope) on a disjoint CUB test split: probes agree with annotations at 82.8% (71.9–94.9), and targeted blur/mask of a predicate's own region moves it ~2–2.5× more than the same edit on a control region. Milestone v2 added the deployed reader heads, three control families (adjacent / farthest / distance-matched), and per-image effect records with bootstrap CIs (30,264 rows). Findings: beak and belly predicates are specific under every control family; crown/breast's apparent non-specificity was mostly adjacent-control contamination (farthest controls: crown 2.69–4.35, 4/4 pass; breast 1.51–2.76, 5/5); and the distance-matched control decomposes the effect into a content component and a distance component. The measured residual bottleneck is the **reader heads** (0.691 agreement deployed vs 0.824 for probes) — perception, not proof.
3. **The relational track (generality).** CLEVR-style existential questions over scene graphs, features including relational literals (`exists(left_of(red cube, sphere))`): 100% of certified reasons cite relational predicates, 100% of conditions are necessary — and accuracy is 69.8%, an honest number we keep in the table. It isolates *where* the error lives: the question is existential over all pairs, the features are per-literal — so the residual error is in feature compilation, not in the proof layer. The proofs remain exact for whatever tree is learned.
4. **The smoke benchmark (the machinery works).** Fresh clone, mock data, end-to-end through all six stages: 100% sufficiency, 4.38-condition mean reasons, 100% NL faithfulness, 0% hallucination, interventions flipping predictions at 43.8%. Mock classification accuracy is meaningless by construction and we label it as such — the run proves plumbing, not science.

A useful contrast within our own results: a cache-only DINO baseline hits 95% training accuracy with 3.60-condition path certificates but provides **no spatial grounding** — accuracy without grounding is easy; the whole point is having both. And read the four results together: each track localizes its residual failure *outside* the proof layer — tree capacity, reader quality, feature compilation, plumbing respectively. That is the thesis, measured.

### Act V — the claim ladder (say this slide slowly)

- **Proved (exact, today):** tree-level sufficiency, minimality, necessity, contrastive boundary conditions. These are theorems about the learned decision function — machine-checked by an exact decision procedure, independently re-checkable by d-DNNF compilation.
- **Measured (with controls, today):** grounding specificity — altering a predicate's region changes that predicate more than matched control edits, with unrelated predicates comparatively stable; reader–annotation agreement; NL faithfulness by audit.
- **Sensitivity, not correction (say the limit):** masking and blurring earn *sensitivity* claims — the model demonstrably relies on that region. A *semantic correction* claim ("changing yellow to black changes the prediction") additionally requires an edit whose attribute change and content preservation are themselves verified. Not yet claimed.
- **Open (say it before a reviewer does):** three-valued predicates (unobserved ≠ absent), binding-based NL audit, channel→attribute semantics (channel wording stays until an association study passes a preset reliability threshold), image-level semantic edits.

### Act VI — where it goes

The roadmap is measurement-driven, in order of what the measurements blamed:

1. **Reader-head training** — the v2 milestone's named bottleneck (0.691 vs 0.824): retrain perception on the full cache, then re-run the milestone.
2. **Relational expansion on validated families** — beak/belly grounding is validated; certified part–attribute explanations over them are backed by measured grounding.
3. **The formal-methods upgrades:** three-valued predicate logic (Requirement 1), binding-based audit so the JSON *is* the audit input (Requirement 3), and the benefit-from-binding controlled comparisons (Requirement 4).
4. **Channel→concept promotion:** semantic names attach to activation literals only after the association study passes a preset threshold on held-out images.
5. **The reviewer-analysis battery, already scripted and scored in proof-size currency:** GOSDT/RuleFit/RF comparison, the 312-attribute ceiling, Matryoshka SAE abstraction levels, prototype-similarity trees, validated tree search over the accuracy-vs-proof-size frontier.

### Epilogue — the anecdote (optional, for lab meetings)

The repo itself proved the provenance point: the data layer silently vanished from git history (the `data/` gitignore rule meant for datasets matched `src/grounded_vision/data/`), and we recovered a snapshot from the Modal image *because the image recorded exactly what was deployed*. Provenance isn't just an explanation feature — it's how the whole project recovered its own state.

---

## Anticipated questions (short answers)

- **"Isn't 56.8% a step down from your old 95%?"** — Yes, deliberately: the review showed 95% was in-cache (train-side). Under the official split the same protocol gives 56.8% held-out — and the proofs held at 100% through the transition. We report the honest pair.
- **"What did the review actually break?"** — Three implementation bugs (verifier class mapping, negated relational templates, tile pooling in four scripts) — fixed with regression tests; two design gaps (three-valued predicates, token-based audit) — open roadmap items. The pooling bug invalidated the original activation artifacts; everything was regenerated under the honest protocol.
- **"Trees are weak classifiers, no?"** — Yes, that's the trade: checkability needs a decidable hypothesis class. The interesting question becomes accuracy-*per-proof-size*, which is exactly what the method-comparison track measures (GOSDT optimal sparse trees, RuleFit, RF).
- **"Why not fine-tune a big model and explain with SHAP?"** — SHAP ranks; it cannot prove sufficiency. On held-out images TreeSHAP's top-k overlaps the certified reason 87.9% of the time — agreement where the signal is strong, and the gap is exactly where a ranking, unmoored from proof, drifts.
- **"Is 100% sufficiency trivial?"** — It holds for pure leaves; the verifier flags impure leaves explicitly (the tree-quality analysis counts "vacuous certificate" leaves), and the proved claim is always about the tree as learned.
- **"Does the contrastive mean editing the image works?"** — No, and we say so: it's model-level counterfactual sensitivity. Image-level *sensitivity* is now measured (the perturbation milestone); *semantic* image edits are future work.
- **"Why 69.8% on CLEVR?"** — Existential semantics vs per-literal features: the approximation lives in feature compilation; the proof layer stays exact. That separation is a feature of the architecture, not a bug in it.

## Related pages

- [`RESULTS.md`](RESULTS.md) — every number, with provenance
- [`REVIEW_VALIDATION_ROADMAP.md`](REVIEW_VALIDATION_ROADMAP.md) — the review, claim by claim, and the open requirements
- [`GROUNDED_VISION_FRAMEWORK.md`](GROUNDED_VISION_FRAMEWORK.md) — the system behind the story
- [`ACTIVATION_GROUNDING_WIKI.md`](ACTIVATION_GROUNDING_WIKI.md) — the real-data research log
- [`NLX_EXAMPLES.md`](NLX_EXAMPLES.md) — example sentences to read aloud
