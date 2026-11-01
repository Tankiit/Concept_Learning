# The research story & talk track

**Use:** spoken narrative for group meetings, talks, or the paper's introduction. Every number below is verified against the committed result files (see [`RESULTS.md`](RESULTS.md)). Three versions: one-liner, 5-minute arc, full story.

---

## The one-liner

> Instead of asking an LLM *why* and hoping it doesn't lie, we make every explanation a **verified theorem about a tree over grounded facts** — and we can trace every word of the sentence back to pixels, prototypes, and probabilities.

## The 5-minute arc

**1. The two failure modes (the hook).**
Concept bottleneck models reduce an image to an attribute bag with no connection to image evidence. LLM-generated explanations sound right and hallucinate freely. Attribution methods like SHAP *rank* features but prove nothing — "these features mattered" is not "these features *suffice*."

**2. The move (the idea).**
Keep the classifier simple enough to be *checked*. A decision tree over grounded predicates — with an independent verifier, not the training procedure, deciding what is a valid explanation. Sufficiency becomes a formal property: a condition set is sufficient iff every tree leaf consistent with it predicts the same class. Explanations are prime implicants: minimal sets where removing any single condition breaks the proof.

**3. The system (one breath).**
Frozen DINOv2 → perception layer (part regions, prototypes, attribute readers) → a fact database where every fact carries provenance (region, detector confidence, reader probability, prototype match) → feature compiler (flat / entity-bound / relational vocabularies) → certifiable tree head → independent verifier producing sufficient reasons and contrastives → evidence cards (boxes, crops, exemplars) → a template verbalizer whose every sentence is audited against the certified tokens.

**4. The evidence (numbers).**
- On 100 real CUB images: 95% tree accuracy, **100% of explanations formally verified sufficient**, mean reason just **3.8 conditions**. TreeSHAP's top-k overlaps our certified reasons 88.6% of the time — so the cheap attribution usually agrees, but only one of the two is a proof. *(These activation-track numbers await regeneration after the tile-pooling fix — see the note in Act III; say so if you quote them.)*
- On CLEVR-style relational QA: **every** certified reason cites relational literals, and every condition is necessary (100% necessity rate) — the certificates are genuinely about *relations*, not attribute bags.
- The full pipeline runs end-to-end on a fresh clone and self-audits: 100% sufficiency, 100% NL faithfulness, **zero hallucinated claims** in the smoke benchmark.

**5. The honesty (why to believe us).**
We are precise about the claim ladder — what is *certified* (tree-level sufficiency, exact), what is *grounded* (tile/part provenance via annotations), and what we *cannot yet claim* (image-level causal interventions; channel→attribute semantics). A contrastive explanation crosses a decision boundary *of the model*; editing the image and observing the class flip is future work.

**6. The trade and the roadmap.**
We trade end-to-end accuracy for verifiability — the CLEVR track shows it honestly: 69.8% accuracy because the *feature compilation* (existential literals) approximates the question semantics, while the proof layer stays exact. Next: PyXAI compiler-based minimum-cardinality proofs (Linux/Modal), region-level image interventions to close the causal gap, and validated channel→concept mappings so NLX can name attributes, not just channels.

---

## The full story

### Act I — the problem with "why"

Explainability has a trust problem on both ends. Downstream, LLM verbalizers generate fluent rationales with no binding to the model: they can say "yellow belly" when the network never looked at the belly. Upstream, the dominant quantitative tools — saliency maps, SHAP — are *attributions*: continuous scores that rank influence. None of them can answer the only question a user ultimately cares about: *is this enough?* If the model predicts *warbler* because of conditions C, is C **sufficient** for that prediction — provably, not plausibly?

Concept bottleneck models (CBMs) promised interpretable intermediate concepts but, as usually built, the concepts float: "has_attribute(belly_color=yellow)" with no region, no exemplar, no probability, no audit trail. The concept vector is as opaque as the pixels it replaced.

### Act II — the design principle

The framework's design principle is **provenance all the way down, proof all the way up**:

- **Down:** every predicate fact in the database carries a `Provenance` record — which region (bbox, detector confidence), which reader probability, which prototype match, which patch indices. A verbalized sentence can be walked backward: sentence → logical conditions → facts → regions → pixels.
- **Up:** the classifier is a decision tree *precisely so that* sufficiency is decidable by exhaustive leaf analysis. The `TreeSufficiencyVerifier` is independent of training: a condition set is sufficient iff at least one leaf is reachable under it and *zero* reachable leaves predict any other class. Minimal sufficient reasons are computed as prime implicants by backward elimination; necessity is audited by leave-one-condition-out. Contrastives name the exact boundary condition that would move the prediction.
- **Constrained:** a domain theory enforces mutual exclusions (a bill cannot be both hooked and needle) and prerequisites (an attribute requires its part) before anything is compiled.

The language layer is deliberately deterministic — template verbalization with polarity-aware phrasing — and an auditor re-reads every sentence against the certified tokens, measuring omitted conditions and unsupported claims. Faithfulness is a *measured* number, not a vibe: 100% in the current benchmark, with hallucination rate 0%.

PyXAI (d4 compiler) cannot run on macOS, so the default backend is the native exact verifier; compiler-based minimum-cardinality proofs run through Modal on Linux — the architecture treats "who proved it" as an explicit field of the certificate.

### Act III — what the experiments say

Four results, each making a different point — **but read the caveat first**: external review (2026-11-01) found the activation track's tile pooling averaged over the wrong axes, so the "tiles" in results 1–2 were full-width row strips. The literal certificates are valid tree conditions; the spatial grounding and headline numbers of results 1–2 await server regeneration ([`RESULTS.md`](RESULTS.md) §1). Results 3–4 are unaffected.

1. **The activation track (real data, the headline).** On 100 real CUB training images with a DINO-activation tree: 95% in-cache accuracy, 100% verified sufficient reasons at a mean of 3.8 conditions. Grounding: each literal is tied to a channel, a 4×4 tile, its patch range, and coordinates — and CUB part names appear **only** where a visible part annotation lies inside the tile; otherwise the explanation says "an unannotated image region." No guessing.
2. **TreeSHAP comparison (why not just SHAP).** At matched length, TreeSHAP's top features overlap the certified reason 88.6% of the time. The right reading: agreement where the signal is strong — and the 11.4% gap is exactly where a ranking, unmoored from proof, drifts. SHAP ranks; it cannot certify.
3. **The relational track (generality).** CLEVR-style existential questions over scene graphs, features including relational literals (`exists(left_of(red cube, sphere))`): 100% of certified reasons cite relational predicates, 100% of conditions are necessary — and accuracy is 69.8%, an honest number we keep in the table. It isolates *where* the error lives: the question is existential over all pairs, the features are per-literal — so the residual error is in feature compilation, not in the proof layer. The certificates remain exact for whatever tree is learned.
4. **The smoke benchmark (the machinery works).** Fresh clone, mock data, end-to-end through all six stages: 100% sufficiency, 4.38-condition mean reasons, 100% NL faithfulness, 0% hallucination, interventions flipping predictions at 43.8%. Mock classification accuracy is meaningless by construction and we label it as such — the run proves plumbing, not science.

A useful contrast within our own results: a cache-only DINO baseline hits 98% training accuracy with 3.30-condition path certificates but provides **no spatial grounding** — accuracy without grounding is easy; the whole point is having both.

### Act IV — the claim ladder (say this slide slowly)

- **Certified (exact, today):** tree-level sufficiency, minimality, necessity, contrastive boundary conditions. These are theorems about the learned tree.
- **Grounded (checked, today):** every condition maps to regions/tiles/patches; part names only where annotations support them; NL faithfulness measured by audit.
- **Not yet claimable (say it before a reviewer does):** image-level causal interventions (mask/blur/inpaint the grounded tile and observe the class flip) — the contrastives certify *model-level* counterfactual sensitivity; channel→attribute semantics ("channel 242 means streaked back") awaits a validation protocol with a reliability threshold.

### Act V — where it goes

- Compiler-based minimum-cardinality proofs via PyXAI on Linux/Modal, including Random-Forest ensemble payloads (majority-vote compilation).
- Region-level image interventions to close the causal gap.
- Channel→concept mapping validation to replace channel-speak with attribute-speak in NLX — only mappings that pass a predefined reliability threshold get promoted.
- The reviewer-analysis battery already scripted: GOSDT/RuleFit/RF scored in "proof-size currency," the 312-attribute ceiling, Matryoshka SAE abstraction levels, prototype-similarity trees, and validated tree search over the accuracy-vs-reason-size frontier.

### Epilogue — the anecdote (optional, for lab meetings)

The repo itself proved the provenance point: the data layer silently vanished from git history (the `data/` gitignore rule meant for datasets matched `src/grounded_vision/data/`), and we recovered a snapshot from the Modal image *because the image recorded exactly what was deployed*. Provenance isn't just an explanation feature — it's how the whole project recovered its own state.

---

## Anticipated questions (short answers)

- **"Trees are weak classifiers, no?"** — Yes, that's the trade: verifiability needs a checkable hypothesis class. The interesting question becomes accuracy-*per-proof-size*, which is exactly what the method-comparison track measures (GOSDT optimal sparse trees, RuleFit, RF).
- **"Why not fine-tune a big model and explain with SHAP?"** — SHAP ranks; it cannot certify sufficiency. Our TreeSHAP comparison quantifies the gap (88.6% overlap — and what SHAP adds beyond the certified reason is unproven).
- **"Is 100% sufficiency trivial?"** — It holds for pure leaves; the verifier flags impure leaves explicitly (the tree-quality analysis counts "vacuous certificate" leaves), and the certified claim is always about the tree as learned.
- **"Does the contrastive mean editing the image works?"** — No, and we say so: it's model-level counterfactual sensitivity. Image interventions are the next experiment.
- **"Why 69.8% on CLEVR?"** — Existential semantics vs per-literal features: the approximation lives in feature compilation; the proof layer stays exact. That separation is a feature of the architecture, not a bug in it.

## Related pages

- [`RESULTS.md`](RESULTS.md) — every number, with provenance
- [`GROUNDED_VISION_FRAMEWORK.md`](GROUNDED_VISION_FRAMEWORK.md) — the system behind the story
- [`ACTIVATION_GROUNDING_WIKI.md`](ACTIVATION_GROUNDING_WIKI.md) — the real-data research log
- [`NLX_EXAMPLES.md`](NLX_EXAMPLES.md) — example sentences to read aloud
