# Review validation roadmap

**Source:** external review, 2026-11-01 — four requirements for claiming that tree facts correspond to the right visual properties, in the right regions, and that explanations preserve those facts accurately.
**Verification status:** every technical claim in the review was checked against the code before this page was written. Three were real defects and are already fixed in this repo (with regression tests); the rest are open work items, each with a concrete spec below.

---

## Claim-by-claim verification (what we found)

| # | Review claim | Verdict | Evidence |
| --- | --- | --- | --- |
| 1 | Missing regions become negative attributes | **Confirmed** | `predicates/grounding.py:88` — `attribute_probabilities.get(attr_id, 0.0)` defaults unobserved attributes to probability 0 → emitted as `truth_value=False` facts, with `provenance=None` when the region is absent. No UNKNOWN state exists anywhere in the schema. |
| 2 | Class-label mapping in the verifier | **Confirmed (latent)** | `explanations/tree_verifier.py` — leaf prediction was `int(np.argmax(counts))`, a **position** in `clf.classes_`, compared against caller-supplied **labels**. Correct only when training labels are contiguous 0..K-1 (true for every run so far — e.g. the activation cache's labels are 0–3 — so no committed result is corrupted; non-contiguous labels would silently falsify everything). |
| 3 | Negative relational templates | **Confirmed** | `language/templates.py` — the relational branch of `verbalize_condition` ignored `negated`: a certified **absence** of `above(crown, beak)` verbalized as its **presence**. |
| 4 | Activation pooling is wrong | **Confirmed — the major one** | `scripts/run_activation_tree_grounding.py:14` — on a `[N,384,16,16]` map reshaped to `(N, C, row-block, row-in-block, col-block, col-in-block)`, `mean(axis=(4,5))` averaged over the **entire column axis**: the 16 "tiles" were full-width row strips. Verified empirically (a delta at patch (row 2, col 13) landed at strip index 2, not tile (0,3)). All tile-derived artifacts are invalidated ([`RESULTS.md`](RESULTS.md) §1). |
| 5 | Audit uses token overlap, not entity–attribute binding | **Confirmed** | `language/checks.py` — `allowed_tokens` is a flat word set; `known_parts` is defined but never used; unsupported-claim scanning checks colors/shapes only. "Black wing" passes when the certified fact is `breast:breast_color=black`. Negation is not audited. |
| 6 | Restore the missing data-loading package | **Already done** | Commit `96e0225` (2026-11-01); root cause was the `data/` gitignore rule. See [`GROUNDED_VISION_FRAMEWORK.md`](GROUNDED_VISION_FRAMEWORK.md). |
| 7 | Activation predicates need validated semantic names | **Already policy, needs the validation experiment** | The NLX policy has always kept channel wording (`tile_r3_c2:dino_ch242`) and attaches part names only from visible annotations — but the channel→attribute association study has not been run. |

**Fixed (regression-tested, `tests/test_review_fixes.py`):** #2 verifier maps leaf argmax through `clf.classes_`; #3 relational negation verbalizes as absence; #4 pooling corrected — in **four** scripts, not one: the same wrong-axes expression also sat in `select_grounded_reasons.py`, `compare_shap_grounding.py`, and `add_exemplar_grounding.py` (found while regenerating). **Regenerated 2026-11-02 with an upgraded protocol** — official train/test split, tree fit on train only, evaluation on held-out test (`scripts/run_activation_track_traintest.py`): held-out accuracy 56.8% (4-species; the historical 95% was in-cache), verified rate 100% on test, mean reason 3.79; full 200-class run in [`RESULTS.md`](RESULTS.md) §1. All activation-track artifacts, evidence PNGs, and NLX examples were replaced.

---

## Requirement 1 — define exactly what each symbol means

**Status: open (design + implementation).**

Adopt three-valued predicates:

| State | Meaning | Emission rule |
| --- | --- | --- |
| Present | part visible (or whole-bird) and reader ≥ τ | positive fact + provenance |
| Absent | part visible and reader < τ | negative fact + provenance |
| Unknown | part not observed / reader input missing | **no truth claim**; explicit unknown marker |

Concrete changes:

- `predicates/grounding.py`: when `part_vis < threshold` (part unobserved) or no region exists, emit no attribute fact for that part; instead record an `unobserved` fact (or a `FactStatus.UNKNOWN` field on `Fact`) so downstream knows the difference between "not yellow" and "not seen".
- `reasoning/features.py`: compile three states separably for the tree. Recommended encoding: `absent = −1`, `unknown = 0`, `present = +1`, so a split at `−0.5`/`+0.5` isolates each state cleanly (a 0/0.5/1 encoding would make `x ≤ 0.5` conflate absent and unknown). Alternative: keep binary features and add companion `unknown(part:attr)` bits — pick one, use it in both compiler and templates.
- `language/templates.py` + `checks.py`: unknown must verbalize as "unobserved", never as absence; the audit must treat an "unobserved" claim as faithful only for unknown-marked conditions.
- Semantic vs activation predicates stay formally separate (already the case): activation literals keep channel wording; a semantic name is attached **only** after Requirement 2's association study passes a predefined reliability threshold on held-out images.

## Requirement 2 — establish that the grounding is meaningful

**Status: executed — first pass (2026-11-01).** See [`RESULTS.md`](RESULTS.md) §4 and `results/predicate_grounding_milestone.json` (script `scripts/run_predicate_grounding_milestone.py`). Headline: agreement is high across all 16 recurring predicates (mean 82.8%), and **specificity splits by part** — beak (2.8–4.2× target-vs-control) and belly (2.7–5.2×) predicates pass the ≥1.5× threshold under both edits; breast (1.1–1.7×) and crown (1.1–1.45×) do not. Per the decision rule below: beak/belly grounding is validated and can back certified part–attribute explanations; crown/breast need region-geometry work first (and note the control was the anatomically adjacent part for crown, which *underestimates* its specificity — a distance-matched non-adjacent control is the immediate refinement).

A box attached to a condition is not evidence. For each recurring predicate, on a **disjoint test set**:

1. **Agreement:** compare the reader's prediction with the annotated attribute (CUB part-attribute labels) — report per-predicate accuracy/AUROC, not just the mean. ✓ done (per-predicate accuracy + AUROC in the JSON)
2. **Targeted perturbation:** mask or blur the predicate's region; re-extract DINO features; recompute predicates and prediction. ✓ done (blur + gray-mask)
3. **Control perturbation:** apply the same edit to a matched control region (same area, different part). ✓ done (same-size box at the next visible scope part's center)
4. **Specificity:** Δ(target predicate) must exceed Δ under control, and unrelated predicates must stay comparatively stable. ✓ measured (plus unrelated-part stability and flip rates)

Required pattern, stated once in the paper: *altering the relevant region changes the relevant predicate more than a matched control, while unrelated predicates remain stable.* Masking/blur earns **sensitivity** claims only; a **semantic correction** claim ("changing yellow to black changes the prediction") additionally requires an edit whose attribute change and content preservation are themselves verified.

DINO caveat (reviewer's point, worth quoting): pooling at a location does not establish that the information *comes from* that location — hence controls.

## Requirement 3 — make the certificate and the language agree exactly

**Status: three items fixed today; two open.**

Every explanation must carry the structured record:

```
entity → attribute → value → polarity → evidence → tree condition
```

- **Open — audit by binding, not tokens:** rewrite `ExplanationAuditor` to check (part, attribute, value, polarity) tuples against the certified conditions: a claimed "black wing" fails when the certified tuple is (breast, breast_color, black, +); a "no white belly" claim fails when polarity is positive. Use `known_parts` (currently dead code); audit negation explicitly.
- **Open — structured record in schema:** extend `VerifiedExplanation` (or its `to_dict`) to serialize the full chain above, so the JSON *is* the audit input — no re-parsing of feature-name strings.
- Fixed: verifier class mapping (#2), negative relational templates (#3), pooling (#4, script fixed — regeneration pending), data package (#6).
- Keep deterministic templates for now (reviewer's guidance): a free-form LM adds another component to validate.

## Requirement 4 — demonstrate a benefit from binding

**Status: open (experiment design).**

Held-out CUB evaluation, three controlled comparisons, readers and tree-capacity budget held comparable:

| Comparison | What it tells you |
| --- | --- |
| Flat attributes vs entity-bound attributes | whether binding adds useful information |
| Correct vs shuffled region assignment | whether the proposed grounding matters (not just extra features) |
| Oracle (`--regions gt`) vs predicted regions | how much perception error weakens the system |

Guardrails:

- The flat baseline must not already encode binding through names (`breast_color` contains the part) — add a truly unbound vocabulary variant, or acknowledge the leakage and interpret the delta as *explicit* vs *implicit* binding.
- Report **classification accuracy, predicate correctness, grounding sensitivity, explanation size, and text correctness separately** — a high certificate rate cannot compensate for incorrect visual facts.

## First milestone (narrow, per the review)

**Executed 2026-11-01** — [`RESULTS.md`](RESULTS.md) §4: 16 recurring part–attribute predicates on the disjoint CUB test split, correct-region and control-region perturbations.

**Outcome (mixed, per the review's decision rule):**
- beak + belly predicates (7/16): reliable *and* specific → **expand to relational explanations on this basis**.
- crown + breast predicates (9/16): reliable but not specific → **next work is region geometry and perception** (smaller/steered boxes, distance-matched non-adjacent controls) before any stronger certification claims for those parts.

Immediate refinements queued: non-adjacent distance-matched control regions; rerun with the deployed `AttributeReader` heads (not just probes) once retrained; extend the scope part list.

**Executed as Milestone v2 (2026-11-02)** — [`RESULTS.md`](RESULTS.md) §4: deployed readers + three control families (adjacent / farthest / distance-matched) + 30,264 per-image effect records with bootstrap CIs. Answers to the two open questions:
1. **Does the separation persist under deployed readers?** Yes — same part-family pattern (beak/belly 2.3–4×, crown/breast ~1.3–1.5× under adjacent controls), CIs tight.
2. **Does region geometry explain the crown/breast failure?** Largely yes: with farthest controls, crown probes recover to 2.69–4.35 (4/4 pass) and breast to 1.51–2.76 (5/5) — the v1 failure was mostly adjacent-control contamination; distance-matched controls (same distance, off-part) split the effect into content and distance components.
3. New bottleneck: **reader head quality** — deployed readers agree at 0.691 vs 0.824 for probes; next work is reader training (perception), per the decision rule.

## PyXAI engine verification (2026-11-06) — executed

Question (external review): *does `PyXAIBackend.explain_sufficient` call PyXAI's `sufficient_reason`, or our own solver with PyXAI only as an audit?* **Answer from the code:** it never calls PyXAI — it delegates to the native exact backward elimination over the decision path, certified by the independent exhaustive `TreeSufficiencyVerifier` (`explanations/pyxai_backend.py`, `solver = "native_tree_verifier"`). Real PyXAI calls live only in `scripts/run_pyxai_modal.py` on Linux/Modal, where PyXAI is both engine (`minimal_sufficient_reason`, `sufficient_reason`) and auditor (`is_sufficient_reason` on our native literals).

**API verification (pyxai 2.0.1, introspected on the installed package):** `set_excluded_features`, `preferred_sufficient_reason(method, n, weights, features_partition)` (DT only), `set_features_type`/`activate_theory`, `is_sufficient_reason`, `n_sufficient_reasons` (DT only), `rectify`, and forest explainers all exist as documented — with three caveats found by execution: (i) abstention under exclusions returns an **empty list `[]`**, not the documented `None` (a genuine empty-sufficient reason returns the empty *tuple*); (ii) `minimal_sufficient_reason` crashes (TypeError/IndexError) on trees using a subset of declared binary features — 99/100 instances in our check; (iii) `n_sufficient_reasons` crashes on some instances (enumeration fallback added). `preferred_sufficient_reason` (the weights=-log(reliability) agenda) routes through the same crashing path — blocked on this tree shape in 2.0.1.

**Executed check** (CLEVR relational track, exactly reproduced tree — test acc 69.8%; 100 test instances; [`results/pyxai_exclusion_check_clevr_armA.json`](../../grounded_vision/results/pyxai_exclusion_check_clevr_armA.json), [`..._armB.json`](../../grounded_vision/results/pyxai_exclusion_check_clevr_armB.json)):

| Measurement | Result |
| --- | --- |
| Prediction agreement, converted tree vs sklearn | **100/100** |
| PyXAI audit of native path-restricted reasons (`is_sufficient_reason`) | **100/100 verified** |
| E0 uniqueness (`n_sufficient_reasons`) | **0/51 instances have a unique sufficient reason**; median count 36–95, max 480 — reasons are highly non-unique, so weighted preferred reasons are meaningful and the path-restricted native reason is one of many |
| Global minimum ⊆ native path-restricted reason | 1/100 — the path-restriction gap is real and quantified |
| Arm A: exclude ALL 333 relational features | coverage 0/65 (uniform abstention), 0 leaks |
| Arm B: exclude the ~4.6 relational literals in each instance's own reason | **coverage 35%, abstention 65%, zero excluded-feature leaks**; alternative reasons mean 7.11 literals vs native 5.01 — the coverage–guarantee trade-off, measured |
| Excluded-feature leak check | 0 across both arms — the exclusion guarantee holds as advertised |

Conversion gotcha documented for future payloads: sklearn node ids are **not** preorder positions — payload node lists must be allocated by nid (the repo's `export_pyxai_comparison.py` does this correctly; an append-in-recursion-order export silently scrambles the tree and corrupts predictions).

Net: the reviewer's mapping is real — excluded-features reasons, abstention semantics, and uniqueness counting are standard PyXAI 2.0.1 calls (modulo the two bugs), and our contribution stays where they placed it: deriving the excluded set from grounding evidence (the milestone v2 per-image reliability records are the input) and measuring the coverage–guarantee trade-off (arm B above is the first point on that curve).

---

## Related pages

- [`RESULTS.md`](RESULTS.md) — consolidated quantitative index (all in-repo tracks regenerated 2026-11-02) and the pending server-results list
- [`GROUNDED_VISION_FRAMEWORK.md`](GROUNDED_VISION_FRAMEWORK.md) — architecture and module map
- [`THE_STORY.md`](THE_STORY.md) — talk track with the review-survival arc and the updated claim ladder
- [`../../grounded_vision/tests/test_review_fixes.py`](../../grounded_vision/tests/test_review_fixes.py) — regression tests for the fixes
