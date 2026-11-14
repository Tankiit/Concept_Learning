# Second-dataset candidates for the premise-support audit — executed survey, 2026-11-12

**Question:** which datasets beyond CUB can carry the paper's headline quantity (premise support — are a certified reason's literals true of the image, against per-image ground truth)? All claims below verified locally on 2026-11-12; downloads staged under `grounded_vision/tmp/datasets/` (gitignored scratch — move to the data volume before any run).

**Deciding constraint:** premise support needs per-image concept ground truth. Fidelity and the exactness decomposition run anywhere; the premise-support column only where annotations exist. This killed EuroSAT (label-only) and AWA (class-level attributes) — see [`BEHAVIOR_EXPLAINER_COMPARISON.md`](BEHAVIOR_EXPLAINER_COMPARISON.md) §3.

## The three candidates the user asked to try — status after execution

| | SUN Attribute DB | CelebA | DeepFashion (fine-grained Attr) |
| --- | --- | --- | --- |
| images × concepts | 14,340 × **102** (continuous [0,1], 3 AMT votes) | 202,599 × **40** (binary) + identity | 14,000 train × **26** (binary), +2k val / 4k test partitions |
| verified locally | `SUNAttributeDB/` mats: `labels_cv` 14340×102, `attributes`, `images` | parquet shard 0: 42 cols, JPEG bytes embedded, official splits (162,770/19,867/19,962) | `anno_fine/*.txt`: `train_attr` 14000×26, names = floral/graphic/striped/… |
| usable concepts (pos-rate 0.05–0.95) | **34/102** at thr 0.5 (sparse: mean label 0.081, 6.7 pos/img) | **30/40** (row-group sample; 9.1 pos/img) | **21/26** (balanced: conventional .80, cotton .68, solid .47 … long_sleeve .31) |
| classes for K-scaling | 611 scene categories (from image paths) | identity 10,177 / attribute-task = multi-label | 50 clothing categories (`list_category_cloth`) |
| task fit | scene classification, global concepts — no parts needed | attribute prediction + identity K-subsets | category + attribute prediction |
| images source | official Brown archive (1.7 GB) — **slow (~3 MB/min) but resumable**; HF sun397 mirrors are 40–110 GB webdatasets without attribute-subset paths | HF `flwrlabs/celeba` parquet — fast (~130 MB/min); 4 shards (~1.7 GB) staged | `Anno_fine` folder on Drive is OPEN (modern IDs, all txt files pulled); **`Img/img.zip` (~2.6 GB) still locked** — no resourcekey opens it; needs a browser session or Kaggle mirror |
| extra value | continuous labels → annotation-noise modeling (audit Q5 analog) | identity-disjoint splits; known noisy/spurious attrs ( Necktie↔Male ) — stress-test for calibration story | official train/val/test; most CUB-like material/texture vocabulary |
| effort to first premise-support number | ~1–1.5 days (readers → tree → certificates; images are the long pole) | **~1 day** (bankexp recipe verbatim on global attrs; altbank controls already exist) | ~1 day *after* images land (annotations are ready) |

Plus carried over from the earlier survey: **RIVAL10** (Box link verified live, 481 MB; per-image attribute labels *unverified* until extracted) and **EuroSAT** (staged at `tmp/behavior-explainer/datasets/EuroSAT/`, 27k images — decomposition/fidelity only, premise support impossible).

## Ranking (premise-support audit fit)

1. **CelebA** — cheapest end-to-end: everything downloadable in minutes, schema verified, official identity-disjoint splits, dense attribute stats, and the bankexp pipeline runs on global attributes without parts. The known attribute noise becomes a feature: it stress-tests the calibration-gated abstention story. 40 concepts is a small vocabulary, but K-scaling runs on identity subsets.
2. **SUN Attributes** — structurally the closest replication of the CUB audit (per-image attribute matrix, 611 categories, continuous labels enabling noise-threshold sensitivity). Costs: sparse (34 usable concepts), image download is the bottleneck (~9 h at current rate; resumable — restart with `curl -C -`).
3. **DeepFashion fine-grained** — best-shaped vocabulary (21/26 usable, material/silhouette semantics), official partitions shipped; blocked on images. Unlock path: browser-download `img.zip` from the Drive folder once (then cache), or find a Kaggle mirror of the Cat&Attr `Img` folder. Do not chase Drive keys further — five candidates tried, none open `Img/`.

## Ground rules if one of these runs (post-deadline / rebuttal)

- Same protocol as CUB: official splits only, no in-cache numbers, premise support reported with n and CIs, abstention curves with matched random/raw baselines.
- These are **not** part of the submitted paper's keep list; they are the rebuttal-window answer to "single dataset?" — position as replication of the *audit method*, not as new headline claims.
- Cheapest-first: CelebA full pipeline → SUN (once images land) → DeepFashion (once images unlock).

## CelebA audit — EXECUTED 2026-11-12 evening (first off-CUB premise-support number)

Pipeline: `scripts/celeba_audit_stage_{a,b,c}*.py` (copied from `tmp/celeba_run/`); artifacts `results/celeba_audit_smiling.json` (per-instance dump included). Setup: official identity-disjoint splits from HF `flwrlabs/celeba` (train 17,134 / valid 6,623 / test 6,654 staged images), DINOv2 ViT-S/14 CLS features (MPS), 40 logistic readers (valid acc mean **0.890** [0.688, 0.985]), linear teacher over the 39 non-target bits, target = **Smiling**; sklearn CART depth-4 student; native `bankexp/utils/tree_ops` minimal sufficient reasons (no PyXAI — arm64 deadlock avoided).

| quantity | CelebA (binary, K=2) | CUB K=4 (committed) |
| --- | --- | --- |
| teacher / tree acc | 0.879 / 0.875 (agree 0.968) | 0.627 / 0.661 (agree 0.627) |
| reason length | 2.02 (max 3) | 2.89 |
| **premise support P(all\|reason)** | **0.693 [0.682, 0.704], n=6,654** | 0.254, n=118 |
| binding vs accuracy | 0.693 ≪ 0.879 ✔ | 0.254 < 0.60 ✔ |
| concentration | two slots carry most failures: `High_Cheekbones=0` (n=3,214, support 0.799), `Mouth_Slightly_Open=0` (n=2,749, 0.823); rare slots worse: `Big_Lips=1` 0.441 (n=118), `Rosy_Cheeks=1` 0.613 ✔ | one literal `beak:bill_color=yellow\|1` in all 88 failures, 0/88 ✔ |
| abstention B_cal (val-isotonic min-conf) | **0.922 [0.909, 0.933] at 27% cov; 0.957 [0.942, 0.968] at 14%** (random 0.693) | 0.697 [0.52, 0.85] at 13.6% (random 0.123) ✔ |
| abstention C_raw (raw min-conf) | **0.946 at 13.6% — works too** ✗ differs | ≤ chance (anti-informative) |
| A_discrete (per-(attr, polarity) bit reliability) | **fails**: 0.628 at 8.8% cov, below random | A_cq ≈ 0 ✔ (same failure) |

**What replicates:** premise support binding and below accuracy; failure concentrated in identifiable (concept, polarity) slots (here the *negative* polarity of the two smile-adjacent concepts — CUB's headline slot was a positive literal, so the transferable claim is slot-concentration, not positive-polarity); calibration-gated abstention buys high support at coverage cost, decisively over random; the *discrete* per-concept reliability channel fails on both datasets.

**What does NOT transfer:** "raw confidence is chance-level" — CelebA's readers (logistic on DINOv2, 17k train images) are well-calibrated enough that raw min-conf nearly matches calibrated (0.946 vs 0.957 at ~14%, CIs overlap). Paper consequence: keep the miscalibration framing **CUB-specific** ("where readers are miscalibrated, calibration is load-bearing"); the dataset-independent constructive claim is the calibrated-min-confidence rule itself, not raw-vs-calibrated separation.

Regime differences to state honestly: binary task (K=2) vs 4/10/200-way; global attributes, no oracle localisation; strong readers (0.89) vs CUB's 0.71; agreement high (0.968) vs 0.627. The replication is of the *audit method and phenomenon* across a very different regime — exactly the rebuttal answer to "single dataset?".

**Not yet run on CelebA:** K-scaling (multi-target or identity-subset tasks), joint teacher-support event, per-reader abstention. SUN audit queued behind the image tarball (resume-loop running); DeepFashion blocked on the locked image zip.

## BDD-OIA (rsbench) audit — EXECUTED 2026-11-12 late (second off-CUB number)

**Why this dataset matters beyond a third data point:** rsbench (NeurIPS 2024 D&B, arXiv:2406.10368) is the Teso–Marconato group's benchmark — the exact reasoning-shortcut literature the claim audit names for Q6 and the Nov 15 related-work pass. Running our audit on *their* preprocessed data grounds that comparison empirically.

Pipeline: `scripts/rsbench_bdd_audit.py`; artifact `results/bdd_oia_audit.json` (all 6,433 per-action reasons). Used their Zenodo `BDD-OIA-preprocessed` bundle as-is (2048-d features + 21 binary concept GT + 5-dim action labels, official splits 16,082/2,270/4,572) — their features, their splits, our audit. Concept order from their `CONCEPTS_ORDER`; actions [forward, stop, left, right, fifth_rare(base 0.003, no positive preds)]. 21 logistic readers (valid acc **0.883** [0.730, 0.992]), per-action linear teacher + depth-4 CART over the 21 concept bits, native minimal reasons, premise support vs GT concept bits.

| action | base | teacher | tree | agree | P(all) | n |
| --- | --- | --- | --- | --- | --- | --- |
| forward | 0.55 | 0.748 | 0.716 | 0.843 | 0.472 | 2,268 |
| stop | 0.45 | 0.747 | 0.747 | 0.997 | 0.596 | 1,914 |
| left | 0.26 | 0.702 | 0.730 | 0.724 | 0.198 | 81 |
| right | 0.28 | 0.689 | 0.632 | 0.819 | 0.558 | 2,170 |
| **pooled** | — | — | — | — | **0.534 [0.522, 0.546]** | 6,433 |

**Polarity asymmetry REPLICATES (the CUB pattern, absent on CelebA):** the failing slots are *positive* traffic/obstacle literals — `green_light=1` (n=1,498, support 0.615), `red_light=1` (0.522), `follow=1` (0.436), `clear=1` (0.388), `stop_sign=1` (0.386), `left_obstacle=1` (0.237) — while negative polarities are near-ceiling (`clear=0` 0.931, `rider=0` 0.918). These are precisely the concepts the rsbench paper flags as reasoning-shortcut-prone (e.g. pedestrian/red-light confusion both implying stop): readers fire without the concept being true, and certificates inherit it.

**Abstention (pooled):** B_cal lifts 0.534 → 0.652 (51% cov) → 0.681 (31%) → 0.723 (5.5%); **C_raw lifts more** (0.711 @25%, 0.755 @13.6%, 0.817 @5%); A_discrete plateaus below 0.60. Cross-dataset verdict now three-way: the discrete reliability channel fails everywhere; min-confidence abstention lifts everywhere; *which* confidence channel wins is reader-calibration-dependent (CUB: raw ≤ chance, calibration load-bearing; CelebA: raw ≈ calibrated; BDD-OIA: raw > calibrated). Paper wording: keep "calibration-gated" as the recommended rule, claim calibration-specific superiority only where readers are demonstrably miscalibrated (CUB).

Three-dataset summary (premise support vs accuracy, binding everywhere):

| dataset | task | readers | acc | P(all\|reason) | n |
| --- | --- | --- | --- | --- | --- |
| CUB K=4 | 4-way bird | 0.71 | 0.60–0.66 | 0.254 | 118 |
| CelebA | binary Smiling | 0.89 | 0.875–0.879 | 0.693 [0.682, 0.704] | 6,654 |
| BDD-OIA | 5-action driving | 0.88 | 0.63–0.75 | 0.534 [0.522, 0.546] | 6,433 |

## BDD-OIA experiment suite — EXECUTED 2026-11-13 (the CUB rungs on rsbench, not just the audit)

Pipeline: `scripts/rsbench_bdd_experiments.py` (7 stages, seed 42, official splits, their preprocessed tensors); artifact `results/bdd_oia_experiments.json`. Stage 1 reproduces the committed audit in the ltn venv (sklearn 1.8 vs the original conda env): pooled P(all) **0.5337 vs committed 0.5343 (Δ −0.0006)**, n 6,423 vs 6,433 — ten reasons differ from solver-version drift; all suite numbers below are from this single run.

**Deciding table vs the CUB keep list:**

| keep-list quantity | CUB | BDD-OIA | verdict |
| --- | --- | --- | --- |
| oracle selection gain (exhaustive ≤2^4 subsets) | 0 | **0.0003** (P(any all-true) 0.534 vs ORD 0.534) | replicates |
| UAR ≡ ORD (max-q sufficient subset) | identical 722/722, gain 0 | identical **98.3%**, q gain cal 0.0014 | replicates in substance |
| clustering above independence | 1.52 [1.23,1.89] (T/K=10) | **1.003 [1.00,1.01]** pooled; right action 0.948 (below 1) | does NOT replicate — CUB-only |
| slot concentration | one literal in 100% of failures | top-1 slot (`green_light=1`) in **22.5%**, top-3 in **78.4%** of 2,995 failures | identifiable but distributed |
| present-precision vs base rate | +0.03…+0.13 | **+0.32** (0.561 vs base 0.241); absent +0.08 | lift larger; low base rate is the driver |
| Q8 marginal-threshold literal filter | 0.70 @ 13.6% cov (D/K=10) | in-sample t=0.5: 0.574 @ 54.6% cov; t≥0.9 and cross-fit: **coverage 0** | weak — the threshold filter is CUB-flavoured |
| stronger-reader null | 25.4 → 24.6% | MLP readers 0.883 → 0.902 val acc, pooled P(all) **0.534 → 0.628** | does NOT replicate |
| certified-concepts retrain | unchanged (0.254 all arms) | vocab restricted to val-acc-certified concepts: pooled **0.630 (20 c) → 0.739 (15) → 0.786 (12)**, but the certificate population reshapes (forward n 2,268→3,383 len 4.0; stop collapses to len-1 reasons; right n → 137); acc ~preserved 0.63–0.75 | does NOT replicate (with reshape caveat) |
| instance-drop retrain (drop train images whose premise fails) | — (not run) | pooled **0.521 vs 0.534** — unchanged | new arm, CUB-spirit null |
| tree-at-GT-bits collapse | 0.254 = majority | forward/stop **0.987/0.986** (no collapse); left 0.763 ≈ maj; right 0.864 | does NOT replicate — their concepts are action-sufficient |
| full-vector trusted-only (CUB guarantee_scaling protocol, fixed δ) | 0.067–0.10 P(kept-true\|covered) | global δ0.6: **0.090 @ cov 1.0** (kept 21.0); per-inst δ0.8: 0.409 @ cov 0.50 | replicates at high coverage |
| reason-scoped trusted ranking | B abstention 0.697 @ 13.6% | per-inst δ0.8: **0.717 @ 22.6% cov**; δ0.6: 0.638 @ 61.6% | replicates |
| label-noise positive control (Q5/Q7) | not run | iid flips: observed P(all) 0.485/0.439/0.355/0.279 vs iid analytic **0.484/0.438/0.354/0.281** (Δ ≤ 0.002 at ε = 5–30%) | positive control delivered |

**Ladder (stage 6):** direct 2048-d probe 0.73–0.76 ≈ deployed-bits teacher 0.685–0.751 (the concept bottleneck buys nothing here); GT-bits teacher 0.913–0.996 (mean ≈ 0.954) — the vocabulary ceiling is near-perfect, so on BDD-OIA the gap is almost entirely premises/readers, unlike CUB where the vocabulary itself was insufficient.

**Two supporting calibration facts (stage 3):** calibrated mean q of ORD reasons = **0.5349 vs observed P(all) 0.5337** — the isotonic completion model is honest on BDD (on CUB raw confidence was anti-informative); and 77.4% of ORD reasons contain a literal with calibrated confidence < 0.8 (CUB X2-P1 analog: 100%).

**The refined three-dataset story (for the paper's replication paragraph):**
1. *Universal:* premise support binds below accuracy, and **no certificate-level selection rule escapes it** — oracle subset selection (gain 0.0003), uncertainty-aware reasons (98.3% identical, gain 0.001), fixed-δ full-vector trust (P(kept-true) 0.09 at full coverage, matching CUB's 0.067–0.10).
2. *Regime-dependent — where the failure lives:* CUB is vocabulary/annotation-limited (GT-bits tree collapses, stronger readers and certified vocabulary change nothing, errors cluster above independence); BDD-OIA is perception-limited (GT concepts nearly sufficient at 0.95+, stronger readers +9.4 points, certified vocabulary +25 points pooled, errors at independence). This is consistent with rsbench's own reasoning-shortcut thesis and is the honest answer to "is the null universal?" — the certificate-side nulls are, the training-side nulls are not.
3. *Transferable constructive tool:* calibrated min-confidence **ranking** (works on all three datasets); discrete per-(concept,polarity) reliability channels and marginal-threshold filters fail or cover nothing off-CUB.

Artifacts uncommitted (same status as the two audit JSONs; commit before the paper cites them).

## BDD-OIA methods comparison — EXECUTED 2026-11-13 (other explanation methods + other model families under the same audit)

Pipeline: `scripts/rsbench_bdd_methods.py`; artifact `results/bdd_oia_methods.json` (uncommitted). Grounding for the rsbench paper itself: their main experiments run on SDD-OIA (synthetic BDD-OIA replacement — "BDD-OIA is not suitable for systematically evaluating RSs OOD"); their reported SDD-OIA means are DPL 0.80 / LTN 0.82 / CBM 0.60 mF1(Y) with concept mF1 ≤ 0.61 everywhere (arXiv:2406.10368 Table 5). Our comparison re-trains **their model families on their real BDD-OIA tensors** (seed 42, official splits) so the audit holds everything but the method fixed.

**Part 1 — explanation methods on the frozen sequential pipeline** (same 6,423 certified-positive instances for every method):

| method | size | frac positive literals | P(all) | exact sufficiency on tree | teacher-flip fidelity |
| --- | --- | --- | --- | --- | --- |
| minimal sufficient reason (ours) | 2.68 | 0.273 | 0.534 [0.521, 0.546] | **1.000** | **0.635** |
| greedy erasure (behavior-explainer style, erase-to-0) | **0.67** | 1.000 | 0.699 | 0.000 | 0.303 |
| top-3 global importance (\|teacher coef\|) | 3.00 | 0.220 | 0.578 | 0.366 | 0.585 |
| random 3 literals (matched size) | 3.00 | 0.105 | 0.682 | 0.009 | 0.307 |

Reading: **P(all) is not comparable across methods without the assertion-mix column** — random matched-size literals are *truer of the image* (0.682) than the certificate's premises precisely because they assert mostly easy absent-literals (10.5% positive), while sufficiency forces our reasons onto rare positive traffic concepts (27.3% positive) — the price of informativeness. Deletion explanations are near-empty here (0.67 kept bits: the depth-4 trees' prediction survives almost any erasure) and are never sufficient — they certify nothing on this pipeline. Certified reasons are the only method with exact sufficiency and the highest teacher-faithfulness at matched size; and **no method's asserted premises are mostly true** — the premise failure is not an artifact of our extractor.

**Part 2 — model families (rsbench taxonomy, their architectures, same readers, same Boolean student):**

| family | mean acc (4 actions) | exact match all-4 | tree↔family agreement | pooled P(all) | n |
| --- | --- | --- | --- | --- | --- |
| joint MLP (their `boiann` arch) | **0.759** | **0.437** | 0.807 | 0.622 [0.611, 0.634] | 6,676 |
| independent CBM (their `boiacbm`: head on GT bits, deployed on predicted) | 0.707 | 0.316 | **0.928** | **0.513** [0.502, 0.524] | 7,592 |
| sequential (head on predicted bits, distilled tree) | 0.721 | 0.370 | 0.891 | 0.542 [0.532, 0.553] | 8,105 |
| sequential-T (committed audit, tree on labels) | 0.721 | 0.370 | 0.849 | 0.534 [0.521, 0.546] | 6,423 |

Findings: (1) **the independent CBM is worst on every column** — accuracy, exact match, and premise support — the classic reasoning-shortcut deployment penalty, replicated under our audit on their own data and model family (matches the rsbench thesis and their SDD-OIA CBM result qualitatively); (2) **the joint model wins accuracy but its post-hoc concept student agrees only 80.7%** — the 21-concept vocabulary cannot express it (their NN/TCAV row analog); (3) the binding gap is uniform across families: P(all) sits 14–19 points below each family's own accuracy; (4) the joint model's higher P(all) (0.622) comes with the loosest student, so it is not evidence that skipping the bottleneck fixes premises — it is a different (post-hoc, imperfectly-faithful) certificate population.

## Download / provenance ledger (2026-11-12)

| dataset | command / path | status |
| --- | --- | --- |
| SUN labels | `curl https://cs.brown.edu/people/gmpatter/Attributes/SUNAttributeDB.tar.gz` | ✅ 532 KB, extracted |
| SUN images | `curl -C - https://cs.brown.edu/people/gmpatter/Attributes/SUNAttributeDB_Images.tar.gz` | ⏳ in progress (resumable) |
| CelebA | HF `flwrlabs/celeba` → `img_align+identity+attr/*.parquet` (4 of 19+3+3 shards) | ⏳ in progress |
| DeepFashion anno | Drive `Anno_fine` folder (ID `19J-FY5NY7s91SiHpQQBo2ad3xjIB42iN`, modern, no key needed) | ✅ 10 txt files |
| DeepFashion images | Drive `Img/` (old-style ID `0B7EVK8r0v71pekpRNUlMS3Z5cUk`) | ❌ locked; browser session or Kaggle mirror |
| RIVAL10 | Box `…shared_name=iflviwl5rbdgtur1rru3t8f7v2vp0gww&file_id=f_944375052992` (504,422,700 B verified) | not downloaded (post-deadline option) |
| EuroSAT | Zenodo `7711810/files/EuroSAT_RGB.zip` | ✅ staged into `tmp/behavior-explainer/datasets/EuroSAT/` |

## SUN Attribute DB audit — EXECUTED 2026-11-14 (third off-CUB row)

Tarball saga: the 1.7 GB Brown download completed but a resumed chunk corrupted the stream at exactly 80 MiB; truncated at the clean offset, refilled, gzip-verified, then extracted (16,657 files; all 14,340 indexed images present). Pipeline: `scripts/sun_audit.py`; artifact `results/sun_audit.json` (per-instance, K=4 and K=10). **Protocol deviation, stated everywhere: SUN ships no official split — category-stratified 70/15/15 carve, seed 42** (10,038/2,151/2,151). Readers: 102 logistics on DINOv2 vits14 over the full train carve — **strongest readers of any bed: 0.952** [0.739, 0.997]. Task: classify top-K scene categories from 102 concept bits (K=4: underwater/waterfall/stadium/theater, n_test=60; K=10: n_test=108).

| K | n | teacher | tree | agree | P(all) | reason len |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 60 | 0.917 | 0.900 | 0.883 | **0.833 [0.720, 0.907]** | 3.28 |
| 10 | 108 | 0.769 | 0.528 | 0.620 | **0.704 [0.612, 0.782]** | 3.83 |

**Binding replicates, mildest gap yet (~8 pts)** — consistent with the reader-strength gradient across beds (0.71→0.88→0.89→0.95 readers; gaps 35→20→19→8 pts). Within-bed scaling direction matches CUB (0.833→0.704 as K grows). **Concentration persists but weakly**: the generic slot `man-made=1` (c88) appears in ~90% of K=10 reasons at support 0.864; scene-defining concepts are well-supported (running_water=0: 0.969 n=96, enclosed_area=0: 1.000 n=40); weakest slot `spectating=1` 0.714 (n=14). **Abstention**: B works (K=10: 0.889 @25%, 0.933 @14%, 1.000 @5.6% vs random 0.704); C comparable (0.926/1.000) — readers well-calibrated, matching the three-dataset prediction. K=4 curves noisy (n=60). Note: K=10 tree (depth 4) badly underfits (0.528 vs teacher 0.769) — fidelity gap large, premise support still binding.

**Paper placement recommendation:** appendix or a hedged half-row — carved split (protocol deviation), small n, single seed. Main replication table stays CelebA + BDD-OIA.
