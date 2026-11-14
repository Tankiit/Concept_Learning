# CVPR 2027 — Body draft: related work, method, experiments, limitations, conclusion

**Drafted:** 2026-11-13. Companion to [`PAPER_INTRO_DRAFT.md`](PAPER_INTRO_DRAFT.md) (v3, number-free). Written from the keep list in [`CVPR2027_CLAIM_AUDIT.md`](CVPR2027_CLAIM_AUDIT.md) (final cut 2026-11-11 + the 2026-11-12 replication addendum). Numbers live in the body by design; every table's provenance is listed below. Citation keys are PLACEHOLDERS until the Nov 15 pass (resolved: `canizales` = arXiv:2605.06640).

---

## §2 Related work

```latex
\section{Related Work}
\label{sec:related}

\paragraph{Concept bottleneck models.}
CBMs \cite{koh2020concept} and their post-hoc and embedding variants
\cite{yuksekgonul2023posthoc,zarlenga2022cem} predict through a layer of
human-readable concepts. A substantial literature documents that this layer
does not behave as intended: concepts are leaked or entangled
\cite{margeloiu2021,mahinpei2021}, and models exploit \emph{reasoning
shortcuts}---concept assignments that yield correct labels while being
wrong \cite{marconato2024,bortolotti2024rsbench}. Our audit is
complementary: rather than asking whether the concept layer is faithful,
we ask what survives a formal sufficiency certificate over that
layer---which of its claims hold by construction, and at what rate the
two premises the proof omits (image truth and teacher agreement) die as
the task grows. Our driving-action replication runs on
the rsbench suite \cite{bortolotti2024rsbench} built to expose reasoning
shortcuts, and the literals our audit flags (green light, red light) are
precisely the shortcut-prone concepts that suite identifies.

\paragraph{Formal explanations.}
Abductive explanations (AXps) certify that a set of features provably
entails a classifier's prediction \cite{ignatiev2019abduction}, with
tooling such as PyXAI \cite{pyxai}. A growing line applies such
certificates at the concept level or over partial input spaces
\cite{teso,canizales,soria,huang}. Closest is the behavior-explainer work
of Canizales et al.\ \cite{canizales}, which computes abductive and
contrastive explanations for \emph{behaviors} of vision models by querying
the network head through concept-erasure operators in aligned CLIP space.
Because their oracle is the network itself, fidelity holds by
construction---but exactness is conditional on an empirical head
monotonicity assumption, explanations come in families of hundreds per
image, and the truth of the cited concepts is graded by an LLM judge
rather than measured, with spurious certificates such as \{Fairy(+)\}
visible in their own analysis. Our audit is the mirror image: we certify
against a symbolic head where exactness is unconditional and reasons are
essentially unique, and we spend the saved effort measuring the premise
their setting cannot express. We do not claim novelty for concept-level
formal explanations; the contribution is the measurement of what
survives the transfer.

\paragraph{Trust, calibration, and abstention.}
Selective prediction and conformal methods gate predictions on confidence
\cite{geifman2017selective,angelopoulos2023conformal}. Our abstention rule
inherits this idea but applies it per-\emph{literal} over a certified
reason, with polarity-aware per-concept calibration fit on a small
validation split; we show the discrete alternative---per-concept
reliability statistics---fails to identify unreliable reasons on every
dataset we test, and that whether calibration beats raw confidence is a
property of reader calibration, not of the rule.
```

## §3 Method — what survives (v3, survival reframe 2026-11-14; formal machinery unchanged from v2)

```latex
\section{What Survives a Sufficient Reason}
\label{sec:method}

A sufficient reason is proved, not measured. This paper treats its
journey toward the image as a survival question: which claims hold by
construction, which hold only with a probability we can estimate, at
what rate the measured ones die as the task grows, and whether any
deployment rule restores them. Figure~\ref{fig:pipeline} summarises the
instrument. We fix notation and the two model families
(\S\ref{sec:setup}); define certified reasons and how we verify them
(\S\ref{sec:cert}); arrange the claims between proof and image as a
survival ladder of exact and measured rungs (\S\ref{sec:ladder}); define
the survival probabilities and the vocabulary bound that caps them
(\S\ref{sec:measured}--\S\ref{sec:ceiling}); and define the trust
channels that might restore survival, together with the pre-registered
test a channel must pass to count as a repair rather than noise
(\S\ref{sec:abstention}).

\subsection{Setup and notation}
\label{sec:setup}
Let $x \in \mathcal{X}$ be an input and
$\mathcal{C} = \{c_1, \dots, c_F\}$ a vocabulary of human-readable
concepts. A \emph{reader} $r_i : \mathcal{X} \to [0,1]$ scores each
concept; thresholding yields concept bits
\begin{equation}
\phi_i(x) = \mathbb{1}\!\left[\, r_i(x) \geq \tfrac12 \,\right],
\qquad \phi(x) \in \{0,1\}^{F}.
\label{eq:bits}
\end{equation}
A Boolean head $c : \{0,1\}^{F} \to \{1, \dots, K\}$ consumes the
bits and predicts a class, and a neural teacher $f : \mathcal{X} \to
\{1,\dots,K\}$ predicts directly from the input. We audit two ways of
obtaining $c$: \emph{Setting~T}, a decision tree trained on predicted
bits, and \emph{Setting~D}, a tree distilled to match $f$. Readers are
trained from images; nothing in what follows depends on how they were
learned. Ground-truth concept annotations
$g(x) \in \{0,1\}^{F}$ are held out and used only for measurement.

\subsection{Certified reasons}
\label{sec:cert}
\begin{definition}[Sufficient reason]
A partial assignment $R \subseteq \{1,\dots,F\} \times \{0,1\}$ is
\emph{sufficient} for prediction $\hat{y}$ if every completion of the
unassigned bits preserves it,
\begin{equation}
\forall z \in \{0,1\}^{F}: \; \bigwedge_{(i,b) \in R} z_i = b
\;\Longrightarrow\; c(z) = \hat{y},
\label{eq:sufficient}
\end{equation}
and \emph{minimal} if no literal can be removed while preserving
\eqref{eq:sufficient}. The literals of $R$ are the \emph{cited
concepts} of the explanation.
\end{definition}
Sufficiency here is exact: for a tree, condition~\eqref{eq:sufficient}
reduces to reachability of leaves under a partial assignment, which we
check by exhaustive enumeration along the decision path plus backward
elimination of literals. This native extractor is the paper's authority;
we cross-check it against PyXAI~2.0.1~\cite{pyxai} on the full audit and
a larger certified-literal pool (prediction match and mutual sufficiency
audits agree on every instance; reason \emph{sets} differ in minimality
choice only---App.~\ref{app:pyxai}). Crucially, \eqref{eq:sufficient}
quantifies over bit completions, not over inputs: it is a statement about
$c$, and the two premises it needs about the world---that the cited bits
are true of the image, and that $c$ agrees with $f$---are exactly what it
cannot express.

\subsection{The survival ladder}
\label{sec:ladder}
Certified explanations are consumed as if the proof covered the image
and the network. We make each intervening claim explicit as a ladder,
each rung either \emph{exact}---it holds by construction, for every
input---or \emph{measured}---it holds with a probability the audit
estimates:
\begin{equation}
\underbrace{\text{proof}}_{\text{exact}}
\;\to\;
\underbrace{\text{robustness}}_{\text{exact}}
\;\to\;
\underbrace{\text{fidelity}}_{\text{measured}}
\;\to\;
\underbrace{\text{premise support}}_{\text{measured}}
\;\to\;
\underbrace{\text{image truth}}_{\text{measured}},
\label{eq:ladder}
\end{equation}
where \emph{proof} is \eqref{eq:sufficient}, \emph{robustness} is
invariance of the certificate under predicate perturbations,
\emph{fidelity} compares $c$ with $f$, and \emph{premise support} and
\emph{image truth} compare the cited literals with $g(x)$. Rungs one
and two always survive; that is what a certificate is. Everything the
user actually cares about lives on the measured rungs, and nothing
forces them to survive with the proof---so ``is this explanation
trustworthy'' decomposes into three survival questions, each with its
own rate. The rest of the paper measures those rates, where they
concentrate, and what restores them.

\subsection{Measured quantities}
\label{sec:measured}
For a test instance $x$ with minimal reason $R(x)$ we define
\begin{align}
\text{support}(R) &= \mathbb{1}\!\left[\, \bigwedge_{(i,b) \in R(x)}
g_i(x) = b \,\right],
\label{eq:premise} \\
\text{fidelity}(R) &= \mathbb{1}\!\left[\, f(x) = c\big(\phi(x)\big)
\,\right],
\label{eq:fidelity}
\end{align}
and report their expectations---the survival rates of the measured
rungs---as \emph{premise support}
$P(\text{support})$---the probability that every cited literal is true
of the image---and \emph{conditional fidelity}
$P\big(f = c \,\big|\, \phi(x') \vDash R\big)$, the teacher's
agreement restricted to inputs whose bits satisfy the reason. The
\emph{joint} event is reported directly rather than as a product of
marginals: a marginal guarantee does not certify each explanation.

\subsection{Vocabulary ceiling}
\label{sec:ceiling}
Low premise support conflates two causes: readers that misread the image,
and a vocabulary that cannot express the task. We separate them by
training heads of one matched family on predicted bits
$\phi(x)$, on continuous reader scores $r(x)$, and on ground-truth bits
$g(x)$. The accuracy gap between the ground-truth head and the predicted
heads bounds what any Boolean head over this vocabulary can achieve: if
the gap is wide, the task itself does not survive this vocabulary and
repair effort belongs there---not in the head or the readers.

\subsection{Trust channels and the repair criterion}
\label{sec:abstention}
Premise failures concentrate in identifiable literals
(\S\ref{sec:audit}), so the deployment question is whether a reason can
be \emph{gated} before it is trusted. For each concept $i$ we fit an
isotonic calibrator $q_i : [0,1] \to [0,1]$ on a small validation split,
mapping the reader's probability to the probability that the concept is
truly present. A reason $R$ is scored by its weakest asserted literal,
\begin{equation}
s(R) = \min_{(i,b) \in R}
\begin{cases}
q_i\big(r_i(x)\big), & b = 1, \\[2pt]
1 - q_i\big(r_i(x)\big), & b = 0,
\end{cases}
\label{eq:score}
\end{equation}
and the deployment abstains when $s(R) < t$, sweeping $t$ to trace a
coverage--support curve. We compare three channels for $s$: \textbf{B},
the calibrated score~\eqref{eq:score}; \textbf{C}, the raw reader
confidence $\min_{(i,b)} r_i(x)$ or its complement; and \textbf{A},
discrete per-(concept, polarity) reliability estimated on validation.
Random ordering at matched coverage is the baseline. The calibrator uses
no test information; the honest-deployment arm is~B.

\paragraph{When is a channel a repair?}
The coverage--support curve is summarised at each class count $K$ by the
\emph{repair ratio}
\begin{equation}
\rho_K(c) \;=\;
\frac{P\big(\text{support}(R) \,\big|\, \text{accepted at coverage } c\big)}
     {P\big(\text{support}(R)\big)},
\label{eq:repair}
\end{equation}
the multiple by which gating lifts premise support above the
unconditional rate $R_K$ measured in the same run. A channel is called a
repair at $(K, c)$ only when the lower bound of a cluster-bootstrap
interval on $\rho_K(c)$ exceeds $2$---the smallest multiple separable at
audit sample sizes and the smallest a practitioner would call a repair
rather than noise. Margins below that bar, or margins that hold at one
coverage and not another, are reported as what they are: a tradeoff
degrading toward the unconditional rate, not a guarantee restored. The
criterion, and the decision each outcome triggers, were fixed before the
scaling runs executed (App.~\ref{app:prereg}).

\paragraph{Ranking, forcing, and selection.}
Two further distinctions keep the survival claim honest. First, a
stricter gate---accept a reason only if the literals passing the
threshold themselves \emph{force} the prediction---accepts the same
instances as the ranking gate at every threshold and class count we
measure (\S\ref{sec:exp-abstain}), so ranking is the operating rule and
forcing a diagnostic. Second, channels fail in two independent ways:
\emph{selection} (which instances they accept) and \emph{support given
acceptance} (whether what they accept is true). The two can dissociate
at the same acceptance rate, so every channel is read at matched
acceptance, separating selection death from support death
(\S\ref{sec:exp-abstain}).
```

## §4 Experiments (v2, CVPR-style rewrite 2026-11-13)

```latex
\section{Experiments}
\label{sec:audit}

We evaluate on three datasets with per-image concept ground truth---CUB
birds (the main audit), CelebA faces, and the BDD-OIA driving-action
benchmark---and answer four questions, each a survival question about a
certified reason. Does premise support survive the growth of the task
(\S\ref{sec:exp-scale})? Where does the failure concentrate
(\S\ref{sec:exp-conc})? Can a gate restore it, and at what cost
(\S\ref{sec:exp-abstain})? Do stronger components or smarter selection
rescue it (\S\ref{sec:exp-nosec})? Context rungs---ceilings,
fidelity, interventions, and a concept-agnosticity control---follow in
\S\ref{sec:exp-context}, the replications in \S\ref{sec:exp-repl},
which of the CUB nulls transfer to a second bed in
\S\ref{sec:exp-transfer}, and a comparison with other explanation
methods and model families in \S\ref{sec:exp-methods}.

\subsection{Experimental setup}
\label{sec:exp-setup}
\paragraph{CUB (main audit).}
We use DINOv3 features with \emph{oracle part localisation}, labelled as
such throughout: readers never see predicted boxes. A vocabulary of
$F{=}36$ part--attribute literals is read by logistic probes over pooled
region features, calibrated by validation-carved isotonic regression. The
teacher is a linear head over concept bits; Boolean heads come in two
configurations, \emph{Setting~T} (tree trained on predicted bits) and
\emph{Setting~D} (tree distilled from the teacher). We use official
train/test splits only and sweep the task difficulty by class
count---four, ten, twenty, fifty, and the full two hundred
classes---auditing $n{=}118$, $243$, $515$, $600$, and $600$ certified
test instances respectively (the fifty- and two-hundred-class audits
subsample $1{,}389$ and $5{,}794$ test images; the four-class subset is
fixed, with five random re-draws in App.~\ref{app:subset}).

\paragraph{Replications.}
The identical premise-support measurement runs on two further beds,
changing domain, features, and task type while holding the audit fixed.
\emph{CelebA}: binary \textsc{Smiling}\ target over forty attribute
concepts, official identity-disjoint splits ($17{,}134/6{,}623/6{,}654$
images), our DINOv2 features. \emph{BDD-OIA}, from the rsbench suite
\cite{bortolotti2024rsbench} built to expose reasoning shortcuts: five
driving actions over twenty-one traffic/obstacle concepts, using their
preprocessed features and splits (the fifth action, with negligible base
rate, yields no positive predictions and is
excluded).\footnote{$0.3\%$ base rate; $n{=}0$ certified reasons.}

\paragraph{Protocol.}
All runs use seed $42$, depth-$4$ trees, Wilson intervals, and
random-ordering baselines at matched coverage. CUB is the fully audited
bed---multiple settings, scales, seeds, and controls; the replications are
single-seed, single-depth evidence that the phenomenon is not an artifact
of one pipeline.

\subsection{Premise support does not survive scale}
\label{sec:exp-scale}
Table~\ref{tab:scaling} reports accuracy, premise support, and the joint
event across the class sweep. Two facts stand out. First, premise support
sits \emph{below} accuracy at every scale: the certified explanation is
less likely to be fully correct than the prediction it certifies. Second,
it collapses as classes are added---from a quarter of instances on the
four-class subset to one in fifty-nine on the full benchmark---while
accuracy degrades far more gracefully. The two intermediate counts,
measured under criteria fixed before the runs, make the decay smooth
rather than a cliff: support roughly halves with each step
($0.254 \to 0.070 \to 0.035 \to 0.017$). Conditional fidelity is partial
($0.814$ at four classes, $0.613$ at ten) but is not the binding premise:
the joint event at two hundred classes ($0.8\%$) is dominated by premise
failure.

\begin{table}[t]
\centering\small
\caption{\textbf{Premise support vs.\ accuracy} (Setting~T, official
splits). $P(\text{all}|R)$: fraction of certified reasons whose literals
are all true of the image. Joint: support and teacher agreement together.
CIs: Wilson. $^{\dagger}$audited subsample of the full test set.}
\label{tab:scaling}
\begin{tabular}{lcccc}
\toprule
classes & $n$ & accuracy & $P(\text{all}\,|\,R)$ & joint \\
\midrule
4  & 118  & 0.60 & 0.254 [$0.184,0.340$] & --- \\
10 & 243  & 0.44 & 0.115 [$0.081,0.161$] & --- \\
20 & 515  & 0.42 & 0.070 [$0.051,0.095$] & 0.037 \\
50$^{\dagger}$ & 600 & 0.30 & 0.035 [$0.023,0.053$] & 0.022 \\
200$^{\dagger}$ & 600 & 0.19 & 0.017 [$0.009,0.031$] & 0.008 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Failure concentrates in (concept, polarity) slots}
\label{sec:exp-conc}
The K=4 failures are not spread across the vocabulary---they are carried
by a single literal. All $88$ failing instances cite
\emph{bill\_colour{=}yellow} present, whose annotation support is
$0/88$; inspecting the reader, it fires on beak regions containing
$3.3\%$ yellow pixels on average---a visual shortcut, not a threshold
artifact. The $30$ passing instances cite only \emph{absent}-valued
literals. Nor is the asymmetry a base-rate effect: positive literals'
precision exceeds their occurrence base rate by $+0.03$ to $+0.13$ across
scales (negative: $+0.06$/$+0.07$). The concentration replicates across
beds with the polarity direction set by the domain: on BDD-OIA the weak
slots are \emph{positive} traffic literals---\emph{green\_light}$=$1
appears in $1{,}498$ reasons at support $0.615$,
\emph{red\_light}$=$1 at $0.522$, \emph{follow}$=$1 at $0.436$---while
negatives sit near ceiling (\emph{clear}$=$0: $0.931$); on CelebA they
are the \emph{negations} of two face attributes. Repair, then, has a
target: specific (concept, polarity) slots, not the model at large.
The one-literal story, however, is a property of this subset: five
random four-class re-draws (App.~\ref{app:subset}) keep premise support
in a band around the committed value ($0.267$--$0.431$) and keep failure
concentrated on a single literal ($0.62$--$0.94$ of failing reasons),
but the carriers are \emph{negated} clauses with high marginal support
($0.90$--$0.98$)---the mirror image of the zero-support yellow-bill
literal---so what transfers is the concentration, not the slot.
Failures are also \emph{positively correlated}---observed joint support
exceeds the independence prediction by $1.52\times$ [$1.23,1.89$] at ten
classes (App.~\ref{app:clust})---so compounding over-predicts the decay
rather than explaining it.

\subsection{Abstention: what a gate restores, and at what cost}
\label{sec:exp-abstain}
Table~\ref{tab:abstain} and Fig.~\ref{fig:abstain} compare the three
gating channels of \S\ref{sec:abstention} at matched coverage. On CUB,
calibrated minimum confidence (B) recovers $0.697$ [$0.52,0.85$] premise
support at $13.6\%$ coverage---statistically indistinguishable from an
oracle that knows each literal's true support ($0.727$)---while raw
confidence (C) selects at $0.030$, \emph{below} the random baseline
($0.123$): reader confidence is not merely uninformative about image
truth, it is anti-informative. The pattern reverses in degree but not in
kind on the replications: B lifts support strongly on both (CelebA
$0.957$ [$0.942,0.968$] at $14\%$ coverage; BDD-OIA $0.723$
[$0.674,0.767$] at $5.5\%$), but there raw confidence tracks or exceeds
the calibrated rule ($0.946$ and $0.817$)---their readers are already
well calibrated, so calibration adds little. The verdict across three
datasets: the \emph{rule} (gate on the weakest asserted literal)
transfers everywhere; the \emph{diagnosis} (calibration is load-bearing)
holds exactly where readers are demonstrably miscalibrated. The discrete
channel (A) never separates failure from success on any bed. And nothing
survives the full class count: at two hundred classes the oracle filter
itself retains only $0.125$ at $1.3\%$ coverage. Judged by the repair
criterion fixed before the scaling runs (\S\ref{sec:abstention}: a gate
repairs only if the lower bound of its support multiple clears a
doubling of the unconditional rate), the calibrated gate survives deep
into the sweep---at twenty classes it holds $0.361$ [$0.24,0.47$] at
$14\%$ coverage against a bar of $0.140$; at fifty, $0.143$
[$0.07,0.23$] against $0.070$, its lower bound sitting at the bar---
while at the full benchmark the margin is fragile: $0.095$
[$0.04,0.15$] against a bar of $0.033$, clearing it at some operating
points and not others. We re-checked the join and the literal marginals
before reporting that case (App.~\ref{app:prereg}); by the criterion's
own language it is a tradeoff thinning toward the unconditional rate,
not a guarantee restored. Finally, the earlier fixed-threshold trust
read-out, whose kept-true rates of $0.067$--$0.133$ sat far below
these curves, is not a counterexample but a diagnosis: requiring the
trusted literals themselves to \emph{force} the prediction accepts the
same instances the ranking gate accepts (at every threshold and class
count we measure), so the gap is the channel---at matched coverage, the
calibrated channel keeps fully supported literal sets ($30/30$ at the
four-class operating point) where the per-instance isotonic channel
keeps $0.067$--$0.133$. Selection death, not support death.

\begin{table}[t]
\centering\small
\caption{\textbf{Premise support at matched coverage.} Channels:
A discrete per-(concept, polarity) reliability, B calibrated
min-confidence (honest deployment), C raw min-confidence, random. Per-dataset
operating points; full curves in Fig.~\ref{fig:abstain}.}
\label{tab:abstain}
\begin{tabular}{llcccc}
\toprule
dataset & coverage & A & B & C & random \\
\midrule
CUB D/10  & 13.6\% & 0.727$^{*}$ & \textbf{0.697} [$0.52,0.85$] & 0.030 & 0.123 \\
CelebA    & 14.2\% & 0.628$^{\S}$ & \textbf{0.957} [$0.942,0.968$] & 0.946 & 0.693 \\
BDD-OIA   & 13.6\% & 0.581$^{\S}$ & 0.755 & \textbf{0.817} & 0.535 \\
\bottomrule
\multicolumn{6}{l}{\footnotesize $^{*}$oracle marginal-support filter;
$^{\S}$A at its own low-coverage operating point (8.8\%).}
\end{tabular}
\end{table}

\subsection{No selection rule escapes the premise failure}
\label{sec:exp-nosec}
If failure concentrates in slots, perhaps better components avoid it. They
do not. \emph{Oracle selection}: the best of two hundred reader
configurations improves the joint rate by exactly zero. \emph{Stronger
readers}: raising reader accuracy from $0.602$ to $0.678$ leaves premise
support unchanged ($25.4\% \to 24.6\%$)---accuracy and image-truth
decouple. \emph{Curated training}: retraining only on instances whose
reasons certified leaves support unchanged as well.
\emph{Uncertainty-aware reasons}---sufficiency defined over per-image
completions rather than bit completions---return set-identical reasons
($722/722$). The only high-support reasons that survive at scale are
single clauses citing the \emph{absence} of a concept ($\le 2\%$ of
instances): explaining by what is not there. The levers that move premise
support are abstention and its coverage cost---nothing else we tested, on
CUB; \S\ref{sec:exp-transfer} tests which of these nulls survive on a
second bed.

\subsection{Context rungs}
\label{sec:exp-context}
Three controls situate the failure. \emph{Vocabulary ceiling}: heads of
matched family reach $0.627$ on predicted bits, $0.746$ on continuous
scores, and $0.805$ on ground-truth bits---the vocabulary can express the
task, so the bottleneck is neither expressiveness nor head capacity.
Conversely, a tree trained directly on ground-truth bits collapses to the
majority ($0.254$/$0.03$ at four/two hundred classes): ground-truth bits
without learned readers yield no usable head, so the reader layer is
indispensable even as it fails. \emph{Fidelity and interventions}:
certified reasons perturb the teacher's margin like random
size-matched perturbations (retention $-6.71$ vs.\ $-6.25$; necessity
$-0.39$ vs.\ $-0.55$), and discrete interventions sit at their chance
floor---the certificate neither tracks nor manipulates the network.
\emph{Concept-agnostic control}: running the identical certification
machinery over non-semantic PCA banks yields graded agreement
$0.612/0.596/0.720$---the machinery certifies whatever bank it is given,
which is precisely why premise support, not the machinery, is where
semantics enters.

\subsection{Replications}
\label{sec:exp-repl}
\emph{CelebA} (readers $0.890$): teacher and tree reach
$0.879$/$0.875$ (agreement $0.968$) with reasons of mean length $2.02$;
premise support $0.693$ [$0.682,0.704$] over $n{=}6{,}654$ certified
reasons---binding by nineteen points. \emph{BDD-OIA} (readers $0.883$):
teacher/tree $0.69$--$0.75$ per action; pooled premise support $0.534$
[$0.522,0.546$] over $6{,}433$ reasons, per action $0.472$
(\textsc{forward}), $0.596$ (\textsc{stop}), $0.198$ (\textsc{left},
$n{=}81$), $0.558$ (\textsc{right}). Both rows are single-seed evidence
(\S\ref{sec:exp-setup}); CUB remains the fully audited bed. The next two
subsections push the BDD-OIA bed further: first the CUB intervention
battery, then other methods and models.

\subsection{Which CUB nulls transfer?}
\label{sec:exp-transfer}
Table~\ref{tab:transfer} re-runs the interventions of
\S\ref{sec:exp-nosec} and controls of \S\ref{sec:exp-context} on
BDD-OIA, and the result splits the ``nothing escapes'' finding in two.
\emph{Certificate-level escape attempts fail identically}: exhaustive
oracle selection over every sufficient subset of the decision path gains
$0.0003$; uncertainty-aware reasons are set-identical on $98.3\%$ of
instances (mean quality gain $0.001$); and the fixed-threshold
trusted-sufficient query keeps fully-true literal sets in only $0.090$ of
covered instances at full coverage---within the $0.067$--$0.10$ band we
measure on CUB. No rule that re-reads or re-selects at deployment time
recovers image truth on either bed. \emph{Training-side nulls reverse}:
stronger readers lift support $0.534 \to 0.628$; retraining with the
vocabulary restricted to certified concepts lifts pooled support to
$0.786$ at twelve of twenty-one concepts (though the certificate
population reshapes---the \textsc{forward} tree certifies more instances
with longer, better-supported reasons, while \textsc{stop} collapses to
single-clause reasons); and the ground-truth-bits tree does
\emph{not} collapse ($0.987$/$0.986$ on \textsc{forward}/\textsc{stop}
vs.\ CUB's $0.254$ majority collapse), with a vocabulary ceiling near
$0.95$. The two beds fail on opposite sides of the same audit: on CUB the
vocabulary itself is insufficient and nothing perceptual helps; on
BDD-OIA the concepts can express the task and the failure is almost
entirely perceptual---consistent with the reasoning-shortcut diagnosis
that motivated the benchmark. Two further measurements sharpen the
picture. Premise errors sit \emph{at} the independence prediction on
BDD-OIA ($1.003$ [$1.00,1.01$] vs.\ $1.52\times$ [$1.23,1.89$] on CUB),
so error clustering is bed-specific (App.~\ref{app:clust}). And as a
positive control, flipping ground-truth concept bits i.i.d.\ at
$\epsilon \in [5\%,30\%]$ moves measured support exactly as the
corresponding noise model predicts (agreement within $0.002$ at every
$\epsilon$): the metric responds to annotation noise, and to nothing
else we injected.

\begin{table}[t]
\centering\small
\caption{\textbf{Which CUB results transfer to BDD-OIA.} Certificate-level
nulls replicate; training-side nulls reverse; the two beds fail on
opposite sides (CUB: vocabulary-limited; BDD-OIA: perception-limited).
Support = $P(\text{all}|R)$ pooled.}
\label{tab:transfer}
\begin{tabular}{lcc}
\toprule
quantity & CUB & BDD-OIA \\
\midrule
oracle subset-selection gain & $0$ & $0.0003$ \\
uncertainty-aware $\equiv$ ordinary reasons & $722/722$ & $98.3\%$ \\
trusted query: kept-true at full coverage & $0.067$--$0.10$ & $0.090$ \\
\midrule
stronger readers & $0.254 \to 0.246$ & $0.534 \to 0.628$ \\
certified-concept retraining & unchanged & $\to 0.786$$^{\dagger}$ \\
tree on ground-truth bits & collapse ($0.254$) & $0.99$ (no collapse) \\
vocabulary ceiling (GT-bits teacher) & $0.805$ & $\approx 0.95$ \\
\midrule
errors vs.\ independence & $1.52\times$ [$1.23,1.89$] & $1.003$ [$1.00,1.01$] \\
\bottomrule
\multicolumn{3}{l}{\footnotesize $^{\dagger}$certificate population
reshapes; see text.}
\end{tabular}
\end{table}

\subsection{Other explanation methods and model families}
\label{sec:exp-methods}
Two comparisons hold everything but the method fixed on BDD-OIA.
\emph{Explanation methods} (Table~\ref{tab:methods}, same frozen
pipeline, same $6{,}423$ certified instances): certified reasons are the
only method whose output is sufficient by construction (deletion-style
erasure \cite{canizales} keeps $0.67$ bits on average and is \emph{never}
sufficient; top-$k$ global importance and random subsets rarely are) and
the most faithful to the teacher at matched size ($0.635$). Yet random
size-matched literals are \emph{more} often true of the image ($0.682$
vs.\ $0.534$)---because they assert mostly easy negations ($11\%$
positive literals), while sufficiency forces certificates onto rare
positive traffic concepts ($27\%$). Premise support is the price of
informativeness, and no method pays it down: every method's asserted
premises are more often false than its model's predictions.
\emph{Model families} (Table~\ref{tab:families}): we re-train the
benchmark's own architectures on its tensors---a joint MLP with no
bottleneck, an independent CBM whose head trains on ground-truth bits,
and our sequential pipeline---under identical readers and an identical
Boolean student. The independent CBM is worst on every column (accuracy,
exact match, premise support $0.513$): the classic reasoning-shortcut
deployment penalty, measured here as premise failure on the benchmark's
own data. The joint MLP is the most accurate but its post-hoc concept
student agrees with it on only $80.7\%$ of instances---the concept
vocabulary cannot express it---so its higher support is a different,
imperfectly faithful certificate population, not evidence that skipping
the bottleneck repairs premises. The binding gap is uniform: support sits
$14$--$19$ points below each family's own accuracy.

\begin{table}[t]
\centering\small
\caption{\textbf{Explanation methods on the frozen BDD-OIA pipeline.}
pos.: fraction of asserted literals that are positive. Sufficiency:
fraction of explanations that force the prediction. Teacher-flip:
fraction where flipping the asserted bits changes the teacher output.}
\label{tab:methods}
\begin{tabular}{lccccc}
\toprule
method & size & pos. & $P(\text{all}|R)$ & suffic. & flip \\
\midrule
minimal sufficient reason & 2.68 & 0.27 & 0.534 [$0.521,0.546$] & \textbf{1.00} & \textbf{0.64} \\
greedy erasure & 0.67 & 1.00 & 0.699 & 0.00 & 0.30 \\
top-$3$ global importance & 3.00 & 0.22 & 0.578 & 0.37 & 0.59 \\
random $3$ literals & 3.00 & 0.11 & \textbf{0.682} & 0.01 & 0.31 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering\small
\caption{\textbf{Model families on BDD-OIA} (their architectures, their
tensors, seed 42; readers and Boolean student held fixed). Agreement:
student vs.\ family on test. Sequential-T is the audit pipeline of
Table~\ref{tab:transfer}.}
\label{tab:families}
\begin{tabular}{lcccc}
\toprule
family & accuracy & exact-4 & agreement & $P(\text{all}|R)$ \\
\midrule
joint MLP (no bottleneck) & \textbf{0.759} & \textbf{0.437} & 0.807 & 0.622 [$0.611,0.634$] \\
independent CBM (head on GT bits) & 0.707 & 0.316 & \textbf{0.928} & 0.513 [$0.502,0.524$] \\
sequential (ours, distilled) & 0.721 & 0.370 & 0.891 & 0.542 [$0.532,0.553$] \\
sequential-T (ours, audit) & 0.721 & 0.370 & 0.849 & 0.534 [$0.521,0.546$] \\
\bottomrule
\end{tabular}
\end{table}
```

## §5 Limitations, §6 Conclusion

```latex
\section{Limitations}
The audit requires per-image concept ground truth, which is scarce; our
replications span faces and driving actions, and the method transfers
wherever such annotations exist, but the measured rates are properties of
these beds. The main audit uses oracle part localisation; region readers
under predicted boxes can only lower premise support. Replications are
single-seed with a single tree depth. The abstention trade-off survives
into the intermediate class counts and thins at the full benchmark,
where its margin holds at some operating points and not others; we do
not claim a repair there---only a measurement and a gate.

\section{Conclusion}
A formal sufficiency certificate is a statement about the symbolic
model; what survives the trip to the image is a measurable, degrading
quantity. We made the certificate's two unstated premises measurable,
found premise support binding, structured, and identifiable across
datasets, measured how fast it dies as the task grows, and showed what
calibration-gated abstention can and cannot restore---robustly at
intermediate scale, marginally at full scale, always paying in
coverage. The audit is cheap---it requires one trained head, certified
reasons, and held-out concept annotations---and we believe it should
accompany any deployment of certified concept explanations.
```

---

## Provenance per table/claim (body-only numbers)

| Table / claim | Quantities | Artifact |
| --- | --- | --- |
| Table scaling | 25.4 [n=118] / 11.5 [n=243] / **0.070 [n=515] / 0.035 [n=600] (K=20/50, new 2026-11-14)** / 1.7 [n=600 of 5,794] / joint 0.037/0.022/0.8; acc 0.60/0.44/0.42/0.30/0.19 | `cbm_cem_audit.json`, `cbm_cem_audit_K200.json`, `task1_cbcem_K20_50.json`, `guarantee_scaling_cub.json` |
| Table scaling CIs | Wilson intervals computed from the same n/p | computed (arithmetic only) |
| Concentration §4.3 | 88 failures one literal 0/88; 3.3% yellow pixels; 30 passes absent-only; +0.03…+0.13 / +0.06,+0.07 | `independence_gap_cub.json`, `yellow_bill_audit.json` |
| Clustering §4.3 (appendix) | 1.52 [1.23,1.89], cross-fit 1.48; D/K10 and K200 n.s. | `independence_gap_cub.json` |
| Table abstain CUB row | B 0.697 [0.52,0.85] @13.6% (n=33, 23); oracle 0.727; raw 0.030; random 0.123; K200 oracle 0.125 @1.3% | `abstention_curves.json`, `independence_gap_cub.json` (Q8) |
| Table abstain CelebA row | B 0.957 [0.942,0.968] @14.2%; C 0.946 @13.6%; A 0.628 @8.8%; random 0.693 | `results/celeba_audit_smiling.json` (**uncommitted**) |
| Table abstain BDD row | B 0.755/C 0.817 @13.6%; A 0.581; random 0.535; B 0.723 @5.5% quoted in text | `results/bdd_oia_audit.json` (**uncommitted**) |
| §4.5 no-escapes | oracle gain 0 (mlp 5/200); 0.602→0.678 / 25.4→24.6%; certified-only unchanged; UAR≡ORD 722/722; single-clause ≤2% | `cbm_cem_audit*.json`, `certified_only_cub.json`, `x2_uncertainty_reasons.json` |
| §4.6 ceilings | 0.627/0.746/0.805; GT-bits tree 0.254/0.03 | `design_b_v2.json` |
| §4.6 fidelity/margins | 0.814/0.613; −6.71 vs −6.25; −0.39 vs −0.55 | `conditional_fidelity.json`, `design_b_v2.json` |
| §4.6 banks | 0.612/0.596/0.720; 0.331/0.203/0.297 | `bankexp/results/comparison_banks_K4.md` |
| §4.7 replications | CelebA 0.693 [0.682,0.704] n=6,654, readers 0.890, acc 0.879/0.875, agree 0.968, len 2.02; BDD 0.534 [0.522,0.546] n=6,433, readers 0.883, per-action 0.472/0.596/0.198/0.558 | the two audit JSONs (**uncommitted**) |
| Table transfer | oracle 0.0003; UAR 98.3%, gain 0.001; trusted-true 0.090 @ cov 1.0 (CUB band 0.067–0.10 from guarantee_scaling trust rows); stronger readers 0.534→0.628 (readers 0.883→0.902); certified concepts 0.630/0.739/0.786 @ 20/15/12 (reshape: forward n 2,268→3,383 len 4.0, stop len→1, right n→137); GT-tree 0.987/0.986, ceiling ≈0.95 (0.913–0.996); independence 1.003 [1.00,1.01] | `results/bdd_oia_experiments.json` (**uncommitted**) + `guarantee_scaling_cub.json` (CUB band) |
| §4.8 label-noise control | observed 0.485/0.439/0.355/0.279 vs analytic 0.484/0.438/0.354/0.281 at ε=5/10/20/30% (Δ≤0.002) | `results/bdd_oia_experiments.json` stage7 (**uncommitted**) |
| Table methods | ours 2.68/0.27/0.534 [0.521,0.546]/1.00/0.64; erasure 0.67/1.00/0.699/0.00/0.30; topk 3.00/0.22/0.578/0.37/0.59; random 3.00/0.11/0.682/0.01/0.31; n=6,423 shared | `results/bdd_oia_methods.json` part1 (**uncommitted**) |
| Table families | joint 0.759/0.437/0.807/0.622 [0.611,0.634]; independent 0.707/0.316/0.928/0.513 [0.502,0.524]; sequential 0.721/0.370/0.891/0.542 [0.532,0.553]; sequential-T 0.721/0.370/0.849/0.534; conceptizer acc 0.883 | `results/bdd_oia_methods.json` part2 (**uncommitted**) |
| §3.6 repair ratio + per-K verdicts | K20 B@.14 0.361 [0.24,0.47] n=72 vs 2×R=0.140 (R=36/515); K50 0.143 [0.07,0.23] n=84 vs 0.070 (R=21/600); K200 0.095 [0.04,0.15] n=84 vs 0.033 (R=10/600, rule-3 verified) | `task1_abstention_K20_50.json`, `task1_abstention_K200.json`, `task1_verdicts.json` |
| §3.6 ranking≡forcing; selection-vs-support | cov_forcing ≡ cov_ranking and support_kept ≡ minimal ≡ ranking at every δ∈{0.5..0.9}, K∈{4,10}; K4 30/118 support 1.000 vs committed 0.067–0.133 at the same 0.254 coverage | `task3_forcing_sweep.json` + `guarantee_scaling_cub.json` (band) |
| app:subset draws | P(all) 0.267–0.431 (per-draw Wilson); shares 0.62–0.94; carriers negated, marginals 0.90–0.98 | `task2_subset_sensitivity.json` |

## Drop-list compliance

- "100% verified" only as by-construction sanity (§3.2 framing). ✔
- No cross-dataset-invariant phrasing; rates "properties of these beds" (§5). ✔
- No CLEVR, no 0/118 headline, no "6×", no K=200 calibrated-collapse claim — the K=200 sentence cites the **oracle** filter (measured, independence_gap Q8), not the calibrated arm. ✔
- Yellow-bill as worked example only; K=4 hedged ("this subset; subset-sensitivity in appendix"). ✔
- Clustering hedged and appendix-routed. ✔

## Flags before this freezes

1. **Commit** `results/celeba_audit_smiling.json`, `results/bdd_oia_audit.json`, `scripts/celeba_audit_stage_*.py`, `scripts/rsbench_bdd_audit.py` — body cites them (committed-JSON rule).
2. **Trust-reconciliation RESOLVED 2026-11-14** (`task3_forcing_sweep.json`): at matched 0.254 coverage the calibrated selection is 30/30 supported vs 0.067–0.133 in `guarantee_scaling` — the 0.07–0.10 is raw-channel **selection** death, not support death, and forcing ≡ ranking at every δ. The §4.4 sentence should say exactly this; §3.6 now carries the distinction.
3. **Subset-draws EXECUTED 2026-11-14** (SUBSET-SPECIFIC, 0/5 dominated; `task2_subset_sensitivity.json`): replace §4.3's "this subset; App." hedge with the draws' range 0.267–0.431 plus the negated-carrier observation; `app:subset` already rewritten with the executed numbers.
4. Appendix stubs referenced: `app:pyxai` (engine ledger, from REVIEW_VALIDATION_ROADMAP), `app:subset`, `app:clust` (independence table), Fig.~`fig:abstain` (curves from the three JSONs).
5. Setting D K=10 premise support is 0.124 (keep list spine says 11.5 = Setting T; the D value differs by 1pt) — Table reports T; state in caption or add D column if space.
6. BDD-OIA fifth action (base rate 0.003, no positive predictions) is silently excluded — one footnote in §4.7.
7. The §4.4 K=200 sentence ("nothing survives the full class count") says *oracle* — keep precise. The **calibrated** arm at K=200 is now MEASURED (2026-11-14, `task1_abstention_K200.json`): fragile — clears the 2× bar at the 14% operating point only — and §4.4 reports it in exactly those words; never merge the two into one claim.
8. **Commit before citing (rule):** also `results/bdd_oia_experiments.json`, `results/bdd_oia_methods.json`, `scripts/rsbench_bdd_experiments.py`, `scripts/rsbench_bdd_methods.py` — Tables transfer/methods/families cite them.
9. New §4.8/§4.9 tables are single-seed, single-depth (same caveat as §4.7; stated once in §4.1 protocol paragraph — covers them).
10. §4.9 families: LTN/DPL (their neuro-symbolic families) not run — probabilistic-logic engines out of scope for the deadline; if a reviewer asks, the answer is the wiki table + "not ported". Consider one clause in §5 Limitations if space.
11. §4.9 "pos." column is load-bearing: never quote method P(all) without assertion-mix (random beats ours otherwise) — keep the caption sentence.
12. Instance-drop retrain arm (0.521 vs 0.534, unchanged) exists in the JSON but is not in the text — available if a reviewer asks "did you try cleaning training data" (answer: yes, null).
13. **Survival reframe vs registered title:** §3 (v3) is survival-voiced; the title/abstract registered Nov 10 say "auditing". The abstract's closer ("we measure it, localise it, and show what calibration-gated abstention can and cannot repair") already reads as survival; decide whether the voice propagates to §2/§4 headings and whether the portal still allows title/abstract edits before Nov 16.

---

## Appendix stubs (referenced from §3–§4; LaTeX drafts)

```latex
\appendix
\section{Explanation-engine ledger}
\label{app:pyxai}
We extract minimal sufficient reasons natively (path literals plus backward
elimination over the tree, each candidate checked by exhaustive
reachability) and cross-check with PyXAI~2.0.1 \cite{pyxai}. On the main
audit, prediction match and mutual sufficiency audits agree on every
instance ($118/118$; mean reason $2.89$ literals), and a real-PyXAI
re-run on a larger certified-literal pool agrees on $472/472$ sufficiency
checks. Reason \emph{sets} match at $0.84$--$0.97$ across feature banks:
the engines differ in minimality choice, never in sufficiency. Four
PyXAI~2.0.1 defects were isolated and avoided (empty-abstention return,
minimal-reason crash on Linux / deadlock on arm64 macOS for
partially-used feature sets, an $n$\_sufficient query crash, and a
PrimeImplicantCNF crash in exclusion queries); one interface gotcha
(sklearn node identifiers) is documented in our code. The native verifier
is the paper's authority.

\section{Four-class subset sensitivity}
\label{app:subset}
Five random four-class draws (seeds fixed before execution) through the
identical pipeline give premise support $0.267$--$0.431$ (per-draw
Wilson intervals; no draw exceeds one half), against $25.4\%$ for the
audited subset. Failing reasons concentrate on a single literal in every
draw (shares $0.62$--$0.94$ of failing reasons), but the carriers are
\emph{negated} clauses with high marginal annotation support
($0.90$--$0.98$)---the mirror image of the audited subset's positive
literal at $0/88$. The one-literal characterisation is therefore a
property of the audited subset, not a law of four-class problems; the
main text hedges accordingly.

\section{Clustering above independence}
\label{app:clust}
Premise errors are positively correlated, so independent-compounding
\emph{over}-predicts the decay of premise support with reason length.
Bootstrap over instances ($10^{4}$ resamples, marginals re-estimated per
resample; cluster-robust): observed/predicted joint support is $1.52$
[$1.23,1.89$] at ten classes in Setting~T (cross-fitted $1.48$), $1.09$
[$0.99,1.21$] n.s.\ in Setting~D, and $1.24$ [$0.56,2.00$] n.s.\ at two
hundred classes. The four-class comparison is degenerate (failure is a
single literal). We therefore report the clustering as a Setting-T,
ten-class observation, not a universal law.

\section{Pre-registered survival criteria}
\label{app:prereg}
The repair criterion of \S\ref{sec:abstention} and the decisions it
triggers were fixed in the repository before the scaling runs executed:
a graceful multiple at both intermediate class counts leaves the paper's
claims unchanged, while a collapse at either shifts the spine of the
paper to the collapse; a margin that first appears only at the largest
class count triggers a verification protocol (join and literal-marginal
checks, both of which passed) before it may be claimed. The
forcing-versus-ranking equivalence check and the five-draw
subset-sensitivity protocol were pre-registered in the same document.
Verdict artefacts ship with the code (\texttt{results/task1\_verdicts.json},
\texttt{task2\_subset\_sensitivity.json}, \texttt{task3\_forcing\_sweep.json}).
```

**Figure draft:** `scripts/plot_abstention_curves.py` → `tmp/abstention_curves_draft.png` — three panels (CUB D/10 from `abstention_curves.json` `at_coverage` curves incl. oracle A_ir; CelebA; BDD-OIA with A/B/C/random). Camera-ready version moves into the LaTeX tree; data paths verified against the committed JSON schemas.

---

## Changelog 2026-11-13 — §3 rewritten CVPR-style (v2)

Replaced the telegraphic §3 with a formal method section: opening roadmap paragraph with `fig:pipeline` reference; §3.1 setup with notation (readers, bits eq.~\eqref{eq:bits}, head, teacher, Settings T/D, held-out GT annotations); §3.2 `definition` environment for sufficient reasons with the forcing condition as display eq.~\eqref{eq:sufficient}, extraction + engine cross-check, and the "quantifies over bit completions, not inputs" point; §3.3 the ladder as a display equation with exact/measured braces; §3.4 premise support / conditional fidelity / joint as display eqs; §3.5 ceiling; §3.6 abstention with the min-score eq.~\eqref{eq:score} and the A/B/C channel definitions. Label `sec:abstention` kept to match the intro bullets; all refs cross-checked (undefined: none). **New dependency: `fig:pipeline` — a method/overview figure (reader bits → head → certificate → premises → gate) needs making for the LaTeX tree.**

---

## Changelog 2026-11-13 — §4 rewritten CVPR-style (v2)

Setup promoted to its own subsection with three named paragraphs (CUB main audit / Replications with the fifth-action footnote / Protocol stating seed, depth, Wilson, matched baselines, and the single-seed caveat for replications). Each results subsection now has a claim-shaped title and prose that walks its table: §4.2 reads Table 1 column-by-column ("one in fifty-nine" for 1.7%); §4.3 tells the yellow-bill shortcut as a story (3.3% pixels → shortcut not threshold) and ends with the repair-target takeaway; §4.4 leads with the CUB anti-informative-raw finding, then the reversal on replications, then the three-dataset verdict sentence, keeping both hard sentences (K=200 oracle-only; fixed-δ vs ranking reconciliation); §4.5 enumerates four escape attempts with inline emphasis and the "explaining by what is not there" line; §4.6 three named controls with the GT-bits paradox stated explicitly; §4.7 compact replication rows. All labels re-verified (zero undefined); all 17 keep-list quantities verified present unchanged.

---

## Changelog 2026-11-13 (evening) — §4.7 expanded + new §4.8/§4.9 (second-dataset part, after the rsbench suite + methods comparison runs)

§4 roadmap now names three more stops. §4.7 keeps the compact replication rows and hands off to the new subsections. **§4.8 "Which CUB nulls transfer?"** (Table `tab:transfer`): certificate-level escapes replicate on BDD-OIA (oracle gain 0.0003; UAR set-identical 98.3%; trusted query 0.090 at full coverage, inside CUB's 0.067–0.10 band) while every training-side null reverses (stronger readers 0.534→0.628; certified vocabulary →0.786 with the reshape caveat stated in text; GT-bits tree does NOT collapse 0.99 vs 0.254; ceiling ≈0.95) — the vocabulary-limited vs perception-limited regime split, with the independence negative control (1.003 vs 1.52×) and the label-noise positive control (Δ≤0.002 vs analytic) in the same paragraph. §4.5 gained the forward pointer "on CUB; §4.8 tests which nulls survive". **§4.9 "Other explanation methods and model families"** (Tables `tab:methods`, `tab:families`): methods table carries the assertion-mix column (pos.) with the "price of informativeness" reading — random-3 beats our P(all) via easy negations; families table = their architectures on their tensors: independent CBM worst on every column (RS deployment penalty as premise failure), joint MLP best-acc but 0.807 post-hoc student agreement, binding gap uniform 14–19 pts. Provenance map + flags updated (items 8–12: commit the two new JSONs+scripts; single-seed caveat inherited from §4.1; LTN/DLP not-run honesty; never quote method P(all) without pos. column; instance-drop null available for rebuttal).

---

## Addendum 2026-11-14 — SUN row (third replication, OPTIONAL placement)

```latex
% OPTIONAL appendix block; main text keeps CelebA + BDD-OIA. Placement per
% user decision (carved split = protocol deviation; small n; single seed).
\section{Third replication: SUN attributes}
\label{app:sun}
The SUN Attribute Database provides $14{,}340$ scene images with $102$
per-image attributes (three crowd votes, thresholded at $0.5$); it ships
no official split, so we use a category-stratified $70/15/15$ carve
(seed $42$) and label this deviation explicitly. Readers are logistic
probes on DINOv2 features (validation accuracy $0.952$---the strongest
of any bed). Classifying the top scene categories from concept bits:
premise support $0.833$ [$0.720,0.907$] against teacher accuracy $0.917$
(four categories, $n{=}60$) and $0.704$ [$0.612,0.782$] against $0.769$
(ten categories, $n{=}108$). The gap is the mildest of any bed---about
eight points---consistent with the cross-dataset gradient that the
premise gap tracks reader quality ($0.71/0.88/0.89/0.95$ reader accuracy
corresponding to $35/20/19/8$-point gaps). Concentration persists in
weaker form: the generic literal \emph{man-made}$=$1 appears in about
nine of ten reasons at support $0.864$ while scene-defining concepts are
near-ceiling. Calibrated abstention again lifts support ($0.933$ at
$14\%$ coverage vs.\ random $0.704$), with raw confidence comparable, as
expected for well-calibrated readers.
```

Provenance: `results/sun_audit.json` via `scripts/sun_audit.py` (**uncommitted**). Intro wording intentionally unchanged ("two further datasets") under the appendix-placement recommendation; if the user promotes SUN to the main table, update intro setup sentence + finding 1 ("both"→"all three") + scope paragraph ("three datasets"→"four") + §4.7.

---

## Changelog 2026-11-14 (evening) — §3 reframed around survival (v3), grounded in the three executed runs

User steer: "I don't want an audit paper — I want to know what survives." §3 re-voiced end to end around the survival question; the formal machinery (definitions, display equations, labels) is unchanged from v2. Section title now **"What Survives a Sufficient Reason"**; the roadmap paragraph asks which claims hold by construction, at what rate the measured rungs die as K grows, and whether any deployment rule restores them. §3.3 "claim ladder" → **"survival ladder"** (exact rungs always survive; the measured rungs carry rates; "is this explanation trustworthy" decomposes into survival questions). §3.5 ceiling gains the task-does-not-survive-this-vocabulary reading. §3.6 "Calibration-gated abstention" → **"Trust channels and the repair criterion"**, adding (i) the **repair ratio ρ_K(c)** display eq. with the pre-registered 2× bar (graceful vs collapsing; margins below it are "a tradeoff degrading toward the unconditional rate, not a guarantee restored") — this is the quantity §4 tabulates per K from `task1_*`; and (ii) the **ranking ≡ forcing** and **selection-vs-support-death** distinctions from `task3_forcing_sweep.json` (both point to `sec:exp-abstain`). New appendix stub `app:prereg` (pre-registered criteria + verdict artifacts); `app:subset` rewritten from [TO RUN] to the executed SUBSET-SPECIFIC numbers; flags 2–3 marked RESOLVED with their artifacts; new flag 13 (survival voice vs the registered Nov-10 title/abstract — user decision). Provenance table gains task1/task2/task3 rows.

---

## Changelog 2026-11-14 (evening, later) — intro v4 + full-paper survival pass (§2/§4/§5/§6), grounded in the executed runs

User steer: survival framing throughout ("what survives"), intro + all parts written clearly. **§2**: CBM positioning sentence now asks "what survives a formal sufficiency certificate"; formal-explanations closer "what survives the transfer". **§4 roadmap**: the four questions are now survival questions. **§4.1**: class sweep is 4/10/20/50/200 (n = 118/243/515/600/600; 50 and 200 subsample 1,389/5,794; fixed 4-class subset + five random re-draws → app:subset). **§4.2** retitled "Premise support does not survive scale"; tab:scaling gains the K=20/50 rows (0.070 [0.051,0.095] n=515; 0.035 [0.023,0.053] n=600; joint 0.037/0.022) + the smooth-decay sentence (0.254→0.070→0.035→0.017) from `task1_cbcem_K20_50.json`. **§4.3**: re-draws sentence (0.267–0.431; shares 0.62–0.94; negated carriers 0.90–0.98 → what transfers is the concentration, not the slot). **§4.4** retitled "what a gate restores, and at what cost"; gains the pre-registered repair-criterion passage (K20 0.361 [0.24,0.47] vs bar 0.140; K50 0.143 [0.07,0.23] vs 0.070, lower bound AT the bar; K200 0.095 [0.04,0.15] vs 0.033, fragile, join+marginals re-checked → app:prereg) and the RESOLVED reconciliation (forcing ≡ ranking at every δ/K; matched coverage keeps 30/30 under the calibrated channel vs 0.067–0.133 per-instance isotonic → "selection death, not support death"). **§5**: trade-off "survives into the intermediate class counts and thins at the full benchmark" (was "collapses"). **§6**: conclusion opens on survival ("what survives the trip to the image is a measurable, degrading quantity") and grades the gate honestly (robustly at intermediate scale, marginally at full scale, always paying in coverage). Flags 7 updated (calibrated K=200 now measured — keep oracle vs calibrated claims separate); scaling provenance row extended. Intro file updated in parallel (v4, same steer; its changelog lives there).
