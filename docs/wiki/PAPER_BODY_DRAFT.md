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

## §3 Method — certify, measure, gate (v4, 2026-11-14: the user skeleton, every TODO filled by decision)

```latex
% ---- macros (move to preamble) --------------------------------------
\newcommand{\phix}{\phi(x)}
\newcommand{\Lits}{\mathcal{L}}
\newcommand{\supp}{\operatorname{supp}}
\newcommand{\acc}{\operatorname{acc}}
\newcommand{\Rset}{R}
\newcommand{\trust}{\tau}

\section{Certify, measure, gate}
\label{sec:method}

We certify, then measure, then gate. A sufficient reason's proof is a
statement about the Boolean head; this section separates what that
proof establishes by construction from what it implicitly asserts
about the image, and defines how we estimate the latter. The pipeline
strip of Fig.~\ref{fig:gap} shows the four stages and where the proof
stops; the subsections below map one-to-one onto them.

\subsection{Pipeline: perception, concepts, reasoning, explanation}
\label{sec:pipeline}

\paragraph{Perception.}
An image $x$ is encoded by a frozen DINOv3 backbone into patch tokens
$z\in\mathbb{R}^{N\times d}$. For each part $p\in\mathcal{P}$ we pool
tokens inside a fixed-size box centred on the annotated part location,
giving a region vector $h_p(x)$. Localisation is \emph{oracle}
throughout---boxes come from the annotated part keypoints, never from
a detector---so every measured quantity below is conditional on
localisation; predicted boxes can only lower the reported rates, and
we label this choice wherever it carries weight.

\paragraph{Concept layer.}
A vocabulary of $F{=}36$ part-bound concepts $c_k=(p_k,a_k)$ (e.g.\
\texttt{beak:bill\_color=yellow}) is read by linear readers,
\begin{equation}
  \phi_k(x) = \mathbb{1}\!\left[\sigma\!\left(w_k^\top h_{p_k}(x)\right) > \theta_k\right],
  \qquad \phix\in\{0,1\}^F,
  \label{eq:readers}
\end{equation}
with $\theta_k=\tfrac12$ for all $k$; each reader's probability is
calibrated by isotonic regression fit on a validation carve-out that
no other stage touches. A \emph{literal} is a concept with a polarity,
$\ell=(k,b)$, $b\in\{0,1\}$; $\Lits$ denotes all $2F$ literals.

\paragraph{Reasoning.}
A decision tree $t:\{0,1\}^F\to[K]$ predicts the class from $\phix$
alone, in two settings: \textbf{(T)} $t$ is trained on labels;
\textbf{(D)} $t$ is distilled from a neural head
$g:\{0,1\}^F\to[K]$ over the same concepts, so that $t$ is a surrogate
for the deployed model. Setting~T carries the headline numbers---every
scaling and abstention table is T unless marked---because its premises
are exactly the bits a deployment would read; Setting~D appears where
the surrogate matters (the abstention comparison and the model-family
table) and is routed there.

\paragraph{Explanation.}
For a prediction $c=t(\phix)$ we extract a \emph{sufficient reason}: a
set of literals $\Rset\subseteq\Lits$, each true of $\phix$, such that
\begin{equation}
  \forall\, u\in\{0,1\}^F:\quad u\models \Rset \;\Rightarrow\; t(u)=c,
  \label{eq:sufficient}
\end{equation}
and no proper subset of $\Rset$ has this property. Reasons are
literal-based and subset-minimal over literals; feature-based
minimality, which may drop both polarities of a concept at once,
differs in minimality choice and is relegated to a
footnote.\footnote{Literal- vs.\ feature-based minimality differ only
in which subsets are eligible; sufficiency is identical under both.
The exchange is documented for the PyXAI engine in
App.~\ref{app:pyxai}.} Reasons are computed by a native reachability
engine and cross-checked against PyXAI~\cite{pyxai}; the two agree on
every audited instance ($472/472$ sufficiency checks,
App.~\ref{app:pyxai}). Concept-level sufficient reasons are the same
object extracted by \citet{teso} and \citet{soria}; we do not claim
the object---we audit it.

\subsection{What the certificate proves, and what it does not}
\label{sec:ladder}

Equation~\eqref{eq:sufficient} holds for every bit vector $u$; it says
nothing about whether the cited literals are true of $x$, and nothing
about whether the tree agrees with the deployed network on inputs
whose bits satisfy $\Rset$. Everything Eq.~\eqref{eq:sufficient} needs
lives to the right of the dashed rule in Fig.~\ref{fig:gap}; everything
it silently needs lives to its left. We organise the claims a
certified reason invites into a \emph{survival ladder}
(Tab.~\ref{tab:ladder}): rungs above the dashed rule hold by
construction; rungs below it are properties of the perception stage
and can only be estimated.

\begin{table}[t]
  \centering\small
  \caption{\textbf{The survival ladder.} Read downward: each rung adds
  a claim the rung above it cannot express. Rungs above the dashed
  rule hold by construction for every input; rungs below it are
  properties of the perception stage, estimated in Sec.~4 with the
  section that measures each.}
  \label{tab:ladder}
  \begin{tabular}{@{}llll@{}}
    \toprule
    Rung & Claim about $\Rset$ & Status & Where measured \\
    \midrule
    1 & sufficient for $t$ (Eq.~\ref{eq:sufficient}) & exact & App.~\ref{app:pyxai} \\
    2 & sufficient for all completions of & exact & \S\ref{sec:nulls}, \\
      & untrusted literals                    &        & \S\ref{sec:nulls}, \S\ref{sec:abstention} \\
    \cdashline{1-4}
    3 & $t$ agrees with $g$ where $\phix\models\Rset$ & measured & \S\ref{sec:exp-fidelity} \\
    4 & every premise true of $x$ (support)            & measured & \S\ref{sec:exp-scale} \\
    5 & premises name the intended concepts            & audited  & \S\ref{sec:exp-verify} \\
    \bottomrule
  \end{tabular}
\end{table}

\subsection{Measured quantities}
\label{sec:measures}

\paragraph{Premise support.}
Let $a(x)\in\{0,1,\star\}^{F}$ be the annotation, with $\star$ for
concepts on parts not visible in $x$. A literal $\ell=(k,b)$ is
\emph{supported} if $a_k(x)=b$, \emph{refuted} if $a_k(x)=1-b$, and
\emph{unresolved} if $a_k(x)=\star$. Premise support of a reason is
\begin{equation}
  \supp(\Rset,x) = \mathbb{1}\!\left[\forall\,\ell\in\Rset:\ \ell \text{ supported}\right].
  \label{eq:support}
\end{equation}
\emph{Convention for unresolved literals:} an unobserved part
contributes no positive concepts---its asserted literals are refuted
and its negated literals are supported. Every number in this paper is
computed under this convention; the unresolved fraction reaches one
half of literals at the larger class counts, and the strict
alternative (support on fully-resolved reasons only) is reported as a
sensitivity in the appendix.

\paragraph{The inversion.}
We compare $\mathbb{E}[\supp(\Rset,x)]$ with
$\acc=\mathbb{E}[\mathbb{1}[t(\phix)=y]]$ on the same audited
instances. The comparison is the baseline a reader already holds: an
explanation should be at least as reliable as the prediction it
explains. The inversion---support below accuracy at every class
count---breaks that default.

\paragraph{Independence null.}
With $\hat p_\ell$ the support rate of literal $\ell$ over its
occurrences in reasons,
\begin{equation}
  \widehat{\supp}_{\perp} = \frac{1}{n}\sum_{i=1}^{n}
  \prod_{\ell\in \Rset_i} \hat p_\ell,
  \label{eq:null}
\end{equation}
is the support expected if premise errors were independent given the
literal. A ratio above one means failures share literals---failure is
predictable compounding, and repair has targets; at one, errors are
independent; below one, errors avoid one another. We measure above one
on CUB at ten classes and at one on BDD-OIA, and therefore treat
clustering as bed-specific (App.~\ref{app:clust}) rather than a law.

\paragraph{Fidelity (Setting D).}
$\operatorname{fid}(\Rset)=\Pr[g(\phi(x'))=c \mid \phi(x')\models\Rset]$,
estimated on held-out data with Wilson intervals.

\paragraph{Vocabulary ceiling.}
Low support conflates weak readers with an inexpressive vocabulary.
Heads of matched family trained on predicted bits, on continuous
reader scores, and on ground-truth bits bound what any Boolean head
over this vocabulary can achieve: if the ground-truth head dominates,
the task does not survive the vocabulary and repair belongs
there---not in the head or the readers.

\subsection{Gating reasons by literal trust}
\label{sec:gate}

A \emph{trust channel} assigns each literal occurrence a score
$\trust(\ell,x)\in[0,1]$. A reason is issued only if its weakest
literal clears a threshold,
\begin{equation}
  \operatorname{issue}(\Rset,x) = \mathbb{1}\!\left[\min_{\ell\in\Rset}\trust(\ell,x)\ge\lambda\right].
  \label{eq:gate}
\end{equation}
Channels compared: \textbf{raw confidence} (the reader's probability,
or its complement for negated literals); \textbf{calibrated,
polarity-aware confidence} (the validation-isotonic probability of the
asserted polarity---the honest-deployment arm); \textbf{discrete
per-(concept, polarity) reliability} (validation-estimated rates);
\textbf{random ordering} at matched coverage as baseline, with the
oracle marginal-support filter as a reference upper bound. No results
here; Sec.~\ref{sec:abstention}.

\paragraph{Repair criterion (fixed before the runs).}
A channel counts as a repair at operating point $\lambda$ if the
cluster-bootstrap lower bound of
$\mathbb{E}[\supp\mid\operatorname{issue}]/\mathbb{E}[\supp]$ is at
least $2$---the smallest multiple separable at audit sample sizes and
the smallest a practitioner would call a repair rather than noise. The
criterion, and the decision each outcome triggers, were fixed on
2026-11-11, before the scaling runs executed (App.~\ref{app:prereg}).
Coverage is the price and is always reported:
$\operatorname{cov}=\mathbb{E}[\operatorname{issue}]$; the ratio is
defined at matched acceptance, separating selection death from support
death (\S\ref{sec:abstention}).

\subsection{Protocol}
\label{sec:protocol}

Main audit: CUB with official train/test splits, class counts
$K\in\{4,10,20,50,200\}$, the four-class subset fixed and re-drawn
five times at seeds 1000--1004 (App.~\ref{app:subset}); audited
reasons cover all certified test instances at $K\le 20$ ($n=118$,
$243$, $515$) and subsamples of $n{=}600$ at $K{=}50$ and $200$ (of
$1{,}389$ and $5{,}794$; deterministic by sid order). Replications
with the measurement held fixed: CelebA (official identity-disjoint
splits) and BDD-OIA (rsbench splits); a third (SUN, carved split) is
relegated to the appendix as a protocol deviation. All runs use seed
$42$; intervals are cluster bootstraps over instances ($10^{4}$
resamples) or Wilson where noted. Pre-registered on 2026-11-11,
before execution: the repair criterion, the verdict rules it triggers,
the dominance criterion for subset re-draws, and the
forcing-versus-ranking equivalence check (App.~\ref{app:prereg}).
## §4 What survives (v3, 2026-11-14: the user skeleton, ladder-ordered, every TODO filled)

```latex
\section{What survives}
\label{sec:experiments}

We walk the ladder of Tab.~\ref{tab:ladder} from the exact rungs down.
The proof holds everywhere and certifies whatever concept bank it is
given (\S\ref{sec:exp-verify}); premise support is the rung that fails,
decaying smoothly with scale (\S\ref{sec:exp-scale}); fidelity survives
partially and is not the binding premise (\S\ref{sec:exp-fidelity});
the failure concentrates in identifiable slots
(\S\ref{sec:audit}); the vocabulary, not head capacity, limits the main
bed (\S\ref{sec:ceiling}); the calibrated gate is the only lever and
pays in coverage (\S\ref{sec:abstention}); a pre-registered list of
escape attempts returns nulls (\S\ref{sec:nulls}); and the beds fail on
opposite sides of the same audit (\S\ref{sec:beds}).

\subsection{Setup}
\label{sec:exp-setup}
Everything experimental is as in \S\ref{sec:protocol}; we state here
only what it deferred. The four-class subset is fixed; five random
re-draws (seeds 1000--1004) are reported where subset identity matters
(\S\ref{sec:audit}, App.~\ref{app:subset}). All CUB results use oracle
localisation; CelebA and BDD-OIA concepts are image-level, so there is
no part localisation to relax there---the oracle caveat is a CUB
statement, labelled as such wherever it carries weight. Audited reasons
cover all certified test instances at $K\le20$ and deterministic
$n{=}600$ subsamples at $K{=}50,200$. All runs fit on one workstation.

\subsection{The proof survives: exactness, and its indifference to meaning}
\label{sec:exp-verify}
Every issued reason is sufficient for its tree, confirmed by an
independent engine on every audited instance. The native extractor and
real PyXAI agree on $118/118$ audited four-class reasons and on
$472/472$ sufficiency checks over the larger certified-literal pool
(the CUB trusted-reason certificates; engine ledger and four isolated
PyXAI defects in App.~\ref{app:pyxai}). Exactness extends to banks
with no semantics at all: on PCA feature banks (CLS-token and
mean-patch), the same machinery issues exact three-literal certificates
whose teacher-sufficiency is zero---the machinery certifies whatever
bank it is given, which is rung five's point: \emph{semantics never
entered the machinery}. Exactness is the floor everything below is
measured against, not a finding.

\subsection{Premise support does not survive scale}
\label{sec:exp-scale}

Premise support falls below accuracy at every class count, and the
decay is smooth---roughly halving at each step of the sweep. Fig.~\ref{fig:scale}
and Tab.~\ref{tab:scale} also carry the mechanism: the decay tracks an
independence null closely (ratios $1.00$--$1.52$), so premise failure
is per-literal compounding over reasons that lengthen with $K$; only at
ten classes do errors cluster significantly above the null ($1.52$
[$1.23,1.89$]), and the four-class ratio of exactly one is degenerate:
one zero-support literal carries every failure and every passing
literal has support one. The same inversion (support well below
accuracy) holds on both replication beds (Appendix table; CelebA
$0.693$ vs.\ $0.879$, BDD-OIA $0.534$ vs.\ $0.69$--$0.75$).

\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{fig_scale.pdf}
  \caption{\textbf{Premise support versus accuracy and versus the
  independence null (Setting~T, CUB).} Solid red: fraction of certified
  reasons whose literals are all true (Wilson 95\% bands). Hollow: the
  independence null of Eq.~\eqref{eq:null}---support expected if
  premise errors were independent given the literal; the decay rides
  the null rather than departing from it. $n$ and mean reason length
  $|R|$ under each $K$. Artefact: \texttt{results/figures/fig\_scale.pdf}.}
  \label{fig:scale}
\end{figure}

\begin{table}[t]
  \centering\small
  \caption{Setting~T on CUB, oracle localisation. Support counts every
  cited literal against part annotations under the unresolved-literal
  convention of Eq.~\eqref{eq:support} (asserted on unobserved parts
  refuted; negated supported). Null: Eq.~\eqref{eq:null}. Ratio:
  observed/ null with cluster-bootstrap 95\% CI.}
  \label{tab:scale}
  \begin{tabular}{@{}rrrrrrr@{}}
    \toprule
    $K$ & $n$ & acc & support & null & ratio [CI] & $|\Rset|$ \\
    \midrule
      4 & 118 & .602 & .254 & .254 & 1.00 (degen.) & 3.6 \\
     10 & 243 & .444 & .115 & .076 & 1.52 [1.23,1.89] & 5.9 \\
     20 & 515 & .416 & .070 & .061 & 1.15 [0.87,1.44] & 7.1 \\
     50 & 600 & .303 & .035 & .034 & 1.02 [0.64,1.40] & 8.6 \\
    200 & 600 & .193 & .017 & .013 & 1.24 [0.56,2.00] & 10.8 \\
    \bottomrule
  \end{tabular}
\end{table}

\subsection{Fidelity survives partially, and is not the binding premise}
\label{sec:exp-fidelity}
Conditional fidelity---the teacher's agreement restricted to inputs
whose bits satisfy the reason---is $0.814$ at four classes and $0.613$
at ten (Wilson-intervaled in \texttt{conditional\_fidelity.json}). (The
$0.627$ in the ceiling ladder is a different quantity: the accuracy of
a tree on predicted bits against all inputs; it belongs to
\S\ref{sec:ceiling}.) Fidelity is not binding because support is the
smaller factor at every scale: the joint event (support and teacher
agreement) is $0.037/0.022/0.008$ at twenty/fifty/two hundred classes,
tracking support, not fidelity. Interventions agree: certified reasons
perturb the teacher's margin like random size-matched perturbations
(retention $-6.71$ vs.\ $-6.25$; necessity $-0.39$ vs.\ $-0.55$), and
discrete interventions sit at their chance floor---the certificate
neither tracks nor manipulates the network. The thirty all-correct
pipit reasons are the exception that frames the rule: perfect
conditional fidelity (1.0 on the pattern) with failure elsewhere.

\subsection{The failure concentrates}
\label{sec:audit}
Failures occupy few (concept, polarity) slots on every bed, but which
slot is a property of the subset, not the method. At four classes, all
$88$ failing reasons cite \emph{bill\_colour{=}yellow} present
(annotation support $0/88$; the reader fires on beak regions averaging
$3.3\%$ yellow pixels---a visual shortcut, not a threshold artifact),
while all $30$ passing reasons cite only absent-valued literals---and
none of the thirty owes its pass to the scoring rule: all would
survive the strict variant that excludes concepts on unobserved parts
(checked directly; \texttt{cbm\_cem\_audit.json} + part visibility).
Re-drawing the subset five times keeps premise support in a band
($0.267$--$0.431$) and keeps failure concentrated (top literal in
$0.62$--$0.94$ of failing reasons) but moves the carrier to
\emph{negated} slots with high marginal support ($0.90$--$0.98$)---the
mirror image of the zero-support yellow-bill literal
(App.~\ref{app:subset}). Across beds the polarity direction is set by
the domain: positive traffic literals fail on BDD-OIA
(\emph{green\_light}$=$1 at support $0.615$ in $1{,}498$ reasons),
negated face attributes on CelebA. Positive-literal precision exceeds
base rates by only $+0.03$--$0.13$ (negative $+0.06/+0.07$): the
asymmetry is not a base-rate effect.

\subsection{The vocabulary ceiling}
\label{sec:ceiling}
The main bed is vocabulary-limited, not capacity-limited. Matched
heads reach $0.627$ on predicted bits, $0.712$ on pooled region
features, $0.746$ on continuous reader scores, and $0.805$ on
ground-truth bits: the vocabulary can express the task, so repair
belongs in the readers/vocabulary, not the head. Conversely, a tree
trained directly on ground-truth bits collapses to the majority
($0.254$/$0.03$ at four/two hundred classes): ground-truth bits without
learned readers yield no usable head---the reader layer is
indispensable even as it fails. BDD-OIA sits on the other side
(\S\ref{sec:beds}): ceiling near $0.95$, failure almost entirely
perceptual.

\subsection{Abstention is the only lever, and it pays in coverage}
\label{sec:abstention}
The calibrated, polarity-aware weakest-literal gate clears the
pre-registered repair bar at the intermediate class counts, sits at it
at fifty, and is fragile at two hundred:

\begin{center}\small
\begin{tabular}{@{}rlrrl@{}}
\toprule
$K$ & operating point & support$|$issue & LB ratio vs.\ $2\times$ & verdict \\
\midrule
20  & $\lambda$ s.t.\ cov $=0.14$ & $0.361$ [$0.24,0.47$] ($n{=}72$) & $3.4$ & cleared \\
50  & cov $=0.14$ & $0.143$ [$0.07,0.23$] ($n{=}84$) & $2.0$ & at the bar \\
200 & cov $=0.14$ & $0.095$ [$0.04,0.15$] ($n{=}84$) & $2.4$ & cleared here only \\
200 & cov $=0.10$ & $0.117$ [$0.03,0.22$] ($n{=}60$) & $1.8$ & not cleared \\
200 & cov $=0.05$ & $0.100$ [$0.00,0.23$] ($n{=}30$) & $\le 1$ & not cleared \\
\bottomrule
\end{tabular}
\end{center}

The K=200 cell was verified before reporting (join exact, no
structural-literal artifacts; App.~\ref{app:prereg}): it is a tradeoff
thinning toward the unconditional rate, not a guarantee restored. Raw
confidence separates exactly where readers are miscalibrated---on CUB
it selects \emph{below} the random baseline ($0.030$ vs.\ $0.123$ at
matched coverage), while on the calibrated-reader beds it tracks the
calibrated rule; we use the raw-vs-calibrated reversal itself as the
miscalibration diagnostic. Two further facts complete the picture.
First, the \emph{oracle} marginal filter dies at two hundred classes
($95\%$ of reason literals have marginal support strictly between zero
and one, so min-marginal ranking has no signal) while the calibrated
per-instance channel still separates---the oracle is dead and
calibration is not, which is why the K=200 row is delicate rather than
null. Second, the stricter forcing gate---accept only if the trusted
literals themselves force the prediction---accepts exactly the
ranking gate's instances at every threshold and class count we
measure, and the earlier fixed-threshold read-out's low kept-true
rates ($0.067$--$0.133$) are the raw channel's \emph{selection} death:
at matched coverage the calibrated channel keeps fully supported sets
($30/30$). Support--coverage curves per bed: Fig.~\ref{fig:abstain}.

\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{fig_abstain.pdf}
  \caption{\textbf{Support--coverage curves for the trust channels of
  Eq.~\eqref{eq:gate}.} One panel per bed (CUB $K{=}20$, CelebA,
  BDD-OIA); dashed line $=2\times$ unconditional support (the repair
  bar, fixed in advance); shading from the cluster bootstrap. [TO
  BUILD at camera-ready from \texttt{task1\_abstention\_K20\_50.json}
  + the two replication audits; draft exists as
  \texttt{results/figures/task1\_abstention\_scaling.png}.]}
  \label{fig:abstain}
\end{figure}

\subsection{What does not survive}
\label{sec:nulls}

\begin{table}[t]
  \centering\small
  \caption{Interventions that do not move premise support (CUB,
  Setting~T).}
  \label{tab:nulls}
  \begin{tabular}{@{}lll@{}}
    \toprule
    Intervention & Effect on support & Source \\
    \midrule
    Oracle concept/reader selection & gain $0$ (best of 200 readers, $5/200$) & \texttt{cbm\_cem\_audit.json} \\
    Stronger reader (MLP) & acc $0.602{\to}0.678$, support $0.254{\to}0.246$ & \texttt{cbm\_cem\_audit\_reader\_mlp.json} \\
    Certified-only retraining & unchanged & \texttt{certified\_only\_cub.json} \\
    Uncertainty-aware reasons & identical ($722/722$) & \texttt{x2\_uncertainty\_reasons.json} \\
    Discrete reliability channel & no separation, any bed & \texttt{abstention\_curves.json} \\
    \bottomrule
  \end{tabular}
\end{table}

These nulls are pre-registered as such, and they are the reason this
is a measured-limit paper rather than a method paper: the levers one
would reach for---better selection, better readers, cleaner training
data, smarter reason queries---leave premise support exactly where it
was. The only lever that moves it is abstention, at its coverage
price.

\subsection{Ablations: the inversion is not a design artifact}
\label{sec:ablations}
Ablations here do a different job from the usual one: they show the
inversion is not an artifact of a choice we made. Predictions were
fixed before the runs (App.~\ref{app:prereg}). \emph{Depth, not just
scale} (Tab.~\ref{tab:ablate}, rows 1--2): varying tree depth at fixed
class count moves mean reason length $2.0\to6.0$ at ten classes and
support $0.115\to0.358$---a three-fold gain by knob-turn alone,
tracking the independence null at every depth (ratios $1.03$--$1.51$;
the pre-registered threshold was a three-fold rise, and it is met at
$3.1\times$). The mechanism is interventional, not observational. At
four classes support is \emph{flat} across depth ($0.254$ exactly, at
every depth): the yellow-bill literal rides every tree, corroborating
concentration. At two hundred classes depth spans support
$0.67\to0.03$---but the shallow tree classifies at chance: the reason
you can fully trust is the reason of a head that cannot solve the
task. \emph{Aggregator} (row 3): ranking reasons by the minimum,
product, or mean of per-literal calibrated trust, min is weakly the
best at ten classes (support $0.667$ vs.\ $0.583$ for product at $5\%$
coverage; lower-bound repair ratio $4.7$ vs.\ $4.5$; mean worst at
$3.4$)---the failure mode is a single bad literal, and min is the
aggregator built for it. Backbone need not be ablated with new
compute: the inversion replicates across three different feature
stacks already (ours on CUB, ours on CelebA, rsbench's on BDD-OIA).
Remaining ablations (predicted localisation, reader threshold,
vocabulary size, pooling) are supplement-tier; the localisation one is
the only claim-relevant unknown and is stated as such in
\S5.

\begin{table}[t]
  \centering\small
  \caption{\textbf{Design ablations} (pre-registered predictions in
  App.~\ref{app:prereg}; Setting~T, same readers throughout). Row 2
  holds $K$ fixed and turns depth; the $K{=}200$ cells use a
  deterministic sid-order subsample, so levels are not comparable to
  Tab.~\ref{tab:scale}---the within-column trend is the measurement.}
  \label{tab:ablate}
  \begin{tabular}{@{}llrrr@{}}
    \toprule
    knob & setting & $|R|$ & support & note \\
    \midrule
    depth $=2$ & $K{=}10$ & 2.00 & 0.358 & ratio 1.03 \\
    depth $=12$ & $K{=}10$ & 6.00 & 0.115 & ratio 1.51 \\
    depth $=2$ & $K{=}200$ & 2.00 & 0.672 & acc $0.00$ \\
    depth $=12$ & $K{=}200$ & 10.4 & 0.038 & acc $0.16$ \\
    \midrule
    agg $=$ min & $K{=}10$, $5\%$ cov & --- & 0.667 & LB ratio $4.7$ \\
    agg $=$ product & $K{=}10$, $5\%$ cov & --- & 0.583 & LB ratio $4.5$ \\
    agg $=$ mean & $K{=}10$, $14\%$ cov & --- & 0.529 & LB ratio $3.4$ \\
    \bottomrule
  \end{tabular}
\end{table}

\subsection{The beds fail on opposite sides}
\label{sec:beds}
CUB is vocabulary-limited and BDD-OIA is perception-limited, so the
same audit points repair at different stages. Certificate-level
escapes replicate on BDD-OIA (oracle subset-selection gain $0.0003$;
uncertainty-aware reasons set-identical on $98.3\%$; the
fixed-threshold trusted query keeps fully-true sets in $0.090$ of
covered instances, inside CUB's $0.067$--$0.10$ band), while every
training-side null reverses: stronger readers lift support
$0.534\to0.628$; retraining with the vocabulary restricted to
certified concepts lifts pooled support to $0.786$ (with a reshaped
certificate population, stated in the text); and the ground-truth-bits
tree does \emph{not} collapse ($0.987/0.986$ on
\textsc{forward}/\textsc{stop} vs.\ CUB's $0.254$ majority collapse),
with a ceiling near $0.95$. Premise errors sit \emph{at} the
independence prediction there ($1.003$ [$1.00,1.01$] vs.\ $1.52\times$
on CUB), and as a positive control, i.i.d.\ annotation noise moves
measured support exactly as the corresponding noise model predicts
(agreement within $0.002$ at every noise level): the metric responds
to annotation noise and to nothing else we injected. A third bed, SUN,
appears in the appendix only (carved, non-official split): premise
support $0.833/0.704$ at four/ten categories against the mildest gap
of any bed, consistent with its being the strongest-reader bed.
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


## Appendix: other explanation methods and model families (moved from §4 v2 at the 2026-11-14 restructure)

```latex
\section{Other explanation methods and model families}
\label{app:methods}
% body preserved verbatim from the v2 §4.9
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
\S\ref{sec:beds}.}
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

```

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

---

## Changelog 2026-11-14 (latest) — fig:pipeline DRAFTED (the last blocking figure)

Method figure rendered (`scripts/plot_fig_pipeline.py` → `results/figures/fig_pipeline.{png,pdf}`) per the user's slot-mockup: pipeline row (image + oracle boxes → φ(x): F=36 bound literals, logistic readers, val-isotonic → CART depth-12 tree, Settings T/D → R ⊨ c, minimal exact, 472/472 cross-check) over an audit panel of one rung per row (proof ✓ exact by construction; fidelity ≈ measured 0.81/0.61 partial-not-binding; premise support ✗ measured 0.254→0.017 binding), then a dashed separator and the GATE row (calibrated min-confidence abstention — the lever, not a rung; restore at partial coverage, pay in coverage). Rung-4 decision recorded: keep as the gate row; robustness earns no row (exact-by-construction like rung 1) and image truth is the same measurement as premise support at our level of treatment. No K-curve inset — that curve is Fig. 1(b) / `fig:gap`; this figure carries endpoints only. LaTeX figure environment + caption added at the end of §3 (label `fig:pipeline`, resolving the roadmap's reference). NOTE for assembly: `Fig.~\ref{fig:gap}` in this caption resolves in the INTRO file — fine once assembled, expected to look unresolved in per-file checks.

---

## Changelog 2026-11-14 (final) — §4 v3 ladder-ordered + intro v5 teaser; both user skeletons filled, every TODO decided

**Decisions made filling the two skeletons** (each TODO named its decision): (1) *teaser caption sentence 4*: headline numbers carried in the caption (captions are body matter; intro prose stays number-free — "eighty-eight" as words, no digits). (2) *pick-rule*: footnote converted to an inline parenthetical in number-words (deterministic rule stated: median-length failing reason citing the yellow-bill literal, middle image id). (3) *naming*: **survival ladder** unified in prose AND contribution bullet; the locked abstract still says "claim ladder" — edit at assembly if the portal allows (flag 13). (4) *SUN*: **appendix, final** — intro says "a third bed (SUN) is reported in the appendix under a carved, non-official split"; sec:beds one sentence. (5) *count*: "three scales" → "five class counts" (findings lead + bullet 3). (6) *verb*: "collapses" → "decays smoothly — roughly halving at each step" (the K=20/50 runs killed the cliff; null-tracking confirms). (7) *blocking check EXECUTED* (cbm_cem_audit per-instance × part visibility): all 30/30 passing K=4 reasons are all-negated AND all 30 survive the strict exclude-unobserved rule — "survivors explain by absence" describes the model, not the scoring rule; stated in intro finding 2 + §4.4. (8) *K=20/50 ratio CIs computed* (cluster bootstrap, marginals re-estimated per resample; method matches independence_gap_cub.json): **1.15 [0.87,1.44]** and **1.02 [0.64,1.40]** — both n.s. → tab:scale carries the CI column; §4.3 finding sentence = (a) inversion lead + (b) hedged mechanism ("tracks an independence null; clusters above it only at ten classes"); K=4 ratio 1.00 stated as degenerate. Committed as `results/independence_gap_K20_50.json`. (9) *fidelity number clash resolved*: 0.814/0.613 = conditional fidelity (conditional_fidelity.json, rung 3); design_b's 0.627 = ceiling-ladder accuracy (rung routing stated in §4.4). (10) *T vs D*: T carries headlines (stated in §3.4); D routed to abstention + families. (11) *bullet 4 scoped*: raw-confidence "no better than chance" now restricted to miscalibrated readers. (12) *oracle boxes*: stated once in §4.1 as a CUB statement ("CelebA/BDD concepts are image-level — no localisation to relax"). (13) *472/472 provenance*: 118/118 main audit + 472/472 certified-literal pool (app:pyxai). (14) *ECE*: not computed — the raw-vs-calibrated reversal is used as the miscalibration diagnostic; ECE optional flag. (15) *methods/families*: moved VERBATIM to appendix `app:methods` (space); tab:families' tab:transfer ref retargeted to §sec:beds.

**Structure changes**: §4 is now "What survives" in ladder order — exp-verify (exactness floor + PCA indifference) → exp-scale (fig:scale NEW: `scripts/plot_fig_scale.py` → `results/figures/fig_scale.{png,pdf}`, adds the independence-null hollow series + n/|R| under each K) → exp-fidelity (incl. margins + pipit leaf one sentence) → audit/concentration (incl. re-draws + strict-rule check + precision-beyond-base-rates) → ceiling → abstention (per-K verdict table incl. the K=200 three-operating-point honesty; oracle-dies/calibration-separates paragraph; forcing≡ranking paragraph; fig:abstain TO BUILD) → nulls (tab:nulls with sources) → beds (BDD reversals + independence + label-noise control + SUN pointer). Old §4.2–§4.9 superseded; their content preserved here or in app:methods. §3 cross-refs updated to the new labels; §3's duplicate sec:ceiling label removed (label lives on §4.5); §3 gate keeps only sec:gate (§4.6 owns sec:abstention).

**New flags**: (16) fig:abstain to build at camera-ready (data: task1_abstention_K20_50 + the two replication audits; draft = results/figures/task1_abstention_scaling.png). (17) concentration heat-strip figure (literal × polarity per bed) still to build — carries §4.5's claim. (18) ECE per bed optional. (19) drawio strip (fig1_strip.pdf) may replace fig1_v2.pdf at camera-ready — both labels attached. (20) strict-rule sensitivity number for tab:scale caption: "all 30/30 pass strictly" is K=4-only; compute the strict variant at other K if a reviewer asks.

---

## Changelog 2026-11-14 (night) — ABLATIONS EXECUTED under pre-registration (addendum committed BEFORE the runs, 2f88002)

**Depth sweep (C1)** `scripts/ablation_depth_support.py` → `results/ablation_depth_support.json`: |R| and support at depths {2,4,6,8,12,16} × K ∈ {4,10,200}. Verdicts vs the pre-registered P-D1..P-D3: **P-D1 ✓** (|R| monotone in depth, saturating), **P-D2 ✓** (ratio to independence null in [1.00, 1.56] at every cell, inside the pre-registered [0.7, 1.7]), **P-D3 ✓ exactly at the bar** (K=10: 0.358/0.115 = 3.1× ≥ 3×). Two unanticipated findings recorded: (i) K=4 support is EXACTLY flat at 0.254 across all depths — the yellow-bill literal rides every tree (concentration corroboration); (ii) K=200 depth-2 gives support 0.672 at accuracy 0.000 — the trustable reasons come from a head that cannot classify; §4.10 states the trade. CAVEAT recorded in table caption: K=200 ablation cells use first-600-by-sid (class-skewed), so LEVELS are not comparable to tab:scale's 0.017 (committed audit subsample differs); the within-column trend is the measurement. K=4/10 cells match committed levels exactly (0.254, 0.115).
**Aggregator (D1)** `scripts/ablation_aggregator.py` → `results/ablation_aggregator.json`: vs P-A1 — outcome ✓ (min weakly best at K=10: 0.667 vs product 0.583 at 5% cov; LB ratio 4.7 vs 4.5; mean worst 3.4), rank-correlation sub-prediction ✗ (Spearman min/product 0.81/0.62 < 0.9 — falsified honestly; literal trusts are less skewed than predicted). K=200 aggregator unmeasured (task3 dump covers K ∈ {4,10}) — stated.
**Not run + rationale:** backbone (A2) — covered by the three replication feature stacks (one sentence in §4.10); predicted localisation (A1) — the only claim-relevant unknown ("can only lower" is an assumption); assigned next, else §5 softens to "untested"; reader threshold / pooling / vocabulary size — supplement tier. New subsection §4.10 "Ablations" + tab:ablate added (10 subsections now — merge with sec:nulls at .tex assembly if space demands).
