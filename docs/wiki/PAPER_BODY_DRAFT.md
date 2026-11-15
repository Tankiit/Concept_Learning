# CVPR 2027 — Body draft: related work, method, experiments, limitations, conclusion

**Drafted:** 2026-11-13. Companion to [`PAPER_INTRO_DRAFT.md`](PAPER_INTRO_DRAFT.md) (v3, number-free). Written from the keep list in [`CVPR2027_CLAIM_AUDIT.md`](CVPR2027_CLAIM_AUDIT.md) (final cut 2026-11-11 + the 2026-11-12 replication addendum). Numbers live in the body by design; every table's provenance is listed below. Citation keys are PLACEHOLDERS until the Nov 15 pass (resolved: `canizales` = arXiv:2605.06640).

---

## §2 Related work

```latex
\section{Related Work}
\label{sec:related}

\paragraph{Concept bottleneck models.}
CBMs \cite{koh2020concept} and their post-hoc and embedding variants
\cite{yuksekgonul2023posthoc,zarlenga2022concept} predict through a
layer of human-readable concepts. A substantial literature documents
that this layer does not behave as intended: concept scores leak
task information \cite{margeloiu2021how}, learned representations
escape their intended semantics \cite{mahinpei2021}, and models
exploit \emph{reasoning shortcuts}---concept assignments that yield
correct labels while being wrong \cite{marconato2024,rsbench}.
\emph{We add}: the question one step further up. Rather than asking
whether the concept layer is faithful, we ask what survives a formal
sufficiency certificate \emph{over} that layer---which of its claims
hold by construction, and at what rate the two premises the proof
omits (image truth and teacher agreement) die as the task grows. Our
driving-action replication runs on the rsbench suite
\cite{rsbench} built to expose reasoning shortcuts, and the literals
our audit flags (green light, red light) are precisely the
shortcut-prone concepts that suite identifies.

\paragraph{Formal explanations.}
Abductive explanations certify that a set of inputs provably entails
a classifier's prediction \cite{ignatiev2018abductionbased}, with the
sufficient/complete-reason vocabulary fixed by Darwiche and Hirth
\cite{darwiche2020reasons} and tooling such as PyXAI
\cite{audemard2024pyxai}. A growing line applies such certificates to
neuro-symbolic pipelines \cite{paul2024formal}, to concept
bottlenecks \cite{teso2026when}, and to interpretable-by-design
architectures such as prototype networks \cite{soria2026formal}---in
contrast to heuristic saliency \cite{selvaraju2017gradcam}, which
offers no guarantee at all. Closest is the behavior-explainer work of
Canizales et al.\ \cite{canizales2026conceptbased}, which computes
abductive and contrastive explanations for \emph{behaviors} of vision
models by querying the network head through concept-erasure operators
in aligned CLIP space. Because their oracle is the network itself,
fidelity holds by construction---but exactness is conditional on an
empirical head-monotonicity assumption, explanations come in families
of hundreds per image, and the truth of the cited concepts is graded
by an LLM judge rather than measured. \emph{We add} the mirror image:
we certify against a symbolic head, where exactness is unconditional
and reasons are essentially unique, and we spend the saved effort
measuring the premise their setting cannot express---against held-out
per-image annotations, with pre-registered criteria. Teso et al.\
\cite{teso2026when} ask when CBM explanations can be faithful
\emph{and compact}, answering at the architecture level: faithfulness
forces all-concept explanations unless concepts are probabilistic and
sparsified at training time. Beyond them, we take the certified
explanation as given and measure the axis they leave open---whether
its premises are \emph{true of the image}, how fast that truth decays
with scale, and what restores it---finding, among other things, that
their sparsification lever (shorter reasons) buys premise truth only
by giving up accuracy. We do
not claim novelty for concept-level formal explanations; the
contribution is the measurement of what survives the transfer.

\paragraph{Trust, calibration, and abstention.}
Selective prediction gates predictions on confidence
\cite{geifman2017selective}, and conformal methods calibrate that
gate \cite{angelopoulos2023conformal}; a large literature supplies
the uncertainty estimates such gates consume, from ensembles and
Bayesian methods \cite{kendall2017uncertainties,gal2016dropout,lakshminarayanan2017simple,malinin2018predictive,sensoy2018evidential}
to their surveys \cite{hullermeier2021aleatoric,gawlikowski2023survey,ulmer2023survey} and behaviour under shift \cite{ovadia2019trust,hendrycks2017baseline}; recent position papers argue prediction-level
uncertainty is fundamentally incomplete
\cite{jimenez2026position,aldoma2026position}. \emph{We add} the level
that argument calls for, inside a certified explanation: the gate
scores each \emph{literal} of a sufficient reason, not the prediction,
with polarity-aware per-concept calibration---and we show the discrete
alternative (population-level reliability rates) never identifies
unreliable reasons on any dataset we test, that raw confidence is
anti-informative exactly where readers are miscalibrated, and that the
repair bar a gate must clear is fixed before the runs, not chosen
after.
```

## §3 Method (v5, 2026-11-14 night: user's CVPR rewrite, all TODOs filled; DINOv3→DINOv2 corrected; protocol moved to §4.1)

```latex
% NOTE: cleveref required (\cref). Macros: \phix \Rset \Lits \supp \acc \trust.
% CVPR numeric citations: "Teso \etal~\cite{teso2026when}", not \citet.

\section{Auditing certified concept reasons}
\label{sec:method}

Our setup is a four-stage pipeline: a frozen backbone produces
part-level features, linear readers turn them into binary concepts, a
decision tree predicts the class from those concepts, and a formal
engine extracts a sufficient reason for each prediction. The
certificate covers only the last two stages. Our goal is to measure how
much of it survives the first two. \cref{sec:pipeline} describes the
pipeline, \cref{sec:guarantees} states what the certificate does and
does not prove, \cref{sec:measures} defines how we measure the gap, and
\cref{sec:gate} introduces a trust gate that withholds reasons built on
unreliable concepts.

\subsection{Pipeline}
\label{sec:pipeline}

\paragraph{Perception.}
Given an image $x$, a frozen DINOv2 ViT-S/14 backbone~\cite{dinov2}
produces patch tokens $z\in\mathbb{R}^{N\times d}$. For each part
$p\in\mathcal{P}$, we pool the tokens in a fixed-size box centred on
the annotated part keypoint to obtain a region feature $h_p(x)$. We use
these oracle part locations throughout, so that our measurements
isolate concept errors from localisation errors. With a learned
detector, the support rates we report can only decrease --- and this
statement is itself untested: predicted localisation is the one
design ablation we flag rather than claim (\cref{sec:ablations}).

\paragraph{Concepts.}
Each of $F{=}36$ concepts $c_k=(p_k,a_k)$ pairs a part with an
attribute (\eg \texttt{beak:bill\_color=yellow}) and is predicted by a
linear reader on its part feature:
\begin{equation}
  \phi_k(x) = \mathbb{1}\!\left[\sigma\!\left(w_k^\top h_{p_k}(x)\right) > \tfrac12\right],
  \qquad \phix\in\{0,1\}^F.
  \label{eq:readers}
\end{equation}
Reader probabilities are calibrated with isotonic regression on a
held-out validation split. A \emph{literal} $\ell=(k,b)$ is a concept
with a polarity $b\in\{0,1\}$, and $\Lits$ is the set of all $2F$
literals.

\paragraph{Reasoning head.}
A decision tree $t:\{0,1\}^F\to[K]$ predicts the class from $\phix$
alone. We study two variants.
\textbf{Label-trained (T):} $t$ is fit to the class labels. Its
premises are exactly the bits a deployed system reads, so all main
results use this variant.
\textbf{Distilled (D):} $t$ is distilled from a neural head
$g:\{0,1\}^F\to[K]$ over the same concepts and acts as its surrogate.
We use it where agreement with $g$ matters (\cref{sec:abstention},
\cref{tab:families}).

\paragraph{Explanation.}
For a prediction $c=t(\phix)$, a \emph{sufficient reason} is a
subset-minimal set of literals $\Rset\subseteq\Lits$, all true of
$\phix$, such that
\begin{equation}
  \forall\, u\in\{0,1\}^F:\quad u\models \Rset \;\Rightarrow\; t(u)=c.
  \label{eq:sufficient}
\end{equation}
We compute reasons with a reachability-based engine and cross-check
them against PyXAI~\cite{audemard2024pyxai}; the two agree on every audited
instance ($118/118$ reasons, $472/472$ sufficiency checks over the
larger certified-literal pool; Supp.~\cref{app:pyxai}, which also
covers literal- \vs feature-level minimality). These concept-level
reasons are the same object used by Teso \etal~\cite{teso2026when} and Soria
\etal~\cite{soria2026formal}. We do not propose a new explanation. We audit what
this one guarantees.

\subsection{What the certificate proves}
\label{sec:guarantees}
\label{sec:ladder}

\cref{eq:sufficient} holds for every binary vector $u$. It does not say
whether the cited concepts are present in $x$, nor whether $t$ agrees
with the deployed network $g$. We make this explicit by distinguishing
five \emph{levels of guarantee} (\cref{tab:levels}). Levels~1--2 hold
by construction, whereas levels~3--5 depend on perception and must be
measured.

\begin{table}[t]
  \centering\small
  \caption{\textbf{Levels of guarantee for a certified reason.} Each
  level adds a claim that the levels above it do not imply. Levels
  above the dashed line hold by construction; those below it are
  measured in \cref{sec:experiments}.}
  \label{tab:levels}
  \begin{tabular}{@{}clll@{}}
    \toprule
    & Claim about $\Rset$ & Status & Measured in \\
    \midrule
    1 & sufficient for $t$ (\cref{eq:sufficient})        & exact    & Supp.~\ref{app:pyxai} \\
    2 & sufficient under any untrusted completion        & exact    & \cref{sec:nulls,sec:abstention} \\
    \cdashline{1-4}
    3 & $t$ agrees with $g$ where $\phix\models\Rset$    & measured & \cref{sec:exp-fidelity} \\
    4 & every premise is true of $x$                     & measured & \cref{sec:exp-scale} \\
    5 & premises name the intended concepts              & audited  & \cref{sec:exp-verify} \\
    \bottomrule
  \end{tabular}
\end{table}

\subsection{Measuring the gap}
\label{sec:measures}

\paragraph{Premise support.}
Let $a(x)\in\{0,1,\star\}^F$ be the ground-truth annotation, where
$\star$ marks concepts whose part is not visible. A literal
$\ell=(k,b)$ is \emph{supported} if $a_k(x)=b$. A reason is supported
if all its literals are:
\begin{equation}
  \supp(\Rset,x) = \mathbb{1}\!\left[\forall\,\ell\in\Rset:\
  a_k(x)=b\right].
  \label{eq:support}
\end{equation}
For an invisible part, we treat all its concepts as absent. Supp.\
\cref{app:unresolved} reports the stricter alternative, which scores
only fully resolved reasons.

\paragraph{Support \vs accuracy.}
We compare $\mathbb{E}[\supp(\Rset,x)]$ with accuracy
$\acc=\mathbb{E}[\mathbb{1}[t(\phix)=y]]$ on the same audited
instances: a reader expects an explanation to be at least as reliable
as the prediction it explains.

\paragraph{Independence baseline.}
With $\hat p_\ell$ the support rate of literal $\ell$ over its
occurrences in reasons, the \emph{independence baseline} is
\begin{equation}
  \widehat{\supp}_{\perp} = \frac{1}{n}\sum_{i=1}^{n}
  \prod_{\ell\in \Rset_i} \hat p_\ell,
  \label{eq:null}
\end{equation}
the support expected if premise errors were independent given the
literal. A support-to-baseline ratio above one means failures share
literals---predictable compounding, with concrete targets for repair;
departures from one are reported where they are measured
(\cref{sec:exp-scale}).

\paragraph{Fidelity.}
$\operatorname{fid}(\Rset)=\Pr\!\left[g(\phi(x'))=c \mid
\phi(x')\models\Rset\right]$, estimated on held-out data with Wilson
intervals.

\paragraph{Vocabulary ceiling.}
Matched heads trained on predicted bits, on continuous reader scores,
and on ground-truth bits bound what any head over this vocabulary can
achieve, locating repair in the readers or the vocabulary rather than
in the head (\cref{sec:ceiling}).

\subsection{Gating reasons by concept trust}
\label{sec:gate}

A \emph{trust channel} assigns each literal occurrence a score
$\trust(\ell,x)\in[0,1]$, and a reason is issued only if its weakest
literal clears a threshold,
\begin{equation}
  \operatorname{issue}(\Rset,x) =
  \mathbb{1}\!\left[\min_{\ell\in\Rset}\trust(\ell,x)\ge\lambda\right].
  \label{eq:gate}
\end{equation}
Channels: \textbf{raw} reader confidence (or its complement, for
negated literals); \textbf{calibrated} per-concept confidence
(validation isotonic, polarity-aware); \textbf{discrete}
per-(concept, polarity) reliability rates; \textbf{random} ordering at
matched coverage; and an oracle marginal-support filter as reference.
The criterion under which a channel counts as a repair, and the
decision rules it triggers, were fixed before evaluation
(\cref{sec:exp-setup}).
```

## §4 Experiments (v4, 2026-11-14 night: user's rewrite adopted, all \todo filled)

```latex
\section{Experiments}
\label{sec:experiments}

Our experiments examine the five levels of guarantee in
\cref{tab:levels} in turn, from those that hold by construction to
those that must be measured empirically. They address four questions.
\textbf{(Q1)} Is the certificate exact, and does exactness depend on
the semantics of the concept bank? \textbf{(Q2)} How often are the
premises of a certified reason true of the image, and how does this
change with the number of classes? \textbf{(Q3)} Where does premise
failure originate: in the vocabulary, in perception, or in the
reasoning head? \textbf{(Q4)} Can any intervention restore premise
support, and at what cost?

\subsection{Experimental setup}
\label{sec:exp-setup}

\paragraph{Datasets.}
We fully audit CUB-200-2011~\cite{cub}: its per-image part annotations
allow every premise of a certified reason to be checked against ground
truth. We
use the official splits and vary the number of classes over
$K\in\{4,10,20,50,200\}$. To test whether our findings depend on the
choice of classes, the four-class subset is redrawn five times (supp.\
\cref{app:subset}). We repeat the full evaluation, with the same
readers, heads, reason extraction and support scoring, on
CelebA~\cite{celeba} using identity-disjoint splits and on
BDD-OIA~\cite{bddoia} using the splits of rsbench~\cite{rsbench}.
Results on SUN, for which no official split exists, are reported in
the supplement.

\paragraph{Evaluation protocol.}
For each test image, we extract a sufficient reason for the tree's
prediction, regardless of whether that prediction is correct, and
compare each cited literal with the ground-truth part annotations
(\cref{eq:support}). At $K\le 20$ we score every certified test
instance ($n=118$, $243$ and $515$, respectively). At $K=50$ and
$K=200$ we score a fixed subsample of $n=600$ instances, drawn from
$1{,}389$ and $5{,}794$ certified instances. All CUB results use
oracle part localisation. Because the concepts in CelebA and BDD-OIA
are defined at the image level, this caveat applies to CUB alone.
Unless stated otherwise, intervals are 95\% cluster-bootstrap
intervals over instances with $10^4$ resamples. The repair criterion
of \cref{sec:gate} and the associated decision rules were fixed before
any experiment was run (supp.\ \cref{app:prereg}). All experiments run
on a single workstation.

\subsection{The certificate is exact, irrespective of semantics}
\label{sec:exp-verify}

Every reason issued by our engine is sufficient for its tree. We
verify this against PyXAI~\cite{audemard2024pyxai}, which agrees with our engine
on all $118$ four-class reasons and on all $472$ sufficiency checks
over the larger pool of certified literals (supp.\ \cref{app:pyxai}).

Exactness, however, carries no information about the meaning of the
concepts. When we replace the concept bank with principal components
of the CLS token or of the mean patch embedding, the same pipeline
returns exact certificates of three literals on average, yet these
certificates are never sufficient for the teacher network. The
certificate is thus equally valid for any bank it is given, which is
precisely why level~5, that the premises name the intended concepts,
must be checked against annotations rather than assumed. We therefore
treat exactness as the baseline against which the remaining levels are
measured, not as a result in its own right. \textbf{(Q1:} yes, exact
everywhere, and indifferent to the semantics of the bank.\textbf{)}

\subsection{Premise support degrades with scale}
\label{sec:exp-scale}

\begin{table}[t]
  \centering\small
  \setlength{\tabcolsep}{4pt}
  \caption{\textbf{Premise support and accuracy on CUB} (Setting~T,
  oracle localisation). \emph{Null} is the support expected under
  independent literal errors (\cref{eq:null}). \emph{Ratio} is
  support/null with a 95\% bootstrap interval. $|\Rset|$ is the mean
  reason length. Joint: support and teacher agreement together.
  $^\dagger$Degenerate: a single literal with zero support accounts for
  every failure.}
  \label{tab:scale}
  \begin{tabular}{@{}rrcccccr@{}}
    \toprule
    $K$ & $n$ & Acc. & Support & Joint & Null & Ratio & $|\Rset|$ \\
    \midrule
      4 & 118 & 0.602 & 0.254 & 0.254 & 0.254 & 1.00$^\dagger$ &  3.6 \\
     10 & 243 & 0.444 & 0.115 & 0.070 & 0.076 & 1.52 {\scriptsize[1.23, 1.89]} &  5.9 \\
     20 & 515 & 0.416 & 0.070 & 0.037 & 0.061 & 1.15 {\scriptsize[0.87, 1.44]} &  7.1 \\
     50 & 600 & 0.303 & 0.035 & 0.022 & 0.034 & 1.02 {\scriptsize[0.64, 1.40]} &  8.6 \\
    200 & 600 & 0.193 & 0.017 & 0.008 & 0.013 & 1.24 {\scriptsize[0.56, 2.00]} & 10.8 \\
    \bottomrule
  \end{tabular}
\end{table}

\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{fig_scale.pdf}
  \caption{\textbf{Premise support versus accuracy and versus the
  independence baseline (Setting~T, CUB).} Solid red: fraction of
  certified reasons whose literals are all true (Wilson 95\% bands).
  Hollow: the independence baseline of \cref{eq:null}---support
  expected if premise errors were independent given the literal; the
  decay tracks the baseline rather than departing from it. Dotted:
  strict scoring (only reasons whose every literal sits on a visible
  part)---indistinguishable from the main curve at four of five class
  counts. $n$ and mean reason length $|\Rset|$ under each $K$.}
  \label{fig:scale}
\end{figure}

\cref{tab:scale} and \cref{fig:scale} report premise support alongside
classification accuracy. At every class count, the probability that
all premises of a certified reason hold is substantially lower than
the probability that the prediction itself is correct. The gap widens
with scale. Between $K=4$ and $K=200$, accuracy decreases by a factor
of roughly three (from $0.602$ to $0.193$), whereas premise support
decreases by a factor of fifteen (from $0.254$ to $0.017$), roughly
halving at each step of the sweep.

The independence null accounts for most of this decay. Mean reason
length grows from $3.6$ to $10.8$ literals as $K$ increases, and the
observed support remains statistically consistent with the product of
per-literal support rates at $K=20$, $50$ and $200$. The only
significant departure occurs at $K=10$, where the ratio of $1.52$
indicates that failures share literals more often than independence
would predict. The ratio of exactly one at $K=4$ is degenerate for the
reason given in \cref{tab:scale}. These results suggest that premise
failure is driven primarily by per-literal error compounding over
reasons that lengthen with the number of classes, rather than by a
small number of correlated failure modes. We test this interpretation
directly in \cref{sec:ablations}.

The scoring convention does not create the decay: under strict
scoring, which counts only reasons whose every literal sits on a
visible part, support is unchanged at four of five class counts
($0.254$, $0.115$, $0.035$, $0.017$) and moves from $0.070$ to $0.054$
at twenty (dotted line, \cref{fig:scale}). The same inversion appears
on both replication datasets (supp.\ \cref{tab:replication}). On CelebA, premise support is $0.693$ against
an accuracy of $0.879$, and on BDD-OIA it is $0.534$ against an
accuracy of $0.690$--$0.755$. \textbf{(Q2:} rarely all true, at every
class count; roughly halving per step and tracking the independence
baseline.\textbf{)}

\paragraph{Failures concentrate in a few literals.}
\label{sec:exp-conc}
Although failures compound across literals, they are not spread
uniformly across the vocabulary (supp.\ \cref{fig:failures}). At $K=4$, all
$88$ failing reasons cite \texttt{bill\_color=yellow} as present, a
literal the annotations never support. The corresponding reader fires
on beak regions in which only $3.3\%$ of pixels are yellow on average,
indicating a visual shortcut rather than a poorly chosen threshold (the reader fires even on beaks with no yellow pixels at all---mean $3.3\%$ across firing beaks---four examples in the supplement, Fig.~\ref{fig:beakstrip}).
All $30$ passing reasons at $K=4$ cite only negated literals, but
this is a property of the small subset rather than of the method: at
$K=20$, $35$ of the $36$ fully supported reasons cite at least one
positive literal (\cref{fig:qualitative}c). All thirty remain valid
under the stricter scoring rule that excludes concepts on unobserved
parts, so none owes its pass to the convention. Concentration persists
across the five redrawn subsets: premise support remains between
$0.267$ and $0.431$, and the most frequent failing literal appears in
$62$--$94\%$ of failing reasons. The identity and polarity of that
literal, however, change with the subset. In the redrawn subsets it is
a negated literal with high marginal support ($0.90$--$0.98$), the
mirror image of the yellow-bill case. The domain likewise determines
the polarity across datasets. Failures on BDD-OIA are carried by
positive traffic literals, such as \texttt{green\_light=1}, which is
supported in only $61.5\%$ of its $1{,}498$ occurrences, whereas
failures on CelebA are carried by negated facial attributes. The same
asymmetry explains why heuristic explanations can look ``truer'' than
certified ones: on BDD-OIA, random three-literal sets enjoy higher
premise support than sufficient reasons ($0.682$ versus $0.534$)
because they assert mostly easy negations, while sufficiency forces
the reason onto rare positive literals---the price of informativeness
(supp.).

\subsection{Fidelity holds partially but is not the bottleneck}
\label{sec:exp-fidelity}

In Setting~D, the tree is a surrogate for a neural head $g$, and a
certified reason transfers to $g$ only if the two agree on inputs that
satisfy it. This conditional fidelity is $0.814$ at $K=4$ and $0.613$
at $K=10$. Although imperfect, fidelity is not what limits the
reliability of certified reasons. The probability that a reason is
both premise-supported and faithful to the teacher is $0.037$,
$0.022$ and $0.008$ at $K=20$, $50$ and $200$ (and $0.254$, $0.070$ at four and ten).
Wherever both factors are measured, support is the smaller one, and
the joint tracks support rather than fidelity.

Interventional evidence leads to the same conclusion. Perturbing the
literals of a certified reason changes the teacher's margin by about
as much as perturbing a random set of literals of the same size
(retention $-6.71$ versus $-6.25$; necessity $-0.39$ versus $-0.55$),
and discrete interventions on certified literals remain at their
chance floor. Even where the tree agrees with the teacher, the
certified reason thus does not identify the inputs that drive the
teacher's decision. \textbf{(Q3, completed in \cref{sec:ceiling}:} on
CUB, the readers that instantiate an expressive vocabulary; on
BDD-OIA, perception alone.\textbf{)}

\subsection{Vocabulary versus perception}
\label{sec:ceiling}

\begin{table}[t]
  \centering\small
  \caption{\textbf{Accuracy of matched heads by input.} The gap
  between predicted and ground-truth bits bounds what better readers
  could recover; the ground-truth row bounds what any head over this
  vocabulary can achieve. BDD-OIA column: \textsc{forward} action
  (per-action values in supp.\ \cref{app:beds}); the continuous-scores
  variant was not run on BDD-OIA.}
  \label{tab:ceiling}
  \begin{tabular}{@{}lcc@{}}
    \toprule
    Head input & CUB ($K{=}4$) & BDD-OIA \\
    \midrule
    Predicted bits $\phix$     & 0.627 & 0.72 \\
    Pooled region features     & 0.712 & 0.75 \\
    Continuous reader scores   & 0.746 & --- \\
    Ground-truth bits $a(x)$   & 0.805 & 0.99 \\
    \bottomrule
  \end{tabular}
\end{table}

To determine where repair should be directed, we train heads of the
same family on progressively more informative inputs
(\cref{tab:ceiling}; visual in supp.\ \cref{fig:ceiling}). On CUB, accuracy increases monotonically from
predicted bits ($0.627$) to ground-truth bits ($0.805$). The concept
vocabulary can therefore express the task, and the shortfall lies in
the readers that instantiate it. BDD-OIA presents the opposite
picture. Its ceiling is close to $0.99$ on the \textsc{forward} and
\textsc{stop} actions, its failures are almost entirely perceptual,
and its premise errors sit at the independence prediction ($1.003$,
CI $[1.00, 1.01]$). Across datasets, the gap between accuracy and
support also tracks reader quality ($35/19/19/8$ points as reader
accuracy rises from $0.71$ to $0.95$). This between-dataset gradient
does not contradict the CUB null of \cref{tab:nulls}: the stronger MLP
reader there gained $7.6$ points without touching the failing slots
---the yellow-bill shortcut fired exactly as before---whereas the
replication beds pair much stronger readers with different,
healthier concept-image alignments. Reader quality helps when it
repairs the specific failing slots, not in the aggregate.

This difference is reflected in how the two datasets respond to
intervention. On BDD-OIA, interventions on the training side, which
leave premise support unchanged on CUB (\cref{sec:abstention}),
improve it substantially. Stronger readers raise support from $0.534$
to $0.628$, and restricting the vocabulary to certified concepts
raises it to $0.786$, albeit with a different population of
certificates (supp.\ \cref{app:beds}). Interventions on the
certificate itself remain ineffective on both datasets. On BDD-OIA,
oracle subset selection gains $0.0003$, and uncertainty-aware reasons
coincide with standard reasons on $98.3\%$ of instances. As a positive
control for the metric, injecting i.i.d.\ annotation noise changes
measured support by the amount the noise model predicts, to within
$0.002$ at every noise level.

\subsection{Abstention restores support at a cost in coverage}
\label{sec:abstention}

\begin{table}[t]
  \centering\small
  \setlength{\tabcolsep}{3.5pt}
  \caption{\textbf{Calibrated weakest-literal gate on CUB.} Setting~T
  except the marked row (Setting~D, the distilled arm, whose
  unconditional support is $0.124$). LB is the bootstrap lower bound
  on the ratio of support among issued reasons to unconditional
  support; the pre-registered repair criterion requires
  $\mathrm{LB}\ge 2$.}
  \label{tab:gate}
  \begin{tabular}{@{}rcccl@{}}
    \toprule
    $K$ & Coverage & Support $\mid$ issued & LB & Criterion \\
    \midrule
     10$^{\mathrm{D}}$ & 0.136 & 0.697 {\scriptsize[0.52, 0.85]} & 4.2 & met \\
     20 & 0.14 & 0.361 {\scriptsize[0.24, 0.47]} & 3.4 & met \\
     50 & 0.14 & 0.143 {\scriptsize[0.07, 0.23]} & 2.0 & at threshold \\
    200 & 0.14 & 0.095 {\scriptsize[0.04, 0.15]} & 2.4 & met \\
    200 & 0.10 & 0.117 {\scriptsize[0.03, 0.22]} & 1.8 & not met \\
    200 & 0.05 & 0.100 {\scriptsize[0.00, 0.23]} & $\le$1.0 & not met \\
    \bottomrule
  \end{tabular}
\end{table}

\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{fig_abstain.pdf}
  \caption{\textbf{The gate, channel by channel and point by point.}
  \textbf{(a)}~Support among issued reasons at matched coverage
  ($K{=}10$, Setting~D): the calibrated weakest-literal channel matches
  the oracle marginal filter, while raw confidence selects below
  random. \textbf{(b)}~Setting~T operating points: issued support
  (orange, bootstrap bars) against the unconditional rate (grey
  circles) and the dashed $2\times$ repair bar, fixed in advance; the
  $K{=}200$ rows show the fragility ourselves---the bar is cleared at
  coverage $0.14$ and not below.}
  \label{fig:abstain}
\end{figure}

\cref{tab:gate} and \cref{fig:abstain} evaluate the gate of
\cref{eq:gate} with calibrated, polarity-aware confidence as the trust
channel. Issuing only those reasons whose weakest literal is
confidently predicted meets the pre-registered criterion at $K=20$,
where support among issued reasons rises from $0.070$ to $0.361$ at a
coverage of $14\%$. The criterion is met exactly at $K=50$, and only
at a single operating point at $K=200$. At this scale, the gain
diminishes toward the unconditional rate as coverage decreases, and it
should be read as a trade-off rather than a restored guarantee.

The choice of trust channel matters, and the strongest gate result
comes early: at ten classes in Setting~D, the calibrated gate reaches
$0.697$ [$0.52,0.85$] at $13.6\%$ coverage against $0.727$ for the
oracle marginal filter---statistically indistinguishable from a filter
that knows each literal's true support. On CUB (Setting~D, $K{=}10$),
raw reader confidence selects reasons with \emph{lower} support than
random selection at matched coverage ($0.030$ versus $0.123$); on
CelebA and BDD-OIA, where the readers are well calibrated, raw
confidence matches or exceeds the calibrated channel ($0.946$ on
CelebA; $0.817$ versus $0.755$ on BDD-OIA), so whether the two
channels agree serves as a diagnostic of reader miscalibration. Calibrated
per-instance confidence also remains informative where
population-level information does not. At $K=200$, $95\%$ of the
literals in reasons have marginal support strictly between zero and
one, so an oracle filter based on marginal support has no signal,
whereas the calibrated channel continues to separate supported from
unsupported reasons. A reviewer-level baseline completes the picture: abstaining on the
\emph{prediction's} own confidence---the standard selective-prediction
rule, scoring every literal of an instance with the tree's maximum
class probability---does not substitute for the per-literal gate. At
$K{=}10$ it raises support among issued reasons to only $0.212$
[$0.11,0.38$] at $13.6\%$ coverage, a bootstrap lower-bound ratio of
$0.8$ (no better than random ordering), where the calibrated
per-literal gate reaches $0.727$ [$0.56,0.85$] with ratio $4.7$:
prediction confidence is blind to \emph{which} literal fails. Finally,
a stricter gate, which accepts a reason
only if its trusted literals alone force the prediction, selects
exactly the same instances as the ranking gate at every threshold and
at the class counts we test (four and ten). The gate's limitations
therefore stem from the ranking it induces rather than from the
acceptance rule. \textbf{(Q4:} calibration-gated abstention, robustly
at intermediate class counts and marginally at the full benchmark,
always paying in coverage; on CUB it is the only accuracy-preserving
lever.\textbf{)}

\begin{table}[t]
  \centering\small
  \setlength{\tabcolsep}{4pt}
  \caption{\textbf{Interventions that do not change premise support}
  (CUB, Setting~T; $K{=}4$ unless noted).}
  \label{tab:nulls}
  \begin{tabular}{@{}ll@{}}
    \toprule
    Intervention & Effect \\
    \midrule
    Oracle concept / reader selection & no gain (best of 200 candidates) \\
    Stronger reader (MLP)             & acc.\ $0.602\to0.678$; support $0.254\to0.246$ \\
    Retraining on certified instances & support stays $0.254$; yellow-bill still $0/88$ \\
    Uncertainty-aware reasons         & identical on $722/722$ instances \\
    Discrete reliability channel      & no separation on any dataset tested \\
    \bottomrule
  \end{tabular}
\end{table}

On CUB, abstention is the only intervention that raises premise
support \emph{without sacrificing accuracy}. \cref{tab:nulls} lists
the alternatives we pre-registered: oracle selection of concepts or
readers, stronger readers, retraining on certified instances,
uncertainty-aware reasons and discrete reliability estimates. None of
them alters premise support on CUB---most notably, the stronger reader
improves accuracy by more than seven points while support stays at
$0.254$ and the yellow-bill literal still fires at $0/88$. The two
interventions that \emph{do} move support each pay elsewhere:
shortening reasons trades against accuracy (\cref{sec:ablations}),
and training-side repairs work only on the perception-limited bed,
where they also help accuracy (\cref{sec:ceiling}). This set of null
results is the principal reason we present the paper as a measured
limit rather than as a method.

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{fig_qualitative.pdf}
  \caption{\textbf{Certified reasons at increasing tree depth} (CUB,
  $K=20$, Setting~T). Each row shows one test image and its sufficient
  reason at four depths. Literals are marked \textcolor{ok}{\ding{51}}
  supported, \textcolor{bad}{\ding{55}} refuted, or $\circ$ on a part
  that is not visible. Boxes mark the parts cited by the default-depth
  reason, coloured by their worst status.
  (a)~A typical failure: the reason is valid for the tree and the
  prediction is correct, but a premise is false.
  (b)~The depth trade-off: the depth-two reason is fully true but
  justifies the wrong class, whereas the default tree is correct; this
  occurs for $92$ of the $515$ instances ($17.9\%$).
  (c)~A success: fully supported default reason with a correct
  prediction, citing positive evidence---as do all $19$ fully
  supported \emph{and} correct default reasons (of the $36$ fully
  supported, $35$ cite a positive literal; the remainder of those are
  wrong-class successes, which is why we require correctness here).
  Rows are selected by fixed rules: (a)~median-length failing default
  reason, (b,\,c)~median length among qualifying instances, with ties
  broken by the middle instance index (supp.\ \cref{app:qualitative}).}
  \label{fig:qualitative}
\end{figure*}

\subsection{Ablations}
\label{sec:ablations}

\begin{table}[t]
  \centering\small
  \caption{\textbf{Ablations on CUB at $K{=}20$} (Setting~T; same
  readers, seed and 515 audited instances throughout; every row from
  one pre-registered sweep). Default = depth 12; its support
  reproduces the committed audit exactly ($0.070$).}
  \label{tab:ablate}
  \begin{tabular}{@{}llccc@{}}
    \toprule
    Component & Variant & Acc. & Support & $|\Rset|$ \\
    \midrule
    Reason length & tree depth $\le 2$ & 0.184 & \textbf{0.528} & 2.0 \\
                  & tree depth $\le 4$ & 0.295 & 0.307 & 3.9 \\
                  & tree depth $\le 6$ & 0.390 & 0.109 & 5.6 \\
                  & tree depth $\le 12$ (default) & 0.396 & 0.070 & 7.2 \\
    \midrule
    Gate aggregator & min (default) & --- & \textbf{0.667}$^{a}$ & --- \\
                    & product        & --- & 0.583$^{a}$ & --- \\
                    & mean           & --- & 0.583$^{a}$ & --- \\
    \bottomrule
  \end{tabular}
\end{table}

$^{a}$Support at $5\%$ coverage at $K{=}10$; bootstrap LB repair
ratios $4.7/4.5/3.4$ respectively (\cref{sec:abstention}).

\cref{tab:ablate} examines whether our findings depend on design
choices, under predictions fixed before the runs (supp.\
\cref{app:prereg}). Shortening reasons at fixed $K$ is the decisive
knob: within one sweep over the same 515 instances, limiting tree
depth to $2$ moves premise support from $0.070$ to $0.528$---within
one percent of the $0.522$ the independence baseline predicts for
reasons of that length---while accuracy falls from $0.396$ to $0.184$.
This turns the compounding account of \cref{sec:exp-scale} from an
observational into an interventional claim, and the accuracy price is
itself the finding: well-supported reasons come from heads that
classify worse. The trade-off row of \cref{fig:qualitative} is an
event, not an anecdote: over the same $515$ instances, $92$ ($17.9\%$)
have a fully true depth-2 reason that predicts the wrong class while
the default prediction is correct (the $35/36$ successes above are
over the same $515$, from the same per-instance dump). Ranking the gate by the product or the mean of
literal trust instead of the minimum changes nothing for the better
(at $K{=}10$, $0.583$ versus $0.667$ support at $5\%$ coverage)---
failure is a single bad literal, and the minimum is the aggregator
built for it. Design choices we did not test are listed as such in
\S5: predicted localisation (the one claim-relevant unknown;
\S\ref{sec:pipeline} states the assumption rather than claiming it),
backbone swaps on CUB (covered by the two-stack, three-dataset
replication above), and minimum-cardinality reasons (deferred by an
isolated engine defect, supp.\ \cref{app:pyxai}).
```

## §5 Discussion (incl. limitations), §6 Conclusion (v2, 2026-11-15)

```latex
\section{Discussion}
\label{sec:discussion}

\paragraph{What a deployer can hold.}
Two things survive the trip from proof to image, and only two: the
exactness of the certificate, at every scale and for any concept bank
it is given; and, at partial coverage, a calibration-gated reason whose
premises are usually true. Everything else we tested---selection,
stronger readers, cleaner training data, smarter reason
queries---leaves premise support where it was on the main bed. A
practical reading of our tables is therefore a menu of what \emph{not}
to buy: a certified explanation is not evidence about the image at
full coverage, and an accuracy improvement in the underlying model is
not an improvement in its explanations.

\paragraph{Repair must target slots, not aggregates.}
The nulls and the reversals triangulate to one lesson. On CUB, a
stronger reader lifts accuracy by seven points without touching
support, because the yellow-bill shortcut fires exactly as before;
on BDD-OIA, the same intervention works, because its failure is
perceptual and slot-wide. Between datasets, the accuracy--support gap
tracks reader quality ($35/19/19/8$ points as readers improve from
$0.71$ to $0.95$). Reader quality helps when it repairs the specific
failing (concept, polarity) slots---not in the aggregate---so the
audit's concentration finding is also its repair manual.

\paragraph{True premises are necessary, not sufficient.}
The qualitative figure sharpens what ``success'' means: the success
row's premises are all true, and its prediction is still wrong. A
fully supported reason certifies the \emph{tree's} reasoning from true
premises; it does not entail the correct class, because the teacher
and the tree can disagree for other reasons. Premise support is the
binding rung, not the only rung---which is exactly why the ladder
keeps fidelity separate.

\paragraph{The level that uncertainty estimation was missing.}
Position papers argue that prediction-level uncertainty is
fundamentally incomplete \cite{jimenez2026position,aldoma2026position}.
Our gate is a small instance of the level they call for: it scores
each \emph{literal} of a certified reason, not the prediction, and the
diagnosis it enables (raw confidence is anti-informative exactly where
readers are miscalibrated; discrete population-level reliability never
separates) is invisible at the prediction level.

\paragraph{Limitations.}
The audit requires per-image concept ground truth, which is scarce;
our replications span faces and driving actions, and the method
transfers wherever such annotations exist, but the measured rates are
properties of these datasets. The main audit uses oracle localisation;
predicted boxes are untested---the claim that they can only lower the
reported support is an assumption we state rather than measure, and
the one design choice a reviewer could still flip. Also untested:
backbone swaps on CUB (the inversion replicates across two feature
stacks spanning three datasets) and minimum-cardinality reasons
(isolated engine defect). Replications are single-seed with a single
tree depth. The gate's margin thins at the full class count, holding
at some operating points and not others; we do not claim a repair
there---only a measurement and a gate.

\section{Conclusion}
\label{sec:conclusion}

A formal sufficiency certificate is a statement about the symbolic
model; what survives the trip to the image is a measurable, degrading
quantity. We made the certificate's two unstated premises measurable,
found premise support binding, structured, and identifiable across
datasets---concentrated in few (concept, polarity) slots that differ
by domain, and compounding per literal as reasons lengthen with the
task, an account we confirmed interventionally. We measured what
restores it: on the main bed, only calibration-gated abstention,
robustly at intermediate scale, marginally at full scale, always
paying in coverage---and nothing that buys accuracy instead. Even a
fully true reason, we saw, does not entail the correct class. The
audit is cheap---one trained head, certified reasons, held-out concept
annotations---and we believe it should accompany any deployment of
certified concept explanations: the certificate tells you what the
tree knows; only the measurement tells you what the bird does.
```

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
erasure \cite{canizales2026conceptbased} keeps $0.67$ bits on average and is \emph{never}
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
\cref{sec:ceiling}.}
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


## Appendix stubs: app:unresolved, app:beds, tab:replication (added 2026-11-14 night)

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.9\linewidth]{fig_ceiling.pdf}
\caption{\textbf{Vocabulary--vs--perception ladder} (matched
linear/logistic head family; the CART-on-ground-truth-bits collapse
quoted in the text is a different head family and is not plotted).
CUB climbs monotonically to a $0.805$ ceiling---its concepts express
the task; BDD-OIA starts higher and ends near ceiling---its failure is
perceptual, not expressive. The continuous-scores variant was not run
on BDD-OIA (dashed).}
\label{fig:ceiling}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{fig_failures.pdf}
  \caption{\textbf{What failure looks like, and where it concentrates.}
  \emph{Top:} the certified-but-false premises of our running
  example (image \#44)---a beak crop asserted \texttt{bill\_colour=yellow} with zero
  yellow pixels (bed average over firing beaks: $3\%$) and a tail crop
  asserted black; both refuted by the annotations. \emph{Bottom left:}
  each replication bed's top carrier of failing reasons (CelebA: a
  negated attribute; BDD-OIA: top fault slot \texttt{green\_light=1} at
  support $0.62$, alongside the stereotyped co-resident
  \texttt{rider=0} in $76\%$ of failing reasons at support $0.92$; the
  deterministic pick is frame \texttt{6c0cc8ca-e1dd9997\_3}---bdd100k
  frames are not local). \emph{Bottom right:} the share of failing
  reasons containing at least one of the top-$k$ carrier literals---CUB
  K=4 (thick red), the five re-drawn subsets (thin red; the shape is
  stable while the top literal's identity changes), CelebA (grey),
  BDD-OIA (green; its top-$1$ coverage is the stereotyped
  \texttt{rider=0}, a usually-true co-resident---the fault sits in the
  traffic-light slots).}
  \label{fig:failures}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=\linewidth]{fig_beakstrip.pdf}
\caption{\textbf{The yellow-bill shortcut, four beaks} (deterministic
pick: min / quartile / median / max of the yellow-pixel distribution
among the 88 firing beaks). Each crop: species, measured yellow-pixel
share, reader probability, annotation. The reader asserts
\texttt{bill\_colour=yellow} with high confidence on beaks that are
near-entirely non-yellow.}
\label{fig:beakstrip}
\end{figure}

\section{Qualitative-figure selection rules}
\label{app:qualitative}
Rows of Fig.~\ref{fig:qualitative} are fixed before inspection, from
the per-instance depth dump (\texttt{ablation\_depth\_K20\_perinstance.%
json}; the full $515$-instance $K{=}20$ audited set---no subsample; its
default-depth aggregate reproduces the committed sweep exactly).
(a)~the median-length failing default (depth-12) reason;
(b)~among instances whose depth-2 reason is fully true, whose depth-2
prediction is wrong, and whose default prediction is correct
($92$ instances), the median-length depth-2 reason; (c)~among fully
supported \emph{and correctly predicting} default reasons citing at
least one positive literal ($19$ instances qualify), the median-length
reason. Ties in all cases are broken by the middle instance index; the
selected instances are $\#513$, $\#714$ and $\#603$.

\section{Unresolved-literal sensitivity}
\label{app:unresolved}
Under the convention of \cref{eq:support}, concepts on invisible parts
count as absent: their negated literals are supported and their
asserted literals refuted. Under the stricter rule---score only
reasons whose every literal sits on a visible part---all $30$ passing
four-class reasons still pass, so the concentration finding does not
depend on the convention. The unresolved fraction reaches one half of
literals at the larger class counts; the strict variant at $K\ge10$
is left to future runs.

\section{BDD-OIA per-action detail}
\label{app:beds}
Per-action head accuracies on BDD-OIA (rsbench splits, seed 42):
\textsc{forward} deployed tree $0.716$, direct feature probe $0.753$,
ground-truth-bits teacher $0.986$; \textsc{stop} $0.747/0.755/0.996$;
\textsc{left} $0.730/0.746/0.913$; \textsc{right}
$0.685/0.731/0.979$. The training-side reversals quoted in
\cref{sec:ceiling} (stronger readers $0.534\to0.628$;
certified-vocabulary retraining $\to0.786$ with a reshaped certificate
population) and the certificate-level escape replications (oracle gain
$0.0003$; uncertainty-aware $\equiv$ ordinary on $98.3\%$) are from
the same committed artifact (\texttt{bdd\_oia\_experiments.json}).

\begin{table}[h]
\centering\small
\caption{\textbf{Replication summary} (same readers, heads,
extraction and scoring as CUB; single seed).}
\label{tab:replication}
\begin{tabular}{@{}lccc@{}}
\toprule
dataset & $n$ (reasons) & accuracy & premise support \\
\midrule
CelebA  & 6{,}654 & 0.879 & 0.693 [0.682, 0.704] \\
BDD-OIA & 6{,}433 & 0.69--0.75 & 0.534 [0.522, 0.546] \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[h]
\centering
\includegraphics[width=0.7\columnwidth]{fig_slope.pdf}
\caption{\textbf{Vocabulary--vs--perception ladder} (matched
linear/logistic head family; the CART-on-ground-truth-bits collapse
quoted in \cref{sec:ceiling} is a different head family and is not
plotted). CUB climbs monotonically to a $0.805$ ceiling---its concepts
express the task; BDD-OIA starts higher and ends near ceiling---its
failure is perceptual, not expressive. BDD continuous rung not run.}
\label{fig:slope}
\end{figure}
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
reachability) and cross-check with PyXAI~2.0.1 \cite{audemard2024pyxai}. On the main
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
Timestamped record: the criteria entered repository history before any
run executed (commit \texttt{848faba}, 2026-11-11: criteria and
decision rules; ablation predictions in \texttt{2f88002} precede
ablation results in \texttt{9905d62}); an anonymised OSF registration
mirrors the same text for review.
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

---

## Changelog 2026-11-14 (latest) — §3 v5 + §4 v4: user's CVPR rewrites adopted, ALL TODOs/\todo filled; DINOv3→DINOv2 CORRECTED

**Factual correction:** the committed pipeline's backbone is **DINOv2 ViT-S/14** (`dinov2_vits14`, 384-dim — `perception/backbone.py`, configs), not DINOv3 as earlier drafts claimed. Fixed in §3.1 (cite `dinov2`), fig_pipeline + fig1_v2 strips (re-rendered, re-committed). This also reframes the advisor's backbone ablation: DINOv2 IS the default; CLIP/supervised remain untested (one sentence in §4.7: three feature stacks already covered by the replications).

**§3 v5** = user's rewrite ("Auditing certified concept reasons", levels-of-guarantee table `tab:levels`, protocol moved to §4.1) with every TODO filled: (i) support-vs-accuracy = one definitional sentence + the ≤15-word rationale ("a reader expects an explanation to be at least as reliable as the prediction it explains"); (ii) independence NULL renamed **independence baseline**, three-case reading cut to one sentence (ratio > 1 = failures share literals — predictable compounding with repair targets), CUB-vs-BDD finding stays in §4; (iii) vocabulary ceiling = one sentence + pointer to tab:ceiling; (iv) gate channels as a run-in list, "honest-deployment arm" jargon dropped, repair criterion moved to §4.1 with a one-line pointer; (v) oracle-localisation sentence now says the decrease claim is an ASSUMPTION we flag rather than claim. Label aliases kept (sec:ladder→sec:guarantees; tab:family→tab:families).

**§4 v4** = user's experiments rewrite, all \todo filled: tab:ceiling CUB (K=4) complete; BDD column filled from committed `bdd_oia_experiments.json` stage6_ladder (predicted bits 0.72 / features 0.75 / GT bits 0.99 forward-action; continuous rung honestly "---"); **tab:ablate completed with a NEW RUN**: depth sweep at K=20 (`ablation_depth_K20.json`; script now takes K from argv) — depth ≤2: acc 0.184, support **0.528** (baseline predicts 0.522 — within 1%), |R| 2.0; ≤4: 0.295/0.307/3.9; ≤6: 0.390/0.109/5.6; the "raises support to \todo vs \todo predicted" sentence is filled with 0.528 vs 0.522; predicted-box and min-cardinality rows filled honestly as NOT RUN (flagged/deferred, engine defect noted); backbone rows removed (sentence instead). fig:scale/fig:abstain environments restored. New supp stubs: `app:unresolved` (strict rule: 30/30 at K=4; K≥10 future), `app:beds` (BDD per-action ladder + reversals), `tab:replication`. `sec:audit` label preserved on the concentration paragraph (intro ref). cleveref now required (\cref) — preamble note.

**Flags:** (21) fig:failures (concentration heat-strip) still to build — §4.3 references it; (22) cleveref in preamble; (23) DINOv2 correction — sweep ALL drafts for stray "DINOv3" at assembly (intro/§2 clean; checked).

---

## Changelog 2026-11-14 (final figures) — the figure set COMPLETE per the advisor plan

Built tonight, all from committed JSONs, all scripts + PNG/PDF committed: **fig_failures** (`plot_fig_failures.py`: sid-44 beak/tail crops — beak measures **0% yellow pixels**, stated alongside the 3% bed average; CelebA/BDD top-carrier cards, images not local; concentration curves CUB K=4 + 5 redraws (thin) + CelebA; BDD has no per-instance records, card instead) — resolves the last unresolved §4.3 ref and kills flag 21. **fig_abstain** (`plot_fig_abstain.py`: K=20 + K=200 panels, channels B/C/A_ir, R and 2×R lines, 0.14 operating point circled; K=200 panel shows the fragility ourselves) — kills flag 16; caption updated (CUB K20+K200, not three beds — CelebA/BDD curves remain supplement-tier). **fig_scale** now log-y (halving = straight line; null markers overlap visibly). **fig_mechanism** (`plot_fig_mechanism.py`: K=20 depth, support rides the baseline; d=2 annotated with acc) wired into §4.7. **fig_slope** (SUPPLEMENT; matched-family caption resolves the 0.805-vs-collapse wrinkle: different head family, not plotted). Advisor questions answered: fig4 = mechanism (the explanation is the spine; the diagnostic is tab:ceiling); deterministic pick rule = Fig-1's rule extended per bed (median-length failing reason citing the bed's top carrier, middle id).

---

## Changelog 2026-11-14 (final, per user) — fig 1 + fig:pipeline DROPPED from the paper; fig_failures v2 with a real CelebA face

User: "i dont need fig 1 and fig pipeline." Removed from the paper: the teaser figure environment and its two intro pointer sentences (¶1 pipit-contrast sentence deleted; intro prose intact otherwise), and §3's opening no longer references any figure ("Our setup is a four-stage pipeline..."). fig:failures caption reworded ("our running example (image #44)" instead of Fig.~\ref{fig:gap}). The artifacts (fig1_v2, fig1_gap, fig_pipeline + scripts) REMAIN COMMITTED as available-if-wanted; no text references them. fig_failures v2: CelebA panel now shows a REAL FACE — annotation-selected (middle test-partition image whose attributes AFFIRM High_Cheekbones, the state the top failing carrier literal denies), because the audit's identity-disjoint parquet export retained no idx→filename mapping (verified: Smiling-gt agreement against the archive order is chance, 3,384/6,654; custom 17,134/6,623/6,654 split). The panel title states both the carrier share (50% of 2,042 failing reasons) and the selection rule. BDD-OIA keeps its card: no frames local (bdd100k not downloaded) and no per-instance records committed — the only honest option. Layout now: [beak | tail | BDD card] / [CelebA face | concentration curves].

## Changelog 2026-11-15 — RSBench per-instance FOUND in the committed audit; fig_failures gains the BDD line

User: "check if we could use RSBench results." Answer: YES — `bdd_oia_audit.json` carried per-instance records all along under the key `reasons` (6,433 records; my earlier schema probe looked for `per_instance` and wrongly concluded none existed — corrected; the fig caption and prior changelog claims about "no per-instance records" are fixed). New from those records: BDD concentration curve added to fig_failures (containment rule, same as CUB/CelebA); top fault-weighted slot green_light=1 (support 0.615) vs the most CONTAINED literal rider=0 in 2,288/2,996 = 76% of failing reasons at support 0.918 — BDD reasons are short and stereotyped, the fault sits in the traffic-light slots (caption states both). Deterministic qualitative pick (median-length failing reason citing green_light=1, middle idx): action forward, test idx 2284, frame 6c0cc8ca-e1dd9997_3.jpg, reason [green_light=1, rider=0, other_obstacle=0] — frame name from the local Zenodo bdd2048 bundle's sorted inputs listing; the image itself still needs bdd100k (registration). BDD card in fig_failures now data-rich (fault slot + stereotyped literal + named pick frame).

---

## Changelog 2026-11-15 — CITATION PASS: paper/refs.bib installed (user-supplied, 119 entries + 11 added); §2 rewritten with "We add" positioning; all cite keys re-keyed to the bib

`paper/refs.bib` = the user's pasted bib verbatim + an "added for this paper" block (cub, celeba, bddoia, rsbench, dinov2, yuksekgonul2023posthoc, margeloiu2021how, mahinpei2021, marconato2024, geifman2017selective, angelopoulos2023conformal; TODO-VERIFY marks on memory-written fields: bddoia authors/pages, rsbench venue/authors, marconato2024 venue/year, margeloiu author list). §2 rewritten in three paragraphs, each ending with an explicit \emph{We add} clause: (1) CBMs — the question one step up from layer-faithfulness (what survives a certificate over the layer); (2) formal explanations — the mirror-image certificate (symbolic head ⇒ unconditional exactness, unique reasons) + premise measurement their settings cannot express (vs Canizales network-oracle/monotonicity/LLM-judge; vs Teso faithful-and-compact axis: we measure true-of-the-image); darwiche2020reasons credited for the sufficient-reason vocabulary; selvaraju2017gradcam one-clause contrast; (3) trust/calibration — the literal-level gate inside a certified explanation, with the jimenez2026/aldoma2026 "prediction-level UQ is incomplete" positions cited as exactly the gap our per-literal gate instantiates. All \cite keys in intro+body now resolve against refs.bib (verified by diff; re-key map: pyxai→audemard2024pyxai, teso→teso2026when, soria→soria2026formal, canizales→canizales2026conceptbased, huang→paul2024formal, ignatiev2019abduction→ignatiev2018abductionbased, margeloiu2021→margeloiu2021how, zarlenga2022cem→zarlenga2022concept, bortolotti2024rsbench→rsbench). Unused-in-text bib entries (uncertainty/active-learning/partial-ID families) stay in refs.bib — available for the supplement. NOTE: bortolotti2026autoformalizing (LLM constraint formalization) is in the bib but not yet cited — candidate for §2 ¶1 one-clause or the supplement; left out to keep §2 tight.

---

## Changelog 2026-11-15 — full review pass applied (advisor check of every number vs RESULTS_SUMMARY)

**Errors fixed:** (1) DINOv3/DINOv2 — verified clean in prose already (only changelogs mention the old name; the reviewer's copy predates the fix). (2) **tab:ablate rebuilt self-contained**: every row from ONE pre-registered sweep over the same 515 instances (default = its own depth-12 row, acc 0.396, whose support 0.070 reproduces the committed audit exactly); depth-4/6 rows now shown (0.307/0.109); the K=200 subsample caveat correctly applies ONLY to the older K-sweep table, not K=20 (n=515 identical; verified in JSON). (3) Scope phrase adopted: **"On CUB, abstention is the only intervention that raises premise support without sacrificing accuracy"** + the two pay-elsewhere clauses (depth pays accuracy; training-side works only on the perception-limited bed). (4) "three feature stacks" → two stacks spanning three datasets. (5) Forcing scoped to "the class counts we test (four and ten)". (6) Raw-vs-random labelled Setting D, K=10, and tab:gate gains the Setting-D row (0.697 [0.52,0.85], LB 4.2 — computed from abstention_curves.json). (7) "matches or exceeds" (BDD raw 0.817 beats calibrated 0.755 — stated, strengthens the diagnostic). (8) K=4/10 joints FOUND in cbm_cem_audit.json (0.254/0.070) → tab:scale gains the Joint column at every K; fidelity claim scoped to "wherever both factors are measured". (9) tab:nulls caption scoped; certified-only row carries numbers (0.254 unchanged; yellow-bill still 0/88).
**Unused results added:** K=10 gate-vs-oracle (0.697 vs 0.727 — now the lead sentence of the gate subsection); aggregator rows replace the not-run rows in tab:ablate (footnoted K=10 @5% cov); the methods-comparison mechanism sentence (random-3 0.682 vs 0.534 — price of informativeness) in the concentration paragraph; the reader-strength gradient (35/20/19/8 pts) + its reconciliation with the CUB MLP null in sec:ceiling (between-dataset vs within-bed; the MLP didn't touch the failing slots).
**Wording:** "rung"→variant, "audit"→sweep, "bed"→dataset, "rides"→tracks; sec:audit → sec:exp-conc (intro ref updated); garbled detector sentence replaced by the §5 not-run list; leading-zero 3-decimal numbers standardized in tab:scale; Q1–Q4 one-sentence answers close their subsections.
**Repair:** §5/§6 had been accidentally deleted by the §4 v4 span replacement — RESTORED (limitations now carries the not-run list: predicted localisation as the flagged assumption, backbone swaps, min-cardinality).

---

## Changelog 2026-11-15 (camera-ready figures) — the advisor's styled figure script, LOADER-WIRED and run

`scripts/plot_figs_v6.py` = the advisor's script (styling, panels, CVPR column width) with its own TODO honoured: every number loaded from the committed JSONs (fig_scale ← 3 audit JSONs + 2 independence JSONs; fig_abstain ← abstention_curves + task1_abstention files; fig_ceiling ← design_b_v2 ladder + bdd stage6; fig_methods ← bdd_oia_methods part1; fig_readers ← all four bed audits). Outputs to results/figures (not /home/claude). Regenerated fig_scale/fig_abstain (restyled) + new fig_ceiling/fig_methods/fig_readers; fig_slope superseded by fig_ceiling (kept on disk). Loader caught a data bug the hard-coded version hid: the BDD reader-gradient gap must average the FOUR ACTIVE actions (43.6 pts if the excluded fifth's ~0.997 teacher accuracy leaks in); correct values 34.8/18.7/18.6/8.3 → gradient stated as 35/19/19/8 in §4.5 and RESULTS_SUMMARY §9 (was 35/20/19/8). fig_readers footnote: SUN marked $^*$ (carved split).

---

## Changelog 2026-11-15 (ablations figure) — fig_mechanism rebuilt camera-ready, loader-wired

`plot_figs_v6.py` extended with the ablations figure (was hard-coded in the older style): support vs |R| across tree depths at fixed K=20, orange points labelled by depth with the accuracy in small grey beneath, dashed ink curve = the independence baseline at the same |R| — loaded from `ablation_depth_K20.json`. `save()` in the script is now skip-existing (never overwrites; delete a file to regenerate it), so the whole set is idempotent. Old `plot_fig_mechanism.py` output superseded (script kept). The aggregator stays in tab:ablate per the advisor's guidance (table rows, not a figure).

## Changelog 2026-11-15 (qualitative figure + beak strip) — advisor layout implemented; row-(c) question answered empirically

**Per-instance dump EXECUTED** (`ablation_depth_K20_perinstance.py` → `ablation_depth_K20_perinstance.json`, new file): K=20 × depths {2,4,6,8,12,16}, per instance sid/pred/correct + literals (feature, bit, supported, part_visible); default-depth aggregate reproduces the committed sweep exactly (0.070). **Row-(c) answer: 35 of 36 fully-true default K=20 reasons cite ≥1 positive literal** — the absence-only character of the K=4 successes does NOT persist at scale; the rule requires a positive literal and one exists (sid 462). Row picks (fixed rules, middle sid at median): (a) sid 513 (7-literal failing default), (b) 92 candidates → sid 714 (depth-2 all-true wrong-class vs correct default), (c) sid 462. **fig_qualitative** rendered per the advisor layout (image with worst-status boxes + 4 depth cells with ✓/✗/∘, footer pred/|R|/all-true below the box; QA-passed). **fig_beakstrip** (supplement) from committed yellow_bill_audit.json: 4 firing beaks at min/quartile/median/max yellow-pixel share with species, %, reader prob — decision: SUPPLEMENT (main text keeps the single 0%-crop in fig:failures + the 3.3% sentence + the new pointer). New §4.3 sentence notes the reader fires at as little as 0% yellow.

## Changelog 2026-11-15 (page budget + the two new findings folded in)

**Denominators (one file, `ablation_depth_K20_perinstance.json`, the full 515-instance K=20 set — no subsample):** row-(b) event = **92/515 = 17.9%**; fully supported default reasons = **36/515**, of which **35** cite ≥1 positive literal. Both now stated together in §4.7. **§4.3 concentration ¶** rewritten per the advisor: absence-only scoped to K=4 with the 35/36 at K=20 as the contrast finding (points at fig:qualitative c); the strict-rule clause survives attached to the K=4 thirty. **Intro** finding 2 scoped likewise, number-free ("a property of the smallest subset, not of the method…"). **Beak wording** fixed: "fires even on beaks with no yellow pixels at all—mean 3.3% across firing beaks". **PAGE BUDGET — two figures moved to the supplement: fig_failures and fig_ceiling.** Rationale: fig_qualitative row (a) now carries the typical-failure-with-boxes role; fig_failures' unique cargo (cross-dataset concentration curves + bed carriers) is fully quantified in §4.3 text, so its main-text sentence = the existing numbers sentence + "(supp. Fig. failures)"; fig_ceiling restates tab:ceiling, so its main-text sentence = the existing §4.5 opener + "(visual in supp.)". Main text now: 5 figures (fig_scale, fig_methods, fig_abstain full-width, fig_qualitative full-width, fig_mechanism) + 5 tables + tab:levels. Note: fig:abstain caption still describes the pre-restyle panel — flagged for .tex assembly.

## Changelog 2026-11-15 — fig_qualitative caption v2 (advisor's camera-ready text)

n filled = **515** (the full K=20 audited set; same file as the 35/36 — no subsample; stated in both caption and app:qualitative). Caption claims verified against the dump: row (a) sid 513 default pred CORRECT + premises false ✓; row (b) sid 714 d2 wrong/default correct ✓; row (c) sid 462 all-supported but pred WRONG → added the honest clause "(here the prediction itself is wrong---full premises do not entail a correct class)". New supp stub `app:qualitative` (fixed selection rules + tie-breaks + row sids + provenance) resolves the caption's \cref. Preamble needs: \usepackage{pifont} (dings) + color defs ok/bad (= green/red).

## Changelog 2026-11-15 — §5 Discussion (new) + §6 Conclusion v2

§5 is now a five-paragraph Discussion that folds limitations in: (i) what a deployer can hold (exactness + gated reasons at partial coverage; the menu of what NOT to buy); (ii) repair targets slots not aggregates (MLP null vs BDD reversal vs between-dataset gradient — the audit as repair manual); (iii) true premises are necessary, not sufficient (the qualitative success row predicts wrong; premise support is the binding rung, fidelity stays separate); (iv) the per-literal gate as the level the UQ position papers call for (jimenez/aldoma cited); (v) limitations (oracle-localisation assumption flagged-not-claimed, two stacks, single-seed, gate thinning at 200). §6 Conclusion v2: survival summary + the interventional confirmation + "fully true ≠ correct class" + closing line "the certificate tells you what the tree knows; only the measurement tells you what the bird does." All claims trace to committed JSONs; no new numbers beyond those already in the ledger.

## Changelog 2026-11-15 (tier-review response) — strict-scoring line, true-success row (c), abstract v2, Teso difference, prereg receipts, caption fixes

**Strict scoring COMPUTED** (`strict_support_cub.py` → `strict_support_cub.json`, committed): strict = convention at K=4/10/50/200 (0.254/0.115/0.035/0.017); only K=20 moves (0.070→0.054). Wired as the dotted line in fig_scale + one §4.2 sentence + fig:scale caption clause — the "convention inflates support" attack is now answered in the headline figure itself. **Row (c) fixed**: select_rows now requires all-supported AND correct AND ≥1 positive literal (19 qualify; pick sid 603 by the median/middle rule); literal labels de-truncated (full attribute + part names); caption (c) rewritten as a true success with the 19/36-vs-35/36 numbers; app:qualitative updated. **Abstract v2** written into plan §9 (survival voice, number-free, scoped raw-confidence claim, polarity-domain correction, and the two non-obvious findings led with: sufficiency's own cost — unguaranteed explanations are more often fully true — and the compounding law confirmed interventionally). **§2**: explicit Teso difference (architecture-level faithfulness/compactness vs our image-truth axis; their sparsification lever = our accuracy-price finding). **app:prereg**: timestamped receipts (commits 848faba → 2f88002 → 9905d62; OSF mirror note). **fig:abstain caption** finally matches the two-panel design; sec:beds dangler (app:methods tab:families) → sec:ceiling. Combined-document ref check: all paper-text refs resolve.

## Changelog 2026-11-15 (prediction-confidence baseline EXECUTED) — the selective-prediction attack closed

`ablation_gate_predconf.py` → `ablation_gate_predconf.json` (committed): K=10 Setting T, task3 protocol (sanity: n=243, R=0.115 = committed exactly). Channels per instance: B calibrated min-p_true, **E = max tree class probability (== leaf purity for a CART; the standard selective-prediction score; Setting-D definition identical — proportions of teacher labels)**, C raw, A oracle marginal. Result at 13.6% coverage: **E = 0.212 [0.11,0.38], LB ratio 0.8 (≤ random-level) vs B = 0.727 [0.56,0.85], LB 4.7**; E never clears the 2× bar at any coverage. fig_abstain panel (b) gains the K=10 (T) row with the gate (orange dot) beside the prediction-confidence baseline (blue ×) over the same 2× bar — the visual answer to "abstention is just selective prediction". §4.6 sentence added. Advisor TODO resolved: for a CART, max class prob and leaf purity coincide (predict_proba = leaf class proportions), so the "fairer baseline" question is moot; the instance-constant score is exactly ranking by prediction confidence.

## Changelog 2026-11-15 (prioritization pass) — decision 1 settled: the compounding law leads

Advisor's order (decide 1 → run 2 → 3 → Tier 0 + abstract → 4): items 2 and 3 were ALREADY EXECUTED (pred-conf channel: E 0.212 LB 0.8 vs gate 0.727 LB 4.7 at K=10/13.6%, `109d45a`; strict scoring: = convention at 4/5 K, dotted line in fig_scale, `670e9e7`). **Decision 1: the compounding law carries the abstract and the contribution order** (prediction-then-intervention beats single-dataset observational surprise; the K=4 degenerate point stated AS the concentration finding, not hidden). Abstract v2 reordered (law first, degeneracy clause added, sufficiency-cost second as cross-dataset corroboration); intro finding 1 gains the interventional clause (number-free). **Item 4 = SCOPED, not run** (no time before Nov 16): Discussion now states the simple pipeline, the 35→8-point gradient, its 15–20-point extrapolation to modern-concept-model reader strength, and that the extrapolation—not a measurement—is what a strong-model run would test; run targeted for the rebuttal window if attacked. The min-operator observation on the pred-conf channel: an instance-constant score ranks reasons, not literals — part of the point, now implicit in §4.6's "blind to which literal fails". Remaining for the assembly: re-export .tex from the wiki drafts (kills Tier 0), pifont/cleveref preamble, fig:abstain caption done.
