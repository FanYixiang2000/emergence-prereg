# Manuscript draft (target: Nature Machine Intelligence)

Status: full skeleton with real numbers; every claim is backed by a file
in `outputs/` and a figure in `figures/`. Claims are deliberately scoped
per the positioning section at the end. Chinese notes in brackets mark
writing decisions still open.

---

## Title

**Counterfactual tests distinguish useful emergence from abrupt change
in learning systems**

This scoped title replaces the earlier ontological “Emergence as ...”
formulation.

## Abstract (~170 words)

Emergent abilities -- capabilities that appear abruptly and were not
explicitly trained -- are reported across multi-agent systems, neural
network training, and large language models, yet the field measures them
through signatures that disagree: performance jumps, representation
jumps, information-theoretic synergy, causal efficacy of macro-variables.
Here we show that the signatures audited here are projections of a measurable
quantity: the distribution over future outcome basins, P_t(B | s_t), and
that emergence corresponds to a specific event in this distribution -- a
selective, endogenously triggered, locally costly, counterfactually
necessary collapse toward high-value basins. We operationalize this with a
six-component episode-level criterion and a separate four-component
training-process proxy, freezing thresholds before confirmatory tests.
Evidence spans gridworld and neural multi-agent systems, grokking in three
architecture families, induction-head formation, public MultiBERTs and Pythia
pretraining series (encoder and decoder, neither trained by us), and 240
externally annotated sacrificial key moves in human chess games. Across their
declared scopes, these instruments accept documented cases, reject reward
shaping, noise, traps and gradual accrual, and identify expert key moves that no
single prior signature identifies. Pre-registered throughout, including
thirteen registered misses, two of which expose a methodological
principle: in converged state-action systems, useful emergence is visible through
counterfactual intervention, not through observed improvement.

## Main text structure (the four questions, one line each)

1. **Why are existing signatures not enough?** They disagree with each
   other and each admits named false positives (Fig. 1, Fig. 2).
2. **What is the mechanism?** Potential-Trigger-Collapse on P_t(B|s_t),
   with a six-component registered criterion (Box 1).
3. **Does it separate emergence from its imitations?** Yes -- battery of
   named counterexamples + prior-detector comparison, internal and
   external (Fig. 2, Fig. 5 right).
4. **Does it hold beyond systems we built?** Yes -- grokking/induction,
   the public MultiBERTs series including a rejection case, the public
   Pythia decoder series with a full accept/reject double dissociation,
   chess key moves, deep MARL (Figs. 3-6).

---

## Introduction argument chain (the five WHYs, each with its witness)

This is the load-bearing argument of the paper. Every step names the
prior definition it addresses and the measured witness. [引言按此顺序
展开，每步一段；数字全部来自 outputs/，不允许出现无 witness 的批评。]

**WHY-1. Why the field needs a definition at all.** The communities do
not merely use different estimators -- they disagree on verdicts. Wei et
al. call LLM abilities emergent; Schaeffer et al. call the same curves
metric artifacts; neither side has a measurement that could settle it,
because both read performance curves, and a curve contains no
information about where the ability came from. Meanwhile causal
emergence and PID-synergy verdicts are computed on systems their
authors built, and have never been asked to rank anything in the wild.
A definition must be able to (a) disagree with a performance curve and
(b) be checked against external ground truth. None of the existing ones
can do either.

**WHY-2. Why each prior definition fails -- not rhetorically, but on
the same data.** The failure pattern is one theorem and one table.
Theorem (Prop. 2, THEORY.md): no single observable of the collapse
event determines the verdict, because for each single observable we
construct two systems equal on it with opposite truth. Table (measured,
hindsight-optimal thresholds -- i.e. giving each rival its best case):

- *Representation jump* ("a point in representation space jumps to
  another point; farther = stronger emergence"): jump distance is
  direction-blind, provenance-blind, and cannot order strength. Three
  measured witnesses. (a) DIRECTION: the harmful decoy has the exact
  same jump (0.102) as the accepted conditional positive, because they
  share a policy -- but one lands at usefulness gap +3.43, the other
  at -8.39; a norm carries no sign. (b) RANKING INVERSION: the accepted
  positive's jump (0.102) is SMALLER than reward-shaped coordination
  (0.403) and ordinary converged team reward (0.328); ranking by
  distance puts the genuine case near the bottom, and even the
  hindsight-optimal threshold caps at 0.9 with the miss being
  precisely the accepted conditional positive. (c) STRENGTH: the noisy learner --
  which the behavioral audit shows acquired the SAME selective
  structure as the clean one -- posts a jump of 1.399, 13.7x larger;
  "farther = stronger" would call it 14x more emergent for the same
  ability, i.e. the scale measures reward noise, not emergence.
  Theoretically collapse upper-bounds the jump (Pinsker; equivalently,
  an observed jump lower-bounds the required distributional movement;
  Prop. 1): it certifies that the distribution moved, never where it
  landed nor who moved it. On external data the unsigned-distance
  reading collapses entirely: it ranks the annotated chess key move
  first in 5% of positions.
- *Performance jump*: usefulness without provenance; accepts the
  reward-shaped system, misses latent conditional ability (hindsight
  acc. 0.8).
- *PID synergy / information-decomposition emergence (Rosas et al.)*:
  joint structure without a value sign. Beyond the sampled synergy
  proxy (best operating point DIRECTION-INVERTED, acc. 0.9), we
  computed Rosas' own practical criterion Psi EXACTLY -- zero
  Monte-Carlo error -- on the enumerated policy-closed chains,
  maximizing over four supervenient features and two micro
  decompositions (the rival's best case). The published verdict
  Psi > 0 scores 0.3: its TOP scorer is `wrong_selector` (+0.59,
  the system that triggers in exactly the wrong mode) and it misses
  `noise_policy`; the hindsight-optimal threshold reaches 0.9 only
  by inverting the theory's own sign convention, and still misses
  the accepted conditional positive (`exact_prior_formalisms.py`).
- *Causal emergence (Hoel EI)*: macro-efficacy without endogeneity.
  The charitable episodic proxy already misfired (top scorer +0.933
  is the reward-SHAPED system -- macro interventions work great
  precisely when a designer forced the macro channel; acc. 0.9).
  Hoel's EXACT EI (max-entropy interventions on the enumerated
  micro TPM, five candidate coarse-grainings, best taken) is
  strictly worse: micro EI is 9.6-11.3 bits, no macro beats it
  (CE < 0 for all 10 systems), and the hindsight-optimal threshold
  in either direction is the trivial all-negative classifier
  (acc. 0.8, missing BOTH true positives); the ordering also puts
  `blind_trigger` past `latent_conditional`.
- The full criterion, with thresholds frozen BEFORE measurement,
  scores 1.000 on the same battery; the exploratory chess move-ranking
  diagnostic is excluded from the main claim because its common baseline
  does not test the martingale lesson.

**WHY-3. Why our definition is TRUE (in the only sense a definition can
be).** A definition cannot be proven; it can be shown to (i) recover
the cases the field already agrees on, (ii) reject the named imitations,
(iii) survive systems whose training its authors do not control, and
(iv) make prospective within-family predictions. Scoped instruments
address all four: process proxies recover grokking, induction-head and
public-checkpoint transitions; episode instruments reject reward
shaping, noise, traps, prewiring, gradual accrual -- each a named
counterexample, including both frequency-tail families on the public
decoder; public-model instruments transfer to MultiBERTs (5 seeds) and
Pythia without control over training, while chess is a selection-bounded
event test; a phase region was derived before a confirmatory same-family
sweep. Thirteen recorded misses remain in place (full index:
PREDICTION_LEDGER.md); heterogeneous ledger rows are not independent
inferential units.

**WHY-4. Why our definition is GOOD (what it buys that others cannot).**
(a) It is counterfactual: verdicts come from do-contrasts, so it can
overrule a performance curve -- which is exactly what settles
Wei-vs-Schaeffer-type disputes; our tail-facts result shows the
criterion confirming the metric-artifact critique from the mechanism
side. (b) It is directional: useful and harmful collapse get opposite
verdicts from the same |jump|. (c) It is compositional: the six
components are conjunctive. Ablations uniquely support selectivity,
usefulness and endogeneity; dedicated controls/refinement support the
other components. (d) It is
estimator-robust where it claims to be (two 12-cell robustness grids)
and honest about where it is not (Potential's absolute scale is
observer-dependent). (e) It prices in the martingale caveat: in
converged systems, observed improvement is structurally silent, and
only the do-contrast speaks -- a principle two independent domains
forced on us.

**WHY-5. The contribution, stated as deliverables.** (1) A common
substrate (P_t(B|s_t)) of which the existing signatures are measured
projections, with each projection's blind spot exhibited on shared
data -- for the two formal rivals (Hoel EI, Rosas Psi), equations are
computed without sampling error within declared candidate families
(scoped derivability result, Prop. 5). (2) A six-component falsifiable criterion with frozen thresholds
and public code. (3) The martingale/do-contrast principle for auditing
converged systems. (4) A within-family prospective parameter methodology
(phase boundary). (5) A prospectively frozen negative-control practice (named counterexamples,
registered failures) that the emergence literature currently lacks.

[对应审稿人一句话总结：prior definitions ask "did something jump?";
ours asks "did the system itself, at a cost, collapse an open future
onto value -- and would the future have stayed open otherwise?" 前者是
运动学，后者是因果动力学。]

## Box 1: the framework (formulas, frozen thresholds)

Root definition (trajectory space). Let tau be a system trajectory with
prior law P(tau) -- the open possibility space -- and M the macro-
structure that materializes. The collapse carried by structure m is

    C(m) = KL( P(tau | M=m) || P(tau) )          [bits]

with two exact identities (Prop. 0, THEORY.md):

    E_m[C(m)] = I(tau; M)                        (average specific
                                                  information)
    C(m) = -log2 P(A_m)   for deterministic M    (event surprisal)

The second identity shows that a broad, high-mass readout carries little
trajectory information (measured: "failed_noise" 0.06 bits untrained).
It does not certify endogeneity: a designer can hardwire a behavior that
is rare because its context is rare. Endogeneity and acquisition therefore
remain separate audited components. Measured anchor: the sacrifice-rescue
structure has 7.16 bits of initial surprisal under the untrained policy;
training raises its mass to the ecological rate P = 1/3, a 5.58-bit
log-likelihood gain.

Observer possibility space (the estimator). Trajectories are not
enumerable in real systems, so fix an observer: a set of outcome basins
{B_1..B_K} = g(tau) (macro-outcomes of an episode, checkpoint, or
position) and a rollout policy for estimating futures. By the
data-processing inequality, basin-level KL/JS contrasts lower-bound
their trajectory-space counterparts (Prop. 0c, verified: basin JS
0.90 <= trajectory JS 1.00). Entropy drops do not inherit this bound
and remain partition-scale dependent. All quantities below are plug-in
estimates from N rollouts.

    P_t(B | s_t)        future-basin distribution at state/time t

    Potential_t   = H(P_t(B | s_t))                        [bits]
    Collapse(a)   = H(P(B | s)) - H(P(B | s, do a))        [bits]
    Specificity   = JS(P(B | s, do a), P(B | s, do a'))    [bits]
    Usefulness(a) = P(B_win | s, do a) - P(B_win | s, do a')
    LocalCost(a)  = immediate return of a relative to greedy alternative

Emergence (registered criterion): an ability/transition is emergent iff

    (1) Potential:    the future was genuinely open before the event
                      (H_pre >= threshold)
    (2) Selectivity:  collapse is conditioned on context/trigger, not
                      unconditional (per-context separation)
    (3) Specificity:  triggered futures differ from non-triggered ones
    (4) Usefulness:   the collapse lands in high-value basins under the
                      realized context (do-contrast > threshold)
    (5) Endogeneity:  the trigger is chosen by the system, not scripted
                      or reward-forced
    (6) Acquisition:  the structure was learned (gain over own
                      initialization), not prewired

Thresholds were frozen in `grokking_collapse_bridge.THRESHOLDS` and the
per-domain preregistrations before each measurement, and never adjusted
in response to outcomes. [Box 引用 THEORY.md 六命题：Prop 0 轨迹空间根
定义（MI 恒等式 + 稀有律 + 数据处理下界）；Prop 1 Pinsker 上界连接
collapse 与表征跳变；Prop 2 无单一观测量充分（含测量见证）；Prop 3
usefulness 恒等式（驱动相界预测）；Prop 4 plug-in 估计一致性；Prop 5
覆盖定理——每个已发表定义都是根对象的可推导投影，且投影严格有损（精确
形式见证）。推导链：0 奠基并授权盆级估计 → 1 把表征跳变降级为有界症状
→ 2 排除一切单信号定义 → 3 给出选择性记账与相界 → 4 控制估计误差 →
5 关闭"稻草人"缺口，统一主张对原始形式成立。]

## The unification table (main text, Q1/Q3)

Prior signatures as projections of P_t(B), each with a measured witness
where the projection fails (all numbers from our runs):

| Prior signature | What it reads off P_t(B) | Measured blind spot (witness) |
|---|---|---|
| Performance jump (Wei et al.) | usefulness only, no provenance | accepts `shaped_process` (reward-forced); misses `latent_conditional` (0.800 hindsight-optimal acc. on battery) |
| Representation jump | projection bounded by collapse (Prop. 1) | ordinary team reward produces large jumps (fig14); 0.900 acc. |
| PID synergy / Psi (Williams-Beer / Rosas, EXACT Psi computed) | joint structure of the collapsed state | exact Psi > 0 (the published verdict) scores 0.3; top scorer is `wrong_selector` (+0.59); best threshold must invert the theory's sign and still misses the accepted conditional positive |
| Causal emergence, EI (Hoel, EXACT EI computed) | macro-vs-micro intervention efficacy | exact CE < 0 on all 10 enumerated chains; best threshold = trivial classifier (0.8), misses BOTH true positives; proxy version's top scorer was the reward-SHAPED system |
| Metric jump / sharpness (Schaeffer et al., as critique) | grid-resolution artifact detection | our tail-facts result CONFIRMS their point from the mechanism side: abruptness is metric- and grid-dependent (fig34) |

[定位句，写进正文：PTC does not replace these definitions. It identifies
the common quantity they project, explains why each admits false
positives, and adds the counterfactual conditions under which an observed
transition becomes useful emergence.]

## Two faces of one collapse: temporal foreshadowing, spatial cooperation

[叙事组织轴，来自项目原始构思（PPT）。这不是新的判据成分，也不是两种
涌现——是同一个根定义 C(m) 作用在两类可能性空间上。写进正文的域组织，
Fig. 6 的讨论段落用它收束。]

The root definition makes no reference to what the possibility space
ranges over. Two instantiations cover the field's classic intuitions,
and we measured both with the same do-operators:

- **Temporal collapse ("foreshadowing")**: the space is futures within
  one episode/training run; emergence is a locally costly or locally
  silent act whose meaning arrives later, when it proves
  counterfactually necessary. Measured witnesses: the chess
  brilliancies (median local cost -3 pawns, yet do(key) - do(best-alt)
  = +0.54 win probability -- the move is a foreshadowing whose payoff
  is only visible retrospectively); the MultiBERTs NPI case (registered
  failure R5, kept: possibility space collapses at step 20k while
  accuracy is still exactly 0.5, usefulness arrives at 40k --
  collapse-then-jump, foreshadowing at training scale); grokking's
  delayed generalization.
- **Spatial collapse ("cooperation")**: the space is joint
  configurations across agents; emergence is a joint basin no
  individual can reach or hold alone, certified by single-agent
  do-blocks destroying the joint outcome. Measured witnesses: LBF
  forced-coop (blocking ONE agent's commitment destroys full clearance,
  69 positive / 21 tied / 0 negative episode contrasts, with positive
  medians in 3/3 policy seeds -- the operational content of "the whole exceeds
  the parts" WITHOUT the Psi/EI machinery that our exact-form audit
  shows cannot carry a verdict); the swarm decoy family; simple_spread
  coverage.

The same event often has both faces at once: an LBF commitment step is
temporally a foreshadowing (locally it just walks toward a food) and
spatially a coordination lock (the other agent's simultaneous LOAD is
what makes it pay). The duality is presentation structure, not new
theory: both are C(m) = KL(P(tau|M=m) || P(tau)) with different tau.

[从 PPT 保留的另外两点：中心一句话定义 -- "开放可能性空间中非预设行为
结构的稳定化坍缩" (the stabilized collapse of a non-prespecified
behavioral structure in an open possibility space) 可以直接做正文的
one-liner；四角的社区映射（PID/因果涌现/LLM 能力/群体智能）就是统一表。
从 PPT 里丢弃/修正的三点，写作时不要带回来：(1) "突变性 (ΔC_t 显著上升)"
不能做必要条件 -- 我们自己的 tail-facts 结果证明 abruptness 依赖 metric
和 grid（Schaeffer 侧的确认），burst 是测量锚点不是定义成分；(2)
"不可分解性 = 整体解释力超过局部" 恰好是 Hoel EI / Rosas Psi 的读法，
精确形式审计显示它带不动判决 -- 正确的操作化是反事实必要性（do-block
单个 agent 摧毁整体 basin），即 fig36 实测的东西；(3) "可复现性" 是
方法论要求（多种子、预注册），不是定义成分 -- 一次性的真坍缩仍是涌现。]

## Main figures (6) -- composites assembled for the manuscript as `paper/main_fig1..6.png`
## (assemble_main_figures.py; panel sources single-sourced from
## generate_paper_figures.py)

- **Fig. 1** (concept, `main_fig1.png` = figure1_concept): four regimes
  of P_t(B); only one is emergence. The editor's three-minute picture.
- **Fig. 2** (criterion vs imitations, `main_fig2.png`): (a) internal
  battery 10 systems / named counterexamples + ablations exact
  (fig19); (b) threshold plateaus + single prior observables capped
  (fig28); (c) EXACT rival formalisms -- Hoel EI trivial classifier,
  Rosas Psi top scorer is wrong_selector (fig37).
- **Fig. 3** (mechanism within episodes, `main_fig3.png`): (a) P_t(B)
  with do-operators, useful vs harmful collapse sign flip (fig15);
  (b) bootstrap CIs + unsupervised basin recovery (fig17+fig16);
  (c) neural DQN replication (fig18).
- **Fig. 4** (zero-authorial-control training series, `main_fig4.png`):
  (a) MultiBERTs main series (fig31); (b) phenomena battery +
  burst-jump alignment (fig32); (c) tail-gradualism REJECTION (fig34a)
  + the Pythia decoder replication (fig38) -- agreement accepted with
  a foreshadow burst, head facts accepted, BOTH frequency-tail
  families rejected: the accept-and-reject double dissociation on the
  architecture family the emergent-abilities debate is about, verified
  at two scales (160m/410m).
- **Fig. 5** (real strategic system, `main_fig5.png`): (a) chess key
  moves -- locally costly, selectively decisive; quiet-position control
  (fig33); (b) 12/12 robustness cells + prior detectors fail externally
  (fig34bc).
- **Fig. 6** (deep MARL + prediction, `main_fig6.png`): (a)
  simple_spread do_commit/do_block counterfactual (fig35); (b) the LBF
  cross-task replication with 4/4 registered passes (fig36); (c) the
  prospective phase-boundary prediction (fig22) as the "predicts new
  facts" panel.

Extended Data: grokking bridge across architectures (fig23/26/29/30),
external swarm transfer + refined criterion confirmation (fig21/24/25),
estimator robustness (fig20), representation-jump bridge (fig13/14),
analytic core (fig1-fig6 internal numbering), alignment statistics.

## The registered-failure narrative (one paragraph, used verbatim-ish)

Thirteen registered predictions or checks missed under frozen
thresholds, and none was discarded or re-thresholded. Several were
auxiliary claims about which abilities are gradual (facts R3, NPI R5,
tail-facts T6, and the Pythia tail-facts route miss where the rejection
verdict was correct but burstiness caught it rather than usefulness):
the data answered that abruptness depends on the metric and the
checkpoint grid -- measured confirmation, from the mechanism side, of
the metric-sensitivity critique of emergent abilities. Two exposed one shared methodological principle
(chess C5, deep-MARL D2): in a converged system the behaving policy
already prices the trigger into the baseline future -- P_t(win) is
approximately a martingale -- so emergence cannot be read from observed
improvement over a baseline that contains the trigger; it must be read
from do-contrasts between triggering and its best alternative. This
principle, forced on us by failures in two unrelated domains, is itself
one of the paper's findings. The remaining failures (marginal-selectivity
acceptance of an untrained network; the R2 normalization check; the
binary-pair tail families) each triggered a registered refinement that
was then confirmed out-of-sample.

## Scoped claims (what we say / what we do not say)

We say:
- We identify and operationalize a measurable mechanism -- useful
  collapse of latent future possibilities -- and a six-component
  falsifiable criterion for it.
- The criterion separates documented emergence from reward-induced
  coordination, noise, traps, prewiring, and gradual accrual, across
  systems spanning seven orders of magnitude of parameters (tabular to
  160M, encoder and decoder), under frozen thresholds, with
  pre-registered predictions including a prospective phase boundary.
- Common emergence signatures are projections of the same underlying
  quantity and each admits measured false positives.

We do NOT say:
- that all emergence is possibility collapse (untested beyond learning
  systems and one strategic game);
- that prior definitions are special cases of ours (they are measured
  projections with characterized blind spots -- a weaker, defensible
  claim);
- that we explain frontier-scale LLM emergence (our largest system is
  a 160M-parameter decoder; the scale decomposition is a bridge, not a
  proof);
- that the criterion adjudicates "true" abruptness (tail-facts showed
  the verdict is metric- and resolution-dependent; no checkpoint-grid
  method can do better).

## Methods pointers (each maps to a script + preregistration)

- Internal battery and refinement: `criterion_ablation_battery.py`,
  `refined_criterion_confirmation.py`.
- Within-episode mechanism: `within_episode_collapse_probe.py`,
  `neural_within_episode_probe.py`, `unsupervised_basin_discovery.py`.
- Training-process bridge: `grokking_collapse_bridge.py`,
  `transformer_grokking_replication.py`, `induction_head_emergence.py`,
  `scale_emergence_decomposition.py`.
- Public series: `multiberts_collapse_probe.py`,
  `multiberts_phenomena_battery.py`, `multiberts_tail_gradualism.py`
  (+ `MULTIBERTS_PREREGISTRATION.md`).
- Chess: `chess_collapse_probe.py`, `chess_robustness_grid.py`,
  `chess_prior_detectors.py` (+ `CHESS_PREREGISTRATION.md`).
- Deep MARL: `deep_marl_collapse_probe.py`, `deep_marl_aggregate.py`
  (+ `DEEP_MARL_PREREGISTRATION.md`).
- External swarm transfer: `external_swarm_criterion_transfer.py`
  (+ `EXTERNAL_TRANSFER_PREREGISTRATION.md`).
- Theory and verification: `THEORY.md`, `verify_theory_bounds.py`.
- Prior-detector comparison: `prior_metrics_comparison.py`.

## Positioning against the published landscape (verified 2026-07)

The quantitative-emergence literature publishes in three separated
communities, and the venue history matters for our claims:

- **Causal emergence (EI) line**: Hoel's EI (PNAS 2013), Causal
  Emergence 2.0 (arXiv 2025), SVD-based CE (npj Complexity 2025),
  NIS+ machine-learning identification (Entropy survey 2024). All
  quantify macro-vs-micro causal power on Markov dynamics. None
  measures usefulness, endogeneity, or counterfactual necessity; none
  validates on systems the authors did not construct. Venue ceiling so
  far: npj Complexity / Entropy / PNAS (2013).
- **Information-decomposition line**: Rosas et al. "Reconciling
  emergences" (PLoS Comput Biol 2020), the Phil Trans A review (2022),
  synergy-in-networks applications (PLoS Comput Biol 2024), and an
  arXiv 2024 paper reading grokking as a synergy phase transition.
  Emergence = synergy / causal decoupling of a supervenient feature.
  No usefulness sign, no trigger, no do-operators; case studies are
  observational (Game of Life, flocking, ECoG).
- **LLM emergent-abilities line**: Wei et al. (TMLR 2022) vs Schaeffer
  et al. (NeurIPS 2023, metric artifacts), and the percolation model
  (arXiv 2024) that predicts WHEN capabilities emerge from data
  structure. Definitions are performance-curve phenomenology; the
  percolation work is the closest prior for prediction but has no
  counterfactual conditions and one system family.

What none of them has, and we do: (i) a detector-level comparison
showing each line's measure admits NAMED false positives on the same
battery, internally and on an external system; (ii) counterfactual
(do-operator) usefulness conditions; (iii) pre-registered transfer to
systems with zero authorial control (public checkpoint series, human
games); (iv) a prospective phase-boundary prediction; (v) registered
failures. This is the gap the paper fills: not a new competing measure,
but the measurable substrate (P_t(B)) the existing measures project,
plus the conditions that turn a transition into useful emergence.

[写作注意——量子隐喻警戒：正文与标题避免 "wavefunction/quantum collapse"
类比。我们的坍缩是统计的（对 rollout 未来分布的收缩）、由系统自身动作
内生驱动，不是观测导致的量子坍缩。审稿人对物理隐喻过敏；Schrödinger 只
能出现在 SI 的一句历史脚注里，或完全不出现。]

## Significance statement (for cover letter)

Emergence is measured differently in complex-systems science (causal
efficacy of macro-variables), information theory (synergy), and machine
learning (abrupt ability curves), and the three communities disagree
about what counts. We provide a single measurable quantity of which all
three signatures are projections, a falsifiable six-component criterion
with frozen thresholds, and validation spanning designed systems,
public model-training series, and human strategic play -- including
prospective prediction and honest registered failures. For machine
intelligence specifically, the framework turns "did an ability emerge?"
from a curve-reading exercise into a counterfactual measurement, with
direct application to capability auditing and forecasting.

## Known reviewer pressure points and the prepared answers

1. "simple_spread is a toy deep-MARL demo." -- CLOSED: the registered
   protocol was replicated on Level-Based Foraging (forced-coop,
   discrete, irreversible consumption, sparse reward -- a structurally
   different task family), 3 seeds, 4/4 registered predictions passed,
   including the cleanest counterfactual-necessity result in the
   project (69/90 positive episode contrasts, 21 ties, none negative;
   positive medians in 3/3 policy seeds; fig36, LBF_PREREGISTRATION.md).
2. "The observer (basins, rollout policy) is a choice." -- Unsupervised
   basin recovery (purity 0.84-1.0 internal and external); estimator
   robustness grids in two domains (12-cell gridworld, 12-cell chess);
   the potential component's absolute scale is observer-dependent and
   is reported as such.
3. "Multiple systems, multiple probes -- garden of forking paths." --
   Every confirmatory stage has a frozen preregistration with outcomes
   recorded in place, including failures. Misses that motivated a
   refinement remain counted as misses; revised rules count only through
   later re-frozen tests. Single-page audit: PREDICTION_LEDGER.md indexes
   ~131 frozen verdicts/checks with 13 registered misses, each row
   pointing to its primary document.
4. "The LBF probe temperature (6.0) was tuned to pass." -- CLOSED:
   re-probing the saved main-run networks across T in {2, 3, 4.5, 8}
   (lbf_robustness_grid.py, success criteria frozen in-script), the
   pooled do-contrast is positive with sign-test p <= 2.7e-20 in EVERY
   cell (worst cell 20W/0L), and the trained-vs-greedy double
   dissociation holds in every cell. The ABSOLUTE potential threshold
   (0.8 bits) fails at T <= 3 -- the same observer-scale dependence
   already documented in the chess grid (C4 at temp 200) and declared
   as the potential component's known limitation: contrasts are
   robust, absolute openness scales with the observer.
5. "The detector comparison never ran on your flagship deep-MARL
   domain." -- CLOSED: five single-signal detectors on 8 LBF systems
   including forced_commit (observer-imposed trigger on the trained
   policy) and scripted_coop (hand-coded coordinator, win rate in the
   trained range). Best single detector: 0.875, each miss a named
   system; scripted_coop scores ABOVE the trained systems on
   specificity (0.94), Psi (0.95) and EI (0.68) -- structure signals
   cannot see who wrote the structure. Round-1 set-composition
   failure (performance separated a set with no competent imitation)
   archived and reported (lbf_prior_detectors.py, both rounds'
   predictions frozen before their runs).
6. "Your rival detectors are flavored strawmen." -- CLOSED: Hoel's
   exact EI (max-entropy interventions, exact enumerated TPMs, five
   candidate macros, best taken) and Rosas' exact practical criterion
   Psi (four supervenient features x two micro decompositions, best
   taken) were computed with zero Monte-Carlo error on the same
   battery. Exact EI: best threshold = trivial classifier (0.8),
   misses both true positives. Exact Psi > 0: accuracy 0.3, top
   scorer is `wrong_selector`. Both are WORSE than our charitable
   proxies; the blind spots are structural (no access to value sign,
   selectivity, or provenance), not artifacts of simplification
   (`exact_prior_formalisms.py`; Prop. 5, THEORY.md).
7. "All your public-model evidence is an encoder (BERT); the emergent-
   abilities debate is about autoregressive decoders." -- CLOSED: the
   same frozen thresholds applied to EleutherAI Pythia-160m's 21
   published checkpoints (PYTHIA_PREREGISTRATION.md, fig38) accept
   subject-verb agreement (0.49 -> 0.93 across steps 512-2000, with
   the foreshadow burst preceding the jump) and head facts, while
   rejecting BOTH frequency-tail families (tail facts via burstiness,
   tail words via usefulness) and both controls. 7/8 registered
   predictions passed; the one route miss (rejection via burstiness
   instead of usefulness) is recorded. The decoder shows the full
   accept/reject double dissociation that MultiBERTs showed only
   partially. Scale check: the identical protocol on Pythia-410m
   passes 3/3 registered predictions with the SAME anchored window
   (step 1000) -- not a one-size accident.

## Headline effect sizes with bootstrap 95% CIs

From `bootstrap_intervals.py` (20,000 percentile resamples; chess
resamples positions, while MARL resamples evaluation episodes conditional
on three trained policies; outputs/bootstrap_intervals.json). Use these
in the main text instead of bare point estimates:

- Chess, counterfactual necessity of the annotated key move:
  P(win | do key) - P(win | do best-alt) mean +0.539 [0.508, 0.570],
  median +0.547 [0.500, 0.594] (n = 240 positions).
- Chess, useful shift of key vs greedy: +0.537 [0.506, 0.568];
  vs random: +0.679 [0.649, 0.709].
- Chess, local cost: median -3.0 pawns [-3.0, -3.0]; strict-loss rate
  0.800 [0.746, 0.850].
- Chess, openness control: quiet - sacrifice potential (median diff)
  +0.764 bits [0.647, 0.939] -- quiet positions are MORE open, so
  openness alone cannot be the signature.
- Deep MARL (simple_spread), pooled do-contrast:
  median +0.083 [0.042, 0.177], mean +0.091 [0.013, 0.169]
  (n = 120 episodes over 3 seeds; per-seed medians 0.094/0.104/0.063,
  all positive).
- Deep MARL cross-task (Level-Based Foraging, forced-coop), pooled
  do-contrast: median +0.042 [0.021, 0.094], mean +0.109
  [0.082, 0.137] (n = 90 episodes over 3 seeds; 69 positive,
  21 tied and 0 negative; positive medians in 3/3 policy seeds).
  All four registered predictions passed
  (LBF_PREREGISTRATION.md).
