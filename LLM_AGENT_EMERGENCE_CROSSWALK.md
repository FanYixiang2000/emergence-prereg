# LLM / Agent-System Emergence Crosswalk

Created 2026-07-29 after checking the user's examples:
`Scaling Large Language Model-based Multi-Agent Collaboration`
(MACNET) and AgentVerse.

## Core mapping

The framework can explain LLM-agent "collaborative emergence", but it
must not treat every performance gain from adding agents as emergence.
The object must be declared.

### Collaborative emergence in LLM multi-agent systems

Slow/control axis: number of agents, topology, role diversity,
communication rounds, or routing policy.

Possibility space:

- nominal answer space: possible final artifacts/answers;
- interaction-trajectory space: who talks to whom, in what order, with
  what revisions;
- role/topology space: chain, hierarchy, mesh, debate, verifier,
  integrator, specialist routes;
- evidence-integration space: which partial arguments survive into the
  final answer.

Under this framework, collaborative emergence is not simply
`score(N) > score(1)`. It is a candidate emergence event only if the
multi-agent process forms a persistent macro-regime of coordination
that was not directly specified and that reorganizes future reasoning
possibilities. Examples:

- a stable division of labor appears across rounds;
- a hierarchy or chain of specialists becomes the attractor;
- a verifier/integrator bottleneck starts selecting which arguments
  survive;
- the system moves from many incompatible drafts to a constrained
  shared artifact trajectory.

MACNET's reported logistic collaborative scaling can be interpreted
as a control-axis capability-realization profile: as agent count or
topology changes, the interaction-trajectory possibility space
contracts into more integrated reasoning regimes. The framework would
measure whether the transition is gradual, saturating, or punctuated,
and whether topology affects M/J/R/G/A.

### Neural emergence in LLM scaling

Slow axis: model scale, training compute, data, or training step.

Possibility space:

- mechanism/circuit space: possible internal algorithms that can solve
  a task;
- behavior-policy space: distributions over solution strategies;
- latent-representation space: separability or reuse of task-relevant
  variables;
- continuation space: distribution over reasoning traces under a fixed
  prompt.

This corresponds to capability-formation emergence. A scaling "jump"
is not enough, because metric thresholding can create apparent
emergence. The framework requires evidence that internal or
behavioral possibility space reorganizes: many possible mechanisms or
solution traces collapse into a stable, reusable regime, ideally
before or alongside metric gain.

### AgentVerse

AgentVerse dynamically recruits expert agents, lets them deliberate,
execute, evaluate, and revise. It reports social behaviors such as
volunteering, conformity and destructive behavior.

Framework mapping:

- D: the future interaction distribution changes, e.g. from many
  possible speakers/roles/plans to a stable chain, hierarchy, or
  consensus route.
- G: the concrete role allocation or social behavior is not directly
  scripted by the framework. If the framework hard-codes solver,
  critic and executor roles, G is lower; if those roles are recruited
  and stabilized by interaction history, G is higher.
- R: the role/social pattern persists across rounds, perturbations or
  similar tasks.
- S: before the interaction, the final role/social configuration was
  low probability under a neutral baseline.
- X: after early conversational cues, the final social organization
  becomes predictable before the final answer is produced.
- A: the macro affordance gain is improved task solving, fewer
  contradictions, better tool use, or robustness.

AgentVerse therefore can contain genuine emergence, but only if the
reported social behavior passes D/G/R. A static pipeline with fixed
roles is coordinated engineering, not high-G emergence. A dynamically
assembled group that repeatedly self-organizes into a useful hierarchy
or volunteer/critic regime is a strong candidate.

## Testable experiments

For LLM agent systems, the framework suggests the following measurable
contracts:

1. Role-distribution collapse: entropy over active roles / speakers /
   routes across rounds.
2. Draft-trajectory collapse: entropy or clustering of possible final
   artifacts under cloned interaction continuations.
3. Topology-source profile: compare chain, hierarchy, mesh and
   irregular MACNET graphs; test whether topology changes M/J/R/A.
4. Intervention utility: perturb the emerging integrator, verifier or
   high-centrality agent and predict damage from source/profile
   measures.
5. Early intelligibility: test whether early micro-cues predict final
   hierarchy, consensus, or failure before performance is visible.

The strongest NMI bridge would be to show that a possibility-space
profile predicts when adding agents helps, when coordination overhead
dominates, and which intervention improves the collective regime.
