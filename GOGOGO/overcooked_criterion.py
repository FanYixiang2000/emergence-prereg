"""Six-component measurement machinery for unmodified Overcooked-AI.

Shared by the design pilots and the (to-be-frozen) confirmation run.
Environment dynamics, observations and the sparse delivery reward are
the benchmark's own; this module only adds measurement: minimal
releasing interventions on the first-potting role commitment, outcome
basins, and the frozen component definitions.

Trigger: agent 0 is the first to place an onion in a pot.
Interventions (minimal, releasing, same style as every other domain):
    do_commit  agent 0 is steered by a shortest-path onion-to-pot
               subpolicy until the first potting event, then released;
    do_block   agent 0's potting interacts are suppressed until agent 1
               has potted once, then released.
Basins: (first potter: 0 / 1 / none) x (>= 1 delivery), 6 cells.
Value: sparse team reward per episode.
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

np.Inf = np.inf
import torch

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action, Direction

HORIZON = 400
BASINS = tuple(f"{p}_{d}" for p in ("pot0", "pot1", "potnone")
               for d in ("deliver", "nodeliver"))
THRESHOLDS = {
    "potential_bits": 0.5,
    "conditional_selectivity": 0.5,
    "specificity_js_bits": 0.2,
    "usefulness_gap": 0.0,
    "acquisition": 0.3,
}


def make_env(layout: str) -> OvercookedEnv:
    mdp = OvercookedGridworld.from_layout_name(layout)
    return OvercookedEnv.from_mdp(mdp, horizon=HORIZON, info_level=0)


def featurize(env: OvercookedEnv) -> List[np.ndarray]:
    return [o.astype(np.float32) for o in env.featurize_state_mdp(env.state)]


# scripted motion

def walkable(mdp: OvercookedGridworld, pos: Tuple[int, int]) -> bool:
    x, y = pos
    if not (0 <= y < len(mdp.terrain_mtx) and 0 <= x < len(mdp.terrain_mtx[0])):
        return False
    return mdp.terrain_mtx[y][x] == " "


def terrain_positions(mdp: OvercookedGridworld, char: str):
    out = []
    for y, row in enumerate(mdp.terrain_mtx):
        for x, c in enumerate(row):
            if c == char:
                out.append((x, y))
    return out


def bfs_next_move(mdp, start, goals, occupied) -> Optional[Tuple[int, int]]:
    """First move of a shortest path from start to any cell adjacent to a
    goal tile; returns the move direction or None if already adjacent."""
    adj_targets = set()
    for g in goals:
        for d in Direction.ALL_DIRECTIONS:
            cell = (g[0] + d[0], g[1] + d[1])
            if walkable(mdp, cell):
                adj_targets.add(cell)
    if start in adj_targets:
        return None
    frontier = deque([(start, None)])
    seen = {start}
    while frontier:
        pos, first = frontier.popleft()
        for d in Direction.ALL_DIRECTIONS:
            nxt = (pos[0] + d[0], pos[1] + d[1])
            if nxt in seen or not walkable(mdp, nxt) or nxt in occupied:
                continue
            nf = first if first is not None else d
            if nxt in adj_targets:
                return nf
            seen.add(nxt)
            frontier.append((nxt, nf))
    return None


def face_or_interact(mdp, player, goals) -> Tuple[int, int] | str:
    """Adjacent to a goal: face it, then interact."""
    for d in Direction.ALL_DIRECTIONS:
        cell = (player.position[0] + d[0], player.position[1] + d[1])
        if cell in goals:
            if player.orientation == d:
                return "interact"
            return d
    return Action.STAY


def pot_states(env: OvercookedEnv):
    """Per-pot classification: accepting / full_idle / cooking / ready."""
    out = {}
    for pos in terrain_positions(env.mdp, "P"):
        soup = env.state.objects.get(pos)
        if soup is None:
            out[pos] = "accepting"
        elif soup.is_ready:
            out[pos] = "ready"
        elif soup.is_cooking:
            out[pos] = "cooking"
        elif len(soup.ingredients) >= 3:
            out[pos] = "full_idle"
        else:
            out[pos] = "accepting"
    return out


def ingredient_sources(mdp) -> list:
    return terrain_positions(mdp, "O") + terrain_positions(mdp, "T")


def onion_to_pot_action(env: OvercookedEnv, agent: int,
                        rng: random.Random):
    """Scripted cook: fetch ingredients, fill a pot, start it cooking;
    yield the pot approach while the pot is busy."""
    mdp = env.mdp
    state = env.state
    player = state.players[agent]
    other = state.players[1 - agent]
    occupied = {other.position}
    held = player.held_object.name if player.held_object else None
    pots = pot_states(env)
    if held in ("onion", "tomato"):
        goals = [p for p, s in pots.items() if s == "accepting"]
        if not goals:
            move = bfs_next_move(mdp, player.position,
                                 ingredient_sources(mdp), occupied)
            return move if move is not None else Action.STAY
    elif held is None:
        full = [p for p, s in pots.items() if s == "full_idle"]
        goals = full if full else ingredient_sources(mdp)
    else:
        return "interact" if rng.random() < 0.5 else Action.STAY
    act = face_or_interact(mdp, player, set(goals))
    if act != Action.STAY:
        return act
    move = bfs_next_move(mdp, player.position, goals, occupied)
    if move is not None:
        return move
    return rng.choice(Direction.ALL_DIRECTIONS)


def is_potting_interact(env: OvercookedEnv, agent: int, action) -> bool:
    if action != "interact":
        return False
    state = env.state
    player = state.players[agent]
    if not player.held_object or player.held_object.name not in (
            "onion", "tomato"):
        return False
    facing = (player.position[0] + player.orientation[0],
              player.position[1] + player.orientation[1])
    return facing in set(terrain_positions(env.mdp, "P"))


# policies

class TeamPolicy:
    """kind: net | scripted_roles | fixed_role | clone"""

    def __init__(self, kind: str, net=None, clone_table=None,
                 cook_agent: int = 0):
        self.kind = kind
        self.net = net
        self.clone_table = clone_table
        self.cook_agent = cook_agent

    def actions(self, env: OvercookedEnv, obs, rng: random.Random):
        if self.kind == "net":
            x = torch.tensor(np.stack(obs))
            with torch.no_grad():
                logits, _ = self.net(x)
                dist = torch.distributions.Categorical(logits=logits)
                acts = dist.sample().tolist()
            return [Action.ALL_ACTIONS[a] for a in acts]
        if self.kind == "scripted_roles":
            return [self._role_action(env, agent, rng) for agent in (0, 1)]
        if self.kind == "clone":
            x = torch.tensor(np.stack(obs))
            with torch.no_grad():
                logits, _ = self.net(x)
                acts = logits.argmax(dim=1).tolist()
            return [Action.ALL_ACTIONS[a] for a in acts]
        raise ValueError(self.kind)

    def _role_action(self, env, agent, rng):
        if agent == self.cook_agent:
            return onion_to_pot_action(env, agent, rng)
        return serve_action(env, agent, rng)


def serve_action(env: OvercookedEnv, agent: int, rng: random.Random):
    """Scripted server: fetch a dish once a soup is under way, collect
    the ready soup, deliver; wait near the serving window otherwise."""
    mdp = env.mdp
    state = env.state
    player = state.players[agent]
    other = state.players[1 - agent]
    occupied = {other.position}
    held = player.held_object.name if player.held_object else None
    pots = pot_states(env)
    soup_underway = any(s in ("cooking", "ready", "full_idle")
                        for s in pots.values())
    if held == "soup":
        goals = terrain_positions(mdp, "S")
    elif held == "dish":
        ready = [p for p, s in pots.items() if s == "ready"]
        cooking = [p for p, s in pots.items()
                   if s in ("cooking", "full_idle")]
        if ready:
            goals = ready
        elif cooking:
            goals = cooking  # approach, wait facing the pot
            act = face_or_interact(mdp, player, set(goals))
            if act == "interact":
                return Action.STAY
            if act != Action.STAY:
                return act
            move = bfs_next_move(mdp, player.position, goals, occupied)
            return move if move is not None else Action.STAY
        else:
            return idle_move(env, agent, rng)
    elif held is None:
        if soup_underway:
            goals = terrain_positions(mdp, "D")
        else:
            return idle_move(env, agent, rng)
    else:
        return "interact"
    act = face_or_interact(mdp, player, set(goals))
    if act != Action.STAY:
        return act
    move = bfs_next_move(mdp, player.position, goals, occupied)
    if move is not None:
        return move
    return rng.choice(Direction.ALL_DIRECTIONS)


def idle_move(env: OvercookedEnv, agent: int, rng: random.Random):
    """Wait adjacent to the serving window, off the pot approaches."""
    mdp = env.mdp
    player = env.state.players[agent]
    other = env.state.players[1 - agent]
    serving = terrain_positions(mdp, "S")
    adj = set()
    for g in serving:
        for d in Direction.ALL_DIRECTIONS:
            cell = (g[0] + d[0], g[1] + d[1])
            if walkable(mdp, cell):
                adj.add(cell)
    if player.position in adj:
        return Action.STAY
    move = bfs_next_move(mdp, player.position, serving,
                         {other.position})
    if move is not None:
        return move
    return rng.choice(Direction.ALL_DIRECTIONS)


def train_bc_clone(layouts, seed: int, n_episodes: int = 150,
                   epochs: int = 20):
    """Supervised clone of the scripted role pair (cook = agent 0 in
    context A's optimal sense is irrelevant -- the clone copies whatever
    the scripted pair does on both layouts)."""
    from overcooked_pilot import PolicyNet
    torch.manual_seed(seed)
    rng = random.Random(seed)
    xs, ys = [], []
    scripted = TeamPolicy("scripted_roles", cook_agent=0)
    for episode in range(n_episodes):
        layout = layouts[episode % len(layouts)]
        env = make_env(layout)
        env.reset()
        while True:
            obs = featurize(env)
            actions = scripted.actions(env, obs, rng)
            for agent in (0, 1):
                xs.append(obs[agent])
                ys.append(Action.ALL_ACTIONS.index(actions[agent]))
            _s, _r, done, _info = env.step(actions)
            if done:
                break
    net = PolicyNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x = torch.tensor(np.stack(xs))
    y = torch.tensor(ys)
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        for start in range(0, len(x), 1024):
            mb = perm[start:start + 1024]
            logits, _ = net(x[mb])
            loss = torch.nn.functional.cross_entropy(logits, y[mb])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    return net


# episodes

def run_episode(policy: TeamPolicy, layout: str, seed: int,
                intervention: Optional[str]) -> Dict:
    env = make_env(layout)
    env.reset()
    rng = random.Random(seed)
    first_potter: Optional[int] = None
    agent1_potted = False
    sparse_total = 0.0
    committed_done = False

    while True:
        obs = featurize(env)
        actions = policy.actions(env, obs, rng)
        if intervention == "do_commit" and first_potter is None \
                and not committed_done:
            actions[0] = onion_to_pot_action(env, 0, rng)
        if intervention == "do_block" and not agent1_potted:
            if is_potting_interact(env, 0, actions[0]):
                actions[0] = Action.STAY
        _s, sparse_r, done, info = env.step(actions)
        sparse_total += sparse_r
        gs = env.game_stats
        pots0 = list(gs.get("potting_onion", [[], []])[0]) + \
            list(gs.get("potting_tomato", [[], []])[0])
        pots1 = list(gs.get("potting_onion", [[], []])[1]) + \
            list(gs.get("potting_tomato", [[], []])[1])
        if first_potter is None:
            t0 = min(pots0) if len(pots0) else None
            t1 = min(pots1) if len(pots1) else None
            if t0 is not None and (t1 is None or t0 <= t1):
                first_potter = 0
            elif t1 is not None:
                first_potter = 1
        if len(pots1):
            agent1_potted = True
        if first_potter == 0:
            committed_done = True
        if done:
            break

    potter = {0: "pot0", 1: "pot1", None: "potnone"}[first_potter]
    deliver = "deliver" if sparse_total > 0 else "nodeliver"
    return {
        "basin": f"{potter}_{deliver}",
        "trigger": int(first_potter == 0),
        "score": sparse_total,
    }


def entropy_bits(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total)
                for c in counts.values() if c > 0)


def js_bits(p: Dict[str, float], q: Dict[str, float]) -> float:
    out = 0.0
    for key in set(p) | set(q):
        a, b = p.get(key, 0.0), q.get(key, 0.0)
        m = 0.5 * (a + b)
        if a > 0:
            out += 0.5 * a * math.log2(a / m)
        if b > 0:
            out += 0.5 * b * math.log2(b / m)
    return out


def evaluate(policy: TeamPolicy, layouts: Tuple[str, str], n_eval: int,
             seed_offset: int) -> Dict:
    rows: List[Dict] = []
    for ctx, layout in enumerate(layouts):
        for episode in range(n_eval):
            seed = seed_offset + 10_000 * ctx + episode
            for mode in (None, "do_commit", "do_block"):
                row = run_episode(policy, layout, seed, mode)
                row["mode"] = mode or "natural"
                row["context"] = ctx
                rows.append(row)

    def subset(mode, ctx=None):
        return [r for r in rows if r["mode"] == mode
                and (ctx is None or r["context"] == ctx)]

    def dist(rows_in):
        counts: Dict[str, int] = {}
        for r in rows_in:
            counts[r["basin"]] = counts.get(r["basin"], 0) + 1
        total = len(rows_in)
        return {b: counts.get(b, 0) / total for b in BASINS}

    natural = subset("natural")
    counts: Dict[str, int] = {}
    for r in natural:
        counts[r["basin"]] = counts.get(r["basin"], 0) + 1
    trig = {str(c): float(np.mean([r["trigger"]
                                   for r in subset("natural", c)]))
            for c in (0, 1)}
    mean = lambda rows_in: float(np.mean([r["score"] for r in rows_in]))
    return {
        "rows_n": len(rows),
        "potential_bits": entropy_bits(counts),
        "trigger_rates": trig,
        "conditional_selectivity": abs(trig["0"] - trig["1"]),
        "specificity_js_bits": js_bits(dist(subset("do_commit")),
                                       dist(subset("do_block"))),
        "usefulness_gap": mean(natural) - mean(subset("do_block")),
        "natural_score": mean(natural),
        "do_block_score": mean(subset("do_block")),
    }


def verdict(metrics: Dict, endogenous: bool, acquisition: float) -> Dict:
    passes = {
        "potential": metrics["potential_bits"]
        >= THRESHOLDS["potential_bits"],
        "conditional_selectivity": metrics["conditional_selectivity"]
        >= THRESHOLDS["conditional_selectivity"],
        "specificity": metrics["specificity_js_bits"]
        >= THRESHOLDS["specificity_js_bits"],
        "usefulness": metrics["usefulness_gap"]
        > THRESHOLDS["usefulness_gap"],
        "endogeneity": endogenous,
        "acquisition": acquisition >= THRESHOLDS["acquisition"],
    }
    return {"passes": passes, "emergent": int(all(passes.values())),
            "failed": [k for k, ok in passes.items() if not ok]}
