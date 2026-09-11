"""Do cached batched proposals sample the same distribution as the per-call path?"""
import sys
from collections import Counter
from pathlib import Path
import numpy as np, torch
REPO = Path(r'C:/Users/ricco/Documents/GitHub/parallel-risk-gym'); sys.path.insert(0, str(REPO))
torch.set_num_threads(2)
from parallel_risk import ParallelRiskEnv
from parallel_risk.agents.mcts_gnn_agent import MCTSGNNAgent
from parallel_risk.agents.mcts_agent import RiskSimulator
CKPT = str(REPO/'experiments/k_sweep_ppo_200/K10/checkpoints/final.pt')
N = 4000
for mp in ('simple_6', 'grid_20'):
    env = ParallelRiskEnv(map_name=mp, max_turns=40, max_actions_per_turn=10, reward_shaping_config=None)
    env.reset(seed=0)
    ag = MCTSGNNAgent.from_checkpoint(CKPT, env.map_config, simulation_budget=200, device='cpu',
                                      max_regions=5, max_turns=40, pw_sampler='gnn')
    s = ag.mcts.samplers['agent_0']; sim = ag.mcts.sim
    gs = RiskSimulator.clone_state(env.game_state)
    obs = sim.state_to_obs(gs, 'agent_0')
    # per-call path (fresh forward + single decode each time)
    torch.manual_seed(0); np.random.seed(0)
    old = [s.get_action_raw(obs)['actions'][:s.action_budget].copy() for _ in range(N)]
    # cached batched path
    torch.manual_seed(0); np.random.seed(0)
    ag.evaluator.new_search()
    new = [s.get_action_for_state(gs, 'agent_0')['actions'][:s.action_budget].copy() for _ in range(N)]
    def marg(rows, col):
        c = Counter()
        for r in rows:
            for slot in r: c[int(slot[col])] += 1
        tot = sum(c.values())
        return {k: v / tot for k, v in c.items()}
    print(f'--- {mp} (n={env.map_config.n_territories}), {N} draws each')
    for col, name in ((0, 'source'), (1, 'dest'), (2, 'troops')):
        a, b = marg(old, col), marg(new, col)
        keys = set(a) | set(b)
        tv = 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)
        print(f'   {name:7s} total-variation distance {tv:.4f}   support {len(a)} vs {len(b)}')
    same = sum(1 for x, y in zip(old, new) if np.array_equal(x, y))
    print(f'   identical draws: {same}/{N} (expected ~0: different RNG consumption)')
