"""Can the MCTS tree even contain the policy's preferred action?

Progressive widening draws candidates from MaskedRandomAgentRLlib, so at
budget 200 the root holds ~14 random joint actions per agent. The GNN can only
rank those. This measures, on real states:

  * log P(policy's own sampled action)      -- what PPO would play
  * log P(best of the 14 random candidates) -- the best the tree can pick
  * the PUCT prior distribution over those candidates after softmax
    (near-uniform => the prior carries no signal into selection)

Usage: python diag_pw_candidates.py [n_states_per_map]
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(r'C:/Users/ricco/Documents/GitHub/parallel-risk-gym')
sys.path.insert(0, str(REPO))
torch.set_num_threads(4)

from parallel_risk import ParallelRiskEnv                       # noqa: E402
from parallel_risk.agents.masked_random_agent import MaskedRandomAgentRLlib  # noqa: E402
from parallel_risk.agents.mcts_agent import RiskSimulator       # noqa: E402
from parallel_risk.agents.mcts_gnn_agent import GNNActionSampler, MCTSGNNAgent  # noqa: E402

CKPT = REPO / 'experiments/k_sweep_ppo_200/K10/checkpoints/final.pt'
K, MR, MT, N_PW = 10, 5, 40, 14      # N_PW = sqrt(200), the root's widening budget
MAPS = ['simple_6', 'dense_12', 'grid_20', 'hub_ring_30']
N_STATES = int(sys.argv[1]) if len(sys.argv) > 1 else 25


def main():
    rows = []
    for map_name in MAPS:
        env = ParallelRiskEnv(map_name=map_name, max_turns=MT, max_actions_per_turn=max(10, K),
                              reward_shaping_config=None)
        agent = MCTSGNNAgent.from_checkpoint(str(CKPT), env.map_config, simulation_budget=200,
                                             device='cpu', max_regions=MR, max_turns=MT)
        sim = RiskSimulator(env.map_config, max_turns=MT)
        rnd = MaskedRandomAgentRLlib(n_territories=env.map_config.n_territories,
                                     adjacency_matrix=env.map_config.adjacency_matrix,
                                     action_budget=K, max_troops=20,
                                     max_actions_per_turn=max(10, K))
        gnn_sampler = GNNActionSampler(agent.policy, agent.decoder, env.map_config, K,
                                       torch.device('cpu'), max_regions=MR,
                                       max_actions_per_turn=max(10, K))
        policy_fn = agent.mcts.policy_fn

        np.random.seed(0)
        obs, _ = env.reset(seed=0)
        beats = 0
        d_logp, spreads = [], []
        for i in range(N_STATES):
            gs = RiskSimulator.clone_state(env.game_state)
            agent.mcts.evaluator.new_search()
            o = sim.state_to_obs(gs, 'agent_0')
            # what the raw policy would play
            lp_policy = policy_fn(gs, 'agent_0', gnn_sampler.get_action_raw(o))
            # what progressive widening puts in the tree
            lp_rand = [policy_fn(gs, 'agent_0', rnd.get_action_raw(o)) for _ in range(N_PW)]
            best = max(lp_rand)
            beats += (best >= lp_policy)
            d_logp.append(lp_policy - best)
            p = np.exp(np.array(lp_rand) - max(lp_rand))
            p /= p.sum()
            spreads.append(p.max())          # 1/14 = uniform, 1.0 = one-hot
            # advance the game so states are diverse
            acts = {a: rnd.get_action_raw(sim.state_to_obs(gs, a)) for a in ('agent_0', 'agent_1')}
            obs, _, te, tr, _ = env.step(acts)
            if te.get('__all__') or tr.get('__all__'):
                obs, _ = env.reset(seed=i + 1)
        rows.append((map_name, np.mean(d_logp), beats, np.mean(spreads)))
        print(f'{map_name:14s} n={env.map_config.n_territories:2d}  '
              f'policy log-prob advantage over best of {N_PW} random: '
              f'{np.mean(d_logp):+7.2f} nats   '
              f'random beat policy in {beats}/{N_STATES} states   '
              f'max PUCT prior {np.mean(spreads):.3f} (uniform={1/N_PW:.3f})', flush=True)

    print()
    print('Reading: a large positive advantage means the action PPO would play is far more')
    print('probable under the policy than anything progressive widening puts in the tree, so')
    print('the search cannot select it. A max prior near 1/14 means the softmaxed priors are')
    print('nearly uniform over the random candidates, i.e. the prior adds little to selection.')


if __name__ == '__main__':
    main()
