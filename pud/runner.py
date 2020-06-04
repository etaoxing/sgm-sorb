from pud.dependencies import *
from pud.collector import Collector
from pud.policies import GaussianPolicy

def train_eval(
    agent,
    replay_buffer,
    env,
    eval_env,
    num_iterations=int(1e6),
    initial_collect_steps=1000,
    num_eval_episodes=10,
    opt_log_interval=100,
    eval_interval=10000,
):
    collector = Collector(GaussianPolicy(agent), replay_buffer, env, initial_collect_steps=initial_collect_steps)
    collector.step(collector.initial_collect_steps)
    for i in range(1, num_iterations + 1):
        collector.step(2)
        agent.train()
        opt_info = agent.optimize(replay_buffer, iterations=1, batch_size=64)

        if i % opt_log_interval == 0:
            print(f'iteration = {i}, opt_info = {opt_info}')

        if i % eval_interval == 0:
            agent.eval()
            print(f'evaluating iteration = {i}')
            eval_agent(agent, eval_env)
            print('-' * 10)


def eval_agent(agent, eval_env, num_evals=10, eval_distances=[2, 5, 10]):
    for dist in eval_distances:
        eval_env.set_sample_goal_args(prob_constraint=1, min_dist=dist, max_dist=dist) # NOTE: samples goal distances in [min_dist, max_dist] closed interval
        returns = Collector.eval_agent(agent, eval_env, num_episodes=num_evals)
        # For debugging, it's helpful to check the predicted distances for
        # goals of known distance.
        states = dict(observation=[], goal=[])
        for _ in range(num_evals):
            state = eval_env.reset()
            states['observation'].append(state['observation'])
            states['goal'].append(state['goal'])
        pred_dist = list(agent.get_dist_to_goal(states))

        print('\tset goal dist = %d' % dist)
        print('\t\treturns = {}'.format(returns))
        print('\t\tpredicted_dists = {}'.format(pred_dist))
        print('\t\taverage return = %d' % np.mean(returns))
        print('\t\taverage predicted_dist = %.1f (%.1f)' %
                                  (np.mean(pred_dist), np.std(pred_dist)))
