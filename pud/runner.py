from pud.dependencies import *
from pud.collector import Collector
from pud.envs.simple_navigation_env import set_env_difficulty

def train_eval(
    policy,
    agent,
    replay_buffer,
    env,
    eval_env,
    num_iterations=int(1e6),
    initial_collect_steps=1000,
    collect_steps=1,
    opt_steps=1,
    batch_size_opt=64,
    num_eval_episodes=10,
    opt_log_interval=100,
    eval_interval=10000,
):
    collector = Collector(policy, replay_buffer, env, initial_collect_steps=initial_collect_steps)
    collector.step(collector.initial_collect_steps)
    for i in range(1, num_iterations + 1):
        collector.step(collect_steps)
        agent.train()
        opt_info = agent.optimize(replay_buffer, iterations=opt_steps, batch_size=batch_size_opt)

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
        returns = Collector.eval_agent(agent, eval_env, num_evals)
        # For debugging, it's helpful to check the predicted distances for
        # goals of known distance.
        states = dict(observation=[], goal=[])
        for _ in range(num_evals):
            state = eval_env.reset()
            states['observation'].append(state['observation'])
            states['goal'].append(state['goal'])
        pred_dist = list(agent.get_dist_to_goal(states))

        print(f'\tset goal dist = {dist}')
        print(f'\t\treturns = {returns}')
        print(f'\t\tpredicted_dists = {pred_dist}')
        print(f'\t\taverage return = {np.mean(returns)}')
        print(f'\t\taverage predicted_dist = {np.mean(pred_dist):.1f} ({np.std(pred_dist):.2f})')


def eval_search_policy(search_policy, eval_env, num_evals=10):
    eval_start = time.process_time()

    successes = 0.
    for _ in range(num_evals):
        try:
            _, _, _, ep_reward_list = Collector.get_trajectory(search_policy, eval_env)
            successes += int(len(ep_reward_list) < eval_env.duration)
        except:
            pass

    eval_end = time.process_time()
    eval_time = eval_end - eval_start
    success_rate = successes / num_evals
    return success_rate, eval_time


def take_cleanup_steps(search_policy, eval_env, num_cleanup_steps):
    set_env_difficulty(eval_env, 0.95)

    search_policy.set_cleanup(True)
    cleanup_start = time.process_time()
    Collector.eval_agent(search_policy, eval_env, num_cleanup_steps, by_episode=False)
    cleanup_end = time.process_time()
    search_policy.set_cleanup(False)
    cleanup_time = cleanup_end - cleanup_start
    return cleanup_time


def cleanup_and_eval_search_policy(search_policy, eval_env, num_evals=10, difficulty=0.5):
    set_env_difficulty(eval_env, difficulty)
    search_policy.reset_stats()
    success_rate, eval_time = eval_search_policy(search_policy, eval_env, num_evals=num_evals)

    # Initial sparse graph
    print(f'Initial {search_policy} has success rate {success_rate:.2f}, evaluated in {eval_time:.2f} seconds')
    initial_g, initial_rb = search_policy.g.copy(), search_policy.rb_vec.copy()

    # Filter search policy
    search_policy.filter_keep_k_nearest()

    set_env_difficulty(eval_env, difficulty)
    search_policy.reset_stats()
    success_rate, eval_time = eval_search_policy(search_policy, eval_env, num_evals=num_evals)
    print(f'Filtered {search_policy} has success rate {success_rate:.2f}, evaluated in {eval_time:.2f} seconds')
    filtered_g, filtered_rb = search_policy.g.copy(), search_policy.rb_vec.copy()

    # Cleanup steps
    num_cleanup_steps = 2500
    cleanup_time = take_cleanup_steps(search_policy, eval_env, num_cleanup_steps)
    print(f'Took {num_cleanup_steps} cleanup steps in {cleanup_time:.2f} seconds')

    set_env_difficulty(eval_env, difficulty)
    search_policy.reset_stats()
    success_rate, eval_time = eval_search_policy(search_policy, eval_env, num_evals=num_evals)
    print(f'Cleaned {search_policy} has success rate {success_rate:.2f}, evaluated in {eval_time:.2f} seconds')
    cleaned_g, cleaned_rb = search_policy.g.copy(), search_policy.rb_vec.copy()

    return (initial_g, initial_rb), (filtered_g, filtered_rb), (cleaned_g, cleaned_rb)