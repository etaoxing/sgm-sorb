from pud.dependencies import *
from pud.collector import Collector
from pud.envs.simple_navigation_env import plot_walls, set_env_difficulty
from pud.utils import set_global_seed, set_env_seed

def visualize_trajectory(agent, eval_env, difficulty=0.5):
    set_env_difficulty(eval_env, difficulty)

    plt.figure(figsize=(8, 4))
    for col_index in range(2):
        plt.subplot(1, 2, col_index + 1)
        plot_walls(eval_env.walls)
        goal, observations_list, _, _ = Collector.get_trajectory(agent, eval_env)
        obs_vec = np.array(observations_list)

        print(f'traj {col_index}, num steps: {len(obs_vec)}')

        plt.plot(obs_vec[:, 0], obs_vec[:, 1], 'b-o', alpha=0.3)
        plt.scatter([obs_vec[0, 0]], [obs_vec[0, 1]], marker='+',
                    color='red', s=200, label='start')
        plt.scatter([obs_vec[-1, 0]], [obs_vec[-1, 1]], marker='+',
                    color='green', s=200, label='end')
        plt.scatter([goal[0]], [goal[1]], marker='*',
                    color='green', s=200, label='goal')
        if col_index == 0:
            plt.legend(loc='lower left', bbox_to_anchor=(0.3, 1), ncol=3, fontsize=16)
    plt.show()


def visualize_buffer(rb_vec, eval_env):
    plt.figure(figsize=(6, 6))
    plt.scatter(*rb_vec.T)
    plot_walls(eval_env.walls)
    plt.show()


def visualize_pairwise_dists(pdist):
    plt.figure(figsize=(6, 3))
    plt.hist(pdist.flatten(), bins=range(20))
    plt.xlabel('predicted distance')
    plt.ylabel('number of (s, g) pairs')
    plt.show()


def visualize_graph(rb_vec, eval_env, pdist, cutoff=7, edges_to_display=8):
    plt.figure(figsize=(6, 6))
    plot_walls(eval_env.walls)
    pdist_combined = np.max(pdist, axis=0)
    plt.scatter(*rb_vec.T)
    for i, s_i in enumerate(rb_vec):
        for count, j in enumerate(np.argsort(pdist_combined[i])):
            if count < edges_to_display and pdist_combined[i, j] < cutoff:
                s_j = rb_vec[j]
                plt.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c='k', alpha=0.5)
    plt.show()


def visualize_graph_ensemble(rb_vec, eval_env, pdist, cutoff=7, edges_to_display=8):
    ensemble_size = pdist.shape[0]
    plt.figure(figsize=(5 * ensemble_size, 4))
    for col_index in range(ensemble_size):
        plt.subplot(1, ensemble_size, col_index + 1)
        plot_walls(eval_env.walls)
        plt.title('critic %d' % (col_index + 1))

        plt.scatter(*rb_vec.T)
        for i, s_i in enumerate(rb_vec):
            for count, j in enumerate(np.argsort(pdist[col_index, i])):
                if count < edges_to_display and pdist[col_index, i, j] < cutoff:
                    s_j = rb_vec[j]
                    plt.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c='k', alpha=0.5)
    plt.show()


def visualize_full_graph(g, rb_vec, eval_env):
    plt.figure(figsize=(6, 6))
    plot_walls(eval_env.walls)
    plt.scatter(rb_vec[g.nodes, 0], rb_vec[g.nodes, 1])

    edges_to_plot = g.edges
    edges_to_plot = np.array(list(edges_to_plot))

    for i, j in edges_to_plot:
        s_i = rb_vec[i]
        s_j = rb_vec[j]
        plt.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c='k', alpha=0.5)

    plt.title(f'|V|={g.number_of_nodes()}, |E|={len(edges_to_plot)}')
    plt.show()


def visualize_search_path(search_policy, eval_env, difficulty=0.5):
    set_env_difficulty(eval_env, difficulty)

    if search_policy.open_loop:
        state = eval_env.reset()
        start = state['observation']
        goal = state['goal']

        search_policy.select_action(state)
        waypoints = search_policy.get_waypoints()
    else:
        goal, observations, waypoints, _ = Collector.get_trajectory(search_policy, eval_env)
        start = observations[0]

    plt.figure(figsize=(6, 6))
    plot_walls(eval_env.walls)

    waypoint_vec = np.array(waypoints)

    print(f'waypoints: {waypoint_vec}')
    print(f'waypoints shape: {waypoint_vec.shape}')
    print(f'start: {start}')
    print(f'goal: {goal}')

    plt.scatter([start[0]], [start[1]], marker='+',
                color='red', s=200, label='start')
    plt.scatter([goal[0]], [goal[1]], marker='*',
                color='green', s=200, label='goal')
    plt.plot(waypoint_vec[:, 0], waypoint_vec[:, 1], 'y-s', alpha=0.3, label='waypoint')
    plt.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.15), ncol=4, fontsize=16)
    plt.show()


def visualize_compare_search(agent, search_policy, eval_env, difficulty=0.5, seed=0):
    set_env_difficulty(eval_env, difficulty)

    plt.figure(figsize=(12, 5))
    for col_index in range(2):
        title = 'no search' if col_index == 0 else 'search'
        plt.subplot(1, 2, col_index + 1)
        plot_walls(eval_env.walls)
        use_search = (col_index == 1)

        set_global_seed(seed)
        set_env_seed(eval_env, seed + 1)

        if use_search:
            policy = search_policy
        else:
            policy = agent
        goal, observations, waypoints, _ = Collector.get_trajectory(policy, eval_env)
        start = observations[0]

        obs_vec = np.array(observations)
        waypoint_vec = np.array(waypoints)

        print(f'policy: {title}')
        print(f'start: {start}')
        print(f'goal: {goal}')
        print(f'steps: {obs_vec.shape[0] - 1}')
        print('-' * 10)

        plt.plot(obs_vec[:, 0], obs_vec[:, 1], 'b-o', alpha=0.3)
        plt.scatter([start[0]], [start[1]], marker='+',
                    color='red', s=200, label='start')
        plt.scatter([obs_vec[-1, 0]], [obs_vec[-1, 1]], marker='+',
                    color='green', s=200, label='end')
        plt.scatter([goal[0]], [goal[1]], marker='*',
                    color='green', s=200, label='goal')
        plt.title(title, fontsize=24)

        if use_search:
            plt.plot(waypoint_vec[:, 0], waypoint_vec[:, 1], 'y-s', alpha=0.3, label='waypoint')
            plt.legend(loc='lower left', bbox_to_anchor=(-0.8, -0.15), ncol=4, fontsize=16)
    plt.show()