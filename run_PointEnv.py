from pud.dependencies import *
from pud.utils import set_global_seed, set_env_seed

max_episode_steps = 20
env_name = 'FourRooms'  # Choose one of the environments shown above. 
resize_factor = 5  # Inflate the environment to increase the difficulty.
# resize_factor = 1
thin = False # If True, resize by expanding open space, not walls, to make walls thin
desc_name = "thinned_" + env_name.lower() if thin else env_name.lower()
seed = 0

set_global_seed(seed)

from pud.envs.simple_navigation_env import env_load_fn
env = env_load_fn(env_name, max_episode_steps,
                     resize_factor=resize_factor,
                     terminate_on_timeout=False,
                     thin=thin)
set_env_seed(env, seed + 1)

eval_env = env_load_fn(env_name, max_episode_steps,
                          resize_factor=resize_factor,
                          terminate_on_timeout=True,
                          thin=thin)
set_env_seed(eval_env, seed + 2)

from pud.ddpg import UVFDDPG
state_dim = env.observation_space['observation'].shape[0] # concatenating goal
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

print(f'state dim: {state_dim}, action dim: {action_dim}, max action: {max_action}')

agent = UVFDDPG(
    int(2 * state_dim),
    action_dim,
    max_action,
    discount=1,
    max_episode_steps=max_episode_steps,
    ensemble_size=3,
    use_distributional_rl=True,
    targets_update_interval=5, # tfagents default
    tau=0.05,
)

print(agent)

from pud.buffer import ReplayBuffer
replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=1000)

if False:
    from pud.runner import train_eval
    
    train_eval(agent,
            replay_buffer,
            env,
            eval_env,
            initial_collect_steps=1000,
            eval_interval=1000,
            num_eval_episodes=10,
            num_iterations=30000,
            )
    torch.save(agent.state_dict(), 'agent.pth')
elif True:
    ckpt_file = os.path.join('workdirs', 'uvfddpg_distributional1_ensemble3_rescale5', 'agent.pth')
    agent.load_state_dict(torch.load(ckpt_file))
    agent.eval()
    
    # from pud.visualize import visualize_trajectory
    # eval_env.duration = 100 # We'll give the agent lots of time to try to find the goal.
    # visualize_trajectory(agent, eval_env, difficulty=0.5)

    # We now will implement the search policy, which automatically finds these waypoints via graph search. 
    # The first step is to fill the replay buffer with random data.
    #
    from pud.collector import Collector
    env.set_sample_goal_args(prob_constraint=0.0, min_dist=0, max_dist=np.inf)
    rb_vec = Collector.sample_initial_states(eval_env, replay_buffer.max_size)

    # from pud.visualize import visualize_buffer
    # visualize_buffer(rb_vec, eval_env)

    pdist = agent.get_pairwise_dist(rb_vec, aggregate=None)
    # from scipy.spatial import distance
    # euclidean_dists = distance.pdist(rb_vec)

    # As a sanity check, we'll plot the pairwise distances between all 
    # observations in the replay buffer. We expect to see a range of values 
    # from 1 to 20. Distributional RL implicitly caps the maximum predicted 
    # distance by the largest bin. We've used 20 bins, so the critic 
    # predicts 20 for all states that are at least 20 steps away from one another.
    # 
    # from pud.visualize import visualize_pairwise_dists
    # visualize_pairwise_dists(pdist)

    # With these distances, we can construct a graph. Nodes in the graph are 
    # observations in our replay buffer. We connect observations with edges 
    # whose lengths are equal to the predicted distance between those observations. 
    # Since it is hard to visualize the edge lengths, we included a slider that 
    # allows you to only show edges whose predicted length is less than some threshold.
    # ---
    # Our method learns a collection of critics, each of which makes an independent 
    # prediction for the distance between two states. Because each network may make 
    # bad predictions for pairs of states it hasn't seen before, we act in 
    # a *risk-averse* manner by using the maximum predicted distance across our 
    # ensemble. That is, we act pessimistically, only adding an edge 
    # if *all* critics think that this pair of states is nearby.
    #
    # from pud.visualize import visualize_graph
    # visualize_graph(rb_vec, eval_env, pdist)

    # We can also visualize the predictions from each critic. 
    # Note that while each critic may make incorrect decisions 
    # for distant states, their predictions in aggregate are correct.
    # 
    # from pud.visualize import visualize_graph_ensemble
    # visualize_graph_ensemble(rb_vec, eval_env, pdist)

    from pud.policies import SearchPolicy
    search_policy = SearchPolicy(agent, rb_vec, pdist, open_loop=False)

    # Plot the search path found by the search policy
    # 
    # from pud.visualize import visualize_search_path
    # visualize_search_path(search_policy, eval_env, difficulty=0.9)

    # Now, we'll use that path to guide the agent towards the goal. 
    # On the left, we plot rollouts from the baseline goal-conditioned policy. 
    # On the right, we use that same policy to reach each of the waypoints 
    # leading to the goal. As before, the slider allows you to change the 
    # distance to the goal. Note that only the search policy is able to reach distant goals.
    #
    from pud.visualize import visualize_compare_search
    eval_env.duration = 300 # We'll give the agent lots of time to try to find the goal.
    visualize_compare_search(agent, search_policy, eval_env, difficulty=0.9)