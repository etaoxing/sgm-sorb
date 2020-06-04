from pud.dependencies import *

class Collector:
    def __init__(self, policy, buffer, env, initial_collect_steps=0):
        self.buffer = buffer
        self.env = env
        self.policy = policy

        self.steps = 0
        self.state = env.reset()
        self.initial_collect_steps = initial_collect_steps

    def step(self, num_steps):
        for _ in range(num_steps):
            if self.steps < self.initial_collect_steps:
                action = self.env.action_space.sample()
            else:
                action = self.policy.select_action(self.state)

            next_state, reward, done, info = self.env.step(np.copy(action))
            if info.get('last_timestep', False):
                self.buffer.add(self.state, action, info['terminal_observation'], reward, done)
                self.state = next_state
            else:
                self.buffer.add(self.state, action, next_state, reward, done)
                self.state = next_state

            self.steps += 1

    @classmethod
    def sample_initial_states(cls, eval_env, num_states):
        rb_vec = []
        for _ in range(num_states):
            rb_vec.append(eval_env.reset())
        rb_vec = np.array([x['observation'] for x in rb_vec])
        return rb_vec

    @classmethod
    def eval_agent(cls, policy, eval_env, num_episodes):
        e = 0
        r = 0
        rewards = []

        state = eval_env.reset()
        while e < num_episodes:
            action = policy.select_action(state)
            state, reward, done, info = eval_env.step(np.copy(action))

            r += reward
            if done:
                rewards.append(r)
                e += 1
                r = 0
        return rewards

    @classmethod
    def get_trajectory(cls, policy, eval_env):
        ep_observation_list = []
        ep_waypoint_list = []
        ep_reward_list = []

        state = eval_env.reset()
        ep_goal = state['goal']
        while True:
            ep_observation_list.append(state['observation'])
            action = policy.select_action(state) # NOTE: state['goal'] may be modified
            ep_waypoint_list.append(state['goal'])
            state, reward, done, info = eval_env.step(np.copy(action))

            ep_reward_list.append(reward)
            if done:
                ep_observation_list.append(info['terminal_observation']['observation'])
                break

        return ep_goal, ep_observation_list, ep_waypoint_list, ep_reward_list