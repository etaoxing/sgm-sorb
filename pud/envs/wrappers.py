from pud.dependencies import *

# https://www.tensorflow.org/agents/tutorials/2_environments_tutorial#using_standard_environments
class TimeStep:
    def __init__(self):
        pass


class TimeLimit(gym.Wrapper):
    """End episodes after specified number of steps.

    Resets the environment if either these conditions holds:
        1. The base environment returns done = True
        2. The time limit is exceeded.

    If terminate_on_timeout=True, then returns done = True in case 1 and 2
    If terminate_on_timeout=False, then returns done = True only in case 1
    """

    def __init__(self, env, duration, terminate_on_timeout=False):
        super().__init__(env)
        self.duration = duration
        self.num_steps = None
        self.terminate_on_timeout = terminate_on_timeout

    def reset(self):
        self.step_count = 0
        observation = self.env.reset()
        observation['first_step'] = True
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        self.step_count += 1
        timed_out = self.step_count >= self.duration
        if timed_out or done:
            info['timed_out'] = timed_out
            info['terminal_observation'] = observation
            info['last_step'] = True
            done = done if not self.terminate_on_timeout else True
            observation = self.reset()

        return observation, reward, done, info
