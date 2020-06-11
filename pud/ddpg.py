from pud.dependencies import *
from pud.utils import variance_initializer_

# Returns an action for a given state
class Actor(nn.Module): # TODO: [256, 256], MLP class
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action
        self.reset_parameters()

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        a = self.max_action * a 
        return a

    def reset_parameters(self):
        # nn.init.kaiming_uniform_(self.l1.weight, mode='fan_in', nonlinearity='relu')
        # torch.nn.init.zeros_(self.l1.bias)
        # nn.init.kaiming_uniform_(self.l2.weight, mode='fan_in', nonlinearity='relu')
        # torch.nn.init.zeros_(self.l2.bias)
        # nn.init.uniform_(self.l3.weight, -0.003, 0.003)
        # torch.nn.init.zeros_(self.l3.bias)

        variance_initializer_(self.l1.weight, scale=1./3., mode='fan_in', distribution='uniform')
        torch.nn.init.zeros_(self.l1.bias)
        variance_initializer_(self.l2.weight, scale=1./3., mode='fan_in', distribution='uniform')
        torch.nn.init.zeros_(self.l2.bias)
        nn.init.uniform_(self.l3.weight, -0.003, 0.003)
        torch.nn.init.zeros_(self.l3.bias)


# Returns a Q-value for given state/action pair
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, output_dim=1):
        super().__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256 + action_dim, 256)
        self.l3 = nn.Linear(256, output_dim)

        self.reset_parameters()

    def forward(self, state, action):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(torch.cat([q, action], dim=1)))
        q = self.l3(q)
        return q

    def reset_parameters(self):
        # nn.init.kaiming_uniform_(self.l1.weight, mode='fan_in', nonlinearity='relu')
        # torch.nn.init.zeros_(self.l1.bias)
        # nn.init.kaiming_uniform_(self.l2.weight, mode='fan_in', nonlinearity='relu')
        # torch.nn.init.zeros_(self.l2.bias)
        # nn.init.uniform_(self.l3.weight, -0.003, 0.003)
        # torch.nn.init.zeros_(self.l3.bias)

        variance_initializer_(self.l1.weight, scale=1./3., mode='fan_in', distribution='uniform')
        torch.nn.init.zeros_(self.l1.bias)
        variance_initializer_(self.l2.weight, scale=1./3., mode='fan_in', distribution='uniform')
        torch.nn.init.zeros_(self.l2.bias)
        nn.init.uniform_(self.l3.weight, -0.003, 0.003)
        torch.nn.init.zeros_(self.l3.bias)


class DDPG(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, 
                 discount=0.99,
                 actor_update_interval=1,
                 targets_update_interval=1,
                 tau=0.005,
                 ActorCls=Actor, CriticCls=Critic):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.actor_update_interval = actor_update_interval
        self.targets_update_interval = targets_update_interval
        self.tau = tau

        self.actor = ActorCls(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4, eps=1e-07)

        self.critic = CriticCls(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4, eps=1e-07)

        self.optimize_iterations = 0

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1))
            return self.actor(state).cpu().detach().numpy().flatten()

    def get_q_values(self, state):
        actions = self.actor(state)
        q_values = self.critic(state, actions)
        return q_values

    def critic_loss(self, current_q, target_q, reward, done):
        td_targets = reward + ((1 - done) * self.discount * target_q).detach()
        critic_loss = F.mse_loss(current_q, td_targets)
        # critic_loss = F.smooth_l1_loss(current_q, td_targets) # Huber loss, if used then actor_loss will diverge to max value
        return critic_loss

    def update_actor_target(self):
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_critic_target(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def optimize(self, replay_buffer, iterations=1, batch_size=128):
        opt_info = dict(actor_loss=[], critic_loss=[])
        for _ in range(iterations):
            self.optimize_iterations += 1

            # Each of these are batches 
            state, next_state, action, reward, done = replay_buffer.sample(batch_size)

            current_q = self.critic(state, action)
            target_q = self.critic_target(next_state, self.actor_target(next_state))
            critic_loss = self.critic_loss(current_q, target_q, reward, done)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            opt_info['critic_loss'].append(critic_loss.cpu().detach().numpy())

            if self.optimize_iterations % self.actor_update_interval == 0:
                # Compute actor loss
                actor_loss = -self.get_q_values(state).mean()

                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                opt_info['actor_loss'].append(actor_loss.cpu().detach().numpy())

            # Update the frozen target models
            if self.optimize_iterations % self.targets_update_interval == 0:
                self.update_actor_target()
                self.update_critic_target()

        return opt_info

    def state_dict(self):
        return dict(
            actor=self.actor.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(),
            critic=self.critic.state_dict(),
            critic_optimizer=self.critic_optimizer.state_dict(),
            optimize_iterations=self.optimize_iterations,
        )

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic.load_state_dict(state_dict['critic'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
        self.optimize_iterations = state_dict['optimize_iterations']


def merge_obs_goal(state):
    if isinstance(state, dict) and ('observation' in state and 'goal' in state):
        obs = state['observation']
        goal = state['goal']
        assert obs.shape == goal.shape
        # For 1D observations, simply concatenate them together.
        assert len(obs.shape) == 2
        modified_state = torch.cat([obs, goal], dim=-1)
        assert obs.shape[0] == modified_state.shape[0]
        assert modified_state.shape[1] == obs.shape[1] + goal.shape[1]
    else:
        raise ValueError("Unsupported observation/goal keys: {}".format(state.keys()))
    return modified_state


class GoalConditionedActor(Actor):
    def forward(self, state):
        modified_state = merge_obs_goal(state)
        return super().forward(modified_state)


class GoalConditionedCritic(Critic):
    def forward(self, state, action):
        modified_state = merge_obs_goal(state)
        return super().forward(modified_state, action)


class EnsembledCritic(nn.Module):
    def __init__(self, CriticInstance, ensemble_size=3):
        super().__init__()
        self.ensemble_size = ensemble_size

        self.critics = nn.ModuleList([CriticInstance])
        for _ in range(self.ensemble_size - 1):
            critic_copy = copy.deepcopy(CriticInstance)
            critic_copy.reset_parameters()
            self.critics.append(critic_copy)

    def forward(self, *args, **kwargs):
        q_list = [critic(*args, **kwargs) for critic in self.critics]
        return q_list

    def state_dict(self):
        state_dict = {}
        for i, critic in enumerate(self.critics):
            state_dict[f'critic_{i}'] = critic.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        for i, critic in enumerate(self.critics):
            critic.load_state_dict(state_dict[f'critic_{i}'])


class UVFDDPG(DDPG):
    def __init__(self, *args,
                 discount=1,
                 num_bins=1,
                 use_distributional_rl=False,
                 ensemble_size=1,
                 CriticCls=GoalConditionedCritic,
                 **kwargs):
        self.num_bins = num_bins # used if distributional_rl=True
        self.use_distributional_rl = use_distributional_rl
        self.ensemble_size = ensemble_size

        if self.use_distributional_rl:
            CriticCls = functools.partial(CriticCls, output_dim=self.num_bins)
            assert discount == 1

        super().__init__(*args, discount=discount, ActorCls=GoalConditionedActor, CriticCls=CriticCls, **kwargs)
        if self.ensemble_size > 1:
            self.critic = EnsembledCritic(self.critic, ensemble_size=ensemble_size)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_target.load_state_dict(self.critic.state_dict())

            # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
            for i in range(1, len(self.critic.critics)): # first copy already added
                critic_copy = self.critic.critics[i]
                self.critic_optimizer.add_param_group({'params': critic_copy.parameters()})
                # https://stackoverflow.com/questions/51756913/in-pytorch-how-do-you-use-add-param-group-with-a-optimizer

    def select_action(self, state):
        with torch.no_grad():
            state = dict(
                observation=torch.FloatTensor(state['observation'].reshape(1, -1)),
                goal=torch.FloatTensor(state['goal'].reshape(1, -1)),
            )
            return self.actor(state).cpu().detach().numpy().flatten()

    def get_q_values(self, state, aggregate='mean'):
        q_values = super().get_q_values(state)
        if not isinstance(q_values, list):
            q_values_list = [q_values]
        else:
            q_values_list = q_values

        expected_q_values_list = []
        if self.use_distributional_rl:
            for q_values in q_values_list:
                q_probs = F.softmax(q_values, dim=1)
                batch_size = q_probs.shape[0]
                # NOTE: We want to compute the value of each bin, which is the
                # negative distance. Without properly negating this, the actor is
                # optimized to take the *worst* actions.
                neg_bin_range = -torch.arange(1, self.num_bins + 1, dtype=torch.float)
                tiled_bin_range = neg_bin_range.unsqueeze(0).repeat(batch_size, 1)
                assert q_probs.shape == tiled_bin_range.shape
                # Take the inner product between these two tensors
                expected_q_values = torch.sum(q_probs * tiled_bin_range, dim=1, keepdim=True)
                expected_q_values_list.append(expected_q_values)
        else:
            expected_q_values_list = q_values_list

        expected_q_values = torch.stack(expected_q_values_list)
        if aggregate is not None:
            if aggregate == 'mean':
                expected_q_values = torch.mean(expected_q_values, dim=0)
            elif aggregate == 'min':
                expected_q_values, _ = torch.min(expected_q_values, dim=0)
            else:
                raise ValueError

        if not self.use_distributional_rl:
            # Clip the q values if not using distributional RL. If using
            # distributional RL, the q values are implicitly clipped.
            min_q_value = -1.0 * self.num_bins
            max_q_value = 0.0
            expected_q_values = torch.clamp(expected_q_values, min_q_value, max_q_value)

        return expected_q_values

    def critic_loss(self, current_q, target_q, reward, done):
        if not isinstance(current_q, list):
            current_q_list = [current_q]
            target_q_list = [target_q]
        else:
            current_q_list = current_q
            target_q_list = target_q

        critic_loss_list = []
        for current_q, target_q in zip(current_q_list, target_q_list):
            if self.use_distributional_rl:
                # Compute distributional td targets
                target_q_probs = F.softmax(target_q, dim=1)
                batch_size = target_q_probs.shape[0]
                one_hot = torch.zeros(batch_size, self.num_bins)
                one_hot[:, 0] = 1

                # Calculate the shifted probabilities
                # Fist column: Since episode didn't terminate, probability that the
                # distance is 1 equals 0.
                col_1 = torch.zeros((batch_size, 1))
                # Middle columns: Simply the shifted probabilities.
                col_middle = target_q_probs[:, :-2]
                # Last column: Probability of taking at least n steps is sum of
                # last two columns in unshifted predictions:
                col_last = torch.sum(target_q_probs[:, -2:], dim=1, keepdim=True)
                shifted_target_q_probs = torch.cat([col_1, col_middle, col_last], dim=1)
                assert one_hot.shape == shifted_target_q_probs.shape
                td_targets = torch.where(done.bool(), one_hot, shifted_target_q_probs).detach()

                critic_loss = torch.mean(-torch.sum(td_targets * torch.log_softmax(current_q, dim=1), dim=1)) # https://github.com/tensorflow/tensorflow/issues/21271
            else:
                critic_loss = super().critic_loss(current_q, target_q, reward, done)
            critic_loss_list.append(critic_loss)
        critic_loss = torch.mean(torch.stack(critic_loss_list))
        return critic_loss

    def get_dist_to_goal(self, state, **kwargs):
        with torch.no_grad():
            state = dict(
                observation=torch.FloatTensor(state['observation']),
                goal=torch.FloatTensor(state['goal']),
            )
            q_values = self.get_q_values(state, **kwargs)
            return -1.0 * q_values.cpu().detach().numpy().squeeze(-1)

    def get_pairwise_dist(self, obs_vec, goal_vec=None, aggregate='mean', max_search_steps=7, masked=False):
        """Estimates the pairwise distances.

          obs_vec: Array containing observations
          goal_vec: (optional) Array containing a second set of observations. If
                    not specified, computes the pairwise distances between obs_tensor and
                    itself.
          aggregate: (str) How to combine the predictions from the ensemble. Options
                     are to take the minimum predicted q value (i.e., the maximum distance),
                     the mean, or to simply return all the predictions.
          max_search_steps: (int)
          masked: (bool) Whether to ignore edges that are too long, as defined by
                  max_search_steps.
        """
        if goal_vec is None:
            goal_vec = obs_vec

        dist_matrix = []
        for obs_index in range(len(obs_vec)):
            obs = obs_vec[obs_index]
            # obs_repeat_tensor = np.ones_like(goal_vec) * np.expand_dims(obs, 0)
            obs_repeat_tensor = np.repeat([obs], len(goal_vec), axis=0)
            state = {'observation': obs_repeat_tensor, 'goal': goal_vec}
            dist = self.get_dist_to_goal(state, aggregate=aggregate)
            dist_matrix.append(dist)

        pairwise_dist = np.stack(dist_matrix)
        if aggregate is None:
            pairwise_dist = np.transpose(pairwise_dist, [1, 0, 2])
        # else:
        #     pairwise_dist = np.expand_dims(pairwise_dist, 0)

        if masked:
            mask = (pairwise_dist > max_search_steps)
            return np.where(mask, np.full(pairwise_dist.shape, np.inf), pairwise_dist)
        else:
            return pairwise_dist
