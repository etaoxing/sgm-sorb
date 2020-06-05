from pud.dependencies import *

class BasePolicy:
    def __init__(self, agent):
        self.agent = agent

    def select_action(self, state):
        return self.agent.select_action(state)


class GaussianPolicy(BasePolicy):
    def __init__(self, agent, noise_scale=1.0):
        super().__init__(agent)
        self.noise_scale = noise_scale

    def select_action(self, state):
        action = super().select_action(state)
        action += np.random.normal(0, self.agent.max_action *
                                   self.noise_scale, size=self.agent.action_dim)
        action = action.clip(-self.agent.max_action, self.agent.max_action)
        return action


class SearchPolicy(BasePolicy):
    def __init__(self,
                 agent,
                 rb_vec,
                 pdist,
                 aggregate='min',
                 max_search_steps=7,
                 open_loop=False,
                 weighted_path_planning=False,
                 no_waypoint_hopping=False,
                 cleanup=False,
                 ):
        """
        Args:
            rb_vec: a replay buffer vector storing the observations that will be used as nodes in the graph
            pdist: a matrix of dimension len(rb_vec) x len(rb_vec) where pdist[i,j] gives the distance going from
                   rb_vec[i] to rb_vec[j]
            max_search_steps: (int)
            open_loop: if True, only performs search once at the beginning of the episode
            weighted_path_planning: whether or not to use edge weights when planning a shortest path from start to goal
            no_waypoint_hopping: if True, will not try to proceed to goal until all waypoints have been reached
            cleanup: if True, will prune edges when fail to reach waypoint after `attempt_cutoff`
        """
        super().__init__(agent)
        self.rb_vec = rb_vec
        self.pdist = pdist

        self.aggregate = aggregate
        self.max_search_steps = max_search_steps
        self.open_loop = open_loop
        self.weighted_path_planning = weighted_path_planning

        self.no_waypoint_hopping = no_waypoint_hopping
        self.cleanup = cleanup
        self.attempt_cutoff = 3 * max_search_steps

        self.g = self.build_rb_graph()
        if not self.open_loop:
            pdist2 = self.agent.get_pairwise_dist(self.rb_vec,
                                                  aggregate=self.aggregate,
                                                  max_search_steps=self.max_search_steps,
                                                  masked=True)
            self.rb_distances = scipy.sparse.csgraph.floyd_warshall(pdist2, directed=True)
        self.reset_stats()

    def reset_stats(self):
        self.stats = dict(
            path_planning_attempts=0,
            path_planning_fails=0,
            graph_search_time=0,
            localization_fails=0,
        )

    def get_stats(self):
        return self.stats

    def build_rb_graph(self):
        g = nx.DiGraph()
        pdist_combined = np.max(self.pdist, axis=0)
        for i, s_i in enumerate(self.rb_vec):
            for j, s_j in enumerate(self.rb_vec):
                length = pdist_combined[i, j]
                if length < self.max_search_steps:
                    g.add_edge(i, j, weight=length)
        return g

    def get_pairwise_dist_to_rb(self, state):
        start_to_rb_dist = self.agent.get_pairwise_dist([state['observation']],
                                                        self.rb_vec,
                                                        aggregate=self.aggregate,
                                                        max_search_steps=self.max_search_steps,
                                                        masked=True)
        rb_to_goal_dist  = self.agent.get_pairwise_dist(self.rb_vec,
                                                        [state['goal']],
                                                        aggregate=self.aggregate,
                                                        max_search_steps=self.max_search_steps,
                                                        masked=True)
        return start_to_rb_dist, rb_to_goal_dist

    def get_closest_waypoint(self, state):
        """
        For closed loop replanning at each step. Uses the precomputed distances
        `rb_distances` b/w states in `rb_vec`
        """
        obs_to_rb_dist, rb_to_goal_dist = self.get_pairwise_dist_to_rb(state)
        # (B x A), (A x B)

        # The search_dist tensor should be (B x A x A)
        search_dist = sum([
            np.expand_dims(obs_to_rb_dist, 2),
            np.expand_dims(self.rb_distances, 0),
            np.expand_dims(np.transpose(rb_to_goal_dist), 1)
        ]) # elementwise sum

        # We assume a batch size of 1.
        min_search_dist = np.min(search_dist)
        waypoint_index = np.argmin(np.min(search_dist, axis=2), axis=1)[0]
        waypoint = self.rb_vec[waypoint_index]

        return waypoint, min_search_dist

    def construct_planning_graph(self, state):
        start_to_rb_dist, rb_to_goal_dist = self.get_pairwise_dist_to_rb(state)
        planning_graph = self.g.copy()

        for i, (dist_from_start, dist_to_goal) in enumerate(zip(start_to_rb_dist.flatten(), rb_to_goal_dist.flatten())):
            if dist_from_start < self.max_search_steps:
                planning_graph.add_edge('start', i, weight=dist_from_start)
            if dist_to_goal < self.max_search_steps:
                planning_graph.add_edge(i, 'goal', weight=dist_to_goal)

        if not np.any(start_to_rb_dist < self.max_search_steps) or not np.any(
                rb_to_goal_dist < self.max_search_steps):
            self.stats['localization_fails'] += 1

        return planning_graph

    def get_path(self, state):
        g2 = self.construct_planning_graph(state)
        try:
            self.stats['path_planning_attempts'] += 1
            graph_search_start = time.process_time()

            if self.weighted_path_planning:
                path = nx.shortest_path(g2, source='start', target='goal', weight='weight')
            else:
                path = nx.shortest_path(g2, source='start', target='goal')
        except:
            self.stats['path_planning_fails'] += 1
            raise RuntimeError(f'Failed to find path in graph (|V|={g2.number_of_nodes()}, |E|={g2.number_of_edges()})')
        finally:
            graph_search_end = time.process_time()
            self.stats['graph_search_time'] += graph_search_end - graph_search_start

        edge_lengths = []
        for (i, j) in zip(path[:-1], path[1:]):
            edge_lengths.append(g2[i][j]['weight'])

        waypoint_to_goal_dist = np.cumsum(edge_lengths[::-1])[::-1]  # Reverse CumSum
        waypoint_indices = list(path)[1:-1]
        return waypoint_indices, waypoint_to_goal_dist[1:]

    def initialize_path(self, state):
        self.waypoint_indices, self.waypoint_to_goal_dist_vec = self.get_path(state)
        self.waypoint_counter = 0
        self.waypoint_attempts = 0
        self.reached_final_waypoint = False

    def get_current_waypoint(self):
        waypoint_index = self.waypoint_indices[self.waypoint_counter]
        waypoint = self.rb_vec[waypoint_index]
        return waypoint, waypoint_index

    def get_waypoints(self):
        waypoints = [self.rb_vec[i] for i in self.waypoint_indices]
        return waypoints

    def reached_waypoint(self, dist_to_waypoint, state, waypoint_index):
        return dist_to_waypoint < self.max_search_steps

    def select_action(self, state):
        goal = state['goal']
        dist_to_goal = self.agent.get_dist_to_goal({k: [v] for k, v in state.items()})[0]
        if self.open_loop or self.cleanup:
            if state.get('first_step', False): self.initialize_path(state)
            
            if self.cleanup and (self.waypoint_attempts >= self.attempt_cutoff):
                # prune edge and replan
                if self.waypoint_counter != 0 and not self.reached_final_waypoint:
                    src_node = self.waypoint_indices[self.waypoint_counter - 1]
                    dest_node = self.waypoint_indices[self.waypoint_counter]
                    self.g.remove_edge(src_node, dest_node)
                self.initialize_path(state)

            waypoint, waypoint_index = self.get_current_waypoint()
            state['goal'] = waypoint
            dist_to_waypoint = self.agent.get_dist_to_goal({k: [v] for k, v in state.items()})[0]

            if self.reached_waypoint(dist_to_waypoint, state, waypoint_index):
                if not self.reached_final_waypoint:
                    self.waypoint_attempts = 0

                self.waypoint_counter += 1
                if self.waypoint_counter >= len(self.waypoint_indices):
                    self.reached_final_waypoint = True
                    self.waypoint_counter = len(self.waypoint_indices) - 1

                waypoint, waypoint_index = self.get_current_waypoint()
                state['goal'] = waypoint
                dist_to_waypoint = self.agent.get_dist_to_goal({k: [v] for k, v in state.items()})[0]

            dist_to_goal_via_waypoint = dist_to_waypoint + self.waypoint_to_goal_dist_vec[self.waypoint_counter]
        else:
            # closed loop, replan waypoint at each step
            waypoint, dist_to_goal_via_waypoint = self.get_closest_waypoint(state)

        if (self.no_waypoint_hopping and not self.reached_final_waypoint) or \
           (dist_to_goal_via_waypoint < dist_to_goal) or \
           (dist_to_goal > self.max_search_steps):
            state['goal'] = waypoint
            if self.open_loop: self.waypoint_attempts += 1
        else:
            state['goal'] = goal
        return super().select_action(state)

