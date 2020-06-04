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
        """
        super().__init__(agent)
        self.rb_vec = rb_vec
        self.pdist = pdist
        self.max_search_steps = max_search_steps
        self.open_loop = open_loop
        self.weighted_path_planning = weighted_path_planning
        self.no_waypoint_hopping = no_waypoint_hopping
        self.aggregate = aggregate
        self.g = self.build_graph()

        pdist2 = self.agent.get_pairwise_dist(self.rb_vec, masked=True, aggregate=self.aggregate)
        self.rb_distances = scipy.sparse.csgraph.floyd_warshall(pdist2, directed=True)

    def build_graph(self):
        g = nx.DiGraph()
        pdist_combined = np.max(self.pdist, axis=0)
        for i, s_i in enumerate(self.rb_vec):
            for j, s_j in enumerate(self.rb_vec):
                length = pdist_combined[i, j]
                if length < self.max_search_steps:
                    g.add_edge(i, j, weight=length)
        return g

    def construct_planning_graph(self, state):
        start_to_rb = self.agent.get_pairwise_dist([state['observation']],
                                                   self.rb_vec,
                                                   aggregate=self.aggregate,
                                                   max_search_steps=self.max_search_steps,
                                                   masked=True)
        rb_to_goal  = self.agent.get_pairwise_dist(self.rb_vec,
                                                  [state['goal']],
                                                  aggregate=self.aggregate,
                                                  max_search_steps=self.max_search_steps,
                                                  masked=True)

        planning_graph = self.g.copy()
        for i, (dist_from_start, dist_to_goal) in enumerate(zip(start_to_rb, rb_to_goal)):
            if dist_from_start < self.max_search_steps:
                planning_graph.add_edge('start', i, weight=dist_from_start)
            if dist_to_goal < self.max_search_steps:
                planning_graph.add_edge(i, 'goal', weight=dist_to_goal)
        return planning_graph

    def get_path(self, state):
        g2 = self.construct_planning_graph(state)
        if self.weighted_path_planning:
            path = nx.shortest_path(g2, source='start', target='goal', weight='weight')
        else:
            path = nx.shortest_path(g2, source='start', target='goal')

        edge_lengths = []
        for (i, j) in zip(path[:-1], path[1:]):
            edge_lengths.append(g2[i][j]['weight'])

        wypt_to_goal_dist = np.cumsum(edge_lengths[::-1])[::-1]  # Reverse CumSum
        wypt_vec = list(path)[1:-1]
        return wypt_vec, wypt_to_goal_dist[1:]

    def get_waypoint(self, state):
        obs_to_rb_dist = self.agent.get_pairwise_dist([state['observation']],
                                                      self.rb_vec,
                                                      aggregate=self.aggregate,
                                                      max_search_steps=self.max_search_steps,
                                                      masked=True) # B x A
        rb_to_goal_dist  = self.agent.get_pairwise_dist(self.rb_vec,
                                                       [state['goal']],
                                                       aggregate=self.aggregate,
                                                       max_search_steps=self.max_search_steps,
                                                       masked=True) # A x B

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

    def select_action(self, state):
        goal = state['goal']
        dist_to_goal = self.agent.get_dist_to_goal({k: [v] for k, v in state.items()})[0]
        if self.open_loop:
            raise NotImplementedError
        else:
            wypt, dist_to_goal_via_wypt = self.get_waypoint(state)

        if (dist_to_goal_via_wypt < dist_to_goal) or (dist_to_goal > self.max_search_steps):
            state['goal'] = wypt
        else:
            state['goal'] = goal
        return super().select_action(state)
