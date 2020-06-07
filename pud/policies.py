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

        self.aggregate = aggregate
        self.max_search_steps = max_search_steps
        self.open_loop = open_loop
        self.weighted_path_planning = weighted_path_planning

        self.no_waypoint_hopping = no_waypoint_hopping
        self.cleanup = False
        self.attempt_cutoff = 3 * max_search_steps

        self.build_rb_graph()
        if not self.open_loop:
            pdist2 = self.agent.get_pairwise_dist(self.rb_vec,
                                                  aggregate=self.aggregate,
                                                  max_search_steps=self.max_search_steps,
                                                  masked=True)
            self.rb_distances = scipy.sparse.csgraph.floyd_warshall(pdist2, directed=True)
        self.reset_stats()

    def __str__(self):
        s = f'{self.__class__.__name__} (|V|={self.g.number_of_nodes()}, |E|={self.g.number_of_edges()})'
        return s

    def reset_stats(self):
        self.stats = dict(
            path_planning_attempts=0,
            path_planning_fails=0,
            graph_search_time=0,
            localization_fails=0,
        )

    def get_stats(self):
        return self.stats

    def set_cleanup(self, cleanup): # if True, will prune edges when fail to reach waypoint after `attempt_cutoff`
        self.cleanup = cleanup

    def build_rb_graph(self):
        g = nx.DiGraph()
        pdist_combined = np.max(self.pdist, axis=0)
        for i, s_i in enumerate(self.rb_vec):
            for j, s_j in enumerate(self.rb_vec):
                length = pdist_combined[i, j]
                if length < self.max_search_steps:
                    g.add_edge(i, j, weight=length)
        self.g = g

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


class SparseSearchPolicy(SearchPolicy):
    def __init__(self, *args,
                 beta=0.05,
                 edge_cutoff=10,
                 norm_cutoff=0.05,
                 consistency_cutoff=5,
                 waypoint_consistency_cutoff=1,
                 k_nearest=5,
                 localize_to_nearest=True,
                 open_loop=True,
                 no_waypoint_hopping=True,
                 **kwargs):
        """
        Args:
            beta: percentage assigned to newest embedding space observation
                  in exponential moving average / variance calculations
            edge_cutoff: draw directed edges between nodes when their qvalue
                         distance is less than edge_cutoff
            norm_cutoff: define neighbors if their embedding 
                         norm distance is less than norm_cutoff
            consistency_cutoff: qval consistency cutoff
            waypoint_consistency_cutoff: waypoint qval consistency cutoff
            k_nearest: for filtering the nearest k nodes using edge weight
            localize_to_nearest: if True, will incrementally add edges with 
                                 incoming start and goal nodes until path
                                 exists from start to goal; otherwise, adds 
                                 all edges with incoming start and goal nodes 
                                 that have distance less than `max_search_steps`
        """
        self.beta = beta
        self.edge_cutoff = edge_cutoff
        self.norm_cutoff = norm_cutoff
        self.consistency_cutoff = consistency_cutoff
        self.waypoint_consistency_cutoff = waypoint_consistency_cutoff
        self.k_nearest = k_nearest
        self.localize_to_nearest = localize_to_nearest
        super().__init__(*args, open_loop=open_loop, no_waypoint_hopping=no_waypoint_hopping, **kwargs)

    def filter_keep_k_nearest(self):
        """
        For each node in the graph, keeps only the k outgoing edges with lowest weight.
        """
        for node in self.g.nodes():
            edges = list(self.g.edges(nbunch=node, data='weight', default=np.inf))
            edges.sort(key=lambda x: x[2])
            try:
                edges_to_remove = edges[self.k_nearest:]
            except IndexError:
                edges_to_remove = []
            self.g.remove_edges_from(edges_to_remove)

    def construct_planning_graph(self, state):
        if not self.localize_to_nearest:
            return super().construct_planning_graph(state)

        start_to_rb_dist, rb_to_goal_dist = self.get_pairwise_dist_to_rb(state)
        start_to_rb_dist, rb_to_goal_dist = start_to_rb_dist.flatten(), rb_to_goal_dist.flatten()
        planning_graph = self.g.copy()

        sorted_start_indices = np.argsort(start_to_rb_dist)
        sorted_goal_indices = np.argsort(rb_to_goal_dist)
        neighbors_added = 0
        while neighbors_added < len(start_to_rb_dist):
            i = sorted_start_indices[neighbors_added]
            j = sorted_goal_indices[neighbors_added]
            planning_graph.add_edge('start', i, weight=start_to_rb_dist[i])
            planning_graph.add_edge(j, 'goal', weight=rb_to_goal_dist[j])
            try:
                nx.shortest_path(planning_graph, source='start', target='goal')
                break
            except nx.NetworkXNoPath:
                neighbors_added += 1

        if not np.any(start_to_rb_dist < self.max_search_steps) or not np.any(
                rb_to_goal_dist < self.max_search_steps):
            self.stats['localization_fails'] += 1

        return planning_graph

    def reached_waypoint(self, dist_to_waypoint, state, waypoint_index):
        waypoint_qvals_combined = np.max(self.pdist, axis=0)[waypoint_index, :]
        obs_qvals = self.agent.get_pairwise_dist([state['observation']],
                                                  self.rb_vec,
                                                  aggregate=None)
        obs_qvals_combined = np.max(obs_qvals, axis=0).flatten()
        qval_diffs = waypoint_qvals_combined - obs_qvals_combined
        qval_inconsistency = np.linalg.norm(qval_diffs, np.inf)
        return qval_inconsistency < self.waypoint_consistency_cutoff

    def build_rb_graph(self):
        """
        Performs dynamic graph building.
        Args:
            edge_cutoff: draw directed edges between nodes when their qvalue
                         distance is less than edge_cutoff
            beta: percentage assigned to newest embedding space observation
                  in exponential moving average / variance calculations
        """
        self.g = nx.DiGraph()

        # add initial node
        cache_index = 0
        self.g.add_node(0)
        sparse_rb_vec = np.array([self.rb_vec[cache_index]])
        sparse_pdist = self.get_cached_pairwise_dist(np.array([cache_index]), np.array([cache_index]))
        sparse_rb_variances = np.zeros((1))
        cache_indices =  np.array([cache_index])

        for cache_index in range(1, len(self.rb_vec)): 
            # Merge with existing node or create new node
            embedding = self.rb_vec[cache_index]

            # Localize to nearest neighbors in embedding space
            neighbor_indices = np.arange(len(sparse_rb_vec))[self.norm_consistency(embedding, sparse_rb_vec)]

            # Get maximum distances (i.e., minimum qvalues)
            embedding_to_rb = self.get_cached_pairwise_dist(np.array([cache_index]), cache_indices)
            rb_to_embedding = self.get_cached_pairwise_dist(cache_indices, np.array([cache_index]))

            pdist_combined = np.max(sparse_pdist, axis=0)
            embedding_to_rb_combined = np.max(embedding_to_rb, axis=0).flatten()
            rb_to_embedding_combined = np.max(rb_to_embedding, axis=0).flatten()

            # Try to merge with a neighbor based on qvalue consistency
            merged = False
            for neighbor in neighbor_indices:
                # Merge if qvalues are consistent
                if self.qvalue_consistency(neighbor, pdist_combined, embedding_to_rb_combined, rb_to_embedding_combined):
                    difference_from_avg = embedding - sparse_rb_vec[neighbor]
                    sparse_rb_vec[neighbor] = sparse_rb_vec[neighbor] + self.beta * difference_from_avg
                    sparse_rb_variances[neighbor] = (1 - self.beta) * \
                        (sparse_rb_variances[neighbor] + self.beta * np.sum(difference_from_avg ** 2))
                    merged = True
                    break

            # Add node if cannot merge
            if not merged:
                # Add node to graph
                new_index = self.g.number_of_nodes()
                in_indices = np.arange(new_index)[rb_to_embedding_combined < self.edge_cutoff]
                in_weights = rb_to_embedding_combined[in_indices]
                out_indices = np.arange(new_index)[embedding_to_rb_combined < self.edge_cutoff]
                out_weights = embedding_to_rb_combined[out_indices]
                self.g.add_node(new_index)
                self.g.add_weighted_edges_from(zip(in_indices, [new_index] * len(in_indices), in_weights))
                self.g.add_weighted_edges_from(zip([new_index] * len(out_indices), out_indices, out_weights))

                # The only qvalue distance we don't yet have is the new node to itself.
                # Can concatenate qvalues we already have to save |V|^2 qvalue query.
                # Used to update sparse_pdist
                embedding_to_embedding = self.get_cached_pairwise_dist(np.array([cache_index]), 
                                                                       np.array([cache_index]))

                # Add node to other attributes
                sparse_rb_vec = np.concatenate((sparse_rb_vec, [embedding]), axis=0)
                sparse_rb_variances = np.append(sparse_rb_variances, [0])
                sparse_pdist = np.concatenate((sparse_pdist, embedding_to_rb), axis=1)
                sparse_pdist = np.concatenate(
                    (sparse_pdist, np.concatenate((rb_to_embedding, embedding_to_embedding), axis=1)), axis=2)
                cache_indices = np.append(cache_indices, cache_index)

        self.rb_vec = sparse_rb_vec
        self.pdist = sparse_pdist

    def get_cached_pairwise_dist(self, row_indices, col_indices):
        assert len(row_indices.shape) == len(col_indices.shape) == 1
        row_entries = row_indices.shape[0]
        col_entries = col_indices.shape[0]
        row_advanced_index = np.tile(row_indices, (col_entries, 1)).T
        col_advanced_index = np.tile(col_indices, (row_entries, 1))
        if len(self.pdist.shape) == 2:
            return self.pdist[row_advanced_index, col_advanced_index]
        elif len(self.pdist.shape) == 3:
            return self.pdist[:, row_advanced_index, col_advanced_index]
        else:
            raise RuntimeError('Cached pdist has unrecognized shape')

    def norm_consistency(self, embedding, embeddings):
        differences = embeddings - embedding
        inconsistency = np.linalg.norm(differences, axis=1)
        return inconsistency < self.norm_cutoff

    def qvalue_consistency(self, neighbor_index, pdist_combined, rb_to_embedding_combined, embedding_to_rb_combined):
        # Find adjacent nodes
        in_indices = np.array(list(self.g.predecessors(neighbor_index)))
        out_indices = np.array(list(self.g.successors(neighbor_index)))

        # Be conservative about merging in this edge case
        if len(in_indices) == 0 and len(out_indices) == 0:
            return False

        # Calculate qvalues with adjacent nodes
        if len(in_indices) != 0:
            existing_in_qvals = pdist_combined[in_indices, neighbor_index]
            new_in_qvals = rb_to_embedding_combined[in_indices]
        else:
            existing_in_qvals = np.array([])
            new_in_qvals = np.array([])
        if len(out_indices) != 0:
            existing_out_qvals = pdist_combined[neighbor_index, out_indices]
            new_out_qvals = embedding_to_rb_combined[out_indices]
        else:
            existing_out_qvals = np.array([])
            new_out_qvals = np.array([])
        existing_qvals = np.append(existing_in_qvals, existing_out_qvals)
        new_qvals = np.append(new_in_qvals, new_out_qvals)

        # Measure qvalue consistency
        qval_diffs = new_qvals - existing_qvals
        qval_inconsistency = np.linalg.norm(qval_diffs, np.inf)

        # Determine if consistent using cutoff
        return qval_inconsistency < self.consistency_cutoff

    def get_goal_in_rb(self):
        goal_index = np.random.randint(low=0, high=self.rb_vec.shape[0])
        return self.rb_vec[goal_index].copy()
