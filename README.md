# Sparse Graphical Memory (SGM) and Search on the Replay Buffer (SoRB) in PyTorch

## Example usage
```
pip install -e .

python run_PointEnv.py configs/config_PointEnv.py
```

## Results

### SoRB (re-planning with closest waypoint) trajectory visualization
![Search comparison](./workdirs/uvfddpg_distributional1_ensemble3_rescale5/sorb_compare_search_openloop0.png)

```
policy: no search
start: [0.03271197 0.99020872]
goal: [0.81310241 0.028764  ]
steps: 300
----------
policy: search
start: [0.03271197 0.99020872]
goal: [0.81310241 0.028764  ]
steps: 127
```

### SoRB (open loop planning) trajectory visualization
![Search comparison](./workdirs/uvfddpg_distributional1_ensemble3_rescale5/sorb_compare_search_openloop1.png)

```
policy: no search
start: [0.03271197 0.99020872]
goal: [0.81310241 0.028764  ]
steps: 300
----------
policy: search
start: [0.03271197 0.99020872]
goal: [0.81310241 0.028764  ]
steps: 111
```

### State graph visualization 

1. SoRB state graph (per critic in ensemble)
![SoRB state graph](./workdirs/uvfddpg_distributional1_ensemble3_rescale5/sorb_state_graph_ensemble.png)

2. SGM state graph (ensembled)
<!-- ![SGM state graph](./workdirs/uvfddpg_distributional1_ensemble3_rescale5/sgm_state_graph.png) -->
<p align="center"><img src="./workdirs/uvfddpg_distributional1_ensemble3_rescale5/sgm_state_graph.png" width="275" alt="SGM state graph"></p>

```
Initial SparseSearchPolicy (|V|=202, |E|=1894) has success rate 0.20, evaluated in 57.72 seconds
Filtered SparseSearchPolicy (|V|=202, |E|=986) has success rate 0.80, evaluated in 34.39 seconds
Took 10000 cleanup steps in 344.87 seconds
Cleaned SparseSearchPolicy (|V|=202, |E|=955) has success rate 1.00, evaluated in 27.10 seconds
```

## Credits
* https://github.com/scottemmons/sgm
* https://github.com/google-research/google-research/tree/master/sorb
* https://github.com/sfujim/TD3

## References
[1]: Michael Laskin, Scott Emmons, Ajay Jain, Thanard Kurutach, Pieter Abbeel, Deepak Pathak, ["Sparse Graphical Memory for Robust Planning"](https://arxiv.org/abs/2003.06417), 2020.

[2]: Benjamin Eysenbach, Ruslan Salakhutdinov, Sergey Levine, ["Search on the Replay Buffer: Bridging Planning and Reinforcement Learning"](https://arxiv.org/abs/1906.05253), 2019.
