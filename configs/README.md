A quick guide on the hydra config for this project.

# Sub Configs
## env
When you define a new config for a new env, you must have
- name: str. The name of the environment (matching gym's name).
- reward_type: str. It should specify what reward function to use. The default can be "original".
- episode_length: int. The maximum number of steps in an episode.
- render_dim: [int, int]. The dimensions of the render window.

## reward_model
When you define a new config for a new reward model, you must have
- name: str. The name of the reward model.

## rl_algo
When you define a new config for a new RL algorithm, you must have
- name: str. The name of the RL algorithm.

