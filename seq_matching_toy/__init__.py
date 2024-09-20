from gymnasium.envs.registration import register

register(
     id="GridNav",
     entry_point="seq_matching_toy.toy_envs.grid_nav:GridNavigationEnv"
)