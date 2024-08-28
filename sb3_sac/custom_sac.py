from stable_baselines3 import SAC

class CustomSAC(SAC):
    """
    StableBaselines3 SAC with added data class to store previous number of timesteps and episodes (used for reward calculation)
    """
    def __init__(self, *args, **kwargs):
        super(CustomSAC, self).__init__(*args, **kwargs)

    def collect_rollouts(self, *args, **kwargs):
        rollout = super().collect_rollouts(*args, **kwargs)

        self.previous_num_timesteps = self.num_timesteps
        self.previous_num_episodes = self._episode_num

        return rollout
    
    def learn(self, *args, **kwargs):
        self.previous_num_timesteps = 0
        self.previous_num_episodes = 0

        # Call the parent learn function
        return super().learn(*args, **kwargs)