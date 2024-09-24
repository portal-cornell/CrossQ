from stable_baselines3 import SAC

from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy

from vlm_reward.vlm_buffer import GeomXposReplayBuffer

class CustomSAC(SAC):
    """
    StableBaselines3 SAC with added data class to store previous number of timesteps and episodes (used for reward calculation)
    """
    def __init__(self, 
                    policy: Union[str, Type[SACPolicy]],
                    env: Union[GymEnv, str],
                    learning_rate: Union[float, Schedule] = 3e-4,
                    buffer_size: int = 1_000_000,  # 1e6
                    learning_starts: int = 100,
                    batch_size: int = 256,
                    tau: float = 0.005,
                    gamma: float = 0.99,
                    train_freq: Union[int, Tuple[int, str]] = 1,
                    gradient_steps: int = 1,
                    action_noise: Optional[ActionNoise] = None,
                    replay_buffer_class: Optional[Type[ReplayBuffer]] = GeomXposReplayBuffer,  # A hack to use the custom replay buffer to log geom-xpos
                    replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
                    optimize_memory_usage: bool = False,
                    ent_coef: Union[str, float] = "auto",
                    target_update_interval: int = 1,
                    target_entropy: Union[str, float] = "auto",
                    use_sde: bool = False,
                    sde_sample_freq: int = -1,
                    use_sde_at_warmup: bool = False,
                    stats_window_size: int = 100,
                    tensorboard_log: Optional[str] = None,
                    policy_kwargs: Optional[Dict[str, Any]] = None,
                    verbose: int = 0,
                    seed: Optional[int] = None,
                    device: Union[th.device, str] = "auto",
                    _init_setup_model: bool = True,
                    **kwargs):
        super(CustomSAC, self).__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model
        )

    def collect_rollouts(self, *args, **kwargs):
        rollout = super().collect_rollouts(*args, **kwargs)

        self.previous_num_timesteps = self.num_timesteps
        self.previous_num_episodes = self._episode_num

        self.replay_buffer.clear_geom_xpos()

        return rollout
    
    def learn(self, *args, **kwargs):
        self.previous_num_timesteps = 0
        self.previous_num_episodes = 0

        # Call the parent learn function
        return super().learn(*args, **kwargs)