from typing import Callable

import gymnasium

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

RENDER_DIM = {
    "HumanoidStandupCurriculum": (480, 480),
}


def get_make_env(
    env_name: str,
    *,
    render_mode: str = "rgb_array",
    seed:int,
    **kwargs,
) -> Callable:
    def make_env_wrapper() -> gymnasium.Env:
        env: gymnasium.Env
        env = gymnasium.make(
            env_name,
            render_mode=render_mode,
            **kwargs,
        )

        env.reset(seed=seed)
        return Monitor(env)
        # return env
    
    set_random_seed(seed)
    return make_env_wrapper
