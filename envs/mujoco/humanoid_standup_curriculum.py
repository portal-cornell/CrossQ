import pathlib
from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.humanoidstandup_v4 import HumanoidStandupEnv as GymHumanoidStandupEnv
from gymnasium.spaces import Box
from numpy.typing import NDArray


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 0.8925)),
    "elevation": -20.0,
}


class HumanoidStandupCurriculum(GymHumanoidStandupEnv):
    # TODO: add init that takes in an stage indicator
    #   in this init, let's define a mapping from stage indicator to reward function

    # TODO: reward function for sit up
    def __init__(
        self, 
        episode_length=240, 
        render_mode = "rgb_array",
        **kwargs,
    ):
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(376,), dtype=np.float64
        )
        MujocoEnv.__init__(
            self,
            "humanoidstandup.xml",
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            render_mode=render_mode,
            **kwargs,
        )
        utils.EzPickle.__init__(self, **kwargs)

        self.episode_length = episode_length
        self.num_steps = 0


    def step(self, action) -> Tuple[NDArray, float, bool, bool, Dict]:
        obs, _, terminated, truncated, _ = super().step(action)

        reward, info = reward_original(self.data, timestep=self.model.opt.timestep)

        self.num_steps += 1
        terminated = self.num_steps >= self.episode_length
        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.num_steps = 0
        return super().reset(seed=seed, options=options)



def reward_original(data, **kwargs):
    timestep = kwargs.get('timestep', None)

    uph_cost = upward_reward(data, timestep)
    quad_ctrl_cost = control_cost(data)
    quad_impact_cost = impact_cost(data)

    reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1

    terms_to_plot = dict(
        pos=data.qpos[2],
        uph=uph_cost,
        ctrl=quad_ctrl_cost,
        imp=quad_impact_cost,
        r=reward
    )
    
    return reward, terms_to_plot


"""++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Helper functions

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""

SITUP_HEIGHT = 0.5  # From looking at the environment

def upward_reward(data, timestep):
    """A reward for moving upward (in an attempt to stand up)
    """
    pos_after = data.qpos[2]
    return (pos_after - 0) / timestep


def control_cost(data):
    """Penalising the humanoid if it has too large of a control force.
    """
    return 0.1 * np.square(data.ctrl).sum()


def impact_cost(data):
    """Penalising the humanoid if the external contact force is too large.
    """
    quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
    return min(quad_impact_cost, 10)

