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

        reward, info = reward_stage0(self.data, timestep=self.model.opt.timestep)

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

    uph_reward = upward_reward(data, timestep)
    quad_ctrl_cost = control_cost(data)
    quad_impact_cost = impact_cost(data)

    reward = uph_reward - quad_ctrl_cost - quad_impact_cost + 1

    terms_to_plot = dict(
        pos=f"{data.qpos[2]:.2f}",
        uph=f"{uph_reward:.2f}",
        ctrl=f"{quad_ctrl_cost:.2f}",
        imp=f"{quad_impact_cost:.2f}",
        com=str([f"{data.qpos.flat[:3][i]:.2f}" for i in range(3)]),
        r=f"{reward:.2f}",
    )
    
    return reward, terms_to_plot


def reward_stage0(data, **kwargs):
    """Original reward with cost to prevent feet from going up
    """
    timestep = kwargs.get('timestep', None)

    uph_reward = upward_reward(data, timestep)
    quad_ctrl_cost = control_cost(data)
    quad_impact_cost = impact_cost(data)
    feet_above_com_cost = feet_are_above_com_cost(data)

    reward = uph_reward - quad_ctrl_cost - quad_impact_cost - feet_above_com_cost + 1

    terms_to_plot = dict(
        pos=f"{data.qpos[2]:.2f}",
        uph=f"{uph_reward:.2f}",
        ctrl=f"{quad_ctrl_cost:.2f}",
        imp=f"{quad_impact_cost:.2f}",
        fabv=f"{feet_above_com_cost:.2f}",
        com=str([f"{data.qpos.flat[:3][i]:.2f}" for i in range(3)]),
        ftR=str([f"{data.geom_xpos[8][i]:.2f}" for i in range(3)]),
        ftL=str([f"{data.geom_xpos[11][i]:.2f}" for i in range(3)]),
        r=f"{reward:.2f}",
    )
    
    return reward, terms_to_plot


def reward_stage1(data, **kwargs):
    """After the agent can sit up, this reward fn encourages the agent to move the feet closer to the COM
    """
    timestep = kwargs.get('timestep', None)

    uph_reward = upward_reward(data, timestep)
    quad_ctrl_cost = control_cost(data)
    quad_impact_cost = impact_cost(data)
    feet_above_com_cost = feet_are_above_com_cost(data)

    feet_dist_cost, dist_info = dist_btw_com_and_feet_cost(data, timestep)
    # This provides a positve baseline that doesn't make stage 1 look worse than stage 0
    feet_dist_offset = 50

    if not is_sitting_up(data):
        reward = uph_reward - quad_ctrl_cost - quad_impact_cost - feet_above_com_cost + 1
    else:
        reward = (uph_reward - quad_ctrl_cost - quad_impact_cost - feet_above_com_cost + 1) + (feet_dist_offset - feet_dist_cost)

    terms_to_plot = dict(
        pos=f"{data.qpos[2]:.2f}",
        uph=f"{uph_reward:.2f}",
        ctrl=f"{quad_ctrl_cost:.2f}",
        imp=f"{quad_impact_cost:.2f}",
        fabv=f"{feet_above_com_cost:.2f}",
        fdist=f"{feet_dist_cost:.2f}",
        sat=f"{is_sitting_up(data):.2f}",
        fmid=dist_info["feet_midpt"],
        dist=dist_info["dist"],
        com=str([f"{data.qpos.flat[:3][i]:.2f}" for i in range(3)]),
        ftR=str([f"{data.geom_xpos[8][i]:.2f}" for i in range(3)]),
        ftL=str([f"{data.geom_xpos[11][i]:.2f}" for i in range(3)]),
        r=f"{reward:.2f}",
    )
    
    return reward, terms_to_plot


"""++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Helper functions

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""

SITUP_HEIGHT = 0.5  # From looking at the environment
SITUP_TOLERANCE = 0.1

def upward_reward(data, timestep):
    """A reward for moving upward (in an attempt to stand up)
    """
    com_z = data.qpos[2]
    return (com_z - 0) / timestep


def control_cost(data):
    """Penalizing the humanoid if it has too large of a control force.
    """
    return 0.1 * np.square(data.ctrl).sum()


def impact_cost(data):
    """Penalizing the humanoid if the external contact force is too large.
    """
    quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
    return min(quad_impact_cost, 10)


def feet_are_above_com_cost(data):
    """Penalizing the cost of putting feet above com
    """
    com_z = data.qpos[2]
    left_foot_z = data.geom_xpos[11][2]
    right_foot_z = data.geom_xpos[8][2]

    # When the humanoid is resting, upward reward is still about 40
    #   It will be suboptimal to just keep my feet up in the air (max=0.75) while my com (0.1) is on the ground
    left_foot_cost = 100 * (left_foot_z > com_z) * (left_foot_z - com_z)
    right_foot_cost = 100 * (right_foot_z > com_z) * (right_foot_z - com_z)
    # We want the maximum value to be bad, but not too bad (TODO: tune these values)
    return min(left_foot_cost + right_foot_cost, 100)


def is_sitting_up(data):
    com_z = data.qpos[2]
    return np.abs(com_z - SITUP_HEIGHT) < SITUP_TOLERANCE

def dist_btw_com_and_feet_cost(data, timestep):
    """Penalising the cost between the distance between the center of mass and feet
    """
    com_xy = data.qpos[:2]
    left_foot_xy = data.geom_xpos[11][:2]
    right_foot_xy = data.geom_xpos[8][:2]

    feet_midpt = np.array([(left_foot_xy[0]-right_foot_xy[0])/2.0, (left_foot_xy[1]-right_foot_xy[1])/2.0])

    dist = np.sqrt(np.sum((com_xy - feet_midpt)**2))

    terms_to_plot = dict(
        feet_midpt = str([f"{feet_midpt[i]:.2f}" for i in range(len(feet_midpt))]),
        dist = f"{dist:.2f}"
    )

    # TODO: Remove debugging terms to plot
    # return dist / timestep, terms_to_plot
    scale = 40
    return min(dist * scale, scale), terms_to_plot