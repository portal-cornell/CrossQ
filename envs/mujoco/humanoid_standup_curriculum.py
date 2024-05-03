import pathlib
from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.humanoidstandup_v4 import HumanoidStandupEnv as GymHumanoidStandupEnv
from gymnasium.spaces import Box
from numpy.typing import NDArray
from envs.humanoid.reward_helpers_humanoid_standup import *

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 0.8925)),
    "elevation": -20.0,
}


class HumanoidStandupCurriculum(GymHumanoidStandupEnv):
    def __init__(
        self, 
        episode_length=240, 
        reward_type="original",
        render_mode = "rgb_array",
        camera_config: Optional[Dict[str, Any]] = DEFAULT_CAMERA_CONFIG,
        textured: bool = True,
        **kwargs,
    ):
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(376,), dtype=np.float64
        )
        env_file_name = None
        if textured:
            env_file_name = "humanoidstandup_textured.xml"
        else:
            env_file_name = "humanoidstandup.xml"
        model_path = str(pathlib.Path(__file__).parent / env_file_name)
        MujocoEnv.__init__(
            self,
            model_path,
            5,
            observation_space=observation_space,
            default_camera_config=camera_config,
            render_mode=render_mode,
            **kwargs,
        )
        utils.EzPickle.__init__(self, **kwargs)

        self.episode_length = episode_length
        self.num_steps = 0

        assert reward_type in REWARD_FN_MAPPING.keys()
        self.reward_fn = REWARD_FN_MAPPING[reward_type]

        if textured:
            self.camera_id = -1


    def step(self, action) -> Tuple[NDArray, float, bool, bool, Dict]:
        obs, _, terminated, truncated, _ = super().step(action)

        reward, info = self.reward_fn(self.data, timestep=self.model.opt.timestep)

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
        ctrl=f"{quad_ctrl_cost:.2f}",
        imp=f"{quad_impact_cost:.2f}",
        uph=f"{uph_reward:.2f}",
        com=str([f"{data.qpos.flat[:3][i]:.2f}" for i in range(3)]),
        og_r=f"{reward:.2f}"
    )
    
    return reward, terms_to_plot


def reward_tassa_mpc(data, **kwargs):
    """Reward described in https://ieeexplore.ieee.org/document/6386025
    """
    original_mujoco_reward, _ = reward_original(data, **kwargs)

    ## All three terms use the smooth-abs norm (Figure 2).

    # penalizes the horizontal distance (in the xy-plane) between
    # the center-of-mass (CoM) and the mean of the feet positions.
    com_and_mean_feet_dist_cost, info = hori_dist_btw_com_and_mean_feet(data)
    com_and_mean_feet_dist_cost_w = 10

    # penalizes the horizontal distance between
    # the torso and the CoM
    torso_and_com_dist_cost = hori_dist_btw_com_and_torso(data)
    torso_and_com_dist_cost_w = 1

    # penalizes the vertical
    # distance between the torso and a point 1.3m over the mean of
    # the feet.
    torso_and_standing_cost = vert_dist_btw_torso_and_standing_height(data)
    torso_and_standing_cost_w = 100

    # quadratic penalty on the horizontal COM velocity
    com_vel_cost = hori_com_vel_cost(data)
    com_vel_cost_w = 0.05

    reward = - com_and_mean_feet_dist_cost_w * com_and_mean_feet_dist_cost - torso_and_com_dist_cost_w * torso_and_com_dist_cost - torso_and_standing_cost_w * torso_and_standing_cost - com_vel_cost_w * com_vel_cost

    terms_to_plot = dict(
        ctrl=f"{com_vel_cost:.2f}",
        uph=f"{torso_and_standing_cost:.2f}",
        ctor=f"{torso_and_com_dist_cost:.2f}",
        cft=f"{com_and_mean_feet_dist_cost:.2f}",
        vel=str([f"{data.qvel.flat[:3][i]:.2f}" for i in range(3)]),
        ft=info['feet_midpt'],
        tor=str([f"{data.qpos.flat[:3][i]:.2f}" for i in range(3)]),
        com=str([f"{data.xipos[1][i]:.2f}" for i in range(3)]),
        r=f"{reward:.2f}",
        og_r=f"{original_mujoco_reward:.2f}",
    )
    
    return reward, terms_to_plot


def reward_tassa_mpc_with_upward_reward(data, **kwargs):
    """Reward described in https://ieeexplore.ieee.org/document/6386025
    """
    timestep = kwargs.get('timestep', None)
    original_mujoco_reward, _ = reward_original(data, **kwargs)

    ## All three terms use the smooth-abs norm (Figure 2).

    # penalizes the horizontal distance (in the xy-plane) between
    # the center-of-mass (CoM) and the mean of the feet positions.
    com_and_mean_feet_dist_cost, info = hori_dist_btw_com_and_mean_feet_no_smooth_abs(data)
    com_and_mean_feet_dist_cost_w = 100

    # penalizes the horizontal distance between
    # the torso and the CoM
    torso_and_com_dist_cost = hori_dist_btw_com_and_torso_no_smooth_abs(data)
    torso_and_com_dist_cost_w = 1
    
    # provide an upward reward
    uph_reward = upward_reward(data, timestep)
    uph_reward_w = 1

    # quadratic penalty on the horizontal COM velocity
    com_vel_cost = hori_com_vel_cost(data)
    com_vel_cost_w = 0.05

    reward = uph_reward_w * uph_reward - com_and_mean_feet_dist_cost_w * com_and_mean_feet_dist_cost - torso_and_com_dist_cost_w * torso_and_com_dist_cost - com_vel_cost_w * com_vel_cost + 1

    terms_to_plot = dict(
        ogctrl=f"{control_cost(data):.2f}",
        ctrl=f"{com_vel_cost:.2f}",
        uph=f"{uph_reward:.2f}",
        ctor=f"{torso_and_com_dist_cost:.2f}",
        cft=f"{com_and_mean_feet_dist_cost:.2f}",
        vel=str([f"{data.qvel.flat[:3][i]:.2f}" for i in range(3)]),
        ft=info['feet_midpt'],
        tor=str([f"{data.qpos.flat[:3][i]:.2f}" for i in range(3)]),
        com=str([f"{data.xipos[1][i]:.2f}" for i in range(3)]),
        r=f"{reward:.2f}",
        og_r=f"{original_mujoco_reward:.2f}",
    )
    
    return reward, terms_to_plot


def reward_tassa_mpc_with_upward_reward_each_feet_dist(data, **kwargs):
    """Reward described in https://ieeexplore.ieee.org/document/6386025
    """
    timestep = kwargs.get('timestep', None)
    original_mujoco_reward, _ = reward_original(data, **kwargs)

    ## All three terms use the smooth-abs norm (Figure 2).

    # penalizes the horizontal distance (in the xy-plane) between
    # the center-of-mass (CoM) and the mean of the feet positions.
    com_and_left_foot_dist_cost, com_and_right_foot_dist_cost, info = hori_dist_btw_com_and_feet_no_smooth_abs(data)
    com_and_feet_dist_cost_w = 100

    # penalizes the horizontal distance between
    # the torso and the CoM
    torso_and_com_dist_cost = hori_dist_btw_com_and_torso_no_smooth_abs(data)
    torso_and_com_dist_cost_w = 1
    
    # provide an upward reward
    uph_reward = upward_reward(data, timestep)
    uph_reward_w = 1

    # quadratic penalty on the horizontal COM velocity
    com_vel_cost = hori_com_vel_cost(data)
    com_vel_cost_w = 0.05

    reward = uph_reward_w * uph_reward \
                - com_and_feet_dist_cost_w * (com_and_left_foot_dist_cost + com_and_right_foot_dist_cost) \
                - torso_and_com_dist_cost_w * torso_and_com_dist_cost \
                - com_vel_cost_w * com_vel_cost \
                + 1

    terms_to_plot = dict(
        ogctrl=f"{control_cost(data):.2f}",
        ctrl=f"{com_vel_cost:.2f}",
        uph=f"{uph_reward:.2f}",
        ctor=f"{torso_and_com_dist_cost:.2f}",
        cLft=f"{com_and_left_foot_dist_cost:.2f}",
        cRft=f"{com_and_right_foot_dist_cost:.2f}",
        vel=str([f"{data.qvel.flat[:3][i]:.2f}" for i in range(3)]),
        Rft=info['right_foot_xy'],
        Lft=info['left_foot_xy'],
        tor=str([f"{data.qpos.flat[:3][i]:.2f}" for i in range(3)]),
        com=str([f"{data.xipos[1][i]:.2f}" for i in range(3)]),
        r=f"{reward:.2f}",
        og_r=f"{original_mujoco_reward:.2f}",
    )
    
    return reward, terms_to_plot


def reward_tassa_mpc_with_upward_reward_each_feet_dist_bottom_up(data, **kwargs):
    """Reward described in https://ieeexplore.ieee.org/document/6386025
    """
    timestep = kwargs.get('timestep', None)
    original_mujoco_reward, _ = reward_original(data, **kwargs)

    ## All three terms use the smooth-abs norm (Figure 2).

    # penalizes the horizontal distance (in the xy-plane) between
    # the center-of-mass (CoM) and the mean of the feet positions.
    com_and_left_foot_dist_cost, com_and_right_foot_dist_cost, info = hori_dist_btw_com_and_feet_no_smooth_abs(data)
    com_and_feet_dist_cost_w = 100

    # provide an reward for distance between the legs and bottom
    bottom_and_left_foot_dist_r, bottom_and_right_foot_dist_r, bottom_info = vert_dist_btw_bottom_and_feet(data)
    bottom_and_feet_dist_r_w = 50
    
    # penalizes the horizontal distance between
    # the torso and the CoM
    torso_and_com_dist_cost = hori_dist_btw_com_and_torso_no_smooth_abs(data)
    torso_and_com_dist_cost_w = 1
    
    # provide an upward reward
    uph_reward = upward_reward(data, timestep)
    uph_reward_w = 1

    # quadratic penalty on the horizontal COM velocity
    com_vel_cost = hori_com_vel_cost(data)
    com_vel_cost_w = 0.05

    reward = uph_reward_w * uph_reward \
                + bottom_and_feet_dist_r_w * (bottom_and_left_foot_dist_r + bottom_and_right_foot_dist_r) \
                - com_and_feet_dist_cost_w * (com_and_left_foot_dist_cost + com_and_right_foot_dist_cost) \
                - torso_and_com_dist_cost_w * torso_and_com_dist_cost \
                - com_vel_cost_w * com_vel_cost \
                + 1

    terms_to_plot = dict(
        ogctrl=f"{control_cost(data):.2f}",
        ctrl=f"{com_vel_cost:.2f}",
        uph=f"{uph_reward:.2f}",
        ctor=f"{torso_and_com_dist_cost:.2f}",
        cLft=f"{com_and_left_foot_dist_cost:.2f}",
        cRft=f"{com_and_right_foot_dist_cost:.2f}",
        cBLft=f"{bottom_and_left_foot_dist_r:.2f}",
        cBRft=f"{bottom_and_right_foot_dist_r:.2f}",
        vel=str([f"{data.qvel.flat[:3][i]:.2f}" for i in range(3)]),
        Rft=info['right_foot_xy'],
        Lft=info['left_foot_xy'],
        btm=bottom_info['bottom_z'],
        tor=str([f"{data.qpos.flat[:3][i]:.2f}" for i in range(3)]),
        com=str([f"{data.xipos[1][i]:.2f}" for i in range(3)]),
        r=f"{reward:.2f}",
        og_r=f"{original_mujoco_reward:.2f}",
    )
    
    return reward, terms_to_plot

def reward_tassa_mpc_with_upward_reward_each_feet_dist_to_torso_bottom_up(data, **kwargs):
    """Reward described in https://ieeexplore.ieee.org/document/6386025
    """
    timestep = kwargs.get('timestep', None)
    original_mujoco_reward, _ = reward_original(data, **kwargs)

    ## All three terms use the smooth-abs norm (Figure 2).

    # penalizes the horizontal distance (in the xy-plane) between
    # the center-of-mass (CoM) and the mean of the feet positions.
    com_and_left_foot_dist_cost, com_and_right_foot_dist_cost, info = hori_dist_btw_torso_and_feet_no_smooth_abs(data)
    com_and_feet_dist_cost_w = 50

    # provide an reward for distance between the legs and bottom
    bottom_and_left_foot_dist_r, bottom_and_right_foot_dist_r, bottom_info = vert_dist_btw_bottom_and_feet(data)
    bottom_and_feet_dist_r_w = 50
    
    # penalizes the horizontal distance between
    # the torso and the CoM
    torso_and_com_dist_cost = hori_dist_btw_com_and_torso_no_smooth_abs(data)
    torso_and_com_dist_cost_w = 1
    
    # provide an upward reward
    uph_reward = upward_reward(data, timestep)
    uph_reward_w = 1

    # quadratic penalty on the horizontal COM velocity
    com_vel_cost = hori_com_vel_cost(data)
    com_vel_cost_w = 0.05
    quad_ctrl_cost = control_cost(data)
    quad_impact_cost = impact_cost(data)

    reward = uph_reward_w * uph_reward \
                + bottom_and_feet_dist_r_w * (bottom_and_left_foot_dist_r + bottom_and_right_foot_dist_r) \
                - com_and_feet_dist_cost_w * (com_and_left_foot_dist_cost + com_and_right_foot_dist_cost) \
                - torso_and_com_dist_cost_w * torso_and_com_dist_cost \
                - com_vel_cost_w * com_vel_cost \
                - quad_ctrl_cost - quad_impact_cost \
                + 1

    terms_to_plot = dict(
        ctrl=f"{com_vel_cost:.2f}",
        uph=f"{uph_reward:.2f}",
        ctor=f"{torso_and_com_dist_cost:.2f}",
        cLft=f"{com_and_left_foot_dist_cost:.2f}",
        cRft=f"{com_and_right_foot_dist_cost:.2f}",
        cBLft=f"{bottom_and_left_foot_dist_r:.2f}",
        cBRft=f"{bottom_and_right_foot_dist_r:.2f}",
        vel=str([f"{data.qvel.flat[:3][i]:.2f}" for i in range(3)]),
        Rft=info['right_foot_xy'],
        Lft=info['left_foot_xy'],
        btm=bottom_info['bottom_z'],
        tor=str([f"{data.qpos.flat[:3][i]:.2f}" for i in range(3)]),
        com=str([f"{data.xipos[1][i]:.2f}" for i in range(3)]),
        r=f"{reward:.2f}",
        og_r=f"{original_mujoco_reward:.2f}",
    )
    
    return reward, terms_to_plot


# From here onwards, tassa mpc that use upward reward + individual feet as reward_tassa_improved
def reward_tassa_improved_feet_to_circle_bottom_up(data, **kwargs):
    """Reward described in https://ieeexplore.ieee.org/document/6386025
    """
    timestep = kwargs.get('timestep', None)
    original_mujoco_reward, _ = reward_original(data, **kwargs)

    ## All three terms use the smooth-abs norm (Figure 2).

    # penalizes the horizontal distance (in the xy-plane) between
    # the center-of-mass (CoM) and the mean of the feet positions.
    torso_and_left_foot_dist_cost, torso_and_right_foot_dist_cost, com_and_left_foot_dist_cost, com_and_right_foot_dist_cost, info = hori_dist_btw_torso_and_com_circular_range_and_feet_no_smooth_abs(data)
    torso_and_feet_dist_cost_w = 50
    com_and_feet_dist_cost_w = 25

    # provide an reward for distance between the legs and bottom
    bottom_and_left_foot_dist_r, bottom_and_right_foot_dist_r, bottom_info = vert_dist_btw_bottom_and_feet(data)
    bottom_and_feet_dist_r_w = 50
    
    # penalizes the horizontal distance between
    # the torso and the CoM
    torso_and_com_dist_cost = hori_dist_btw_com_and_torso_no_smooth_abs(data)
    torso_and_com_dist_cost_w = 1
    
    # provide an upward reward
    uph_reward = upward_reward(data, timestep)
    uph_reward_w = 1

    # quadratic penalty on the horizontal COM velocity
    com_vel_cost = hori_com_vel_cost(data)
    com_vel_cost_w = 0.05

    reward = uph_reward_w * uph_reward \
                + bottom_and_feet_dist_r_w * (bottom_and_left_foot_dist_r + bottom_and_right_foot_dist_r) \
                - torso_and_feet_dist_cost_w * (torso_and_left_foot_dist_cost + torso_and_right_foot_dist_cost) \
                - com_and_feet_dist_cost_w * (com_and_left_foot_dist_cost + com_and_right_foot_dist_cost) \
                - torso_and_com_dist_cost_w * torso_and_com_dist_cost \
                - com_vel_cost_w * com_vel_cost \
                + 1

    terms_to_plot = dict(
        ctrl=f"{com_vel_cost:.2f}",
        uph=f"{uph_reward:.2f}",
        ctor=f"{torso_and_com_dist_cost:.2f}",
        cLft=f"{torso_and_left_foot_dist_cost:.2f}, {com_and_left_foot_dist_cost:.2f}",
        cRft=f"{torso_and_right_foot_dist_cost:.2f}, {com_and_right_foot_dist_cost:.2f}",
        cBLft=f"{bottom_and_left_foot_dist_r:.2f}",
        cBRft=f"{bottom_and_right_foot_dist_r:.2f}",
        # vel=str([f"{data.qvel.flat[:3][i]:.2f}" for i in range(3)]),
        Rft=info['right_foot_xy'],
        Lft=info['left_foot_xy'],
        btm=bottom_info['bottom_z'],
        tor=str([f"{data.qpos.flat[:3][i]:.2f}" for i in range(3)]),
        com=str([f"{data.xipos[1][i]:.2f}" for i in range(3)]),
        r=f"{reward:.2f}",
        og_r=f"{original_mujoco_reward:.2f}",
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

    original_reward, _ = reward_original(data, **kwargs)

    terms_to_plot = dict(
        ctrl=f"{quad_ctrl_cost:.2f}",
        imp=f"{quad_impact_cost:.2f}",
        uph=f"{uph_reward:.2f}",
        fabv=f"{feet_above_com_cost:.2f}",
        com=str([f"{data.qpos.flat[:3][i]:.2f}" for i in range(3)]),
        ftR=str([f"{data.geom_xpos[8][i]:.2f}" for i in range(3)]),
        ftL=str([f"{data.geom_xpos[11][i]:.2f}" for i in range(3)]),
        r=f"{reward:.2f}",
        og_r=f"{original_reward:.2f}",
    )
    
    return reward, terms_to_plot


def reward_stage1_v0(data, **kwargs):
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

    original_reward, _ = reward_original(data, **kwargs)

    terms_to_plot = dict(
        ctrl=f"{quad_ctrl_cost:.2f}",
        imp=f"{quad_impact_cost:.2f}",
        uph=f"{uph_reward:.2f}",
        fabv=f"{feet_above_com_cost:.2f}",
        fdist=f"{feet_dist_cost:.2f}",
        sat=f"{is_sitting_up(data):.2f}",
        # fmid=dist_info["feet_midpt"],
        fmid_d=dist_info["feet_midpt_dist"],
        ftR_d=dist_info["right_foot_dist"],
        ftL_d=dist_info["left_foot_dist"],
        com=str([f"{data.qpos.flat[:3][i]:.2f}" for i in range(3)]),
        ftR=str([f"{data.geom_xpos[8][i]:.2f}" for i in range(3)]),
        ftL=str([f"{data.geom_xpos[11][i]:.2f}" for i in range(3)]),
        r=f"{reward:.2f}",
        og_r=f"{original_reward:.2f}",
    )
    
    return reward, terms_to_plot


def reward_stage1_v1(data, **kwargs):
    """After the agent can sit up, this reward fn encourages the agent to move the feet closer to the COM
    
    v1: remove the mid point, which could be problematic
    """
    timestep = kwargs.get('timestep', None)

    uph_reward = upward_reward(data, timestep)
    quad_ctrl_cost = control_cost(data)
    quad_impact_cost = impact_cost(data)
    feet_above_com_cost = feet_are_above_com_cost(data)

    feet_dist_cost, dist_info = dist_btw_com_and_feet_cost_v1(data, timestep)
    # This provides a positve baseline that doesn't make stage 1 look worse than stage 0
    feet_dist_offset = 50

    if not is_sitting_up(data):
        reward = uph_reward - quad_ctrl_cost - quad_impact_cost - feet_above_com_cost + 1
    else:
        reward = (uph_reward - quad_ctrl_cost - quad_impact_cost - feet_above_com_cost + 1) + (feet_dist_offset - feet_dist_cost)

    original_reward, _ = reward_original(data, **kwargs)

    terms_to_plot = dict(
        ctrl=f"{quad_ctrl_cost:.2f}",
        imp=f"{quad_impact_cost:.2f}",
        uph=f"{uph_reward:.2f}",
        fabv=f"{feet_above_com_cost:.2f}",
        fdist=f"{feet_dist_cost:.2f}",
        sat=f"{is_sitting_up(data):.2f}",
        # fmid=dist_info["feet_midpt"],
        # fmid_d=dist_info["feet_midpt_dist"],
        ftR_d=dist_info["right_foot_dist"],
        ftL_d=dist_info["left_foot_dist"],
        com=str([f"{data.qpos.flat[:3][i]:.2f}" for i in range(3)]),
        ftR=str([f"{data.geom_xpos[8][i]:.2f}" for i in range(3)]),
        ftL=str([f"{data.geom_xpos[11][i]:.2f}" for i in range(3)]),
        r=f"{reward:.2f}",
        og_r=f"{original_reward:.2f}",
    )
    
    return reward, terms_to_plot


# v2: prevent cross the legs

# stage 3: force the feet to touch the ground?
    

REWARD_FN_MAPPING = dict(
        original = reward_original,
        tassa_mpc = reward_tassa_mpc,
        tassa_mpc_uph = reward_tassa_mpc_with_upward_reward,
        tassa_mpc_each_feet = reward_tassa_mpc_with_upward_reward_each_feet_dist,
        tassa_mpc_bottom_up = reward_tassa_mpc_with_upward_reward_each_feet_dist_bottom_up,
        tassa_mpc_torso_bottom_up = reward_tassa_mpc_with_upward_reward_each_feet_dist_to_torso_bottom_up,
        tassa_imp_circle_dist = reward_tassa_improved_feet_to_circle_bottom_up,
        stage0 = reward_stage0,
        stage1_v0 = reward_stage1_v0,
        stage1_v1 = reward_stage1_v1,
    )