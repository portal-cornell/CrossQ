import pathlib
from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv as GymHumanoidEnv
from gymnasium.spaces import Box
from numpy.typing import NDArray

from envs.humanoid.reward_helpers import *

def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 3.5,
    "lookat": np.array((0.25, 0.0, 1.25)),
    "elevation": -10.0,
}

class VLMRewardedHumanoidEnv(GymHumanoidEnv):
    def __init__(
        self,
        episode_length=240,
        reward_type="remain_standing",
        render_mode: str = "rgb_array",
        forward_reward_weight: float = 1.25,
        ctrl_cost_weight: float = 0.1,
        healthy_reward: float = 5.0,
        healthy_z_range: Tuple[float] = (1.0, 2.0),
        reset_noise_scale: float = 1e-2,
        exclude_current_positions_from_observation: bool = True,
        camera_config: Optional[Dict[str, Any]] = DEFAULT_CAMERA_CONFIG,
        textured: bool = True,
        **kwargs,
    ):
        terminate_when_unhealthy = False
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            render_mode=render_mode,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(376,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(378,), dtype=np.float64
            )
        env_file_name = None
        if textured:
            env_file_name = "humanoid_textured.xml"
        else:
            env_file_name = "humanoid.xml"
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
        self.episode_length = episode_length
        self.num_steps = 0
        if camera_config:
            self.camera_id = -1

        assert reward_type in REWARD_FN_MAPPING.keys()
        self.reward_fn = REWARD_FN_MAPPING[reward_type]

    def step(self, action) -> Tuple[NDArray, float, bool, bool, Dict]:
        xy_position_before = mass_center(self.model, self.data)

        obs, reward, terminated, truncated, info = super().step(action)

        reward, info = self.reward_fn(self.data, model=self.model, 
                                        xy_position_before=xy_position_before, dt=self.dt,
                                        ctrl_cost=self.control_cost(action),
                                        healthy_reward=self.healthy_reward,
                                        forward_reward_weight=self._forward_reward_weight)

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
    model = kwargs.get("model", None)
    xy_position_before = kwargs.get("xy_position_before", None)
    dt = kwargs.get("dt", None)

    ctrl_cost = kwargs.get("ctrl_cost", None)
    healthy_reward = kwargs.get("healthy_reward", None)

    forward_reward_weight = kwargs.get("forward_reward_weight", None)

    xy_position_after = mass_center(model, data)

    xy_velocity = (xy_position_after - xy_position_before) / dt
    x_velocity, y_velocity = xy_velocity

    forward_reward = forward_reward_weight * x_velocity

    rewards = forward_reward + healthy_reward - ctrl_cost

    info = {
            "reward_linvel": f"{forward_reward:.2f}",
            "reward_quadctrl": f"{-ctrl_cost:.2f}",
            "reward_alive": f"{healthy_reward:.2f}",
            "x_position": f"{xy_position_after[0]:.2f}",
            "y_position": f"{xy_position_after[1]:.2f}",
            "distance_from_origin": f"{np.linalg.norm(xy_position_after, ord=2):.2f}",
            "x_velocity": f"{x_velocity:.2f}",
            "y_velocity": f"{y_velocity:.2f}",
            "forward_reward": f"{forward_reward:.2f}",
        }
    
    return rewards, info


def reward_remain_standing(data, **kwargs):
    original_mujoco_reward, _ = reward_original(data, **kwargs)

    ctrl_cost = kwargs.get("ctrl_cost", None)
    ctrl_cost_w = 1

    upward_cost = vert_dist_btw_torso_and_standing_height(data)
    upward_cost_w = 1

    # penalizes the horizontal distance (in the xy-plane) between
    # the center-of-mass (CoM) and the mean of the feet positions.
    torso_and_left_foot_dist_cost, torso_and_right_foot_dist_cost, com_and_left_foot_dist_cost, com_and_right_foot_dist_cost, info = hori_dist_btw_torso_and_com_circular_range_and_feet_no_smooth_abs(data)
    torso_and_feet_dist_cost_w = 1
    com_and_feet_dist_cost_w = 1

    # provide an reward for distance between the legs and bottom
    bottom_and_left_foot_dist_r, bottom_and_right_foot_dist_r, bottom_info = vert_dist_btw_bottom_and_feet(data)
    bottom_and_feet_dist_r_w = 1
    
    # penalizes the horizontal distance between
    # the torso and the CoM
    torso_and_com_dist_cost = hori_dist_btw_com_and_torso_no_smooth_abs(data)
    torso_and_com_dist_cost_w = 1

    rewards = - upward_cost - ctrl_cost \
                + bottom_and_feet_dist_r_w * (bottom_and_left_foot_dist_r + bottom_and_right_foot_dist_r) \
                - torso_and_feet_dist_cost_w * (torso_and_left_foot_dist_cost + torso_and_right_foot_dist_cost) \
                - com_and_feet_dist_cost_w * (com_and_left_foot_dist_cost + com_and_right_foot_dist_cost)

    terms_to_plot = dict(
            uph_c= - upward_cost,
            ctrl_c= -ctrl_cost,
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
            r= f"{rewards:.2f}",
            og_r= f"{original_mujoco_reward:.2f}",
    )
    
    return rewards, terms_to_plot

# reward_tassa_improved_feet_to_circle_bottom_up
# torso_and_feet_dist_cost_w = 50, com_and_feet_dist_cost_w = 25
# bottom_and_feet_dist_r_w = 50
# torso_and_com_dist_cost_w = 1
# uph_reward_w = 1
# com_vel_cost_w = 0.05
def best_standing_from_lying_down(data, **kwargs):
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

REWARD_FN_MAPPING = dict(
        original = reward_original,
        remain_standing = reward_remain_standing,
        best_standing_up = best_standing_from_lying_down,
    )