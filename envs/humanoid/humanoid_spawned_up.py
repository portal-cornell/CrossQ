import pathlib
from typing import Any, Dict, Optional, Tuple
from loguru import logger
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
    "azimuth": 180
}

class HumanoidEnvCustom(GymHumanoidEnv):
    DEMOS_DICT = {
        "both_arms_out_goal_only_euclidean": ["create_demo/demos/both-arms-out_joint-state.npy"]
    }

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

        self._ref_joint_states = np.array([])

        if "_goal_only_" in reward_type or "_seq_" in reward_type:
            self._load_reference_joint_states(self.DEMOS_DICT[reward_type])

        # Spawned the humanoid not so high
        self.init_qpos[2] = 1.3

    def step(self, action) -> Tuple[NDArray, float, bool, bool, Dict]:
        xy_position_before = mass_center(self.model, self.data)

        obs, reward, terminated, truncated, info = super().step(action)

        reward, info = self.reward_fn(self.data, model=self.model, 
                                        dt=self.dt,
                                        timestep=self.model.opt.timestep,
                                        xy_position_before=xy_position_before,
                                        ctrl_cost=self.control_cost(action),
                                        healthy_reward=self.healthy_reward,
                                        forward_reward_weight=self._forward_reward_weight,
                                        ref_joint_states=self._ref_joint_states)

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

    def get_obs(self):
        return self._get_obs()

    def _load_reference_joint_states(self, joint_state_fp_list):
        """
        Parameters:
            joint_state_fp_list (list): list of path to the saved reference joint state

        Effects:
            self._ref_joint_states gets updated
        """
        ref_joint_states_list = []
        for fp in joint_state_fp_list:
            ref_joint_states_list.append(np.load(fp))
        self._ref_joint_states = np.stack(ref_joint_states_list)

        # logger.debug(f"Updated self._ref_joint_states: {self._ref_joint_states.shape}\n{self._ref_joint_states}")

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

def reward_simple_remain_standing(data, **kwargs):
    original_mujoco_reward, _ = reward_original(data, **kwargs)

    ctrl_cost = kwargs.get("ctrl_cost", None)
    ctrl_cost_w = 1

    upward_cost = vert_dist_btw_torso_and_standing_height(data)
    upward_cost_w = 1

    rewards = - upward_cost - ctrl_cost

    terms_to_plot = dict(
            uph_c= - upward_cost,
            ctrl_c= - ctrl_cost,
            tor=str([f"{data.qpos.flat[:3][i]:.2f}" for i in range(3)]),
            com=str([f"{data.xipos[1][i]:.2f}" for i in range(3)]),
            r= f"{rewards:.2f}",
            og_r= f"{original_mujoco_reward:.2f}",
    )
    
    return rewards, terms_to_plot

def reward_simple_remain_standing_exp_dist(data, **kwargs):
    original_mujoco_reward, _ = reward_original(data, **kwargs)

    ctrl_cost = kwargs.get("ctrl_cost", None)
    ctrl_cost_w = 1

    upward_reward = np.exp(-(data.qpos.flat[2] - 1.3)**2)
    upward_reward_w = 1

    rewards = upward_reward_w * upward_reward - ctrl_cost_w * ctrl_cost

    terms_to_plot = dict(
            uph_r= upward_reward,
            ctrl_c= ctrl_cost,
            tor=str([f"{data.qpos.flat[:3][i]:.2f}" for i in range(3)]),
            com=str([f"{data.xipos[1][i]:.2f}" for i in range(3)]),
            r= f"{rewards:.2f}",
            og_r= f"{original_mujoco_reward:.2f}",
    )
    
    return rewards, terms_to_plot

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


def reward_kneeling(data, **kwargs):
    """
    """
    num_steps = kwargs.get('num_steps', 0)
    
    # terms_to_plot = dict(
    #     n=num_steps,
    #     com=str([f"{data.xipos[1][i]:.2f}" for i in range(3)])
    # )
    
    # naming = {0: "floor", 1:"torso", 2:"head", 3:"uwaist", 4:"lwaist", 5:"bottom",
    #             6:"R_thigh", 7:"R_shin", 8:"R_foot",
    #             9:"L_thigh", 10:"L_shin", 11:"L_foot",
    #             12:"R_uarm", 13:"R_larm", 14:"R_hand",
    #             15:"left_uarm", 16:"L_arm", 17:"L_hand"}
    # for i in range(1, len(data.geom_xpos)):
    #     terms_to_plot[naming[i]] = str([f"{data.geom_xpos[i][j]:.2f}" for j in range(3)])

    
    # Shin and foot are on the floor
    l_shin = data.geom_xpos[10]
    r_shin = data.geom_xpos[7]
    l_foot = data.geom_xpos[11]
    r_foot = data.geom_xpos[8]
    bottom = data.geom_xpos[5]
    torso = data.geom_xpos[1]
    head = data.geom_xpos[2]

    good_shin_foot_height = 0.10  # Needs to be below this
    shin_height_r = (good_shin_foot_height - l_shin[2]) + (good_shin_foot_height - r_shin[2])
    foot_height_r = (good_shin_foot_height - l_foot[2]) + (good_shin_foot_height - r_foot[2])

    good_torso_height = 0.6
    good_bottom_height = 0.3

    dist_with_tolerance_list = [
        (np.abs(torso[2] - good_torso_height), 0.1),  # Torso's height needs to be at around the right range
        (np.abs(bottom[2] - good_bottom_height), 0.1),  # Bottom's height needs to be at around the right range
        (np.linalg.norm(torso[:2] - bottom[:2]), 0.2),  # Torse and bottom need to be close to each other in (x,y) questions
        (np.linalg.norm(head[:2] - torso[:2]), 0.1),  # Torse and head need to be close to each other in (x,y) questions
    ]

    dist_with_tolerance_cost_list = dist_cost_with_tol(dist_with_tolerance_list)

    ctrl_cost = kwargs.get("ctrl_cost", None)
    ctrl_cost_w = 1

    reward = shin_height_r + foot_height_r - np.sum(dist_with_tolerance_cost_list) - ctrl_cost * ctrl_cost_w

    terms_to_plot = dict(
        steps=num_steps,
    )

    for i in [7, 10, 8, 11, 5, 1, 2]:
        terms_to_plot[GEOM_XPOS_NAMING[i]] = str([f"{data.geom_xpos[i][j]:.2f}" for j in range(3)])

    terms_to_plot["shin_r"] = f"{shin_height_r:.2f}"
    terms_to_plot["foot_r"] = f"{foot_height_r:.2f}"

    for i, name in enumerate(["torso_r", "btm_r", "tor-btm", "head-tor"]):
        terms_to_plot[name] = f"{dist_with_tolerance_list[i][0]:.2f}"

    terms_to_plot["r"] = f"{reward:.2f}"

    # if num_steps == 53:
    #     with open("./debugging/geom_xpos.npy", "wb") as fout:
    #         np.save(fout, data.geom_xpos)

    #     with open("./debugging/qpos.npy", "wb") as fout:
    #         np.save(fout, data.qpos)

    return reward, terms_to_plot


def reward_splitting(data, **kwargs):
    num_steps = kwargs.get('num_steps', 0)

    # L ft: 1.13, 0.10
    # R ft: 1.13 -0.09
    l_foot = data.geom_xpos[11]
    r_foot = data.geom_xpos[8]
    bottom = data.geom_xpos[5]
    torso = data.geom_xpos[1]
    head = data.geom_xpos[2]

    good_torso_height = 0.5
    good_bottom_height = 0.15
    good_feet_dist = 1.5

    dist_with_tolerance_list = [
        (np.abs(torso[2] - good_torso_height), 0.1, 1),  # Torso's height needs to be at around the right range
        (np.abs(bottom[2] - good_bottom_height), 0.1, 1),  # Bottom's height needs to be at around the right range
        (np.linalg.norm(torso[:2] - bottom[:2]), 0.2, 1),  # Torse and bottom need to be close to each other in (x,y) questions
        (np.linalg.norm(head[:2] - torso[:2]), 0.1, 1),  # Torse and head need to be close to each other in (x,y) questions
        (np.abs(np.linalg.norm(l_foot[:2] - r_foot[:2]) - good_feet_dist), 0.2, 5)  # Two feet need to be far apart
    ]

    dist_with_tolerance_cost_list = dist_cost_with_tol(dist_with_tolerance_list)

    ctrl_cost = kwargs.get("ctrl_cost", None)
    ctrl_cost_w = 1

    reward = - np.sum(dist_with_tolerance_cost_list) - ctrl_cost * ctrl_cost_w

    terms_to_plot = dict(
        steps=num_steps,
        ctrl_c=ctrl_cost
    )

    for i in [8, 11, 5, 1, 2]:
        terms_to_plot[GEOM_XPOS_NAMING[i]] = str([f"{data.geom_xpos[i][j]:.2f}" for j in range(3)])

    for i, name in enumerate(["torso_r", "btm_r", "tor-btm", "head-tor", "feet-apart-r"]):
        terms_to_plot[name] = f"{dist_with_tolerance_list[i][0]:.2f}"

    terms_to_plot["r"] = f"{reward:.2f}"

    return reward, terms_to_plot


def reward_both_arms_out_goal_only_euclidean(data, **kwargs):
    original_mujoco_reward, _ = reward_original(data, **kwargs)

    ctrl_cost = kwargs.get("ctrl_cost", None)
    ctrl_cost_w = 1

    upward_reward = np.exp(-(data.qpos.flat[2] - 1.3)**2)
    upward_reward_w = 0

    assert "ref_joint_states" in kwargs, "ref_joint_states must be passed in as part of the kwargs"
    ref_joint_states = kwargs.get('ref_joint_states', None)
    num_steps = kwargs.get('num_steps', 0)

    assert ref_joint_states.shape[0] == 1, "there should only be the goal image/joint position"

    # Mimicking how they get the observation
    # https://github.com/Farama-Foundation/Gymnasium/blob/b6046caeb30c9938789aeeec183147c7ffd1983b/gymnasium/envs/mujoco/humanoid_v4.py#L119
    curr_qpos = data.qpos.flat.copy()[2:]

    pose_matching_reward = np.exp(-np.linalg.norm(curr_qpos - ref_joint_states[0]))
    pose_matching_reward_w = 1
    
    reward = upward_reward_w * upward_reward + pose_matching_reward_w * pose_matching_reward - ctrl_cost_w * ctrl_cost
    
    terms_to_plot = dict(
        tor=str([f"{data.qpos.flat[:3][i]:.2f}" for i in range(3)]),
        com=str([f"{data.xipos[1][i]:.2f}" for i in range(3)]),
        l2_norm=f"{np.linalg.norm(curr_qpos - ref_joint_states[0]):.2f}",
        uph_r= f"{upward_reward:.2f}",
        ctrl_c= f"{ctrl_cost:.2f}",
        pose_r = f"{pose_matching_reward:.2f}",
        r = f"{reward:.2f}",
        og_r= f"{original_mujoco_reward:.2f}",
        steps=num_steps,
    )

    return reward, terms_to_plot


REWARD_FN_MAPPING = dict(
        original = reward_original,
        simple_remain_standing = reward_simple_remain_standing,
        simple_remain_standing_exp_dist = reward_simple_remain_standing_exp_dist,
        remain_standing = reward_remain_standing,
        best_standing_up = best_standing_from_lying_down,
        kneeling = reward_kneeling,
        splitting = reward_splitting,
        both_arms_out_goal_only_euclidean = reward_both_arms_out_goal_only_euclidean
    )
    