import pathlib
from typing import Any, Dict, Optional, Tuple
from loguru import logger
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv as GymHumanoidEnv
from gymnasium.spaces import Box
from numpy.typing import NDArray
import copy

from seq_reward.seq_utils import load_reference_seq
from seq_reward.cost_fns import euclidean_distance_advanced

from envs.humanoid.reward_helpers import *

from constants import TASK_SEQ_DICT

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
    def __init__(
        self,
        episode_length=240,
        reward_type="remain_standing",
        task_name=None,
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
        self.stage = 0  # For the stage detector reward function
        if camera_config:
            self.camera_id = -1

        assert reward_type in REWARD_FN_MAPPING.keys()
        self.reward_fn = REWARD_FN_MAPPING[reward_type]

        self._use_geom_xpos = "geom_xpos" in reward_type

        self._ref_joint_states = np.array([])

        # Determine the references to use
        #   Either to calculate the reward (when the reward_type is goal_only_euclidean_geom_xpos)
        #   Or to evaluate how well the agent is doing wrt the references (when the reward_type is basic_r_geom_xpos)
        if task_name in list(TASK_SEQ_DICT.keys()) and "key_frames" in list(TASK_SEQ_DICT[task_name]["sequences"].keys()):
            # We can do this because both the goal_only_euclidean_geom_xpos and basic_r_geom_xpos use the same reference sequence
            ref_seq_to_load = "key_frames"
            
            self._ref_joint_states = load_reference_seq(task_name, ref_seq_to_load, use_geom_xpos=self._use_geom_xpos)

            logger.info(f"[Env] Loaded reference sequence for {reward_type}. task_name={task_name}. seq_name={ref_seq_to_load}. shape={self._ref_joint_states.shape}")
        else:
            logger.info(f"[Env] Warning: {task_name} is not in TASK_SEQ_DICT.")

        # Spawned the humanoid not so high
        self.init_qpos[2] = 1.3

    def step(self, action) -> Tuple[NDArray, float, bool, bool, Dict]:
        xy_position_before = mass_center(self.model, self.data)

        obs, reward, terminated, truncated, info = super().step(action)

        reward, info = self.reward_fn(self.data, model=self.model, 
                                        dt=self.dt,
                                        num_steps=self.num_steps,
                                        curr_stage=self.stage,
                                        timestep=self.model.opt.timestep,
                                        xy_position_before=xy_position_before,
                                        ctrl_cost=self.control_cost(action),
                                        healthy_reward=self.healthy_reward,
                                        forward_reward_weight=self._forward_reward_weight,
                                        ref_joint_states=self._ref_joint_states)
        
        # Allows us to access geom_xpos during evaluation
        #   Because we store the geom_xpos after the step, we don't need to do any post-processing
        info["geom_xpos"] = copy.deepcopy(self.data.geom_xpos) 

        self.num_steps += 1
        self.stage = int(info.get("stage", 0))
        terminated = self.num_steps >= self.episode_length

        if terminated:
            # TODO: We need to verify that this is actually render the last array 
            #   (Should be slighltly different from the last frame you can get from ReplayBuffer's render_arrays)
            #   (Should not be the initial frame)
            info["last_render_array"] = self.render()

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.num_steps = 0
        self.stage = 0
        return super().reset(seed=seed, options=options)
    
    def get_obs(self):
        return self._get_obs()


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

def reward_goal_only_euclidean_geom_xpos(data, **kwargs):
    """Only use the goal joint states to calculate the reward
    - The reward is based on the euclidean distance between the current joint states and the reference joint states

    Final goal: Both arms out

    This task is a goal-reaching task (i.e. doesn't matter how you get to the goal, as long as you get to the goal)
    """
    basic_standing_reward, terms_to_plot = basic_remain_standing_rewards(data, 
                                                            upward_reward_w=1, 
                                                            ctrl_cost_w=1, 
                                                            **kwargs)

    # Calculate the reward based on the euclidean distance between the current joint states and the goal joint states
    #   Assume ref_joint_states are already normalized when they are loaded
    assert "ref_joint_states" in kwargs, "ref_joint_states must be passed in as part of the kwargs"
    ref_joint_states = kwargs.get('ref_joint_states', None)
    
    assert ref_joint_states.shape[0] == 1, f"there should only be the goal image/joint position, but got shape: {ref_joint_states.shape}"

    # Mimicking how they get the observation
    # https://github.com/Farama-Foundation/Gymnasium/blob/b6046caeb30c9938789aeeec183147c7ffd1983b/gymnasium/envs/mujoco/humanoid_v4.py#L119
    # Ignore the 1st joint state, which is the floor
    curr_geom_xpos = copy.deepcopy(data.geom_xpos)  # (n, 3)
    # Normalize the current pose by the torso's position (which is at index 1)
    curr_geom_xpos = curr_geom_xpos - curr_geom_xpos[1]

    # Only the arms are relevant
    curr_geom_xpos_relevant = curr_geom_xpos[12:, :]
    ref_joint_states_relevant = ref_joint_states[0, 12:, :]

    pose_matching_reward = np.exp(-np.linalg.norm(curr_geom_xpos_relevant - ref_joint_states_relevant))
    # pose_matching_reward = np.exp(-np.linalg.norm(curr_geom_xpos - ref_joint_states[0]))
    pose_matching_reward_w = 1

    # if kwargs.get("num_steps", 0) == 0 and pose_matching_reward_w != 1:
    #     print(f"pose_matching_reward_w: {pose_matching_reward_w:.2f}")
    
    reward = basic_standing_reward + pose_matching_reward_w * pose_matching_reward

    terms_to_plot["pose_r"] = f"{pose_matching_reward:.2f}"
    terms_to_plot["r"] = f"{reward:.2f}"
    terms_to_plot["steps"] = kwargs.get("num_steps", 0)

    return reward, terms_to_plot


def reward_seq_euclidean(data, **kwargs):
    """Use a sequence of reference joint states to calculate the reward

    This task is a goal-reaching task (i.e. doesn't matter how you get to the goal, as long as you get to the goal)
    """
    basic_standing_reward, terms_to_plot = basic_remain_standing_rewards(data, 
                                                            upward_reward_w=1, 
                                                            ctrl_cost_w=1, 
                                                            **kwargs)

    # Calculate the reward based on the euclidean distance between the current joint states and a sequence of referenece joint states
    assert "ref_joint_states" in kwargs, "ref_joint_states must be passed in as part of the kwargs"
    ref_joint_states = kwargs.get('ref_joint_states', None)

    num_ref_joint_states = ref_joint_states.shape[0]
    # Assumption, the reference joint states are in order
    ref_joint_states_weights = np.array([2**(-x) for x in range(num_ref_joint_states)])[::-1]

    curr_qpos = data.qpos.flat.copy()[2:]

    unweighted_reward_for_each_ref = np.exp(-np.linalg.norm(curr_qpos - ref_joint_states, axis=1))
    pose_matching_reward = np.sum(ref_joint_states_weights * unweighted_reward_for_each_ref)

    reward = pose_matching_reward + basic_standing_reward

    terms_to_plot["pose_r_l"] = str([f"{unweighted_reward_for_each_ref[i]:.2f}" for i in range(num_ref_joint_states)])
    terms_to_plot["pose_r"] = f"{pose_matching_reward:.2f}"
    terms_to_plot["r"] = f"{reward:.2f}"
    terms_to_plot["steps"] = kwargs.get("num_steps", 0)
    
    return reward, terms_to_plot


def reward_seq_stage_detector(data, **kwargs):
    """Use a sequence of reference joint states to calculate the reward

    This task is a goal-reaching task (i.e. doesn't matter how you get to the goal, as long as you get to the goal)
    """
    original_mujoco_reward, _ = reward_original(data, **kwargs)

    basic_standing_reward, terms_to_plot = basic_remain_standing_rewards(data, 
                                                            upward_reward_w=1, 
                                                            ctrl_cost_w=1, 
                                                            **kwargs)

    # Calculate the reward based on the euclidean distance between the current joint states and a sequence of referenece joint states
    assert "ref_joint_states" in kwargs, "ref_joint_states must be passed in as part of the kwargs"
    ref_joint_states = kwargs.get('ref_joint_states', None)

    num_ref_joint_states = ref_joint_states.shape[0]
    # Assumption, the reference joint states are in order
    # ref_joint_states_weights = np.array([2**(-x) for x in range(num_ref_joint_states)])[::-1]
    ref_joint_states_weights = np.array([1 for _ in range(num_ref_joint_states)])[::-1]  # Trying uniform weights

    curr_stage = int(kwargs.get('curr_stage', 0))

    curr_qpos = data.qpos.flat.copy()[2:]

    unweighted_reward_for_each_ref = np.exp(-np.linalg.norm(curr_qpos - ref_joint_states, axis=1))
    curr_stage_unweighted_reward  = unweighted_reward_for_each_ref[curr_stage]
    
    # If the reward for the current stage is large enough, move to the next stage
    if curr_stage_unweighted_reward > 0.4:
        new_stage = min(curr_stage + 1, num_ref_joint_states - 1)
    else:
        new_stage = curr_stage

    if curr_stage > 0:
        # Add a base reward for each stage that has been completed
        stage_reward = np.sum(ref_joint_states_weights[:curr_stage])
    else:
        stage_reward = 0

    pose_matching_reward = stage_reward + curr_stage_unweighted_reward * ref_joint_states_weights[curr_stage]

    reward = pose_matching_reward + basic_standing_reward

    terms_to_plot["pose_r_l"] = str([f"{unweighted_reward_for_each_ref[i]:.2f}" for i in range(num_ref_joint_states)])
    terms_to_plot["pose_r"] = f"{pose_matching_reward:.2f}"
    terms_to_plot["stage"] = new_stage
    terms_to_plot["r"] = f"{reward:.2f}"
    terms_to_plot["og_r"] = f"{original_mujoco_reward:.2f}"
    terms_to_plot["steps"] = kwargs.get("num_steps", 0)
    
    return reward, terms_to_plot


def reward_seq_avg(data, **kwargs):
    """Use a sequence of reference joint states to calculate the reward

    This task is a goal-reaching task (i.e. doesn't matter how you get to the goal, as long as you get to the goal)
    """
    original_mujoco_reward, _ = reward_original(data, **kwargs)

    basic_standing_reward, terms_to_plot = basic_remain_standing_rewards(data, 
                                                            upward_reward_w=1, 
                                                            ctrl_cost_w=1, 
                                                            **kwargs)

    # Calculate the reward based on the euclidean distance between the current joint states and a sequence of referenece joint states
    assert "ref_joint_states" in kwargs, "ref_joint_states must be passed in as part of the kwargs"
    ref_joint_states = kwargs.get('ref_joint_states', None)

    num_ref_joint_states = ref_joint_states.shape[0]

    curr_qpos = data.qpos.flat.copy()[2:]

    unweighted_reward_for_each_ref = np.exp(-np.linalg.norm(curr_qpos - ref_joint_states, axis=1))
    # Average of the reward from each reference joint state
    pose_matching_reward = np.mean(unweighted_reward_for_each_ref)

    reward = pose_matching_reward + basic_standing_reward

    terms_to_plot["pose_r_l"] = str([f"{unweighted_reward_for_each_ref[i]:.2f}" for i in range(num_ref_joint_states)])
    terms_to_plot["pose_r"] = f"{pose_matching_reward:.2f}"
    terms_to_plot["r"] = f"{reward:.2f}"
    terms_to_plot["og_r"] = f"{original_mujoco_reward:.2f}"
    terms_to_plot["steps"] = kwargs.get("num_steps", 0)
    
    return reward, terms_to_plot

def remain_standing_reward(data, **kwargs):
    """Only provide basic reward to remain standing and control cost
    """
    basic_standing_reward, terms_to_plot = basic_remain_standing_rewards(data, 
                                                            upward_reward_w=1, 
                                                            ctrl_cost_w=1, 
                                                            **kwargs)
    
    reward = basic_standing_reward

    return reward, terms_to_plot


def reward_only_basic_r_geom_xpos(data, **kwargs):
    """Only provide basic reward to remain standing and control cost

    However, this also plot the pose matching reward based on the ground-truth reference joint state
        Note: this pose matching reward is not used in the final reward calculation
    """
    basic_standing_reward, terms_to_plot = basic_remain_standing_rewards(data, 
                                                            upward_reward_w=1, 
                                                            ctrl_cost_w=1, 
                                                            **kwargs)
    
    reward = basic_standing_reward

    # Still calculating the pose matching reward (to individual poses) to show in the terms_to_plot
    assert "ref_joint_states" in kwargs, "ref_joint_states must be passed in as part of the kwargs"
    # Assume ref_joint_states are already normalized when they are loaded
    ref_joint_states = kwargs.get('ref_joint_states', None)

    num_ref_joint_states = ref_joint_states.shape[0]

    curr_geom_xpos = copy.deepcopy(data.geom_xpos)  # (n, 3)
    # Normalize the current pose by the torso's position (which is at index 1)
    curr_geom_xpos = curr_geom_xpos - curr_geom_xpos[1]
    curr_geom_xpos = curr_geom_xpos.reshape(1, *curr_geom_xpos.shape)

    unweighted_reward_for_each_ref = np.exp(-euclidean_distance_advanced(curr_geom_xpos, ref_joint_states))

    terms_to_plot["pose_r_l"] = str([f"{unweighted_reward_for_each_ref[0][i]:.2f}" for i in range(num_ref_joint_states)])
    terms_to_plot["r"] = f"{reward:.2f}"
    terms_to_plot["steps"] = kwargs.get("num_steps", 0)
    
    return reward, terms_to_plot


REWARD_FN_MAPPING = dict(
        # Hand engineered reward functions
        original = reward_original,
        remain_standing = remain_standing_reward,
        best_standing_up = best_standing_from_lying_down,
        kneeling = reward_kneeling,
        splitting = reward_splitting,

        # Generic reward functions used for tasks that have reference
        #   reference can either be just a final goal or a sequence to follow
        goal_only_euclidean_geom_xpos = reward_goal_only_euclidean_geom_xpos,  # Only works when reference is just a final goal
        basic_r_geom_xpos = reward_only_basic_r_geom_xpos,  # Only supplies the basic remain standing reward (but also plots how well the agent does wrt the reference)
    )
    