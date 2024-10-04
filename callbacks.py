import os
import time
import json
from typing import Any, Dict, Optional
import imageio
import gymnasium
import torch as th
import numpy as np
from numpy import array
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import (
    CheckpointCallback as SB3CheckpointCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
from stable_baselines3.common.logger import Image as LogImage  # To avoid conflict with PIL.Image
from wandb.integration.sb3 import WandbCallback as SB3WandbCallback
from stable_baselines3.common.base_class import BaseAlgorithm

from PIL import Image, ImageDraw, ImageFont
from numbers import Number

from loguru import logger
from einops import rearrange

from seq_reward.seq_utils import get_matching_fn, load_reference_seq, load_images_from_reference_seq, seq_matching_viz
from seq_reward.cost_fns import euclidean_distance_advanced

from vlm_reward.reward_main import compute_rewards
from vlm_reward.reward_transforms import half_gaussian_filter_1d
from constants import TASK_SEQ_DICT
from utils import calc_iqm

class JointBasedSeqRewardCallback(BaseCallback):
    """
    Custom callback for calculating joint based sequence matching rewards after rollouts are collected.
    """
    def __init__(self, task_name, matching_fn_cfg, use_geom_xpos, verbose=0):
        super(JointBasedSeqRewardCallback, self).__init__(verbose)

        self._ref_seq = load_reference_seq(task_name=task_name, seq_name=matching_fn_cfg["seq_name"], use_geom_xpos=use_geom_xpos)
        logger.info(f"[JointBasedSeqRewardCallback] Loaded reference sequence. task_name={task_name}, seq_name={matching_fn_cfg['seq_name']}, use_geom_xpos={use_geom_xpos}, self._ref_seq.shape={self._ref_seq.shape}")

        self._scale = matching_fn_cfg['scale']
        self._use_geom_xpos = use_geom_xpos

        self._matching_fn, self._matching_fn_name = get_matching_fn(matching_fn_cfg, matching_fn_cfg["cost_fn"])

        logger.info(f"[JointBasedSeqRewardCallback] Loaded matching fn {self._matching_fn_name} with {matching_fn_cfg}")


    def on_rollout_end(self) -> None:
        """
        This method is called after the rollout ends.
        You can access and modify the rewards in the ReplayBuffer here.
        """
        # Time this function
        start_time = time.time()

        replay_buffer_pos = self.model.replay_buffer.pos
        total_timesteps = self.model.num_timesteps - self.model.previous_num_timesteps  # Total number of timesteps that we have collected
        env_episode_timesteps = total_timesteps // self.model.env.num_envs  # Number of timesteps that we have collected per environment

        # logger.debug(f"\nreplay_buffer_pos={replay_buffer_pos}, total_timesteps={total_timesteps}, \nenv_episode_timesteps={env_episode_timesteps}, self.model.num_timesteps={self.model.num_timesteps}")

        # Get the observation from the replay buffer
        #   size: (train_freq, n_envs, obs_size)
        #   For OT-based reward, train_freq = episode_length
        if self._use_geom_xpos:
            obs_to_process = np.array(self.model.replay_buffer.geom_xpos)
            # Normalize along the center of mass (index 1)
            obs_to_process = obs_to_process - obs_to_process[:, :, 1:2, :]
        else:
            if replay_buffer_pos - env_episode_timesteps >= 0:
                # logger.debug(f"not circular, check replay buffer: {self.model.replay_buffer.observations[replay_buffer_pos - env_episode_timesteps : replay_buffer_pos, :].shape}")
                
                obs_to_process = np.array(self.model.replay_buffer.observations[replay_buffer_pos - env_episode_timesteps : replay_buffer_pos, :])
            else:
                # Split reward assignment (circular buffer)
                # logger.debug(f"\ncircular, part 1={self.model.replay_buffer.observations[-(env_episode_timesteps - replay_buffer_pos) :, :].shape} \n part 2={self.model.replay_buffer.observations[:replay_buffer_pos, :].shape}")

                obs_to_process = np.concatenate((self.model.replay_buffer.observations[-(env_episode_timesteps - replay_buffer_pos) :, :], self.model.replay_buffer.observations[:replay_buffer_pos, :]), axis=0)

        matching_reward_list = []
        for env_i in range(self.model.env.num_envs):
            # TODO: A hard-coded value (22 is matching qpos of the environment)
            if self._use_geom_xpos:
                # Don't need to do anything here, geom_xpos is getting normalized when we get it from the replay buffer
                obs = obs_to_process[:, env_i]
            else:
                obs = obs_to_process[:, env_i, :22]  # size: (train_freq, 22)
            
            matching_reward, _ = self._matching_fn(obs, self._ref_seq)  # size: (train_freq,)
            # logger.debug(f"matching_reward={matching_reward.shape}")
            matching_reward_list.append(matching_reward)

        rewards = np.stack(matching_reward_list, axis=1)  # size: (train_freq, n_envs)

        # Add the optimal transport reward to exisiting rewards
        if replay_buffer_pos - env_episode_timesteps >= 0:
            self.model.replay_buffer.rewards[
                replay_buffer_pos - env_episode_timesteps : replay_buffer_pos, :
            ] += rewards[:, :]
        else:
            # Split reward assignment (circular buffer)
            self.model.replay_buffer.rewards[
                -(env_episode_timesteps - replay_buffer_pos) :, :
            ] += rewards[: env_episode_timesteps - replay_buffer_pos, :]

            self.model.replay_buffer.rewards[:replay_buffer_pos, :] += rewards[
                env_episode_timesteps - replay_buffer_pos :, :
            ]

        self.model.replay_buffer.clear_geom_xpos()

        print(f"JointBasedSeqRewardCallback took {time.time() - start_time} seconds")


    def _on_step(self) -> bool:
        """
        Just need to define this method to avoid NotImplementedError

        Return: 
            If the callback returns False, training is aborted early.
        """
        return True

class VLMRewardCallback(BaseCallback):
    """
    Custom callback for calculating Optimal Transport (OT) rewards after rollouts are collected.
    """
    def __init__(self, scale=1, filter_rewards=False, add_to_gt_rewards=True, verbose=0):
        super(VLMRewardCallback, self).__init__(verbose)

        self._scale = scale
        self._filter_rewards = filter_rewards
        self._add_to_gt_rewards = add_to_gt_rewards

    def on_rollout_end(self) -> None:
        """
        This method is called after the rollout ends.
        You can access and modify the rewards in the ReplayBuffer here.
        """
        # Time this function
        start_time = time.time()

        replay_buffer_pos = self.model.replay_buffer.pos
        total_timesteps = self.model.num_timesteps - self.model.previous_num_timesteps  # Total number of timesteps that we have collected
        env_episode_timesteps = total_timesteps // self.model.env.num_envs  # Number of timesteps that we have collected per environment
        total_episodes = self.model.get_episode_num() - self.model.previous_num_episodes
        env_episodes = total_episodes // self.model.env.num_envs

        ### Prepare the frame to be processed
        frames = th.from_numpy(np.array(self.model.replay_buffer.render_arrays))

        print(f"Start calculating rewards: frames.shape={frames.shape}")

        frames = rearrange(frames, "n_steps n_envs ... -> (n_steps n_envs) ...")
        
        ### Compute rewards
        # NOTE: distributed will be off if num_workers == 1
        rewards = compute_rewards(
            model=self.model.reward_model,
            frames=frames,
            rank0_batch_size_pct=self.model.reward_model_config["rank0_batch_size_pct"],
            batch_size=self.model.reward_model_config["reward_batch_size"],  # This is the total batch size
            num_workers=self.model.n_gpu_workers,
            worker_frames_tensor=self.model.worker_frames_tensor
            )
        
        rewards = rearrange(
            rewards,
            "(n_steps n_envs) ... -> (n_envs n_steps) ...",
            n_envs=self.model.env.num_envs,
        )

        # Scale the rewards
        rewards = rewards * self._scale

        # Filter the rewards
        if self._filter_rewards:
            print("Filtering rewards")
            rewards = half_gaussian_filter_1d(rewards, sigma=4, smooth_last_N=True) 
            
        # Clear the rendered images in the ReplayBuffer
        self.model.replay_buffer.clear_render_arrays()

        ### Update the rewards
        if self._add_to_gt_rewards:
            print("Adding VLM rewards to GT rewards")
            # Add the VLM reward to exisiting rewards
            if replay_buffer_pos - env_episode_timesteps >= 0:
                self.model.replay_buffer.rewards[
                    replay_buffer_pos - env_episode_timesteps : replay_buffer_pos, :
                ] += rewards[:, :]
            else:
                # Split reward assignment (circular buffer)
                self.model.replay_buffer.rewards[
                    -(env_episode_timesteps - replay_buffer_pos) :, :
                ] += rewards[: env_episode_timesteps - replay_buffer_pos, :]

                self.model.replay_buffer.rewards[:replay_buffer_pos, :] += rewards[
                    env_episode_timesteps - replay_buffer_pos :, :
                ]
        else:
            print("Overwriting GT rewards with VLM rewards")
            # Overwrite the rewards with VLM rewards
            if replay_buffer_pos - env_episode_timesteps >= 0:
                self.model.replay_buffer.rewards[
                    replay_buffer_pos - env_episode_timesteps : replay_buffer_pos, :
                ] = rewards[:, :]
            else:
                # Split reward assignment (circular buffer)
                self.model.replay_buffer.rewards[
                    -(env_episode_timesteps - replay_buffer_pos) :, :
                ] = rewards[: env_episode_timesteps - replay_buffer_pos, :]

                self.model.replay_buffer.rewards[:replay_buffer_pos, :] = rewards[
                    env_episode_timesteps - replay_buffer_pos :, :
                ]

        ### Logging the rewards 
        rewards = rearrange(rewards, "n_steps n_envs -> n_envs n_steps")
        for env_idx in range(self.model.env.num_envs):
            # Compute sum of rewards per episode
            rewards_per_episode = np.sum(
                np.reshape(
                    rewards[env_idx], (env_episodes, self.model.episode_length)
                ),
                axis=1,
            )
            self.model.ep_vlm_info_buffer.extend([rewards_per_episode.tolist()])

        print(f"VLMRewardCallback took {time.time() - start_time} seconds")


    def _on_step(self) -> bool:
        """
        Just need to define this method to avoid NotImplementedError

        Return: 
            If the callback returns False, training is aborted early.
        """
        return True
    

def plot_info_on_frame(pil_image, info, font_size=20):
    """
    Parameters:
        pil_image: PIL.Image
            The image to plot the info on
        info: Dict
            The information to plot on the image
        font_size: int
            The size of the font to use for the text
    
    Effects:
        pil_image is modified to include the info
    """
    # TODO: this is a hard-coded path
    font = ImageFont.truetype("/share/portal/hw575/vlmrm/src/vlmrm/cli/arial.ttf", font_size)
    draw = ImageDraw.Draw(pil_image)

    x = font_size  # X position of the text
    y = pil_image.height - font_size  # Beginning of the y position of the text
    
    i = 0
    for k in info:
        # TODO: This is pretty ugly
        if not any([text in k for text in ["TimeLimit", "render_array", "geom_xpos"]]):
            reward_text = f"{k}:{info[k]}"
            # Plot the text from bottom to top
            text_position = (x, y - (font_size + 10)*(i+1))
            draw.text(text_position, reward_text, fill=(255, 255, 255), font=font)
        i += 1


class VideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        eval_env: gymnasium.Env,
        rollout_save_path: str,
        render_freq: int,
        render_dim: tuple = (480, 480, 3),
        n_eval_episodes: int = 1,
        deterministic: bool = True,
        use_geom_xpos: bool = True,
        task_name: str = "",
        threshold: float = 0.5,
        success_fn_cfg: dict = {},
        matching_fn_cfg: dict = {}, 
        calc_visual_reward: bool = False,
        verbose=0
    ):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to
        TensorBoard

        Pararmeters
            eval_env: A gym environment from which the trajectory is recorded
                Assumes that there's only 1 environment
            rollout_save_path: The path to save the rollouts (states and rewards)
            render_freq: Render the agent's trajectory every eval_freq call of the callback.
            n_eval_episodes: Number of episodes to render
            deterministic: Whether to use deterministic or stochastic policy
            goal_seq_name: The name of the reference sequence to compare with (This defines the unifying metric that all approaches attempting to solve the same task gets compared against)
            seq_name: The name of the reference sequence to compare with
                You only need to set this if you want to calculate the OT reward
            matching_fn_cfg: The configuration for the matching function
        """
        super().__init__(verbose)
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._render_dim = render_dim
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic
        self._rollout_save_path = rollout_save_path  # Save the state of the environment
        self._use_geom_xpos = use_geom_xpos
        self._threshold = threshold
        self._calc_visual_reward = calc_visual_reward

        if task_name != "":
            self._goal_ref_seq = load_reference_seq(task_name=task_name, seq_name="key_frames", use_geom_xpos=self._use_geom_xpos)
            logger.info(f"[VideoRecorderCallback] Loaded reference sequence. task_name={task_name}, seq_name=key_frames, use_geom_xpos={self._use_geom_xpos}, shape={self._goal_ref_seq.shape}")

            self.set_ground_truth_goal_matching_fn(task_name, use_geom_xpos)
            self.set_success_fn(success_fn_cfg)

            self._calc_gt_reward = True
        else:
            self._calc_gt_reward = False

        if matching_fn_cfg != {}:
            # The reference sequence that is used to calculate the sequence matching reward
            self._seq_matching_ref_seq = load_reference_seq(task_name=task_name, seq_name=matching_fn_cfg["seq_name"], use_geom_xpos=self._use_geom_xpos)
            # For the frames, we remove the initial frame which matches the initial position
            self._seq_matching_ref_seq_frames = load_images_from_reference_seq(task_name=task_name, seq_name=matching_fn_cfg["seq_name"])[1:]
            # TODO: For now, we can only visualize this when the reference frame is defined via a gif
            self._plot_matching_visualization = len(self._seq_matching_ref_seq_frames) > 0

            self._calc_matching_reward = True
            self._scale = matching_fn_cfg['scale']
            self._matching_fn, self._matching_fn_name = get_matching_fn(matching_fn_cfg, matching_fn_cfg["cost_fn"])

            self._reward_vmin = matching_fn_cfg.get("reward_vmin", -1)
            self._reward_vmax = matching_fn_cfg.get("reward_vmax", 0)

            logger.info(f"[VideoRecorderCallback] Loaded reference sequence for seq level matching. task_name={task_name}, seq_name={matching_fn_cfg['seq_name']}, use_geom_xpos={self._use_geom_xpos}, shape={self._seq_matching_ref_seq.shape}, image_frames_shape={self._seq_matching_ref_seq_frames.shape}")
        else:
            self._calc_matching_reward = False
        

    def set_ground_truth_goal_matching_fn(self, task_name: str, use_geom_xpos: bool):
        """Set the ground-truth goal matching function based on the goal_seq_name.

        This will be unifying metric that we measure the performance of different methods against.

        The function will return an reward array of size (n_timesteps,) where each element is the reward for the corresponding timestep.
        """
        is_goal_reaching_task = TASK_SEQ_DICT[task_name]["task_type"].lower() == "goal_reaching"

        if is_goal_reaching_task:
            logger.info(f"Goal Reaching Task. The ground-truth reward will be calculated based on the final joint state only. Task name = {task_name}")

            assert len(self._goal_ref_seq) == 1, f"Expected only 1 reference sequence, got {len(self._goal_ref_seq)}"
            
            axis_to_norm = (1,2) if use_geom_xpos else 1

            self._gt_goal_matching_fn = lambda rollout: np.exp(-np.linalg.norm(rollout - self._goal_ref_seq, axis=axis_to_norm))
        else:
            def seq_matching_fn(ref, rollout, threshold):
                """
                Calculate the reward based on the sequence matching to the goal_ref_seq

                Parameters:
                    rollout: np.array (rollout_length, ...)
                        The rollout sequence to calculate the reward
                """
                # Calculate reward from the rollout to self.goal_ref_seq
                reward_matrix = np.exp(-euclidean_distance_advanced(rollout, ref))

                # Detect when a stage is completed (the rollout is close to the goal_ref_seq) (under self._threshold)
                stage_completed = 0
                stage_completed_matrix = np.zeros(reward_matrix.shape) # 1 if the stage is completed, 0 otherwise
                current_stage_matrix = np.zeros(reward_matrix.shape) # 1 if the current stage, 0 otherwise
                
                for i in range(len(reward_matrix)):  # Iterate through the timestep
                    current_stage_matrix[i, stage_completed] = 1
                    if reward_matrix[i][stage_completed] > threshold and stage_completed < len(ref) - 1:
                        stage_completed += 1
                    stage_completed_matrix[i, :stage_completed] = 1

                # Find the highest reward to each reference sequence
                highest_reward = np.max(reward_matrix, axis=0)

                # Reward (shape: (rollout)) at each timestep is
                #   Stage completion reward + Reward at the current stage
                reward = np.sum(stage_completed_matrix * highest_reward + current_stage_matrix * reward_matrix, axis=1)/len(ref)

                return reward
            
            self._gt_goal_matching_fn = lambda rollout: seq_matching_fn(self._goal_ref_seq, rollout, self._threshold)


    def set_success_fn(self, success_fn_cfg):
        """
        Whether the entire body is above an threshold (0.5)
        Whether the arm is above an threshold (0.55)

        Binary success: whether at any point has the key poses have been hit
            # of the key poses that have been hit (in the right order)
        The percentage of time that it's holding the key pose
            For each key pose, we find the time interval that each key poses hold
        """
        def success_fn(obs_seq, ref_seq, threshold):
            """
            Calculate the binary success based on the rollout and the reference sequence

            Parameters:
                rollout: np.array (rollout_length, ...)
                    The rollout sequence to calculate the reward

            Return:
                pct_stage_completed: float
                    The percentage of stages that are completed
                pct_timesteps_completing_the_stages: float
                    The percentage of timesteps that are completing the stages
            """
            # Calculate reward from the rollout to self.goal_ref_seq
            reward_matrix = np.exp(-euclidean_distance_advanced(obs_seq, ref_seq))

            # Detect when a stage is completed (the rollout is close to the goal_ref_seq) (under self._threshold)
            stage_completed = 0
            # Track the number of steps where a stage is being completed
            n_steps_completing_each_stage = [0] * len(ref_seq)

            for i in range(len(reward_matrix)):  # Iterate through the timestep
                if stage_completed > 1:
                    # Once at least 1 stage is counted, if it's still above the threshold for the current stage, we will add to the count
                    n_steps_completing_each_stage[stage_completed-1] += 1

                if reward_matrix[i][stage_completed] > threshold and stage_completed < len(ref_seq) - 1:
                    stage_completed += 1
                    # stage_completed-1 because stage_completed is counting the number of stages completed
                    n_steps_completing_each_stage[stage_completed-1] += 1

            pct_stage_completed = stage_completed/len(ref_seq)

            # The last pose is never reached
            if n_steps_completing_each_stage[-1] == 0:
                # We don't count any of the previous stage's steps
                pct_timesteps_completing_the_stages = 0
            else:
                pct_timesteps_completing_the_stages = np.sum(n_steps_completing_each_stage)/len(obs_seq)

            return pct_stage_completed, pct_timesteps_completing_the_stages
        
        self._success_fn_based_on_all_pos = lambda obs_seq, ref_seq=self._goal_ref_seq, threshold=success_fn_cfg["threshold_for_all_pos"]: success_fn(obs_seq, ref_seq, threshold)

        self._success_fn_based_on_only_arm_pos = lambda obs_seq, ref_seq=self._goal_ref_seq, threshold=success_fn_cfg["threshold_for_arm_pos"]: success_fn(obs_seq[:, 12:], ref_seq[:, 12:], threshold)

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            # Saving for only one env (the first env)
            #   Because we are using this to plot
            raw_screens = []
            screens = []
            infos = []
            # Saving for each env
            states = []
            rewards = []
            geom_xposes = [[] for _ in range(self._n_eval_episodes)]

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in
                the captured `screens` list

                :param _locals: A dictionary containing all local variables of the
                 callback's scope
                :param _globals: A dictionary containing all global variables of the
                 callback's scope
                """
                env_i = _locals['i']

                if env_i == 0:
                    screen = self._eval_env.render()

                    image_int = np.uint8(screen)[:self._render_dim[0], :self._render_dim[1], :]

                    raw_screens.append(Image.fromarray(image_int))
                    screens.append(Image.fromarray(image_int))  # The frames here will get plotted with info later
                    infos.append(_locals.get('info', {}))

                    states.append(_locals["observations"][:, :22])
                    rewards.append(_locals["rewards"])

                geom_xpos = _locals.get('info', {})["geom_xpos"]

                # Normalize the joint states based on the torso (index 1)
                geom_xpos = geom_xpos - geom_xpos[1]
                geom_xposes[env_i].append(geom_xpos)
            
            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )

            # Save the raw_screens locally
            imageio.mimsave(os.path.join(self._rollout_save_path, f"{self.num_timesteps}_rollouts.gif"), raw_screens, duration=1/30, loop=0)

            states = np.array(states)  # size: (rollout_length, n_eval_episodes, 22)
            rewards = np.array(rewards) # size: (rollout_length, n_eval_episodes)
            geom_xposes = np.array(geom_xposes) # (n_eval_episodes, rollout_length, 18, 3)

            if self._calc_gt_reward:
                # Calculate the goal matching reward
                if self._use_geom_xpos:            
                    full_pos_success_rate_list = []
                    full_pos_pct_success_timesteps_list = []
                    arm_pos_success_rate_list = []
                    arm_pos_pct_success_timesteps_list = []

                    for env_i in range(self._n_eval_episodes):
                        # Don't need to do anything here, geom_xpos is getting normalized in the grab_screens function
                        geom_xposes_to_process = geom_xposes[env_i]

                        full_pos_success_rate, full_pos_pct_success_timesteps = self._success_fn_based_on_all_pos(geom_xposes_to_process)
                        arm_pos_success_rate, arm_pos_pct_success_timesteps = self._success_fn_based_on_only_arm_pos(geom_xposes_to_process)

                        full_pos_success_rate_list.append(full_pos_success_rate)
                        full_pos_pct_success_timesteps_list.append(full_pos_pct_success_timesteps)
                        arm_pos_success_rate_list.append(arm_pos_success_rate)
                        arm_pos_pct_success_timesteps_list.append(arm_pos_pct_success_timesteps)

                    # Save the result as a json
                    with open(os.path.join(self._rollout_save_path, f"{self.num_timesteps}_success_results.json"), "w") as f:
                        json.dump({
                            "full_pos_success_rate": full_pos_success_rate_list,
                            "full_pos_pct_success_timesteps": full_pos_pct_success_timesteps_list,
                            "arm_pos_success_rate": arm_pos_success_rate_list,
                            "arm_pos_pct_success_timesteps": arm_pos_pct_success_timesteps_list
                        }, f)

                    full_pos_success_rate_iqm, _ = calc_iqm(full_pos_success_rate_list)
                    full_pos_pct_success_timesteps_iqm, _ = calc_iqm(full_pos_pct_success_timesteps_list)
                    arm_pos_success_rate_iqm, _ = calc_iqm(arm_pos_success_rate_list)
                    arm_pos_pct_success_timesteps_iqm, _ = calc_iqm(arm_pos_pct_success_timesteps_list)
                    
                    self.logger.record("eval/full_pos_success", 
                                        full_pos_success_rate_iqm, 
                                        exclude=("stdout", "log", "json", "csv"))
                    
                    self.logger.record("eval/full_pos_pct_success_timesteps", 
                                        full_pos_pct_success_timesteps_iqm, 
                                        exclude=("stdout", "log", "json", "csv"))
                    
                    self.logger.record("eval/arm_pos_success",
                                        arm_pos_success_rate_iqm,
                                        exclude=("stdout", "log", "json", "csv"))
                    
                    self.logger.record("eval/arm_pos_pct_success_timesteps",
                                        arm_pos_pct_success_timesteps_iqm,
                                        exclude=("stdout", "log", "json", "csv"))
                else:
                    raise NotImplementedError(f"Ground truth reward calculation for self._use_geom_xpos={self._use_geom_xpos} is False")

                # Show the success rate for the 1st env's rollout
                reward_matrix = np.exp(-euclidean_distance_advanced(geom_xposes[0], self._goal_ref_seq))
                for i in range(len(infos)):
                    infos[i]["per_ref_seq_r"] = str([f"{reward_matrix[i][j]:.2f}" for j in range(len(self._goal_ref_seq))])
                    infos[i]["full_pos_success"] = f"{full_pos_success_rate_list[0]:.2f}"
                    infos[i]["arm_pos_success"] = f"{arm_pos_success_rate_list[0]:.2f}"

            frames = th.from_numpy(np.array(screens)).float().cuda(0).permute(0,3,1,2) / 255.0
            
            if self._calc_visual_reward:
                logger.info("Evaluating rollout for recorder callback")
                self.model.reward_model.requires_grad_(False)
                vlm_rewards = self.model._compute_joint_rewards(
                            model=self.model.reward_model,
                            transform=self.model.image_transform,
                            frames=frames,
                            ref_joint_states=self.model._ref_joint_states,
                            batch_size=self.model.reward_model_config["reward_batch_size"],
                            ).detach().cpu().numpy()

                # To write the values on the rollout frames
                for i in range(len(infos)):
                    infos[i]["vlm_joint_match_r"] = f"{vlm_rewards[i]:.4f}"

                self.logger.record("rollout/avg_vlm_total_reward", 
                                np.mean(vlm_rewards + rewards), 
                                exclude=("stdout", "log", "json", "csv"))

            # TODO: We can potentially also do VLM reward calculation
            if self._calc_matching_reward:
                if self._use_geom_xpos:
                    # Don't need to do anything here, geom_xpos is getting normalized in the grab_screens function
                    matching_reward, matching_reward_info = self._matching_fn(geom_xposes[0], self._seq_matching_ref_seq)
                else:
                    matching_reward, matching_reward_info = self._matching_fn(np.array(states[:, 0])[:, :22], self._seq_matching_ref_seq)

                self.logger.record("rollout/avg_matching_reward", 
                                np.mean(matching_reward)/self._scale, 
                                exclude=("stdout", "log", "json", "csv"))

                # Add the matching_reward to the infos so that we can plot it
                for i in range(len(infos)):
                    infos[i]["matching_reward"] = f"{matching_reward[i]:.2f}"

                # Save the matching_rewards locally    
                with open(os.path.join(self._rollout_save_path, f"{self.num_timesteps}_rollouts_matching_rewards.npy"), "wb") as f:
                    np.save(f, np.array(matching_reward))

                if self._plot_matching_visualization:
                    # TODO: For now, we can only visualize this when the reference frame is defined via a gif
                    matching_reward_viz_save_path = os.path.join(self._rollout_save_path, f"{self.num_timesteps}_matching_fn_viz.png")

                    # Subsample the frames. Otherwise, the visualization will be too long
                    raw_screens_used_to_plot = [raw_screens[i] for i in range(0, len(raw_screens), 10)]
                    ref_seqs_used_to_plot = [self._seq_matching_ref_seq_frames[i] for i in range(0, len(self._seq_matching_ref_seq_frames), 10)]
                    seq_matching_viz(
                        matching_fn_name=self._matching_fn_name,
                        obs_seq=np.array(raw_screens_used_to_plot),
                        ref_seq=np.array(ref_seqs_used_to_plot),
                        matching_reward=matching_reward,
                        info=matching_reward_info,
                        reward_vmin=self._reward_vmin,
                        reward_vmax=self._reward_vmax,
                        path_to_save_fig=matching_reward_viz_save_path,
                        rolcol_size=2
                    )

                    # Log the image to wandb
                    img = Image.open(matching_reward_viz_save_path)
                    self.logger.record(
                        "trajectory/matching_fn_viz",
                        LogImage(np.array(img), dataformats="HWC"),
                        exclude=("stdout", "log", "json", "csv"),
                    )

            # Plot info on the frames  
            for i in range(len(screens)):
                plot_info_on_frame(screens[i], infos[i])

            screens = [np.uint8(s).transpose(2, 0, 1) for s in screens]

            # Log to wandb
            self.logger.record(
                "trajectory/video",
                Video(th.ByteTensor(array([screens])), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )

            # Save the rollouts locally    
            with open(os.path.join(self._rollout_save_path, f"{self.num_timesteps}_rollouts_states.npy"), "wb") as f:
                np.save(f, np.array(states))

            with open(os.path.join(self._rollout_save_path, f"{self.num_timesteps}_rollouts_geom_xpos_states.npy"), "wb") as f:
                np.save(f, np.array(geom_xposes))
            
            with open(os.path.join(self._rollout_save_path, f"{self.num_timesteps}_rollouts_rewards.npy"), "wb") as f:
                np.save(f, np.array(rewards))

        return True


class WandbCallback(SB3WandbCallback):
    def __init__(
        self,
        model_save_path: str,
        model_save_freq: int,
        **kwargs,
    ):
        super().__init__(
            model_save_path=model_save_path,
            model_save_freq=model_save_freq,
            **kwargs,
        )

    def save_model(self) -> None:
        model_path = os.path.join(
        self.model_save_path, f"model_{self.model.num_timesteps}_steps.zip"
        )
        self.model.save(model_path)