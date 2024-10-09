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
from seq_reward.cost_fns import euclidean_distance_advanced, euclidean_distance_advanced_arms_only
from vlm_reward.reward_models.joint_pred import load_joint_prediction_model

from vlm_reward.reward_main import compute_rewards
from vlm_reward.reward_transforms import half_gaussian_filter_1d
from constants import TASK_SEQ_DICT
from utils import calc_iqm

class VisualJointBasedSeqRewardCallback(BaseCallback):
    """
    Custom callback for calculating sequence matching rewards using visual model predictions
    for joint positions after rollouts are collected.
    """
    def __init__(self, task_name, matching_fn_cfg, visual_model_cfg, use_geom_xpos, use_image_for_ref, verbose=0):
        super(VisualJointBasedSeqRewardCallback, self).__init__(verbose)

        self.use_geom_xpos = use_geom_xpos
        self.use_image_for_ref = use_image_for_ref
        
        # Load the ref seq. Note: if using image, it must be processed into an actual ref seq in _on_training_start (once the model has been initialized)
        self._ref_seq = load_reference_seq(task_name=task_name, seq_name=matching_fn_cfg["seq_name"], use_geom_xpos=self.use_geom_xpos, use_image=self.use_image_for_ref)
        logger.info(f"[VisualSeqRewardCallback] Loaded reference sequence. task_name={task_name}, seq_name={matching_fn_cfg['seq_name']}, shape={self._ref_seq.shape}")
        
        self._scale = matching_fn_cfg['scale']
        self._matching_fn, self._matching_fn_name = get_matching_fn(matching_fn_cfg, matching_fn_cfg["cost_fn"])
        
        # visual specific attributes
        self.visual_model_cfg = visual_model_cfg
        self.batch_size = self.visual_model_cfg["reward_batch_size"]
        self.kappa = self.visual_model_cfg["confidence_kappa"] # from wild visual navigation (higher -> relaxed uncertainty)
        self.add_to_gt_rewards = self.visual_model_cfg["add_to_gt_rewards"]
        rank = self.visual_model_cfg['rank']
        self.visual_model_device = f"cuda:{rank}"

        self.uncertainty_stats_for_logging = RunningStats()
        
        logger.info(f"[VisualSeqRewardCallback] Initialized with matching fn {self._matching_fn_name} and batch_size={self.batch_size}")

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """

        if self.use_image_for_ref:
            ref_seq_torch = th.as_tensor(np.array(self._ref_seq)).permute(0,3,1,2).to(self.visual_model_device) / 255.0
            joint_positions, _ = self.model.compute_joint_predictions_with_uncertainty(ref_seq_torch)
            joint_positions = joint_positions.view(-1, 18, 3) # TODO: hard coding joint dim here for ease
            self._ref_seq = joint_positions.cpu().numpy()

    def _on_training_end(self) -> None:
        """
        This method is called after the last rollout ends.
        """

        final_mean_uncertainty = self.uncertainty_stats_for_logging.get_mean()
        final_std_uncertainty = self.uncertainty_stats_for_logging.get_std()
        np.save("mu_std_uncertainty.npy", np.array([final_mean_uncertainty, final_std_uncertainty]))

    def on_rollout_end(self) -> None:
        """Calculate rewards based on visual predictions after rollout ends"""
        start_time = time.time()
        
        replay_buffer_pos = self.model.replay_buffer.pos
        total_timesteps = self.model.num_timesteps - self.model.previous_num_timesteps
        env_episode_timesteps = total_timesteps // self.model.env.num_envs
        
        # Get frames from replay buffer
        frames = th.from_numpy(np.array(self.model.replay_buffer.render_arrays)).float().to(self.visual_model_device) / 255.0

        # TODO, IMPORTANT: we cut off the last frame because it is always the reset frame (which has high confidence and good joint predictions, resulting in a really large reward at the end)
        frames_chopped = frames[:-1]
        frames_chopped = rearrange(frames_chopped, "n_steps n_envs h w c -> (n_steps n_envs) c h w")

        # NOTE: this assumes we are always using an uncertainty-based model
        joint_positions, uncertainties = self.model.compute_joint_predictions_with_uncertainty(frames_chopped)

        # Update uncertainty statistics and compute confidence weights
        #self.model.uncertainty_stats.update(uncertainties.mean().item())
        confidence_weights = self.model.compute_confidence_weights(uncertainties, self.model.uncertainty_stats)
        self.uncertainty_stats_for_logging.update_batch(uncertainties) # only used to log, not update anything
        
        # Rearrange predictions and confidence weights to env shape
        joint_positions = rearrange(joint_positions, "(n_steps n_envs) (n_joints d_joint) -> n_steps n_envs n_joints d_joint", n_envs=self.model.env.num_envs, n_joints = self._ref_seq.shape[1])
        confidence_weights = rearrange(confidence_weights, "(n_steps n_envs) -> n_steps n_envs", n_envs=self.model.env.num_envs)
        
        # Just set the values for the last frame as the same as the second to last frame (because last frame is corrupted)
        joint_positions = th.cat((joint_positions, joint_positions[-1][None]), dim=0)
        confidence_weights = th.cat((confidence_weights, confidence_weights[-1][None]), dim=0)
 
        # Clear the render arrays once computations have been run on them
        self.model.replay_buffer.clear_render_arrays()

        # Compute rewards for each environment
        joint_positions = joint_positions.cpu().numpy()
        confidence_weights = confidence_weights.cpu().numpy()
        matching_reward_list = []
        for env_i in range(self.model.env.num_envs):
            # Apply confidence weighting to joint positions
            env_joint_positions = joint_positions[:, env_i]
            env_confidence_weights = confidence_weights[:, env_i] 

            matching_reward, _ = self._matching_fn(env_joint_positions, self._ref_seq) 
            matching_reward *= env_confidence_weights
            matching_reward_list.append(matching_reward)
        
        rewards = np.stack(matching_reward_list, axis=1)
        
        # Update replay buffer rewards
        if replay_buffer_pos - env_episode_timesteps >= 0:
            self.model.replay_buffer.rewards[
                replay_buffer_pos - env_episode_timesteps : replay_buffer_pos, :
            ] += rewards
        else:
            self.model.replay_buffer.rewards[
                -(env_episode_timesteps - replay_buffer_pos):
            ] += rewards[: env_episode_timesteps - replay_buffer_pos]
            
            self.model.replay_buffer.rewards[:replay_buffer_pos] += rewards[
                env_episode_timesteps - replay_buffer_pos:
            ]
        
        logger.debug(f"VisualBasedSeqRewardCallback took {time.time() - start_time} seconds")

    def _on_step(self) -> bool:
        return True

class RunningStats:
    """Helper class to compute running mean and standard deviation"""
    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0
    
    def update_batch(self, x_batch: np.array):
        self.n += len(x_batch)
        x = x_batch.mean().item()

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def update(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def get_mean(self):
        return self.new_m if self.n else None

    def get_std(self):
        return np.sqrt(self.new_s / (self.n - 1)) if self.n > 1 else None

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
        use_image_for_ref: bool = False,
        visual_model_rank: bool = 0,
        task_name: str = "",
        threshold: float = 0.5,
        success_fn_cfg: dict = {},
        matching_fn_cfg: dict = {}, 
        visual_fn_cfg: dict = {},
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
        self._use_image_for_ref = use_image_for_ref
        self._threshold = threshold
        self._calc_visual_reward = calc_visual_reward
        self._visual_model_device = f"cuda:{visual_model_rank}"

        if self._calc_visual_reward:
            self.all_uncertainties = []

        if task_name != "":
            self._goal_ref_seq = load_reference_seq(task_name=task_name, seq_name="key_frames", use_geom_xpos=self._use_geom_xpos, use_image=self._use_image_for_ref)
            logger.info(f"[VideoRecorderCallback] Loaded reference sequence. task_name={task_name}, seq_name=key_frames, use_geom_xpos={self._use_geom_xpos}, shape={self._goal_ref_seq.shape}")

            self.set_ground_truth_goal_matching_fn(task_name, use_geom_xpos)
            self.set_success_fn(success_fn_cfg)

            self._calc_gt_reward = True

            self._success_json_save_path = os.path.join(self._rollout_save_path, "success_results.json")
            self._success_results = {}
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
    
    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        if self._use_image_for_ref:
            ref_seq_torch = th.as_tensor(np.array(self._goal_ref_seq)).permute(0,3,1,2).to(self._visual_model_device) / 255.0
            joint_positions, _ = self.model.compute_joint_predictions_with_uncertainty(ref_seq_torch)
            joint_positions = joint_positions.view(-1, 18, 3) # TODO: hard coding joint dim here for ease
            self._goal_ref_seq = joint_positions.cpu().numpy()

    def _on_training_end(self) -> None:
        if self._calc_visual_reward:
            unc = np.concatenate(self.all_uncertainties)
            np.save("all_uncertainties.npy", unc)

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
            current_stage = 0
            stage_completed = 0
            # Track the number of steps where a stage is being completed
            #   Offset by 1 to play nicely with the stage_completed
            n_steps_completing_each_stage = [0] * (len(ref_seq) + 1)

            for i in range(len(reward_matrix)):  # Iterate through the timestep
                if reward_matrix[i][current_stage] > threshold and stage_completed < len(ref_seq):
                    stage_completed += 1
                    current_stage = min(current_stage + 1, len(ref_seq)-1)
                    n_steps_completing_each_stage[stage_completed] += 1
                elif current_stage == len(ref_seq)-1 and reward_matrix[i][current_stage] > threshold:
                    # We are at the last stage
                    n_steps_completing_each_stage[stage_completed] += 1
                elif current_stage > 0 and reward_matrix[i][current_stage-1] > threshold:
                    # Once at least 1 stage is counted, if it's still above the threshold for the current stage, we will add to the count
                    n_steps_completing_each_stage[stage_completed] += 1

            pct_stage_completed = stage_completed/len(ref_seq)

            # The last pose is never reached
            if n_steps_completing_each_stage[-1] == 0:
                # We don't count any of the previous stage's steps
                pct_timesteps_completing_the_stages = 0
            else:
                pct_timesteps_completing_the_stages = np.sum(n_steps_completing_each_stage)/len(ref_seq)

            return pct_stage_completed, pct_timesteps_completing_the_stages
        
        self._success_fn_based_on_all_pos = lambda obs_seq, ref_seq=self._goal_ref_seq, threshold=success_fn_cfg["threshold_for_all_pos"]: success_fn(obs_seq, ref_seq, threshold)

        self._success_fn_based_on_only_arm_pos = lambda obs_seq, ref_seq=self._goal_ref_seq, threshold=success_fn_cfg["threshold_for_arm_pos"]: success_fn(obs_seq[:, 12:], ref_seq[:, 12:], threshold)


    def add_success_results(self, curr_timestep, timestep_success_dict):
        """
        Add the success results to the success_results dictionary
        """
        self._success_results[curr_timestep] = timestep_success_dict

        with open(self._success_json_save_path, "w") as f:
            json.dump(self._success_results, f, indent=4)

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

                    full_pos_success_rate_iqm, full_pos_success_rate_std = calc_iqm(full_pos_success_rate_list)
                    full_pos_pct_success_timesteps_iqm, full_pos_pct_success_timesteps_std = calc_iqm(full_pos_pct_success_timesteps_list)
                    arm_pos_success_rate_iqm, arm_pos_success_rate_std = calc_iqm(arm_pos_success_rate_list)
                    arm_pos_pct_success_timesteps_iqm, arm_pos_pct_success_timesteps_std = calc_iqm(arm_pos_pct_success_timesteps_list)

                    # Save the success results locally
                    self.add_success_results(self.num_timesteps, {
                        "full_pos_success_rate": full_pos_success_rate_list,
                        "full_pos_success_rate_iqm": float(full_pos_success_rate_iqm),
                        "full_pos_success_rate_std": float(full_pos_success_rate_std),
                        "full_pos_pct_success_timesteps": full_pos_pct_success_timesteps_list,
                        "full_pos_pct_success_timesteps_iqm": float(full_pos_pct_success_timesteps_iqm),
                        "full_pos_pct_success_timesteps_std": float(full_pos_pct_success_timesteps_std),
                        "arm_pos_success_rate": arm_pos_success_rate_list,
                        "arm_pos_success_rate_iqm": float(arm_pos_success_rate_iqm),
                        "arm_pos_success_rate_std": float(arm_pos_success_rate_std),
                        "arm_pos_pct_success_timesteps": arm_pos_pct_success_timesteps_list,
                        "arm_pos_pct_success_timesteps_iqm": float(arm_pos_pct_success_timesteps_iqm),
                        "arm_pos_pct_success_timesteps_std": float(arm_pos_pct_success_timesteps_std)
                    })
                    
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
                arm_reward_matrix = np.exp(-euclidean_distance_advanced_arms_only(geom_xposes[0], self._goal_ref_seq))
                if self._calc_matching_reward:
                    reward_matrix_using_seq_req = np.exp(-euclidean_distance_advanced(geom_xposes[0], self._seq_matching_ref_seq))
                    arm_reward_matrix_using_seq_req = np.exp(-euclidean_distance_advanced_arms_only(geom_xposes[0], self._seq_matching_ref_seq))
                for i in range(len(infos)):
                    # Plot the reward (exp of the negative distance) based on the ground-truth goal reference sequence
                    infos[i]["rf_r"] = str([f"{reward_matrix[i][j]:.2f}" for j in range(len(self._goal_ref_seq))]) + " | " +  str([f"{arm_reward_matrix[i][j]:.2f}" for j in range(len(self._goal_ref_seq))])
                    # Success Rate based on the entire body + based on only the arm
                    infos[i]["[all, arm] success"] = f"{full_pos_success_rate_list[0]:.2f}, {arm_pos_success_rate_list[0]:.2f}"
                    if self._calc_matching_reward:
                        # Plot the reward (exp of the negative distance) based on the reference sequence USED FOR SEQUENCE MATCHING
                        infos[i]["seqrf_r"] = str([f"{reward_matrix_using_seq_req[i][j]:.2f}" for j in range(len(self._seq_matching_ref_seq))]) + " | " + str([f"{arm_reward_matrix_using_seq_req[i][j]:.2f}" for j in range(len(self._seq_matching_ref_seq))])

            frames = th.from_numpy(np.array(screens)).float().to(self._visual_model_device).permute(0,3,1,2) / 255.0
            
            if self._calc_visual_reward and self._calc_matching_reward: # visual + matching
                logger.info("Evaluating rollout for recorder callback, visual and sequence")
                # NOTE: this assumes we are always using an uncertainty-based model
                # TODO IMPORTANT: chopping off the last frame because it is corrupted
                frames_chopped = frames[:-1]
                joint_positions, uncertainties = self.model.compute_joint_predictions_with_uncertainty(frames_chopped)
                # Do not update running uncertainty stats, since this is eval mode
                confidence_weights = self.model.compute_confidence_weights(uncertainties, self.model.uncertainty_stats)
                joint_positions = rearrange(joint_positions, "steps (n_joints d_joint) -> steps n_joints d_joint", n_joints = self._seq_matching_ref_seq.shape[1])
                
                # TODO IMPORTANT: using positions and confidence weights for second to last frame as the last frame (because it is corrupted)
                joint_positions = th.cat((joint_positions, joint_positions[-1][None]), dim=0)
                confidence_weights = th.cat((confidence_weights, confidence_weights[-1][None]), dim=0)

                joint_positions = joint_positions.cpu().numpy()
                confidence_weights = confidence_weights.cpu().numpy()
                vlm_matching_reward, vlm_matching_reward_info = self._matching_fn(joint_positions, self._seq_matching_ref_seq)
                vlm_matching_reward *= confidence_weights
                vlm_matching_reward_info["confidence"] = confidence_weights

                self.all_uncertainties.append(uncertainties.cpu().numpy().flatten())

                self.logger.record("rollout/avg_vlm_matching_reward", 
                                np.mean(vlm_matching_reward)/self._scale, 
                                exclude=("stdout", "log", "json", "csv"))
                

                self.logger.record("rollout/avg_uncertainty", 
                                np.mean(uncertainties.cpu().numpy()), 
                                exclude=("stdout", "log", "json", "csv"))
                
                self.logger.record("rollout/avg_confidence_weight", 
                                np.mean(confidence_weights), 
                                exclude=("stdout", "log", "json", "csv"))
                
                # Add the matching_reward to the infos so that we can plot it
                for i in range(len(infos)):
                    infos[i]["vlm_matching_reward"] = f"{vlm_matching_reward[i]:.2f}"                    
                # Save the matching_rewards locally    
                with open(os.path.join(self._rollout_save_path, f"{self.num_timesteps}_rollouts_vlm_matching_rewards.npy"), "wb") as f:
                    np.save(f, np.array(vlm_matching_reward))
                
                if self._plot_matching_visualization:
                    self.plot_matching_visualization(raw_screens, vlm_matching_reward, vlm_matching_reward_info)


            elif self._calc_visual_reward and not self._calc_matching_reward: # just visual reward
                logger.info("Evaluating rollout for recorder callback")
                self.model.reward_model.requires_grad_(False)
                vlm_rewards = self.model._compute_joint_rewards(
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

                if self._plot_matching_visualization and not self._calc_visual_reward: # only plot if not already plotted using visual rewards
                    self.plot_matching_visualization(raw_screens, matching_reward, matching_reward_info)

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

    def plot_matching_visualization(self, raw_screens, matching_reward, matching_reward_info):
        # TODO: For now, we can only visualize this when the reference frame is defined via a gif
        matching_reward_viz_save_path = os.path.join(self._rollout_save_path, f"{self.num_timesteps}_matching_fn_viz.png")

        # Subsample the frames. Otherwise, the visualization will be too long
        if len(raw_screens) > 20:
            obs_seq_skip_step = int(0.1 * len(raw_screens))
            raw_screens_used_to_plot = np.array([raw_screens[i] for i in range(obs_seq_skip_step, len(raw_screens), obs_seq_skip_step)])
        else:
            raw_screens_used_to_plot = np.array(raw_screens)
            
        if len(self._seq_matching_ref_seq_frames) > 8:
            ref_seq_skip_step = max(int(0.1 * len(self._seq_matching_ref_seq_frames)), 2)
            ref_seqs_used_to_plot = np.array([self._seq_matching_ref_seq_frames[i] for i in range(ref_seq_skip_step, len(self._seq_matching_ref_seq_frames), ref_seq_skip_step)])
        else:
            ref_seqs_used_to_plot = self._seq_matching_ref_seq_frames

        seq_matching_viz(
            matching_fn_name=self._matching_fn_name,
            obs_seq=raw_screens_used_to_plot,
            ref_seq=ref_seqs_used_to_plot,
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