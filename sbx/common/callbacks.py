import os
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
from wandb.integration.sb3 import WandbCallback as SB3WandbCallback
from stable_baselines3.common.base_class import BaseAlgorithm

from PIL import Image, ImageDraw, ImageFont
from numbers import Number

from loguru import logger
from einops import rearrange

import vlm_reward.utils.optimal_transport as custom_ot
import vlm_reward.utils.soft_dtw as custom_sdtw
import time

from vlm_reward.reward_main import compute_rewards
from vlm_reward.reward_transforms import half_gaussian_filter_1d

class JointBasedSeqRewardCallback(BaseCallback):
    """
    Custom callback for calculating joint based sequence matching rewards after rollouts are collected.
    """
    def __init__(self, seq_name, matching_fn_cfg, use_geom_xpos, verbose=0):
        super(JointBasedSeqRewardCallback, self).__init__(verbose)

        self._ref_seq = custom_ot.load_reference_seq(seq_name, use_geom_xpos=use_geom_xpos)
        logger.info(f"[JointBasedSeqRewardCallback] Loaded reference sequence. seq_name={seq_name}, use_geom_xpos={use_geom_xpos}, self._ref_seq.shape={self._ref_seq.shape}")

        self._scale = matching_fn_cfg['scale']
        self._use_geom_xpos = use_geom_xpos

        self.set_matching_fn(matching_fn_cfg)

    def set_matching_fn(self, matching_fn_cfg):
        assert "joint_wasserstein" == matching_fn_cfg["name"] or "joint_soft_dtw", f"Currently only supporting joint_wasserstein or joint soft dynamic time warping, got {matching_fn_cfg['name']}"
        logger.info(f"[JointBasedSeqRewardCallback] Using the following reward model:\n{matching_fn_cfg}")

        matching_fn_name = matching_fn_cfg["name"]

        if matching_fn_name == "joint_wasserstein":
            self._matching_fn = lambda rollout, ref: custom_ot.compute_ot_reward(rollout, ref, custom_ot.COST_FN_DICT[matching_fn_cfg['cost_fn']], self._scale, modification_dict=dict(matching_fn_cfg['modification']))
        elif matching_fn_name == "joint_soft_dtw":
            self._matching_fn = lambda rollout, ref: custom_sdtw.compute_soft_dtw_reward(rollout, ref, custom_ot.COST_FN_DICT[matching_fn_cfg['cost_fn']], matching_fn_cfg['gamma'], self._scale, modification_dict=dict(matching_fn_cfg['modification']))

        logger.info(f"Set matching function to {matching_fn_name} with cost_fn={matching_fn_cfg['cost_fn']} and scale={self._scale}")

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
        frames = torch.from_numpy(np.array(self.model.replay_buffer.render_arrays))

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
        n_eval_episodes: int = 1,
        deterministic: bool = True,
        goal_seq_name: str = "",
        threshold: float = 0.5,
        use_geom_xpos: bool = False,
        seq_name: str = "",
        matching_fn_cfg: dict = {}, 
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
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic
        self._rollout_save_path = rollout_save_path  # Save the state of the environment
        self._use_geom_xpos = use_geom_xpos
        self._threshold = threshold

        # TODO: Figure out how to calculate the joint matching reward for sequence following tasks
        #   For now, we will just support calculating the ground-truth reward for matching the goal joint state
        self._goal_ref_seq = custom_ot.load_reference_seq(goal_seq_name, use_geom_xpos)
        logger.info(f"[VideoRecorderCallback] Loaded reference sequence. seq_name={seq_name}, use_geom_xpos={use_geom_xpos}")

        self.set_ground_truth_goal_matching_fn(goal_seq_name, use_geom_xpos)

        if matching_fn_cfg != {}:
            self._ref_seq = custom_ot.load_reference_seq(seq_name, use_geom_xpos)
            self._calc_matching_reward = True
            self._scale = matching_fn_cfg['scale']
            self.set_matching_fn(matching_fn_cfg)
        else:
            self._calc_matching_reward = False

        # TODO: We can potentially also do VLM reward calculation

    def set_ground_truth_goal_matching_fn(self, goal_seq_name: str, use_geom_xpos: bool):
        """Set the ground-truth goal matching function based on the goal_seq_name.

        This will be unifying metric that we measure the performance of different methods against.

        The function will return an reward array of size (n_timesteps,) where each element is the reward for the corresponding timestep.
        """
        if "final_only" in goal_seq_name:
            logger.info(f"The ground-truth reward will be calculated based on the final joint state only: {goal_seq_name}")

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
                reward_matrix = np.exp(-custom_ot.euclidean_distance_advanced(rollout, ref))

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


    def set_matching_fn(self, matching_fn_cfg):
        assert "joint_wasserstein" == matching_fn_cfg["name"] or "joint_soft_dtw", f"Currently only supporting joint_wasserstein or joint soft dynamic time warping, got {matching_fn_cfg['name']}"
        logger.info(f"[VideoRecorderCallback] Using the following reward model:\n{matching_fn_cfg}")
        
        matching_fn_name = matching_fn_cfg["name"]

        if matching_fn_name == "joint_wasserstein":
            self._matching_fn = lambda rollout, ref: custom_ot.compute_ot_reward(rollout, ref, custom_ot.COST_FN_DICT[matching_fn_cfg['cost_fn']], self._scale, modification_dict=dict(matching_fn_cfg['modification']))
        elif matching_fn_name == "joint_soft_dtw":
            self._matching_fn = lambda rollout, ref: custom_sdtw.compute_soft_dtw_reward(rollout, ref, custom_ot.COST_FN_DICT[matching_fn_cfg['cost_fn']], matching_fn_cfg['gamma'], self._scale, modification_dict=dict(matching_fn_cfg['modification']))

        logger.info(f"Set matching function to {matching_fn_name} with cost_fn={matching_fn_cfg['cost_fn']} and scale={self._scale}")


    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            raw_screens = []
            screens = []
            states = []
            geom_xposes = []
            infos = []
            rewards = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in
                the captured `screens` list

                :param _locals: A dictionary containing all local variables of the
                 callback's scope
                :param _globals: A dictionary containing all global variables of the
                 callback's scope
                """
                screen = self._eval_env.render()

                image_int = np.uint8(screen)

                raw_screens.append(Image.fromarray(image_int))
                screens.append(Image.fromarray(image_int))  # The frames here will get plotted with info later
                infos.append(_locals.get('info', {}))
                states.append(_locals["observations"])
                rewards.append(_locals["rewards"])

                geom_xpos = _locals.get('info', {})["geom_xpos"]

                # Normalize the joint states based on the torso (index 1)
                geom_xpos = geom_xpos - geom_xpos[1]
                geom_xposes.append(geom_xpos)

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )

            # Save the raw_screens locally
            imageio.mimsave(os.path.join(self._rollout_save_path, f"{self.num_timesteps}_rollouts.gif"), raw_screens, duration=1/30, loop=0)

            # Originally, states is a list of np.arrays size (1, env_obs_size)
            #   We want to concatenate them to get a single np.array size (n_timesteps, env_obs_size)
            states = np.concatenate(states)
            rewards = np.concatenate(rewards)

            # Calculate the goal matching reward
            if self._use_geom_xpos:
                # Don't need to do anything here, geom_xpos is getting normalized in the grab_screens function
                goal_matching_reward = self._gt_goal_matching_fn(np.array(geom_xposes))
            else:
                goal_matching_reward = self._gt_goal_matching_fn(np.array(states)[:, :22])

            for i in range(len(infos)):
                infos[i]["gt_joint_match_r"] = f"{goal_matching_reward[i]:.4f}"

            with open(os.path.join(self._rollout_save_path, f"{self.num_timesteps}_goal_matching_reward.npy"), "wb") as f:
                np.save(f, np.array(goal_matching_reward))
            self.logger.record("rollout/sum_total_reward_per_epsisode", 
                                np.sum(goal_matching_reward), 
                                exclude=("stdout", "log", "json", "csv"))

            # TODO: We can potentially also do VLM reward calculation
            if self._calc_matching_reward:
                if self._use_geom_xpos:
                    # Don't need to do anything here, geom_xpos is getting normalized in the grab_screens function
                    matching_reward, _ = self._matching_fn(np.array(geom_xposes), self._ref_seq)
                else:
                    matching_reward, _ = self._matching_fn(np.array(states)[:, :22], self._ref_seq)

                self.logger.record("rollout/avg_matching_reward_unscaled", 
                                np.mean(matching_reward)/self._scale, 
                                exclude=("stdout", "log", "json", "csv"))
                
                self.logger.record("rollout/avg_total_reward_unscaled", 
                                np.mean(matching_reward/self._scale + rewards), 
                                exclude=("stdout", "log", "json", "csv"))
                
                self.logger.record("rollout/avg_total_reward", 
                                np.mean(matching_reward + rewards), 
                                exclude=("stdout", "log", "json", "csv"))

                # Add the matching_reward to the infos so that we can plot it
                for i in range(len(infos)):
                    infos[i]["matching_reward"] = f"{matching_reward[i]:.2f}"

                # Save the matching_rewards locally    
                with open(os.path.join(self._rollout_save_path, f"{self.num_timesteps}_rollouts_matching_rewards.npy"), "wb") as f:
                    np.save(f, np.array(states))

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