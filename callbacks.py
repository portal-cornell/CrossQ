import os
from typing import Any, Dict, Optional

from scipy.spatial.distance import cdist
import ot
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

class OTRewardCallback(BaseCallback):
    """
    Custom callback for calculating Optimal Transport (OT) rewards after rollouts are collected.
    """
    SEQ_DICT = {
        "arms_up_then_down": ["create_demo/demos/left-arm-out_joint-state.npy", "create_demo/demos/both-arms-out_joint-state.npy", "create_demo/demos/right-arm-out_joint-state.npy"],
    }

    def __init__(self, seq_name, cost_fn_type="cosine", verbose=0):
        super(OTRewardCallback, self).__init__(verbose)

        COST_FN_DICT = {
            "cosine": OTRewardCallback.cosine_distance,
            "euclidean": OTRewardCallback.euclidean_distance,
        }

        self._ref_seq = self.load_reference_seq(seq_name)

        self._cost_fn = COST_FN_DICT[cost_fn_type]

    def load_reference_seq(self, seq_name: str) -> np.ndarray:
        """
        Load the reference sequence for the given sequence name
        """
        ref_seq = []
        for joint in self.SEQ_DICT[seq_name]:
            ref_seq.append(np.load(joint))
        return np.stack(ref_seq)

    def _compute_ot_reward(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute the Optimal Transport (OT) reward between the reference sequence and the observed sequence

        Parameters:
            obs: np.ndarray
                The observed sequence of joint states
                size: (train_freq, 22)
                    For OT-based reward, train_freq == episode_length
                    22 is the observation size that we want to calculate
        """
        # Calculate the cost matrix between the reference sequence and the observed sequence
        #   size: (train_freq, ref_seq_len)
        cost_matrix = self._cost_fn(obs, self._ref_seq)

        # Calculate the OT plan between the reference sequence and the observed sequence
        obs_weight = np.ones(obs.shape[0]) / obs.shape[0]
        ref_weight = np.ones(self._ref_seq.shape[0]) / self._ref_seq.shape[0]
        T = ot.sinkhorn(obs_weight, ref_weight, cost_matrix, reg=0.01, log=False)  # size: (train_freq, ref_seq_len)

        # Calculate the OT reward for each timestep
        #   sum by row of (cost matrix * OT plan)
        ot_reward = np.sum(cost_matrix * T, axis=1)  # size: (train_freq,)

        return ot_reward


    def on_rollout_end(self) -> None:
        """
        This method is called after the rollout ends.
        You can access and modify the rewards in the ReplayBuffer here.
        """
        replay_buffer_pos = self.model.replay_buffer.pos
        total_timesteps = self.model.num_timesteps - self.model.previous_num_timesteps  # Total number of timesteps that we have collected
        env_episode_timesteps = total_timesteps // self.model.env.num_envs  # Number of timesteps that we have collected per environment

        logger.debug(f"\nreplay_buffer_pos={replay_buffer_pos}, total_timesteps={total_timesteps}, \nenv_episode_timesteps={env_episode_timesteps}, self.model.num_timesteps={self.model.num_timesteps}")

        # Get the observation from the replay buffer
        #   size: (train_freq, n_envs, obs_size)
        #   For OT-based reward, train_freq = episode_length
        if replay_buffer_pos - env_episode_timesteps >= 0:
            # logger.debug(f"not circular, check replay buffer: {self.model.replay_buffer.observations[replay_buffer_pos - env_episode_timesteps : replay_buffer_pos, :].shape}")

            obs_to_process = np.array(self.model.replay_buffer.observations[replay_buffer_pos - env_episode_timesteps : replay_buffer_pos, :])
        else:
            # Split reward assignment (circular buffer)
            logger.debug(f"\ncircular, part 1={self.model.replay_buffer.observations[-(env_episode_timesteps - replay_buffer_pos) :, :].shape} \n part 2={self.model.replay_buffer.observations[:replay_buffer_pos, :].shape}")

            obs_to_process = np.concatenate((self.model.replay_buffer.observations[-(env_episode_timesteps - replay_buffer_pos) :, :], self.model.replay_buffer.observations[:replay_buffer_pos, :]), axis=0)

        ot_reward_list = []
        for env_i in range(self.model.env.num_envs):
            # TODO: A hard-coded value (22 is matching qpos of the environment)
            obs = obs_to_process[:, env_i, :22]  # size: (train_freq, 22)
            ot_reward = self._compute_ot_reward(obs)  # size: (train_freq,)
            logger.debug(f"ot_reward={ot_reward.shape}")
            ot_reward_list.append(ot_reward)

        rewards = np.stack(ot_reward_list, axis=1)  # size: (train_freq, n_envs)

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


    def _on_step(self) -> bool:
        """
        Just need to define this method to avoid NotImplementedError

        Return: 
            If the callback returns False, training is aborted early.
        """
        return True
    
    @classmethod
    def cosine_distance(cls, x, y):
        distance = np.dot(x, y.T) / np.linalg.norm(x, axis=1, keepdims=True) / np.linalg.norm(y.T, axis=0, keepdims=True) # Transpose B to match dimensions

        # Rescale to be between 0 and 1
        distance_rescaled = (distance + 1) / 2
        return 1 - distance_rescaled
    
    @classmethod
    def euclidean_distance(cls, x, y):
        return cdist(x, y, metric="euclidean")


def plot_info_on_frame(pil_image, info, font_size=20):
    # TODO: this is a hard-coded path
    font = ImageFont.truetype("/share/portal/hw575/vlmrm/src/vlmrm/cli/arial.ttf", font_size)
    draw = ImageDraw.Draw(pil_image)

    x = font_size  # X position of the text
    y = pil_image.height - font_size  # Beginning of the y position of the text
    
    i = 0
    for k in info:
        # TODO: This is pretty ugly
        if not any([text in k for text in ["TimeLimit", "render_array"]]):
            reward_text = f"{k}:{info[k]}"
            # Plot the text from bottom to top
            text_position = (x, y - 30*(i+1))
            draw.text(text_position, reward_text, fill=(255, 255, 255), font=font)
        i += 1


class VideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        eval_env: gymnasium.Env,
        render_freq: int,
        n_eval_episodes: int = 1,
        deterministic: bool = True,
    ):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to
        TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the
         callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

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

                image = np.uint8(screen)
                pil_image = Image.fromarray(image)
                info = _locals.get('info', {})

                plot_info_on_frame(pil_image, info)

                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(np.uint8(pil_image).transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )

            self.logger.record(
                "trajectory/video",
                Video(th.ByteTensor(array([screens])), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )

        return True



class DinoVideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        model,
        eval_env: gymnasium.Env,
        render_freq: int,
        n_eval_episodes: int = 1,
        deterministic: bool = True,
    ):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to
        TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the
         callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

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

                image = np.uint8(screen)
                pil_image = Image.fromarray(image)
                info = _locals.get('info', {})

               # plot_info_on_frame(pil_image, info)

                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(np.uint8(pil_image).transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )

            self.logger.record(
                "trajectory/video",
                Video(th.ByteTensor(array([screens])), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )
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


