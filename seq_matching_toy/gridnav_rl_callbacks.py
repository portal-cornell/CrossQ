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

from vlm_reward.utils.optimal_transport import COST_FN_DICT, compute_ot_reward
from vlm_reward.utils.soft_dtw import compute_soft_dtw_reward
from vlm_reward.utils.dtw import compute_dtw_reward

def set_matching_fn(fn_config, cost_fn_name="nav_manhattan"):
    assert  fn_config["name"] in ["optimal_transport", "dtw", "soft_dtw"], f"Currently only supporting ['optimal_transport', 'dtw', 'soft_dtw'], got {fn_config['name']}"
    logger.info(f"[GridNavSeqRewardCallback] Using the following reward model:\n{fn_config}")

    cost_fn = COST_FN_DICT[cost_fn_name]
    scale = float(fn_config["scale"])
    fn_name = fn_config["name"]

    if fn_name == "optimal_transport":
        return lambda obs_seq, ref_seq: compute_ot_reward(obs_seq, ref_seq, cost_fn, scale)
    elif fn_name == "dtw":
        return lambda obs_seq, ref_seq: compute_dtw_reward(obs_seq, ref_seq, cost_fn, scale)
    elif fn_name == "soft_dtw":
        gamma = float(fn_config["gamma"])
        return lambda obs_seq, ref_seq, cost_fn=cost_fn, gamma=gamma, scale=scale: compute_soft_dtw_reward(obs_seq, ref_seq, cost_fn, gamma, scale)
    else:
        raise NotImplementedError(f"Unknown sequence matching function: {fn_name}")

class GridNavSeqRewardCallback(BaseCallback):
    """
    Custom callback for calculating seq matching reward in the GridNavigation environment.
    """
    def __init__(self, ref_seq, matching_fn_cfg, cost_fn_name, verbose=0):
        super(GridNavSeqRewardCallback, self).__init__(verbose)

        self._ref_seq = ref_seq

        logger.info(f"[GridNavSeqRewardCallback] Loaded reference sequence. self._ref_seq.shape={self._ref_seq.shape} self._ref_seq=\n{self._ref_seq}")

        self._matching_fn = set_matching_fn(matching_fn_cfg, cost_fn_name)

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

        logger.debug(f"\nreplay_buffer_pos={replay_buffer_pos}, total_timesteps={total_timesteps}, \nenv_episode_timesteps={env_episode_timesteps}, self.model.num_timesteps={self.model.num_timesteps}")

        # Get the observation from the replay buffer
        #   size: (train_freq, n_envs, obs_size)
        #   For OT-based reward, train_freq = episode_length
        obs_to_process = np.array(self.model.replay_buffer.frames)

        logger.debug(f"obs_to_process={obs_to_process.shape}")

        matching_reward_list = []
        for env_i in range(self.model.env.num_envs):
            obs = obs_to_process[:, env_i]

            logger.debug(f"obs={obs.shape}")
            
            matching_reward, _ = self._matching_fn(obs, self._ref_seq)  # size: (train_freq,)

            logger.debug(f"matching_reward={matching_reward.shape}")

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

        print(f"GridNavSeqRewardCallback took {time.time() - start_time} seconds")


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


class GridNavVideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        eval_env: gymnasium.Env,
        rollout_save_path: str,
        render_freq: int,
        n_eval_episodes: int = 1,
        deterministic: bool = True,
        ref_seq: str = "",
        matching_fn_cfg: dict = {}, 
        cost_fn_name: str = "nav_manhattan",
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

        self._ref_seq = ref_seq
        logger.info(f"[GridNavVideoRecorderCallback] Loaded reference sequence. self._ref_seq.shape={self._ref_seq.shape} self._ref_seq=\n{self._ref_seq}")

        if matching_fn_cfg != {}:
            self._calc_matching_reward = True
            self._matching_fn = set_matching_fn(matching_fn_cfg, cost_fn_name)
        else:
            self._calc_matching_reward = False


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

            if self._calc_matching_reward:
                if self._use_geom_xpos:
                    # Don't need to do anything here, geom_xpos is getting normalized in the grab_screens function
                    matching_reward, _ = self._matching_fn(np.array(geom_xposes), self._ref_seq)
                else:
                    matching_reward, _ = self._matching_fn(np.array(states)[:, :22], self._ref_seq)
                matching_reward = self._matching_fn(np.array(states), self._ref_seq)

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