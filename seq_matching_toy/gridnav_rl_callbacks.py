import os
from typing import Any, Dict, Optional
import imageio
import gymnasium
import wandb
import torch as th
import numpy as np
from numpy import array
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
from stable_baselines3.common.logger import Image as WandbImage
from wandb.integration.sb3 import WandbCallback as SB3WandbCallback

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from loguru import logger

import time

from seq_matching_toy.run_seq_matching_on_examples import plot_matrix_as_heatmap_on_ax
from seq_matching_toy.seq_utils import get_matching_fn, update_location, render_map_and_agent
    
def convert_obs_to_frames(map_array, obs):
    """
    Frames is represented as map_array with the agent's position marked by 1

    obs represents the agent's (x, y) position in the grid

    Parameters:
        map_array: np.ndarray (row_size, col_size)
        obs: np.ndarray (n_timesteps, 2)

    Returns:
        frames: np.ndarray (n_timesteps, row_size, col_size)
    """
    frames = []
    for i in range(len(obs)):
        frame = np.copy(map_array)
        frame[int(obs[i][0]), int(obs[i][1])] = 1
        frames.append(frame)

    return np.array(frames)

class GridNavSeqRewardCallback(BaseCallback):
    """
    Custom callback for calculating seq matching reward in the GridNavigation environment.
    """
    def __init__(self, map_array, ref_seq, matching_fn_cfg, cost_fn_name, verbose=0):
        super(GridNavSeqRewardCallback, self).__init__(verbose)

        self._map = map_array

        self._ref_seq = ref_seq

        logger.info(f"[GridNavSeqRewardCallback] Loaded reference sequence. self._ref_seq.shape={self._ref_seq.shape} self._ref_seq=\n{self._ref_seq}")

        self._matching_fn, self._matching_fn_name = get_matching_fn(matching_fn_cfg, cost_fn_name)

    def on_rollout_end(self) -> None:
        """
        This method is called after the rollout ends.
        You can access and modify the rewards in the ReplayBuffer here.
        """
        # Time this function
        start_time = time.time()

        matching_reward_list = []
        for env_i in range(self.model.env.num_envs):
            obs_to_use = self.model.rollout_buffer.observations[1:, env_i]  # Skip the first observation because we are calculating the reward based on what it looks like in the next state
            final_obs = update_location(agent_pos=obs_to_use[-1].astype(np.int64), action=int(self.model.rollout_buffer.actions[-1, env_i]), map_array=self._map)
            obs_to_use = np.concatenate([obs_to_use, np.expand_dims(final_obs, 0)], axis=0)
            frames = convert_obs_to_frames(self._map, obs_to_use)
            
            matching_reward, _ = self._matching_fn(frames, self._ref_seq)  # size: (n_steps,)

            matching_reward_list.append(matching_reward)

            # for i in range(len(frames)):
            #     print(f"[{i}] action={self.model.rollout_buffer.actions[i, env_i]} matching_reward={matching_reward[i]}")
            #     if i > 0:
            #         print(f"before:\n{frames[i-1]}")
            #     print(f"after:\n{frames[i]}")
            #     input("stop")

        rewards = np.stack(matching_reward_list, axis=1)  # size: (n_steps, n_envs)

        self.model.rollout_buffer.rewards += rewards

        # print(f"GridNavSeqRewardCallback took {time.time() - start_time} seconds")

    
    def on_rollout_end_no_buffer(self, states, actions, rewards):
        obs_to_use = states[1:]  # Skip the first observation because we are calculating the reward based on what it looks like in the next state
        final_obs = update_location(agent_pos=obs_to_use[-1].astype(np.int64), action=int(actions[-1]), map_array=self._map)
        obs_to_use = np.concatenate([obs_to_use, np.expand_dims(final_obs, 0)], axis=0)
        frames = convert_obs_to_frames(self._map, obs_to_use)
        
        matching_rewards, _ = self._matching_fn(frames, self._ref_seq)  # size: (n_steps,)

        return rewards + matching_rewards


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
        if not any([text in k for text in ["TimeLimit", "render_array", "geom_xpos", "episode"]]):
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
        map_array: np.ndarray = None,
        ref_seq: str = "",
        matching_fn_cfg: dict = {}, 
        cost_fn_name: str = "nav_manhattan",
        reward_vmin: int = 0, 
        reward_vmax: int = 0, 
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

        self._map = map_array
        self._ref_seq = ref_seq
        logger.info(f"[GridNavVideoRecorderCallback] Loaded reference sequence. self._ref_seq.shape={self._ref_seq.shape} self._ref_seq=\n{self._ref_seq}")

        self._reward_vmin = reward_vmin
        self._reward_vmax = reward_vmax

        self._gt_reward_fn = self.set_ground_truth_fn()

        if matching_fn_cfg != {}:
            self._calc_matching_reward = True
            self._matching_fn, self._matching_fn_name = get_matching_fn(matching_fn_cfg, cost_fn_name)
        else:
            self._calc_matching_reward = False

    def set_ground_truth_fn(self):
        """
        Set the ground truth function for the matching function
        """
        def nav_key_point_following(obs_seq, ref_seq):
            """
            Counting the number of key points that the agent has followed
            """
            score = 0
            j = 0

            score_at_each_timestep = []

            for i in range(len(obs_seq)):
                if j < len(ref_seq):
                    if np.array_equal(obs_seq[i], ref_seq[j]):
                        score += 1
                        j += 1

                score_at_each_timestep.append(score/len(ref_seq))
            
            return score_at_each_timestep
        
        return nav_key_point_following

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            raw_screens = []
            screens = []
            states = []
            infos = []
            rewards = []
            actions = []

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
                actions.append(_locals["actions"])

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )

            # Originally, states is a list of np.arrays size (1, env_obs_size)
            #   We want to concatenate them to get a single np.array size (n_timesteps, env_obs_size)
            states = np.concatenate(states)
            actions = np.concatenate(actions)

            obs_to_use = states[1:]  # Skip the first observation because we are calculating the reward based on what it looks like in the next state
            final_obs = update_location(agent_pos=obs_to_use[-1].astype(np.int64), action=int(actions[-1]), map_array=self._map)
            obs_to_use = np.concatenate([obs_to_use, np.expand_dims(final_obs, 0)], axis=0)
            obs_seq = convert_obs_to_frames(self._map, obs_to_use)

            # Add the first frame and replace final frame to the screens
            raw_screens = [Image.fromarray(np.uint8(render_map_and_agent(self._map, states[0].astype(np.int64))))] + raw_screens
            raw_screens[-1] = Image.fromarray(np.uint8(render_map_and_agent(self._map, final_obs)))
            screens = [Image.fromarray(np.uint8(render_map_and_agent(self._map, states[0].astype(np.int64))))] + screens
            screens[-1] = Image.fromarray(np.uint8(render_map_and_agent(self._map, final_obs)))

            # Save the raw_screens locally
            imageio.mimsave(os.path.join(self._rollout_save_path, f"{self.num_timesteps}_rollouts.gif"), raw_screens, duration=1/30, loop=0)
            
            gt_rewards = self._gt_reward_fn(obs_seq, self._ref_seq)

            for i in range(len(infos)):
                infos[i]["gt_r"] = f"{gt_rewards[i]:.4f}"

            self.logger.record("rollout/mean_gt_reward_per_epsisode", 
                                np.mean(gt_rewards), 
                                exclude=("stdout", "log", "json", "csv"))

            if self._calc_matching_reward:
                matching_reward, info = self._matching_fn(obs_seq, self._ref_seq)  # size: (n_steps,)

                self.logger.record("rollout/avg_total_reward", 
                                np.mean(matching_reward + rewards), 
                                exclude=("stdout", "log", "json", "csv"))

                # Add the matching_reward to the infos so that we can plot it
                for i in range(len(infos)):
                    infos[i]["matching_reward"] = f"{matching_reward[i]:.2f}"

                # Save the matching_rewards locally    
                with open(os.path.join(self._rollout_save_path, f"{self.num_timesteps}_rollouts_matching_rewards.npy"), "wb") as f:
                    np.save(f, np.array(states))

                # TODO: messy code taken from seq_matching_toy/run_seq_matching_on_examples.py
                #   Basically allows us to visualizing the matching function on a rollout
                rolcol_size = 1

                # 3 * because we have 3 figure columns
                #   In each figure columns, we have len(ref_seq) for the reference sequence/cost matrix, 1 column for the vertical stack of obs seq, and 1 column for the colorbar
                fig_width = 3 * (rolcol_size * (len(self._ref_seq) + 2))

                #  We have len(obs_seq) for the observed sequence/cost matrix, 1 row for the horizontal stack of ref seq
                fig_height = rolcol_size * (len(obs_seq) + 1)

                # Create the figure (2 columns, and the number of rows will be the number of sequence matching algorithms)
                fig, axs = plt.subplots(1, 3, figsize=(fig_width, fig_height))

                # Plot the cost matrix
                ax = axs[0]
                plot_matrix_as_heatmap_on_ax(ax, fig, obs_seq, self._ref_seq, info["cost_matrix"], f"{self._matching_fn_name} Cost Matrix", cmap="gray_r", rolcol_size=rolcol_size)

                # Plot the assignment matrix
                ax = axs[1]
                
                plot_matrix_as_heatmap_on_ax(ax, fig, obs_seq, self._ref_seq, info["assignment"], f"{self._matching_fn_name} Assignment Matrix", cmap="Greens", rolcol_size=rolcol_size, vmin=0, vmax=1)

                # Plot the reward
                ax = axs[2]
                plot_matrix_as_heatmap_on_ax(ax, fig, obs_seq, self._ref_seq, np.expand_dims(matching_reward,1), f"{self._matching_fn_name} Reward (Sum = {np.sum(matching_reward):.2f})", cmap="Greens", rolcol_size=rolcol_size,  
                                            vmin=self._reward_vmin, vmax=self._reward_vmax)
                

                plt.tight_layout()

                img_path = os.path.join(self._rollout_save_path, f"{self.num_timesteps}_matching_fn_viz.png")
                plt.savefig(img_path)

                img = Image.open(img_path)

                # Log to wandb
                self.logger.record(
                    "trajectory/matching_fn_viz",
                    WandbImage(np.array(img), "HWC"),
                    exclude=("stdout", "log", "json", "csv"),
                )

                plt.close(fig)

            # Plot info on the frames  
            for i in range(1, len(screens)):
                plot_info_on_frame(screens[i], infos[i-1])

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
            
            with open(os.path.join(self._rollout_save_path, f"{self.num_timesteps}_rollouts_rewards.npy"), "wb") as f:
                np.save(f, np.array(rewards))

        return True


    def _on_step_no_buffer(self, raw_screens, screens, states, actions, rewards, infos, num_timesteps) -> bool:
        obs_to_use = states[1:]  # Skip the first observation because we are calculating the reward based on what it looks like in the next state
        final_obs = update_location(agent_pos=obs_to_use[-1].astype(np.int64), action=int(actions[-1]), map_array=self._map)
        obs_to_use = np.concatenate([obs_to_use, np.expand_dims(final_obs, 0)], axis=0)
        obs_seq = convert_obs_to_frames(self._map, obs_to_use)

        # Add the first frame and replace final frame to the screens
        raw_screens = [Image.fromarray(np.uint8(render_map_and_agent(self._map, states[0].astype(np.int64))))] + raw_screens
        raw_screens[-1] = Image.fromarray(np.uint8(render_map_and_agent(self._map, final_obs)))
        screens = [Image.fromarray(np.uint8(render_map_and_agent(self._map, states[0].astype(np.int64))))] + screens
        screens[-1] = Image.fromarray(np.uint8(render_map_and_agent(self._map, final_obs)))

        # Save the raw_screens locally
        imageio.mimsave(os.path.join(self._rollout_save_path, f"{num_timesteps}_rollouts.gif"), raw_screens, duration=1/30, loop=0)
        
        gt_rewards = self._gt_reward_fn(obs_seq, self._ref_seq)

        for i in range(len(infos)):
            infos[i]["gt_r"] = f"{gt_rewards[i]:.4f}"
        
        log_dict = {"rollout/mean_gt_reward_per_epsisode": np.mean(gt_rewards)}

        if self._calc_matching_reward:
            matching_reward, info = self._matching_fn(obs_seq, self._ref_seq)  # size: (n_steps,)

            log_dict["rollout/avg_total_reward"] = np.mean(matching_reward + rewards)
            # Add the matching_reward to the infos so that we can plot it
            for i in range(len(infos)):
                infos[i]["matching_reward"] = f"{matching_reward[i]:.2f}"

            # Save the matching_rewards locally    
            with open(os.path.join(self._rollout_save_path, f"{num_timesteps}_rollouts_matching_rewards.npy"), "wb") as f:
                np.save(f, np.array(states))

            # TODO: messy code taken from seq_matching_toy/run_seq_matching_on_examples.py
            #   Basically allows us to visualizing the matching function on a rollout
            rolcol_size = 1

            # 3 * because we have 3 figure columns
            #   In each figure columns, we have len(ref_seq) for the reference sequence/cost matrix, 1 column for the vertical stack of obs seq, and 1 column for the colorbar
            fig_width = 3 * (rolcol_size * (len(self._ref_seq) + 2))

            #  We have len(obs_seq) for the observed sequence/cost matrix, 1 row for the horizontal stack of ref seq
            fig_height = rolcol_size * (len(obs_seq) + 1)

            # Create the figure (2 columns, and the number of rows will be the number of sequence matching algorithms)
            fig, axs = plt.subplots(1, 3, figsize=(fig_width, fig_height))

            # Plot the cost matrix
            ax = axs[0]
            plot_matrix_as_heatmap_on_ax(ax, fig, obs_seq, self._ref_seq, info["cost_matrix"], f"{self._matching_fn_name} Cost Matrix", cmap="gray_r", rolcol_size=rolcol_size)

            # Plot the assignment matrix
            ax = axs[1]
            
            plot_matrix_as_heatmap_on_ax(ax, fig, obs_seq, self._ref_seq, info["assignment"], f"{self._matching_fn_name} Assignment Matrix", cmap="Greens", rolcol_size=rolcol_size, vmin=0, vmax=1)

            # Plot the reward
            ax = axs[2]
            plot_matrix_as_heatmap_on_ax(ax, fig, obs_seq, self._ref_seq, np.expand_dims(matching_reward,1), f"{self._matching_fn_name} Reward (Sum = {np.sum(matching_reward):.2f})", cmap="Greens", rolcol_size=rolcol_size,  
                                        vmin=self._reward_vmin, vmax=self._reward_vmax)
            

            plt.tight_layout()

            img_path = os.path.join(self._rollout_save_path, f"{num_timesteps}_matching_fn_viz.png")
            plt.savefig(img_path)

            img = Image.open(img_path)

            # Log to wandb
            log_dict["trajectory/matching_fn_viz"] = wandb.Image(np.array(img))

            plt.close(fig)

        # Plot info on the frames  
        for i in range(1, len(screens)):
            plot_info_on_frame(screens[i], infos[i-1])

        screens = [np.uint8(s).transpose(2, 0, 1) for s in screens]

        # Log to wandb
        log_dict["trajectory/video"] = wandb.Video(th.ByteTensor(array([screens])), fps=40)

        # Save the rollouts locally    
        with open(os.path.join(self._rollout_save_path, f"{num_timesteps}_rollouts_states.npy"), "wb") as f:
            np.save(f, np.array(states))
        
        with open(os.path.join(self._rollout_save_path, f"{num_timesteps}_rollouts_rewards.npy"), "wb") as f:
            np.save(f, np.array(rewards))

        wandb.log(log_dict, step=num_timesteps)

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