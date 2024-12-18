import os
import time
import json
from typing import Any, Dict, Optional
import imageio
import gymnasium
import torch as th
from torchvision import transforms
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
from seq_reward.cost_fns import euclidean_distance_advanced, euclidean_distance_advanced_arms_only, COST_FN_DICT

from vlm_reward.vlm_buffer import GeomXposReplayBuffer
from constants import HUMANOID_TASK_SEQ_DICT
from utils import calc_iqm

from torchvision.utils import save_image

from vlm_reward.reward_models.resnet import load_resnet50_backbone

def run_model_on_batch(frames, model, batch_size):
    results = []

    for batch in th.split(frames, batch_size):
        with th.no_grad():
            result = model(batch).squeeze()
            if len(result.shape) < 2:
                results = result[None]
        results.append(result)
    return th.cat(results)

class SeqRewardCallback(BaseCallback):
    """
    Custom callback for calculating state based sequence matching rewards after rollouts are collected.
    """
    def __init__(self, env_name, matching_fn_cfg, verbose=0, **kwargs):
        """
        Parameters:
            env_name: str
                The name of the environment
            matching_fn_cfg: dict
                The configuration for the matching function
            use_geom_xpos: bool
                Whether to use geom_xpos for the observation
            verbose: int
                The verbosity level
        """
        super(SeqRewardCallback, self).__init__(verbose)

        self.matching_fn, self.matching_fn_name = get_matching_fn(matching_fn_cfg, matching_fn_cfg["cost_fn"])
        logger.info(f"[SeqRewardCallback] Loaded matching fn {self.matching_fn_name} with {matching_fn_cfg}")

    def add_to_buffer_rewards(self, seq_matching_rewards):
        """add_to the calculated sequence matching reward to the replay buffer

        Parameters:
            seq_matching_rewards: np.array
                The sequence matching reward to put in the replay buffer
                Shape: (train_freq, n_envs)

        Effects:
            The rewards in the replay buffer are modified to be the sequence matching reward
        """
        replay_buffer_pos = self.model.replay_buffer.pos
        total_timesteps = self.model.num_timesteps - self.model.previous_num_timesteps  # Total number of timesteps that we have collected
        env_episode_timesteps = total_timesteps // self.model.env.num_envs  # Number of timesteps that we have collected per environment

        if replay_buffer_pos - env_episode_timesteps >= 0:
            self.model.replay_buffer.rewards[
                replay_buffer_pos - env_episode_timesteps : replay_buffer_pos, :
            ] += seq_matching_rewards[:, :]
        else:
            # Split reward assignment (circular buffer)
            self.model.replay_buffer.rewards[
                -(env_episode_timesteps - replay_buffer_pos) :, :
            ] += seq_matching_rewards[: env_episode_timesteps - replay_buffer_pos, :]

            self.model.replay_buffer.rewards[:replay_buffer_pos, :] += seq_matching_rewards[
                env_episode_timesteps - replay_buffer_pos :, :
            ]

class StateBasedSeqRewardCallback(SeqRewardCallback):
    """
    Custom callback for calculating state based sequence matching rewards after rollouts are collected.
    """
    def __init__(self, env_name, task_name, matching_fn_cfg, verbose=0, **kwargs):
        """
        Parameters:
            env_name: str
                The name of the environment
            task_name: str
                The name of the task in the environment
            matching_fn_cfg: dict
                The configuration for the matching function
            use_geom_xpos: bool
                Whether to use geom_xpos for the observation
            verbose: int
                The verbosity level
        """
        super(StateBasedSeqRewardCallback, self).__init__(env_name, matching_fn_cfg, verbose=0, **kwargs)

        self.task_name = task_name
        self.env_name = env_name
        self.env_kwargs = kwargs
        self.seq_name = matching_fn_cfg["seq_name"]

        self._ref_seq = load_reference_seq(env_name=env_name, task_name=task_name, seq_name=matching_fn_cfg["seq_name"], load_visual=False, use_geom_xpos=kwargs.get('use_geom_xpos', False))
        logger.info(f"[StateBasedSeqRewardCallback] Loaded reference sequence. env_name={env_name}, task_name={task_name}, seq_name={matching_fn_cfg['seq_name']}, self._ref_seq.shape={self._ref_seq.shape}")

    def on_rollout_end(self) -> None:
        """
        This method is called after the rollout ends.
        You can access and modify the rewards in the ReplayBuffer here.

        Effect:
            The rewards in the replay buffer are modified to add the sequence matching reward
        """
        # Time this function
        start_time = time.time()

        # Get the observation from the replay buffer
        #   size: (train_freq, n_envs, obs_size)
        obs_to_process = self.get_obs_to_process_from_buffer()
        matching_reward_list = []
        # For each environment, calculate the sequence matching reward
        for env_i in range(self.model.env.num_envs):
            obs = obs_to_process[:, env_i]
            
            matching_reward, _ = self.matching_fn(obs, self._ref_seq)  # size: (train_freq,)

            matching_reward_list.append(matching_reward)

        rewards = np.stack(matching_reward_list, axis=1)  # size: (train_freq, n_envs)

        # Add the sequence matching reward to exisiting rewards
        self.add_to_buffer_rewards(rewards)

        if type(self.model.replay_buffer) == GeomXposReplayBuffer:
            self.model.replay_buffer.clear_geom_xpos()

        print(f"StateBasedSeqRewardCallback took {time.time() - start_time} seconds")

    def get_obs_to_process_from_buffer(self):
        """Get the observation from the replay buffer
        
        Returns:
            obs_to_process: np.array
                The observation to process (we will calculate the distance between these observation and the reference sequence)
                Shape: (train_freq, n_envs, obs_size), where train_freq is the number of timesteps in the episode
        """
        replay_buffer_pos = self.model.replay_buffer.pos
        total_timesteps = self.model.num_timesteps - self.model.previous_num_timesteps  # Total number of timesteps that we have collected
        env_episode_timesteps = total_timesteps // self.model.env.num_envs  # Number of timesteps that we have collected per environment

        if self.env_name == "HumanoidSpawnedUpCustom":
            if self.env_kwargs.get('use_geom_xpos'):
                # Because we manually stored geom_xpos in the replay buffer
                obs_to_process = np.array(self.model.replay_buffer.geom_xpos)
                # Normalize along the center of mass (index 1)
                obs_to_process = obs_to_process - obs_to_process[:, :, 1:2, :]
            else:
                # TODO: A hard-coded value (22 is matching qpos of the environment)
                if replay_buffer_pos - env_episode_timesteps >= 0:
                    obs_to_process = np.array(self.model.replay_buffer.observations[replay_buffer_pos - env_episode_timesteps : replay_buffer_pos, :22])
                else:
                    # Split reward assignment (circular buffer)
                    obs_to_process = np.concatenate((self.model.replay_buffer.observations[-(env_episode_timesteps - replay_buffer_pos) :, :22], self.model.replay_buffer.observations[:replay_buffer_pos, :22]), axis=0)
        elif self.env_name == "Metaworld":
            if replay_buffer_pos - env_episode_timesteps >= 0:
                obs_to_process = np.array(self.model.replay_buffer.observations[replay_buffer_pos - env_episode_timesteps : replay_buffer_pos, :])
            else:
                # Split reward assignment (circular buffer)
                obs_to_process = np.concatenate((self.model.replay_buffer.observations[-(env_episode_timesteps - replay_buffer_pos) :, :], self.model.replay_buffer.observations[:replay_buffer_pos, :]), axis=0)

            obs_to_process = obs_to_process[:, :, :18]  # We only want the first 18 features (which corresponds to the current state)
        else:
            raise NotImplementedError(f"env_name={self.env_name} is not supported")
        
        return obs_to_process

    def _on_step(self) -> bool:
        """
        Just need to define this method to avoid NotImplementedError

        Return: 
            If the callback returns False, training is aborted early.
        """
        return True


class VisualSeqRewardCallback(SeqRewardCallback):
    def __init__(self, env_name, task_name, matching_fn_cfg, verbose=0, device='cuda', encoder_batch_size=32, use_image_for_ref=True, **kwargs):
        super(VisualSeqRewardCallback, self).__init__(env_name, matching_fn_cfg, verbose, **kwargs)
        
        self.task_name = task_name
        self.env_name = env_name
        self.env_kwargs = kwargs
        self.seq_name = matching_fn_cfg["seq_name"]
        self.matching_fn_cfg = matching_fn_cfg
        self.device = device
        self.use_geom_xpos=kwargs.get('use_geom_xpos', False) # Only matters for humanoid environment

        self.matching_fn, self.matching_fn_name = get_matching_fn(matching_fn_cfg, matching_fn_cfg["cost_fn"])

        self.visual_encoder = load_resnet50_backbone(self.device)

        self.encoder_batch_size = encoder_batch_size
        
        self.pil_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])  
        self.torch_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])  
    
        self.use_image_for_ref = use_image_for_ref
        
    def on_training_start(self, *args, **kwargs) -> None:    
        # Wait to initialize reference until training is starting, in case inference is necessary
        self.initialize_ref()

    def initialize_ref(self):
        # Ref may be a sequence of images or states at this point
        ref = load_reference_seq(env_name=self.env_name, task_name=self.task_name, seq_name=self.seq_name, load_visual=self.use_image_for_ref, use_geom_xpos=self.use_geom_xpos)

        if self.use_image_for_ref:
            transformed_frames = [self.pil_transform(frame) for frame in ref]

            # Stack the transformed frames into a batch tensor
            frames = th.stack(transformed_frames).to(self.device)
            
            ref_seq = run_model_on_batch(frames, self.visual_encoder, self.encoder_batch_size)
            self.matching_ref_seq = ref_seq.detach().cpu().numpy()
                        
            logger.info(f"[VisualSeqRewardCallback] Loaded reference GIF sequence. env_name={self.env_name}, task_name={self.task_name}, seq_name={self.seq_name}, self._ref_seq.shape={self.matching_ref_seq.shape}")
        else:
            self.matching_ref_seq = ref
            
            logger.info(f"[VisualSeqRewardCallback] Loaded reference GROUND TRUTH sequence. env_name={self.env_name}, task_name={self.task_name}, seq_name={self.seq_name}, self._ref_seq.shape={self.matching_ref_seq.shape}")  

    def on_rollout_end(self) -> None:
        start_time = time.time()
        
        replay_buffer_pos = self.model.replay_buffer.pos
        total_timesteps = self.model.num_timesteps - self.model.previous_num_timesteps
        env_episode_timesteps = total_timesteps // self.model.env.num_envs
        
        # Get frames from replay buffer
        frames = self.get_obs_to_process_from_buffer()
        

        matching_reward_list = []
        for env_i in range(self.model.env.num_envs):            
            # IMPORTANT: we cut off the last frame because it is always the reset frame
            env_frames = frames[:-1, env_i, ...]
            env_frames_transformed = self.torch_transform(env_frames) #th.stack([self.torch_transform(frame) for frame in env_frames]).to(self.device)
            learner_embeddings = run_model_on_batch(env_frames_transformed, self.visual_encoder, self.encoder_batch_size)
            
            # Just set the values for the last frame as the same as the second to last frame (because last frame is corrupted)
            
            learner_embeddings = th.cat((learner_embeddings, learner_embeddings[-1][None]), dim=0) 

            learner_embeddings = learner_embeddings.detach().cpu().numpy()
            matching_reward, _ = self.matching_fn(learner_embeddings, self.matching_ref_seq) 
            matching_reward_list.append(matching_reward)
        
        # Clear the render arrays once computations have been run on them
        self.model.replay_buffer.clear_render_arrays()
        rewards = np.stack(matching_reward_list, axis=1)
        self.add_to_buffer_rewards(rewards)
        
        logger.debug(f"VisualBasedSeqRewardCallback took {time.time() - start_time} seconds")

    def get_obs_to_process_from_buffer(self):
        """Get the observation from the replay buffer
        
        Returns:
            obs_to_process: np.array
                The observation to process (we will calculate the distance between these observation and the reference sequence)
                Shape: (train_freq, n_envs, obs_size), where train_freq is the number of timesteps in the episode
        """
        replay_buffer_pos = self.model.replay_buffer.pos
        total_timesteps = self.model.num_timesteps - self.model.previous_num_timesteps  # Total number of timesteps that we have collected
        env_episode_timesteps = total_timesteps // self.model.env.num_envs  # Number of timesteps that we have collected per environment

        # if replay_buffer_pos - env_episode_timesteps >= 0:
        #     obs = np.array(self.model.replay_buffer.render_arrays[replay_buffer_pos - env_episode_timesteps : replay_buffer_pos, ...])
        # else:
        #     # Split reward assignment (circular buffer)
        #     obs = np.concatenate((self.model.replay_buffer.render_arrays[-(env_episode_timesteps - replay_buffer_pos):, ...], self.model.replay_buffer.render_arrays[:replay_buffer_pos, ...]), axis=0)
        # torch_obs = th.stack([self.pil_transform(img) for img in obs]).to(self.device)
        
        torch_obs = th.from_numpy(np.array(self.model.replay_buffer.render_arrays)).float().to(self.device) / 255.0
        frames = rearrange(torch_obs, "n_steps n_envs h w c -> n_steps n_envs c h w")
       
        if self.env_name.lower() == "metaworld":
            # metaworld observations are flipped depending on camera angle
            if self.model.env.camera_name == 'corner4':
                frames = th.flip(frames, [4]) # flip along horizontal
            elif self.model.env.camera_name in ['corner1', 'corner2', 'corner3']:
                frames = th.flip(frames, [3]) # flip along vertical
        return frames

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
        env_name: str = "",
        camera_name: str = "",
        task_name: str = "",
        use_geom_xpos: bool = True,
        threshold: float = 0.5,
        success_fn_cfg: dict = {},
        matching_fn_cfg: dict = {}, 
        calc_visual_reward: bool = False,
        verbose=0,
        encoder_batch_size=32,
        discount_factor=.99,
        device='cuda'
    ):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to
        TensorBoard

        Pararmeters
            eval_env: A gym environment from which the trajectory is recorded
                Assumes that there's only 1 environment
            rollout_save_path: The path to save the rollouts (states and rewards)
            render_freq: Render the agent's trajectory every eval_freq call of the callback.
            render_dim: The dimensions of the rendered frames
            n_eval_episodes: Number of episodes to render
            deterministic: Whether to use deterministic or stochastic policy
            env_name: The name of the environment
            camera_name: For Metaworld only, the name of the camera affects the transform for the image.
            task_name: The name of the task in the environment
            use_geom_xpos: Whether to use geom_xpos for the observation (only for HumanoidSpawnedUpCustom)
            threshold: The threshold to consider a success
            success_fn_cfg: The configuration for the success function
            matching_fn_cfg: The configuration for the matching function
            calc_visual_reward: Whether to calculate the visual reward using the VLM reward model
        """
        super().__init__(verbose)
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._render_dim = render_dim
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

        self._rollout_save_path = rollout_save_path  # Save the state of the environment

        self.matching_fn_cfg = matching_fn_cfg
        self.success_fn_cfg = success_fn_cfg

        self.env_name = env_name
        self.task_name = task_name
        self.seq_name = matching_fn_cfg['seq_name']
        self.calc_visual_reward = calc_visual_reward
        self.use_geom_xpos = use_geom_xpos
        self.threshold = threshold
        self.calc_visual_reward = calc_visual_reward
        self.discount_factor = discount_factor
        self.camera_name = camera_name

        if self.calc_visual_reward:
            self.device=device
            self.visual_encoder = load_resnet50_backbone(self.device)
            self.encoder_batch_size = encoder_batch_size
            
            self.pil_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])  
            self.torch_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])  
    
    def on_training_start(self, *args, **kwargs):
        """
        Effect: 
          self.calc_matching_reward (bool) - Whether to calculate the sequence matching reward
          self.matching_ref_seq (np.array) - The reference sequence that is used to calculate the sequence matching reward
          self.matching_fn (fn) - The function to calculate the sequence matching reward
        """
        self._setup_seq_matching()
        
        # Set up ground-truth metric that all the models will be compared against
        if self.env_name == "HumanoidSpawnedUpCustom":
            self._humanoid_env_setup_eval(self.task_name, self.success_fn_cfg, self.use_geom_xpos)
        elif self.env_name == "Metaworld":
            self._metaworld_env_setup_eval(self.task_name, self.success_fn_cfg)

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            # Saving for only one env (the first env)
            #   Because we are using this to plot
            raw_screens = []
            screens = []
            # Saving for each env
            states = []
            rewards = []
            geom_xposes = [[] for _ in range(self._n_eval_episodes)]
            all_infos = [[] for _ in range(self._n_eval_episodes)]

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

                    if self._env_name == "Metaworld":
                        if self._camera_name == "corner" or self._camera_name == "corner2" or self._camera_name == "corner3":
                            # For some reason, the image is flipped upside down
                            image_int = np.flipud(image_int)
                        elif self._camera_name == "corner4":
                            # For some reason, the image is flipped left-right
                            image_int = np.fliplr(image_int)

                    raw_screens.append(Image.fromarray(image_int))
                    screens.append(Image.fromarray(image_int))  # The frames here will get plotted with info later
                    
                    states.append(_locals["observations"])
                    rewards.append(_locals["rewards"])
                
                all_infos[env_i].append(_locals.get('info', {}))
                
                if self.use_geom_xpos:
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

            states = np.array(states)  # size: (rollout_length, n_eval_episodes, state_feature_size)
            rewards = np.array(rewards) # size: (rollout_length, n_eval_episodes)
            geom_xposes = np.array(geom_xposes) # If self._user_geom_xpose, size: (n_eval_episodes, rollout_length, 18, 3). Else, empty list
            
            # Calculate different rewards/metrics (and update what will be plotted on the 0th env's info)
            infos_0th_env = all_infos[0]

            if self.env_name == "HumanoidSpawnedUpCustom":
                infos_0th_env = self._calc_and_record_humanoid_gt_reward(geom_xposes, infos_0th_env)
            elif self.env_name == "Metaworld":
                self._calc_and_record_metaworld_gt_reward(states, all_infos)

            if self.calc_visual_reward:
                obs_seq = self.get_obs_embeddings_from_screens(screens)
            else:
                obs_seq = self.get_obs_clean_states(states, geom_xposes)

            infos_0th_env = self.calc_and_record_seq_matching_reward_for_0th_env(obs_seq, raw_screens, infos_0th_env)

            # Plot info on the frames  
            for i in range(len(screens)):
                plot_info_on_frame(screens[i], infos_0th_env[i])

            # Log to wandb
            self.logger.record(
                "trajectory/video",
                Video(th.ByteTensor(array([[np.uint8(s).transpose(2, 0, 1) for s in screens]])), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )

            # Save the rollouts locally    
            with open(os.path.join(self._rollout_save_path, f"{self.num_timesteps}_rollouts_states.npy"), "wb") as f:
                np.save(f, np.array(states))
                
            with open(os.path.join(self._rollout_save_path, f"{self.num_timesteps}_rollouts_rewards.npy"), "wb") as f:
                np.save(f, np.array(rewards))

            if self.use_geom_xpos:
                with open(os.path.join(self._rollout_save_path, f"{self.num_timesteps}_rollouts_geom_xpos_states.npy"), "wb") as f:
                    np.save(f, np.array(geom_xposes))

        return True

    """+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        Calculate Ground-truth Reward/Metric (e.g., success rate) that all the models get compared against

    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""
    def _calc_and_record_humanoid_gt_reward(self, geom_xposes, infos):
        """
        Parameters:
            geom_xposes: np.array
                The geom_xposes for the n_eval_episodes
                size: (n_eval_episodes, rollout_length, 18, 3)
            infos: List[Dict]
                The information for the 0th environment (later used for plotting
        
        Effects:
            - Save the success results locally
            - Log various success rate to the logger (so it can appear on wandb)

        Returns:
            infos: List[Dict]
                The information for the 0th environment (later used for plotting)
                
                It gets information about
                    - "rf_r" = reward based on the ground-truth goal reference sequence. This is used to compute the success rate
                    - "[all, arm] success" = success rate based on the entire body + based on only the arm
        """
        if self._calc_gt_reward:
            # Calculate the goal matching reward
            if self.use_geom_xpos:            
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
                raise NotImplementedError(f"Ground truth reward calculation for self.use_geom_xpos={self.use_geom_xpos} is False")

            # Plot success rate and reward information for the 0th env's rollout
            reward_matrix = np.exp(-euclidean_distance_advanced(geom_xposes[0], self._goal_ref_seq))
            arm_reward_matrix = np.exp(-euclidean_distance_advanced_arms_only(geom_xposes[0], self._goal_ref_seq))

            for i in range(len(infos)):
                # Plot the reward (exp of the negative distance) based on the ground-truth goal reference sequence
                infos[i]["rf_r"] = str([f"{reward_matrix[i][j]:.2f}" for j in range(len(self._goal_ref_seq))]) + " | " +  str([f"{arm_reward_matrix[i][j]:.2f}" for j in range(len(self._goal_ref_seq))])
                # Success Rate based on the entire body + based on only the arm
                infos[i]["[all, arm] success"] = f"{full_pos_success_rate_list[0]:.2f}, {arm_pos_success_rate_list[0]:.2f}"
        
        return infos
    

    def _calc_and_record_metaworld_gt_reward(self, states, all_infos):
        """Record Metaworld's ground-truth reward (sparse and dense) to the logger

        Parameters:
            states: observed states, of shape (episode_length, n_envs, obs_shape)
            all_infos: List[List[Dict]]
                The information for all the environment (we will extract environment's sparse and dense rewards)
        
        Effects:
            - Log the ground-truth sparse and dense reward from the environment to the logger (so it can appear on wandb)
        """
        if self._calc_gt_reward:
            env_sparse_reward_list = [[] for _ in range(self._n_eval_episodes)]
            env_dense_reward_list = [[] for _ in range(self._n_eval_episodes)]

            for i in range(self._n_eval_episodes):
                env_sparse_reward_list[i] = np.sum([float(info["success"]) for info in all_infos[i]])
                env_dense_reward_list[i] = np.sum([float(info["dense_r"]) for info in all_infos[i]])

            avg_env_sparse_reward = np.mean(env_sparse_reward_list)
            avg_env_dense_reward = np.mean(env_dense_reward_list)

            self.logger.record("eval/env_sparse_reward", 
                                    avg_env_sparse_reward, 
                                    exclude=("stdout", "log", "json", "csv"))
            
            self.logger.record("eval/env_dense_reward",
                                    avg_env_dense_reward,
                                    exclude=("stdout", "log", "json", "csv"))

            # We can only do the calculation below if we have a ref seq
            #   (see the _set_metaworld_success_fn function)
            if self._success_fn_based_on_all_pos:
                full_pos_success_rate_list = []
                full_pos_pct_success_timesteps_list = []

                for env_i in range(self._n_eval_episodes):
                    # Don't need to do anything here, geom_xpos is getting normalized in the grab_screens function
                    states_to_process = states[:, env_i, ...]

                    full_pos_success_rate, full_pos_pct_success_timesteps = self._success_fn_based_on_all_pos(states_to_process)
                    full_pos_success_rate_list.append(full_pos_success_rate)
                    full_pos_pct_success_timesteps_list.append(full_pos_pct_success_timesteps)

                full_pos_success_rate_iqm, full_pos_success_rate_std = calc_iqm(full_pos_success_rate_list)
                full_pos_pct_success_timesteps_iqm, full_pos_pct_success_timesteps_std = calc_iqm(full_pos_pct_success_timesteps_list)

                # Save the success results locally
                self.add_success_results(self.num_timesteps, {
                    "full_pos_success_rate": full_pos_success_rate_list,
                    "full_pos_success_rate_iqm": float(full_pos_success_rate_iqm),
                    "full_pos_success_rate_std": float(full_pos_success_rate_std),
                    "full_pos_pct_success_timesteps": full_pos_pct_success_timesteps_list,
                    "full_pos_pct_success_timesteps_iqm": float(full_pos_pct_success_timesteps_iqm),
                    "full_pos_pct_success_timesteps_std": float(full_pos_pct_success_timesteps_std)
                })
                
                self.logger.record("eval/full_pos_success", 
                                    full_pos_success_rate_iqm, 
                                    exclude=("stdout", "log", "json", "csv"))
        
        return all_infos

    """+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        Calculate Sequence Matching Reward for the 0th Environment

    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""

    def get_obs_embeddings_from_screens(self, frames):
        """
        Get the embeddings to use for sequence matching given the screens
        """
        frames = frames[:-1]
        frames = th.from_numpy(np.array(frames)).float().cuda(0).permute(0,3,1,2) / 255.0
        frames = self.torch_transform(frames) #th.stack([self.torch_transform(frame) for frame in env_frames]).to(self.device)

        embeddings = run_model_on_batch(frames, self.visual_encoder, self.encoder_batch_size)

        # Just set the values for the last frame as the same as the second to last frame (because last frame is corrupted)
        embeddings = th.cat((embeddings, embeddings[-1][None]), dim=0) 
        embeddings = embeddings.detach().cpu().numpy()

        return embeddings

    def get_obs_clean_states(self, states, geom_xposes=None):
        """
        Get the correct states to use for sequence matching given the full states
        only need to define geom_xposes if using humanoid with xpos
        """
        if self.env_name == "HumanoidSpawnedUpCustom":
            if self.use_geom_xpos:
                # Don't need to do anything here, geom_xpos is getting normalized in the grab_screens function
                obs_seq_to_process= geom_xposes[0]
            else:
                # size: (rollout_length, n_eval_episodes, state_feature_size)
                #   We want only the first 22 features
                obs_seq_to_process = np.array(states[:, 0])[:, :22]
        elif self.env_name == "Metaworld":
            # size: (rollout_length, n_eval_episodes, state_feature_size)
            #   We want only the first 18 features (which corresponds to the current state)
            obs_seq_to_process = np.array(states[:, 0][:, :18])
        return obs_seq_to_process


    def calc_and_record_seq_matching_reward_for_0th_env(self, obs_seq, raw_screens, infos):
        """Calculate the sequence matching reward for the 0th environment

        Parameters:
            states: np.array
                The states for the 0th environment
                size: (rollout_length, n_eval_episodes, state_feature_size)
            geom_xposes: np.array
                The geom_xposes for the 0th environment
                size: (n_eval_episodes, rollout_length, 18, 3)
            raw_screens: List[PIL.Image]
                The raw_screens for the 0th environment
            infos: List[Dict]
                The information for the 0th environment (later used for plotting)

        Returns:
            infos: List[Dict]
                Updated information for the 0th environment (later used for plotting
                    - added 'matching_reward' to the info
        """
        
        if self.calc_matching_reward:
            
            matching_reward, matching_reward_info = self.matching_fn(obs_seq, self.matching_ref_seq)

            self.logger.record("eval/avg_matching_reward", 
                            np.mean(matching_reward)/self.scale, 
                            exclude=("stdout", "log", "json", "csv"))

            # Add the matching_reward to the infos so that we can plot it
            for i in range(len(infos)):
                infos[i]["matching_reward"] = f"{matching_reward[i]:.2f}"

            # Save the matching_rewards locally    
            with open(os.path.join(self._rollout_save_path, f"{self.num_timesteps}_rollouts_matching_rewards.npy"), "wb") as f:
                np.save(f, np.array(matching_reward))

            if self.plot_matching_visualization:
                # TODO: For now, we can only visualize this when the reference frame is defined via a gif
                matching_reward_viz_save_path = os.path.join(self._rollout_save_path, f"{self.num_timesteps}matching_fn_viz.png")

                # Subsample the frames. Otherwise, the visualization will be too long
                if len(raw_screens) > 20:
                    obs_seq_skip_step = int(0.1 * len(raw_screens))
                    raw_screens_used_to_plot = np.array([raw_screens[i] for i in range(obs_seq_skip_step, len(raw_screens), obs_seq_skip_step)])
                else:
                    raw_screens_used_to_plot = np.array(raw_screens)
                    
                if len(self.matching_ref_seq_frames) > 8:
                    ref_seq_skip_step = max(int(0.1 * len(self.matching_ref_seq_frames)), 2)
                    ref_seqs_used_to_plot = np.array([self.matching_ref_seq_frames[i] for i in range(ref_seq_skip_step, len(self.matching_ref_seq_frames), ref_seq_skip_step)])
                else:
                    ref_seqs_used_to_plot = self.matching_ref_seq_frames
                
                seq_matching_viz(
                    matching_fn_name=self.matching_fn_name,
                    obs_seq=raw_screens_used_to_plot,
                    ref_seq=ref_seqs_used_to_plot,
                    matching_reward=matching_reward,
                    info=matching_reward_info,
                    reward_vmin=self.reward_vmin,
                    reward_vmax=self.reward_vmax,
                    path_to_save_fig=matching_reward_viz_save_path,
                    r_discount_factor=self.discount_factor,
                    rolcol_size=2
                )

                # Log the image to wandb
                img = Image.open(matching_reward_viz_save_path)
                self.logger.record(
                    "trajectory/matching_fn_viz",
                    LogImage(np.array(img), dataformats="HWC"),
                    exclude=("stdout", "log", "json", "csv"),
                )

        return infos
    
            
    """+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        Helper functions to set up
            - Whether to calculate the sequence matching reward and how
            - Whether to calculate the ground-truth reward/success rate and how

    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""

    def _setup_seq_matching(self):
        """If needed, set up the sequence matching reward calculation. 
        
        Criteria: matching_fn_cfg != {}

        Parameters:
            env_name: str
                The name of the environment
            task_name: str
                The name of the task in the environment
            matching_fn_cfg: dict
                The configuration for the sequence matching function

        Effects:
            self.calc_matching_reward (bool) - Whether to calculate the sequence matching reward
            self.plot_matching_visualization (bool) - Whether to plot the matching visualization

            self.matching_ref_seq (np.array) - The reference sequence that is used to calculate the sequence matching reward
            self.matching_ref_seq_frames (np.array) - The reference frames (for plotting)

            self.matching_fn (fn) - The function to calculate the sequence matching reward
        """
        if self.matching_fn_cfg != {}:
            # The reference sequence that is used to calculate the ground truth sequence matching performance
            self.gt_ref_seq= load_reference_seq(env_name=self.env_name, task_name=self.task_name, seq_name=self.seq_name, load_visual=False, use_geom_xpos=self.use_geom_xpos)

            if self.calc_visual_reward:
                # Infer the reference sequence that is used to calculate the predicted sequence matching reward
                ref_frames = load_reference_seq(env_name=self.env_name, task_name=self.task_name, seq_name=self.seq_name, load_visual=True, use_geom_xpos=self.use_geom_xpos)

                transformed_frames = [self.pil_transform(frame) for frame in ref_frames]
                
                # Stack the transformed frames into a batch tensor
                frames = th.stack(transformed_frames).to(self.device)
                
                ref_seq = run_model_on_batch(frames, self.visual_encoder, self.encoder_batch_size)
                self.matching_ref_seq = ref_seq.detach().cpu().numpy()
                logger.info(f"[VideoRecorderCallback] Loaded reference GIF embedding sequence. env_name={self.env_name}, task_name={self.task_name}, seq_name={self.seq_name}, self._ref_seq.shape={self.matching_ref_seq.shape}")
            else:
                # If not visual reward, use the ground truth states as the reference sequence
                self.matching_ref_seq = self.gt_ref_seq
                
                logger.info(f"[VideoRecorderCallback] Loaded reference GROUND TRUTH sequence. env_name={self.env_name}, task_name={self.task_name}, seq_name={self.seq_name}, self._ref_seq.shape={self.matching_ref_seq.shape}")                                      

            # These are the frames used for plotting (regardless of visual inference). We remove the initial frame which matches the initial position
            ref_frames_pil = load_reference_seq(env_name=self.env_name, task_name=self.task_name, seq_name=self.seq_name, load_visual=True)[1:]
            self.matching_ref_seq_frames = np.stack([np.array(frame) for frame in ref_frames_pil])

            # TODO: For now, we can only visualize this when the reference frame is defined via a gif
            self.plot_matching_visualization = len(self.matching_ref_seq_frames) > 0

            self.calc_matching_reward = True
            self.scale = self.matching_fn_cfg.get('scale', 1)
            self.matching_fn, self.matching_fn_name = get_matching_fn(self.matching_fn_cfg, self.matching_fn_cfg["cost_fn"])

            self.reward_vmin = self.matching_fn_cfg.get("reward_vmin", -1)
            self.reward_vmax = self.matching_fn_cfg.get("reward_vmax", 0)

            logger.info(f"[VideoRecorderCallback] Loaded reference sequence for seq level matching. task_name={self.task_name}, seq_name={self.seq_name}, use_geom_xpos={self.use_geom_xpos}, shape={self.matching_ref_seq.shape}, image_frames_shape={self.matching_ref_seq_frames.shape}")
        else:
            self.calc_matching_reward = False
            logger.info(f"[VideoRecorderCallback] env_name={self.env_name}, calc_matching_reward=False")



    """++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    Humanoid Environment Setup

    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""
    def _metaworld_env_setup_eval(self, task_name, success_fn_cfg):
        """If the environment is Metaworld, set up the evaluation for the environment

        Effects:
            - Set the flag _calc_gt_reward to True (It's always True for Metaworld before we get environment reward for free)
        """
        self._calc_gt_reward = True
        self._set_metaworld_success_fn(success_fn_cfg)
        self._success_results = {}
        self._success_json_save_path = os.path.join(self._rollout_save_path, "success_results.json")

        logger.info(f"[VideoRecorderCallback] env_name=Metaworld, _calc_gt_reward=True")

    def _humanoid_env_setup_eval(self, task_name, success_fn_cfg, use_geom_xpos):
        """If the environment is HumanoidSpawnedUpCustom, set up the evaluation for the environment

        Effects:
            - Set the ground-truth reward function (that all methods will be compared against)
            - Set the success function
        """
        if task_name != "":
            # We are assuming that "key_frames" represent the key point goal reference sequences
            self._goal_ref_seq = load_reference_seq(env_name=self.env_name, task_name=task_name, seq_name="key_frames", use_geom_xpos=self.use_geom_xpos)
            logger.info(f"[VideoRecorderCallback] Loaded reference sequence for ground-truth reward calculation. task_name={task_name}, seq_name=key_frames, use_geom_xpos={self.use_geom_xpos}, shape={self._goal_ref_seq.shape}")

            self._set_humanoid_ground_truth_reward_fn(task_name, use_geom_xpos)
            self._set_humanoid_success_fn(success_fn_cfg)

            self._calc_gt_reward = True

            self._success_json_save_path = os.path.join(self._rollout_save_path, "success_results.json")
            self._success_results = {}
        else:
            self._calc_gt_reward = False

            logger.info(f"[VideoRecorderCallback] env_name=HumanoidSpawnedUpCustom, _calc_gt_reward=False")
        
    def _set_humanoid_ground_truth_reward_fn(self, task_name: str, use_geom_xpos: bool):
        """For humanoid environment, set the ground-truth goal matching function based on the goal_seq_name.

        This will be unifying metric that we measure the performance of different methods against.

        The function will return an reward array of size (n_timesteps,) where each element is the reward for the corresponding timestep.
        """
        is_goal_reaching_task = HUMANOID_TASK_SEQ_DICT[task_name]["task_type"].lower() == "goal_reaching"

        if is_goal_reaching_task:
            logger.info(f"Goal Reaching Task. The ground-truth reward will be calculated based on the final joint state only. Task name = {task_name}")

            assert len(self._goal_ref_seq) == 1, f"Expected only 1 reference sequence, got {len(self._goal_ref_seq)}"
            
            axis_to_norm = (1,2) if use_geom_xpos else 1

            self._gt_goalmatching_fn = lambda rollout: np.exp(-np.linalg.norm(rollout - self._goal_ref_seq, axis=axis_to_norm))
        else:
            def stage_progress_fn(ref, rollout, threshold):
                """
                Calculate the reward based on the sequence matching to the goal_ref_seq

                Parameters:
                    rollout: np.array (rollout_length, ...)
                        The rollout sequence to calculate the reward
                    threshold: float
                        The threshold to determine if the stage is completed
                """
                # Calculate reward from the rollout to self.gogal_ref_seq
                reward_matrix = np.exp(-euclidean_distance_advanced(rollout, ref))

                # Detect when a stage is completed (the rollout is close to the goal_ref_seq) (under self.threshold)
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
            
            self._gt_goalmatching_fn = lambda rollout: stage_progress_fn(self._goal_ref_seq, rollout, self.threshold)

    def success_fn(self, obs_seq, ref_seq, threshold):
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

        cost_fn_name = "euclidean"
        cost_fn = COST_FN_DICT[cost_fn_name]

        reward_matrix = np.exp(-cost_fn(obs_seq, ref_seq))

        # Detect when a stage is completed (the rollout is close to the goal_ref_seq) (under self.threshold)
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
    

    def _set_metaworld_success_fn(self, success_fn_cfg):
        if self.calc_matching_reward:
            # Because a ref seq is supplied, self._seq_matching_ref_seq is already set
            self._success_fn_based_on_all_pos = lambda obs_seq, ref_seq=self._seq_matching_ref_seq, threshold=success_fn_cfg["threshold_for_all_pos"]: self.success_fn(obs_seq[:, :18], ref_seq, threshold)
        else:
            # Else, we cannot calculate the success wrt the reference sequence
            #   (e.g., when we are training with environment reward)
            self._success_fn_based_on_all_pos = None
        
    def _set_humanoid_success_fn(self, success_fn_cfg):
        """
        Whether the entire body is above an threshold (0.5)
        Whether the arm is above an threshold (0.55)

        Binary success: whether at any point has the key poses have been hit
            # of the key poses that have been hit (in the right order)
        The percentage of time that it's holding the key pose
            For each key pose, we find the time interval that each key poses hold
        """
       
        self._success_fn_based_on_all_pos = lambda obs_seq, ref_seq=self._goal_ref_seq, threshold=success_fn_cfg["threshold_for_all_pos"]: self.success_fn(obs_seq, ref_seq, threshold)
        self._success_fn_based_on_only_arm_pos = lambda obs_seq, ref_seq=self._goal_ref_seq, threshold=success_fn_cfg["threshold_for_arm_pos"]: self.success_fn(obs_seq[:, 12:], ref_seq[:, 12:], threshold)

    def add_success_results(self, curr_timestep, timestep_success_dict):
        """
        Add the success results to the success_results dictionary
        """
        self._success_results[curr_timestep] = timestep_success_dict

        with open(self._success_json_save_path, "w") as f:
            json.dump(self._success_results, f, indent=4)


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