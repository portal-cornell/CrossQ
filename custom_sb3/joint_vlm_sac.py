from stable_baselines3 import SAC
import torch
from collections import deque
import copy

from loguru import logger

#### From stable_baselines3/sac/sac.py ####
from typing import Any, ClassVar, Dict, Optional, Tuple, Type, TypeVar, Union
import time
from einops import rearrange
import numpy as np

import warnings

from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from stable_baselines3.common.utils import check_for_correct_spaces, get_system_info, safe_mean
from stable_baselines3.common.vec_env.patch_gym import _convert_space
from stable_baselines3.sac.policies import SACPolicy
#### From stable_baselines3/sac/sac.py ####

from torchvision import models, transforms
from torch import nn


from vlm_reward.vlm_buffer import VLMReplayBuffer
from vlm_reward.reward_transforms import half_gaussian_filter_1d
from vlm_reward.reward_models.joint_pred import load_joint_prediction_model


class JointVLMSAC(SAC):
    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "steps"),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = VLMReplayBuffer,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        ### VLM SAC Custom Parameters ###
        inference_only: bool = False,
        reward_model_config: dict = None,
        n_cpu_workers: int = 1,
        n_gpu_workers: int = 1,
        episode_length: int = 120,
        render_dim: Tuple[int, int] = (480, 480),
        add_to_gt_rewards: bool = True,
        ref_joint_states = None,
        confidence_kappa=2,
    ):
        # TODO: Add a parameter to point to the dataset relevant to the task
        # train_freq[0] because we are assuming that the train_freq is a tuple
        stats_window_size = (
            (learning_starts + train_freq[0] * env.num_envs)
            // episode_length
            // env.num_envs
        ) * env.num_envs

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[torch.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[torch.optim.Adam] = None
        
        # VLM SAC Custom Parameters
        self.reward_model_config = reward_model_config
        self.n_cpu_workers = n_cpu_workers
        self.n_gpu_workers = n_gpu_workers
        self.episode_length = episode_length
        self.render_dim = render_dim

        self.ep_vlm_info_buffer = None  # type: Optional[deque]

        # Joint model parameters
        self.filter_rewards = False # whether or not to gaussian filter the rewards after computing
        self.estimate_confidence = self.reward_model_config["estimate_confidence"]
        self.reward_model_batch_size = self.reward_model_config["reward_batch_size"]
        self._add_to_gt_rewards = add_to_gt_rewards
        if ref_joint_states is not None:
            self._ref_joint_states = ref_joint_states.cuda(0)
        self.uncertainty_stats = RunningNormalStats(alpha=.99)
        self.kappa = confidence_kappa # wvn used 2, but this leads to lots of 0 preds for good states (higher -> relaxed uncertainty)
        
        self.inference_only = inference_only
        if not self.inference_only:
            self._setup_reward_model()
            self.previous_num_timesteps = 0
            self.previous_num_episodes = 0

            if self.reward_model_config["rank0_batch_size_pct"] < 1.0:
                # Uneven workload split between workers
                worker_batch_size = int((1 - self.reward_model_config["rank0_batch_size_pct"]) * self.reward_model_config["reward_batch_size"]) // (self.n_gpu_workers - 1)
            else:
                worker_batch_size = self.reward_model_config["reward_batch_size"] // self.n_gpu_workers
            
            self.worker_frames_tensor = torch.zeros(
                    (worker_batch_size, self.render_dim[0], self.render_dim[1], 3),
                    dtype=torch.uint8,
                ).cuda(0)  # (Batch size per worker, w, h, 3)

        if _init_setup_model:
            self._setup_model()

    """
    Added for VLM reward
    """
    def _setup_reward_model(self):
        logger.info(f"Setting up VLM reward model: {self.reward_model_config['vlm_model']}")
        
        # This is the actual batch size for rank0 inference worker
        #   because this batch_size is used to decide how many copies of the reference human image to use
        if self.reward_model_config["rank0_batch_size_pct"] < 1.0:
            rank0_worker_batch = int(self.reward_model_config["rank0_batch_size_pct"] * self.reward_model_config["reward_batch_size"])
        else:
            rank0_worker_batch = self.reward_model_config["reward_batch_size"] // self.n_gpu_workers

        reward_model, transform = load_joint_prediction_model(device='cuda:0', 
                                    checkpoint=self.reward_model_config["checkpoint"],
                                    output_dim=self.reward_model_config["output_dim"],
                                    is_reco=self.estimate_confidence
                                    )
        
        self.reward_model = reward_model
        self.image_transform = transform

        logger.debug(f"Finished loading up VLM reward model: {self.reward_model_config['vlm_model']}")

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        *args,
    ) -> Tuple[int, BaseCallback]:
        total_timesteps, callback = super()._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            *args,
        )
        if self.ep_vlm_info_buffer is None or reset_num_timesteps:
            self.ep_vlm_info_buffer = deque(maxlen=self._stats_window_size)
        return total_timesteps, callback

    def collect_rollouts(self, *args, **kwargs):
        rollout = super().collect_rollouts(*args, **kwargs)
        if not self.inference_only:
            # TODO, IMPORTANT: commented this out for now, to move reward computation to the callback
            #self._compute_vlm_rewards()
            self.previous_num_timesteps = self.num_timesteps
            self.previous_num_episodes = self._episode_num

        return rollout
    
    def compute_joint_predictions(self, frames, apply_transform=True):
        """Compute joint positions using the visual model"""
        batches = torch.split(frames, self.reward_model_batch_size)
        all_preds = []
        
        for batch in batches:
            if apply_transform:
                xpos_preds = self.forward_visual_model(batch)
            else: 
                xpos_preds, emb, emb_reco = self.reward_model(batch)
            all_preds.append(xpos_preds)

        joint_positions = torch.cat(all_preds, dim=0)
        
        return joint_positions

    def compute_joint_predictions_with_uncertainty(self, frames, apply_transform=True):
        """Compute joint positions using the visual model"""
        batches = torch.split(frames, self.reward_model_batch_size)
        all_preds = []
        all_uncertainties = []
        
        for batch in batches:
            if apply_transform:
                xpos_preds, emb, emb_reco = self.forward_visual_model(batch)
            else:
                xpos_preds, emb, emb_reco = self.reward_model(batch)

            _, D = emb_reco.shape
            uncertainty = 1/D * (torch.linalg.vector_norm(emb - emb_reco, dim=1)**2) # MSE = 1/N * norm^2
            
            all_preds.append(xpos_preds)
            all_uncertainties.append(uncertainty)
        
        joint_positions = torch.cat(all_preds, dim=0)
        uncertainties = torch.cat(all_uncertainties, dim=0)
        
        return joint_positions, uncertainties

    def compute_confidence_weights(self, uncertainties, uncertainty_stats):
        """Compute confidence weights based on uncertainties, and given an uncertainty stats object
        This allows you to use different running means/stds (for example, in different callbacks)"""
        mu = uncertainty_stats.get_mean()
        std = uncertainty_stats.get_std()
        scale = 1
        
        if mu is None:
            confidence = torch.ones_like(uncertainties)
        else:
            confidence = torch.exp(-(uncertainties - mu) ** 2 / (2 * (std * self.kappa) ** 2))
            confidence[uncertainties < mu] = 1.0
        
        return confidence * scale

    def _compute_vlm_rewards(self):
        """from VLMRewardCallback.on_rollout_end
        """
        # Time this function
        start_time = time.time()

        replay_buffer_pos = self.replay_buffer.pos
        total_timesteps = self.num_timesteps - self.previous_num_timesteps  # Total number of timesteps that we have collected
        env_episode_timesteps = total_timesteps // self.env.num_envs  # Number of timesteps that we have collected per environment
        total_episodes = self.get_episode_num() - self.previous_num_episodes
        env_episodes = total_episodes // self.env.num_envs

        ref_joint_states = self._ref_joint_states # target joint states
        
        ### Prepare the frame to be processed
        frames = torch.from_numpy(np.array(self.replay_buffer.render_arrays)).float().cuda(0 ) / 255.0

        print(f"Start calculating rewards: frames.shape={frames.shape}")

        frames = rearrange(frames, "n_steps n_envs h w c -> (n_steps n_envs) c h w")
 
        rewards = (1 - .999**self.get_episode_num()) * self._compute_joint_rewards(
            frames=frames,
            ref_joint_states=ref_joint_states,
            batch_size=self.reward_model_batch_size,
            )
        # TODO: this assumes 1D (DreamSim). Potentially to adapt for other reward models (using above)
        rewards = rearrange(
            rewards,
            "(n_steps n_envs) -> n_steps n_envs",
            n_envs=self.env.num_envs,
        )
        self.replay_buffer.clear_render_arrays()
        rewards_np = rewards.cpu().numpy()

        cur_height = np.zeros_like(rewards_np)
        if replay_buffer_pos - env_episode_timesteps >= 0:
            cur_height = self.replay_buffer.observations[replay_buffer_pos - env_episode_timesteps : replay_buffer_pos, :, 0] 
        else:
            # Split reward assignment (circular buffer)
            cur_height[: env_episode_timesteps - replay_buffer_pos, :] = self.replay_buffer.observations[
                -(env_episode_timesteps - replay_buffer_pos) :, :, 0
            ]
            cur_height[
                env_episode_timesteps - replay_buffer_pos :, :
            ] = self.replay_buffer.observations[:replay_buffer_pos,:, 0]

        rewards_np = rewards_np * (cur_height > 1.1) # only take vlm rewards when the gt rewards are large (so it is standing)
        ### Update the rewards
        # import pdb; pdb.set_trace()
        if self._add_to_gt_rewards:
            print("Adding VLM rewards to GT rewards")
            # Convert rewards tensor to np array for compatibility with self.replay_buffer.rewards
            # Add the VLM reward to existing rewards
            if replay_buffer_pos - env_episode_timesteps >= 0:
                self.replay_buffer.rewards[replay_buffer_pos - env_episode_timesteps : replay_buffer_pos, :] += rewards_np[:, :]
            else:
                # Split reward assignment (circular buffer)
                self.replay_buffer.rewards[
                    -(env_episode_timesteps - replay_buffer_pos) :, :
                ] += rewards_np[: env_episode_timesteps - replay_buffer_pos, :]

                self.replay_buffer.rewards[:replay_buffer_pos, :] += rewards_np[
                    env_episode_timesteps - replay_buffer_pos :, :
                ]
        else:
            print("Overwriting GT rewards with VLM rewards")
            # Overwrite the rewards with VLM rewards
            if replay_buffer_pos - env_episode_timesteps >= 0:
                self.replay_buffer.rewards[
                    replay_buffer_pos - env_episode_timesteps : replay_buffer_pos, :
                ] = rewards_np[:, :]
            else:
                # Split reward assignment (circular buffer)
                self.replay_buffer.rewards[
                    -(env_episode_timesteps - replay_buffer_pos) :, :
                ] = rewards_np[: env_episode_timesteps - replay_buffer_pos, :]

                self.replay_buffer.rewards[:replay_buffer_pos, :] = rewards_np[
                    env_episode_timesteps - replay_buffer_pos :, :
                ]

        ### Logging the rewards 
        # TODO: compatibility with torch vs numpy, for now it assumes rewards is a Tensor
        rewards = rearrange(rewards, "n_steps n_envs -> n_envs n_steps")
        if isinstance(rewards, torch.Tensor):
            rewards_np = rewards.cpu().numpy()
        for env_idx in range(self.env.num_envs):
            # Compute sum of rewards per episode
            rewards_per_episode = np.sum(
                np.reshape(
                    rewards_np[env_idx], (env_episodes, self.episode_length)
                ),
                axis=1,
            )
            self.ep_vlm_info_buffer.extend([rewards_per_episode.tolist()])

        print(f"VLMRewardCallback took {time.time() - start_time} seconds")


    def get_vram(self):
        free = torch.cuda.mem_get_info()[0] / 1024 ** 3
        total = torch.cuda.mem_get_info()[1] / 1024 ** 3
        total_cubes = 24
        free_cubes = int(total_cubes * free / total)
        return f'VRAM: {total - free:.2f}/{total:.2f}GB\t VRAM:[' + (
                total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']'
                
    def forward_visual_model(self, batch):
        """
        Batch is shape (N, C, H, W) in [0,1] containing images
        Applies transform to the batch and runs the model on it
        """
        batch_transformed = self.image_transform(batch)
        return self.reward_model(batch_transformed)

    def _compute_joint_rewards(self, frames, ref_joint_states, batch_size, apply_transform=True):
        """Only use the goal joint xpos states to calculate the reward
        - The reward is based on the euclidean distance between the current joint states and the reference joint states

        Final goal: Both arms out

        This task is a goal-reaching task (i.e. doesn't matter how you get to the goal, as long as you get to the goal)
        """
        batches = torch.split(frames, batch_size)
        all_preds = []
        all_uncertainties = []
        for i, batch in enumerate(batches):
            if apply_transform:
                xpos_preds, emb, emb_reco = self.forward_visual_model(batch)
            else:
                xpos_preds, emb, emb_reco = self.reward_model(batch)
                
            uncertainty = torch.linalg.vector_norm(emb - emb_reco, dim=1)
            
            all_preds.append(xpos_preds)
            all_uncertainties.append(uncertainty)
        
        def gaussian_likelihood(l, mu, std, kappa):
            return torch.exp(- (l - mu) ** 2 / (2 * (std * kappa) ** 2))
        curr_geom_xpos = torch.cat(all_preds, dim=0)
        xpos_uncertainties = torch.cat(all_uncertainties, dim=0)
        #self.uncertainty_stats.update(xpos_uncertainties.mean().item()) # update before calculating confidence for now (problem: how to assign the first values for confidence?)

        mu = self.uncertainty_stats.get_mean()
        std = self.uncertainty_stats.get_std()
        scale = 1# - .99 ** self.uncertainty_stats.count 
        
        if mu is None:
            confidence = torch.ones_like(xpos_uncertainties)
        else:
            confidence = gaussian_likelihood(xpos_uncertainties, mu, std, self.kappa)
            confidence[xpos_uncertainties < mu] = 1.0
        confidence *= scale

        n, d = ref_joint_states.shape # n is the number of joints, d is the dimension of each joint vector
        curr_geom_xpos = curr_geom_xpos.view(len(frames), n, d)
        # Normalize the current pose by the torso's position (which is at index 1)
        # curr_geom_xpos = curr_geom_xpos - curr_geom_xpos[:, 1, :]

        joint_pos_relevant = curr_geom_xpos[:, 12:, :].flatten(start_dim=1)
        target_joint_pos_relevant = ref_joint_states[None, 12:, :].flatten(start_dim=1)

        pose_matching_reward = confidence * torch.exp(-torch.linalg.vector_norm(target_joint_pos_relevant - joint_pos_relevant, dim=1))
        
        return pose_matching_reward

    def learn(self, *args, **kwargs):
        self.previous_num_timesteps = 0
        self.previous_num_episodes = 0

        # Call the parent learn function
        return super().learn(*args, **kwargs)
    
    def train(self, *args, **kwargs):
        # Call the parent train function
        super().train(*args, **kwargs)

        # Log the VLM reward information
        if len(self.ep_vlm_info_buffer) > 0 and len(self.ep_vlm_info_buffer[0]) > 0:
            self.logger.record(
                "rollout/ep_vlm_rew_mean",
                safe_mean([ep_reward for ep_reward in self.ep_vlm_info_buffer]),
            )
    
    def get_episode_num(self):
        # A hack to get the episode number
        return self._episode_num
    

    # Need to have this because VLM SAC has additional parameters
    @classmethod
    def load(  # noqa: C901
        cls,
        path,
        env = None,
        inference_only = True,
        device = "auto",
        custom_objects = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        reward_model_config: dict = None,
        episode_length: int = 120,
        render_dim: Tuple[int, int] = (480, 480),
        **kwargs,
    ):
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            if "net_arch" in data["policy_kwargs"] and len(data["policy_kwargs"]["net_arch"]) > 0:
                saved_net_arch = data["policy_kwargs"]["net_arch"]
                if isinstance(saved_net_arch, list) and isinstance(saved_net_arch[0], dict):
                    data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        model = cls(
            policy=data["policy_class"],
            env=env,
            inference_only = inference_only,
            device=device,
            _init_setup_model=False,  # type: ignore[call-arg]
            train_freq=(episode_length, "step"),
            reward_model_config=reward_model_config,
            episode_length=episode_length,
            render_dim=render_dim
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load Policy saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e
        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # type: ignore[operator]
        return model

class RunningNormalStats:
    def __init__(self, alpha=0.98):
        """
        Initialize the RunningStats object.
        TODO: Hard coded values from mean on training set right now
        
        Args:
            alpha (float): The smoothing factor. Default is 0.98.
                           Higher values give more weight to past observations.
        """
        # stats for old model (9/26):
        self.gt_mean = 0.0031699459068477154
        self.gt_std = 0.002458093920722604

        # stats for fine tuned autoencoder on reference (10/11), successful for sdtw+ visual ref (kappa=.5)
        # self.gt_mean = 0.00631257938221097
        # self.gt_std = 0.005063631106168032

        self.alpha = alpha
        self.mean = None
        self.squared_mean = None
        self.count = 0

    def update(self, value):
        """
        Update the running statistics with a new value.
        
        Args:
            value (torch.Tensor or float): The new value to incorporate.
        
        Returns:
            tuple: (mean, std) The updated mean and standard deviation.
        """
        self.count += 1
        
        if isinstance(value, (int, float)):
            value = torch.tensor(value, dtype=torch.float32)
        
        if self.mean is None:
            self.mean = value.clone().detach()
            self.squared_mean = value.pow(2).clone().detach()
        else:
            if self.count > 1/(1-self.alpha):
                self.mean = self.alpha * self.mean + (1 - self.alpha) * value
                self.squared_mean = self.alpha * self.squared_mean + (1 - self.alpha) * value.pow(2)
            else:
                self.mean = (self.mean * (self.count - 1) + value) / self.count
                self.squared_mean = (self.squared_mean * (self.count - 1) + value.pow(2)) / self.count
    
    def get_mean(self):
        """
        Get the current running mean 
        
        Returns:
            tuple: (mean, std) The current mean and standard deviation,
                   or (None, None) if no updates have been made.
        """
        if self.gt_mean is not None:
            return self.gt_mean
        return self.mean

    def get_std(self):
        """
        Get the current running stdev 
        
        Returns:
            tuple: (mean, std) The current mean and standard deviation,
                   or (None, None) if no updates have been made.
        """
        if self.gt_std is not None: # lets you specify a ground truth std and override this
            return self.gt_std

        variance = self.squared_mean - self.mean.pow(2)
        # Clamp to prevent negative variance due to numerical instability
        std = torch.sqrt(torch.clamp(variance, min=1e-8))
        return std