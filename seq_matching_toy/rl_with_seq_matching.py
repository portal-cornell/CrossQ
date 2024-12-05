from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.global_hydra import GlobalHydra

import os
import wandb
from typing import Any, Dict, List, Union
from numpy.typing import NDArray
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList
from loguru import logger

from custom_sb3 import PPO, RECURRENT_PPO
from reinforce_model import REINFORCE
from seq_matching_toy.toy_envs.grid_nav import *
from seq_matching_toy.toy_envs.minigrid_sequence import make_sequence_env
from seq_matching_toy.toy_examples_main import load_map_from_example_dict, load_ref_seq_from_example_dict, load_reward_vmin_vmax_from_example_dict, load_starting_pos_from_example_dict
from seq_matching_toy.gridnav_rl_callbacks import WandbCallback, GridNavVideoRecorderCallback, GridNavSeqRewardCallback, MinigridSeqRewardCallback, MinigridVideoRecorderCallback


# Define sweep config
sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "rollout/mean_gt_reward_per_epsisode"},
    "parameters": {
        # "lr": {"max": 0.001, "min": 0.00001},

        "lr": {"values": [2.5e-4]},
        "ent_coef": {"values": [0.25, 0.5, 1.0, 1.5, 2.0]},
        "vf_coef": {"values": [0.25, 0.5, 1.0, 1.5, 2.0]},
        "episode_length": {"values": [5]},

        # "lr": {"values": [2.5e-4]},
        # "ent_coef": {"max": 2.0, "min": 0.1},
        # "episode_length": {"values": [9, 10, 11, 12]},

        # "lr": {"values": [2.5e-4]},
        # "ent_coef": {"max": 1.5, "min": 0.25},
        # "episode_length": {"values": [9]},
        
        # REINFORCE tuning
        # "lr": {"values": [0.01, 0.001, 0.0001]},
        # "ent_coef": {"max": 1.0, "min": 0.1},
        # "episode_length": {"values": [8]},
    },
}

def get_output_folder_name(data_log_dir) -> str:
    """
    Return:
        folder_name: str
            - The name of the folder that holds all the logs/outputs for the current run
            - Not a complete path
    """
    # Remove the path to the repo directory
    folder_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir.replace(str(os.getcwd()),"")
    # Remove the name of the folder that holds all the outputs
    folder_name = folder_path.replace(data_log_dir, "").replace("/", "") # TODO: a hack
    
    return folder_name

def get_output_path() -> str:
    """
    Return:
        output_path: str
            - The absolute path to the folder that holds all the logs/outputs for the current run
    """
    return hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

@hydra.main(version_base=None, config_path="configs", config_name="train_rl_config")
def train_or_sweep(cfg: DictConfig):
    # Set the path for logging
    cfg.logging.run_name = get_output_folder_name(cfg.log_folder)
    cfg.logging.run_path = get_output_path()

    logger.info(f"Logging to {cfg.logging.run_path}\nRun name: {cfg.logging.run_name}")

    if cfg.sweep.enabled:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=f"sweep_{cfg.env.example_name}")
        wandb.agent(sweep_id, function=lambda cfg=cfg: train(cfg), count=cfg.sweep.n_trails)
    else:
        train(cfg)


def train(cfg: DictConfig):
    os.makedirs(os.path.join(cfg.logging.run_path, "eval"), exist_ok=True)

    reinforce = cfg.rl_algo.name.lower() == "reinforce"
    policy_gradient = cfg.rl_algo.name.lower() == "policy_gradient"
    include_history = cfg.env.include_history

    # Initialize the environment
    map_array = load_map_from_example_dict(cfg.env.example_name)
    starting_pos = load_starting_pos_from_example_dict(cfg.env.example_name)
    ref_seq = load_ref_seq_from_example_dict(cfg.env.example_name)
    reward_vmin, reward_vmax = load_reward_vmin_vmax_from_example_dict(cfg.env.example_name)

    if include_history:
        grid_class = GridNavigationEnvHistory
    else:
        grid_class = GridNavigationEnv

    tags =cfg.logging.wandb_tags + [f"ep_{cfg.env.episode_length}", cfg.env.example_name, cfg.seq_reward_model.name, f"disc_{cfg.rl_algo.gamma}", f"ent_{cfg.rl_algo.ent_coef}"] + (["temporal"] if cfg.env.temporal_encoding else []) + ([f"rank_{cfg.seq_reward_model.rank_weighting}"] if cfg.seq_reward_model.get('rank_weighting') else [])

    with wandb.init(
        project=cfg.logging.wandb_project,
        name=cfg.logging.run_name,
        tags=tags,
        sync_tensorboard=True,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        mode=cfg.logging.wandb_mode,
        monitor_gym=True,  # auto-upload the videos of agents playing the game
    ) as wandb_run:
        if cfg.sweep.enabled:
            lr = wandb.config.lr
            ent_coef = wandb.config.ent_coef
            vf_coef = wandb.config.vf_coef
            episode_length = wandb.config.episode_length

            logger.info(f"Running sweep with lr={wandb.config.lr}, ent_coef={wandb.config.ent_coef}, vf_coef={wandb.config.vf_coef}, episode_length={wandb.config.episode_length}")
        else:
            lr = cfg.rl_algo.lr
            ent_coef = cfg.rl_algo.ent_coef
            vf_coef = cfg.rl_algo.vf_coef
            episode_length = cfg.env.episode_length
        
        if cfg.env.minigrid:
            make_env_fn = lambda: Monitor(make_sequence_env(map_array=np.copy(map_array), starting_pos=starting_pos, render_mode="rgb_array", temporal_encoding= cfg.env.temporal_encoding, episode_length=episode_length))
            training_env = make_vec_env(
                make_env_fn,
                n_envs=cfg.compute.n_cpu_workers,
                seed=cfg.seed,
                vec_env_cls=SubprocVecEnv,
            )
        else:
            if reinforce or policy_gradient:
                training_env = grid_class(map_array=np.copy(map_array), starting_pos=starting_pos, render_mode="rgb_array", episode_length=episode_length)
            else:
                make_env_fn = lambda: Monitor(grid_class(map_array=np.copy(map_array), starting_pos=starting_pos, render_mode="rgb_array", episode_length=episode_length))

                training_env = make_vec_env(
                    make_env_fn,
                    n_envs=cfg.compute.n_cpu_workers,
                    seed=cfg.seed,
                    vec_env_cls=SubprocVecEnv,
                )

        if reinforce or policy_gradient:
            model = REINFORCE(
                        env = training_env,
                        n=cfg.rl_algo.n,
                        learning_rate=lr,
                        ent_coef=ent_coef,
                        gamma=cfg.rl_algo.gamma,
                        policy_gradient=policy_gradient,
                        use_relative_reward=cfg.rl_algo.use_relative_reward if "use_relative_reward" in cfg.rl_algo else False,
                        video_save_freq=cfg.logging.video_save_freq)
        else:
            ppo_class = RECURRENT_PPO if cfg.rl_algo.recurrent_policy else PPO
            policy_type = "MlpLstmPolicyMlpPolicy" if cfg.rl_algo.recurrent_policy else "MlpPolicy"

            # Define the model
            model = ppo_class(policy_type, 
                        training_env, 
                        n_steps=cfg.env.episode_length,
                        n_epochs=cfg.rl_algo.n_epochs,
                        batch_size=cfg.rl_algo.batch_size,
                        learning_rate=lr,
                        tensorboard_log=os.path.join(cfg.logging.run_path, "tensorboard"),
                        gamma=cfg.rl_algo.gamma,
                        ent_coef=ent_coef,
                        vf_coef=vf_coef,
                        verbose=1,
                        seed=cfg.seed)
    
        # Make an alias for the wandb in the run_path
        if cfg.logging.wandb_mode != "disabled" and not cfg.sweep.enabled:
            os.symlink(os.path.abspath(wandb_run.dir), os.path.join(cfg.logging.run_path, "wandb"), target_is_directory=True)

        checkpoint_dir = os.path.join(cfg.logging.run_path, "checkpoint")

        wandb_callback = WandbCallback(
            model_save_path=str(checkpoint_dir),
            model_save_freq=cfg.logging.model_save_freq // cfg.compute.n_cpu_workers,
            verbose=2,
        )

        matching_fn_cfg = dict(cfg.seq_reward_model)
        matching_fn_cfg["reward_vmin"] = reward_vmin
        matching_fn_cfg["reward_vmax"] = reward_vmax

        if cfg.env.minigrid:
            video_callback = MinigridVideoRecorderCallback(
                make_env_fn,
                rollout_save_path=os.path.join(cfg.logging.run_path, "eval"),
                render_freq=cfg.logging.video_save_freq // cfg.compute.n_cpu_workers,
                ref_seq = np.copy(ref_seq),
                matching_fn_cfg = matching_fn_cfg,
                cost_fn_name = cfg.cost_fn,
                temporal_encoding= cfg.env.temporal_encoding,
                reward_vmin = reward_vmin,
                reward_vmax = reward_vmax,
                reward_discount_factor=cfg.rl_algo.gamma
            )

            seq_matching_callback = MinigridSeqRewardCallback(
                ref_seq = np.copy(ref_seq),
                matching_fn_cfg = matching_fn_cfg,
                cost_fn_name = cfg.cost_fn,
                temporal_encoding= cfg.env.temporal_encoding,
                env_fn = make_env_fn
            )

        else:
            video_callback = GridNavVideoRecorderCallback(
                SubprocVecEnv([make_env_fn]) if not (reinforce or policy_gradient) else env,
                rollout_save_path=os.path.join(cfg.logging.run_path, "eval"),
                render_freq=cfg.logging.video_save_freq // cfg.compute.n_cpu_workers,
                map_array = np.copy(map_array),
                ref_seq = np.copy(ref_seq),
                matching_fn_cfg = matching_fn_cfg,
                cost_fn_name = cfg.cost_fn,
                reward_vmin = reward_vmin,
                reward_vmax = reward_vmax,
                use_history=cfg.env.include_history
            )

            seq_matching_callback = GridNavSeqRewardCallback(
                map_array = np.copy(map_array),
                ref_seq = np.copy(ref_seq),
                matching_fn_cfg = matching_fn_cfg,
                cost_fn_name = cfg.cost_fn,
                use_history=cfg.env.include_history
            )

        callback_list = [wandb_callback, video_callback, seq_matching_callback]

        # Train the model
        model.learn(
            total_timesteps=cfg.rl_algo.total_timesteps,
            progress_bar=True, 
            callback=CallbackList(callback_list) if not (reinforce or policy_gradient) else callback_list)

        logger.info("Saving final model")
        model.save(str(os.path.join(checkpoint_dir, "final_model")))

        logger.info("Done.")
        wandb_run.finish()

# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    train_or_sweep()
# pylint: enable=no-value-for-parameter