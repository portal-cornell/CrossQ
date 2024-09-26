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

from custom_sb3 import PPO
from seq_matching_toy.toy_envs.grid_nav import *
from seq_matching_toy.toy_examples_main import examples
from seq_matching_toy.gridnav_rl_callbacks import WandbCallback, GridNavVideoRecorderCallback, GridNavSeqRewardCallback


def load_map_from_example_dict(example_name: str) -> NDArray:
    """
    Load the map from the example dictionary.

    Parameters:
        example_name: str
            - The name of the example

    Returns:
        map_array: NDArray
            - The map array
    """
    return examples[example_name]["map_array"]

def load_starting_pos_from_example_dict(example_name: str) -> NDArray:
    """
    Load the starting position from the example dictionary.

    Parameters:
        example_name: str
            - The name of the example

    Returns:
        starting_pos: NDArray
            - The starting position of the agent
    """
    return examples[example_name]["starting_pos"]

def load_ref_seq_from_example_dict(example_name: str) -> NDArray:
    """
    Load the reference seq from the example dictionary.

    Parameters:
        example_name: str
            - The name of the example

    Returns:
        ref_seq: NDArray
            - The array of reference sequences
    """
    return examples[example_name]["ref_seq"]

def load_reward_vmin_vmax_from_example_dict(example_name: str) -> NDArray:
    """
    Load the reard vmin and vmax from the example dictionary.

    Parameters:
        example_name: str
            - The name of the example

    Returns:
        reward_vmin: float
            - The minimum value of the reward to receive in the environment
        reward_vmax: float
            - The maximum value of the reward to receive in the environment
    """
    return examples[example_name]["plot"]["reward_vmin"], examples[example_name]["plot"]["reward_vmax"]

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
def train(cfg: DictConfig):
    # Set the path for logging
    cfg.logging.run_name = get_output_folder_name(cfg.log_folder)
    cfg.logging.run_path = get_output_path()

    logger.info(f"Logging to {cfg.logging.run_path}\nRun name: {cfg.logging.run_name}")

    os.makedirs(os.path.join(cfg.logging.run_path, "eval"), exist_ok=True)

    # Initialize the environment
    map_array = load_map_from_example_dict(cfg.env.example_name)
    starting_pos = load_starting_pos_from_example_dict(cfg.env.example_name)
    ref_seq = load_ref_seq_from_example_dict(cfg.env.example_name)
    reward_vmin, reward_vmax = load_reward_vmin_vmax_from_example_dict(cfg.env.example_name)

    make_env_fn = lambda: Monitor(GridNavigationEnv(map_array=np.copy(map_array), starting_pos=starting_pos, render_mode="rgb_array", episode_length=cfg.env.episode_length))

    training_env = make_vec_env(
        make_env_fn,
        n_envs=cfg.compute.n_cpu_workers,
        seed=cfg.seed,
        vec_env_cls=SubprocVecEnv,
    )

    # Define the model
    model = PPO("MlpPolicy", 
                training_env, 
                n_steps=cfg.env.episode_length,
                n_epochs=cfg.rl_algo.n_epochs,
                batch_size=cfg.rl_algo.batch_size,
                learning_rate=cfg.rl_algo.lr,
                tensorboard_log=os.path.join(cfg.logging.run_path, "tensorboard"),
                verbose=1)
    
    with wandb.init(
        project=cfg.logging.wandb_project,
        name=cfg.logging.run_name,
        tags=cfg.logging.wandb_tags,
        sync_tensorboard=True,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        mode=cfg.logging.wandb_mode,
        monitor_gym=True,  # auto-upload the videos of agents playing the game
    ) as wandb_run:
        # Make an alias for the wandb in the run_path
        if cfg.logging.wandb_mode != "disabled":
            os.symlink(os.path.abspath(wandb_run.dir), os.path.join(cfg.logging.run_path, "wandb"), target_is_directory=True)

        checkpoint_dir = os.path.join(cfg.logging.run_path, "checkpoint")

        wandb_callback = WandbCallback(
            model_save_path=str(checkpoint_dir),
            model_save_freq=cfg.logging.model_save_freq // cfg.compute.n_cpu_workers,
            verbose=2,
        )

        video_callback = GridNavVideoRecorderCallback(
            SubprocVecEnv([make_env_fn]),
            rollout_save_path=os.path.join(cfg.logging.run_path, "eval"),
            render_freq=cfg.logging.video_save_freq // cfg.compute.n_cpu_workers,
            map_array = np.copy(map_array),
            ref_seq = np.copy(ref_seq),
            matching_fn_cfg = dict(cfg.seq_reward_model),
            cost_fn_name = cfg.cost_fn,
            reward_vmin = reward_vmin,
            reward_vmax = reward_vmax,
        )

        seq_matching_callback = GridNavSeqRewardCallback(
            map_array = np.copy(map_array),
            ref_seq = np.copy(ref_seq),
            matching_fn_cfg = dict(cfg.seq_reward_model),
            cost_fn_name = cfg.cost_fn,
        )

        callback_list = [wandb_callback, video_callback, seq_matching_callback]

        # Train the model
        model.learn(
            total_timesteps=cfg.rl_algo.total_timesteps,
            progress_bar=True, 
            callback=CallbackList(callback_list))

        logger.info("Saving final model")
        model.save(str(os.path.join(checkpoint_dir, "final_model")))

        logger.info("Done.")
        wandb_run.finish()


if __name__ == "__main__":
    train()
