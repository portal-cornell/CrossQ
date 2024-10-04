import datetime
import secrets
import os
import numpy as np

from omegaconf import DictConfig
from loguru import logger

import hydra

from constants import WANDB_DIR

def get_run_hash() -> str:
    return f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_', f"{secrets.token_hex(4)}"

def get_output_folder_name() -> str:
    """
    Return:
        folder_name: str
            - The name of the folder that holds all the logs/outputs for the current run
            - Not a complete path
    """
    # Remove the path to the repo directory
    folder_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir.replace(str(os.getcwd()),"")
    # Remove the name of the folder that holds all the outputs
    # Ignoring the first "/" and splitting the path by "/"
    folder_name = folder_path[1:].split("/")[1]
    
    return folder_name

def get_output_path() -> str:
    """
    Return:
        output_path: str
            - The absolute path to the folder that holds all the logs/outputs for the current run
    """
    return hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

def use_vlm_for_reward(cfg: DictConfig) -> bool:
    return "hand_engineered" not in cfg.reward_model.name.lower() and "ot" not in cfg.reward_model.name.lower() and "dtw" not in cfg.reward_model.name.lower()

def use_joint_vlm_for_reward(cfg: DictConfig) -> bool:
    return "joint_pred" in cfg.reward_model.name.lower()

def set_os_vars() -> None:

    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    # Get egl (mujoco) rendering to work on cluster
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "all"
    os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "compute,graphics,utility,video"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["EGL_PLATFORM"] = "device"
    # Get wandb file (e.g. rendered) gif more accessible
    os.environ["WANDB_DIR"] = WANDB_DIR
    # os.environ["LOGURU_LEVEL"] = "INFO"  # Uncomment this if you don't want logger to print logger.debug stuff

def validate_and_preprocess_cfg(cfg: DictConfig):
    """
    Parameters:
        cfg: DictConfig
            - The hydra config object

    Effects:
        - Sets the logging.run_name to the name of the folder that holds all the logs/outputs for the current run
        - Sets the logging.run_path to the absolute path to the folder that holds all the logs/outputs for the current run
    """
    if use_vlm_for_reward(cfg):
        assert cfg.reward_model.reward_batch_size % cfg.compute.n_gpu_workers == 0, f"({cfg.reward_model.reward_batch_size=}) corresponds to the total size of the batch do be distributed among workers and therefore must be divisible by ({cfg.compute.n_gpu_workers=})"

        assert (cfg.compute.n_cpu_workers * cfg.env.episode_length) % cfg.reward_model.reward_batch_size == 0, f"({cfg.compute.n_cpu_workers=}) * ({cfg.episode_length=}) must be divisible by ({cfg.reward_model.reward_batch_size=}) so that all batches are of the same size."

        if cfg.compute.n_gpu_workers == 1:
            assert cfg.reward_model.rank0_batch_size_pct == 1.0, f"When there's only one worker, rank0 has to handle the entire batch so {cfg.reward_model.rank0_batch_size_pct}"
        else:
            assert cfg.reward_model.rank0_batch_size_pct < 1.0, f"When there are multiple workers, rank0 should not have the entire batch so {cfg.reward_model.rank0_batch_size_pct} can't be 1."

        if cfg.reward_model.rank0_batch_size_pct < 1.0:
            assert float(((1 - cfg.reward_model.rank0_batch_size_pct) * cfg.reward_model.reward_batch_size)).is_integer(), f"({cfg.reward_model.reward_batch_size=}) needs to be an integer when multiplying by {cfg.reward_model.rank0_batch_size_pct=}"
            assert int((1 - cfg.reward_model.rank0_batch_size_pct) * cfg.reward_model.reward_batch_size) % (cfg.compute.n_gpu_workers - 1) == 0, f"({cfg.reward_model.reward_batch_size=}) corresponds to the total size of the batch do be distributed among workers and therefore must be divisible by ({cfg.compute.n_gpu_workers=})"

    cfg.logging.run_name = get_output_folder_name()
    cfg.logging.run_path = get_output_path()

    os.makedirs(os.path.join(cfg.logging.run_path, "eval"), exist_ok=True)

def get_make_env_kwargs(cfg: DictConfig):
    """
    Set the make_env_kwargs based on the config

    Parameters:
        cfg: DictConfig
            - The hydra config object
    """
    # if use_vlm_for_reward(cfg):
    #     make_env_kwargs = dict(
    #         episode_length = cfg.env.episode_length,
    #     )
    # else:
    #     make_env_kwargs = dict(
    #         max_episode_steps = cfg.env.episode_length,
    #     )
    make_env_kwargs = dict(
            episode_length = cfg.env.episode_length,
        )
    
    if "custom" in cfg.env.name.lower():
        make_env_kwargs["reward_type"] = cfg.env.reward_type
        if cfg.env.task_name:
            make_env_kwargs["task_name"] = cfg.env.task_name

    return make_env_kwargs

def calc_iqm(results):
    """
    Parameters:
        results: List (len = n_eval_env)
            The list of results to calculate the IQM from

    Return:
        The IQM of the results
        The std of the trimmed results that fall within the 25th and 75th percentile
    """
    # Step 1: Sort the results
    sorted_results = np.sort(results)
    
    # Step 2: Compute the 25th and 75th percentiles (interquartile range)
    q25 = np.percentile(sorted_results, 25)
    q75 = np.percentile(sorted_results, 75)
    
    # Step 3: Select data within the interquartile range
    trimmed_results = sorted_results[(sorted_results >= q25) & (sorted_results <= q75)]
    
    # Step 4: Compute the Interquartile Mean (IQM)
    iqm = np.mean(trimmed_results)
    
    # Step 5: Compute the standard deviation or variance (for trimmed data)
    std = np.std(trimmed_results)  # If you need the variance, use np.var(trimmed_results)
    
    return iqm, std