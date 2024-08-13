import argparse
import functools
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
import wandb
import json
import yaml

from omegaconf import DictConfig, OmegaConf
import hydra

import torch
from torch import multiprocessing
import torch.distributed as dist

import numpy as np
import jax
import jax.numpy as jnp
import rlax
import flax.linen as nn

from stable_baselines3.common import type_aliases
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped, sync_envs_normalization
from sbx import SAC, VLM_SAC

from sbx.common.make_vec_env import make_vec_env
from sbx.common.subproc_vec_env import SubprocVecEnv
from sbx.sac.actor_critic_evaluation_callback import CriticBiasCallback, EvalCallback
from sbx.sac.utils import *

import gymnasium as gym

from loguru import logger

import utils
import multiprocess
from envs.base import get_make_env
from sbx.vlm_reward.reward_main import load_reward_model, dist_worker_compute_reward
from callbacks import VideoRecorderCallback, WandbCallback
    

def primary_worker(cfg: DictConfig, stop_event: Optional[multiprocessing.Event] = None):
    """
    This defines the main worker that will be used for training the agent

    Parameters:
        cfg: DictConfig
            - The hydra config object
        stop_event: multiprocessing.Event
            The event to signal the workers to stop
    """
    # Save logging also into a file
    logger.add(os.path.join(utils.get_output_path(), "logs.txt"), enqueue=True)

    # Initialize the environment
    use_vlm_for_reward = utils.use_vlm_for_reward(cfg)
    if use_vlm_for_reward:
        make_env_kwargs = dict(
            episode_length = cfg.env.episode_length,
        )
    else:
        make_env_kwargs = dict(
            max_episode_steps = cfg.env.episode_length,
        )
    if "custom" in cfg.env.name.lower():
        make_env_kwargs["reward_type"] = cfg.env.reward_type

    logger.info(f"Creating environment={cfg.env.name} instances with {make_env_kwargs=}")

    make_env_fn = get_make_env(cfg.env.name, **make_env_kwargs)
    training_env = make_vec_env(
        make_env_fn,
        n_envs=cfg.compute.n_cpu_workers,
        seed=cfg.seed,
        vec_env_cls=SubprocVecEnv,
        use_gpu_ids=list(range(cfg.compute.n_gpu_workers)),
        vec_env_kwargs=dict(render_dim=(cfg.env.render_dim[0], cfg.env.render_dim[1], 3)),
    )

    import optax

    logger.info("Creating the learner...")
    # Train a model from scatch
    sac_class = VLM_SAC if use_vlm_for_reward else SAC
    model = sac_class(
        "MultiInputPolicy" if isinstance(training_env.observation_space, gym.spaces.Dict) else "MlpPolicy",
        training_env,
        policy_kwargs=dict({
            'activation_fn': activation_fn[cfg.rl_algo.critic_activation],  # From sbx.sac.utils import *
            'layer_norm': cfg.rl_algo.ln,
            'batch_norm': bool(cfg.rl_algo.bn),
            'batch_norm_momentum': float(cfg.rl_algo.bn_momentum),
            'batch_norm_mode': cfg.rl_algo.bn_mode,
            'dropout_rate': cfg.rl_algo.dropout_rate,
            'n_critics': cfg.rl_algo.n_critics,
            'net_arch': cfg.rl_algo.net_arch,
            'optimizer_class': optax.adam,
            'optimizer_kwargs': dict({
                'b1': cfg.rl_algo.adam_b1,
                'b2': 0.999 # default
            })
        }),
        gradient_steps=cfg.rl_algo.utd,
        policy_delay=cfg.rl_algo.policy_delay,
        crossq_style=bool(cfg.rl_algo.crossq_style),
        td3_mode=cfg.rl_algo.td3_mode if "td3_mode" in cfg.rl_algo else False,
        use_bnstats_from_live_net=bool(cfg.rl_algo.bnstats_live_net),
        policy_q_reduce_fn=jax.numpy.min,  # Both CrossQ and SAC use min
        learning_starts=5000,
        learning_rate=cfg.rl_algo.lr,
        qf_learning_rate=cfg.rl_algo.lr,
        tau=cfg.rl_algo.tau,
        gamma=0.99,
        verbose=0,
        buffer_size=1_000_000,
        seed=cfg.seed,
        stats_window_size=1,  # don't smooth the episode return stats over time
        tensorboard_log=os.path.join(utils.get_output_path(), "tensorboard"),
        reward_model_config = cfg.reward_model if use_vlm_for_reward else None,
        n_cpu_workers = cfg.compute.n_cpu_workers,
        n_gpu_workers = cfg.compute.n_gpu_workers,
        episode_length = cfg.env.episode_length,
        render_dim = cfg.env.render_dim,
    )

    model.use_distributed = cfg.compute.distributed

    # TODO: Not sure if .load() is better than .set_parameters()
    if cfg.model_base_path:
        existing_checkpoint_path = os.path.join(cfg.model_base_path, cfg.model_checkpoint)
        logger.info(f"Loading parameters from checkpoint: {existing_checkpoint_path}")
        model.set_parameters(existing_checkpoint_path)
    logger.debug(f"Created the learned and initialized if needed: allocated={round(torch.cuda.memory_allocated(0)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(0)/1024**3,1)}")
    
    with wandb.init(
        project=cfg.logging.wandb_project,
        name=utils.get_output_folder_name(),
        tags=[],
        sync_tensorboard=True,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        mode=cfg.logging.wandb_mode,
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        dir=utils.get_output_path(),
    ) as wandb_run:
        checkpoint_dir = os.path.join(utils.get_output_path(), "checkpoint")

        wandb_callback = WandbCallback(
            model_save_path=str(checkpoint_dir),
            model_save_freq=cfg.logging.model_save_freq // cfg.compute.n_cpu_workers,
            verbose=2,
        )

        video_callback = VideoRecorderCallback(
            SubprocVecEnv([make_env_fn], render_dim=(cfg.env.render_dim[0], cfg.env.render_dim[1], 3)),
            render_freq=cfg.logging.video_save_freq // cfg.compute.n_cpu_workers,
        )

        model.learn(
            total_timesteps=cfg.total_timesteps, 
            progress_bar=True, 
            callback=CallbackList([wandb_callback, video_callback])
        )

        if stop_event is not None:
            stop_event.set()

        logger.info("Saving final model")
        model.save(str(os.path.join(checkpoint_dir, "final_model")))

        logger.info("Done.")
        wandb_run.finish()


def vlm_inference_worker(rank: int, cfg: DictConfig, stop_event: multiprocessing.Event):
    """
    Creates a VLM reward model and runs the inference (reward calculcation) on the frames sent by the main worker

    Parameters:
        rank: int
            The rank of the worker
        cfg: DictConfig
            The hydra config object
        stop_event: multiprocessing.Event
            The event to signal the workers to stop
    """
    # Save logging also into a file
    logger.add(os.path.join(utils.get_output_path(), "logs.txt"), enqueue=True)

    logger.info(f"[Worker {rank}] Loading Reward model....")

    if cfg.reward_model.rank0_batch_size_pct < 1.0:
        worker_batch_size = int((1 - cfg.reward_model.rank0_batch_size_pct) * cfg.reward_model.reward_batch_size) // (cfg.compute.n_gpu_workers - 1)
    else:
        worker_batch_size = cfg.reward_model.reward_batch_size // cfg.compute.n_gpu_workers
    
    reward_model = load_reward_model(rank, 
                                        worker_actual_batch_size=worker_batch_size,  # Note that this is different size compared to rank 0's reward model when rank0_batch_size_pct < 1.0
                                        model_name=cfg.reward_model.vlm_model, 
                                        model_config_dict=cfg.reward_model).eval().cuda(rank)
    logger.debug(f"Loaded the reward model at rank={rank}: allocated={round(torch.cuda.memory_allocated(rank)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(rank)/1024**3,1)}")

    worker_frames_tensor = torch.zeros(
                (worker_batch_size, cfg.env.render_dim[0], cfg.env.render_dim[1], 3),
                dtype=torch.uint8,
            ).cuda(rank)
    while not stop_event.is_set():
        logger.info(f"[Worker {rank}] Entering wait for compute_embeddings_dist...")
        dist_worker_compute_reward(
            rank,
            rank0_batch_size_pct=cfg.reward_model.rank0_batch_size_pct,
            reward_model=reward_model,
            render_dim=(cfg.env.render_dim[0], cfg.env.render_dim[1], 3),
            total_batch_size=cfg.reward_model.reward_batch_size,  # Because this is not rank = 0, this helper doesn't actually use this value
            num_workers=cfg.compute.n_gpu_workers,
            worker_frames_tensor=worker_frames_tensor,
        )
    logger.info(f"[Worker {rank}] Received stop event. Exiting worker")


def init_process(
    rank: int,
    stop_event: multiprocessing.Event,
    /,
    backend: str,
    cfg: DictConfig,
):
    """Used by multiprocessing to spawn worker
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    # if backend == "nccl":
    # TODO: come back to this after fixing the kube setup
    # os.environ["NCCL_SHM_DISABLE"] = "1"
    dist.init_process_group(backend, rank=rank, world_size=cfg.compute.n_gpu_workers)
    if rank == 0:
        primary_worker(cfg, stop_event)
    else:
        vlm_inference_worker(rank, cfg, stop_event)


@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg: DictConfig):
    utils.validate_cfg(cfg)

    logger.info(f"Started run with run_name={utils.get_output_path()}")

    @logger.catch
    def _train():
        use_vlm_for_reward = utils.use_vlm_for_reward(cfg)
        if use_vlm_for_reward:
            logger.info("Running VLM-rewarded RL. Spawning workers.")
            args_with_multiprocessing = ("nccl", cfg)
            multiprocess.spawn(
                fn=init_process,
                args=args_with_multiprocessing,
                nprocs=args.n_gpu_workers,
                join=True,
                daemon=False,
                start_method="spawn",
            )
        else:
            logger.info("Running RL for ground truth.")
            primary_worker(cfg)

    if cfg.compute.distributed:
        _train()
    else:
        primary_worker(cfg)


if __name__ == "__main__":
    utils.set_os_vars()

    main()