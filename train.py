import argparse
import functools
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
import wandb
import json
import yaml

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
# from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped, sync_envs_normalization
from sbx import SAC, VLM_SAC

from sbx.common.make_vec_env import make_vec_env
from sbx.common.subproc_vec_env import SubprocVecEnv
from sbx.sac.actor_critic_evaluation_callback import CriticBiasCallback, EvalCallback
from sbx.sac.utils import *

import gymnasium as gym
# from shimmy.registration import DM_CONTROL_SUITE_ENVS

from loguru import logger

import utils
import multiprocess
from envs.base import get_make_env
from sbx.vlm_reward.reward_main import load_reward_model, dist_worker_compute_reward
from callbacks import VideoRecorderCallback, WandbCallback
    

# TODO:
def primary_worker(run_name, args, stop_event: Optional[multiprocessing.Event] = None):
    """
    Load the environments
    Initialize wandb
    Define policy
    Define wandb callback fn
    Start learning
    Stop event catching
    """
    train_log_dir = os.path.join("./train_logs", run_name)
    logger.add(os.path.join(train_log_dir, "logs.txt"), enqueue=True)
    args, args_dict = utils.get_model_args_dict(args)

    use_vlm_for_reward = utils.vlm_for_reward(args)

    if use_vlm_for_reward:
        make_env_kwargs = dict(
            episode_length = args.episode_length,
        )
    else:
        make_env_kwargs = dict(
            max_episode_steps = args.episode_length,
        )

    if "custom" in args.env.lower():
        make_env_kwargs["reward_type"] = args.reward_type
    logger.info(f"Creating environment={args.env} instances with {make_env_kwargs=}")
    # TODO: Not sure if the thing below still works for vanilla RL
    # training_env = SubprocVecEnv([get_make_env(args.env, seed=args.seed+i, **make_env_kwargs) for i in range(args.n_envs)], start_method="spawn")
    make_env_fn = get_make_env(args.env, **make_env_kwargs)
    training_env = make_vec_env(
        make_env_fn,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        use_gpu_ids=list(range(args.n_workers)),
        vec_env_kwargs=dict(render_dim=(args.render_dim[0], args.render_dim[1], 3)),
    )

    import optax

    logger.info("Creating the learner...")
    # Train a model from scatch
    sac_class = VLM_SAC if use_vlm_for_reward else SAC
    model = sac_class(
        "MultiInputPolicy" if isinstance(training_env.observation_space, gym.spaces.Dict) else "MlpPolicy",
        args,
        training_env,
        policy_kwargs=dict({
            'activation_fn': activation_fn[args.critic_activation],  # From sbx.sac.utils import *
            'layer_norm': args_dict["layer_norm"],
            'batch_norm': bool(args.bn),
            'batch_norm_momentum': float(args.bn_momentum),
            'batch_norm_mode': args.bn_mode,
            'dropout_rate': args_dict["dropout_rate"],
            'n_critics': args.n_critics,
            'net_arch': args_dict["net_arch"],
            'optimizer_class': optax.adam,
            'optimizer_kwargs': dict({
                'b1': args.adam_b1,
                'b2': 0.999 # default
            })
        }),
        gradient_steps=args.utd,
        policy_delay=args.policy_delay,
        crossq_style=bool(args.crossq_style),
        td3_mode=args_dict["td3_mode"],
        use_bnstats_from_live_net=bool(args.bnstats_live_net),
        policy_q_reduce_fn=args_dict["policy_q_reduce_fn"],
        learning_starts=5000,
        learning_rate=args.lr,
        qf_learning_rate=args.lr,
        tau=args.tau,
        gamma=0.99 if not args.env == 'Swimmer-v4' else 0.9999,
        verbose=0,
        buffer_size=1_000_000,
        seed=args.seed,
        stats_window_size=1,  # don't smooth the episode return stats over time
        tensorboard_log=os.path.join("./train_logs", run_name),
    )
    # TODO: Not sure if .load() is better than .set_parameters()
    if args.model_base_path:
        existing_checkpoint_path = os.path.join(args.model_base_path, args.model_checkpoint)
        logger.info(f"Loading parameters from checkpoint: {existing_checkpoint_path}")
        model.set_parameters(existing_checkpoint_path)
    logger.debug(f"Created the learned and initialized if needed: allocated={round(torch.cuda.memory_allocated(0)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(0)/1024**3,1)}")
    

    with wandb.init(
        project=args.wandb_project,
        name=run_name,
        tags=[],
        sync_tensorboard=True,
        config=args_dict,
        mode=args.wandb_mode,
        monitor_gym=True,  # auto-upload the videos of agents playing the game
    ) as wandb_run:
        # TODO: This is not the most efficient setup
        eval_log_dir = os.path.join("./eval_logs", run_name, "eval")
        qbias_log_dir = os.path.join("./eval_logs", run_name, "qbias") # CrossQ doesn't actually use this
        checkpoint_dir = os.path.join("./train_logs", run_name, "checkpoint")

        # Create callback that evaluates agent
        # TODO: These callback somehow triggers compute_rewards
        #   since we are using the VideoCallback as evaluation, this is fine for now
        # eval_callback = EvalCallback(
        #     # TODO: this does not actually match the train env, so far the default value makes it ok
        #     # make_vec_env(args.env, n_envs=1, seed=args.seed, vec_env_cls=SubprocVecEnv),
        #     # make_env_fn(),
        #     SubprocVecEnv([make_env_fn], render_dim=(args.render_dim[0], args.render_dim[1], 3)),
        #     jax_random_key_for_seeds=args.seed,
        #     best_model_save_path=None,
        #     log_path=eval_log_dir, eval_freq=args.eval_freq // args.n_envs,
        #     n_eval_episodes=1, deterministic=True, render=False
        # )

        # Callback that evaluates q bias according to the REDQ paper.
        # q_bias_callback = CriticBiasCallback(
        #     # make_vec_env(args.env, n_envs=1, seed=args.seed, vec_env_cls=SubprocVecEnv), 
        #     make_env_fn(),
        #     jax_random_key_for_seeds=args.seed,
        #     best_model_save_path=None,
        #     log_path=qbias_log_dir, eval_freq=args_dict["eval_freq"],
        #     n_eval_episodes=1, render=False
        # )

        wandb_callback = WandbCallback(
            model_save_path=str(checkpoint_dir),
            model_save_freq=args.model_save_freq // args.n_envs,
            verbose=2,
        )

        eval_env = make_vec_env(
            make_env_fn,
            n_envs=args.n_envs,
            seed=args.seed,
            vec_env_cls=SubprocVecEnv,
            use_gpu_ids=list(range(args.n_workers)),
            vec_env_kwargs=dict(render_dim=(args.render_dim[0], args.render_dim[1], 3)),
        )

        video_callback = VideoRecorderCallback(
            eval_env=eval_env,
            render_freq=args.video_save_freq // args.n_envs,
        )


        callback_list = CallbackList(
            [wandb_callback, video_callback]
        )

        model.learn(total_timesteps=args.total_timesteps, progress_bar=False, callback=callback_list)

        if stop_event is not None:
            stop_event.set()

        logger.info("Saving final model")
        model.save(str(os.path.join(checkpoint_dir, "final_model")))

        logger.info("Done.")
        wandb_run.finish()


def vlm_inference_worker(run_name:str, rank: int, args, stop_event: multiprocessing.Event):
    train_log_dir = os.path.join("./train_logs", run_name)
    logger.add(os.path.join(train_log_dir, "logs.txt"), enqueue=True)

    logger.info(f"[Worker {rank}] Loading Reward model....")

    with open(args.reward_config, "r") as fin:
        model_config_dict = yaml.safe_load(fin)

    if args.rank0_batch_size_pct < 1.0:
        worker_batch_size = int((1 - args.rank0_batch_size_pct) * args.reward_batch_size) // (args.n_workers - 1)
    else:
        worker_batch_size = args.reward_batch_size // args.n_workers
    reward_model = load_reward_model(rank, 
                                        worker_actual_batch_size=worker_batch_size,  # Note that this is different size compared to rank 0's reward model when rank0_batch_size_pct < 1.0
                                        model_name=args.reward_model_name, 
                                        model_config_dict=model_config_dict).eval().cuda(rank)
    logger.debug(f"Loaded the reward model at rank={rank}: allocated={round(torch.cuda.memory_allocated(rank)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(rank)/1024**3,1)}")

    worker_frames_tensor = torch.zeros(
                (worker_batch_size, args.render_dim[0], args.render_dim[1], 3),
                dtype=torch.uint8,
            ).cuda(rank)
    while not stop_event.is_set():
        logger.info(f"[Worker {rank}] Entering wait for compute_embeddings_dist...")
        dist_worker_compute_reward(
            rank,
            rank0_batch_size_pct=args.rank0_batch_size_pct,
            reward_model=reward_model,
            render_dim=(args.render_dim[0], args.render_dim[1], 3),
            total_batch_size=args.reward_batch_size,  # Because this is not rank = 0, this helper doesn't actually use this value
            num_workers=args.n_workers,
            worker_frames_tensor=worker_frames_tensor,
        )
    logger.info(f"[Worker {rank}] Received stop event. Exiting worker")


def init_process(
    rank: int,
    stop_event: multiprocessing.Event,
    /,
    backend: str,
    run_name: str,
    args: argparse.Namespace,
):
    """Used by multiprocessing to spawn worker
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    # if backend == "nccl":
    # TODO: come back to this after fixing the kube setup
    # os.environ["NCCL_SHM_DISABLE"] = "1"
    dist.init_process_group(backend, rank=rank, world_size=args.n_workers)
    if rank == 0:
        primary_worker(run_name, args, stop_event)
    else:
        vlm_inference_worker(run_name, rank, args, stop_event)


if __name__ == "__main__":
    utils.set_os_vars()

    parser = argparse.ArgumentParser()
    parser.add_argument("-total_timesteps",   type=int,   required=False, default=5e6, help="total number of training steps")
    parser.add_argument("-episode_length",   type=int,   required=False, default=240, help="maximum timestep in an episode")

    parser.add_argument("-env",         type=str, required=False, default="HumanoidStandup-v4", help="Set Environment.")
    parser.add_argument("-reward_type", type=str, required=False, default="original", help='Type of rewards to use')
    parser.add_argument("-render_dim", type=int, nargs="+", default=[480, 480], help="Dimension of the rendered frames of the environment")
    
    parser.add_argument("-algo",        type=str, required=False, default='sac', choices=['crossq', 'sac', 'redq', 'droq', 'td3'], help="critic activation function")
    parser.add_argument("-seed",        type=int, required=False, default=1, help="Set Seed.")
    parser.add_argument("-n_envs",     type=int, required=False, default=1, help="Set up multiple env (one per cpu)")

    # Multiprocessing
    parser.add_argument("-n_workers",         type=int, required=False, default=1, help="Number of workers (GPUs) in the run")
    parser.add_argument("-rank0_batch_size_pct", type=float, required=False, default=1.0, help="Determine how many percentage of the batch the main worker (rank = 0) need to process when doing reward calculation")

    # VLM reward model
    parser.add_argument("-reward_model_name", type=str, required=False, default="", help="Name of the reward model")
    parser.add_argument("-reward_batch_size", type=int, required=False, help="Batch size sent to the VLM reward model")
    parser.add_argument("-reward_config", type=str, required=False, default="", help="Path to the reward config file")

    # Checkpoint and logging
    parser.add_argument("-model_checkpoint",  type=str, required=False, default="final_model", help="Model checkpoint zip file name (without .zip).")
    parser.add_argument("-model_base_path",        type=str, required=False, default="", help="Folder to all the checkpoints in a run.")

    parser.add_argument("-log_freq",    type=int, required=False, default=300, help="how many times to log during training")
    parser.add_argument("-eval_freq",    type=int, required=False, default=300, help="how many times to evaluate during training")
    parser.add_argument("-model_save_freq",   type=int, required=False, default=1e6, help="frequency to save the model")
    parser.add_argument("-video_save_freq",   type=int, required=False, default=1e6, help="frequency to save the model")

    # Wandb related
    parser.add_argument('-wandb_project', type=str, required=False, default='crossQ', help='wandb project name')
    parser.add_argument("-wandb_mode",    type=str, required=False, default='disabled', choices=['disabled', 'online'], help="enable/disable wandb logging")
    parser.add_argument("-eval_qbias",    type=int, required=False, default=0, choices=[0,1], help="enable/diasble q bias evaluation (expensive)")
    
    # Trainer related dataset
    parser.add_argument("-adam_b1",           type=float, required=False, default=0.5, help="adam b1 parameter")
    parser.add_argument("-bn",                type=float, required=False, default=False,  choices=[0,1], help="Use batch norm layers in the actor and critic networks")
    parser.add_argument("-bn_momentum",       type=float, required=False, default=0.99, help="batch norm momentum parameter")
    parser.add_argument("-bn_mode",           type=str,   required=False, default='brn_actor', help="batch norm mode (bn or brn)")
    parser.add_argument("-critic_activation", type=str,   required=False, default='relu', help="critic activation function")
    parser.add_argument("-crossq_style",      type=float, required=False, default=1,choices=[0,1], help="crossq style joint forward pass through critic network")
    parser.add_argument("-dropout",           type=int,   required=False, default=0, choices=[0,1], help="whether to use dropout for SAC")
    parser.add_argument("-ln",                type=float, required=False, default=False, choices=[0,1], help="layernorm in critic network")
    parser.add_argument("-lr",                type=float, required=False, default=1e-3, help="actor and critic learning rate")
    parser.add_argument("-n_critics",         type=int,   required=False, default=2, help="number of critics to use")
    parser.add_argument("-n_neurons",         type=int,   required=False, default=256, help="number of neurons for each critic layer)")
    parser.add_argument("-policy_delay",      type=int,   required=False, default=1, help="policy is updated after this many critic updates")
    parser.add_argument("-tau",               type=float, required=False, default=0.005, help="target network averaging")
    parser.add_argument("-utd",               type=int,   required=False, default=1, help="number of critic updates per env step (update to data ratio)")
    parser.add_argument("-bnstats_live_net",  type=int,   required=False, default=0,choices=[0,1], help="use bn running statistics from live network within the target network")

    parser.add_argument('--distributed', default=True, action=argparse.BooleanOptionalAction)
    # TODO: do some args validation

    experiment_time, run_id = utils.get_run_hash()
    args = parser.parse_args()

    utils.validate_args(args)

    if utils.vlm_for_reward(args):
        run_name = f"{args.algo}_{args.env}_rm={args.reward_model_name[:4]}_r={args.reward_type}_s={args.seed}_{experiment_time}_{run_id}"
    else:
        run_name = f"{args.algo}_{args.env}_r={args.reward_type}_s={args.seed}_{experiment_time}_{run_id}"

    # Create log dir where evaluation results will be saved
    eval_log_dir = os.path.join("./eval_logs", run_name, "eval")
    qbias_log_dir = os.path.join("./eval_logs", run_name, "qbias") # CrossQ doesn't actually use this
    checkpoint_dir = os.path.join("./train_logs", run_name, "checkpoint")
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(qbias_log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Register logging files and save parameteres
    train_log_dir = os.path.join("./train_logs", run_name)
    
    with open(os.path.join(train_log_dir, "params.json"), "w") as fout:
        params = json.dumps(vars(args), indent=4)
        logger.info(params)
        fout.write(params)

    logger.info(f"Started run with run_name={run_name}")

    @logger.catch
    def _train():
        use_vlm_for_reward = utils.vlm_for_reward(args)
        if use_vlm_for_reward:
            logger.info("Running VLM-rewarded RL. Spawning workers.")
            args_with_multiprocessing = ("nccl", run_name, args)
            multiprocess.spawn(
                fn=init_process,
                args=args_with_multiprocessing,
                nprocs=args.n_workers,
                join=True,
                daemon=False,
                start_method="spawn",
            )
        else:
            logger.info("Running RL for ground truth.")
            primary_worker(run_name, args)
    if args.distributed:
        _train()
    else:
        primary_worker(run_name, args)