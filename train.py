import utils

utils.set_os_vars()

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import wandb

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.global_hydra import GlobalHydra

import numpy as np

import torch
from torch import multiprocessing
import torch.distributed as dist

from stable_baselines3.common.callbacks import CallbackList

from custom_sb3 import SAC, VLM_SAC, JOINT_VLM_SAC
from stable_baselines3.sac.policies import MultiInputPolicy

from sbx.common.make_vec_env import make_vec_env
from sbx.common.subproc_vec_env import SubprocVecEnv

import gymnasium as gym


from loguru import logger

import multiprocess
from envs.base import get_make_env
from vlm_reward.reward_models.model_factory import load_reward_model
from vlm_reward.reward_main import dist_worker_compute_reward
from callbacks import VideoRecorderCallback, WandbCallback, JointBasedSeqRewardCallback
from constants import REWARDS_TO_ENTRY_IN_SEQ

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
    logger.add(os.path.join(cfg.logging.run_path, "logs.txt"), enqueue=True)

    # Initialize the environment
    use_vlm_for_reward = utils.use_vlm_for_reward(cfg)
    use_joint_vlm_for_reward = utils.use_joint_vlm_for_reward(cfg)

    logger.info(f"using_vlm_for_reward={use_vlm_for_reward}")
    logger.info(f"using vlm to predict joint pos: {use_joint_vlm_for_reward}")

    make_env_kwargs = utils.get_make_env_kwargs(cfg)

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

    logger.info("Creating the learner...")

    assert cfg.rl_algo.name == "sb3_sac", "Only StableBaseline3 SAC is supported for now"
    # Train a model from scatch

    ref_joint_states = None
    if use_joint_vlm_for_reward:
        sac_class = JOINT_VLM_SAC
        ref_joint_states = torch.as_tensor(np.load(cfg.reward_model.target_joint_state))
    elif use_vlm_for_reward:
        sac_class = VLM_SAC
    else:
        sac_class = SAC
    

    model = sac_class(
        MultiInputPolicy if isinstance(training_env.observation_space, gym.spaces.Dict) else "MlpPolicy",
        training_env,
        learning_rate=cfg.rl_algo.lr,
        buffer_size=1_000_000,
        learning_starts=5000,
        batch_size=256,
        tau=cfg.rl_algo.tau,
        gamma=0.99,
        train_freq=(cfg.env.episode_length, "step"),
        gradient_steps=cfg.env.episode_length,
        stats_window_size=1,  # don't smooth the episode return stats over time
        tensorboard_log=os.path.join(cfg.logging.run_path, "tensorboard"),
        policy_kwargs=dict({
            'activation_fn': torch.nn.ReLU,
            'net_arch': dict(cfg.rl_algo.net_arch),
            'n_critics': cfg.rl_algo.n_critics,
        }),
        verbose=0,
        seed=cfg.seed,
        ### VLM_SAC specific reward (SAC will ignore this)
        inference_only=False,
        reward_model_config = OmegaConf.to_container(cfg.reward_model, resolve=True, throw_on_missing=True) if use_vlm_for_reward else None,
        n_cpu_workers = cfg.compute.n_cpu_workers,
        n_gpu_workers = cfg.compute.n_gpu_workers,
        episode_length = cfg.env.episode_length,
        render_dim = cfg.env.render_dim,
        add_to_gt_rewards = cfg.reward_model.add_to_gt_rewards if use_vlm_for_reward else False,
        ref_joint_states=ref_joint_states
    )

    # TODO: Not sure if .load() is better than .set_parameters()
    if cfg.model_base_path:
        existing_checkpoint_path = os.path.join(cfg.model_base_path, cfg.model_checkpoint)
        logger.info(f"Loading parameters from checkpoint: {existing_checkpoint_path}")
        model.set_parameters(existing_checkpoint_path)
    logger.debug(f"Created the learned and initialized if needed: allocated={round(torch.cuda.memory_allocated(0)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(0)/1024**3,1)}")
    
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
        
        goal_seq_name = REWARDS_TO_ENTRY_IN_SEQ[cfg.env.reward_type] if "reward_type" in cfg.env and cfg.env.reward_type in REWARDS_TO_ENTRY_IN_SEQ else ""

        # If it's a goal reaching task
        # For non-goal reaching reward, we should set the goal sequence name to be the final image only
        if not ("goal_only" in cfg.env.reward_type):
            if "basic_r" in cfg.env.reward_type:
                goal_only_reward_type = cfg.env.reward_type.replace("_basic_r", "_goal_only_euclidean")
            elif "seq" in cfg.env.reward_type:
                base_name = cfg.env.reward_type.split("_seq")[0]
                goal_only_reward_type = base_name + "_goal_only_euclidean"
            else:
                goal_only_reward_type = ""

            if goal_only_reward_type in REWARDS_TO_ENTRY_IN_SEQ:
                goal_seq_name = REWARDS_TO_ENTRY_IN_SEQ[goal_only_reward_type]
            else:
                goal_seq_name = ""
        else:
            goal_seq_name = ""

        video_callback = VideoRecorderCallback(
            SubprocVecEnv([make_env_fn], render_dim=(cfg.env.render_dim[0], cfg.env.render_dim[1], 3)),
            rollout_save_path=os.path.join(cfg.logging.run_path, "eval"),
            render_freq=cfg.logging.video_save_freq // cfg.compute.n_cpu_workers,
            # This allow us to calculate the unifying reward/metric that all methods are compared against
            #   i.e. it defines "rollout/sum_total_reward_per_epsisode" in wandb
            goal_seq_name=REWARDS_TO_ENTRY_IN_SEQ[cfg.env.reward_type] if "reward_type" in cfg.env and cfg.env["reward_type"] in REWARDS_TO_ENTRY_IN_SEQ else "",
            threshold=cfg.env.pose_matching_stage_threshold,
            use_geom_xpos="geom_xpos" in cfg.env.reward_type if "reward_type" in cfg.env else False,
            # For joint based reward (this allow us to visualize the sequence matching reward in a rollout
            seq_name=cfg.reward_model.seq_name if cfg.reward_model.name == "joint_wasserstein" or cfg.reward_model.name == "joint_soft_dtw" else "",
            matching_fn_cfg=dict(cfg.reward_model) if cfg.reward_model.name == "joint_wasserstein" or cfg.reward_model.name == "joint_soft_dtw" else {},
        )

        callback_list = [wandb_callback, video_callback]

        if cfg.reward_model.name == "joint_wasserstein" or cfg.reward_model.name == "joint_soft_dtw":
            # Add the OT reward callback if we are using joint_wasserstein as the reward model
            callback_list.append(JointBasedSeqRewardCallback(
                                    seq_name = cfg.reward_model.seq_name,
                                    matching_fn_cfg = dict(cfg.reward_model),
                                    use_geom_xpos = "geom_xpos" in cfg.env.reward_type
            ))

        model.learn(
            total_timesteps=cfg.total_timesteps, 
            progress_bar=True, 
            callback=CallbackList(callback_list),
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
    logger.add(os.path.join(cfg.logging.run_path, "logs.txt"), enqueue=True)

    logger.info(f"[Worker {rank}] Loading Reward model....")

    if cfg.reward_model.rank0_batch_size_pct < 1.0:
        worker_batch_size = int((1 - cfg.reward_model.rank0_batch_size_pct) * cfg.reward_model.reward_batch_size) // (cfg.compute.n_gpu_workers - 1)
    else:
        worker_batch_size = cfg.reward_model.reward_batch_size // cfg.compute.n_gpu_workers
    
    reward_model = load_reward_model(rank, 
                                        worker_actual_batch_size=worker_batch_size,  # Note that this is different size compared to rank 0's reward model when rank0_batch_size_pct < 1.0
                                        model_name=cfg.reward_model.name, 
                                        model_config_dict=OmegaConf.to_container(cfg.reward_model, resolve=True, throw_on_missing=True))
    
    reward_model.eval()
    reward_model.cuda(rank)
    
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
    torch.cuda.set_device(rank)

    dist.init_process_group(backend, rank=rank, world_size=cfg.compute.n_gpu_workers)
    if rank == 0:
        primary_worker(cfg, stop_event)
    else:
        vlm_inference_worker(rank, cfg, stop_event)


@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg: DictConfig):
    utils.validate_and_preprocess_cfg(cfg)

    logger.info(f"Started run with run_name={cfg.logging.run_path}")

    @logger.catch
    def _train():
        use_vlm_for_reward = utils.use_vlm_for_reward(cfg)
        if use_vlm_for_reward:
            logger.info("Running VLM-rewarded RL. Spawning workers.")
            args_with_multiprocessing = ("nccl", cfg)
            multiprocess.spawn(
                fn=init_process,
                args=args_with_multiprocessing,
                nprocs=cfg.compute.n_gpu_workers,
                join=True,
                daemon=False,
                start_method="spawn",
            )
        else:
            logger.info("Running RL for ground truth.")
            primary_worker(cfg)

    if cfg.compute.n_gpu_workers > 1:
        _train()
    else: # If only 1 worker, no need to spawn process on each worker 
        primary_worker(cfg)


if __name__ == "__main__":
    utils.set_os_vars()

    # solve a weird bug that sometimes occurs with global hydra being already initialized
    GlobalHydra.instance().clear()
    main()