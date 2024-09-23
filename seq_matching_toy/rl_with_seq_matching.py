from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.global_hydra import GlobalHydra

import os
import wandb
from typing import Any, Dict, List, Union
from numpy.typing import NDArray
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList
from loguru import logger

from seq_matching_toy.toy_envs.grid_nav import *
from callbacks import WandbCallback

class GridNavReplayBuffer(RolloutBuffer):
    # Directly inherit from ReplayBuffer
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )
        self.render_arrays: List[NDArray] = []

    def add(
        self,
        obs,
        next_obs: NDArray,
        action: NDArray,
        reward: NDArray,
        done: NDArray,
        infos: List[Dict[str, Any]],
    ) -> None:
        super().add(
            obs,
            next_obs,
            action,
            reward,
            done,
            infos,
        )

        assert len(self.render_arrays) < self.buffer_size
        self.render_arrays.append([info["geom_xpos"] for info in infos])

    def clear_geom_xpos(self) -> None:
        self.geom_xpos = []

@hydra.main(version_base=None, config_path="configs", config_name="rl_config")
def train(cfg: DictConfig):
    # Initialize the environment
    training_env = make_vec_env(
        lambda: GridNavigationEnv(map_config=dict(env.map_config), ref_seq=[], render_mode="rgb_array", max_episode_steps=cfg.env.episode_length),
        n_envs=cfg.compute.n_cpu_workers,
        seed=cfg.seed,
        vec_env_cls=SubprocVecEnv,
        use_gpu_ids=list(range(cfg.compute.n_gpu_workers)),
    )

    # Define the model
    model = PPO("MlpPolicy", 
                training_env, 
                learning_rate=cfg.rl_algo.lr,
                rollout_buffer_class=
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

        callback_list = [wandb_callback]

        # Train the model
        model.learn(
            total_timesteps=cfg.total_timesteps,
            progress_bar=True, 
            callback=CallbackList(callback_list))

        logger.info("Saving final model")
        model.save(str(os.path.join(checkpoint_dir, "final_model")))

        logger.info("Done.")
        wandb_run.finish()


if __name__ == "__main__":
    train()
    # env = GridNavigationEnv(map_config={"name": "3x3"}, ref_seq=[], render_mode="rgb_array")
    # env.reset()
    # env.render()
    
    # path = [RIGHT, RIGHT, DOWN, DOWN, LEFT, LEFT, UP, STAY]

    # frames = []
    # frames.append(env.render())

    # for action in path:
    #     env.step(action)
    #     frames.append(env.render())

    # imageio.mimsave("testing.gif", frames, duration=1/20, loop=0)

    # writer = imageio.get_writer('testing.mp4', fps=20)

    # for im in frames:
    #     writer.append_data(im)
    
    # writer.close()