from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

from typing import Callable, List, Tuple, Dict, Optional

from vlm_reward.utils.optimal_transport import load_reference_seq, compute_ot_reward, plot_ot_plan, COST_FN_DICT
from vlm_reward.utils.soft_dtw import compute_soft_dtw_reward

from vlm_reward.eval.eval_utils import gt_vs_source_heatmap

import numpy as np

import os

from loguru import logger
from tqdm import tqdm

def eval_one_traj(
    ref_seq: np.ndarray,
    gif_path: str,
    traj_path: str,
    gt_reward_path: str,
    reward_fn_type: str,
    reward_fn: callable,
    eval_result_path: str,
):
    """

    Parameters:
        ref
    """
    os.makedirs(eval_result_path, exist_ok=True)

    # Make an symlink to the gif path
    os.symlink(os.path.abspath(gif_path), os.path.join(eval_result_path, "rollout.gif"))

    # Load the saved .np trajectory states file
    traj = np.load(traj_path)[:, :22]
      
    # Load the path to the ground truth reward
    gt_reward = np.load(gt_reward_path)

    # Calculate the reward
    pred_reward, info = reward_fn(ref_seq, traj)

    # Plot the ground-truth vs predicted reward heat map
    gt_vs_source_heatmap(gt_reward, pred_reward, os.path.join(eval_result_path, "within_sequence_rewards.png"))

    # Plot the OT plan
    plot_ot_plan(info["assignment"], os.path.join(eval_result_path, "assignment_plan.png"))
        

def eval_from_config(cfg: DictConfig):
    assert cfg.joint_eval_data.sequence_and_reward_dir is not None, "Please provide the path to the gif folder"

    # Define the reward function
    dist_fn = COST_FN_DICT[cfg.reward_model.cost_fn]

    if cfg.reward_model.name == "joint_wasserstein":
        reward_fn = lambda ref, obs: compute_ot_reward(obs, ref, dist_fn, scale=cfg.reward_model.scale)
    elif cfg.reward_model.name == "joint_soft_dtw":
        reward_fn = lambda ref, obs: compute_soft_dtw_reward(obs, ref, dist_fn, gamma=cfg.reward_model.gamma, scale=cfg.reward_model.scale)

    # Load the reference joint states
    ref_seq = load_reference_seq(cfg.joint_eval_data.name)

    logger.debug(f"Reference sequence shape: {ref_seq.shape}")

    # Prune and get all the gifs in cfg.joint_eval_data.sequence_and_reward_dir
    #   (only get the gif files)
    gif_files = [f for f in os.listdir(cfg.joint_eval_data.sequence_and_reward_dir) if f.endswith(".gif")]

    # Filter the gif_files based on cfg.skip_gifs (how many gifs to skip)
    gif_files = [gif_files[i] for i in range(0, len(gif_files), cfg.skip_gifs)]

    data_save_dir = HydraConfig.get().runtime.output_dir

    for f in tqdm(gif_files):
        base_name = str(f).replace(".gif", "")

        states_npy_fp = os.path.join(cfg.joint_eval_data.sequence_and_reward_dir, base_name + "_states.npy")
        rewards_npy_fp = os.path.join(cfg.joint_eval_data.sequence_and_reward_dir, base_name + "_rewards.npy")

        eval_one_traj(
            ref_seq=ref_seq,
            gif_path=os.path.join(cfg.joint_eval_data.sequence_and_reward_dir, f),
            traj_path=states_npy_fp,
            gt_reward_path=rewards_npy_fp,
            reward_fn_type=cfg.reward_model.name,
            reward_fn=reward_fn,
            eval_result_path=os.path.join(data_save_dir, f),
        )


@hydra.main(version_base=None, config_path="configs", config_name="joint_eval_config")
def main(cfg: DictConfig):

    eval_from_config(cfg)


if __name__=="__main__":
    main()
    
    # ot_reward_fn = lambda ref, obs: compute_ot_reward(obs, ref, euclidean_distance, scale=10)

    # eval_one_traj(
    #     ref_seq_name = "both_arms_out_with_intermediate",
    #     traj_path = "train_logs/2024-08-26-235406_sb3-sac_envr=both_arms_out_goal_only_euclidean_rm=hand_engineered_s=9_nt=2M_debug_train-freq=120steps_gradient-step=120/eval/2000000_rollouts_states.npy",
    #     gt_reward_path = "train_logs/2024-08-26-235406_sb3-sac_envr=both_arms_out_goal_only_euclidean_rm=hand_engineered_s=9_nt=2M_debug_train-freq=120steps_gradient-step=120/eval/2000000_rollouts_rewards.npy",
    #     reward_fn_type = "ot",
    #     reward_fn = ot_reward_fn,
    #     eval_result_path = "vlm_reward/eval/joint_eval_logs/test_2000000",
    # )