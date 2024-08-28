from omegaconf import DictConfig, OmegaConf
import hydra

from typing import Callable, List, Tuple, Dict, Optional

from vlm_reward.utils.optimal_transport import load_reference_seq, compute_ot_reward, plot_ot_plan, euclidean_distance

from vlm_reward.eval.eval_utils import gt_vs_source_heatmap

import numpy as np

import os

from loguru import logger

def eval_one_traj(
    ref_seq_name: str,
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
    
    # Load the reference joint states
    ref_seq = load_reference_seq(ref_seq_name)

    logger.debug(f"Reference sequence shape: {ref_seq.shape}")

    # Load the saved .np trajectory states file
    traj = np.load(traj_path)[:, :22]

    logger.debug(f"Trajectory sequence shape: {traj.shape}")
      
    # Load the path to the ground truth reward
    gt_reward = np.load(gt_reward_path)

    logger.debug(f"Ground truth reward shape: {gt_reward.shape}")

    # Calculate the reward
    pred_reward, info = reward_fn(ref_seq, traj)

    logger.debug(f"Predicted reward shape: {pred_reward.shape}")

    # Plot the ground-truth vs predicted reward heat map
    gt_vs_source_heatmap(gt_reward, pred_reward, os.path.join(eval_result_path, "within_sequence_rewards.png"))

    if reward_fn_type == "ot":
        # Plot the OT plan
        plot_ot_plan(info["T"], os.path.join(eval_result_path, "ot_plan.png"))


def eval_from_config(cfg: DictConfig):
    # Define the reward function
    reward_fn = lambda ref, obs: compute_ot_reward(obs, ref, euclidean_distance)


@hydra.main(version_base=None, config_path="configs", config_name="eval_config")
def main(cfg: DictConfig):

    eval_from_config(cfg)


if __name__=="__main__":
    ot_reward_fn = lambda ref, obs: compute_ot_reward(obs, ref, euclidean_distance)

    eval_one_traj(
        ref_seq_name = "both_arms_out_with_intermediate",
        traj_path = "train_logs/2024-08-26-235406_sb3-sac_envr=both_arms_out_goal_only_euclidean_rm=hand_engineered_s=9_nt=2M_debug_train-freq=120steps_gradient-step=120/eval/2000000_rollouts_states.npy",
        gt_reward_path = "train_logs/2024-08-26-235406_sb3-sac_envr=both_arms_out_goal_only_euclidean_rm=hand_engineered_s=9_nt=2M_debug_train-freq=120steps_gradient-step=120/eval/2000000_rollouts_rewards.npy",
        reward_fn_type = "ot",
        reward_fn = ot_reward_fn,
        eval_result_path = "vlm_reward/eval/joint_eval_logs/2000000",
    )