from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

from typing import Callable, List, Tuple, Dict, Optional

from seq_reward.optimal_transport import compute_ot_reward
from seq_reward.soft_dtw import compute_soft_dtw_reward

from seq_reward.seq_utils import load_reference_seq
from seq_reward.cost_fns import COST_FN_DICT, euclidean_distance_advanced


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
    use_geom_xpos: bool = False,
):
    """

    Parameters:
        ref
    """
    os.makedirs(eval_result_path, exist_ok=True)

    # Make an symlink to the gif path
    os.symlink(os.path.abspath(gif_path), os.path.join(eval_result_path, "rollout.gif"))

    # Load the saved .np trajectory states file
    traj = np.load(traj_path)
    if not use_geom_xpos:
        traj = traj[:, :22]
      
    # Load the path to the ground truth reward
    gt_reward = np.load(gt_reward_path)

    # Calculate the reward
    pred_reward, info = reward_fn(ref_seq, traj)

    # Plot the ground-truth vs predicted reward heat map
    gt_vs_source_heatmap(gt_reward, pred_reward, os.path.join(eval_result_path, "within_sequence_rewards.png"))
    
    if info != {}:
        # Plot the OT plan
        plot_matrix_as_heatmap(info["assignment"], f"{reward_fn_type} Assignment Matrix", os.path.join(eval_result_path, "assignment_plan.png"), cmap="hot")

        # Plot the cost matrix
        plot_matrix_as_heatmap(info["cost_matrix"], f"{reward_fn_type} Cost Matrix", os.path.join(eval_result_path, "cost_matrix.png"), cmap="bone_r")

        # Plot the transported cost matrix
        plot_matrix_as_heatmap(info["transported_cost"], f"{reward_fn_type} Transported Cost Matrix (Assignment * Cost)", os.path.join(eval_result_path, "transported_cost_matrix.png"), cmap="bone_r")
            

def eval_from_config(cfg: DictConfig):
    assert cfg.joint_eval_data.sequence_and_reward_dir is not None, "Please provide the path to the gif folder"

    logger.info(f"Using the following reward model:\n{OmegaConf.to_yaml(cfg.reward_model)}")

    # Define the reward function
    if cfg.reward_model.name == "joint_wasserstein":
        reward_fn = lambda ref, obs: compute_ot_reward(obs, ref, COST_FN_DICT[cfg.reward_model.cost_fn], scale=cfg.reward_model.scale, modification_dict=dict(cfg.reward_model.modification))
    elif cfg.reward_model.name == "joint_soft_dtw":
        reward_fn = lambda ref, obs: compute_soft_dtw_reward(obs, ref, COST_FN_DICT[cfg.reward_model.cost_fn], gamma=cfg.reward_model.gamma, scale=cfg.reward_model.scale, modification_dict=dict(cfg.reward_model.modification))
    else:
        def seq_matching_fn(ref, rollout, threshold=0.2):
            """
            Calculate the reward based on the sequence matching to the goal_ref_seq

            Parameters:
                rollout: np.array (rollout_length, ...)
                    The rollout sequence to calculate the reward
            """
            # Calculate reward from the rollout to self.goal_ref_seq
            reward_matrix = np.exp(-euclidean_distance_advanced(rollout, ref))

            # Detect when a stage is completed (the rollout is close to the goal_ref_seq) (under self._threshold)
            stage_completed = 0
            stage_completed_matrix = np.zeros(reward_matrix.shape) # 1 if the stage is completed, 0 otherwise
            current_stage_matrix = np.zeros(reward_matrix.shape) # 1 if the current stage, 0 otherwise
            
            for i in range(len(reward_matrix)):  # Iterate through the timestep
                current_stage_matrix[i, stage_completed] = 1
                if reward_matrix[i][stage_completed] > threshold and stage_completed < len(ref) - 1:
                    stage_completed += 1
                stage_completed_matrix[i, :stage_completed] = 1

            # Find the highest reward to each reference sequence
            highest_reward = np.max(reward_matrix, axis=0)

            # Reward (shape: (rollout)) at each timestep is
            #   Stage completion reward + Reward at the current stage
            reward = np.sum(stage_completed_matrix * highest_reward + current_stage_matrix * reward_matrix, axis=1)/len(ref)

            return reward, {}

        reward_fn = seq_matching_fn

    # Load the reference joint states
    ref_seq = load_reference_seq(cfg.joint_eval_data.name, cfg.use_geom_xpos)

    logger.debug(f"Reference sequence shape: {ref_seq.shape}")

    # Prune and get all the gifs in cfg.joint_eval_data.sequence_and_reward_dir
    #   (only get the gif files)
    all_gif_files = [f for f in os.listdir(cfg.joint_eval_data.sequence_and_reward_dir) if f.endswith(".gif")]

    # Filter the gif_files based on cfg.skip_gifs (how many gifs to skip)
    gif_files = [all_gif_files[i] for i in range(0, len(all_gif_files), cfg.skip_gifs)]
    # gif_files += [all_gif_files[72]]g

    data_save_dir = HydraConfig.get().runtime.output_dir

    for f in tqdm(gif_files):
        base_name = str(f).replace(".gif", "")

        if cfg.use_geom_xpos:
            # This geom_xpos_states.npy has already been normalized
            states_npy_fp = os.path.join(cfg.joint_eval_data.sequence_and_reward_dir, base_name + "_geom_xpos_states.npy")
        else:
            states_npy_fp = os.path.join(cfg.joint_eval_data.sequence_and_reward_dir, base_name + "_states.npy")

        rewards_npy_fp = os.path.join(cfg.joint_eval_data.sequence_and_reward_dir, base_name.replace("_rollouts", "") + "_goal_matching_reward.npy")
        # rewards_npy_fp = os.path.join(cfg.joint_eval_data.sequence_and_reward_dir, base_name + "_rewards.npy")

        eval_one_traj(
            ref_seq=ref_seq,
            gif_path=os.path.join(cfg.joint_eval_data.sequence_and_reward_dir, f),
            traj_path=states_npy_fp,
            gt_reward_path=rewards_npy_fp,
            reward_fn_type=cfg.reward_model.name,
            reward_fn=reward_fn,
            eval_result_path=os.path.join(data_save_dir, f),
            use_geom_xpos=cfg.use_geom_xpos,
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