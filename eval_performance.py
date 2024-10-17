"""
Usage to plot workshop results

Create plots for the workshop paper's joint-based distance metrics
    python eval_performance.py -w

Create plots for the workshop paper's visual-based distance metrics
    python eval_performance.py -w -v
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from typing import List, Tuple
import os
import glob
import yaml
from seq_reward.seq_utils import load_reference_seq
from seq_reward.cost_fns import euclidean_distance_advanced
import numpy as np
import scipy.stats as stats
from torchvision.utils import save_image
import argparse
import pdb

def weighted_euclidean_distance(batch_1, batch_2, weights):
    """
    Computes the weighted distance matrix between two batches of samples.
    
    Parameters:
    - batch_1: np.ndarray of shape (batch_1, n_joints, joint_dim)
    - batch_2: np.ndarray of shape (batch_2, n_joints, joint_dim)
    - weights: np.ndarray of shape (n_joints, joint_dim)
    
    Returns:
    - distance_matrix: np.ndarray of shape (batch_1, batch_2)
    """
    batch_1_size, n_joints, joint_dim = batch_1.shape
    batch_2_size = batch_2.shape[0]
    
    # Initialize the distance matrix of shape (batch_1, batch_2)
    distance_matrix = np.zeros((batch_1_size, batch_2_size))
    
    # Compute the weighted squared distance for each pair of samples
    for i in range(batch_1_size):
        for j in range(batch_2_size):
            diff = batch_1[i] - batch_2[j]  # Shape: (n_joints, joint_dim)
            weighted_diff = weights * diff  # Element-wise weighting
            distance_matrix[i, j] = np.linalg.norm(weighted_diff) # norm of difference to get distance
    
    return distance_matrix

def rollout_matching_metric(obs_seq, ref_seq, threshold, N_timesteps_threshold=5, joint_weights=None):
    """
    Calculate the binary success based on the rollout and the reference sequence

    Parameters:
        rollout: np.array (rollout_length, ...)
            The rollout sequence to calculate the reward

    Return:
        pct_stage_completed: float
            The percentage of stages that are completed
        pct_timesteps_completing_the_stages: float
            The percentage of timesteps that are completing the stages
    """
    # Calculate reward from the rollout to self.goal_ref_seq
    if joint_weights is None:
        joint_weights = np.ones_like(ref_seq[0]) # default weights are just 1 (evenly distributed)
    reward_matrix = np.exp(-weighted_euclidean_distance(obs_seq, ref_seq, joint_weights)) # euclidean_distance_advanced
    # Detect when a stage is completed (the rollout is close to the goal_ref_seq) (under self._threshold)
    current_stage = 0
    stage_completed = 0
    # Track the number of steps where a stage is being completed
    #   Offset by 1 to play nicely with the stage_completed
    n_steps_completing_each_stage = [0] * (len(ref_seq) + 1)
    i=0
    while i + N_timesteps_threshold < len(reward_matrix):  # Iterate through the timestep
        if np.all(reward_matrix[i: i+N_timesteps_threshold, current_stage] > threshold) and stage_completed < len(ref_seq):
            stage_completed += 1
            current_stage = min(current_stage + 1, len(ref_seq)-1)
            # stage_completed-1 because stage_completed is counting the number of stages completed
            n_steps_completing_each_stage[stage_completed] += 1

            i += N_timesteps_threshold
        elif len(ref_seq) == 1 and np.all(reward_matrix[i: i+N_timesteps_threshold, current_stage] > threshold):
            # If there's only 1 stage
            n_steps_completing_each_stage[stage_completed] += 1
            i+= N_timesteps_threshold
        elif current_stage > 0 and np.all(reward_matrix[i: i+N_timesteps_threshold, current_stage-1] > threshold):
            # Once at least 1 stage is counted, if it's still above the threshold for the current stage, we will add to the count
            n_steps_completing_each_stage[stage_completed] += 1
            i+= N_timesteps_threshold
        else:
            i+=1
    pct_stage_completed = stage_completed/len(ref_seq)

    # The last pose is never reached
    if n_steps_completing_each_stage[-1] == 0:
        # We don't count any of the previous stage's steps
        pct_timesteps_completing_the_stages = 0
    else:
        pct_timesteps_completing_the_stages = np.sum(n_steps_completing_each_stage)/len(ref_seq)

    # IMPORTANT: pct_timesteps_completing is broken now because of changes to include min number of stagtes
    return pct_stage_completed#, pct_timesteps_completing_the_stages 

def rollout_matching_metric_with_torso_height(obs_seq, ref_seq, obs_qpos, threshold, N_timesteps_threshold=5, joint_weights=None, min_torso_height=1.1):
    """
    Calculate the binary success based on the rollout and the reference sequence

    Parameters:
        rollout: np.array (rollout_length, ...)
            The rollout sequence to calculate the reward
        
        obs_qpos: np.array (n_joints, joint_dim)

    Return:
        pct_stage_completed: float
            The percentage of stages that are completed
        pct_timesteps_completing_the_stages: float
            The percentage of timesteps that are completing the stages
    """
    # Calculate reward from the rollout to self.goal_ref_seq
    if joint_weights is None:
        joint_weights = np.ones_like(ref_seq[0]) # default weights are just 1 (evenly distributed)
    obs_seq_torso_height = obs_qpos[:, 0]  # Torso height is at index 0 in qpos (states returned from the env)
    torso_above_min = obs_seq_torso_height > min_torso_height  # shape: (rollout_length,)
    # reward_matrix is of shape (rollout_length, ref_seq_length)
    reward_matrix = np.exp(-weighted_euclidean_distance(obs_seq, ref_seq, joint_weights)) # euclidean_distance_advanced
    # reshape the torso_above_min to match the shape of the reward_matrix
    torso_above_min = np.repeat(torso_above_min[:, None], reward_matrix.shape[1], axis=1)

    reward_matrix = torso_above_min * reward_matrix
    # Detect when a stage is completed (the rollout is close to the goal_ref_seq) (under self._threshold)
    current_stage = 0
    stage_completed = 0
    # Track the number of steps where a stage is being completed
    #   Offset by 1 to play nicely with the stage_completed
    n_steps_completing_each_stage = [0] * (len(ref_seq) + 1)
    i=0
    while i + N_timesteps_threshold < len(reward_matrix):  # Iterate through the timestep
        # print(f"[BEFORE] i: {i}, reward_matrix[i: i+N_timesteps_threshold, current_stage]: {reward_matrix[i: i+N_timesteps_threshold, current_stage]}, stage_completed: {stage_completed}, current_stage: {current_stage}")
        # Case 1: The rollout has stayed above the threshold for the current stage for N_timesteps_threshold
        if np.all(reward_matrix[i: i+N_timesteps_threshold, current_stage] > threshold) and stage_completed < len(ref_seq):
            stage_completed += 1
            current_stage = min(current_stage + 1, len(ref_seq)-1)
            n_steps_completing_each_stage[stage_completed] += N_timesteps_threshold

            i += N_timesteps_threshold
        # Case 2: When we are at the last stage:
        #   We don't increment the stage_completed since the last_stage is already completed
        #   However, we keep track of the number of steps that are completing the last stage
        elif current_stage == len(ref_seq)-1 and np.all(reward_matrix[i: i+N_timesteps_threshold, current_stage] > threshold):
            n_steps_completing_each_stage[stage_completed] += N_timesteps_threshold
            i+= N_timesteps_threshold
        # Case 3: Once at least 1 stage is counted, but we have not moved on to the next stage
        #   if it's still above the threshold for the current stage, we will add to the count
        elif current_stage > 0 and np.all(reward_matrix[i: i+N_timesteps_threshold, current_stage-1] > threshold):
            n_steps_completing_each_stage[stage_completed] += N_timesteps_threshold
            i+= N_timesteps_threshold
        else:
            i+=1
        # print(f"[AFTER] i: {i}, reward_matrix[i: i+N_timesteps_threshold, current_stage]: {reward_matrix[i: i+N_timesteps_threshold, current_stage]}, stage_completed: {stage_completed}, current_stage: {current_stage}")
        # input("stop")

    pct_stage_completed = stage_completed/len(ref_seq)

    # print(f"FINAL: pct_stage_completed: {pct_stage_completed}, n_steps_completing_each_stage: {n_steps_completing_each_stage}")
    # input("FINAL")

    # The last pose is never reached
    if n_steps_completing_each_stage[-1] == 0:
        # We don't count any of the previous stage's steps
        pct_timesteps_completing_the_stages = 0
    else:
        pct_timesteps_completing_the_stages = np.sum(n_steps_completing_each_stage)/len(ref_seq)

    return pct_stage_completed, pct_timesteps_completing_the_stages 


# given a reward matrix, threshold, and number of steps K, find the max number of distinct frames in the reward matrix where reward_matrix[i:i+K] > threshold


def extract_timestep(filename: str) -> int:
    """Extract timestep from filename of format '{timestep}_rollouts_geom_xpos_states.npy'"""
    match = re.match(r'(\d+)_rollouts_geom_xpos_states\.npy', Path(filename).name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Invalid filename format: {filename}")

def extract_exp_label_from_dir(exp_dir):
    return Path(exp_dir).name.split('=')[-1]


def plot_multiple_directories(directory_results, 
                            output_file: str = 'multi_directory_performance.png',
                            labels: List[str] = [],
                            title: str = "",
                            smoothing=5):
    """
    Create and save plot comparing performance across multiple directories
    
    Args:
        directory_results: Dictionary mapping directory names to (timesteps, performances) tuples
        output_file: Path to save the output plot
    """
    
    # Get a good color palette for the number of directories
    colors = plt.get_cmap('Dark2').colors
    
    min_last_timestep = float('inf')

    # Hack to make zip work well
    if labels:
        label_ids = list(range(len(labels)))
    else:
        label_ids = list(range(len(directory_results)))
    
    # Plot each directory's data
    for (dir_name, (performances, lower, upper, timesteps)), color, label_id in zip(directory_results.items(), colors, label_ids):
        # Skip every other performance
        performances = performances[1:][::2]
        lower = lower[1:][::2]
        upper = upper[1:][::2]
        timesteps = timesteps[1:][::2]

        # Plot main line with confidence band
        timesteps = np.array(timesteps)
        # performances = smooth(np.array(performances), alpha=smoothing)
        # lower = smooth(np.array(lower), alpha=smoothing)
        # upper = smooth(np.array(upper), alpha=smoothing)
        window_size = 3
        performances = smooth_with_pd_rolling(np.array(performances), window_size)
        lower = smooth_with_pd_rolling(np.array(lower), window_size)
        upper = smooth_with_pd_rolling(np.array(upper), window_size)

        print(f"After Smoothing Performance: {np.array(performances).shape}")

        # Plot confidence interval
        plt.fill_between(timesteps, lower, upper, color=color, alpha=0.2)
      
        # Plot the main line
        if labels:
            exp_label = labels[label_id]
        else:
            exp_label = extract_exp_label_from_dir(dir_name)
        plt.plot(timesteps, performances, color=color, linewidth=1.5, 
                label=exp_label, alpha=0.8)

        min_last_timestep = min(max(timesteps), min_last_timestep)
    
    # Customize plot
    ax = plt.gca()
    ax.set_xlim([0, min_last_timestep]) # Constrain to the shortest sequence (in case some are 2M long)

    plt.xlabel('Environment Steps', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title(title if title else 'Geometric State Performance Comparison', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust legend
    # plt.legent()

    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved as {output_file}")

def load_rollouts(directory: str):
    """
    Load all state files and compute performance against target state.
    
    Args:
        directory: Directory containing the rollout state files
        target_file: Path to the target state file
        
    Returns:
        Tuple of lists: (timesteps, performances)
    """

    # Find all matching files
    pattern = str(os.path.join(directory, "eval", "*_rollouts_geom_xpos_states.npy"))
    files = glob.glob(pattern)
    if not files:
        raise ValueError(f"No matching files found in {directory}")
    
    # Process each file
    timesteps = []
    states = []
    qposes = []
    
    for file in sorted(files, key=extract_timestep):
        timestep = extract_timestep(file)
        state = np.load(file)
        qpos_fp = file.replace("geom_xpos_states.npy", "states.npy")
        qpos = np.load(qpos_fp)  # When it's loaded, it's in the shape (n_timesteps, n_envs, n_qpos)
        # We reshape it to match the shape of the state
        qpos = np.transpose(qpos, (1, 0, 2))
        states.append(state)
        qposes.append(qpos)
        timesteps.append(timestep)

    return np.stack(states), np.array(timesteps), np.array(qposes)
        
def load_ref(directory: str, seq_name: str = ""):
    config_path = os.path.join(directory, ".hydra", "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    task_name = config.get("env").get("task_name", "right_arm_extend_wave_higher")
    if seq_name == "":
        seq_name = config.get("reward_model").get("seq_name", "key_frames")
    use_geom_xpos = True

    # Because we want to compare all runs against the same reference sequence, this reference sequence should be key_frames
    ref = load_reference_seq(task_name=task_name, seq_name=seq_name, use_geom_xpos=use_geom_xpos)

    print(f"Loaded reference sequence of shape {ref.shape}")
    return ref

def smooth(x, alpha:int):
    if alpha > 1:
        """
        smooth data with moving window average.
        that is, smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(alpha)
        z = np.ones(len(x))
        smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
        return smoothed_x
    return x

def smooth_with_pd_rolling(data, window_size):
    import pandas as pd
    data = pd.Series(data)
    return data.rolling(window=window_size).mean()

def interquartile_mean_and_ci(values, confidence=0.95):
    # Sort the array
    sorted_values = np.sort(values)
    
    # Calculate the first and third quartile
    Q1 = np.percentile(sorted_values, 25)
    Q3 = np.percentile(sorted_values, 75)
    
    # Get the values between Q1 and Q3 (inclusive)
    interquartile_values = sorted_values[(sorted_values >= Q1) & (sorted_values <= Q3)]
    
    # Compute the interquartile mean
    interquartile_mean = np.mean(interquartile_values)
    
    # Compute the sample mean and standard error of the mean (SEM)
    sample_mean = np.mean(values)
    sem = stats.sem(values)  # Standard Error of the Mean
    
    # Compute the margin of error for the 95% confidence interval
    margin_of_error = sem * stats.t.ppf((1 + confidence) / 2., len(values)-1)
    
    # Compute the confidence interval
    ci_lower = sample_mean - margin_of_error
    ci_upper = sample_mean + margin_of_error
    
    return interquartile_mean, ci_lower, ci_upper

def mean_and_ci(values, confidence=0.95):
    """Calculate the mean and confidence interval of a list of values"""
    sample_mean = np.mean(values)
    sem = stats.sem(values)  # Standard Error of the Mean
    
    # Compute the margin of error for the 95% confidence interval
    margin_of_error = sem * stats.t.ppf((1 + confidence) / 2., len(values)-1)
    
    # Compute the confidence interval
    ci_lower = sample_mean - margin_of_error
    ci_upper = sample_mean + margin_of_error
    
    return sample_mean, ci_lower, ci_upper

def compute_performance(rollout_directory, performance_metric, ref_seq_name=""):
    
    ref = load_ref(rollout_directory, seq_name=ref_seq_name)
    rollouts, timesteps, rollout_qpos = load_rollouts(rollout_directory)

    all_rollout_performances_across_timesteps = []  # Store all the rollout performances across timesteps (# of timesteps, 8), where 8 is the number of eval runs per timestep
    performances = []
    cis_lower = []
    cis_upper = []
    for i in range(len(rollouts)):
        rollouts_for_a_timestep = rollouts[i]
        rollouts_qpos_for_a_timestep = rollout_qpos[i]

        rollout_performances = []
        if len(rollouts_for_a_timestep.shape) == 3: 
            # if there aren't multiple eval runs for this rollout
            # This is a hack to get around some older ground truth runs not having multiple eval samples
            # TODO: remove this, and get multiple runs for all (especially ground truth)
            rollouts_for_a_timestep = rollouts_for_a_timestep[None]

        for j in range(len(rollouts_for_a_timestep)):
            sample = rollouts_for_a_timestep[j]
            sample_qpos = rollouts_qpos_for_a_timestep[j]
            performance, _ = performance_metric(sample, ref, sample_qpos)
            rollout_performances.append(performance)

        all_rollout_performances_across_timesteps.extend(rollout_performances)

        # iqm, ci_lower, ci_upper = interquartile_mean_and_ci(rollout_performances)
        mean, ci_lower, ci_upper = mean_and_ci(rollout_performances)
        performances.append(mean)
        cis_lower.append(ci_lower)
        cis_upper.append(ci_upper)

    return performances, cis_lower, cis_upper, timesteps, all_rollout_performances_across_timesteps

def compute_performance_many_experiments(rollout_directories, performance_metric, ref_seq_name=""):
    """
    Return
        all_rollout_performances - Dictionary mapping rollout directories to (performances, cis_lower, cis_upper, timesteps)
        raw_all_rollout_performances - Dictionary mapping rollout directories to all_rollout_performances_across_timesteps
            Essentially, this is the raw success rate that was used to compute performances, cis_lower, and cis_upper
                We accumulate this so that we can plot the IQM (which is across all tasks)
    """
    all_rollout_performances = {}
    raw_all_rollout_performances = {}
    for rollout_directory in rollout_directories:
        print(f"Computing performance for {rollout_directory}")
        rollout_performances, cis_lower, cis_upper, timesteps, all_rollout_performances_across_timesteps = compute_performance(rollout_directory, performance_metric, ref_seq_name=ref_seq_name)
        all_rollout_performances[rollout_directory] = (rollout_performances,cis_lower, cis_upper, timesteps)
        raw_all_rollout_performances[rollout_directory] = all_rollout_performances_across_timesteps
    
    return all_rollout_performances, raw_all_rollout_performances

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--workshop", default=False,  action="store_true", help="Generate plots for the workshop experiments defined in workshop_experiments_folders.py")    
    parser.add_argument("-v", "--visual_result", default=False,  action="store_true", help="Generate plots for the visual reward results for the workshop experiments defined in workshop_experiments_folders.py")

    args = parser.parse_args()

    def metric(rollout, reference, rollout_qpos=None):
        """
        Compute based on the entire body's joint positions
        """
        N_timesteps_threshold = 3
        joint_distance_threshold = .45

        return rollout_matching_metric(rollout, reference, joint_distance_threshold, N_timesteps_threshold)
    
    def torso_metric(rollout, reference, rollout_qpos=None):
        """
        Compute based on the
        - Torso height
        - Arms
        """
        N_timesteps_threshold = 3
        joint_distance_threshold = .5
        joint_weights = np.zeros((18, 3))
        joint_weights[1] = 1  # torso height
        joint_weights[12:] = 1

        return rollout_matching_metric(rollout, reference, joint_distance_threshold, N_timesteps_threshold, joint_weights)
    def arms_metric(rollout, reference, rollout_qpos=None):
        """
        Compute based on the
        - Arms
        """
        N_timesteps_threshold = 3
        joint_distance_threshold = .5
        joint_weights = np.zeros((18, 3))
        joint_weights[12:] = 1 # Just the arms

        return rollout_matching_metric(rollout, reference, joint_distance_threshold, N_timesteps_threshold, joint_weights)
    
    def workshop_metric(rollout, reference, rollout_qpos):
        """
        Compute based on the
        - Torso height
        - Arms

        Arms are based on weighted euclidean distance. Torso height is an indicator function. 
            If the torso height is above a certain threshold, we look at arm based success. Otherwise, there's no success. 

        Because rollout is using geom_xpos (it has already been normalized with respect to the torso)
            Instead, to get the torso height, we will use the qpos of the rollout
        """
        N_timesteps_threshold = 3
        joint_distance_threshold = .5
        joint_weights = np.zeros((18, 3))
        joint_weights[12:] = 1
        min_torso_height = 1.1

        return rollout_matching_metric_with_torso_height(rollout, reference, rollout_qpos, joint_distance_threshold, N_timesteps_threshold, joint_weights, min_torso_height)
    
    plot_folder = "workshop_figs"

    if args.workshop:
        """
        Usage to plot workshop results (Based on the experiments defined in workshop_experiments_folders.py)

        Create plots for the workshop paper's joint-based distance metrics (the plots are stored in workshop_figs/joint_distance_metric_exp_figs/{task_name}/)
            python eval_performance.py -w

        Create plots for the workshop paper's visual-based distance metrics (the plots are stored in workshop_figs/visual_distance_metric_exp_figs/{task_name}/)
            python eval_performance.py -w -v
        """
        from workshop_experiments_folders import joint_based_experiments_dict, visual_based_experiments_dict, task_name_to_plot

        if args.visual_result:
            experiments_dict = visual_based_experiments_dict
            plot_folder = os.path.join(plot_folder, "visual_distance_metric_exp_figs")
        else:
            experiments_dict = joint_based_experiments_dict
            plot_folder = os.path.join(plot_folder, "joint_distance_metric_exp_figs")

        performance_metric_name = "torso-and-arms"
        performance_metric = workshop_metric

        for task_name in experiments_dict.keys():
            baseline = experiments_dict[task_name]['ground_truth_baseline']

            task_plot_folder = os.path.join(plot_folder, task_name)
            if not os.path.exists(task_plot_folder):
                os.makedirs(task_plot_folder)

            for sequence_type in experiments_dict[task_name].keys():
                if sequence_type != 'ground_truth_baseline':
                    print(f"==== Computing performance for {task_name} - {sequence_type} ====")

                    # baseline_labels = list(baseline.keys())
                    # baseline_dirs = [baseline[baseline_label] for baseline_label in baseline_labels]

                    exp_labels = list(experiments_dict[task_name][sequence_type].keys())
                    exp_dirs = [experiments_dict[task_name][sequence_type][exp_label] for exp_label in exp_labels]

                    # all_exp_labels = baseline_labels + exp_labels
                    # all_exp_dirs = baseline_dirs + exp_dirs
                    all_exp_labels = exp_labels
                    all_exp_dirs = exp_dirs

                    performance, raw_all_rollout_performances = compute_performance_many_experiments(all_exp_dirs, performance_metric, ref_seq_name=sequence_type)
                    
                    plot_file = os.path.join(task_plot_folder, f"{task_name}_{performance_metric_name}_{sequence_type}")
                    plot_multiple_directories(performance, labels=all_exp_labels, title=task_name_to_plot[task_name], output_file=plot_file) 

                    # breakpoint()                 
    else:
        experiment_directories = [
        "/share/portal/hw575/CrossQ/train_logs/2024-10-04-004735_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=right_arm_extend_wave_higher_rm=hand_engineered_nt=None", # training for reference rollout
        "/share/portal/hw575/CrossQ/train_logs/2024-09-26-152534_sb3_sac_envr=right_arm_extend_wave_higher_goal_only_euclidean_geom_xpos_rm=hand_engineered_s=9_nt=arms-only-geom", # baseline 
        "/share/portal/hw575/CrossQ/train_logs/2024-10-05-121349_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_last_60_frames_exp-r+bonus", 
        "/share/portal/hw575/CrossQ/train_logs/2024-10-05-121403_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_last_60_frames_exp-r",
        # "/share/portal/hw575/CrossQ/train_logs/2024-10-04-171641_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=rollout_19_frames_exp-r+bonus",
        # "/share/portal/hw575/CrossQ/train_logs/2024-10-04-171648_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=rollout_19_frames_exp-r"
        ]

        baseline_exps = [
            "/share/portal/hw575/CrossQ/train_logs/2024-10-04-004735_sb3_sac_envr=goal_only_euclidean_geom_xpos-t=right_arm_extend_wave_higher_rm=hand_engineered_nt=None", # training for reference 
            "/share/portal/hw575/CrossQ/train_logs/2024-09-26-152534_sb3_sac_envr=right_arm_extend_wave_higher_goal_only_euclidean_geom_xpos_rm=hand_engineered_s=9_nt=arms-only-geom", # baseline 
        ]
        

        # Write the sequence_types sorted by the number of frames
        sequence_types = [
            "intermediate_last_10_frames",
            "intermediate_10_frames",
            "rollout_9_frames",
            "intermediate_last_20_frames",
            "intermediate_20_frames",
            "rollout_19_frames",
            "intermediate_last_30_frames",
            "intermediate_30_frames",
            "rollout_29_frames",
            "intermediate_last_40_frames",
            "intermediate_40_frames",
            "rollout_39_frames",
            "intermediate_last_50_frames",
            "intermediate_50_frames",
            "intermediate_last_60_frames",
            "intermediate_60_frames",
            "rollout_59_frames",
        ]

        experiments = [
            [
                "/share/portal/hw575/CrossQ/train_logs/2024-10-05-031124_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_last_10_frames_exp-r+bonus",
                "/share/portal/hw575/CrossQ/train_logs/2024-10-05-031127_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_last_10_frames_exp-r"
            ],
            [
                "/share/portal/hw575/CrossQ/train_logs/2024-10-05-115210_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_10_frames_exp-r+bonus",
                "/share/portal/hw575/CrossQ/train_logs/2024-10-05-115212_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_10_frames_exp-r"
            ],
            [
                "/share/portal/hw575/CrossQ/train_logs/2024-10-04-171602_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=rollout_9_frames_exp-r+bonus",
                "/share/portal/hw575/CrossQ/train_logs/2024-10-04-171614_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=rollout_9_frames_exp-r"
            ],


            [
                "/share/portal/hw575/CrossQ/train_logs/2024-10-05-031158_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_last_20_frames_exp-r+bonus",
                "/share/portal/hw575/CrossQ/train_logs/2024-10-05-031250_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_last_20_frames_exp-r"
            ],
            [
                "/share/portal/hw575/CrossQ/train_logs/2024-10-04-182525_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_20_frames_exp-r+bonus",
                "/share/portal/hw575/CrossQ/train_logs/2024-10-04-182525_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_20_frames_exp-r"
            ],
            [
                "/share/portal/hw575/CrossQ/train_logs/2024-10-04-171641_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=rollout_19_frames_exp-r+bonus",
                "/share/portal/hw575/CrossQ/train_logs/2024-10-04-171648_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=rollout_19_frames_exp-r",
            ],


            [
                "/share/portal/hw575/CrossQ/train_logs/2024-10-05-031242_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_last_30_frames_exp-r+bonus",
                "/share/portal/hw575/CrossQ/train_logs/2024-10-05-031306_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_last_30_frames_exp-r"
            ],
            [
                "/share/portal/hw575/CrossQ/train_logs/2024-10-05-115214_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_30_frames_exp-r+bonus",
                "/share/portal/hw575/CrossQ/train_logs/2024-10-05-115319_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_30_frames_exp-r"
            ],
            [
                "/share/portal/hw575/CrossQ/train_logs/2024-10-05-120242_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=rollout_29_frames_exp-r+bonus",
                "/share/portal/hw575/CrossQ/train_logs/2024-10-05-120250_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=rollout_29_frames_exp-r"
            ],


            [
                "/share/portal/hw575/CrossQ/train_logs/2024-10-05-121307_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_last_40_frames_exp-r+bonus",
                "/share/portal/hw575/CrossQ/train_logs/2024-10-05-121305_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_last_40_frames_exp-r",
            ],
            [
                "/share/portal/hw575/CrossQ/train_logs/2024-10-05-115406_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_40_frames_exp-r+bonus",
                "/share/portal/hw575/CrossQ/train_logs/2024-10-05-115411_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_40_frames_exp-r"
            ],
            [
                "/share/portal/hw575/CrossQ/train_logs/2024-10-05-120306_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=rollout_39_frames_exp-r+bonus",
                "/share/portal/hw575/CrossQ/train_logs/2024-10-05-120308_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=rollout_39_frames_exp-r"
            ],


            [
                "/share/portal/hw575/CrossQ/train_logs/2024-10-05-121336_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_last_50_frames_exp-r+bonus",
                "/share/portal/hw575/CrossQ/train_logs/2024-10-05-121343_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_last_50_frames_exp-r"
            ],
            [
                "/share/portal/hw575/CrossQ/train_logs/2024-10-05-115654_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_50_frames_exp-r+bonus",
                "/share/portal/hw575/CrossQ/train_logs/2024-10-05-115658_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_50_frames_exp-r"
            ],


            [
                "/share/portal/hw575/CrossQ/train_logs/2024-10-05-121349_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_last_60_frames_exp-r+bonus",
                "/share/portal/hw575/CrossQ/train_logs/2024-10-05-121403_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_last_60_frames_exp-r"
            ],
            [
                "/share/portal/hw575/CrossQ/train_logs/2024-10-04-182539_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_60_frames_exp-r+bonus",
                "/share/portal/hw575/CrossQ/train_logs/2024-10-04-182533_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=intermediate_60_frames_exp-r"
            ],
            [
                "/share/portal/hw575/CrossQ/train_logs/2024-10-04-005215_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=rollout_59_frames_exp-r+bonus",
                "/share/portal/hw575/CrossQ/train_logs/2024-10-04-005301_sb3_sac_envr=basic_r_geom_xpos-t=right_arm_extend_wave_higher_rm=soft_dtw_nt=rollout_59_frames"
            ]
        ]

        plot_smoothing = 5
        plot_folder = "workshop_figs"
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        
        for exp_dirs, sequence_type in zip(experiments, sequence_types):
            for d, d_name in zip([metric, arms_metric, torso_metric], ["full_body_metric", "arm_metric", "torso_metric"]):
                
                all_exp_dirs = baseline_exps + exp_dirs
                performance = compute_performance_many_experiments(all_exp_dirs, d)
                plot_file = os.path.join(plot_folder, f"{d_name}_{sequence_type}")
                plot_multiple_directories(performance, output_file=plot_file, smoothing=plot_smoothing)