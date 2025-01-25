

import pandas as pd
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

from eval_performance import compute_performance_many_experiments, plot_multiple_directories, workshop_metric, weighted_euclidean_distance

APPROACH_COLOR_DICT = {
    "ORCA+TOT pretrained (500k-500k)": "#19825F",
    "ORCA": "#83CE74",
    "TemporalOT": "#69B3FF",
    "OT": "#7D5CBD",
    "DTW": "#E3247F",
    "Threshold": "#EF772B",
    "RoboCLIP": "#5B5B5B"
}
APPROACH_NAME_TO_PLOT = {
    "ORCA+TOT pretrained (500k-500k)": "ORCA(P)",
    "ORCA": "ORCA",
    "TemporalOT": "TOT",
    "OT": "OT",
    "DTW": "DTW",
    "Threshold": "Threshold",
    "RoboCLIP": "RoboCLIP"
}

OUTPUT_DIR = "eval/sparse_metric"


def sparse_metric(rollout, reference, rollout_qpos):
    joint_distance_threshold = .8
    joint_weights = np.zeros((18, 3))
    joint_weights[12:] = 1 # Just the arms
    min_torso_height = 1.1

    distance_matrix = weighted_euclidean_distance(rollout, reference, joint_weights)
    arm_successes = distance_matrix[:, -1] < joint_distance_threshold
    
    rollout_torso_height = rollout_qpos[:, 0]  # Torso height is at index 0 in qpos (states returned from the env)
    torso_above_min = rollout_torso_height > min_torso_height  # shape: (rollout_length,)

    successes = torso_above_min * arm_successes.astype(np.float64)
    return arm_successes.sum(), successes[-1]

def mean_and_se(values):
    mean = np.mean(values)
    se = np.std(values) / np.sqrt(len(values))
    return mean, se

def smooth_with_pd_rolling(data, window_size):
    import pandas as pd
    data = pd.Series(data)
    return data.rolling(window=window_size).mean()

def interquartile_mean_and_se(values):
    # Sort the array
    sorted_values = np.sort(values)
    
    # Calculate the first and third quartile
    Q1 = np.percentile(sorted_values, 25)
    Q3 = np.percentile(sorted_values, 75)
    
    # Get the values between Q1 and Q3 (inclusive)
    interquartile_values = sorted_values[(sorted_values >= Q1) & (sorted_values <= Q3)]
    
    # Compute the interquartile mean
    interquartile_mean = np.mean(interquartile_values)
    
    # Compute the standard error of the mean
    sem = np.std(interquartile_values) / np.sqrt(len(interquartile_values))

    return interquartile_mean, sem

def plot_training_curve(means, ses, timesteps, approaches, task):
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.3)

    for approach in approaches:
        # Plot main line with confidence band
        
        color = APPROACH_COLOR_DICT[approach]

        approach_name = APPROACH_NAME_TO_PLOT[approach]

        window_size = 5
        smoothed_means = smooth_with_pd_rolling(means[approach], window_size)
        smoothed_lower_bound = smooth_with_pd_rolling(np.array(means[approach])-np.array(ses[approach]), window_size)
        smoothed_upper_bound = smooth_with_pd_rolling(np.array(means[approach])+np.array(ses[approach]), window_size)

        # Plot mean[approach]-ses[approach]
        plt.fill_between(timesteps, smoothed_lower_bound, smoothed_upper_bound, color=color, alpha=0.2)
    
        # Plot the main line
        plt.plot(timesteps, smoothed_means, color=color, linewidth=1.5, 
                label=approach_name)
    
    # Customize plot
    ax = plt.gca()
    ax.set_xlim([0, max(timesteps)])
    #ax.set_ylim([0, 1])
    plt.xlabel('Environment Steps', fontsize=20)
    plt.ylabel('Performance', fontsize=20)
    
    # # Put the legend out of the figure (make the legend line thicker)
    #leg = plt.legend(loc='upper left', bbox_to_anchor=(4, 4), fontsize=20, ncol=5)

    # change the line width for the legend
    # for line in leg.get_lines():
    #     line.set_linewidth(8.0)

    plt.tight_layout()
    plt.title(task.replace("-", " ").title() + f" (ep-len=100)", fontsize=20)
    
    # Save plot
    plt_save_path = os.path.join(f"{OUTPUT_DIR}", f"{task.lower()}_training_curves.png")

    plt.savefig(plt_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {plt_save_path}")  

def plot_bar_performance(values, approaches):
    plt.grid(True, linestyle='--', alpha=0.3, zorder=0)

    for approach in approaches:
        all_values = np.array([values[approach][task] for task in values[approach].keys()])
        

        # only take the last step
        flatten_values = all_values[..., -1, :].flatten()
        
        # Calculate the interquartile mean (IQM)
        iqm, sem = interquartile_mean_and_se(flatten_values)

        approach_name = APPROACH_NAME_TO_PLOT[approach]

        # Plot the bar (with label's font size at 18)
        plt.bar(approach_name, iqm, yerr=sem, color=APPROACH_COLOR_DICT[approach], zorder=3, capsize=10)

        # Add the IQM value above the bar
        plt.text(approach_name, iqm + 0.005, f"{iqm:.2f}", ha='center', va='bottom', fontsize=16)

    plt.xticks(fontsize=16)
    plt.ylabel('IQM Success Rate', fontsize=20)

    plt.tight_layout()

    # Save plot
    plt.savefig(os.path.join(f"{OUTPUT_DIR}/cross_task_performance.png"), dpi=300, bbox_inches='tight')
    plt.close()

def save_performance_csv(values, approaches, tasks):

    table = {}

    for approach in approaches:
        table[approach] = []
        for task in tasks:
            task_values = np.array(values[approach][task])
            
            final_frame_values = task_values[..., -1, :].flatten()
            mean, se = mean_and_se(final_frame_values)

            table[approach].append(f"{mean:.2f} ({se:.2f})")
    
    df = pd.DataFrame(table, index=tasks)
    df.index.name = "Task"
    
    # Save to CSV
    df.to_csv(f"{OUTPUT_DIR}/performance.csv")

def main():
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    tasks_to_plot = ["Left-arm-extend", "Right-arm-extend", "Both-arms-out", "Both-arms-down"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Load the CSV file
    csv_file = os.path.join("eval/eval_path_csv", f"mujoco.csv")
    df = pd.read_csv(csv_file)

    # Columns for approaches
    #approaches = ["Threshold", "DTW", "OT", "TemporalOT", "ORCA"]
    approaches = ["Threshold", "DTW", "OT", "TemporalOT", "ORCA"] # # ,

    performance_metric = sparse_metric
    sequence_type = "intermediate_10_frames"

    # Get a dict of the form {task: {approach: [runs]}} 
    all_exp_runs = {}
    for task in tasks_to_plot:
        """
        {
            approach1: [run_1_values, run_2_values, ...],
        }
        """
        # Iterate through each task and approach
        approach_runs = defaultdict(list)
        for index, row in df.iterrows():
            if task == row["Tasks"]:
                for approach in approaches:
                    approach_dir = row[approach]
                    approach_runs[approach].append(approach_dir)
        
        all_exp_runs[task] = approach_runs
    
    # reduce the runs in each approach
    all_approach_values = {approach: {task: [] for task in tasks_to_plot} for approach in approaches}
    for task in tasks_to_plot:
        task_runs = all_exp_runs[task]

        means = {approach: [] for approach in approaches}
        ses = {approach: [] for approach in approaches}

        for approach in approaches:
            approach_runs = task_runs[approach]
            approach_performances = compute_performance_many_experiments(approach_runs, performance_metric, ref_seq_name=sequence_type, eval_metric=None)

            approach_evals = []
            for _, perf in approach_performances.items():
                approach_evals.append(perf[0])
                approach_ts = perf[1] # IMPORTANT: assume this is the same for all runs
            
            approach_evals = np.array(approach_evals)            
            all_approach_values[approach][task].append(approach_evals)

            for i in range(approach_evals.shape[1]):
                values = approach_evals[:, i, :].flatten()
                
                mean, se = mean_and_se(values)
                
                means[approach].append(mean)
                ses[approach].append(se)

        plot_training_curve(means, ses, approach_ts, approaches, task)


        final_means = {approach: mean[-1] for mean in means}
        final_ses = {approach: se[-1] for se in ses}

    plot_bar_performance(all_approach_values, approaches)
    print("saved bar chart")
    save_performance_csv(all_approach_values, approaches, tasks_to_plot)
    print("saved performance csv")
    
if __name__ == "__main__":
    main()