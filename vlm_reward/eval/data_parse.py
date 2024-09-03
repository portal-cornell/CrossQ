import os
from typing import List, Tuple
import pandas as pd

def consolidate_experiment_metrics(eval_log_folder, output_file):
    consolidated_df = pd.DataFrame()
    
    for folder in os.listdir(eval_log_folder):
        # Construct the path to the performance.csv file
        file_path = os.path.join(eval_log_folder, folder, 'performance.csv')
        
        df = pd.read_csv(file_path)
        
        # Add a new column 'Folder' with the name of the current folder
        df['Experiment'] = os.path.basename(folder).split("_rm=")[-1]
        
        # Append the DataFrame to the consolidated DataFrame
        consolidated_df = pd.concat([consolidated_df, df], ignore_index=True)
    
    # Reorder columns to have 'Folder' as the first column
    cols = ['Experiment'] + [col for col in consolidated_df.columns if col != 'Experiment']
    consolidated_df = consolidated_df[cols]
    
    consolidated_df.to_csv(output_file, index=False)

def parse_mujoco_eval_dir(directory, get_every_nth=1) -> List[Tuple[str, str]]:
    """
    Parse a directory containing files like 
    [{i}_rollouts.gif, {i}_rollouts_rewards.npy)]
    
    get_every_nth lets you skip every few rollouts, to make the dataset a little sparser
    """
    filenames = os.listdir(directory)
    # Separate .gif files and .npy files
    gif_files = [f for f in filenames if f.endswith('.gif')]
    reward_files = [f for f in filenames if f.endswith('_rewards.npy')]
    pairs = []
    # Extract the common numeric part and pair them
    for i, gif in enumerate(sorted(gif_files)):
        if i % get_every_nth == 0:
            number = gif.split('_')[0]
            reward_file = f"{number}_rollouts_rewards.npy"
            reward_path = os.path.join(directory, reward_file)

            assert os.path.exists(reward_path), f"Error: reward file does not exist corresponding to source {gif}"
            if reward_file in reward_files:
                pairs.append((os.path.join(directory, gif), reward_path))

    return pairs

def match_rewards_to_sources(source_filenames, reward_filenames):
    """
    Assumes there is a 1:1 matching between source filenames and reward_filenames, but they are both unordered
    Then, once they are sorted, source_filenames[i] is guaranteed to correspond to reward_filenames[i]
    """
    return sorted(source_filenames), sorted(reward_filenames)

def get_matched_source_and_reward_files(source_sequence_dir, reward_dir) -> List[Tuple[str, str]]:
    """
    Returns (source_paths[i], reward_paths[i])
    
    Assumes there is a 1:1 matching between source filenames and reward_filenames, but they are both unordered
    """

    source_filenames = sorted(os.listdir(source_sequence_dir))
    reward_filenames = sorted(os.listdir(reward_dir))

    source_paths = [os.path.join(source_sequence_dir, name) for name in source_filenames]
    reward_paths = [os.path.join(reward_dir, name) for name in reward_filenames]

    return zip(source_paths, reward_paths)

if __name__=="__main__":
    consolidate_experiment_metrics("eval_logs/human-goal/", "human_experiment_outputs.csv")