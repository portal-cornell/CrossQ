from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

import os
import numpy as np

import pdb

from matplotlib import pyplot as plt

def load_npy_in_folder(folder_path: str):
    """
    Parameters:
        folder_path: str
            - The path to the folder that holds the npy files

    Return:
        npy_files: List[np.ndarray]
            - The list of npy files in the folder
    """
    npy_files = []
    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            npy_files.append(np.load(os.path.join(folder_path, file)))
    return np.array(npy_files)

def compute_iqm(npy_files):
    """
    Parameters:
        npy_files: List[np.ndarray]
            - The list of npy files to compute the IQM for
    """
    print(f"npy_files: {npy_files}, {npy_files.shape}")
    # Find the mean for each array
    mean = np.mean(npy_files, axis=1)

    print("Mean: ", mean, "Length: ", len(mean))

    # Find the 25th percentile based on the mean
    percentile_25 = np.percentile(mean, 25)
    # Find the 75th percentile based on the mean
    percentile_75 = np.percentile(mean, 75)

    print("25th percentile: ", percentile_25)
    print("75th percentile: ", percentile_75)

    # Find the idx of the runs that are within the 25th and 75th percentile
    iqm_values_idx = np.where((mean >= percentile_25) & (mean <= percentile_75))

    print("IQM values: ", iqm_values_idx, "Length: ", len(iqm_values_idx))

    # Find the mean of the runs that are within the 25th and 75th percentile
    iqm_values = np.mean(npy_files[iqm_values_idx], axis=0)

    print(f"npy_files[iqm_values_idx]: {npy_files[iqm_values_idx]}, {npy_files[iqm_values_idx].shape}")
    print(f"iqm_values: {iqm_values}, {iqm_values.shape}")

    return iqm_values

@hydra.main(version_base=None, config_path="configs", config_name="summarize_eval_config")
def main(cfg: DictConfig):
    assert cfg.model_inference_paths != {}, "model_inference_paths is empty"
    data_save_dir = HydraConfig.get().runtime.output_dir

    # Plot all the rewards in the same plot
    plt.figure(figsize=(10, 5))

    overall_iqms = []

    for model_name in cfg.model_inference_paths:
        model_folder_path = cfg.model_inference_paths[model_name]

        model_npy_files = load_npy_in_folder(os.path.join(model_folder_path, "video"))

        iqm_values = compute_iqm(model_npy_files)

        overall_iqms.append(np.mean(iqm_values))

        plt.plot(iqm_values, label=model_name)
    
    # Compute the overall ranking based on overall_iqms (from Greatest to Least)
    overall_iqms = np.argsort(overall_iqms)[::-1]
    best_to_worst_model = [list(cfg.model_inference_paths.keys())[idx] for idx in overall_iqms]

    plt.xlabel('Timesteps')  # Replace with actual labels if needed
    plt.ylabel('Rewards')
    plt.title(f'Timesteps vs Rewards (IQM)\n{best_to_worst_model}')

    # Add a legend
    plt.legend()

    plt.savefig(os.path.join(data_save_dir, f"all_rewards_iqm.png"))

    plt.clf()
    plt.close()

if __name__ == "__main__":
    main()