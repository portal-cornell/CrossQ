from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import copy
import os
from loguru import logger
from tqdm import tqdm

from seq_matching_toy.toy_examples_main import examples
from seq_matching_toy.seq_utils import get_matching_fn

def plot_matrix_as_heatmap_on_ax(ax, fig, obs_seq, ref_seq, matrix: np.ndarray, title:str, cmap: str, rolcol_size: int, vmin=None, vmax=None):
    """
    Plot the Matrix with obs_seq on the left and ref_seq on top of the heatmap.
    """
    obs_len = len(obs_seq)
    ref_len = len(ref_seq)

    # Create a GridSpec layout for each subplot
    #   nrows: the number of frames in obs_seq + 1 (for the top row for ref_seq)
    #   ncols: the number of frames in ref_seq + 1 (for the left column for obs_seq) + 1 (for the colorbar)
    #   width_ratios: the width of each column
    #       the first column is for the obs_seq
    #       the 2nd to 1 + ref_len columns are for the ref_seq
    #       the last column is for the colorbar
    #           so (ref_len + 1) has the size of 0.2 and the last one (for the colorbar) has the size of 0.05
    #   height_ratios: the height of each row
    gs = gridspec.GridSpecFromSubplotSpec(1 + obs_len, 2 + ref_len, subplot_spec=ax.get_subplotspec(), width_ratios=[0.2] * (ref_len + 1) + [0.05], height_ratios=[0.2] * (obs_len + 1))
    
    # Plot the matrices from array `a` on the left (aligned vertically)
    for i in range(obs_seq.shape[0]):
        ax_a = fig.add_subplot(gs[i + 1, 0])  # Move down 1 row to align with the heatmap
        ax_a.imshow(obs_seq[i], cmap='plasma')
        ax_a.axis('off')

    # Plot the heatmap (cost matrix) in the center
    ax_heatmap = fig.add_subplot(gs[1:obs_len+1, 1:ref_len+1])
    im = ax_heatmap.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

    if vmin is not None and vmax is not None:
        mid_val = (vmin + vmax) / 2
    else:
        mid_val = (np.max(matrix) + np.min(matrix)) / 2

    # Add text annotations (numbers) on each cell in the heatmap
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text_color = 'white' if matrix[i, j] > mid_val else 'black'
            ax_heatmap.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color=text_color, fontsize=10*rolcol_size)

    ax_colorbar = fig.add_subplot(gs[1:obs_len+1, ref_len + 1])
    cbar = fig.colorbar(im, cax=ax_colorbar, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=5*rolcol_size)  # Adjust the colorbar tick labels if neede

    # # Adjust the layout and remove padding
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Plot the matrices from array `b` on the top (aligned horizontally)
    for j in range(ref_seq.shape[0]):
        ax_b = fig.add_subplot(gs[0, 1 + j])
        ax_b.imshow(ref_seq[j], cmap='plasma')
        ax_b.axis('off')

    ax.set_title(title, fontsize=15*rolcol_size)

    # Turn off the axis
    ax.axis('off')


def prepare_seq_matching_fns(seq_matching_fn_configs, cost_fn_name):
    """
    Return:
        a dictionary that maps the function to its name
    """
    seq_matching_fns_dict = {}

    for fn_config in seq_matching_fn_configs:
        fn, fn_name = get_matching_fn(fn_config, cost_fn_name)
        seq_matching_fns_dict[fn_name] = fn
        
    return seq_matching_fns_dict

    
def run_examples_from_config(cfg: DictConfig):
    """
    Examples are stored in dictionaries in indvidual python files  
        Each dictionary has
        - ref_seq: reference sequence, which is a list of numpy arrays
        - obs_seq: a dictionary of observed sequences, key is the id, the value is a observation sequence, which is a dictionary with
            - descriptions: a string describing the sequence
            - seq: a list of numpy arrays

    There's a main dictionary that map the name of the example to the example dictionary

    For each refernece sequence, 
        For each observation sequence, we create a main plot that shows
            for each sequence matching algorithm
                the cost matrix and the assignment matrix
    So for this plot, it will have 2 columns, and the number of rows will be the number of sequence matching algorithms
    """
    data_save_dir = HydraConfig.get().runtime.output_dir

    logger.info(f"Saving the plots to {data_save_dir}")

    n_seq_matching_fns = len(cfg.seq_matching_fns)
    seq_matching_fns_dict = prepare_seq_matching_fns(cfg.seq_matching_fns, cfg.cost_fn)

    example_name = cfg.example

    for obs_id in tqdm(examples[example_name]["obs_seqs"].keys()):
        ref_seq = np.array(examples[example_name]["ref_seq"])
        obs_seq = np.array(examples[example_name]["obs_seqs"][obs_id]["seq"])

        rolcol_size = cfg.plot.rolcol_size

        # 3 * because we have 3 figure columns
        #   In each figure columns, we have len(ref_seq) for the reference sequence/cost matrix, 1 column for the vertical stack of obs seq, and 1 column for the colorbar
        fig_width = 3 * (rolcol_size * (len(ref_seq) + 2))
        # n_seq_matching_fns * because we need a figure row for each sequence matching function
        #   In each figure row, we have len(obs_seq) for the observed sequence/cost matrix, 1 row for the horizontal stack of ref seq
        fig_height = n_seq_matching_fns * (rolcol_size * (len(obs_seq) + 1))

        # Create the figure (2 columns, and the number of rows will be the number of sequence matching algorithms)
        fig, axs = plt.subplots(n_seq_matching_fns, 3, figsize=(fig_width, fig_height))

        for fn_idx, fn_name in enumerate(seq_matching_fns_dict.keys()):
            seq_matching_fn = seq_matching_fns_dict[fn_name]
            
            reward, info = seq_matching_fn(obs_seq, ref_seq)

            # Plot the cost matrix
            ax = axs[fn_idx, 0]
            plot_matrix_as_heatmap_on_ax(ax, fig, obs_seq, ref_seq, info["cost_matrix"], f"{fn_name} Cost Matrix", cmap="gray_r", rolcol_size=rolcol_size,)

            # Plot the assignment matrix
            ax = axs[fn_idx, 1]   
            plot_matrix_as_heatmap_on_ax(ax, fig, obs_seq, ref_seq, info["assignment"], f"{fn_name} Assignment Matrix", cmap="Greens", rolcol_size=rolcol_size, vmin=0, vmax=1)

            # Plot the reward
            ax = axs[fn_idx, 2]
            plot_matrix_as_heatmap_on_ax(ax, fig, obs_seq, ref_seq, np.expand_dims(reward,1), f"{fn_name} Reward (Sum = {np.sum(reward):.2f})", cmap="Greens", rolcol_size=rolcol_size,
                                         vmin=examples[example_name]["plot"]["reward_vmin"], vmax=examples[example_name]["plot"]["reward_vmax"])

        plt.tight_layout()
        plt.savefig(os.path.join(data_save_dir, f"{example_name}_obs_{obs_id}.png"))

        plt.close(fig)


@hydra.main(version_base=None, config_path="configs", config_name="run_examples_config")
def main(cfg: DictConfig):

    run_examples_from_config(cfg)

if __name__ == "__main__":
    main()