from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

import numpy as np
import matplotlib.pyplot as plt
import os
from loguru import logger
from PIL import Image
from tqdm import tqdm

from seq_matching_toy.toy_envs.minigrid_sequence import make_sequence_env
from seq_reward.seq_utils import get_matching_fn, plot_matrix_as_heatmap_on_ax
from seq_matching_toy.toy_examples_main import load_map_from_example_dict, load_ref_seq_from_example_dict, load_reward_vmin_vmax_from_example_dict, load_starting_pos_from_example_dict, load_observations_from_examples_dict

def prepare_seq_matching_fns(seq_matching_fn_configs, cost_fn_name, reward_vmin, reward_vmax):
    """
    Return:
        a dictionary that maps the function to its name
    """
    seq_matching_fns_dict = {}

    for fn_config in seq_matching_fn_configs:
        fn_config = dict(fn_config)
        fn, fn_name = get_matching_fn(fn_config, cost_fn_name)
        seq_matching_fns_dict[fn_name] = fn
        
    return seq_matching_fns_dict

def render_poses(render_env, pose_seq):
    """
    Create renders of the pose sequence for visualization
    """
    # return np.ones((len(self._ref_seq),5,5,3))
    render_env.unwrapped.reset()
    
    renders = []
    for pose in pose_seq:
        agent_pos = (pose[0], pose[1])
        agent_dir = pose[2] 

        render_env.unwrapped.set_state(agent_pos, agent_dir)
        render = render_env.render()
        renders.append(render)
    renders = np.stack(renders)
    return renders
    
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

    example_name = cfg.env.example_name
    map_array = load_map_from_example_dict(example_name)
    starting_pos = load_starting_pos_from_example_dict(example_name)
    ref_seq = load_ref_seq_from_example_dict(example_name)
    obs_seqs = load_observations_from_examples_dict(example_name)
    vmin, vmax = load_reward_vmin_vmax_from_example_dict(example_name)

    n_seq_matching_fns = len(cfg.seq_matching_fns)
    
    seq_matching_fns_dict = prepare_seq_matching_fns(cfg.seq_matching_fns, cfg.cost_fn, vmin, vmax)

    render_env = make_sequence_env(map_array=np.copy(map_array), starting_pos=starting_pos, render_mode="rgb_array", temporal_encoding= cfg.env.temporal_encoding, episode_length=cfg.env.episode_length)
    ref_render = render_poses(render_env, ref_seq)

    for obs_id in tqdm(obs_seqs.keys()):
        obs_seq = np.array(obs_seqs[obs_id]["seq"])
        obs_render = render_poses(render_env, obs_seq)

        # Save a gif of the example observation
        obs_pil = [Image.fromarray(img) for img in obs_render]
        obs_pil[0].save(os.path.join(data_save_dir, f"{example_name}_obs_{obs_id}.gif"), save_all=True, append_images=obs_pil[1:], duration=len(obs_pil)*5, loop=0)

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
                    
            discounts = np.array([cfg.rl_algo.gamma ** t for t in range(len(reward))])
            rl_return = np.dot(discounts, reward)
            
            if "sparse_reward" in fn_name:
                print(f"sparse reward: {reward}")
                continue
            else: 
                # Plot the cost matrix
                ax = axs[fn_idx, 0]
                plot_matrix_as_heatmap_on_ax(ax, fig, obs_render, ref_render, info["cost_matrix"], f"{fn_name} C", seq_cmap="plasma", matrix_cmap="gray_r", rolcol_size=rolcol_size,)

                # Plot the assignment matrix
                ax = axs[fn_idx, 1]   
                plot_matrix_as_heatmap_on_ax(ax, fig, obs_render, ref_render, info["assignment"], f"{fn_name} A", seq_cmap="plasma", matrix_cmap="Greens", rolcol_size=rolcol_size, vmin=0, vmax=1)

                # Plot the reward
                ax = axs[fn_idx, 2]
                plot_matrix_as_heatmap_on_ax(ax, fig, obs_render, ref_render, np.expand_dims(reward,1), f"{fn_name} Return (y={cfg.rl_algo.gamma}): {rl_return:.5f})", seq_cmap="plasma", matrix_cmap="Greens", rolcol_size=rolcol_size,
                                            vmin=vmin, vmax=vmin)

        plt.tight_layout()
        plt.savefig(os.path.join(data_save_dir, f"{example_name}_obs_{obs_id}.png"))

        plt.close(fig)


@hydra.main(version_base=None, config_path="configs", config_name="run_examples_config")
def main(cfg: DictConfig):

    run_examples_from_config(cfg)

if __name__ == "__main__":
    main()