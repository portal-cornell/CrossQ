import numpy as np
from loguru import logger

import matplotlib.gridspec as gridspec

from seq_reward.optimal_transport import compute_ot_reward
from seq_reward.soft_dtw import compute_soft_dtw_reward
from seq_reward.dtw import compute_dtw_reward
from seq_reward.cost_fns import COST_FN_DICT

from constants import TASK_SEQ_DICT

def load_reference_seq(task_name:str, seq_name: str, use_geom_xpos: bool = False) -> np.ndarray:
    """
    Load the reference sequence for the given task name and sequence name from constants.TASK_SEQ_DICT

    Parameters:
        task_name: str
            Specifies the specific task that we want to load the reference sequence for
        seq_name: str
            Specifies the specific sequence that we want to load the reference sequence for
            (e.g. "key_frames")
        use_qpos: bool
            True then we load path that ends with "joint-state.npy"
            False then we load path that ends with "geom-xpos.npy"
    """
    assert task_name in TASK_SEQ_DICT, f"Unknown task name: {task_name}"

    ref_defined_via_a_list = type(TASK_SEQ_DICT[task_name]["sequences"][seq_name]) == list

    if ref_defined_via_a_list:
        # TODO: We assume that when ref is defined via the list, it doesn't not contain the initial state
        ref_seq = []

        for joint in TASK_SEQ_DICT[task_name]["sequences"][seq_name]:
            new_fp = joint
            if use_geom_xpos:
                if "joint-state" in new_fp:
                    new_fp = new_fp.replace("joint-state", "geom-xpos")
            else:
                # Using qpos (labeled as joint-state)
                if "geom-xpos" in new_fp:
                    new_fp = new_fp.replace("geom-xpos", "joint-state")

            loaded_joint_states = np.load(new_fp)

            if use_geom_xpos:
                # Because we are using geom_xpos
                #    Normalize the joint states based on the torso (index 1)
                loaded_joint_states = loaded_joint_states - loaded_joint_states[1]

            ref_seq.append(loaded_joint_states)
        
        return np.stack(ref_seq)
    else:
        assert type(TASK_SEQ_DICT[task_name]["sequences"][seq_name]) == str, f"Unknown type for TASK_SEQ_DICT[{task_name}]['sequences'][{seq_name}]. Has to be either a list or a string, but got {type(TASK_SEQ_DICT[task_name]['sequences'][seq_name])}"
        
        new_fp = TASK_SEQ_DICT[task_name]["sequences"][seq_name]
        if use_geom_xpos:
            if "joint-state" in new_fp:
                new_fp = new_fp.replace("joint-state", "geom-xpos")
        else:
            # Using qpos (labeled as joint-state)
            if "geom-xpos" in new_fp:
                new_fp = new_fp.replace("geom-xpos", "joint-state")
        
        loaded_joint_states = np.load(new_fp)

        if use_geom_xpos:
            # Because we are using geom_xpos
            #    Normalize the joint states based on the torso (index 1)
            loaded_joint_states = loaded_joint_states - loaded_joint_states[1]

        # Because of how these sequences are generated, we need to remove the 1st frame (which is the initial state)
        return loaded_joint_states[1:]
    
def get_matching_fn(fn_config, cost_fn_name="nav_manhattan"):
    """
    Return
        fn: Callable
            - The function that computes the reward
            - The function should take in the following arguments:
                - obs_seq: np.ndarray (obs_seq_len, obs_seq_dim)
                - ref_seq: np.ndarray (ref_seq_len, ref_seq_dim)
        fn_name: str
            - The name of the function
    """
    assert  fn_config["name"] in ["ot", "dtw", "soft_dtw"], f"Currently only supporting ['optimal_transport', 'dtw', 'soft_dtw'], got {fn_config['name']}"
    logger.info(f"Loading the following reward model:\n{fn_config}")

    cost_fn = COST_FN_DICT[cost_fn_name]
    scale = float(fn_config["scale"])
    fn_name = fn_config["name"]

    if fn_name == "ot":
        gamma = float(fn_config["gamma"])
        fn, fn_name = lambda obs_seq, ref_seq, cost_fn=cost_fn, gamma=gamma, scale=scale: compute_ot_reward(obs_seq, ref_seq, cost_fn, scale, gamma), f"{fn_name}_g={gamma}"
    elif fn_name == "dtw":
        fn, fn_name = lambda obs_seq, ref_seq, cost_fn=cost_fn, scale=scale: compute_dtw_reward(obs_seq, ref_seq, cost_fn, scale), fn_name
    elif fn_name == "soft_dtw":
        gamma = float(fn_config["gamma"])
        fn, fn_name = lambda obs_seq, ref_seq, cost_fn=cost_fn, gamma=gamma, scale=scale: compute_soft_dtw_reward(obs_seq, ref_seq, cost_fn, gamma, scale), f"{fn_name}_g={gamma}"
    else:
        raise NotImplementedError(f"Unknown sequence matching function: {fn_name}")
    
    post_processing_method = fn_config.get("post_processing_method", None)

    if post_processing_method:
        if post_processing_method == "stage_reward_based_on_last_state":
            fn_name += "_stg_lst"

            def post_processor(reward, matching_matrix):
                """
                Assuming the matching_matrix is time consistent.

                Parameters:
                    reward: np.ndarray (obs_seq_len, )
                    matching_matrix: np.ndarray (obs_seq_len, ref_seq_len)
                """
                previous_step_assignment = 0
                reward_bonus = 0

                rewards = []

                max_reward_range = fn_config["reward_vmax"] - fn_config["reward_vmin"]

                # print(f"reward={reward}")
                # print(f"matching_matrix={matching_matrix}")

                for i in range(len(reward)):
                    assignment = matching_matrix[i].argmax()

                    # print(f"i={i} assignment={assignment} previous_step_assignment={previous_step_assignment}")

                    if assignment != previous_step_assignment:
                        # Since reward[i] is the last reward at the end of the current stage, the reward bonus for 
                        #   the next stage should get updated
                        reward_bonus += -(-max_reward_range/2) + reward[i-1]
                    
                    new_reward = reward[i] + reward_bonus

                    rewards.append(new_reward)

                    previous_step_assignment = assignment

                    # print(f"i={i} reward[i]={reward[i]} reward_bonus={reward_bonus} new_reward={new_reward}")
                    # input("stop")
                
                return np.array(rewards)
                    
            def augmented_fn(*args, **kwargs):
                reward, info = fn(*args, **kwargs)
                new_reward = post_processor(reward, info["assignment"])

                return new_reward, info

            return augmented_fn, fn_name
        else:
            raise NotImplementedError(f"Unknown post processing method: {post_processing_method}")
    else:
        return fn, fn_name


def plot_matrix_as_heatmap_on_ax(ax, fig, obs_seq, ref_seq, matrix: np.ndarray, title:str, seq_cmap: str, matrix_cmap: str, rolcol_size: int, vmin=None, vmax=None):
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
        ax_a.imshow(obs_seq[i], cmap=seq_cmap)
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
        ax_b.imshow(ref_seq[j], cmap=seq_cmap)
        ax_b.axis('off')

    ax.set_title(title, fontsize=15*rolcol_size)

    # Turn off the axis
    ax.axis('off')