import os
import csv
import numpy as np
from loguru import logger
from PIL import Image
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

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
                assert loaded_joint_states.shape[0] == 18 and loaded_joint_states.shape[1] == 3, f"Expected the shape to be (18, 3), but got {loaded_joint_states.shape}"
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
            assert loaded_joint_states.shape[1] == 18 and loaded_joint_states.shape[2] == 3, f"Expected the shape to be (num_frames, 18, 3), but got {loaded_joint_states.shape}"
            # Because we are using geom_xpos
            #    Normalize the joint states based on the torso (index 1)
            loaded_joint_states = loaded_joint_states - loaded_joint_states[:, 1:2]

        # Because of how these sequences are generated, we need to remove the 1st frame (which is the initial state)
        return loaded_joint_states[1:]
    

def load_images_from_reference_seq(task_name:str, seq_name: str) -> (np.ndarray):
    assert task_name in TASK_SEQ_DICT, f"Unknown task name: {task_name}"
    assert seq_name in TASK_SEQ_DICT[task_name]["sequences"], f"Unknown sequence name: {seq_name}."

    if type(TASK_SEQ_DICT[task_name]["sequences"][seq_name]) == str:
        gif_path = TASK_SEQ_DICT[task_name]["sequences"][seq_name]

        if "joint-state" in gif_path:
            gif_path = gif_path.replace("_joint-state.npy", ".gif")
        elif "geom-xpos" in gif_path:
            gif_path = gif_path.replace("_geom-xpos.npy", ".gif")
        
        gif_obj = Image.open(gif_path)
        frames = [gif_obj.seek(frame_index) or gif_obj.convert("RGB") for frame_index in range(gif_obj.n_frames)]

        return np.stack([np.array(frame) for frame in frames])
    else:
        # This is a list of image paths (likely only one image in that list)
        return np.array([])

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

    inverted_cost = False #cost_fn_name == "nav_shortest_path"

    if fn_name == "ot":
        gamma = float(fn_config["gamma"])
        fn, fn_name = lambda obs_seq, ref_seq, cost_fn=cost_fn, gamma=gamma, scale=scale: compute_ot_reward(obs_seq, ref_seq, cost_fn, scale, gamma), f"{fn_name}_g={gamma}"
    elif fn_name == "dtw":
        fn, fn_name = lambda obs_seq, ref_seq, cost_fn=cost_fn, scale=scale: compute_dtw_reward(obs_seq, ref_seq, cost_fn, scale, inverted_cost=inverted_cost), fn_name
    elif fn_name == "soft_dtw":
        gamma = float(fn_config["gamma"])
        if gamma == 10000.0:
            fn_name = f"{fn_name}_g=F"
        else:
            fn_name = f"{fn_name}_g={gamma}"
        fn = lambda obs_seq, ref_seq, cost_fn=cost_fn, gamma=gamma, scale=scale: compute_soft_dtw_reward(obs_seq, ref_seq, cost_fn, gamma, scale,inverted_cost=inverted_cost)
    else:
        raise NotImplementedError(f"Unknown sequence matching function: {fn_name}")
    
    post_processing_method = fn_config.get("post_processing_method", [])

    if post_processing_method:
        augmented_fn = fn

        for method in post_processing_method:
            if method == "exp_reward":
                augmented_fn, fn_name = augment_fn_with_exp_reward(
                    original_fn=augmented_fn, 
                    original_fn_name=fn_name)
            elif method == "stage_reward_based_on_last_state":
                augmented_fn, fn_name = augment_fn_with_stage_reward_based_on_last_state(
                    original_fn=augmented_fn, 
                    original_fn_name=fn_name, 
                    stage_bonus=float(fn_config.get("stage_bonus", 0)))
            elif method == "stage_multiplier_based_on_last_state":
                augmented_fn, fn_name = augment_fn_with_stage_multiplier_based_on_last_state(
                    original_fn=augmented_fn, 
                    original_fn_name=fn_name)
            elif method == "stage_multiplier_plus_based_on_max":
                augmented_fn, fn_name = augment_fn_with_stage_multiplier_plus_based_on_max(
                    original_fn=augmented_fn, 
                    original_fn_name=fn_name)
            elif method == "stage_multiplier_plus_based_on_last_state":
                augmented_fn, fn_name = augment_fn_with_stage_multiplier_plus_based_on_last_state(
                    original_fn=augmented_fn, 
                    original_fn_name=fn_name)
            elif method == "reward_scaled_by_stage":
                augmented_fn, fn_name = augment_fn_with_reward_scaled_by_stage(
                    original_fn=augmented_fn, 
                    original_fn_name=fn_name)
            elif method == "convert_to_positive_reward":
                augmented_fn, fn_name = augment_fn_convert_to_positive_reward(
                    original_fn=augmented_fn, 
                    original_fn_name=fn_name, 
                    pos_offset=float(fn_config.get("pos_offset", 0)))
            else:
                raise NotImplementedError(f"Unknown post processing method: {post_processing_method}")
            
        return augmented_fn, fn_name
    else:
        return fn, fn_name



def augment_fn_with_exp_reward(original_fn, original_fn_name):
    new_fn_name = original_fn_name + "_exp"

    def post_processor(reward):
        """
        The reward is - cost right now. We can exponentiate it to make it positive and between 0 and 1.
        """
        return np.exp(reward)
    
    def new_fn(*args, **kwargs):
        reward, info = original_fn(*args, **kwargs)
        new_reward = post_processor(reward)

        return new_reward, info
    
    return new_fn, new_fn_name


def augment_fn_with_stage_reward_based_on_last_state(original_fn, original_fn_name, stage_bonus):
    """
    Parameters:
        stage_bonus: float
            The bonus that we add to the reward when we progress from one stage to another stage.
    """
    new_fn_name = original_fn_name + "_stg_lst"

    def post_processor(reward, matching_matrix):
        """
        Assuming the matching_matrix is time consistent.

        When the assignment changes (progress from one ref frame to another ref frame), we add a bonus to the reward.

        Parameters:
            reward: np.ndarray (obs_seq_len, )
            matching_matrix: np.ndarray (obs_seq_len, ref_seq_len)
        """
        previous_step_assignment = 0
        reward_bonus = 0

        rewards = []

        # print(f"reward={reward}")
        # print(f"matching_matrix={matching_matrix}")

        for i in range(len(reward)):
            assignment = matching_matrix[i].argmax()

            # print(f"i={i} assignment={assignment} previous_step_assignment={previous_step_assignment}")

            if assignment != previous_step_assignment:
                # Since reward[i] is the last reward at the end of the current stage, the reward bonus for 
                #   the next stage should get updated
                reward_bonus += stage_bonus + reward[i-1]
            
            new_reward = reward[i] + reward_bonus

            rewards.append(new_reward)

            previous_step_assignment = assignment

            # print(f"i={i} reward[i]={reward[i]} reward_bonus={reward_bonus} new_reward={new_reward}")
            # input("stop")
        
        # Normalize the rewards to be 0 and 1
        # return np.array(rewards) / matching_matrix.shape[1]
        return np.array(rewards)
                        
    def new_fn(*args, **kwargs):
        reward, info = original_fn(*args, **kwargs)
        new_reward = post_processor(reward, info["assignment"])

        return new_reward, info
    
    return new_fn, new_fn_name


def augment_fn_with_stage_multiplier_based_on_last_state(original_fn, original_fn_name):
    """
    Parameters:
        stage_bonus: float
            The bonus that we add to the reward when we progress from one stage to another stage.
    """
    new_fn_name = original_fn_name + "_x-stg"

    def post_processor(reward, matching_matrix):
        """
        Assuming the matching_matrix is time consistent.

        When the assignment changes (progress from one ref frame to another ref frame), we add a bonus to the reward.

        Parameters:
            reward: np.ndarray (obs_seq_len, )
            matching_matrix: np.ndarray (obs_seq_len, ref_seq_len)
        """
        previous_step_assignment = 0
        reward_multiplier = 1
        rewards = []

        # print(f"reward={reward}")
        # print(f"matching_matrix={matching_matrix}")

        for i in range(len(reward)):
            assignment = matching_matrix[i].argmax()

            # print(f"i={i} assignment={assignment} previous_step_assignment={previous_step_assignment}")

            if assignment != previous_step_assignment:
                # Since reward[i] is the last reward at the end of the current stage, the reward bonus for 
                #   the next stage should get updated
                reward_multiplier *= reward[i-1]
            
            new_reward = reward[i] * reward_multiplier

            rewards.append(new_reward)

            previous_step_assignment = assignment

            # print(f"i={i} reward[i]={reward[i]} reward_bonus={reward_bonus} new_reward={new_reward}")
            # input("stop")
        
        # Normalize the rewards to be 0 and 1
        # return np.array(rewards) / matching_matrix.shape[1]
        return np.array(rewards)
                        
    def new_fn(*args, **kwargs):
        reward, info = original_fn(*args, **kwargs)
        new_reward = post_processor(reward, info["assignment"])

        return new_reward, info
    
    return new_fn, new_fn_name


def augment_fn_with_stage_multiplier_plus_based_on_max(original_fn, original_fn_name):
    """
    Parameters:
        stage_bonus: float
            The bonus that we add to the reward when we progress from one stage to another stage.
    """
    new_fn_name = original_fn_name + "_x+m"

    def post_processor(reward, matching_matrix):
        """
        Assuming the matching_matrix is time consistent.

        When the assignment changes (progress from one ref frame to another ref frame), we add a bonus to the reward.

        Parameters:
            reward: np.ndarray (obs_seq_len, )
            matching_matrix: np.ndarray (obs_seq_len, ref_seq_len)
        """
        previous_step_assignment = 0
        reward_multiplier = 1
        reward_bonus = 0
        rewards = []

        # print(f"reward={reward}")
        # print(f"matching_matrix={matching_matrix}")

        for i in range(len(reward)):
            assignment = matching_matrix[i].argmax()

            # print(f"i={i} assignment={assignment} previous_step_assignment={previous_step_assignment}")

            if assignment != previous_step_assignment:
                # We find all the reward whose argmax is the same as the previous_step_assignment
                print(f"i={i} reward[i]={reward[i]} reward_bonus={reward_bonus} reward_multiplier={reward_multiplier}")
                print(reward[matching_matrix.argmax(axis=1) == previous_step_assignment])
                previous_step_max_reward = np.max(reward[matching_matrix.argmax(axis=1) == previous_step_assignment])
                print(f"previous_step_max_reward={previous_step_max_reward}")
                reward_bonus += previous_step_max_reward
                reward_multiplier *= previous_step_max_reward
                # input("stop")
                           
            new_reward = reward[i] * reward_multiplier + reward_bonus

            rewards.append(new_reward)

            previous_step_assignment = assignment

            # print(f"i={i} reward[i]={reward[i]} reward_bonus={reward_bonus} new_reward={new_reward}")
            # input("stop")
        
        # Normalize the rewards to be 0 and 1
        # return np.array(rewards) / matching_matrix.shape[1]
        return np.array(rewards)
                        
    def new_fn(*args, **kwargs):
        reward, info = original_fn(*args, **kwargs)
        new_reward = post_processor(reward, info["assignment"])

        return new_reward, info
    
    return new_fn, new_fn_name


def augment_fn_with_stage_multiplier_plus_based_on_last_state(original_fn, original_fn_name):
    """
    Parameters:
        stage_bonus: float
            The bonus that we add to the reward when we progress from one stage to another stage.
    """
    new_fn_name = original_fn_name + "_x+l"

    def post_processor(reward, matching_matrix):
        """
        Assuming the matching_matrix is time consistent.

        When the assignment changes (progress from one ref frame to another ref frame), we add a bonus to the reward.

        Parameters:
            reward: np.ndarray (obs_seq_len, )
            matching_matrix: np.ndarray (obs_seq_len, ref_seq_len)
        """
        previous_step_assignment = 0
        reward_multiplier = 1
        reward_bonus = 0
        rewards = []

        # print(f"reward={reward}")
        # print(f"matching_matrix={matching_matrix}")

        for i in range(len(reward)):
            assignment = matching_matrix[i].argmax()

            # print(f"i={i} assignment={assignment} previous_step_assignment={previous_step_assignment}")

            if assignment != previous_step_assignment:
                reward_bonus += reward[i-1]
                reward_multiplier *= reward[i-1]
                # input("stop")
                           
            new_reward = reward[i] * reward_multiplier + reward_bonus

            rewards.append(new_reward)

            previous_step_assignment = assignment

            # print(f"i={i} reward[i]={reward[i]} reward_bonus={reward_bonus} new_reward={new_reward}")
            # input("stop")
        
        # Normalize the rewards to be 0 and 1
        # return np.array(rewards) / matching_matrix.shape[1]
        return np.array(rewards)
                        
    def new_fn(*args, **kwargs):
        reward, info = original_fn(*args, **kwargs)
        new_reward = post_processor(reward, info["assignment"])

        return new_reward, info
    
    return new_fn, new_fn_name

def augment_fn_convert_to_positive_reward(original_fn, original_fn_name, pos_offset):
    new_fn_name = original_fn_name + "_p"

    def post_processor(cost_matrix, matching_matrix):
        """
        Convert the reward to be positive by adding a constant offset.
        """
        reward_matrix = pos_offset - cost_matrix
        return np.sum(reward_matrix * matching_matrix, axis=1) / pos_offset
                        
    def new_fn(*args, **kwargs):
        reward, info = original_fn(*args, **kwargs)
        new_reward = post_processor(info["cost_matrix"], info["assignment"])

        return new_reward, info
    
    return new_fn, new_fn_name


def augment_fn_with_reward_scaled_by_stage(original_fn, original_fn_name):

    new_fn_name = original_fn_name + "_stg_lst"

    def post_processor(reward, matching_matrix):
        """
        Scale the reward such that reward[i] *= N / N_stage, where N_stage is the number of assigned 
        frames to the stage of i. This discounts rewards for stages with high assignment, in order to encourage progress.
        """
        assignment = matching_matrix.argmax(axis=1) # find the best assignment for each obs
        reward = scale_rewards_by_class(reward, assignment)
        
        return reward
                        
    def new_fn(*args, **kwargs):
        reward, info = original_fn(*args, **kwargs)
        new_reward = post_processor(reward, info["assignment"])

        return new_reward, info
    
    return new_fn, new_fn_name


def scale_rewards_by_class(rewards: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """
    Scales rewards based on class frequencies using NumPy arrays.
    
    Args:
        rewards: np.ndarray - Original rewards array
        classes: np.ndarray - Class labels array (in range [0, k))
        
    Returns:
        np.ndarray - Modified rewards where each reward is scaled by (N / N_in_class)
    """
    if rewards.shape != classes.shape:
        raise ValueError("Shape of rewards and classes must match")
    
    N = len(rewards)
    
    # Count frequency of each class using bincount
    class_counts = np.bincount(classes)
    
    # Create array of class counts corresponding to each reward
    counts_per_reward = class_counts[classes]
    
    # Scale rewards vectorized
    scaled_rewards = rewards / counts_per_reward

    return scaled_rewards


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
    im = ax_heatmap.imshow(matrix, cmap=matrix_cmap, aspect='auto', vmin=vmin, vmax=vmax)

    if vmin is not None and vmax is not None:
        mid_val = (vmin + vmax) / 2
    else:
        mid_val = (np.max(matrix) + np.min(matrix)) / 2

    # Add text annotations (numbers) on each cell in the heatmap
    label_text_font_size = max(obs_len, ref_len) / min(matrix.shape[0], matrix.shape[1]) * rolcol_size * 2
    if label_text_font_size >= 1:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text_color = 'white' if matrix[i, j] > mid_val else 'black'
                ax_heatmap.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color=text_color, fontsize=label_text_font_size)

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


def seq_matching_viz(matching_fn_name, obs_seq, ref_seq, matching_reward, info, reward_vmin, reward_vmax, path_to_save_fig, seq_cmap=None, rolcol_size=1):
    # 2 * because we have 2 figure columns (where we will plot the entire ref seq)
    #   In each figure columns, we have len(ref_seq) for the reference sequence/cost matrix, 1 column for the vertical stack of obs seq, and 1 column for the colorbar
    # The last column (for the reward) will just have 4 things (1 column for the vertical stack of obs seq, 1 column for the colorbar, and 2 column for the reward)
    fig_width = rolcol_size * (2 * (len(ref_seq) + 2) + 4)

    #  We have len(obs_seq) for the observed sequence/cost matrix, 1 row for the horizontal stack of ref seq
    fig_height = rolcol_size * (len(obs_seq) + 1)

    # Create the figure (2 columns, and the number of rows will be the number of sequence matching algorithms)
    fig, axs = plt.subplots(1, 3, figsize=(fig_width, fig_height))

    # Plot the cost matrix
    ax = axs[0]
    plot_matrix_as_heatmap_on_ax(ax, fig, obs_seq, ref_seq, info["cost_matrix"], f"{matching_fn_name} Cost", seq_cmap=seq_cmap,  matrix_cmap="gray_r", rolcol_size=rolcol_size)
    
    # Plot the assignment matrix
    ax = axs[1]
    plot_matrix_as_heatmap_on_ax(ax, fig, obs_seq, ref_seq, info["assignment"], f"{matching_fn_name} Assign", seq_cmap=seq_cmap, matrix_cmap="Greens", rolcol_size=rolcol_size, vmin=0, vmax=1)

    # Plot the reward
    ax = axs[2]
    # Only plot the last 2 frames of the ref seq
    if len(ref_seq) > 2:
        plot_matrix_as_heatmap_on_ax(ax, fig, obs_seq, ref_seq[-3:-1], np.expand_dims(matching_reward,1), f"{matching_fn_name} Reward (Sum = {np.sum(matching_reward):.2f})", seq_cmap=seq_cmap, matrix_cmap="Greens", rolcol_size=rolcol_size, vmin=reward_vmin, vmax=reward_vmax)
    else:
        plot_matrix_as_heatmap_on_ax(ax, fig, obs_seq, ref_seq, np.expand_dims(matching_reward,1), f"{matching_fn_name} Reward (Sum = {np.sum(matching_reward):.2f})", seq_cmap=seq_cmap, matrix_cmap="Greens", rolcol_size=rolcol_size, vmin=reward_vmin, vmax=reward_vmax)

    plt.tight_layout()

    plt.savefig(path_to_save_fig)

    plt.close(fig)

def append_to_csv(items, item_headers, filename):
    """
    Logs a set of items with corresponding headers to a CSV file.
    Creates the file with headers if it doesn't exist.

    """
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write headers if new file
        if not file_exists:
            writer.writerow(item_headers)
        
        # Write the new row
        writer.writerow(items)
