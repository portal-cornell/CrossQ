import numpy as np
import ot
from scipy.spatial.distance import cdist
from constants import SEQ_DICT

import matplotlib.pyplot as plt

def load_reference_seq(seq_name: str, use_geom_xpos: bool) -> np.ndarray:
    """
    Load the reference sequence for the given sequence name
    """
    ref_seq = []
    for joint in SEQ_DICT[seq_name]:
        if use_geom_xpos:
            new_fp = str(joint).replace("joint-state", "geom-xpos")
        else:
            new_fp = joint

        loaded_joint_states = np.load(new_fp)

        if use_geom_xpos:
            # Normalize the joint states based on the torso (index 1)
            loaded_joint_states = loaded_joint_states - loaded_joint_states[1]

        ref_seq.append(loaded_joint_states)
    return np.stack(ref_seq)

def compute_ot_reward(obs: np.ndarray, ref: np.ndarray, cost_fn, scale=1) -> np.ndarray:
    """
    Compute the Optimal Transport (OT) reward between the reference sequence and the observed sequence

    Parameters:
        obs: np.ndarray
            The observed sequence of joint states
            size: (train_freq, 22)
                For OT-based reward, train_freq == episode_length
                22 is the observation size that we want to calculate
        ref: np.ndarray
            The reference sequence of joint states
            size: (ref_seq_len, 22)
                22 is the observation size that we want to calculate
        cost_fn: function
            Options: cosine_distance, euclidean_distance
        scale: float
            The scaling factor for the OT reward
    """
    # Calculate the cost matrix between the reference sequence and the observed sequence
    #   size: (train_freq, ref_seq_len)
    cost_matrix = cost_fn(obs, ref)

    # Calculate the OT plan between the reference sequence and the observed sequence
    obs_weight = np.ones(obs.shape[0]) / obs.shape[0]
    ref_weight = np.ones(ref.shape[0]) / ref.shape[0]
    T = ot.sinkhorn(obs_weight, ref_weight, cost_matrix, reg=0.01, log=False)  # size: (train_freq, ref_seq_len)

    # Calculate the OT cost for each timestep
    #   sum by row of (cost matrix * OT plan)
    ot_cost = np.sum(cost_matrix * T, axis=1)  # size: (train_freq,)

    info = dict(
        assignment=T,
        cost_matrix=cost_matrix,
        transported_cost=cost_matrix * T,
    )

    return - scale * ot_cost, info

def plot_matrix_as_heatmap(matrix: np.ndarray, title: str, fp: str):
    """
    Plot the Assignment Matrix
    """
    # Because there are way less reference frames than observed frames, we need to copy the values and pad the reference frames for visualization
    padded_matrix = np.repeat(matrix, 10, axis=1)
    
    plt.imshow(padded_matrix, cmap='hot', interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.show()

    plt.savefig(fp)

    plt.clf()
    plt.close()


def cosine_distance(x, y):
    distance = np.dot(x, y.T) / np.linalg.norm(x, axis=1, keepdims=True) / np.linalg.norm(y.T, axis=0, keepdims=True) # Transpose B to match dimensions

    # Rescale to be between 0 and 1
    distance_rescaled = (distance + 1) / 2
    return 1 - distance_rescaled

def euclidean_distance(x, y):
    return cdist(x, y, metric="euclidean")

def squared_euclidean_distance(x, y):
    return cdist(x, y, metric="euclidean") ** 2

COST_FN_DICT = {
    "cosine": cosine_distance,
    "euclidean": euclidean_distance,
    "squared_euclidean": squared_euclidean_distance,
}