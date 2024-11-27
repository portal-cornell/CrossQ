import numpy as np

def compute_testing_dist_reward(obs: np.ndarray, ref: np.ndarray, cost_fn) -> np.ndarray:
    """
    The main purpose is testing and ensuring that the cost function is working properly. A mock function to compute a test reward between the reference sequence and the observed sequence
    - The reward is the distance between the observed sequence and the last frame of the reference sequence

    Parameters:
        obs: np.ndarray
            The observed sequence of joint states
            size: (train_freq, state_feat_size)
                train_freq == episode_length
        ref: np.ndarray
            The reference sequence of joint states
            size: (ref_seq_len, state_feat_size)
        cost_fn: function
            Options: cosine_distance, euclidean_distance
        scale: float
            The scaling factor for the OT reward

    Returns:
        reward: np.ndarray
            The reward for each frame in the observed sequence
            size: (train_freq, )
        info: dict
            Required to have the following (for downstream visualization)
                - cost_matrix: np.ndarray (train_freq, ref_seq_len)
                - assignment_matrix: np.ndarray (train_freq, ref_seq_len)
    """
    # Calculate the cost matrix between the reference sequence and the observed sequence
    #   size: (train_freq, ref_seq_len)
    cost_matrix = cost_fn(obs, ref)

    # Assignment matrix is not actually used, but it's needed for visualization
    assignment = np.zeros_like(cost_matrix)
    # Set the last column to 1
    assignment[:, -1] = 1

    info = dict(
        cost_matrix=cost_matrix,
        assignment=assignment
    )

    return - cost_matrix[:, -1], info