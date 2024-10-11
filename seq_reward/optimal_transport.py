import numpy as np
import ot

def compute_ot_reward(obs: np.ndarray, ref: np.ndarray, cost_fn, scale=1, gamma=0.01,uncertainty_scaling_matrix=None, modification_dict={}) -> np.ndarray:
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
    if uncertainty_scaling_matrix is not None:
        cost_matrix = cost_matrix * uncertainty_scaling_matrix

    if modification_dict != {}:
        if modification_dict["method"] == "equal_dist_cost":
            cost_scale = modification_dict["cost_scale"]
            scaling_matrix = cost_scale * np.ones_like(cost_matrix)

            n_obs_to_not_scale_per_ref = len(obs) // len(ref)

            i = 0
            for j in range(len(ref)):
                if j == len(ref) - 1:
                    scaling_matrix[i:, j] = 1
                else:
                    scaling_matrix[i:i+n_obs_to_not_scale_per_ref, j] = 1
                i += n_obs_to_not_scale_per_ref

            cost_matrix = cost_matrix * scaling_matrix

            # scale the cost matrix back to between 0 and 1
            cost_matrix = cost_matrix / cost_scale
        elif modification_dict["method"] == "nothing":
            pass
        else:
            raise NotImplementedError(f"Unknown method: {modification_dict['method']}")

    # Calculate the OT plan between the reference sequence and the observed sequence
    obs_weight = np.ones(obs.shape[0]) / obs.shape[0]
    ref_weight = np.ones(ref.shape[0]) / ref.shape[0]

    if gamma == 0:
        T = ot.emd(obs_weight, ref_weight, cost_matrix)  # size: (train_freq, ref_seq_len)
    else:
        T = ot.sinkhorn(obs_weight, ref_weight, cost_matrix, reg=gamma, log=False)  # size: (train_freq, ref_seq_len)

    # Normalize the path so that each row sums to 1
    normalized_T = T / np.expand_dims(np.sum(T, axis=1), 1)

    # Calculate the OT cost for each timestep
    #   sum by row of (cost matrix * OT plan)
    ot_cost = np.sum(cost_matrix * normalized_T, axis=1)  # size: (train_freq,)

    info = dict(
        assignment=normalized_T,
        original_assignment=T,
        cost_matrix=cost_matrix,
        transported_cost=cost_matrix * T,
    )

    return - scale * ot_cost, info