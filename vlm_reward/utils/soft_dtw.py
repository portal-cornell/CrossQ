from tslearn.metrics import SoftDTW

import numpy as np

def compute_soft_dtw_reward(obs: np.ndarray, ref: np.ndarray, cost_fn, gamma=1, scale=1, modification_dict={}) -> np.ndarray:
    assert gamma > 0, "Currently not supporting gamma == 0"

    # Calculate the cost matrix between the reference sequence and the observed sequence
    #   size: (train_freq, ref_seq_len)
    cost_matrix = cost_fn(obs, ref)

    # if modification_dict != {}:
    #     if modification_dict["method"] == "equal_dist_cost":
    #         cost_scale = modification_dict["cost_scale"]
    #         scaling_matrix = cost_scale * np.ones_like(cost_matrix)

    #         n_obs_to_not_scale_per_ref = len(obs) // len(ref)

    #         i = 0
    #         for j in range(len(ref)):
    #             if j == len(ref) - 1:
    #                 scaling_matrix[i:, j] = 1
    #             else:
    #                 scaling_matrix[i:i+n_obs_to_not_scale_per_ref, j] = 1
    #             i += n_obs_to_not_scale_per_ref

    #         cost_matrix = cost_matrix * scaling_matrix

    #         # scale the cost matrix back to between 0 and 1
    #         cost_matrix = cost_matrix / cost_scale
    #     elif modification_dict["method"] == "nothing":
    #         pass
    #     else:
    #         raise NotImplementedError(f"Unknown method: {modification_dict['method']}")

    # sdtw = SoftDTW(cost_matrix, gamma=gamma)
    # dist_sq = sdtw.compute()  # We don't actually use this
    # a = sdtw.grad()

    # TODO: remove this
    # Manually set the assignment matrix a to be evenly distributed across the reference sequence

    a = np.zeros_like(cost_matrix)

    n_obs_per_ref = cost_matrix.shape[0] // cost_matrix.shape[1]

    i = 0
    for j in range(cost_matrix.shape[1]):
        if j == cost_matrix.shape[1] - 1:
            a[i:, j] = 1
        else:
            a[i:i+n_obs_per_ref, j] = 1
        i += n_obs_per_ref
    
    soft_dtw_cost = np.sum(cost_matrix * a, axis=1)  # size: (train_freq,)

    info = dict(
        assignment=a,
        cost_matrix=cost_matrix,
        transported_cost=cost_matrix * a,
    )

    return np.exp(- scale * soft_dtw_cost), info