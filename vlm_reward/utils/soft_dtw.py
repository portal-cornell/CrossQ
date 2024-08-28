from tslearn.metrics import SoftDTW

import numpy as np

def compute_soft_dtw_reward(obs: np.ndarray, ref: np.ndarray, cost_fn, gamma=1, scale=1) -> np.ndarray:
    assert gamma > 0, "Currently not supporting gamma == 0"

    # Calculate the cost matrix between the reference sequence and the observed sequence
    #   size: (train_freq, ref_seq_len)
    cost_matrix = cost_fn(obs, ref)

    sdtw = SoftDTW(cost_matrix, gamma=gamma)
    dist_sq = sdtw.compute()  # We don't actually use this
    a = sdtw.grad()
    
    soft_dtw_cost = np.sum(cost_matrix * a, axis=1)  # size: (train_freq,)

    info = dict(
        assignment=a,
        cost_matrix=cost_matrix,
        transported_cost=soft_dtw_cost,
    )

    return - scale * soft_dtw_cost, info