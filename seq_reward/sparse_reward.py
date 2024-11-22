import numpy as np

def compute_sparse_reward(obs: np.ndarray, ref: np.ndarray, cost_fn, radius, goal_bonus):
    """
    Provide a bonus for when the agent reaches a given reference state, scaled by how far along the reference state is
    """
    cost_matrix = cost_fn(obs, ref)

    closest_ref = np.argmin(cost_matrix, axis=1)
    rewards = np.zeros(len(obs))

    for i, r in enumerate(closest_ref):
       
        if cost_matrix[i, r] < radius:
            rewards[i] = goal_bonus * (r+1) / len(ref)
    info = {}
    
    return rewards, info