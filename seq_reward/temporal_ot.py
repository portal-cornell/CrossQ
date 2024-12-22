"""
Code taken primarily from https://github.com/fuyw/TemporalOT/blob/58e59c19b3afd11986e0e734da2efa0361956c9e/models/temporalot.py
"""

import numpy as np
import torch
import ot
import numpy as np
from scipy.special import logsumexp

def bordered_identity_like(N, M, k):
    """
    Create an identity-like matrix of shape (N, M), such that each column has N // M 1s,
    the remainder is distributed as evenly as possible starting from the last column,
    and a border of width k is added on each side of the ones
    """
    # Base number of 1s per column
    base_ones = N // M
    # Remainder to distribute among the first (N % M) columns
    remainder = N % M

    # Initialize an (N, M) zero matrix
    matrix = np.zeros((N, M), dtype=int)

    # Fill each column with `base_ones` 1s, plus 1 additional 1 for the first `remainder` columns
    current_row = 0
    for col in range(M):
        num_ones = base_ones + 1 if M - col - 1 < remainder else base_ones
        matrix[current_row:current_row + num_ones, col] = 1
        current_row += num_ones  # Move to the next starting row

    # Create the border by adding k ones to the left and right of each row's 1s
    bordered_matrix = np.zeros_like(matrix)

    for row in range(N):
        for col in range(M):
            if matrix[row, col] == 1:
                start_col  = max(0, col - k)
                end_col = min(N, col + k + 1)
                bordered_matrix[row, start_col:end_col] = 1

    return bordered_matrix

def compute_temporal_ot_reward(obs, ref, cost_fn, mask_k=2, scale=1, niter=100, epsilon=.01):
    """
    Compute the TemporalOT reward, as defined in https://arxiv.org/abs/2410.21795
    Applies a diagonal mask to the OT objective, of width 2*mask_k + 1

    mask_k: the size of the mask window on each side
    """

    # context observations should be computed within cost_fn
    # with torch.no_grad():
    #     obs = self.cost_encoder(obs)
    # obs = self.get_context_observations(obs)
    # context cost matrix
    # cost_matrix = 0
    # for i in range(self.context_num):
    #     cost_matrix += cosine_distance(obs[i], exp[i])
    # cost_matrix /= self.context_num

    cost_matrix = cost_fn(obs, ref)

    mask = bordered_identity_like(cost_matrix.shape[0], cost_matrix.shape[1], k=mask_k)

    # optimal weights 
    transport_plan = mask_optimal_transport_plan(obs,
                                                    ref,
                                                    cost_matrix,
                                                    mask,
                                                    niter=niter,
                                                    epsilon=epsilon)

    # NOTE: they do not do this in TemporalOT paper, but we do for our OT baseline
    # They do an adaptive scaling of the OT based on the first min/max costs, but that's kinda weird
    normalized_transport_plan = transport_plan / np.expand_dims(np.sum(transport_plan, axis=1), 1)

    final_reward = -scale * np.diag(
        np.matmul(normalized_transport_plan, cost_matrix.T))
    
    info = dict(
        assignment=normalized_transport_plan,
        original_assignment=transport_plan,
        cost_matrix=cost_matrix,
        transported_cost=normalized_transport_plan,
    )
    return final_reward, info

def mask_sinkhorn(a, b, M, Mask, reg=0.01, numItermax=1000, stopThr=1e-9):
    # set a large value (1e6) for masked entry
    Mr = -M/reg*Mask + (-1e6)*(1-Mask)
    loga = np.log(a)
    logb = np.log(b)

    u = np.zeros(len(a))
    v = np.zeros(len(b))
    err = 1

    for i in range(numItermax):
        v = logb - logsumexp(Mr + u[:, None], 0)
        u = loga - logsumexp(Mr + v[None, :], 1)
        if i % 10 == 0:
            tmp_pi = np.exp(Mr + u[:, None] + v[None, :])
            err = np.linalg.norm(tmp_pi.sum(0) - b)
            if err < stopThr:
                return tmp_pi

    pi = np.exp(Mr + u[:, None] + v[None, :])
    return pi

def mask_optimal_transport_plan(X,
                                Y,
                                cost_matrix,
                                Mask,
                                niter=100,
                                epsilon=0.01):
    X_pot = np.ones(X.shape[0]) / X.shape[0]
    Y_pot = np.ones(Y.shape[0]) / Y.shape[0]
    transport_plan = mask_sinkhorn(X_pot,
                                   Y_pot,
                                   cost_matrix,
                                   Mask,
                                   epsilon,
                                   numItermax=niter)
    
    return transport_plan
