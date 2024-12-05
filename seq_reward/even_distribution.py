import numpy as np

def identity_like(N, M):
    """
    Create an identity matrix of shape (N, M), such that each column has N // M 1s
    And the remainder is distributed as evenly as possible starting from the last column
    """

    # Base number of 1s per column
    k = N // M
    # Remainder to distribute among the first (N % M) columns
    remainder = N % M
    
    # Initialize an (N, M) zero matrix
    matrix = np.zeros((N, M), dtype=int)
    
    # Fill each column with k 1s, plus 1 additional 1 for the first `remainder` columns
    current_row = 0
    for col in range(M):
        num_ones = k + 1 if M - col - 1 < remainder else k
        matrix[current_row:current_row + num_ones, col] = 1
        current_row += num_ones  # Move to the next starting row
    return matrix

def compute_even_distribution_reward(obs: np.ndarray, ref: np.ndarray, cost_fn, scale=1, inverted_cost=False, modification_dict={}):
    """
    Compute reward between obs and ref based on an assignment matrix that evenly distributes the frames from obs to ref

    i.e., the first N frames from obs will be distributed to the first frame of ref, and so on, where N is len(obs) // len(ref)
    """
    # Calculate the cost matrix between the reference sequence and the observed sequence
    assignment = identity_like(len(obs), len(ref))
    normalized_assignment = assignment / np.expand_dims(np.sum(assignment, axis=1), 1)

    cost_matrix = cost_fn(obs, ref)

    even_distributed_cost = np.sum(normalized_assignment * cost_matrix, axis=1)

    if inverted_cost:
        final_reward = scale * even_distributed_cost
    else:
        final_reward = -scale * even_distributed_cost

    info = dict(
        assignment=normalized_assignment,
        original_assignment=assignment,
        cost_matrix=cost_matrix,
        transported_cost=cost_matrix * assignment,
    )

    return final_reward, info
