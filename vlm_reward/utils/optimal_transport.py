import numpy as np
import ot
from scipy.spatial.distance import cdist
from constants import SEQ_DICT
import heapq
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

def compute_ot_reward(obs: np.ndarray, ref: np.ndarray, cost_fn, scale=1, gamma=0.01, modification_dict={}) -> np.ndarray:
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

def plot_matrix_as_heatmap(matrix: np.ndarray, title: str, fp: str, cmap: str):
    """
    Plot the Assignment Matrix
    """
    # Because there are way less reference frames than observed frames, we need to copy the values and pad the reference frames for visualization
    padded_matrix = np.repeat(matrix, 10, axis=1)
    
    plt.imshow(padded_matrix, cmap=cmap, interpolation='nearest')
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

def euclidean_distance_advanced(x, y):
    """
    x: (x_batch_size, ...)
    y: (y_batch_size, ...)

    Specific use case:
        e.g. when we use geom_xpos, we have (batch_size, num_joints, 3)
        we want to calculate the euclidean distance between the joints

    Return: 
        euclidean distance matrix (x_batch_size, y_batch_size)
    """
    # To allow x and y to have different batch sizes, we will expand the dimensions of x and y
    #   to allow for broadcasting
    x_exp = np.expand_dims(x, axis=1)  # (x_batch_size, 1, ...)
    y_exp = np.expand_dims(y, axis=0)  # (1, y_batch_size, ...)
    
    # Calculate the norm along all axes except the batch axis
    norm_axis = tuple(range(2, len(x.shape)+1))

    return np.linalg.norm(x_exp - y_exp, axis=norm_axis)

def euclidean_distance_advanced_arms_only(x, y):
    """
    x: (x_batch_size, ...)
    y: (y_batch_size, ...)
    """
    x_arms_only = x[:, 12:, :]
    y_arms_only = y[:, 12:, :]
    # print(f"x: {x.shape}, y: {y.shape}")
    # To allow x and y to have different batch sizes, we will expand the dimensions of x and y
    #   to allow for broadcasting
    x_exp = np.expand_dims(x_arms_only, axis=1)  # (x_batch_size, 1, ...)
    y_exp = np.expand_dims(y_arms_only, axis=0)  # (1, y_batch_size, ...)
    
    # Calculate the norm along all axes except the batch axis
    norm_axis = tuple(range(2, len(x.shape)+1))

    return np.linalg.norm(x_exp - y_exp, axis=norm_axis)

def squared_euclidean_distance_advanced(x, y):
    return euclidean_distance_advanced(x, y) ** 2

def sigmoid_euclidean_distance(x, y):
    # Since euclidean is positive, we can scale the positive values of sigmoid to be between 0 and 1
    return 2 / (1 + np.exp(-euclidean_distance_advanced(x, y))) - 1

def sigmoid_euclidean_distance_arms_only(x, y):
    assert len(x.shape) == 3 and len(y.shape) == 3, f"x and y must have 3 dimensions, but got x={x.shape} and y={y.shape}"
    
    # Since euclidean is positive, we can scale the positive values of sigmoid to be between 0 and 1
    return 2 / (1 + np.exp(-euclidean_distance_advanced(x[:, 12:, :], y[:, 12:, :]))) - 1

def edit_distance(x, y):
    """
    x: (x_batch_size, A, B)
    y: (y_batch_size, A, B)

    Given two binary matrix M_1 (size (A, B)) and M_2 (size (A, B)), the edit distance is the minimum number of operations to transform M_1 to M_2
        We can calculate it by np.sum(np.abs(M_1 - M_2))

    Returns:
        edit_distance: (x_batch_size, y_batch_size)
    """
    # Expand the dimensions of a and b to make them broadcastable
    x_expanded = x[:, np.newaxis, :, :]  # Shape becomes (T, 1, 2, 2)
    y_expanded = y[np.newaxis, :, :, :]  # Shape becomes (1, T', 2, 2)
    
    return np.sum(np.abs(x_expanded - y_expanded), axis=(2, 3))

def nav_manhantan_distance(x, y):
    """
    x: (x_batch_size, A, B)
    y: (y_batch_size, A, B)

    Given two binary matrix M_1 (size (A, B)) and M_2 (size (A, B)). Each matrix has 1 to represent the agent, -1 to represent the obstacles, and 0 to represent the empty space. We want to calculate the Manhattan distance between the agent in M_1 and the agent in M_2.

    Warning: This function does not account for obstacles
    """
    x_batch_size = x.shape[0]
    y_batch_size = y.shape[0]

    cost_matrix = np.zeros((x_batch_size, y_batch_size))

    for i in range(x_batch_size):
        for j in range(y_batch_size):
            matrix1 = x[i]
            matrix2 = y[j]

            # Get the indices of the position of 1 (the agent) in each matrix
            pos1 = np.argwhere(matrix1 == 1)[0]  # Get the position of the '1' in matrix1
            pos2 = np.argwhere(matrix2 == 1)[0]  # Get the position of the '1' in matrix2

            cost_matrix[i, j] = np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1])

    return cost_matrix


def manhattan_heuristic(pos, goal):
    """
    Manhattan distance heuristic for A*.
    
    pos: (x, y) current position
    goal: (x, y) goal position
    """
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


# Directions for moving up, down, left, right
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def a_star_shortest_path(matrix, start, goal):
    """
    A* algorithm to find the shortest path from start to goal in a matrix avoiding obstacles.
    
    matrix: (A, B) binary matrix
    start: (x, y) tuple representing the starting position of the agent
    goal: (x, y) tuple representing the goal position of the agent
    """
    rows, cols = matrix.shape
    open_list = []
    heapq.heappush(open_list, (0, start))  # Priority queue (f_score, position)
    g_score = {start: 0}  # Cost from start to current position
    f_score = {start: manhattan_heuristic(start, goal)}  # Estimated cost from start to goal
    
    visited = set()

    while open_list:
        _, current = heapq.heappop(open_list)
        
        if current == goal:
            return g_score[current]
        
        if current in visited:
            continue
        visited.add(current)
        
        for direction in DIRECTIONS:
            new_row, new_col = current[0] + direction[0], current[1] + direction[1]
            neighbor = (new_row, new_col)
            
            if 0 <= new_row < rows and 0 <= new_col < cols and matrix[new_row, new_col] != -1:  # Stay within bounds and avoid obstacles
                tentative_g_score = g_score[current] + 1  # Distance between neighbors is always 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + manhattan_heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return float('inf')  # If there's no valid path to the goal

def nav_shortest_path_distance(x, y):
    """
    x: (x_batch_size, A, B)
    y: (y_batch_size, A, B)
    
    Given two binary matrices M_1 (size (A, B)) and M_2 (size (A, B)), 
    with 1 as the agent, -1 as obstacles, and 0 as empty space, 
    calculates the shortest path distance considering obstacles using A* algorithm.
    
    TODO: WILL
    modify this to create reward instead of cost
    - find maximum possible distance (using matrix shape, just height*width)
    - subtract the cost from this distance to get a reward
    """

    x_batch_size = x.shape[0]
    y_batch_size = y.shape[0]

    cost_matrix = np.zeros((x_batch_size, y_batch_size))

    for i in range(x_batch_size):
        for j in range(y_batch_size):
            matrix1 = x[i]
            matrix2 = y[j]

            max_cost = matrix1.shape[0] * matrix1.shape[1] # never will have to visit more than every state

            # Get the positions of the agent (marked as 1)
            pos1 = np.argwhere(matrix1 == 1)[0]
            pos2 = np.argwhere(matrix2 == 1)[0]

            # Find the shortest path between pos1 and pos2 in matrix1, considering obstacles
            sp_length = a_star_shortest_path(matrix1, tuple(pos1), tuple(pos2))
            
            cost_matrix[i, j] = max_cost - sp_length

    return cost_matrix


COST_FN_DICT = {
    "cosine": cosine_distance,
    "euclidean": euclidean_distance_advanced,
    "euclidean_arms_only": euclidean_distance_advanced_arms_only,
    "squared_euclidean": squared_euclidean_distance_advanced,
    "sigmoid_euclidean": sigmoid_euclidean_distance,
    "sigmoid_euclidean_arms_only": sigmoid_euclidean_distance_arms_only,
    "edit_distance": edit_distance,
    "nav_manhattan": nav_manhantan_distance,
    "nav_shortest_path": nav_shortest_path_distance,
}