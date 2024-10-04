import numpy as np

def compute_dtw_reward(obs: np.ndarray, ref: np.ndarray, cost_fn, scale=1, inverted_cost=False, modification_dict={}):
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

    _, accumulated_cost_matrix = _dtw(cost_matrix)
    path = _dtw_path(accumulated_cost_matrix)

    # Normalize the path so that each row sums to 1
    normalized_path = path / np.expand_dims(np.sum(path, axis=1), 1)

    info = dict(
        assignment=normalized_path,
        original_assignment=path,
        cost_matrix=cost_matrix,
        transported_cost=cost_matrix * path,
    )

    soft_dtw_cost = np.sum(cost_matrix * normalized_path, axis=1)  # size: (train_freq,)
        
    if inverted_cost:
        final_reward = scale * soft_dtw_cost
    else:
        final_reward = -scale * soft_dtw_cost

    return final_reward, info

def _dtw(cost_matrix):
    l1, l2 = cost_matrix.shape
    acc_cost_mat = np.full((l1 + 1, l2 + 1), np.inf)
    acc_cost_mat[0, 0] = 0.0

    for i in range(1, l1+1):
        for j in range(1, l2+1):
                # cost_matrix is 0-indexed, acc_cost_mat is 1-indexed
                acc_cost_mat[i, j] = cost_matrix[i-1, j-1] + min(
                    acc_cost_mat[i-1, j-1], acc_cost_mat[i-1, j], acc_cost_mat[i, j-1],
                )
    return acc_cost_mat[-1, -1], acc_cost_mat[1:, 1:]

def _dtw_path(acc_cost_mat):
    sz1, sz2 = acc_cost_mat.shape
    path = [(sz1 - 1, sz2 - 1)]

    while path[-1] != (0, 0):
        i, j = path[-1]
        if i == 0:
            path.append((0, j - 1))
        elif j == 0:
            path.append((i - 1, 0))
        else:
            arr = np.array(
                [
                    acc_cost_mat[i - 1][j - 1],
                    acc_cost_mat[i - 1][j],
                    acc_cost_mat[i][j - 1],
                ]
            )
            argmin = np.argmin(arr)

            if argmin == 0:
                path.append((i - 1, j - 1))
            elif argmin == 1:
                path.append((i - 1, j))
            else:
                path.append((i, j - 1))

    path_matrix = np.zeros_like(acc_cost_mat)

    for i, j in path:
        path_matrix[i, j] = 1

    return path_matrix