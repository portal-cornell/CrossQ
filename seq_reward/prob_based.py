import numpy as np


def compute_final_frame_reward(obs, ref, cost_fn, tau=20):
    """
    Reward = -d(obs, ref[-1])
    Just distance from final reference state (ignore the sequence)
    """
    cost_matrix = cost_fn(obs, ref) 
    assignment = np.zeros_like(cost_matrix)
    assignment[:, -1] = 1

    info = dict(
        assignment=assignment, # plot the inverse normalized cumulative cost
        original_assignment=assignment,
        cost_matrix=cost_matrix,
        transported_cost=cost_matrix,
    )

    final_reward = - np.sum(cost_matrix * assignment, axis=1)  # size: (train_freq,)

    return final_reward, info

def compute_log_coverage_reward(obs, ref, cost_fn, tau=1):

    """
    # max of coverage in previous learner timestep, coverage in current learner timestep up to previous state and occupying current state
    covered[t, t'] = max(covered[t-1, t'], covered[t, t'-1] * exp(-cost(t, t')) ) 
    covered[:, -1] = 1 # ref -1 is always covered
    covered[-1, :] = 0 # nothing has been covered by learner frame -1
    """
    cost_matrix = cost_fn(obs, ref) 
    prob_matrix = 1 - (1/tau)*cost_matrix

    covered = np.zeros_like(prob_matrix)
    covered[0,0] = prob_matrix[0,0]

    for i in range(1, covered.shape[0]):
        covered[i, 0] = max(covered[i-1, 0], prob_matrix[i, 0])

    for j in range(1, covered.shape[1]):
        covered[0, j] = covered[0, j-1] + prob_matrix[0, j]

    for i in range(1, covered.shape[0]):
        for j in range(1, covered.shape[1] - 1):
            covered[i,j] = max(covered[i-1, j], covered[i, j-1] + prob_matrix[i, j])
    
    covered[:, -1] = covered[:, -2] + prob_matrix[:, -1]


    # inverse_coverage = 1 - covered
    # normed_inverse_coverage = inverse_coverage / np.linalg.norm(inverse_coverage, axis=1, keepdims=True)
    # # Reward should be probability of being in states with low coverage
    # coverage_scaled_probabilities = prob_matrix * normed_inverse_coverage

    info = dict(
        assignment=covered, # plot the inverse normalized cumulative cost
        original_assignment=covered,
        cost_matrix=prob_matrix,
        transported_cost=covered,
    )

    # final_reward = coverage_scaled_probabilities.sum(axis=1)
    final_reward = covered[:, -1]
    return final_reward, info

def compute_coverage_reward(obs, ref, cost_fn, tau=1):

    """
    # max of coverage in previous learner timestep, coverage in current learner timestep up to previous state and occupying current state
    covered[t, t'] = max(covered[t-1, t'], covered[t, t'-1] * exp(-cost(t, t')) ) 
    covered[:, -1] = 1 # ref -1 is always covered
    covered[-1, :] = 0 # nothing has been covered by learner frame -1
    """
    cost_matrix = cost_fn(obs, ref) 
    prob_matrix = np.exp(-cost_matrix / tau)

    covered = np.zeros_like(prob_matrix)
    covered[0,0] = prob_matrix[0,0]

    for i in range(1, covered.shape[0]):
        covered[i, 0] = max(covered[i-1, 0], prob_matrix[i, 0])

    for j in range(1, covered.shape[1]):
        covered[0, j] = covered[0, j-1] * prob_matrix[0, j]

    for i in range(1, covered.shape[0]):
        for j in range(1, covered.shape[1] - 1):
            covered[i,j] = max(covered[i-1, j], covered[i, j-1] * prob_matrix[i, j])
    
    covered[:, -1] = covered[:, -2] * prob_matrix[:, -1]


    # inverse_coverage = 1 - covered
    # normed_inverse_coverage = inverse_coverage / np.linalg.norm(inverse_coverage, axis=1, keepdims=True)
    # # Reward should be probability of being in states with low coverage
    # coverage_scaled_probabilities = prob_matrix * normed_inverse_coverage

    info = dict(
        assignment=covered, # plot the inverse normalized cumulative cost
        original_assignment=covered,
        cost_matrix=prob_matrix,
        transported_cost=covered,
    )

    # final_reward = coverage_scaled_probabilities.sum(axis=1)
    final_reward = covered[:, -1]
    return final_reward, info


def compute_log_probability_reward(obs, ref, cost_fn, tau=20):
    """
    see overleaf
    hypothesis: tau should be ref seq length * max distance between two frames
    """
    cost_matrix = cost_fn(obs, ref) 
    
    # max_probs[i, j] represents the max probability that reference j was reached at any timestep before i
    # this is a lower bound on the total probability that reference j was reached by timestep i
    min_cost = np.zeros_like(cost_matrix)
    min_cost[0, :] = cost_matrix[0, :]
    min_cost[:, -1] = cost_matrix[:, -1]
    for i in range(1, min_cost.shape[0]):
        for j in range(min_cost.shape[1] - 1): # we want current probability of being in the last reference, not max so far (discourage moving out of final state)
            min_cost[i, j] = min(min_cost[i-1, j], cost_matrix[i, j]) # monotonically increasing probability matrix
                
    cumulative_cost = np.zeros_like(min_cost)
    cumulative_cost[:, 0] = min_cost[:,0]
    for i in range(min_cost.shape[0]):
        for j in range(1, min_cost.shape[1]):
            cumulative_cost[i, j] = min_cost[i,j] + cumulative_cost[i, j-1] 

    # cumulative_cost bounded above by tau (at most d for each reference subgoal, tau = d*len(ref))
    final_reward = 1 - (1/tau) * cumulative_cost[:,  -1] 
    
    info = dict(
        assignment=1 - (1/tau) * cumulative_cost, # plot the inverse normalized cumulative cost
        original_assignment=cumulative_cost,
        cost_matrix=cost_matrix,
        transported_cost=cumulative_cost,
    )
    return final_reward, info

def compute_probability_reward(obs, ref, cost_fn, tau, scale=1):
    cost_matrix = cost_fn(obs, ref)

    probability_matrix = np.exp(-cost_matrix/tau)
    
    # max_probs[i, j] represents the max probability that reference j was reached at any timestep before i
    # this is a lower bound on the total probability that reference j was reached by timestep i
    max_probs = np.zeros_like(probability_matrix)
    max_probs[0, :] = probability_matrix[0, :]
    max_probs[:, -1] = probability_matrix[:, -1]
    for i in range(1, max_probs.shape[0]):
        for j in range(max_probs.shape[1] - 1): # we want current probability of being in the last reference, not max so far (discourage moving out of final state)
            max_probs[i, j] = max(max_probs[i-1, j], probability_matrix[i, j]) # monotonically increasing probability matrix
                
    # cumulative_probs[i, j] represents a lower bound on the probability that reference j and all previous references were reached by timestep i
    cumulative_probs = np.zeros_like(probability_matrix)
    cumulative_probs[:, 0] = max_probs[:,0]
    #cumulative_probs[:, 0] = np.log(max_probs[:,0])
    for i in range(max_probs.shape[0]):
        for j in range(1, max_probs.shape[1]):
            # TODO: this can be converted to a sum of log probs if numerical instability occurs
            # especially likely to happen with a large number of reference states
            cumulative_probs[i, j] = max_probs[i,j] * cumulative_probs[i, j-1] 
            #cumulative_probs[i, j] = np.log(max_probs[i,j]) + cumulative_probs[i, j-1] # sum log probs to prevent numerical instability

    final_reward = scale*cumulative_probs[:,  -1] # only because we are doing np.log( ... np.exp())
    
    info = dict(
        assignment=cumulative_probs,
        original_assignment=cumulative_probs,
        cost_matrix=max_probs,
        transported_cost=cumulative_probs,
    )
    return final_reward, info


def compute_diagonal_probability_reward(obs, ref, cost_fn, max_cost):
    cost_matrix = cost_fn(obs, ref)
    cost_matrix /= max_cost # max cost is sort of like temperature
    probability_matrix = np.exp(-cost_matrix)
    #probability_matrix = (max_cost-cost_matrix) / max_cost
    max_probs = np.zeros((probability_matrix.shape[0], probability_matrix.shape[0], probability_matrix.shape[1], ))

    for t1 in range(max_probs.shape[0]):
        for t2 in range(t1, max_probs.shape[1]):
            for j in range(max_probs.shape[2]): 
                if t2 == t1: # if getting p(in ref j between time t and time t) = p(in ref j at time t)
                    max_probs[t1, t2, j] = probability_matrix[t1, j]
                else: # if getting p (in ref j between time t1 and time t2)
                    max_probs[t1, t2, j] = max(max_probs[t1, t2-1, j], probability_matrix[t2, j])

    # cumulative_probs[i, j] represents a lower bound on the probability that reference j reached in step i and all previous references were reached before i
    cumulative_probs = np.zeros_like(probability_matrix)
    cumulative_probs[:, 0] = max_probs[0,:,0] # cumulative_probs[i, 0] = max prob of being in ref 0 at any time between (0, i)

    for j in range(1, probability_matrix.shape[1]):
        for i in range(j, probability_matrix.shape[0]):
            if j == probability_matrix.shape[1] - 1:
                cumulative_probs[i, j] = probability_matrix[i, j] * cumulative_probs[i-1, j-1]
            else:
                max_probs_up_to_i = max_probs[1:i+1, i, j] 
                all_previous_before_i = cumulative_probs[:i, j-1] 
                p_in_i_and_reached_previous = max_probs_up_to_i * all_previous_before_i

                cumulative_probs[i, j] = max(p_in_i_and_reached_previous)


    final_reward = np.zeros(probability_matrix.shape[0])
    for i in range(len(final_reward)):
        if i < cumulative_probs.shape[1]:
            final_reward[i] = cumulative_probs[i, i]
        else:
            final_reward[i] = cumulative_probs[i, -1]

    info = dict(
        assignment=cumulative_probs,
        original_assignment=cumulative_probs,
        cost_matrix=probability_matrix,
        transported_cost=cumulative_probs,
    )
    return final_reward, info

def compute_ordered_probability_reward(obs, ref, cost_fn, max_cost):
    cost_matrix = cost_fn(obs, ref)
    cost_matrix /= max_cost
    probability_matrix = np.exp(-cost_matrix)
    #probability_matrix = (max_cost-cost_matrix) / max_cost

    # max_probs[i] represents max probability that the column's state was reached by timestep i
    max_probs = np.zeros_like(probability_matrix)
    max_probs[0, :] = probability_matrix[0, :]
    for i in range(1, max_probs.shape[0]):
        for j in range(max_probs.shape[1]):
            # monotonically increasing along rows, decreasing along columns probability matrix
            max_probs[i, j] = max(max_probs[i-1, j], probability_matrix[i, j])

    for i in range(max_probs.shape[0]):
        for j in range(1, max_probs.shape[1]):
            # monotonically increasing along rows, decreasing along columns probability matrix
            max_probs[i, j] = min(max_probs[i, j-1], max_probs[i, j])
                         
    # if there is an inversion, send probability to 0
    # new_max_probs = np.copy(max_probs)
    # for j in range(max_probs.shape[1]-1):
    #     for i in range(max_probs.shape[0]):
    #         if max_probs[i, j+1] > max_probs[i, j]:
    #             new_max_probs[i, j] = 0
    #             new_max_probs[i, j+1] = 0

    # max_probs = new_max_probs

    # cumu_probs[i, j] represents a lower bound on the probability that reference j was reached by timestep i
    cumu_log_probs = np.zeros_like(probability_matrix)
    #cumu_log_probs[:, 0] = np.log(max_probs[:,0])
    cumu_log_probs[:, 0] = max_probs[:,0]
    for i in range(max_probs.shape[0]):
        for j in range(1, max_probs.shape[1]):
            #normalized_log_prob = np.log(max_probs[i,j]) 
            #cumu_log_probs[i, j] = normalized_log_prob + cumu_log_probs[i, j-1]
            cumu_log_probs[i, j] = max_probs[i,j] * cumu_log_probs[i, j-1]

    final_reward = cumu_log_probs[:,  -1] # only because we are doing np.log( ... np.exp())
    #final_reward = cumu_log_probs[:,  -1] / max_cost /  cumu_log_probs.shape[1] # only because we are doing np.log( ... np.exp())
    #cumu_probs = np.sum(np.log(max_probs), axis=1)
    info = dict(
        assignment=cumu_log_probs,
        original_assignment=cumu_log_probs,
        cost_matrix=max_probs,
        transported_cost=cumu_log_probs,
    )
    return final_reward, info
