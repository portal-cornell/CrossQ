from loguru import logger
import numpy as np

from vlm_reward.utils.optimal_transport import COST_FN_DICT, compute_ot_reward
from vlm_reward.utils.soft_dtw import compute_soft_dtw_reward
from vlm_reward.utils.dtw import compute_dtw_reward

def get_matching_fn(fn_config, cost_fn_name="nav_manhattan"):
    """
    """
    assert  fn_config["name"] in ["ot", "dtw", "soft_dtw"], f"Currently only supporting ['optimal_transport', 'dtw', 'soft_dtw'], got {fn_config['name']}"
    logger.info(f"[GridNavSeqRewardCallback] Using the following reward model:\n{fn_config}")

    cost_fn = COST_FN_DICT[cost_fn_name]
    scale = float(fn_config["scale"])
    fn_name = fn_config["name"]

    if fn_name == "ot":
        gamma = float(fn_config["gamma"])
        fn, fn_name = lambda obs_seq, ref_seq, cost_fn=cost_fn, gamma=gamma, scale=scale: compute_ot_reward(obs_seq, ref_seq, cost_fn, scale, gamma), f"{fn_name}_g={gamma}"
    elif fn_name == "dtw":
        fn, fn_name = lambda obs_seq, ref_seq, cost_fn=cost_fn, scale=scale: compute_dtw_reward(obs_seq, ref_seq, cost_fn, scale), fn_name
    elif fn_name == "soft_dtw":
        gamma = float(fn_config["gamma"])
        fn, fn_name = lambda obs_seq, ref_seq, cost_fn=cost_fn, gamma=gamma, scale=scale: compute_soft_dtw_reward(obs_seq, ref_seq, cost_fn, gamma, scale), f"{fn_name}_g={gamma}"
    else:
        raise NotImplementedError(f"Unknown sequence matching function: {fn_name}")
    
    post_processing_method = fn_config.get("post_processing_method", None)

    if post_processing_method:
        if post_processing_method == "stage_reward_based_on_last_state":
            fn_name += "_stg_lst"

            def post_processor(reward, matching_matrix):
                """
                Assuming the matching_matrix is time consistent.

                Parameters:
                    reward: np.ndarray (obs_seq_len, )
                    matching_matrix: np.ndarray (obs_seq_len, ref_seq_len)
                """
                previous_step_assignment = 0
                reward_bonus = 0

                rewards = []

                max_reward_range = fn_config["reward_vmax"] - fn_config["reward_vmin"]

                # print(f"reward={reward}")
                # print(f"matching_matrix={matching_matrix}")

                for i in range(len(reward)):
                    assignment = matching_matrix[i].argmax()

                    # print(f"i={i} assignment={assignment} previous_step_assignment={previous_step_assignment}")

                    if assignment != previous_step_assignment:
                        # Since reward[i] is the last reward at the end of the current stage, the reward bonus for 
                        #   the next stage should get updated
                        reward_bonus += -(-max_reward_range/2) + reward[i-1]
                    
                    new_reward = reward[i] + reward_bonus

                    rewards.append(new_reward)

                    previous_step_assignment = assignment

                    # print(f"i={i} reward[i]={reward[i]} reward_bonus={reward_bonus} new_reward={new_reward}")
                    # input("stop")
                
                return np.array(rewards)
                    
            def augmented_fn(*args, **kwargs):
                reward, info = fn(*args, **kwargs)
                new_reward = post_processor(reward, info["assignment"])

                return new_reward, info

            return augmented_fn, fn_name
        else:
            raise NotImplementedError(f"Unknown post processing method: {post_processing_method}")
    else:
        return fn, fn_name
    

action_to_direction = {
            0: np.array([1, 0]), # Down
            1: np.array([0, 1]), # Right
            2: np.array([-1, 0]), # Up
            3: np.array([0, -1]), # Left
            4: np.array([0, 0]) # Stay in place
        }

def update_location(agent_pos, action, map_array):
    direction = action_to_direction[action]

    new_pos = agent_pos + direction

    if is_valid_location(new_pos, map_array):
        return new_pos
    else:
        return agent_pos

def is_valid_location(pos, map_array):
    within_x_bounds = 0 <= pos[0] < map_array.shape[0]
    within_y_bounds = 0 <= pos[1] < map_array.shape[1]

    if within_x_bounds and within_y_bounds:
        not_a_hole = map_array[pos[0], pos[1]] != -1
        return  not_a_hole
    else:
        return False
    
def render_map_and_agent(map_array, agent_pos):
    map_with_agent = map_array.copy()

    map_with_agent[agent_pos[0], agent_pos[1]] = 1

    # Convert the map (0: empty, 1: agent, -1: hole) to an RGB image
    map_with_agent_rgb = np.zeros((map_array.shape[0], map_array.shape[1], 3), dtype=np.uint8)
    map_with_agent_rgb[map_with_agent == 0] = [203, 71, 119]
    map_with_agent_rgb[map_with_agent == 1] = [239, 248, 33]
    map_with_agent_rgb[map_with_agent == -1] = [12, 7, 134]

    # Increase the size of the image by a factor of 120
    map_with_agent_rgb = np.kron(map_with_agent_rgb, np.ones((120, 120, 1), dtype=np.uint8))

    return map_with_agent_rgb