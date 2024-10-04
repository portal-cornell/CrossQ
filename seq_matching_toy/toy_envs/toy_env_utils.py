import numpy as np

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