import numpy as np

action_to_direction = {
            0: np.array([1, 0]), # Down
            1: np.array([0, 1]), # Right
            2: np.array([-1, 0]), # Up
            3: np.array([0, -1]), # Left
            4: np.array([0, 0]) # Stay in place
        }

def update_location(agent_pos, action, map_array, history):
    """
    Update the agent's location based on the action taken.
    
    If the new location is invalid, the agent stays in place.
        Invalid locations include:
            - Locations outside the map
            - Locations with a hole
            - Locations where the location has been visited before
    """
    direction = action_to_direction[action]

    new_pos = agent_pos + direction

    if is_valid_location(new_pos, map_array, history):
        return new_pos
    else:
        return agent_pos

def is_valid_location(pos, map_array, history):
    within_x_bounds = 0 <= pos[0] < map_array.shape[0]
    within_y_bounds = 0 <= pos[1] < map_array.shape[1]

    if within_x_bounds and within_y_bounds:
        not_a_hole = map_array[pos[0], pos[1]] != -1
        not_visited = pos.tolist() not in history

        return  not_a_hole and not_visited
    else:
        return False
    
def render_map_and_agent(map_array, agent_pos):
    map_with_agent = map_array.copy()

    map_with_agent[agent_pos[0], agent_pos[1]] = 1

    # Convert the map (0: empty, 1: agent, -1: hole) to an RGB image
    map_with_agent_rgb = np.zeros((map_array.shape[0], map_array.shape[1], 3), dtype=np.uint8)
    map_with_agent_rgb[map_with_agent == 0] = [203, 71, 119]
    map_with_agent_rgb[map_with_agent == 1] = [62, 245, 34] #[239, 248, 33]
    map_with_agent_rgb[map_with_agent == -1] = [12, 7, 134]

    # Increase the size of the image by a factor of 120
    map_with_agent_rgb = np.kron(map_with_agent_rgb, np.ones((120, 120, 1), dtype=np.uint8))

    return map_with_agent_rgb


def render_map(map_array):
    map_with_agent = map_array.copy()

    # Convert the map (0: empty, 1: agent, -1: hole) to an RGB image
    map_with_agent_rgb = np.zeros((map_array.shape[0], map_array.shape[1], 3), dtype=np.uint8)
    map_with_agent_rgb[map_with_agent == 0] = [203, 71, 119]
    map_with_agent_rgb[map_with_agent == 1] = [239, 248, 33]
    map_with_agent_rgb[map_with_agent == 2] = [62, 245, 34] 
    map_with_agent_rgb[map_with_agent == -1] = [12, 7, 134]

    # Increase the size of the image by a factor of 120
    map_with_agent_rgb = np.kron(map_with_agent_rgb, np.ones((120, 120, 1), dtype=np.uint8))

    return map_with_agent_rgb


if __name__=="__main__":
    from toy_examples.nav_3by5_0 import nav_3by5_0
    from toy_examples.nav_3by5_0_diff_pace_2 import nav_3by5_0_diff_pace_2
    from PIL import Image

    map_array = nav_3by5_0_diff_pace_2["map_array"]
    agent_pos = (1,0)
    g1 = (0, 0)
    g2 = (0, 3)
    g3 = (2, 3)

    map_array[*agent_pos] = 2
   # map_array[*g1] = 1
    map_array[*g2] = 1
    map_array[*g3] = 1

    render = render_map(map_array)
    Image.fromarray(render).save(f'same_pace.png')
