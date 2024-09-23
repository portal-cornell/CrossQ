import numpy as np

import gymnasium as gym
from gymnasium import spaces

import imageio

DOWN = 0
RIGHT = 1
UP = 2
LEFT = 3
STAY = 4

PREDEFINED_MAPS = {
    "3x3": np.array([
        [0, 0, 0],
        [0, -1, 0],
        [0, 0, 0]
    ])
}

class GridNavigationEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, map_config, render_mode=None, grid_size=(3, 3)):
        self.grid_size = grid_size  # The size of the square grid

        # Observations is the agent's location in the grid
        self.observation_space = spaces.Discrete(grid_size[0] * grid_size[1])

        # We have 5 actions, corresponding to "right", "up", "left", "down", "do nothing"
        self.action_space = spaces.Discrete(5)

        self.map_config = map_config

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]), # Down
            1: np.array([0, 1]), # Right
            2: np.array([-1, 0]), # Up
            3: np.array([0, -1]), # Left
            4: np.array([0, 0]) # Stay in place
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    
    def step(self, action):
        direction = self._action_to_direction[action]

        self._update_location(direction)

        observation = self._get_obs()
        info = self._get_info()
        reward = 0  # We are using sequence matching function to produce reward

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, False, info

    def _update_location(self, direction):
        new_pos = self._agent_pos + direction

        if self._is_valid_location(new_pos):
            self._agent_pos = new_pos
        else:
            self._agent_pos = self._agent_pos

    def _is_valid_location(self, pos):
        within_x_bounds = 0 <= pos[0] < self.grid_size[0]
        within_y_bounds = 0 <= pos[1] < self.grid_size[1]
        not_a_hole = self._map[pos[0], pos[1]] != -1
        return within_x_bounds and within_y_bounds and not_a_hole

    def _get_obs(self):
        return self._agent_pos

    def _get_info(self):
        return {
            
        }
    
    def reset(self):
        self._map = self._load_map(self.map_config)
        self._agent_pos = np.array([0, 0])
        return self._get_obs()

    def _load_map(self, map_config):
        if "name" in map_config:
            return PREDEFINED_MAPS[map_config["name"]]
        elif "map" in map_config:
            return map_config["map"]
        else:
            raise ValueError(f"Invalid map config: {map_config}. Need to specify either 'name' or 'map'")
        
    def render(self):
        if self.render_mode == "rgb_array":
            map_with_agent = self._map.copy()

            map_with_agent[self._agent_pos[0], self._agent_pos[1]] = 1

            # Convert the map (0: empty, 1: agent, -1: hole) to an RGB image
            map_with_agent_rgb = np.zeros((self.grid_size[0], self.grid_size[1], 3), dtype=np.uint8)
            map_with_agent_rgb[map_with_agent == 0] = [203, 71, 119]
            map_with_agent_rgb[map_with_agent == 1] = [239, 248, 33]
            map_with_agent_rgb[map_with_agent == -1] = [12, 7, 134]

            # Increase the size of the image by a factor of 120
            map_with_agent_rgb = np.kron(map_with_agent_rgb, np.ones((120, 120, 1), dtype=np.uint8))

            return map_with_agent_rgb
        else:
            return
    
if __name__ == "__main__":
    env = GridNavigationEnv({"name": "3x3"}, [], render_mode="rgb_array")
    env.reset()
    env.render()
    
    path = [RIGHT, RIGHT, DOWN, DOWN, LEFT, LEFT, UP, STAY]

    frames = []
    frames.append(env.render())

    for action in path:
        env.step(action)
        frames.append(env.render())

    imageio.mimsave("testing.gif", frames, duration=1/20, loop=0)

    writer = imageio.get_writer('testing.mp4', fps=20)

    for im in frames:
        writer.append_data(im)
    
    writer.close()