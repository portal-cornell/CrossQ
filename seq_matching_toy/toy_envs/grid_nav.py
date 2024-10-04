import numpy as np

import gymnasium as gym
from gymnasium import spaces

import imageio

from seq_matching_toy.seq_utils import update_location, update_obs, render_map_and_agent

DOWN = 0
RIGHT = 1
UP = 2
LEFT = 3
STAY = 4

class GridNavigationEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, map_array, starting_pos, render_mode=None, episode_length=10):
        self.grid_size = map_array.shape  # The size of the square grid

        # Observations is the agent's location in the grid
        self.observation_space = spaces.Box(np.zeros((2,)), np.array([grid_size - 1 for grid_size in self.grid_size]), shape=(2,), dtype=np.int64)

        # We have 5 actions, corresponding to "right", "up", "left", "down", "do nothing"
        self.action_space = spaces.Discrete(5)
        
        self.num_steps = 0
        self.episode_length = episode_length
        self.map = map_array

        self._starting_pos = starting_pos

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    
    def step(self, action):
        self._agent_pos = update_location(agent_pos=self._agent_pos, action=action, map_array=self.map)

        observation = self._get_obs()
        info = self._get_info()
        reward = 0  # We are using sequence matching function to produce reward

        self.num_steps += 1
        terminated = self.num_steps >= self.episode_length

        return observation, reward, terminated, False, info

    def _get_obs(self):
        return self._agent_pos
    
    def _get_info(self):
        return {"step": self.num_steps}

    def reset(self, seed=0):
        """
        This is a deterministic environment, so we don't use the seed."""
        self.num_steps = 0
        self._agent_pos = np.copy(self._starting_pos)

        return self._get_obs(), self._get_info()

    def render(self):
        """
        Render the environment as an RGB image.
        
        The agent is represented by a yellow square, empty cells are white, and holes are blue."""
        if self.render_mode == "rgb_array":
            return render_map_and_agent(self.map, self._agent_pos)
        else:
            return
    
if __name__ == "__main__":
    env = GridNavigationEnv(
        np.array([
            [0, 0, 0],
            [0, -1, 0],
            [0, 0, 0]
        ]), 
        render_mode="rgb_array")
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

class GridNavigationEnvHistory(GridNavigationEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_cells = self.grid_size[0] * self.grid_size[1]
        self.observation_space = spaces.MultiDiscrete(np.ones((n_cells,))*3) # 3 possible states for each grid location (unvisited, visited, current)
        self.visited = np.zeros(self.grid_size)
    
    def _get_obs(self):
        observation = self.visited.copy()
        observation[tuple(self._agent_pos)] = 2
        observation = observation.flatten()
        return observation

    def reset(self, seed=0):
        """
        This is a deterministic environment, so we don't use the seed."""
        self.num_steps = 0
        self._agent_pos = np.copy(self._starting_pos)
        self.visited = np.zeros(self.grid_size)

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.visited[tuple(self._agent_pos)] = 1
        self._agent_pos = update_location(agent_pos=self._agent_pos, action=action, map_array=self.map)
        
        observation = self._get_obs()
        assert len(np.nonzero(observation == 2)) == 1
        info = self._get_info()
        reward = 0  # We are using sequence matching function to produce reward

        self.num_steps += 1
        terminated = self.num_steps >= self.episode_length
        print(f"step {self.num_steps}: {observation}")

        return observation, reward, terminated, False, info