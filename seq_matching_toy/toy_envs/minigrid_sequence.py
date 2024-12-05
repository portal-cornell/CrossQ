import gymnasium as gym
from gymnasium.core import ObservationWrapper
import minigrid
from minigrid.minigrid_env import MiniGridEnv, MissionSpace, Grid
from minigrid.wrappers import FullyObsWrapper, NoDeath
from minigrid.manual_control import ManualControl
import pygame 
import numpy as np
from minigrid.core.world_object import Door, Lava, Ball, Wall
from minigrid.core.constants import COLORS
from enum import IntEnum
from minigrid.core.actions import Actions

class SequenceActions(IntEnum):
    # Turn left, turn right, move forward
    # Remove the options to pick up/drop objects, since these are unobservable
    left = 0
    right = 1
    forward = 2
    stay = 3

class SequenceEnv(MiniGridEnv):
    metadata = {"render_modes": ["rgb_array"]}
    
    def __init__(
        self,
        map_array,
        starting_pos=(1, 1, 0),
        episode_length=10,
        render_mode="rgb_array",
        temporal_encoding=False,
        highlight=False,
        **kwargs,
    ):
        self.agent_start_pos = np.array([starting_pos[0], starting_pos[1]])
        self.agent_start_dir = starting_pos[2]
        self.episode_length = episode_length

        self.map = map_array
        height = self.map.shape[0]
        width = self.map.shape[1]

        self.lava_cost = -5

        mission_space = MissionSpace(mission_func=self._gen_mission)
        actions = SequenceActions

        super().__init__(
            mission_space=mission_space,
            height=height, # +2 for space for outer barrier
            width=width,
            max_steps=episode_length,
            highlight=highlight,
            actions = actions,
            **kwargs,
        )

        self.lava_indices = np.argwhere(self.map.T==1)
        self.door_indices = np.argwhere(self.map.T==2)
        self.wall_indices = np.argwhere(self.map.T==-1)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.temporal_encoding = temporal_encoding

        # Observation space is (x,y,dir) or (x, y, dir, t)
        obs_shape = (4,) if self.temporal_encoding else (3,)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=max(3, self.width-1, self.height - 1),
            shape=obs_shape, 
            dtype="float32",
        )

    @staticmethod
    def _gen_mission():
        return "sequence mission"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        for i, j in self.lava_indices:
            self.put_obj(Lava(), i, j) # +1 because top/left outer barrier

        for i, j in self.wall_indices:
            self.put_obj(Wall(), i, j) # +1 because top/left outer barrier

        for idx, (i, j) in enumerate(self.door_indices):
            color = list(COLORS.keys())[idx]
            self.put_obj(Door(color), i, j)

        # Set i,j to be empty
        # self.grid.set(i, j, None)

    def set_state(self, agent_pos, agent_dir):
        self.agent_pos = agent_pos
        self.agent_dir = agent_dir

    def reset(
        self,
        *,
        seed: int | None = None,
        options = None
    ):
        # Reinitialize episode-specific variables
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert (
            self.agent_pos >= (0, 0)
            if isinstance(self.agent_pos, tuple)
            else all(self.agent_pos >= 0) and self.agent_dir >= 0
        )

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        #obs = self.gen_obs()
        obs = self.observation()

        return obs, {}

    def step(self, action):
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Automatically pick up objects
        if fwd_cell and fwd_cell.can_pickup():
            if self.carrying is None:
                self.carrying = fwd_cell
                self.carrying.cur_pos = np.array([-1, -1])
                self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        # Removed termination for reaching goal/lava
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type in ['lava']:
                reward = self.lava_cost

        # Do nothing
        elif action == self.actions.stay:
            pass
        else:
            raise ValueError(f"Unknown action: {action}")
        # # Drop an object
        # elif action == self.actions.drop:
        #     if not fwd_cell and self.carrying:
        #         self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
        #         self.carrying.cur_pos = fwd_pos
        #         self.carrying = None

        # # Toggle/activate an object
        # elif action == self.actions.toggle:
        #     if fwd_cell:
        #         fwd_cell.toggle(self, fwd_pos)


        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        #obs = self.gen_obs()
        obs = self.observation()
        info = {}
        
        return obs, reward, terminated, truncated, info

    def observation(self):
        agent_pos = self.agent_pos
        agent_dir = self.agent_dir
        temporal_encoding = self.step_count / self.episode_length
        
        if self.temporal_encoding:
            new_obs = np.array([agent_pos[0],agent_pos[1] , agent_dir, temporal_encoding], dtype="float32")
        else:
            new_obs = np.array([agent_pos[0],agent_pos[1] , agent_dir], dtype="float32")
        return new_obs

class LocationObsWrapper(ObservationWrapper):
    """
    Use the location as the only observation output, no image/language/mission.

    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import ImgObsWrapper
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> obs, _ = env.reset()
        >>> obs.keys()
        dict_keys(['image', 'direction', 'mission'])
        >>> env = ImgObsWrapper(env)
        >>> obs, _ = env.reset()
        >>> obs.shape
        (3)
    """

    def __init__(self, env):
        """A wrapper that makes image the only observation.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=max(3, env.width-1, env.height - 1), # TODO: this is upper bound on the box, but maybe not valid??
            shape=(3,),
            dtype="float32",
        )

        #self.observation_space = env.observation_space.spaces["image"]

    def observation(self, obs):
        agent_pos = self.env.agent_pos
        agent_dir = self.env.agent_dir
        
        new_obs = np.array([agent_pos[0],agent_pos[1] , agent_dir], dtype="float32")
        return new_obs
        
def make_sequence_env(**kwargs):
    env = SequenceEnv(**kwargs)

    # Disable deaths in the environment, so it does not reset when reaching a death state
    #env = NoDeath(env, no_death_types=("lava", "ball"), death_cost=-1)
    
    return env

# if __name__=="__main__":
    
#     # minigrid.register_minigrid_envs()
#     # #env = gym.make("MiniGrid-LavaCrossingS9N1-v0", render_mode="human", highlight=False)
#     # # env = NoDeath(env, no_death_types=("lava", "ball"))
#     # # env = FullyObsWrapper(env)
    
#     # gym.register(
#     #     id="sequence-v0",
#     #     entry_point=make_sequence_env,
#     # )
#     # env = gym.make("sequence-v0", tile_size=32, render_mode="human", highlight=False)

#     # agent_pos = env.unwrapped.agent_pos

#     pygame.init()
#     manual_control = ManualControl(env, seed=1234)
    
#     manual_control.start()

