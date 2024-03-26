import pathlib
from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.humanoidstandup_v4 import HumanoidStandupEnv as GymHumanoidStandupEnv
from gymnasium.spaces import Box
from numpy.typing import NDArray


class HumanoidStandupCurriculum(GymHumanoidStandupEnv):
    # TODO: add init that takes in an stage indicator
    #   in this init, let's deifne a mapping from stage indicator to reward function

    # TODO: reward function for sit up
   
    def step(self, a):
         # TODO: step function should call the corresponding reward function based on the stage indicator
        self.do_simulation(a, self.frame_skip)
        pos_after = self.data.qpos[2]
        data = self.data
        uph_cost = (pos_after - 0) / self.model.opt.timestep

        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return (
            self._get_obs(),
            reward,
            False,
            False,
            dict(
                reward_linup=uph_cost,
                reward_quadctrl=-quad_ctrl_cost,
                reward_impact=-quad_impact_cost,
            ),
        )

    

