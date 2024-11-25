# This is not actually how we use metaworld
# import metaworld
# import random

# print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

# ml1 = metaworld.ML1('pick-place-v2') # Construct the benchmark, sampling tasks

# env = ml1.train_classes['pick-place-v2']()  # Create an environment with task `pick_place`
# task = random.choice(ml1.train_tasks)
# print(task)
# env.set_task(task)  # Set task

# obs = env.reset()  # Reset environment
# print(f"obs: {obs}")
# a = env.action_space.sample()  # Sample an action
# print(f"a: {a}")
# obs, reward, done, truncated, info = env.step(a)  # Step the environment with the sampled random action

# print(f"obs: {obs}")
# print(f"reward: {reward}")
# print(f"done: {done}")
# print(f"truncated: {truncated}")
# print(f"info: {info}")

import numpy as np
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
                            # these are ordered dicts where the key : value
                            # is env_name : env_constructor

door_open_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["door-open-v2-goal-observable"]
door_open_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN["door-open-v2-goal-hidden"]

env = door_open_goal_hidden_cls()
env.reset()  # Reset environment
a = env.action_space.sample()  # Sample an action
print(f"a: {a.shape}, {a}")
obs, reward, done, truncated, info = env.step(a)  # Step the environment with the sampled random action
assert (obs[-3:] == np.zeros(3)).all() # goal will be zeroed out because env is HiddenGoal

print(f"obs: {obs.shape}, {obs}")
print(f"reward: {reward}")

"""Information about observation (page 5 of the paper and https://github.com/Farama-Foundation/Metaworld/issues/337)

Observation shape: (39,)
It includes:
    - [3] 3D positions of the end-effector
    - [1] normalized measurement of how open the gripper is
    - [3] the 3D position of the first object
    - [4] the quaternion of the first object
    - [3] the 3D position of the second object
    - [4] the quaternion of the second object
    - [18] all of the previous measurements in the environment
    - [3] the 3D Cartesian position of the goal

If the second object is not present, the corresponding 7 elements are zeros.
If the goal is not observable, the last 3 elements are zeros.
"""
