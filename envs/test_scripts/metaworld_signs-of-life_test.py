import metaworld
import random

print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

ml1 = metaworld.ML1('pick-place-v2') # Construct the benchmark, sampling tasks

env = ml1.train_classes['pick-place-v2']()  # Create an environment with task `pick_place`
task = random.choice(ml1.train_tasks)
print(task)
env.set_task(task)  # Set task

obs = env.reset()  # Reset environment
print(f"obs: {obs}")
a = env.action_space.sample()  # Sample an action
print(f"a: {a}")
obs, reward, done, truncated, info = env.step(a)  # Step the environment with the sampled random action

print(f"obs: {obs}")
print(f"reward: {reward}")
print(f"done: {done}")
print(f"truncated: {truncated}")
print(f"info: {info}")