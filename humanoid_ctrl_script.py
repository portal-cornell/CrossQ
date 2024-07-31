import gymnasium
from loguru import logger
import os
import imageio
import numpy as np
import copy
from PIL import Image
import argparse

import envs
from create_demo.pose_config import pose_config_dict

# Set up
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# Get egl (mujoco) rendering to work on cluster
os.environ["NVIDIA_VISIBLE_DEVICES"] = "all"
os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "compute,graphics,utility,video"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_PLATFORM"] = "device"

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pose_name",   type=str,   required=True, help="Name of the demo to generate data for (correspond to create_demo/pose_config.py)")
parser.add_argument("--debug", default=False, action="store_true")

args = parser.parse_args()

assert args.pose_name in pose_config_dict, f"-pose_name={args.pose_name} does not match any of the key in pose_config_dict in create_demo/pose_config.py"

if args.debug:
    # Initialize video writer for debugging 
    video_writer = imageio.get_writer("debugging/humanoid_env/testing.mp4", fps=30)

# Load the humanoid environment
make_env_kwargs = dict(
    episode_length = 120
)

env = gymnasium.make(
            'HumanoidSpawnedUpCustom',
            render_mode="rgb_array",
            **make_env_kwargs,
        )

# Reset the environment
env.reset()

init_qpos = env.unwrapped.init_qpos

if args.debug:
    video_writer.append_data(np.uint8(env.render()))

    init_qvel = env.unwrapped.init_qvel
    print(f"env.init_qpos=({init_qpos.shape}), \n{init_qpos}")
    print(f"env.init_qvel=({init_qvel.shape}), \n{init_qvel}")

    obs = env.unwrapped.get_obs()
    print(f"before: {obs.shape}\n{obs[:22].shape}\n{obs[:22]}")

# Set the humanoid to the pose specified in pose_config_dict
joint_config = pose_config_dict[args.pose_name]

new_qpos = copy.deepcopy(init_qpos)
for idx in joint_config.keys():
    new_qpos[int(idx)] = joint_config[int(idx)] # z-coordinate of torso

# Set the humanoid joint state to the specified ones, velocity stays at 0
env.unwrapped.set_state(qpos=new_qpos, qvel=np.zeros((23,)))

obs = env.unwrapped.get_obs()
with open(f"create_demo/demos/{args.pose_name}_joint-state.npy", "wb") as fout:
    print(f"Obs after setting the pose: {obs.shape}\n{obs[:22].shape}\n{obs[:22]}")
    # Based on https://www.gymlibrary.dev/environments/mujoco/humanoid/ and
    #   https://docs.google.com/spreadsheets/d/17xxmlh8oLAh7vLlRGtMz4b79ueb0KwccQIvbBa0m3kk/edit?usp=sharing
    np.save(fout, obs[:22])

# Render the environment to visualize the pose
frame = env.render()
image = Image.fromarray(frame)
image.save(f"create_demo/demos/{args.pose_name}.png")

if args.debug:
    video_writer.append_data(np.uint8(frame))

    # Check for pose's stability
    # Action space size from https://www.gymlibrary.dev/environments/mujoco/humanoid/
    for i in range(20):
        env.step(np.zeros((17,)))
        video_writer.append_data(np.uint8(env.render()))

        obs = env.unwrapped.get_obs()
        print(f"{i}:")
        for j in range(len(obs[:22])):
            print(f"  qpos: {j+2}: {obs[:22][j]:.2f}")
        
    video_writer.close()

# Close the environment
env.close()
