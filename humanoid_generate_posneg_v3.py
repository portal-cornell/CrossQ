"""
Generate triplets with random joint poses or taking steps in the environment.

Example commands:

# python humanoid_generate_poses.py -p both-arms-out --debug
# p humanoid_generate_posneg_v3.py -r -k 1 --output_log --debug
# p humanoid_generate_posneg_v3.py -r -k 1000 --output_log --debug
"""
import gymnasium
from loguru import logger
import os
import imageio
import numpy as np
import copy
from PIL import Image
import argparse
import json, random

import envs
from create_demo.pose_config import pose_config_dict
from inference import plot_info_on_frame

from utils_data_gen.random_poses import generate_random_pose_config
# Read the poses threshold
from utils_data_gen.ft_qpos_stats import poses_thres
from utils_humanoid_generate import *

# Set up
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# Get egl (mujoco) rendering to work on cluster
os.environ["NVIDIA_VISIBLE_DEVICES"] = "all"
os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "compute,graphics,utility,video"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_PLATFORM"] = "device"

OUTPUT_ROOT = "finetuning/data/"
FOLDER = f"{OUTPUT_ROOT}/v3_random_joints"

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pose_name",   type=str, help="Name of the demo to generate data for (correspond to create_demo/pose_config.py)")
parser.add_argument("-r", "--random_pose", action="store_true", help="randomly sample a pose")
parser.add_argument("-k", "--k", type=int, default=1, help="Sample k random poses")
parser.add_argument("--debug", default=False, action="store_true", help="Store a video and add more prints to help visualize the data")
parser.add_argument("--output_log", action="store_true", help="Output the data as a json file")

args = parser.parse_args()

# Load the humanoid environment
make_env_kwargs = dict(
    episode_length = 120,
    reward_type = "both_arms_out_goal_only_euclidean"
)

env = gymnasium.make(
    'HumanoidSpawnedUpCustom',
    render_mode="rgb_array",
    **make_env_kwargs,
)

set_seed(1231)


# Log: uid, type (anchor, pos, neg), all the joint states, image path, npy path
output_logs = []



def save_image(frame, path):
    image = Image.fromarray(frame)
    image.save(path)

def save_joint_state(obs, path):
    with open(path, "wb") as fout:
        np.save(fout, obs[:22])

def log_data(curr_log, qpos, npy_path, image_path):
    for idx in range(2, len(qpos)):
        curr_log[f"qpos_{idx}"] = qpos[idx]
    curr_log["npy_path"] = npy_path
    curr_log["image_path"] = image_path
    return curr_log

def generate_anchor_sample(args, env, iteration, joint_config, init_qpos):
    curr_log = {f"qpos_{i}": 0.0 for i in range(2, 24)}
    curr_log.update({"uid": iteration, "itype": 0, "step_type": None})

    new_qpos = copy.deepcopy(init_qpos)
    for idx in joint_config.keys():
        new_qpos[int(idx)] = joint_config[int(idx)]
    
    env.unwrapped.set_state(qpos=new_qpos, qvel=np.zeros((23,)))

    obs = env.unwrapped.get_obs()
    frame = env.render()

    anchor_npy_path = f"{FOLDER}/anchor/{iteration}_joint_state.npy"
    save_joint_state(obs, f"{anchor_npy_path}")

    anchor_image_path = f"{FOLDER}/anchor/{iteration}_pose.png"
    save_image(frame, anchor_image_path)

    return log_data(curr_log, new_qpos, anchor_npy_path, anchor_image_path)

def generate_positive_sample(args, env, iteration, pos_i, joint_config, init_qpos_copy, step_type):
    curr_log = {f"qpos_{i}": 0.0 for i in range(2, 24)}
    curr_log.update({"uid": iteration, "itype": 1, "step_type": step_type})

    reset_initial_qpos = set_joints(joint_config, init_qpos_copy)
    env.unwrapped.set_state(qpos=reset_initial_qpos, qvel=np.zeros((23,)))

    if step_type == "step":
        env.step(np.random.uniform(-0.3, 0.3, (17,)))
    else:  # "pose"
        joints_to_change, _ = generate_random_pose_config()
        new_qpos = perturb_joints_positively(reset_initial_qpos, joints_to_change, poses_thres)
        env.unwrapped.set_state(qpos=new_qpos, qvel=np.zeros((23,)))

    obs = env.unwrapped.get_obs()
    frame = env.render()

    if step_type == "step":
        new_qpos = obs

    image_path = f"{FOLDER}/pos/{iteration}_{pos_i}_{step_type}.png"
    save_image(frame, image_path)

    npy_path = f"{FOLDER}/pos/{iteration}_{pos_i}_{step_type}_joint_state.npy"
    save_joint_state(obs, f"{npy_path}")

    return log_data(curr_log, new_qpos, npy_path, image_path)

def generate_negative_sample(args, env, iteration, neg_i, joint_config, init_qpos_copy):
    curr_log = {f"qpos_{i}": 0.0 for i in range(2, 24)}
    curr_log.update({"uid": iteration, "itype": 2, "step_type": "pose"})

    reset_initial_qpos = set_joints(joint_config, init_qpos_copy)
    env.unwrapped.set_state(qpos=reset_initial_qpos, qvel=np.zeros((23,)))

    joints_to_change, _ = generate_random_pose_config()
    new_qpos = perturb_joints_negatively(reset_initial_qpos, joints_to_change, poses_thres)
    env.unwrapped.set_state(qpos=new_qpos, qvel=np.zeros((23,)))

    obs = env.unwrapped.get_obs()
    frame = env.render()

    image_path = f"{FOLDER}/neg/{iteration}_{neg_i}_pose.png"
    save_image(frame, image_path)

    npy_path = f"{FOLDER}/neg/{iteration}_{neg_i}_pose_joint_state.npy"
    save_joint_state(obs, f"{npy_path}")

    return log_data(curr_log, new_qpos, npy_path, image_path)




# Sample k random poses
for iteration in range(args.k):

    # Dictionary to store the current json entry
    if args.output_log:
        curr_log = {f"qpos_{i}": 0.0 for i in range(2, 24)}
        curr_log["uid"] = iteration
        curr_log["itype"] = 0 # 0: anchor, 1: pos, 2: neg
        curr_log["step_type"] = None # None, "pose", "step"

    # if args.debug:
    #     # Initialize video writer for debugging 
    #     video_writer = imageio.get_writer(f"{FOLDER}/video/{iteration}.mp4", fps=30)

    # Reset the environment
    env.reset()

    init_qpos = env.unwrapped.init_qpos
    init_qpos_copy = copy.deepcopy(init_qpos) # may not be necessary?

    if args.debug:
        # video_writer.append_data(np.uint8(env.render()))

        init_qvel = env.unwrapped.init_qvel
        print(f"env.init_qpos=({init_qpos.shape}), \n{init_qpos}")
        print(f"env.init_qvel=({init_qvel.shape}), \n{init_qvel}")

        obs = env.unwrapped.get_obs()
        print(f"before: {obs.shape}\n{obs[:22].shape}\n{obs[:22]}")

    # Set the humanoid to the pose specified in pose_config_dict
    if args.pose_name:
        joint_config = pose_config_dict[args.pose_name]
    elif args.random_pose:
        _, joint_config = generate_random_pose_config()

    # with open(f"{FOLDER}/anchor/{iteration}_joint_config.json", "w") as fout:
    #     json.dump(joint_config, fout)

    # Generate anchor sample
    output_logs.append(generate_anchor_sample(args, env, iteration, joint_config, init_qpos))

    # Positive samples
    pos_i = 0
    for i in range(1):
        pos_i += 1
        output_logs.append(generate_positive_sample(args, env, iteration, pos_i, joint_config, init_qpos_copy, "step"))

    for i in range(2):
        pos_i += 1
        output_logs.append(generate_positive_sample(args, env, iteration, pos_i, joint_config, init_qpos_copy, "pose"))

    # Negative samples
    neg_i = 0
    for i in range(3):
        neg_i += 1
        output_logs.append(generate_negative_sample(args, env, iteration, neg_i, joint_config, init_qpos_copy))

    # if args.debug:
    #     video_writer.close()

env.close()

if args.output_log:
    with open(f"{FOLDER}/output_log_{args.k}.json", "w") as fout:
        json.dump(output_logs, fout)