"""
Generate frames from sequence of rollouts.
"""
import gymnasium
from loguru import logger
import os
import copy
import imageio
import numpy as np
import copy
from PIL import Image
import argparse
import json, random
from tqdm import tqdm
import matplotlib.pyplot as plt
import random, shutil

import envs
from inference import plot_info_on_frame

from utils_data_gen.utils_humanoid_generate import *

from clean_folder import clean_folder


OUTPUT_ROOT = "finetuning/data/"
FOLDER = f"{OUTPUT_ROOT}/v4_seq_frames_every5_debug_sharded"
folder_path = "/share/portal/hw575/CrossQ/train_logs"

STEP_SIZE = 5

os.makedirs(FOLDER, exist_ok=True)
os.makedirs(f"{FOLDER}/images", exist_ok=True)

import shutil
shutil.copy(f"/home/aw588/git_annshin/CrossQ/finetuning/data/v4_seq_frame20-25/list_folder_used_for_seq.txt", 
            f"{FOLDER}/list_folder_used_for_seq.txt")


os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# Get egl (mujoco) rendering to work on cluster
os.environ["NVIDIA_VISIBLE_DEVICES"] = "all"
os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "compute,graphics,utility,video"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_PLATFORM"] = "device"



def get_folder_with_gifs():
    has_gif_folder_list = []
    for f in os.listdir(folder_path):
        has_gif = clean_folder(os.path.join(folder_path, f))
        if has_gif:
            has_gif_folder_list.append(os.path.join(folder_path, f, "eval"))
    # print(has_gif_folder_list)
    return has_gif_folder_list

def get_npy_stats(folder_list):
    total_npy_files = 0
    for folder in folder_list:
        npy_files = [f for f in os.listdir(folder) if f.endswith(".npy")]
        total_npy_files += len(npy_files)
    return total_npy_files


def generate_sample(env, folder_uid, init_qpos, iteration, frame_idx, state_data):
    qpos = copy.deepcopy(init_qpos)
    qpos[2:24] = copy.deepcopy(state_data[0:22])
    env.unwrapped.set_state(qpos=qpos, qvel=np.zeros((23,)))

    obs = env.unwrapped.get_obs()
    frame = env.render()

    # Create a subfolder for each folder_uid
    subfolder = f"{FOLDER}/images/{folder_uid}"
    os.makedirs(subfolder, exist_ok=True)

    image_path = f"{subfolder}/folder{folder_uid}_{iteration}_frame{frame_idx}.png"
    save_image(frame, image_path)

    joint_npy_path = f"{subfolder}/folder{folder_uid}_{iteration}_frame{frame_idx}_joint_state.npy"
    save_joint_state(obs, joint_npy_path)

    geom_xpos_npy_path = f"{subfolder}/folder{folder_uid}_{iteration}_frame{frame_idx}_geom_xpos.npy"
    geom_xpos = copy.deepcopy(env.unwrapped.data.geom_xpos)
    save_geom_xpos(geom_xpos, geom_xpos_npy_path)

    return image_path

def generate_samples(args, env, folder_path, folder_uid):
    npy_files = [f for f in os.listdir(folder_path) if f.endswith("_states.npy")]
    print(f"Number of .npy files: {len(npy_files)}")

    total_npy_processed = 0
    all_png_paths = []

    for idx, npy_file in enumerate(npy_files):
        print(f"Processing folder {folder_uid}, {idx}/{len(npy_files)}: {npy_file}")
        seq_data = np.load(os.path.join(folder_path, npy_file))

        if len(seq_data.shape) != 2:
            continue

        iteration = npy_file.split("_")[0]
        env.reset()
        init_qpos = env.unwrapped.init_qpos

        for frame_idx in range(0, len(seq_data), STEP_SIZE):
            # For the first 4 sampled frames, randomly select within the step range
            if frame_idx < STEP_SIZE * 4:
                if random.random() > 0.25:
                    continue
                # Randomly sample within the current step range
                start = frame_idx
                end = min(frame_idx + STEP_SIZE, len(seq_data))
                sampled_idx = random.randint(start, end - 1)
                state_data = seq_data[sampled_idx]
            else:
                sampled_idx = frame_idx
                state_data = seq_data[sampled_idx]

            png_path = generate_sample(env, folder_uid, init_qpos, iteration, sampled_idx, state_data)
            all_png_paths.append(png_path)

        total_npy_processed += 1

    return total_npy_processed, all_png_paths

if __name__ == "__main__":
    """
    python humanoid_generate_seq_data.py --debug
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    set_seed(1231)

    # Load the list from file
    with open(f"{FOLDER}/list_folder_used_for_seq.txt", "r") as f:
        folder_list_with_uids = f.readlines()
        folder_list_with_uids = [line.strip().split(": ", 1) for line in folder_list_with_uids]
        folder_uids = {uid: folder for uid, folder in folder_list_with_uids}

    if args.debug:
        folder_uids = {k: v for k, v in folder_uids.items() if k in ["uid_0000"]} #, "uid_0001", "uid_0002"]}
    
    make_env_kwargs = dict(
        episode_length = 120,
        reward_type = "original",
    )

    env = gymnasium.make(
        'HumanoidSpawnedUpCustom',
        render_mode="rgb_array",
        **make_env_kwargs,
    )

    total_npy_processed = 0
    all_png_paths = []
    num_folders = len(folder_uids)
    for idx, (uid, folder) in tqdm(enumerate(folder_uids.items()), total=num_folders):
        print(f"Processing {idx}/{num_folders}: {folder}")
        npy_processed, png_paths = generate_samples(args, env, folder, uid)
        all_png_paths.extend(png_paths)
        total_npy_processed += npy_processed
    
    print(f"Total number of npy processed: {total_npy_processed}")
    env.close()

    # Save all PNG paths to a JSON file
    with open(f"{FOLDER}/all_png_paths.json", "w") as fout:
        json.dump(all_png_paths, fout)

    print("Done")