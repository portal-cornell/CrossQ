"""
Generate triplets from sequence of rollouts.
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
FOLDER = f"{OUTPUT_ROOT}/v3_seq"
folder_path = "/share/portal/hw575/CrossQ/train_logs"



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

def sample_states(seq_data):
    # All indices available at the beginning, as we just sample one triplet per sequence
    available_indices = set(range(len(seq_data)))
    
    anchor_idx = random.choice(list(available_indices))
    available_indices.remove(anchor_idx)

    # At least 3 frames and at most 10 frames away from anchor
    pos_index_range = set(range(max(0, anchor_idx - 10), anchor_idx - 2)) | set(range(anchor_idx + 3, min(len(seq_data), anchor_idx + 11)))
    pos_index_range &= available_indices
    pos_idx = random.choice(list(pos_index_range)) if pos_index_range else None
    if pos_idx is not None:
        available_indices.remove(pos_idx)

    # At least 15 frames away from anchor
    neg_index_range = set(range(0, max(0, anchor_idx - 15))) | set(range(min(len(seq_data), anchor_idx + 16), len(seq_data)))
    neg_index_range &= available_indices
    neg_idx = random.choice(list(neg_index_range)) if neg_index_range else None
    
    return {"anchor": anchor_idx, "pos": pos_idx, "neg": neg_idx}

def generate_sample(env, folder_uid, init_qpos, iteration, state_type, chosen_frame_idx, state_data):
    curr_log = {f"qpos_{i}": 0.0 for i in range(2, 24)}
    curr_log.update({"uid": iteration, "itype": {"anchor": 0, "pos": 1, "neg": 2}[state_type], "step_type": "sequence"})

    # s[0] corresponds to qpos[2], s[21] corresponds to qpos[23]
    qpos = copy.deepcopy(init_qpos)
    qpos[2:24] = copy.deepcopy(state_data[0:22])
    env.unwrapped.set_state(qpos=qpos, qvel=np.zeros((23,)))

    obs = env.unwrapped.get_obs()
    frame = env.render()

    image_path = f"{FOLDER}/{state_type}/folder{folder_uid}_{iteration}_{state_type}_frame{chosen_frame_idx}.png"
    save_image(frame, image_path)

    joint_npy_path = f"{FOLDER}/{state_type}/folder{folder_uid}_{iteration}_{state_type}_frame{chosen_frame_idx}_joint_state.npy"
    save_joint_state(obs, joint_npy_path)

    geom_xpos_npy_path = f"{FOLDER}/{state_type}/folder{folder_uid}_{iteration}_{state_type}_frame{chosen_frame_idx}_geom_xpos.npy"
    geom_xpos = copy.deepcopy(env.unwrapped.data.geom_xpos)
    save_geom_xpos(geom_xpos, geom_xpos_npy_path)

    return log_data(curr_log, qpos, joint_npy_path, geom_xpos_npy_path, image_path), frame, geom_xpos


def generate_seq_triplets(args, env, folder_path, folder_uid):
    skip_viz = args.skip_viz
    viz_until = args.viz_until

    npy_files = [f for f in os.listdir(folder_path) if f.endswith("_states.npy")]
    print(f"Number of .npy files: {len(npy_files)}")

    output_logs = []
    total_npy_processed = 0

    # For every .npy file, we sample a triplet
    for idx, npy_file in enumerate(npy_files):
        print(f"Processing folder {folder_uid}, {idx}/{len(npy_files)}: {npy_file}")
        seq_data = np.load(os.path.join(folder_path, npy_file)) # seq is 120 long

        if len(seq_data.shape) != 2:
            continue

        iteration = npy_file.split("_")[0]

        env.reset()
        
        init_qpos = env.unwrapped.init_qpos
        init_geom_pos = copy.deepcopy(env.unwrapped.data.geom_xpos)

        # print(f"env.init_qpos=({init_qpos.shape}), \n{init_qpos}")
        # print(f"env.init_geom_pos=({init_geom_pos.shape}), \n{init_geom_pos}")

        chosen_states = sample_states(seq_data)
        print(f"Chosen states: {chosen_states}")

        geom_xpos_list = []
        log_data_list = []
        frames = []

        for i, state_type in enumerate(["anchor", "pos", "neg"]):
            chosen_frame_idx = chosen_states[state_type]
            log_data, frame, geom_xpos = generate_sample(env, folder_uid, init_qpos, iteration, state_type, chosen_frame_idx, seq_data[chosen_frame_idx])
            geom_xpos_list.append(geom_xpos)
            log_data_list.append(log_data)
            frames.append(frame)

        # Normalize geom_xpos
        anchor_geom_xpos_normalized = geom_xpos_list[0] - geom_xpos_list[0][1]
        pos_geom_xpos_normalized = geom_xpos_list[1] - geom_xpos_list[1][1]
        neg_geom_xpos_normalized = geom_xpos_list[2] - geom_xpos_list[2][1]

        # Check if positive sample is closer to anchor than negative sample
        if np.linalg.norm(anchor_geom_xpos_normalized - pos_geom_xpos_normalized) <= np.linalg.norm(anchor_geom_xpos_normalized - neg_geom_xpos_normalized):
            output_logs.extend(log_data_list)

            if not skip_viz and (viz_until == -1 or idx <= viz_until):
                fig, axes = plt.subplots(1, 3, figsize=(15, 6))
                for i, (frame, state_type) in enumerate(zip(frames, ["anchor", "pos", "neg"])):
                    axes[i].imshow(frame)
                    axes[i].set_title(f"{state_type.capitalize()} (frame {chosen_states[state_type]})")
                    axes[i].axis('off')
                
                plt.tight_layout()
                debug_folder = os.path.join(FOLDER, "debug")
                plt.suptitle(f"Sequence Triplet: Folder {folder_uid}, Iteration {iteration}")
                plt.savefig(f"{debug_folder}/triplet_{folder_uid}_{iteration}.png")
                plt.close(fig)
        else:
            print(f"Warning: Negative sample is closer to the anchor than the positive sample\npos-distance: {np.linalg.norm(anchor_geom_xpos_normalized - pos_geom_xpos_normalized)}\nneg-distance: {np.linalg.norm(anchor_geom_xpos_normalized - neg_geom_xpos_normalized)}")
            print(f"Skipping triplet for folder {folder_uid}, iteration {iteration}: positive sample farther than negative")

        total_npy_processed += 1

    return output_logs, total_npy_processed


def select_random_debug_samples(source_folder, dest_folder, num_samples=200):
    os.makedirs(dest_folder, exist_ok=True)

    all_files = [f for f in os.listdir(source_folder) if f.endswith('.png')]
    selected_files = random.sample(all_files, min(num_samples, len(all_files)))

    for file in selected_files:
        shutil.copy(os.path.join(source_folder, file), os.path.join(dest_folder, file))

    print(f"Copied {len(selected_files)} files to {dest_folder}")


if __name__ == "__main__":
    """
    python humanoid_generate_seq_triplets.py --output_log   # Used # --viz_until 100 --skip-viz
    python humanoid_generate_seq_triplets.py --debug --output_log --viz_until 10
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--output_log", action="store_true", help="Output log")
    parser.add_argument("--skip_viz", action="store_true", help="Skip visualization")
    parser.add_argument("--viz_until", type=int, default=-1, help="Viz until a certain iteration")
    args = parser.parse_args()

    set_seed(1231)

    
    # # folder_list = get_folder_with_gifs()

    # # # Attribute a UID to each folder and save the mapping
    # # folder_uids = {folder: f"uid_{i:04d}" for i, folder in enumerate(folder_list)}

    # # # Save the folder-UID mapping to a file
    # # with open(f"{FOLDER}/folder_uid_mapping.json", "w") as f:
    # #     json.dump(folder_uids, f, indent=2)

    # # # Save the list of folders with their UIDs
    # # with open(f"{FOLDER}/list_folder_used_for_seq.txt", "w") as f:
    # #     for folder, uid in folder_uids.items():
    # #         f.write(f"{uid}: {folder}\n")


    # Load the list from file
    with open(f"{FOLDER}/list_folder_used_for_seq.txt", "r") as f:
        folder_list_with_uids = f.readlines()
        folder_list_with_uids = [line.strip().split(": ", 1) for line in folder_list_with_uids]
        folder_uids = {uid: folder for uid, folder in folder_list_with_uids}
    

    if args.debug:
        folder_uids = {k: v for k, v in folder_uids.items() if k == "uid_0000"}
    
    make_env_kwargs = dict(
        episode_length = 120,
        reward_type = "original",
    )

    env = gymnasium.make(
        'HumanoidSpawnedUpCustom',
        render_mode="rgb_array",
        **make_env_kwargs,
    )

    # Stat: total number of gifs: 8796
    # total_num_gifs = 0
    # for folder in folder_list:
    #     # For every .npy in the folder (10k steps), we sample a triplet.
    #     # Check how many gifs we have in total
    #     gif_list = [f for f in os.listdir(folder) if f.endswith(".gif")]
    #     total_num_gifs += len(gif_list)

    # print(f"Total number of gifs: {total_num_gifs}")

    total_npy_processed = 0
    num_folders = len(folder_uids)
    for idx, (uid, folder) in enumerate(folder_uids.items()):
        print(f"Processing {idx}/{num_folders}: {folder}")
        output_logs, npy_processed = generate_seq_triplets(args, env, folder, uid)
        total_npy_processed += npy_processed
    
    print(f"Total number of npy processed: {total_npy_processed}")
    env.close()

    if args.output_log:
        suffix = f"_npy{total_npy_processed}"
        if args.debug:
            suffix = "_debug"
        with open(f"{FOLDER}/output_log_seq{suffix}.json", "w") as fout:
            json.dump(output_logs, fout)
    


    # select_random_debug_samples(f"{FOLDER}/debug", f"{FOLDER}/debug_200")


    print("Done")

"""
Total number of npy processed: 9037, 7308 triplets in total
Done
"""