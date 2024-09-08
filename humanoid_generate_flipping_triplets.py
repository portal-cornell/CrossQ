"""
Generate (anchor, pos, neg) triplets with flipping (hand poses)

Approach:
- Get an anchor pose
- Pos: slight perturbation of the joints/take steps
- Neg: same as pos but mirrored
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
import matplotlib.pyplot as plt

import envs
from utils_data_gen.random_poses import generate_random_pose_config
from utils_data_gen.utils_humanoid_generate import *

# Set up
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# Get egl (mujoco) rendering to work on cluster
os.environ["NVIDIA_VISIBLE_DEVICES"] = "all"
os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "compute,graphics,utility,video"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_PLATFORM"] = "device"

OUTPUT_ROOT = "finetuning/data/"
FOLDER = f"{OUTPUT_ROOT}/v3_flipping_debug"


##########################################
########### From Yuki's branch ###########
##########################################

SEQ_ROOT = "/home/aw588/git_annshin/CrossQ_yuki/create_demo/demos"
SEQ_DICT = {
    "arms_bracket_right_final_only": [f"{SEQ_ROOT}/arms_bracket_right_joint-state.npy"],

    "arms_bracket_down_final_only": [f"{SEQ_ROOT}/arms_bracket_down_joint-state.npy"],

    "left_arm_extend_wave_higher_final_only": [f"{SEQ_ROOT}/left-arm-extend-wave-higher_joint-state.npy"],

    "both_arms_out_final_only": [f"{SEQ_ROOT}/both-arms-out_joint-state.npy"],
    "both_arms_out_with_intermediate": [f"{SEQ_ROOT}/left-arm-out_joint-state.npy", f"{SEQ_ROOT}/both-arms-out_joint-state.npy"],
    
    "both_arms_up_final_only": [f"{SEQ_ROOT}/arms_bracket_up_joint-state.npy"],
    "both_arms_up_with_intermediate": [f"{SEQ_ROOT}/both-arms-out_joint-state.npy", f"{SEQ_ROOT}/arms_bracket_up_joint-state.npy"],

    "arms_up_then_down": [f"{SEQ_ROOT}/left-arm-out_joint-state.npy", f"{SEQ_ROOT}/both-arms-out_joint-state.npy", f"{SEQ_ROOT}/right-arm-out_joint-state.npy"],
}

# Function to load an anchor pose
def load_reference_seq(seq_name: str, use_geom_xpos: bool) -> np.ndarray:
    """
    Load the reference sequence for the given sequence name
    """
    ref_seq = []
    for joint in SEQ_DICT[seq_name]:
        if use_geom_xpos:
            new_fp = str(joint).replace("joint-state", "geom-xpos")
        else:
            new_fp = joint

        loaded_joint_states = np.load(new_fp)

        if use_geom_xpos:
            # Normalize the joint states based on the torso (index 1)
            loaded_joint_states = loaded_joint_states - loaded_joint_states[1]

        ref_seq.append(loaded_joint_states)
    return np.stack(ref_seq)

###################################################
########### Define the flipping values ############
###################################################

# along x
flipping_map_arms = {
    18: 21,
    19: 22,
    # 20: 23,
}
flipping_map_pelvis_hip = {
    10: 14,
    11: 15,
    # 12: 16,
    13: 17,
}
flipping_map_body_x = [3, 9]


##########################################
########### Generation script ############
##########################################

# TODO: deduplicate with humanoid_generate_posneg_v3.py
def generate_anchor_sample(args, env, iteration, joint_config, init_qpos):
    curr_log = {f"qpos_{i}": 0.0 for i in range(2, 24)}
    curr_log.update({"uid": iteration, "itype": 0, "step_type": None})

    new_qpos = copy.deepcopy(init_qpos)
    for idx in joint_config.keys():
        new_qpos[int(idx)] = joint_config[int(idx)]
    
    env.unwrapped.set_state(qpos=new_qpos, qvel=np.zeros((23,)))

    obs = env.unwrapped.get_obs()
    frame = env.render()

    anchor_joint_npy_path = f"{FOLDER}/anchor/{iteration}_joint_state.npy"
    save_joint_state(obs, f"{anchor_joint_npy_path}")

    anchor_geom_xpos_npy_path = f"{FOLDER}/anchor/{iteration}_geom_xpos.npy"
    geom_xpos = copy.deepcopy(env.unwrapped.data.geom_xpos)
    save_geom_xpos(geom_xpos, f"{anchor_geom_xpos_npy_path}")

    anchor_image_path = f"{FOLDER}/anchor/{iteration}_pose.png"
    save_image(frame, anchor_image_path)

    return log_data(curr_log, new_qpos, anchor_joint_npy_path, anchor_geom_xpos_npy_path, anchor_image_path), new_qpos


def mirror_qpos(neg_qpos):
    # Mirror arms
    for left, right in flipping_map_arms.items():
        neg_qpos[left], neg_qpos[right] = neg_qpos[right], neg_qpos[left]

    # Mirror pelvis and hip
    for left, right in flipping_map_pelvis_hip.items():
        neg_qpos[left], neg_qpos[right] = neg_qpos[right], neg_qpos[left]
    
    # Flip body angles
    for idx in flipping_map_body_x:
        neg_qpos[idx] = -neg_qpos[idx]  # Flip the angle
        # Ensure the angle stays within [-pi, pi]
        if neg_qpos[idx] < -np.pi:
            neg_qpos[idx] += 2 * np.pi
        elif neg_qpos[idx] > np.pi:
            neg_qpos[idx] -= 2 * np.pi
    
    return neg_qpos

def generate_flipping_triplets(args, env, seq_name: str, use_geom_xpos: bool, num_triplets: int = 1000, output_logs: list = []):
    triplets = []
    for iteration in range(num_triplets):
        env.reset()

        init_qpos = env.unwrapped.init_qpos
        init_geom_pos = copy.deepcopy(env.unwrapped.data.geom_xpos)

        print(f"env.init_qpos=({init_qpos.shape}), \n{init_qpos}")
        print(f"env.init_geom_pos=({init_geom_pos.shape}), \n{init_geom_pos}")

        # Get anchor pose

        # We only use the poses in SEQ_DICT for test data
        if seq_name:
            # Load the reference sequence
            ref_seq = load_reference_seq(seq_name, use_geom_xpos)
            if len(ref_seq) == 1:
                new_qpos = ref_seq[0]
            else:
                raise ValueError("Only single sequence is supported for now")
            anchor_qpos = copy.deepcopy(init_qpos)
            for idx in range(2, 24):
                anchor_qpos[idx] = new_qpos[idx - 2] # because anchor_qpos is (22) while init_qpos is (24)
        else:
            # Otherwise, random pose = anchor pose
            _, joint_config = generate_random_pose_config()
            anchor_qpos = copy.deepcopy(init_qpos)
        
        # Get anchor pose
        anchor_log_data, anchor_qpos = generate_anchor_sample(args, env, iteration, joint_config, anchor_qpos)
        output_logs.append(anchor_log_data)

        # Negative: Mirror the anchor state
        neg_qpos = copy.deepcopy(anchor_qpos)
        neg_qpos = mirror_qpos(neg_qpos)

        # Positive: take a few steps from the anchor
        pos_qpos = copy.deepcopy(anchor_qpos)
        pos_log = generate_positive_sample(args, env, iteration, pos_i, joint_config, init_qpos_copy, "pose")
        output_logs.append(pos_log)


        # Visualization
        if not args.skip_viz and (args.viz_until == -1 or iteration <= args.viz_until):
            fig, axes = plt.subplots(1, 3, figsize=(15, 6))
            frames = []

            # Anchor
            anchor_frame = plt.imread(anchor_log_data["image_path"])
            frames.append(anchor_frame)
            axes[0].imshow(anchor_frame)
            axes[0].set_title("Anchor")
            axes[0].axis('off')

            # Positive
            env.unwrapped.set_state(qpos=pos_qpos, qvel=np.zeros((23,)))
            pos_frame = env.render()
            frames.append(pos_frame)
            axes[1].imshow(pos_frame)
            axes[1].set_title("Positive")
            axes[1].axis('off')

            # Negative
            env.unwrapped.set_state(qpos=neg_qpos, qvel=np.zeros((23,)))
            neg_frame = env.render()
            frames.append(neg_frame)
            axes[2].imshow(neg_frame)
            axes[2].set_title("Negative")
            axes[2].axis('off')

            plt.tight_layout()
            plt.suptitle(f"Flipping Triplet: Iteration {iteration}")
            plt.savefig(f"{debug_folder}/flipping_triplet_{iteration}.png")
            plt.close(fig)

        triplets.append((anchor_qpos, pos_qpos, neg_qpos))
    
    return triplets



if __name__ == "__main__":
    """
    python humanoid_generate_flipping_triplets.py --num_triplets 10 --viz_until 5
    python humanoid_generate_flipping_triplets.py --seq_name both_arms_up_final_only --num_triplets 2 --use_geom_xpos 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, default="", help="Name of the sequence to generate triplets for. Only for manual test set")
    parser.add_argument("--use_geom_xpos", action="store_true", help="Use geom xpos instead of joint states")
    parser.add_argument("--num_triplets", type=int, default=1000, help="Number of triplets to generate")
    parser.add_argument("--skip_viz", action="store_true", help="Skip visualization")
    parser.add_argument("--viz_until", type=int, default=-1, help="Viz until a certain iteration")
    args = parser.parse_args()

    make_env_kwargs = dict(
        episode_length = 120,
        # reward_type = "both_arms_out_goal_only_euclidean"
        reward_type = "original"
    )

    env = gymnasium.make(
        'HumanoidSpawnedUpCustom',
        render_mode="rgb_array",
        **make_env_kwargs,
    )

    # Create a debug folder for visualizations
    debug_folder = os.path.join(FOLDER, "debug")
    os.makedirs(debug_folder, exist_ok=True)

    output_logs = []

    triplets = generate_flipping_triplets(args, env, args.seq_name, args.use_geom_xpos, args.num_triplets, output_logs)    # import pdb; pdb.set_trace()
    # env.close()

    # # Save the triplets
    # os.makedirs(f"{FOLDER}/{args.seq_name}", exist_ok=True)
    # process_triplets(args, env, triplets)
