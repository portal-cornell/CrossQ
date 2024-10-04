"""
Example Command

python create_demo/convert_rollout_to_seq.py -p /share/portal/hw575/CrossQ/train_logs/2024-09-26-152534_sb3_sac_envr=right_arm_extend_wave_higher_goal_only_euclidean_geom_xpos_rm=hand_engineered_s=9_nt=arms-only-geom/eval/2000000_rollouts.gif -t right-arm-extend-wave-higher -l 2 --debug --manual

python create_demo/convert_rollout_to_seq.py -p /share/portal/hw575/CrossQ/create_demo/seq_demos/right-arm-extend-wave-higher_5-frames.gif -t right-arm-extend-wave-higher -l 2 --debug --manual
"""

import argparse
import imageio
import os
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--gif_path",   type=str,   required=True, help="Name of the demo to generate data for (correspond to create_demo/pose_config.py)")
parser.add_argument("-t", "--task_name", type=str, required=True, help="Name of the task to generate data for")
parser.add_argument("-l", "--subsample_length", default=1, type=int, help="After subsampling, this is the desired length for the subsample sequence")
parser.add_argument("--debug", default=False, action="store_true", help="Store a video and add more prints to help visualize the data")
parser.add_argument("--manual", default=False, action="store_true", help="If true, will manually select the subsample indices")

args = parser.parse_args()

# Load the original gif (ignore the last frame which is the reset frame)
gif_obj = Image.open(args.gif_path)
frames = [gif_obj.seek(frame_index) or gif_obj.convert("RGB") for frame_index in range(gif_obj.n_frames)]

if args.manual:
    reward_path = args.gif_path.replace(".gif", "_rewards.npy")

    # Find the indices that sort the reward from highest to lowest
    rewards = np.load(reward_path)
    sorted_indices = np.argsort(rewards)[::-1]

    print(f"top 5 sorted_indices={sorted_indices[:5]}, rewards={rewards[sorted_indices[:5]]}")
    print(f"bottom 5 sorted_indices={sorted_indices[-5:]}, rewards={rewards[sorted_indices[-5:]]}")
    for i in range(1, 6):
        print(f"bottom {5*i} to {5*(i+1)} sorted_indices={sorted_indices[-5*(i+1): -5*i]}, rewards={rewards[sorted_indices[-5*(i+1): -5*i]]}")
    
    # Warning: To align with how interpolation generates rollout, we will keep the initial frame in the subsampled frames. The initial frame will get ignored when this gets loaded as a reference seq
    subsample_indices = [0]
    while True:
        # Get the next index to subsample
        next_index = int(input("Enter the next index to subsample: "))
        if next_index == -1:
            break
        subsample_indices.append(next_index)

    rollout_tag = "hand-picked-rollout"
else:
    # Generate the index to subsample the gif to the desired subsample length
    n_frames_to_skip = len(frames) // args.subsample_length
    print(f"n_frames_to_skip={n_frames_to_skip}")
    subsample_indices = [0] + [idx for idx in range(n_frames_to_skip, len(frames)-1, n_frames_to_skip)]

    rollout_tag = "real-rollout"

# Warning: To align with how interpolation generates rollout, we will keep the initial frame in the subsampled frames. The initial frame will get ignored when this gets loaded as a reference seq
# Make the subsampled gif
subsampled_frames = [frames[idx] for idx in subsample_indices]

# We subtract out the 1st frame
total_ref_frames = len(subsampled_frames) - 1

# Save the subsampled gif
imageio.mimsave(os.path.join("create_demo/seq_demos", f"{args.task_name}_{rollout_tag}_{total_ref_frames}-frames.gif"), subsampled_frames, duration=1/30, loop=0)

if args.debug:
    # Initialize video writer for debugging 
    video_writer = imageio.get_writer("debugging/humanoid_env/testing.mp4", fps=30)

    for frame in subsampled_frames:
        video_writer.append_data(np.uint8(frame))

    video_writer.close()

# Load the original geom_xpos.npy file
geom_xpos_path = args.gif_path.replace(".gif", "_geom_xpos_states.npy")

geom_xpos = np.load(geom_xpos_path)

# Subsample the geom_xpos to the desired subsample length
subsampled_geom_xpos = geom_xpos[subsample_indices]

# Save the subsampled geom_xpos
np.save(os.path.join("create_demo/seq_demos", f"{args.task_name}_{rollout_tag}_{total_ref_frames}-frames_geom-xpos.npy"), subsampled_geom_xpos)