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


args = parser.parse_args()

# Load the original gif (ignore the last frame which is the reset frame)
gif_obj = Image.open(args.gif_path)
frames = [gif_obj.seek(frame_index) or gif_obj.convert("RGB") for frame_index in range(gif_obj.n_frames)]

# Generate the index to subsample the gif to the desired subsample length
n_frames_to_skip = len(frames) // args.subsample_length
print(f"n_frames_to_skip={n_frames_to_skip}")
subsample_indices = [idx for idx in range(n_frames_to_skip, len(frames)-1, n_frames_to_skip)]

# Make the subsampled gif
subsampled_frames = [frames[idx] for idx in subsample_indices]

# Save the subsampled gif
imageio.mimsave(os.path.join("create_demo/seq_demos", f"{args.task_name}_real-rollout_{len(subsampled_frames)}-frames.gif"), subsampled_frames, duration=1/30, loop=0)

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
np.save(os.path.join("create_demo/seq_demos", f"{args.task_name}_real-rollout_{len(subsampled_frames)}-frames_geom-xpos.npy"), subsampled_geom_xpos)