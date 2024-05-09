"""
Generate an unlabelled dataset from a model checkpoint.
"""
import argparse
import json
import os
import subprocess

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from sbx import SAC, VLM_SAC

from utils import get_run_hash, set_os_vars, vlm_for_reward
from envs.base import get_make_env

def plot_info_on_frame(pil_image, info, font_size=20):
    # TODO: this is a hard-coded path
    font = ImageFont.truetype("/share/portal/hw575/vlmrm/src/vlmrm/cli/arial.ttf", font_size)
    draw = ImageDraw.Draw(pil_image)

    x = font_size  # X position of the text
    y = pil_image.height - font_size  # Beginning of the y position of the text
    
    i = 0
    for k in info:
        reward_text = f"{k}:{info[k]}"
        # Plot the text from bottom to top
        text_position = (x, y - 30*(i+1))
        draw.text(text_position, reward_text, fill=(255, 255, 255), font=font)
        i += 1



def generate_dataset(args):
    assert torch.cuda.is_available()

    set_os_vars()

    experiment_time, run_id = get_run_hash()
    run_name = f"{args.env}_s={args.seed}_{experiment_time}_{run_id}"
    
    checkpoint_path = os.path.join(args.model_base_path, args.model_checkpoint)
    print(f"Checkpoint: {checkpoint_path}")
    checkpoint_name = args.model_checkpoint[:-4]
    checkpoint_prefix = checkpoint_name.replace("_steps", "")

    torch.cuda.manual_seed(args.seed)

    # Initialize the environment
    use_vlm_for_reward = vlm_for_reward(args)

    if use_vlm_for_reward:
        make_env_kwargs = dict(
            episode_length = args.episode_length,
        )
    else:
        make_env_kwargs = dict(
            max_episode_steps = args.episode_length,
        )
    
    if "custom" in args.env.lower():
        make_env_kwargs["reward_type"] = args.reward_type

    make_env_fn = get_make_env(args.env, **make_env_kwargs)
    env = make_env_fn()

    # Load the model
    if use_vlm_for_reward:
        algo = VLM_SAC.load(
            path=str(checkpoint_path),
            args=args,
            inference_only=True,
            env=env,
            device="cuda:0"
        )
    else:
        algo = SAC.load(
            path=str(checkpoint_path),
            env=env,
            device="cuda:0"
        )

    print("Loaded learner model")

    inference_log_dir = f"./inference_logs/{run_name}/"
    # inference_img_log_dir = f"./inference_logs/{run_name}/img"
    inference_video_log_dir = f"./inference_logs/{run_name}/video"
    os.makedirs(inference_log_dir, exist_ok=True)
    # os.makedirs(inference_img_log_dir, exist_ok=True)
    os.makedirs(inference_video_log_dir, exist_ok=True)

    print(f"Generating dataset in {inference_log_dir} ...")
    
    for episode_idx in range(args.n_rollouts):
        if args.n_rollouts == 1:
            # if only 1 rollout, save by video name alone
            video_file_name = f"{args.video_base_name}.mp4"
        else:
            video_file_name = f"{args.video_base_name}_{checkpoint_prefix}_{episode_idx}.mp4"

        video_path = os.path.join(inference_video_log_dir, video_file_name)
        video_writer = imageio.get_writer(video_path, fps=30)

        video_with_info_file_name = f"with-info_{video_file_name}"
        video_with_info_writer = imageio.get_writer(os.path.join(inference_video_log_dir, video_with_info_file_name), 
                                                        fps=30)

        obs = env.reset(seed=args.seed + episode_idx)[0]
        for step_idx in range(args.episode_length):
            action = algo.predict(obs)[0]
            obs, _, _, _, info = env.step(action)

            image = env.render()
            image = np.uint8(image)
            pil_image = Image.fromarray(image)

            # Plot the ground truth reward
            plot_info_on_frame(pil_image, info)

            # image_file_name = f"{checkpoint_prefix}_{episode_idx}_{step_idx}.png"
            # image_path = str(os.path.join(inference_img_log_dir, image_file_name))
            # pil_image.save(image_path)
            
            video_writer.append_data(image)
            video_with_info_writer.append_data(np.uint8(pil_image))

        video_writer.close()
        video_with_info_writer.close()

        env.close()

        subprocess.run(["ffmpeg", "-i", str(video_path), "-vf", "fps=10,scale=1280:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse", "-loop", "0", str(video_path)[:-4] + ".gif"])

        print(f"Rollout {episode_idx+1} saved at {video_file_name}.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-env",         type=str, required=False, default="HumanoidStandup-v4", help="Set Environment.")
    parser.add_argument("-reward_type", type=str, required=False, default="original", help='Type of rewards to use')
    
    parser.add_argument("-seed",        type=int, required=False, default=1, help="Set Seed.")
    
    parser.add_argument("-reward_model_name", type=str, required=False, default="", help="Name of the reward model")
    parser.add_argument("-reward_batch_size", type=int, required=False, help="Batch size sent to the VLM reward model")
    parser.add_argument("-reward_config", type=str, required=False, default="", help="Path to the reward config file")

    parser.add_argument("-n_rollouts",        type=int, required=False, default=1, help="Number of rollouts / videos to generate.")
    parser.add_argument("-video_base_name",        type=str, required=False, default="standing_up", help="Name of the video when we just generate one video")
    
    parser.add_argument("-model_checkpoint",  type=str, required=False, default="final_model", help="Model checkpoint zip file name (without .zip).")
    parser.add_argument("-model_base_path",        type=str, required=True, help="Folder to all the checkpoints in a run.")

    parser.add_argument("-episode_length",   type=int,   required=False, default=240, help="maximum timestep in an episode")

    args = parser.parse_args()

    print(json.dumps(vars(args), indent=4))

    generate_dataset(args)
