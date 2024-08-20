"""
Generate an unlabelled dataset from a model checkpoint.
"""
import os

from omegaconf import DictConfig, OmegaConf
import hydra

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from sbx import SAC, VLM_SAC
from vlm_reward.utils.utils import rewards_matrix_heatmap, rewards_line_plot, pad_to_longest_sequence
from vlm_reward.reward_main import compute_rewards, load_reward_model
from vlm_reward.reward_transforms import half_gaussian_filter_1d
import utils

from envs.base import get_make_env

def plot_info_on_frame(pil_image, info, font_size=20):
    """
    Parameters:
        pil_image: PIL.Image
            - The image to plot the text on
        info: Dict
            - The information to plot on the image
        font_size: int
            - The size of the font
    """
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


@hydra.main(version_base=None, config_path="configs", config_name="eval_config")
def generate_dataset(cfg: DictConfig):
    """
    Generate videos/gif of the agent's behavior in the environment.

    Parameters:
        cfg: DictConfig
            - The hydra config object
    """
    assert torch.cuda.is_available()

    utils.set_os_vars()
    
    checkpoint_path = os.path.join(cfg.model_base_path, cfg.model_checkpoint)
    print(f"Checkpoint: {checkpoint_path}")
    checkpoint_name = cfg.model_checkpoint[:-4]  # Remove the .zip
    checkpoint_prefix = checkpoint_name.replace("_steps", "")

    torch.cuda.manual_seed(cfg.seed)

    use_vlm_for_reward = utils.use_vlm_for_reward(cfg)

    # Initialize the environment
    make_env_kwargs = utils.get_make_env_kwargs(cfg)

    make_env_fn = get_make_env(cfg.env.name, **make_env_kwargs)
    env = make_env_fn()

    # Load the model
    sac_class = VLM_SAC if use_vlm_for_reward else SAC
    algo = sac_class.load(
            path=str(checkpoint_path),
            inference_only=True,
            env=env,
            device="cuda:0",
            reward_model_config = OmegaConf.to_container(cfg.reward_model, resolve=True, throw_on_missing=True) if use_vlm_for_reward else None,
            episode_length = cfg.env.episode_length,
            render_dim = cfg.env.render_dim,
        )

    print("Loaded learner model")

    inference_video_log_dir = os.path.join(utils.get_output_path(), "video")

    os.makedirs(inference_video_log_dir, exist_ok=True)

    print(f"Generating dataset in {inference_video_log_dir} ...")
    
    for episode_idx in range(cfg.n_rollouts):
        if cfg.n_rollouts == 1:
            # if only 1 rollout, save by video name alone
            video_file_name = f"{cfg.video_base_name}.mp4"
        else:
            video_file_name = f"{cfg.video_base_name}_{checkpoint_prefix}_{episode_idx}.mp4"
            
        gif_file_name = video_file_name[:-4] + ".gif"

        video_path = os.path.join(inference_video_log_dir, video_file_name)
        video_writer = imageio.get_writer(video_path, fps=30)

        video_with_info_file_name = f"with-info_{video_file_name}"
        video_with_info_writer = imageio.get_writer(os.path.join(inference_video_log_dir, video_with_info_file_name), 
                                                        fps=30)

        obs = env.reset(seed=cfg.seed + episode_idx)[0]
        gt_rewards = []
        images = []
        pil_images = []
        for step_idx in range(cfg.env.episode_length):
            action = algo.predict(obs)[0]
            obs, _, _, _, info = env.step(action)

            image = env.render()
            images.append(torch.as_tensor(image.copy()).permute(2,0,1).float() / 255)
            image_int = np.uint8(image)
 
            pil_image = Image.fromarray(image_int)
            pil_images.append(pil_image)
            gt_rewards.append(float(info['r']))

            # Plot the ground truth reward
            plot_info_on_frame(pil_image, info)
            
            video_writer.append_data(image_int)
            video_with_info_writer.append_data(np.uint8(pil_image))

        imageio.mimsave(os.path.join(inference_video_log_dir, gif_file_name), pil_images, duration=1/30, loop=0)
        video_writer.close()
        video_with_info_writer.close()

        env.close()      

        print(f"Rollout {episode_idx+1} saved at {video_file_name}.")

        if use_vlm_for_reward:
            reward_model = load_reward_model(rank, 
                                        worker_actual_batch_size=cfg.reward_model.reward_batch_size,  # Note that this is different size compared to rank 0's reward model when rank0_batch_size_pct < 1.0
                                        model_name=cfg.reward_model.vlm_model, 
                                        model_config_dict=OmegaConf.to_container(cfg.reward_model, resolve=True, throw_on_missing=True)).eval().cuda(rank)                                 

            frames = torch.stack(images)
            dino_rewards = compute_rewards(
                model=reward_model,
                frames=frames,
                rank0_batch_size_pct=1,
                batch_size=cfg.reward_model.reward_batch_size,  # This is the total batch size
                num_workers=1,
                dist=False
                )
            smoothed_dino_rewards = half_gaussian_filter_1d(dino_rewards, sigma=4, smooth_last_N=True) 
            # all_rewards=np.stack((gt_rewards, smoothed_dino_rewards))

            rewards_matrix_heatmap(np.array(gt_rewards)[None], os.path.join(inference_video_log_dir,f'gt_heatmap_{step_idx}.png'))
            rewards_matrix_heatmap(smoothed_dino_rewards[None], os.path.join(inference_video_log_dir,f'dino_heatmap_{step_idx}.png'))


if __name__ == "__main__":
    generate_dataset()
