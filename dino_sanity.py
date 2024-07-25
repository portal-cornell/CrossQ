"""
Sanity checks for dino inference (plot heatmaps)
"""
import torch
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
import yaml
import os
import numpy as np
from einops import rearrange

from sbx.vlm_reward.reward_main import compute_rewards, load_reward_model
from sbx.vlm_reward.reward_transforms import half_gaussian_filter_1d

from sbx.vlm_reward.reward_models.language_irl.utils import rewards_matrix_heatmap, rewards_line_plot, pad_to_longest_sequence

from loguru import logger

all_source_thresh = [0, .001]

def load_frames_to_torch(gif_path):
    gif_obj = Image.open(gif_path)
    frames = [gif_obj.seek(frame_index) or gif_obj.convert("RGB") for frame_index in range(gif_obj.n_frames)]
    torch_frames = torch.stack([pil_to_tensor(frame) for frame in frames])

    # many of the gifs have the last frame standing, which will mess with smoothing
    torch_frames = torch_frames[:-1] # chop off last one 
    return  torch_frames 

def rewards_from_gifs(gif_paths, reward_config_dict, reward_model_name, batch_size, sigma, transform):
    """
    requires: all gifs must be the same length (have same # of frames)
    """
    
    all_frames = []
    for gif_path in gif_paths:
        frames = load_frames_to_torch(gif_path)
        frames = frames.permute(0, 2, 3, 1) # B, 3, H, W -> B, H, W, 3
        all_frames.append(frames)    

    logger.info("Loading reward model...")
    reward_model = load_reward_model(rank=0, worker_actual_batch_size=batch_size,
                                        model_name=reward_model_name,
                                        model_config_dict=reward_config_dict).eval().cuda(0)
    logger.info("Finished loading reward model...")

    all_rewards = []

    for frames in all_frames:
        rewards = compute_rewards(
            model=reward_model,
            frames=frames,
            rank0_batch_size_pct=1,
            batch_size=batch_size,  # This is the total batch size
            num_workers=1,
            dist=False
            )
        
        logger.info(f"\nsmoothed_size={half_gaussian_filter_1d(rewards, sigma=0.4, smooth_last_N=True).shape}, rewards_size={rewards.cpu().numpy().shape}")
        if sigma:
            smoothed_rewards = half_gaussian_filter_1d(rewards, sigma=sigma, smooth_last_N=True) 
        else:
            smoothed_rewards = rewards.cpu().numpy()
        smoothed_transformed_rewards = transform(smoothed_rewards)

        all_rewards.append(smoothed_transformed_rewards)
    
    all_rewards = pad_to_longest_sequence(all_rewards)
    all_rewards = np.stack(all_rewards)

    n_gifs = len(gif_paths)


    labels = [gif_path.split('/')[-1].split('.')[0] for gif_path in gif_paths]


    return all_rewards, labels



if __name__=="__main__":
    gif_paths =  [
                'axis_exp/kneeling_gifs/0_success_crossq_kneel.gif',
                'axis_exp/kneeling_gifs/1_kneel-at-20_fall-backward.gif',
                'axis_exp/kneeling_gifs/2_some-move-close-to-kneeling.gif',
                'axis_exp/kneeling_gifs/3_crossq_stand_never-on-ground.gif'
               ]

    base_save_path = 'axis_exp/threshold/outputs_cos_l'

    # TODO: Support loading CLIP as a feature encoder
    reward_model_name = 'dinov2_vitl14_reg' # Based on here, DINOv2 large is good enough https://docs.google.com/document/d/14BrYHRFW4cVW2FSIi3-Js2o4rHiPFJSR0TLpUw2Sn2k/edit#bookmark=kix.ogvltufldfdu
    # TODO: Support using Pooled feature
    use_patch = True
    human_demo = True
    batch_size = 32
    # Based on comments by Will
    #   "Experiment run with best settings so far (source thresh = .001, cosine distance function, wasserstein, anne_kneeling_front, no scaling, sigma=4)"
    """
    image_metric: 'wasserstein'
    human_seg_model_path: '/share/portal/hw575/language_irl/pretrained_checkpoints/SGHM-ResNet50.pth'
    pos_image_path: 
    - sbx/vlm_reward/reward_models/language_irl/preference_data/humans_selected_images/kneeling/anne_kneeling_front_final.png
    source_mask_thresh: 0.001
    target_mask_thresh: .5
    """
    sigma = 0
    if human_demo:
        reward_config = 'configs/dino_kneeling_config.yml'
    else:
        reward_config = 'configs/dino_kneeling_config_robot_demo.yml'

    def transform_linear(rew):
        return 4.8541 * rew + 159.2704
    
    def identity(rew):
        return rew

    with open(reward_config, "r") as fin:
        reward_config_dict = yaml.safe_load(fin)

    rewards, all_labels = rewards_from_gifs(gif_paths, 
                                    reward_config_dict=reward_config_dict, 
                                    reward_model_name=reward_model_name, 
                                    batch_size=batch_size, 
                                    sigma=sigma, 
                                    transform=identity)

    heatmap_name = f"rm={reward_model_name}_patch={use_patch}_h-demo={human_demo}.png"

    rewards_matrix_heatmap(rewards, f'axis_exp/heatmaps/{heatmap_name}')
