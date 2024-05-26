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

from sbx.vlm_reward.reward_models.language_irl.utils import rewards_matrix_heatmap, rewards_line_plot

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
    all_frames = torch.stack(all_frames)
    frames = rearrange(all_frames, "n_gifs n_steps ... -> (n_steps n_gifs) ...")
    

    
    reward_model = load_reward_model(rank=0, worker_actual_batch_size=batch_size,
                                        model_name=reward_model_name,
                                        model_config_dict=reward_config_dict).eval().cuda(0)

    all_rews = []
    all_labels=[]
    for thresh in all_source_thresh:
        reward_model.source_mask_thresh = thresh

        rewards = compute_rewards(
            model=reward_model,
            frames=frames,
            rank0_batch_size_pct=1,
            batch_size=batch_size,  # This is the total batch size
            num_workers=1,
            dist=False
            )

        n_gifs = len(gif_paths)
        rewards = rearrange(
            rewards,
            "(n_steps n_gifs) ... -> n_gifs n_steps ...",
            n_gifs=n_gifs,
        )  

        smoothed_rewards = half_gaussian_filter_1d(rewards, sigma=sigma, smooth_last_N=True) 

        smoothed_transformed_rewards = transform(smoothed_rewards)

        all_rews.append(smoothed_transformed_rewards)
        labels = [gif_path.split('/')[-1].split('.')[0] for gif_path in gif_paths]

        all_labels.append(labels)

    return np.array(all_rews), all_labels




if __name__=="__main__":
#    gif_paths = ['sbx/vlm_reward/reward_models/language_irl/kneeling_gifs_ranked/kneeling_5.gif']

    gif_paths =  ['debugging/gifs/kneeling_gifs/kneeling_almost.gif',
                    'debugging/gifs/kneeling_gifs/kneeling_bad.gif',
                    'debugging/gifs/kneeling_gifs/decent_kneeling.gif',
                'debugging/gifs/kneeling_gifs/kneel_adversary.gif',
                'debugging/gifs/kneeling_gifs/leaning_forward.gif',
                'debugging/gifs/kneeling_gifs/one_leg.gif',
                'debugging/gifs/standing_gifs/crossq_stand.gif',
                'sbx/vlm_reward/reward_models/language_irl/kneeling_gifs_ranked/kneeling_5.gif'
                
               ]
    # gif_paths =  ['debugging/gifs/standing_gifs/step_291.gif', 'debugging/gifs/standing_gifs/step_531.gif', 'debugging/gifs/standing_gifs/step_781.gif'] 
    
    #gif_paths = ['debugging/gifs/standing_gifs/crossq_stand.gif']
    base_save_path = 'debugging/threshold/outputs_cos_l'
    
    reward_config = 'configs/dino_kneeling_config.yml'
    reward_model_name = 'dinov2_vitl14_reg' # TODO: change to L
    batch_size=32
    sigma = 4

    def transform_linear(rew):
        return 4.8541 * rew + 159.2704
    
    def identity(rew):
        return rew

    with open(reward_config, "r") as fin:
        reward_config_dict = yaml.safe_load(fin)


    for gif_path in gif_paths:
        rewards, all_labels = rewards_from_gifs([gif_path], 
                                    reward_config_dict=reward_config_dict, 
                                    reward_model_name=reward_model_name, 
                                    batch_size=batch_size, 
                                    sigma=sigma, 
                                    transform=identity)


        for i, rew in enumerate(rewards):
            colors = ['blue', 'green']
            fname = f"gif={gif_path.split('/')[-1]}_t={all_source_thresh[i]}"
            fp = os.path.join(base_save_path, fname)
            rewards_line_plot(rew, labels = [f"t={all_source_thresh[i]}"], fp=fp, c=colors[i % len(colors)])


     #rewards_matrix_heatmap(np.array(rewards), os.path.join(save_base, 'heatmap'))
    #rewards_matrix_heatmap(np.array(smoothed_rewards), os.path.join(save_base, 'heatmap_smooth'))
