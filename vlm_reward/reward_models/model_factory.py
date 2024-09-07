from typing import List, Optional, Tuple, overload, Union
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

from loguru import logger

from vlm_reward.reward_models.model_interface import RewardModel
from vlm_reward.reward_models.dino_refactor import load_dino_wasserstein_reward_model, load_dino_pooled_reward_model
# TODO: Temporarily commented out to avoid long import times
# from vlm_reward.reward_models.lpips import load_lpips_reward_model
from vlm_reward.reward_models.dreamsim import load_dreamsim_reward_model
# from vlm_reward.reward_models.sam2 import load_sam2_reward_model

def load_reward_model(
                    rank: int,
                    worker_actual_batch_size: int,
                    model_name: str, 
                    model_config_dict) -> RewardModel:
    """
    Load a reward model of type model_interface.RewardModel, which is designed for a single goal image

    Since we currently consider the goal image as part of a reward model closure, this will also load the 
    goal image defined by model_config_dict['pos_image_path'][0]
    """
    logger.info(model_config_dict)

    if "dino" in model_name.lower():
        if "wasser" in model_name.lower():
            reward_model = load_dino_wasserstein_reward_model(
                                            rank = rank,
                                            batch_size=worker_actual_batch_size,
                                            model_name=model_config_dict["vlm_model"],
                                            image_size=model_config_dict["image_size"],
                                            human_seg_model_path=model_config_dict["human_seg_model_path"],
                                            source_mask_thresh=model_config_dict["source_mask_thresh"],
                                            target_mask_thresh=model_config_dict["target_mask_thresh"])
            reward_model.cuda(rank)
            logger.debug(f"Loaded DINO wasserstein reward model. model_name={model_config_dict['vlm_model']}, pos_image={model_config_dict['pos_image_path']}, neg_image={model_config_dict.get('neg_image_path', [])}")
        elif "pooled" in model_name.lower():
            reward_model = load_dino_pooled_reward_model(
                                            rank = rank,
                                            batch_size=worker_actual_batch_size,
                                            model_name=model_config_dict["vlm_model"],
                                            image_size=model_config_dict["image_size"],
                                            human_seg_model_path=model_config_dict["human_seg_model_path"])
            reward_model.cuda(rank)
            logger.debug(f"Loaded DINO pooled reward model. model_name={model_config_dict['vlm_model']}, pos_image={model_config_dict['pos_image_path']}, neg_image={model_config_dict.get('neg_image_path', [])}")
        else:
            exception = f"Illegal dino model type {model_name}. Try changing name in your config."
            raise Exception(exception)
    elif "lpips" in model_name.lower():
        reward_model = load_lpips_reward_model()
        reward_model.cuda(rank)
        logger.debug(f"Loaded lpips reward model")
    elif "dreamsim" in model_name.lower():
        reward_model = load_dreamsim_reward_model()
        reward_model.cuda(rank)
        logger.debug(f"Loaded dreamsim reward model")
    elif "sam2" in model_name.lower():
        reward_model = load_sam2_reward_model(
                                    rank, 
                                    sam2_model_id=model_config_dict['sam2_model_id'],
                                    sam2_cfg_path=model_config_dict['sam2_cfg_path'],
                                    batch_size=worker_actual_batch_size)
    elif "clip" in model_name.lower():
        raise Exception("Error: CLIP RewardModel interface not yet implemented")
    else:
        exception = f"Illegal model name {model_name}. Try changing name in your config."
        raise Exception(exception)

    # # Load the target image
    # # All of the currently used models have this transform as their preprocessing step 
    image_transform = lambda image: F.interpolate(image, size=(224,224), mode="bilinear").to(reward_model.device)
    target_image_path = model_config_dict["pos_image_path"][0]
    target_image_raw = load_image_from_path(target_image_path, output_type="torch")
    target_image_tensor = image_transform(target_image_raw[None])[0] # image transform takes in batched input
    reward_model.set_target_embedding(target_image_tensor)
    logger.debug(f"Loaded target image from path: {target_image_path}")

    return reward_model

def load_image_from_path(path: str, output_type="torch") -> Union[Image, torch.Tensor]:
    """
    output_type is either "torch" or "pil"

    Load the image at the path into a torch tensor with shape (channel, height, width)
    """
    frame = Image.open(path).convert("RGB")
    if output_type == "pil":
        return frame

    frame_torch = torch.tensor(np.array(frame)).permute(2,0,1)
    return frame_torch

def load_images_from_paths(paths: List[str], output_type="torch")-> Union[List[Image], torch.Tensor]:
    """
    output_type is either "torch" or "pil". If "pil", frames will be loaded to a list of PIL.Image

    Load a batch of images at the given paths into a torch tensor with shape (batch, channel, height, width)
    """
    if output_type == "pil":
        return [load_image_from_path(path, output_type="pil") for path in paths]
    
    return torch.stack([load_image_from_path(path, output_type="torch") for path in paths])
