from typing import List, Optional, Tuple, overload

from loguru import logger

from vlm_reward.eval.dino import load_dino_reward_model
from vlm_reward.eval.model_interface import RewardModel

def load_reward_model(
                    rank: int,
                    worker_actual_batch_size: int,
                    model_name: str, 
                    model_config_dict) -> RewardModel:
    assert any([model_base_name in model_name.lower() for model_base_name in ["vit", "dino"]])

    print(model_config_dict)

    if "dino" in model_name.lower():
        reward_model = load_dino_reward_model(rank=rank,
                                        batch_size=worker_actual_batch_size,
                                        model_name=model_name,
                                        image_metric="wasserstein",
                                        image_size=model_config_dict["image_size"],
                                        human_seg_model_path=model_config_dict["human_seg_model_path"],
                                        source_mask_thresh=model_config_dict["source_mask_thresh"],
                                        target_mask_thresh=model_config_dict["target_mask_thresh"])

        logger.debug(f"Loaded DINO reward model. model_name={model_name}, pos_image={model_config_dict['pos_image_path']}, neg_image={model_config_dict.get('neg_image_path', [])}")
    
    return reward_model