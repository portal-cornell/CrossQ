from typing import List, Optional, Tuple, overload

from loguru import logger

from vlm_reward.reward_models.model_interface import RewardModel

from vlm_reward.reward_models.dino_refactor import load_dino_wasserstein_reward_model, load_dino_pooled_reward_model
from vlm_reward.reward_models.lpips import load_lpips_reward_model
from vlm_reward.reward_models.dreamsim import load_dreamsim_reward_model

def load_reward_model(
                    rank: int,
                    worker_actual_batch_size: int,
                    model_name: str, 
                    model_config_dict) -> RewardModel:
    assert any([model_base_name in model_name.lower() for model_base_name in ["vit", "dino", "lpips", "dreamsim"]])

    print(model_config_dict)

    if "dino" in model_name.lower():
        if "wasser" in model_name.lower():
            reward_model = load_dino_wasserstein_reward_model(rank=rank,
                                            batch_size=worker_actual_batch_size,
                                            model_name=model_config_dict["vlm_model"],
                                            image_size=model_config_dict["image_size"],
                                            human_seg_model_path=model_config_dict["human_seg_model_path"],
                                            source_mask_thresh=model_config_dict["source_mask_thresh"],
                                            target_mask_thresh=model_config_dict["target_mask_thresh"])

            logger.debug(f"Loaded DINO wasserstein reward model. model_name={model_config_dict['vlm_model']}, pos_image={model_config_dict['pos_image_path']}, neg_image={model_config_dict.get('neg_image_path', [])}")
        elif "pooled" in model_name.lower():
            reward_model = load_dino_pooled_reward_model(rank=rank,
                                            batch_size=worker_actual_batch_size,
                                            model_name=model_config_dict["vlm_model"],
                                            image_size=model_config_dict["image_size"],
                                            human_seg_model_path=model_config_dict["human_seg_model_path"])

            logger.debug(f"Loaded DINO ppoled reward model. model_name={model_config_dict['vlm_model']}, pos_image={model_config_dict['pos_image_path']}, neg_image={model_config_dict.get('neg_image_path', [])}")
        else:
            raise Exception("Illegal dino model type. Try changing name in your config.")
    elif "lpips" in model_name.lower():
        reward_model = load_lpips_reward_model()
        logger.debug(f"Loaded lpips reward model")
    elif "dreamsim" in model_name.lower():
        reward_model = load_dreamsim_reward_model()
        logger.debug(f"Loaded dreamsim reward model")
    else:
        raise Exception("Illegal model name. Try changing name in your config.")

    return reward_model