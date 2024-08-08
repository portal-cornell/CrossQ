from sbx.vlm_reward.reward_models.language_irl.dino_reward_model import metric_factory
from sbx.vlm_reward.reward_models.language_irl.siglip_reward_model import SigLIPFeatureExtractor
from sbx.vlm_reward.reward_models.language_irl.human_seg import HumanSegmentationModel
from sbx.vlm_reward.reward_models.dino_models import DINORewardModelWrapper

from loguru import logger

import torch

def load_siglip_reward_model(
    rank, batch_size, model_name, image_metric, 
    human_seg_model_path, pos_image_path_list, 
    neg_image_path_list,source_mask_thresh, target_mask_thresh,
    baseline_image_path, baseline_mask_thresh
):  
    feature_extractor = SigLIPFeatureExtractor(model_name=model_name, edge_size=224)
    logger.debug("Initialized feature extractor")

    human_seg_model = HumanSegmentationModel(rank, human_seg_model_path)
    logger.debug("Intialized human seg model")

    # Anne: using DINO one just because it's not particular to DINO
    # TODO: factorize
    siglip_wrapper = DINORewardModelWrapper(
                            rank=rank,
                            batch_size=batch_size,
                            dino_metric_model = metric_factory(image_metric=image_metric,
                                                                feature_extractor=feature_extractor,
                                                                patch_masker=human_seg_model),
                            pos_image_path_list = pos_image_path_list, 
                            neg_image_path_list=neg_image_path_list,
                            source_mask_thresh = source_mask_thresh,
                            target_mask_thresh=target_mask_thresh,
                            baseline_image_path=baseline_image_path,
                            baseline_mask_thresh=baseline_mask_thresh
    ).to(f"cuda:{rank}")
    siglip_wrapper.initialize_target_images()
    siglip_wrapper.initialize_baseline_images()
   
    logger.debug(f"Initialized siglip wrapper: allocated={round(torch.cuda.memory_allocated(0)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(0)/1024**3,1)}")

    return siglip_wrapper