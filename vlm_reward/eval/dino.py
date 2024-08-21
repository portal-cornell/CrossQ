from vlm_reward.utils.dino_reward_model import Dino2FeatureExtractor, metric_factory
from vlm_reward.utils.human_seg import HumanSegmentationModel
from vlm_reward.eval.model_interface import RewardModel

from loguru import logger

import torch

from jaxtyping import Float
from typing import Tuple, NewType, Any


def load_dino_reward_model(
    rank: int, batch_size: int, model_name: str, image_metric, image_size: int,
    human_seg_model_path: str, source_mask_thresh: str, target_mask_thresh: str,
    ):  
    feature_extractor = Dino2FeatureExtractor(model_name=model_name, edge_size=image_size)
    logger.debug("Initialized feature extractor")

    human_seg_model = HumanSegmentationModel(rank, human_seg_model_path)
    logger.debug("Intialized human seg model")

    dino_metric_model = metric_factory(image_metric=image_metric,
                                        feature_extractor=feature_extractor,
                                        patch_masker=human_seg_model)

    dino_interface = DinoRewardModel(dino_metric_model=dino_metric_model,
                                    rank=rank,
                                    batch_size=batch_size,
                                    source_mask_thresh=source_mask_thresh,
                                    target_mask_thresh=target_mask_thresh)    

    return dino_interface.cuda(rank) # eval model should always be on gpu                        

class DinoRewardModel(RewardModel):
    """
    Barebones model conforming to the reward model interface indicated in model_interface.py
    Currently supports just 1 target image
    """
    def __init__(self, dino_metric_model, rank, batch_size, 
            source_mask_thresh, target_mask_thresh):
        self.reward_model = dino_metric_model
        self.device = f"cuda:{rank}"
        self.batch_size = batch_size

        self.source_mask_thresh = source_mask_thresh
        self.target_mask_thresh = target_mask_thresh    

    def set_target_embedding(self, target_image: Float[torch.Tensor, "c h w"]) -> None:
        """
        Cache an embedding of the target image
        
        self.target_mask_thresh is masking the threshold to apply to target images
        """
        with torch.no_grad():
            # self.reward_model methods are set up to take in batches, so batch and debatch the input/output
            target_image_prepared = self.reward_model.feature_extractor.prepare_images_parallel(target_image[None])
            target_embeddings, target_masks = self.reward_model.extract_masked_features(batch=target_image_prepared, use_patch_mask=True, mask_thresh=self.target_mask_thresh)

            self.target_embedding = target_embeddings[0]
            self.target_mask = target_masks[0]

    def set_source_embeddings(self, source_images: Float[torch.Tensor, "b c h w"]) -> None:
        """
        Cache an embedding of the source image
        
        self.target_mask_thresh is masking the threshold to apply to source images
        """
        with torch.no_grad():
            source_images_prepared = self.reward_model.feature_extractor.prepare_images_parallel(source_images)
            source_embeddings, source_masks = self.reward_model.extract_masked_features(batch=source_images_prepared, use_patch_mask= self.source_mask_thresh!=0, mask_thresh=self.source_mask_thresh)

            self.source_embeddings = source_embeddings
            self.source_masks = source_masks

    def predict(self) -> Float[torch.Tensor, 'rews']:
        all_ds = []
        target_embeddings_repeat = target_image_embeddings.repeat(self.batch_size, 1, 1, 1).permute(1,0,2,3)
        target_masks_repeat = target_image_masks.repeat(self.batch_size, 1, 1).permute(1,0,2)

        for source_embedding, source_mask in zip(self.source_embeddings, self.source_masks):
            with torch.no_grad():

                distance = torch.Tensor(self.reward_model.compute_distance_parallel(source_features=source_embedding,
                                                                        source_masks=source_mask,
                                                                        target_features=target_embeddings_repeat,
                                                                        target_masks=target_masks_repeat
                                                                        )).to(self.device)
                
                distance = torch.Tensor(raw_output_dict["wasser"]).to(self._device)
                
                all_ds.append(distance)
                logger.debug(f"__call__: {distance.size()=}")

        all_ds = torch.stack(all_ds)

        return - all_ds  # Reward is the negative of the distance

    def initialize_baseline_images(self):
        if self.baseline_image_path is not None:
            with torch.no_grad():
                baseline_embedding = self.reward_model.feature_extractor.load_and_prepare_images_parallel([self.baseline_image_path])
                baseline_embedding, baseline_mask = self.reward_model.extract_masked_features(batch=baseline_embedding, use_patch_mask=True, mask_thresh=self.baseline_mask_thresh)

                self.reward_model.set_baseline_projection(self.target_image_embeddings[0,0,...].cpu(), self.target_image_masks[0,0,...].cpu(), baseline_embedding.cpu(), baseline_mask.cpu())
                

    def eval(self):
        """A placeholder for the reward model wrapper. DINO should not needed to be trained
        """
        return self

    def to(self, device):
        # TODO (yuki): This is not elegant all
        if type(device) == str:
            rank = int(str(device)[-1])
        else:
            rank = device.index

        self.reward_model = self.reward_model.to(device)
        self.reward_model.device = device

        
        self.reward_model.feature_extractor.model = self.reward_model.feature_extractor.model.to(device)
        self.reward_model.feature_extractor.device = device

        self.reward_model.patch_masker = self.reward_model.patch_masker.to(device)
        self.reward_model.patch_masker.model = self.reward_model.patch_masker.model.to(device)
        self.reward_model.patch_masker.device = device

        self.device = device

        return self

    def cuda(self, rank):
        device = f"cuda:{rank}"
        return self.to(device)
