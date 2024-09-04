from vlm_reward.utils.dino_reward_model import Dino2FeatureExtractor, metric_factory
from vlm_reward.utils.human_seg import HumanSegmentationModel
from vlm_reward.reward_models.model_interface import RewardModel

from loguru import logger

import torch

from jaxtyping import Float
from typing import Tuple, NewType, Any


def load_dino_wasserstein_reward_model(rank: int, batch_size: int, model_name: str, image_size: int,
    human_seg_model_path: str, source_mask_thresh: str, target_mask_thresh: str,
    ):  
    feature_extractor = Dino2FeatureExtractor(model_name=model_name, edge_size=image_size)
    logger.debug("Initialized feature extractor")

    human_seg_model = HumanSegmentationModel(rank, human_seg_model_path)
    logger.debug("Intialized human seg model")

    dino_metric_model = metric_factory(image_metric="wasserstein",
                                        feature_extractor=feature_extractor,
                                        patch_masker=human_seg_model)

    dino_interface = DinoWassersteinRewardModel(dino_metric_model=dino_metric_model,
                                    rank=rank,
                                    batch_size=batch_size,
                                    source_mask_thresh=source_mask_thresh,
                                    target_mask_thresh=target_mask_thresh)    

    return dino_interface # eval model should always be on gpu                        

def load_dino_pooled_reward_model(rank: int, batch_size: int, model_name: str, image_size: int, human_seg_model_path: str):  
    feature_extractor = Dino2FeatureExtractor(model_name=model_name, edge_size=image_size)
    logger.debug("Initialized feature extractor")

    # NOTE: not actually used for this model (only in here for compatibility with dino interface)
    human_seg_model = HumanSegmentationModel(rank, human_seg_model_path)

    dino_metric_model = metric_factory(image_metric="feature",
                                        feature_extractor=feature_extractor,
                                        patch_masker=human_seg_model)

    dino_interface = DinoPooledRewardModel(dino_metric_model=dino_metric_model,
                                    rank=rank,
                                    batch_size=batch_size)    

    return dino_interface # eval model should always be on gpu                        

class DinoWassersteinRewardModel(RewardModel):
    """
    Barebones model conforming to the reward model interface indicated in model_interface.py
    Currently supports just 1 target image
    """
    def __init__(self, dino_metric_model, rank, batch_size, 
            source_mask_thresh, target_mask_thresh, 
            cost_fn='cosine', return_ot_plan=False):
        self.reward_model = dino_metric_model
        self.device = f"cuda:{rank}"
        self.batch_size = batch_size

        self.source_mask_thresh = source_mask_thresh
        self.target_mask_thresh = target_mask_thresh   

        self.cost_fn = cost_fn
        self.return_ot_plan = return_ot_plan

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

        # These will store batches of embeddings and masks
        self.source_embeddings = []
        self.source_masks = [] 
        
        with torch.no_grad():
            source_images_prepared = self.reward_model.feature_extractor.prepare_images_parallel(source_images)

            source_image_batches = torch.split(source_images_prepared, self.batch_size)

            # inference on all at once will cause an out of memory error
            for batch in source_image_batches:
                batch_source_embeddings, batch_source_masks = self.reward_model.extract_masked_features(batch=batch, use_patch_mask=self.source_mask_thresh!=0, mask_thresh=self.source_mask_thresh)
                # Cache batches of embeddings
                self.source_embeddings.append(batch_source_embeddings)
                self.source_masks.append(batch_source_masks)

    def predict(self) -> Float[torch.Tensor, 'rews']:
        if not hasattr(self, 'source_embeddings') or not hasattr(self, 'source_masks'):
            raise Exception("Source not yet initialized. Try calling set_source_embeddings")
        if not hasattr(self, 'target_embedding') or not hasattr(self, 'target_mask'):
            raise Exception("Target not yet initialized. Try calling set_target_embedding")

        all_ds = []
        target_embedding_repeat = self.target_embedding.repeat(self.batch_size, 1, 1)
        target_mask_repeat = self.target_mask.repeat(self.batch_size, 1)

        for source_embedding_batch, source_mask_batch in zip(self.source_embeddings, self.source_masks):
            with torch.no_grad():
                
                # The last batch might sometimes be smaller
                source_batch_size = source_embedding_batch.shape[1]
                if source_batch_size != target_embedding_repeat.shape[1]: 
                    target_embedding_repeat =  self.target_embedding.repeat(source_batch_size, 1, 1)
                    target_mask_repeat = self.target_mask.repeat(source_batch_size, 1)
                raw_output_dict = self.reward_model.compute_distance_parallel(source_features=source_embedding_batch,
                                                                        source_masks=source_mask_batch,
                                                                        target_features=target_embedding_repeat,
                                                                        target_masks=target_mask_repeat,
                                                                        cost_fn=self.cost_fn,
                                                                        return_ot_plan=self.return_ot_plan
                                                                        )
                
                distance = torch.Tensor(raw_output_dict["wasser"]).to(self.device)
                all_ds.append(distance)

                if self.return_ot_plan:
                    self.saved_ot_plan.extend(raw_output_dict.get("T", None))
                    self.saved_costs.extend(raw_output_dict.get("C", None))
                

        all_ds = torch.cat(all_ds)

        return - all_ds  # Reward is the negative of the distance     

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

class DinoPooledRewardModel(RewardModel):
    """
    Barebones model conforming to the reward model interface indicated in model_interface.py
    Currently supports just 1 target image
    """
    def __init__(self, dino_metric_model, rank, batch_size):
        self.reward_model = dino_metric_model
        self.device = f"cuda:{rank}"
        self.batch_size = batch_size

    def set_target_embedding(self, target_image: Float[torch.Tensor, "c h w"]) -> None:
        """
        Cache an embedding of the target image
        
        self.target_mask_thresh is masking the threshold to apply to target images
        """
        with torch.no_grad():
            # self.reward_model methods are set up to take in batches, so batch and debatch the input/output
            target_image_prepared = self.reward_model.feature_extractor.prepare_images_parallel(target_image[None])
            target_embeddings = self.reward_model.extract_features_final(batch=target_image_prepared)

            self.target_embedding = target_embeddings[0]

    def set_source_embeddings(self, source_images: Float[torch.Tensor, "b c h w"]) -> None:
        """
        Cache an embedding of the source image
        
        self.target_mask_thresh is masking the threshold to apply to source images
        """

        # These will store batches of embeddings
        self.source_embeddings = []
        
        with torch.no_grad():
            source_images_prepared = self.reward_model.feature_extractor.prepare_images_parallel(source_images)

            source_image_batches = torch.split(source_images_prepared, self.batch_size)

            # inference on all at once will cause an out of memory error
            for batch in source_image_batches:
                batch_source_embeddings = self.reward_model.extract_features_final(batch=batch)
                # Cache batches of embeddings
                self.source_embeddings.append(batch_source_embeddings)
        self.source_embeddings = torch.cat(self.source_embeddings)

    def predict(self) -> Float[torch.Tensor, 'rews']:
        """
        Compute the cosine similarity between the target and each source embedding
        """
        if not hasattr(self, 'source_embeddings'):
            raise Exception("Source not yet initialized. Try calling set_source_embeddings")
        if not hasattr(self, 'target_embedding'):
            raise Exception("Target not yet initialized. Try calling set_target_embedding")

        # cosine similarity
        with torch.no_grad():
            target_norm = torch.norm(self.target_embedding, p=2)  # ||x||
            source_norm = torch.norm(self.source_embeddings, p=2, dim=1)  # ||y_i|| for each i
            
            dot_product = torch.matmul(self.source_embeddings, self.target_embedding)  # y_i ⋅ x for each i
            
            cosine_similarity = dot_product / (target_norm * source_norm)
                            
        return cosine_similarity 

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

        self.device = device

        return self

    def cuda(self, rank):
        device = f"cuda:{rank}"
        return self.to(device)
