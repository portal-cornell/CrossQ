from sbx.vlm_reward.reward_models.language_irl.dino_reward_model import Dino2FeatureExtractor, metric_factory
from sbx.vlm_reward.reward_models.language_irl.human_seg import HumanSegmentationModel

from loguru import logger

import torch

def load_dino_reward_model(
    rank, batch_size, model_name, image_metric, human_seg_model_path, ref_image_path_list
):  
    feature_extractor = Dino2FeatureExtractor(model_name=model_name)
    logger.debug("Initialized feature extractor")

    human_seg_model = HumanSegmentationModel(rank, human_seg_model_path)
    logger.debug("Intialized human seg model")

    dino_wrapper = DINORewardModelWrapper(
                            rank=rank,
                            batch_size=batch_size,
                            dino_metric_model = metric_factory(image_metric=image_metric,
                                                                feature_extractor=feature_extractor,
                                                                patch_masker=human_seg_model),
                            ref_image_path_list = ref_image_path_list
    ).to(f"cuda:{rank}")
    dino_wrapper.initialize_ref_images()
   
    logger.debug(f"Initialized dino wrapper: allocated={round(torch.cuda.memory_allocated(0)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(0)/1024**3,1)}")

    return dino_wrapper


class DINORewardModelWrapper:
    def __init__(self, rank, batch_size, dino_metric_model, ref_image_path_list):
        self.reward_model = dino_metric_model
        self._device = f"cuda:{rank}"
        self.batch_size = batch_size
        self.ref_image_path_list = ref_image_path_list
        
    def embed_module(self, image_batch):
        logger.debug(f"[{self._device}] embed_module. {image_batch.device} {image_batch.size()=}, allocated={round(torch.cuda.memory_allocated(0)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(0)/1024**3,1)}")

        with torch.no_grad():
            if image_batch.shape[1] != 3:
                image_batch = image_batch.permute(0, 3, 1, 2)
            transformed_image = self.reward_model.feature_extractor.transform(image_batch)

            logger.debug(f"[{self._device}] transformed image. {transformed_image.size()=}, allocated={round(torch.cuda.memory_allocated(0)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(0)/1024**3,1)}")

            image_embeddings, image_masks = self.reward_model.extract_masked_features(batch=transformed_image)

            logger.debug(f"[{self._device}] {image_embeddings.size()=}, {image_masks.size()=} allocated={round(torch.cuda.memory_allocated(0)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(0)/1024**3,1)}")

            return image_embeddings, image_masks

    def __call__(self, embedding_tuple):
        with torch.no_grad():
            embeddings, embeddings_masks = embedding_tuple
            distance = torch.Tensor(self.reward_model.compute_distance_parallel(source_features=embeddings,
                                                                    source_masks=embeddings_masks,
                                                                    target_features=self.ref_image_embeddings,
                                                                    target_masks=self.ref_image_masks)).to(self._device)
            logger.debug(f"__call__: {distance.size()=}")
            return - distance  # Reward is the negative of the distance

    def initialize_ref_images(self):
        # TODO: For now, we just support loading one image
        logger.debug("Embedding reference image")
        with torch.no_grad():
            self.ref_image_embeddings = self.reward_model.feature_extractor.load_and_prepare_images_parallel(self.ref_image_path_list)
            self.ref_image_embeddings, self.ref_image_masks = self.reward_model.extract_masked_features(batch=self.ref_image_embeddings)
            self.ref_image_embeddings = self.ref_image_embeddings.repeat(self.batch_size, 1, 1)
            self.ref_image_masks = self.ref_image_masks.repeat(self.batch_size, 1)
            logger.debug(f"{self.ref_image_embeddings.size()=}, {self.ref_image_masks.size()=},allocated={round(torch.cuda.memory_allocated(0)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(0)/1024**3,1)} ")

    def eval(self):
        """A placeholder for the reward model wrapper. DINO should not needed to be trained
        """
        return self

    def to(self, device):
        if type(device) == str:
            rank = int(str(device)[-1])
        else:
            rank = device.index

        self.reward_model = self.reward_model.to(device)

        self.reward_model.device = device
        logger.debug(f"Change reward model to {device}: allocated={round(torch.cuda.memory_allocated(rank)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(rank)/1024**3,1)}")
        
        self.reward_model.feature_extractor.model = self.reward_model.feature_extractor.model.to(device)
        self.reward_model.feature_extractor.device = device

        logger.debug(f"Change feature_extractor to {device}: allocated={round(torch.cuda.memory_allocated(rank)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(rank)/1024**3,1)}")

        self.reward_model.patch_masker = self.reward_model.patch_masker.to(device)
        self.reward_model.patch_masker.model = self.reward_model.patch_masker.model.to(device)
        self.reward_model.patch_masker.device = device

        logger.debug(f"Change patch_masker to {device}: allocated={round(torch.cuda.memory_allocated(rank)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(rank)/1024**3,1)}")

        self._device = device

        return self

    def cuda(self, rank):
        device = f"cuda:{rank}"

        return self.to(device)