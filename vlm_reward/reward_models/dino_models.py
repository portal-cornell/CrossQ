from vlm_reward.utils.dino_reward_model import Dino2FeatureExtractor, metric_factory
from vlm_reward.utils.human_seg import HumanSegmentationModel

from loguru import logger

import torch

def load_dino_reward_model(
    rank, batch_size, model_name, image_metric, image_size,
    human_seg_model_path, pos_image_path_list, 
    neg_image_path_list,source_mask_thresh, target_mask_thresh,
    baseline_image_path, baseline_mask_thresh, cost_fn, return_ot_plan
):  
    feature_extractor = Dino2FeatureExtractor(model_name=model_name, edge_size=image_size)
    logger.debug("Initialized feature extractor")

    human_seg_model = HumanSegmentationModel(rank, human_seg_model_path)
    logger.debug("Intialized human seg model")

    dino_wrapper = DINORewardModelWrapper(
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
                            baseline_mask_thresh=baseline_mask_thresh,
                            cost_fn=cost_fn,
                            return_ot_plan=return_ot_plan
    ).to(f"cuda:{rank}")
    dino_wrapper.initialize_target_images()
    dino_wrapper.initialize_baseline_images()
   
    logger.debug(f"Initialized dino wrapper: allocated={round(torch.cuda.memory_allocated(0)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(0)/1024**3,1)}")

    return dino_wrapper


class DINORewardModelWrapper:
    def __init__(self, rank, batch_size, dino_metric_model, 
            pos_image_path_list, neg_image_path_list,
            source_mask_thresh, target_mask_thresh, baseline_mask_thresh=.5,baseline_image_path=None, cost_fn='cosine', return_ot_plan=False):
        self.reward_model = dino_metric_model
        self._device = f"cuda:{rank}"
        self.batch_size = batch_size
        self.pos_image_path_list = pos_image_path_list
        self.neg_image_path_list = neg_image_path_list
        self.source_mask_thresh = source_mask_thresh
        self.target_mask_thresh = target_mask_thresh

        self.baseline_mask_thresh = baseline_mask_thresh
        self.baseline_image_path = baseline_image_path
        self.gb = baseline_image_path is not None

        self.cost_fn = cost_fn

        # Whether the model should store the optimal transport plan
        self.return_ot_plan = return_ot_plan
        self.saved_ot_plan = []
        self.saved_costs = []
        self.saved_mask = []
        
    def embed_module(self, image_batch, mask_thresh):
        """
        mask_thresh is used for human masking, and applied to the image batch
        """

        logger.debug(f"[{self._device}] embed_module. {image_batch.device} {image_batch.size()=}, allocated={round(torch.cuda.memory_allocated(0)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(0)/1024**3,1)}")
        with torch.no_grad():
            if image_batch.shape[1] != 3:
                image_batch = image_batch.permute(0, 3, 1, 2)
            transformed_image = self.reward_model.feature_extractor.transform(image_batch)

            logger.debug(f"[{self._device}] transformed image. {transformed_image.size()=}, allocated={round(torch.cuda.memory_allocated(0)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(0)/1024**3,1)}")

            image_embeddings, image_masks = self.reward_model.extract_masked_features(batch=transformed_image, use_patch_mask= mask_thresh!=0, mask_thresh=mask_thresh)
            self.saved_mask.append(image_masks)

            logger.debug(f"[{self._device}] {image_embeddings.size()=}, {image_masks.size()=} allocated={round(torch.cuda.memory_allocated(0)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(0)/1024**3,1)}")

            return image_embeddings, image_masks

    def __call__(self, embedding_tuple):
        all_ds = []
        with torch.no_grad():
            for i, (target, target_mask) in enumerate(zip(self.target_image_embeddings, self.target_image_masks)):
                source, source_mask = embedding_tuple

                raw_output_dict = self.reward_model.compute_distance_parallel(source_features=source,
                                                                        source_masks=source_mask,
                                                                        target_features=target,
                                                                        target_masks=target_mask,
                                                                        # gb=self.gb,  # 7/24/2024 (Yuki): Will has some extra stuff with this ("goal baseline?") but it's not on any language_irl branch
                                                                        cost_fn=self.cost_fn,
                                                                        return_ot_plan=self.return_ot_plan
                                                                        )
                
                distance = torch.Tensor(raw_output_dict["wasser"]).to(self._device)
                all_ds.append(distance)

                if self.return_ot_plan:
                    self.saved_ot_plan.extend(raw_output_dict.get("T", None))
                    self.saved_costs.extend(raw_output_dict.get("C", None))
                
                logger.debug(f"__call__: {distance.size()=}")

        all_ds = torch.stack(all_ds)

        if self.pos_idx_split < len(self.target_image_embeddings):
            total_distance = torch.sum(all_ds[:self.pos_idx_split], axis=0) / len(all_ds[:self.pos_idx_split]) - (torch.sum(all_ds[self.pos_idx_split:], axis=0) / len(all_ds[self.pos_idx_split:]))
            
        else:
            total_distance = torch.sum(all_ds[:self.pos_idx_split], axis=0) / len(all_ds[:self.pos_idx_split])
            #total_distance = total_distance - 33.5 # TODO: MAGIC NUMBER offset to near 0 (tend to be around 33.4)

        #total_distance = 500*total_distance # TODO: magic number scales to rewards for RL

        return - total_distance  # Reward is the negative of the distance

    def initialize_target_images(self):
        """
        self.target_mask_thresh is the threshold to apply to target images
        """
        # TODO: For now, we just support loading one image
        logger.debug(f"[{self._device}] Embedding human target image")

        target_image_path_list = self.pos_image_path_list + self.neg_image_path_list

        with torch.no_grad():
            target_image_embeddings = self.reward_model.feature_extractor.load_and_prepare_images_parallel(target_image_path_list)
            target_image_embeddings, target_image_masks = self.reward_model.extract_masked_features(batch=target_image_embeddings, use_patch_mask=True, mask_thresh=self.target_mask_thresh)

            self.target_image_embeddings = target_image_embeddings.repeat(self.batch_size, 1, 1, 1).permute(1,0,2,3)
            self.target_image_masks = target_image_masks.repeat(self.batch_size, 1, 1).permute(1,0,2)
            logger.debug(f"[{self._device}] {self.target_image_embeddings.size()=}, {self.target_image_masks.size()=},allocated={round(torch.cuda.memory_allocated(0)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(0)/1024**3,1)} ")
            
        self.pos_idx_split = len(self.pos_image_path_list) # index at which self.target_image_embeddings and masks become negative examples
        logger.debug(f"self.pos_idx_split={self.pos_idx_split}, len(self.target_image_embeddings)={len(self.target_image_embeddings)}")

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
        # logger.debug(f"Change reward model to {device}: allocated={round(torch.cuda.memory_allocated(rank)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(rank)/1024**3,1)}")
        
        self.reward_model.feature_extractor.model = self.reward_model.feature_extractor.model.to(device)
        self.reward_model.feature_extractor.device = device

        # logger.debug(f"Change feature_extractor to {device}: allocated={round(torch.cuda.memory_allocated(rank)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(rank)/1024**3,1)}")

        self.reward_model.patch_masker = self.reward_model.patch_masker.to(device)
        self.reward_model.patch_masker.model = self.reward_model.patch_masker.model.to(device)
        self.reward_model.patch_masker.device = device

        # logger.debug(f"Change patch_masker to {device}: allocated={round(torch.cuda.memory_allocated(rank)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(rank)/1024**3,1)}")

        self._device = device

        return self

    def cuda(self, rank):
        device = f"cuda:{rank}"

        return self.to(device)

class DINOFullFeatureWrapper:
    def initialize_target_images(self):
        """
        self.target_mask_thresh is the threshold to apply to target images
        """
        # TODO: For now, we just support loading one image
        logger.debug(f"[{self._device}] Embedding human target image")

        target_image_path_list = self.pos_image_path_list + self.neg_image_path_list

        with torch.no_grad():
            target_image_embeddings = self.reward_model.feature_extractor.load_and_prepare_images_parallel(target_image_path_list)
            target_image_embeddings, target_image_masks = self.reward_model.extract_features_final(target_image_embeddings)

            self.target_image_embeddings = target_image_embeddings.repeat(self.batch_size, 1, 1, 1).permute(1,0,2,3)
            
        self.pos_idx_split = len(self.pos_image_path_list) # index at which self.target_image_embeddings and masks become negative examples

    def embed_module(self, image_batch):
        with torch.no_grad():
            if image_batch.shape[1] != 3:
                image_batch = image_batch.permute(0, 3, 1, 2)
            transformed_image = self.reward_model.feature_extractor.transform(image_batch)

            image_embeddings = self.reward_model.extract_features_final(transformed_image)

            return image_embeddings

    def __call__(self, source):
        all_ds = []
        with torch.no_grad():
            for i, target in enumerate(self.target_image_embeddings):

                distance = torch.linalg.vector_norm(target - source, dim=1)
                
                
                all_ds.append(distance)

                logger.debug(f"__call__: {distance.size()=}")
        if self.pos_idx_split < len(self.target_image_embeddings):
            total_distance = sum(all_ds[:self.pos_idx_split]) / len(all_ds[:self.pos_idx_split]) - (sum(all_ds[self.pos_idx_split:]) / len(all_ds[self.pos_idx_split:]))
            
        else:
            total_distance = sum(all_ds[:self.pos_idx_split]) / len(all_ds[:self.pos_idx_split])
            #total_distance = total_distance - 33.5 # TODO: MAGIC NUMBER offset to near 0 (tend to be around 33.4)

        #total_distance = 500*total_distance # TODO: magic number scales to rewards for RL
        return - total_distance  # Reward is the negative of the distance