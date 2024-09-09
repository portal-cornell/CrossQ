from typing import Any, List
from PIL import Image

from jaxtyping import Float
from typing import Tuple, NewType, Any, Self

from loguru import logger
import torch
from torch import Tensor
from torchvision import transforms
import torch.nn.functional as F
from torchvision.utils import save_image

import numpy as np
import ot
import concurrent
from scipy.spatial.distance import cdist

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
from vlm_reward.reward_models.model_interface import RewardModel # this is the interface class

from huggingface_hub import hf_hub_download
from vlm_reward.utils.human_seg import HumanSegmentationModel

"""
Current implementation lacking the following features
- currently not using any memory encoding, but this is expensive and it seems to be good so far anyways

Plan: 

Prompt method:
- initialize a prompt based on the target image and embed it
- initialize a prompt based on the first frame in the source image
    - cannot be target image because it requires a point specific to the first image
    - after this, the current mask is used as the prompts for the next frame (put it into video mode)

Embedding method:
- get the best mask prediction using argmax(iou_pred), since sam gives us multiple mask predictions
- downsize the image embeddings and mask to a size suitable for wasserstein ex. (16, 16)
- index the embeddings using the mask to get patches only in the masked segment

Prediction method:
- find the wasserstein distance between the masked patches for target and source

Why can't we just use mask_tokens_out? Because this should be some "prompt" that is sort of specific to the mask
- i.e., it is some embedding for "humanoid", so that the model knows how to segment the humanoid in the future
- the whole point is that, even if the humanoid is in different positions or part is occluded, it is still a humanoid and thus should be masked. This token allows for that
    - So basically, this will probably "undo" all the differences we care about (like position/rotation)

later: find wasserstein distance between all the mask predictions (for when there is uncertainty in the prompts)
"""

def load_sam2_mean_feature_reward_model(rank: int, sam2_model_id: str, sam2_cfg_path:str, human_seg_model_path,
                                    source_mask_thresh, target_mask_thresh, batch_size:int):

    return SAM2MeanFeatureRewardModel(rank, sam2_model_id, sam2_cfg_path, human_seg_model_path,
                                    source_mask_thresh, target_mask_thresh, batch_size)
    


def load_sam2_wasserstein_reward_model(rank: int, sam2_model_id: str, sam2_cfg_path:str, human_seg_model_path,
                                    source_mask_thresh, target_mask_thresh, batch_size:int):

    return SAM2WassersteinRewardModel(rank, sam2_model_id, sam2_cfg_path, human_seg_model_path,
                                    source_mask_thresh, target_mask_thresh, batch_size)

class SAM2RewardModelBase(RewardModel):
    
    def __init__(self, rank: int, sam2_model_id: str, sam2_cfg_path:str, human_seg_model_path: str, 
                source_mask_thresh: str, target_mask_thresh: str, batch_size: int, patch_dim=(12,12)):
        """
        Initialize the SAM2-large model with the specified network type.
        
        :param sam2_model_id: the sam2 checkpoint id (for huggingface)
        :param sam2_cfg_path: the path to a config expected by build_sam2
        :param batch_size: the size of batches on which to perform embedding calculations
        :param first_frame_src_point_coords: an (N,2) np array containing the normalized coordinates of points for the initial prompt of the source image
        :param first_frame_src_point_labels: an (N,) array containing the labels (1 for foreground, 0 for background) of the initial prompts
        :param trg_point_coords: same as src, but for the target image
        :param trg_point_labels: same as src, but for the target image
        :param patch_dim: the dimension to interpolate embedding patches to (for more efficient storage and eventually wasserstein calculations)

        Note that the coordinates are normalized by dividing width by total width and height by total height
        """
        self.device = f'cuda:{rank}'
        self.batch_size = batch_size
        self.model_id = sam2_model_id
        self.cfg_path = sam2_cfg_path
        self.model = self._build_sam2_hf(sam2_model_id, sam2_cfg_path, self.device)

        self.source_embedding = None
        self.target_embedding = None
        self.patch_dim = patch_dim

        self.mask_pooling = torch.nn.MaxPool2d(patch_dim) # if any mask logit is positive, use it
        self.embed_pooling = torch.nn.AvgPool2d(patch_dim)

        # the threshold for mask logits (in HumanSeg, found that .01 is good for mujoco and .5 is good for real life)
        self.human_seg_model = self.instantiate_human_seg(rank, human_seg_model_path)
        self.source_mask_thresh = source_mask_thresh
        self.target_mask_thresh = target_mask_thresh

    def instantiate_human_seg(self, rank, human_seg_model_path):
        human_seg_model = HumanSegmentationModel(rank, human_seg_model_path)
        human_seg_model = human_seg_model.to(self.device)
        human_seg_model.model = human_seg_model.model.to(self.device)
        human_seg_model.device = self.device
        return human_seg_model

    def _build_sam2_hf(self, model_id, sam2_cfg_path, device):

        model_id_to_filenames = {
            "facebook/sam2-hiera-large": ("sam2_hiera_l.yaml", "sam2_hiera_large.pt"),
        }
        _, checkpoint_name = model_id_to_filenames[model_id]

        ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
        return SAM2ImagePredictor(build_sam2(config_file=sam2_cfg_path, ckpt_path=ckpt_path, device=device))

    def predict(self) -> Tensor:
        """
        Calculate the distance between the source and target embeddings using Sam2.
        
        SAM gives dense embeddings (not a single vector). So this needs to some feature pooling or histogram-based metric (like wasserstein)

        Returns:
            A tensor containing the distance between the embeddings
        """
        raise NotImplementedError("predict() for SAM2RewardModelBase purposefully not implemented. Subclass and implement it yourself")

    def _get_all_sam2_encoder_features(self, image_batch: Tensor) -> Tensor:
        """
        For each image in the batch, get the low and high res features from the sam2 encoder
        This is essentially just running hiera(image_batch) because the embedding is retrieved before the masked decoder (no prompts included)
        """
        self.model.reset_predictor()
        self.model.set_image_batch(image_batch)

        low_res_feats = self.model._features['image_embed'].clone()
        high_res_feats = [feature.clone() for feature in self.model._features['high_res_feats']]
        all_feats = [low_res_feats] + high_res_feats
        return all_feats

    def set_source_embeddings(self, images: Tensor) -> None:
        """
        Set the source embeddings.
        images: a set of images
        
        For each image in the batch, get embeddings and masks, downsize them to patch_dim, and apply the masks
        Embedding is retrieved after applying the masked decoder (cross attention on the prompts)

        The first image in the batch is prompted with self.first_frame_src_point_coords and self.first_frame_src_point_labels.
        Subsequent images are prompted with the previous image's masks

        """
        if images.device != self.device: 
            images = images.to(self.device)
        if images.dtype == torch.uint8:
            images = images.float() / 255

        source_image_batches = torch.split(images, self.batch_size)

        source_embeddings = []
        source_masks = []
        # inference on all at once will cause an out of memory error
        for i, image_batch in enumerate(source_image_batches):
            logger.info(f"batch {i} / {len(source_image_batches)}")
            mask_input_batch = self.human_seg_model.forward_logits(image_batch)
            # transform to input space for sam2 (weird hacky thing)
            # 8 comes from qualitative evaluation (sam2 produces good masks when input is 8)
            mask_input_batch_logits = torch.clamp(torch.log(torch.clamp(mask_input_batch, 1e-9, 1)), -32, 32)  + 8 
            
            self.model.set_image_batch(image_batch) # model expects HWC numpy input
            
            embeddings, masks, ious, sam_tokens = self.model.predict_batch_embeddings(mask_input_batch=mask_input_batch_logits)
            embeddings_downsize = self.embed_pooling(embeddings)
            masks_downsize = self.mask_pooling(masks) # create artificial channel dim and convert to float tensor for interpolate, then convert back
            binary_masks_downsize = masks_downsize > self.source_mask_thresh
            source_embeddings.append(embeddings_downsize)
            source_masks.append(binary_masks_downsize)

        masked_embeddings = self._index_embeddings(torch.cat(source_embeddings, dim=0), torch.cat(source_masks, dim=0))
        self.source_embedding = masked_embeddings

    def set_target_embedding(self, target_image: Tensor) -> None:
        """
        Set the target embedding, using the trg point coords and labels as prompts
        
        :param target_image: The target image as a tensor.
        """
        if target_image.device != self.device:
            target_image = target_image.to(self.device)
        if target_image.dtype == torch.uint8:
            target_image = target_image.float() / 255
        if len(target_image.shape) == 3: 
            # batchify input
            target_image_batch = target_image[None]
        
        mask_input_batch = self.human_seg_model.forward_logits(target_image_batch)
        mask_input_batch_logits = torch.clamp(torch.log(torch.clamp(mask_input_batch, 1e-9, 1)), -32, 32) # get log probs from true probs, and clip them into range

        self.model.set_image_batch(target_image_batch) # model expects HWC numpy input
        embeddings, masks, ious, sam_tokens = self.model.predict_batch_embeddings(mask_input_batch=mask_input_batch_logits)

        embeddings_downsize = self.embed_pooling(embeddings)
        masks_downsize = self.mask_pooling(masks) # create artificial channel dim and convert to float tensor for interpolate, then convert back
        binary_masks_downsize = masks_downsize > self.source_mask_thresh
        masked_embeddings = self._index_embeddings(embeddings_downsize, binary_masks_downsize)

        # unbatchify output
        self.target_embedding = masked_embeddings[0]

        
    def to(self, device: str) -> None:
        """
        TODO: this is hacky, because sam2 only lets you set device at model instantation
        """
        self.human_seg_model =self.human_seg_model.to(device)
        self.human_seg_model.model = self.human_seg_model.model.to(device)
        self.human_seg_model.device = device

        assert 'cuda' in device, "Error: cuda not found in device. Must be on gpu"
        if self.device != device:
            return self.__class__(device[-1], self.model_id, self.cfg_path, self.human_seg_model_path, self.source_mask_thresh, self.target_mask_thresh, self.batch_size)
        
        
    def cuda(self, rank: int = 0) -> None:
        """
        TODO: see self.to
        Shortcut for sending the model to a CUDA device.
        
        :param rank: CUDA device rank.
        """
        return self.to(f"cuda:{rank}")

    def _index_embeddings(self, embeddings: Float[Tensor, 'b c h w'], masks: Float[Tensor, 'b h w']) -> List[Tensor]:
        """
        embeddings: a set of dense embeddings for an image
        masks: a boolean mask for each embedding
        patch_dim: the dimension 

        Index the embeddings by the masks
        """
        b, c, h, w = embeddings.shape
        masks_flat = masks.view(b, h*w)
        embeddings_flat = embeddings.view(b,h*w, c)

        masked_embeddings = []
        for emb, mask in zip(embeddings_flat, masks_flat):
            if mask.sum() == 0:
                logger.warning("WARNING: mask is empty, defaulting to all embeddings for this image")
                masked_embeddings.append(emb)
            else:    
                masked_embeddings.append(emb[mask])
        return masked_embeddings

    def torch_to_numpy(self, tensor):
        """
        safely convert any tensor to numpy
        """
        return tensor.float().detach().cpu().numpy()

    def eval(self):
        return self

class SAM2MeanFeatureRewardModel(SAM2RewardModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self) -> Tensor:
        if self.source_embedding is None or self.target_embedding is None:
            raise ValueError("Source and target embeddings must be set before prediction.")
        
        return self._mean_feature_distance(self.source_embedding, self.target_embedding)
    
    def _mean_feature_distance(self, masked_sources: List[Float[Tensor, "n_masked_patches channels"]], target: Float[Tensor, "n_masked_patches channels"]):
        target_mean = target.mean(dim=0)
        source_means = torch.stack([source.mean(dim=0) for source in masked_sources])
        
        d = F.cosine_similarity(source_means, target_mean[None])
        return d

class SAM2WassersteinRewardModel(SAM2RewardModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self) -> Tensor:
        if self.source_embedding is None or self.target_embedding is None:
            raise ValueError("Source and target embeddings must be set before prediction.")
        
        return torch.as_tensor(self._wasserstein_distance(self.source_embedding, self.target_embedding))
    
    def _wasserstein_distance(self, sources: List[Float[Tensor, "n_masked_patches channels"]], target: Float[Tensor, "n_masked_patches channels"], return_ot_plan=False):
        sources_cpu_np = [source.cpu().numpy() for source in sources]
        target_cpu_np = target.cpu().numpy()

        def compute_wasser_given_target(source_cpu_np):
            # compute the wasserstein using the cached target
            return self.compute_patchwise_wasserstein(source_cpu_np, target_cpu_np)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = executor.map(compute_wasser_given_target, sources_cpu_np) 

            raw_output_dict = [future for future in futures]
            wassersteins = [o["wasser"] for o in raw_output_dict]
            if return_ot_plan:
                ot_plan = [o["T"] for o in raw_output_dict]
                costs = [o["C"] for o in raw_output_dict]
        if return_ot_plan: # for debugging
            return dict(wasser=wassersteins, T=ot_plan, C=costs)
        else:
            return wassersteins

    def compute_patchwise_wasserstein(self, features, target_features, d='cosine'):
        """
        Return: a dictionary that contains field
            "wasser": the wasserstein distance
            "T": (optional) the optimal transport plan
        """   
        if len(features) == 0:
            print('Error: no source features found. Distance is considered -1. Consider decreasing the human threshold for the source or target')
            return dict(wasser = -1)
        elif len(target_features) == 0:
            print('Error: no target features found. Distance is considered -1. Consider decreasing the human threshold for the source or target')
            return dict(wasser = -1)
        
        M = cdist(target_features, features, metric=d)
        
        # Sinkhorn2 directly outputs the distance (compared to sinkhorn which outputs the OT solution)
        # For regularizing, using this paper: (https://github.com/siddhanthaldar/ROT/blob/41ef7b98ca3950b9f31dd174f306cbe6916a09c9/ROT/rewarder.py#L4)
        T = ot.sinkhorn([], [], M, reg=0.05, log=False)

        # Calculate the wasserstein distance (ref: )
        wasser = np.sum(T*M)

        return dict(wasser=wasser, T=T, C=M)