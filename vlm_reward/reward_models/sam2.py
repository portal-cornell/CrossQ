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

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
from vlm_reward.reward_models.model_interface import RewardModel # this is the interface class

from huggingface_hub import hf_hub_download

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

def load_sam2_mean_feature_reward_model(rank: int, sam2_model_id: str, sam2_cfg_path:str,batch_size:int):

    # normalized coordinates of points on the robot in the initial frame of source trajectories and target image
    # note that the coordinates are centered (i.e, (0,0) corresponds to the center of the image) 

    # coords[0,:] corresponds to width (distance from left)
    # and coords[1, :] corresponds to height (distance from top, so 1 is the bottom of the image)
    # the below correspond to left arm, right arm, hips, face of robot, ground, and sky respectively
    first_frame_src_point_coords = np.array([[.5, .6], [.4125, .5], [.6, .5], [.5, .4125], [.125, .833], [.833, .125]])
    first_frame_src_point_labels = np.array([1,1,1,1,0,0]) # sky is a background point
    trg_point_coords = np.array([[.4125, .5], [.6, .4], [.125, .833]]) # TODO: get correct point coords for trg here
    trg_point_labels = np.array([1, 1, 0])

    return SAM2MeanFeatureRewardModel(rank, sam2_model_id, sam2_cfg_path, batch_size,
                                    first_frame_src_point_coords, first_frame_src_point_labels,
                                    trg_point_coords, trg_point_labels)
    
class SAM2RewardModelBase(RewardModel):
    
    def __init__(self, rank: int, sam2_model_id: str, sam2_cfg_path:str, batch_size: int,
                first_frame_src_point_coords: np.ndarray, first_frame_src_point_labels: np.ndarray,
                trg_point_coords: np.ndarray, trg_point_labels: np.ndarray, patch_dim=(12,12)):
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
        self.cfg_path = sam2_cfg_path
        self.model = self._build_sam2_hf(sam2_model_id, sam2_cfg_path, self.device)

        self.source_embedding = None
        self.target_embedding = None
        self.patch_dim = patch_dim

        self.first_frame_src_point_coords = first_frame_src_point_coords
        self.first_frame_src_point_labels = first_frame_src_point_labels

        # these embeddings will be computed when goal image is set (doesn't need to be precomputed here)
        self.trg_point_coords = trg_point_coords
        self.trg_point_labels = trg_point_labels

        self.mask_pooling = torch.nn.MaxPool2d(patch_dim) # if any mask logit is positive, use it
        self.embed_pooling = torch.nn.AvgPool2d(patch_dim)
        self.mask_threshold = 0 # the threshold for mask logits (0 in standard SAM, but can be tuned for more/less refined masks)

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

    def set_source_embeddings(self, image_sequence: Tensor) -> None:
        """
        Set the source embeddings.
        image_sequence: a sequence of images (it is important that they occur in order), where the given prompts 
        correspond to the first image in the batch
        
        For each image in the batch, get embeddings and masks, downsize them to patch_dim, and apply the masks
        Embedding is retrieved after applying the masked decoder (cross attention on the prompts)

        The first image in the batch is prompted with self.first_frame_src_point_coords and self.first_frame_src_point_labels.
        Subsequent images are prompted with the previous image's masks

        """
        if image_sequence.device != self.device: # because LPIPS doesn't have a device attribute
            image_sequence = image_sequence.to(self.device)
        if image_sequence.dtype == torch.uint8:
            image_sequence = image_sequence.float() / 255
        
        all_masked_embeddings = []
        all_ious = []        
        all_sam_tokens = []

        for i, image in enumerate(image_sequence):
            self.model.set_image(image) # model expects HWC numpy input

            if i == 0:
                # use the point prompts for the first frame
                embeddings, masks, ious, sam_tokens = self.model.predict_embeddings(
                    sparse_prompt_embeddings=self.first_frame_sparse_embeddings, 
                    dense_prompt_embeddings=self.first_frame_dense_embeddings, 
                    multi_object_mode=self.first_frame_multi_object_mode
                )
            else:
                # use the mask prompts from the previous frame
                sparse_prompt_embeddings, dense_prompt_embeddings, multi_object_mode= self.model.get_mask_embeddings(masks)
                embeddings, masks, ious, sam_tokens = self.model.predict_embeddings(
                    sparse_prompt_embeddings=sparse_prompt_embeddings, 
                    dense_prompt_embeddings=dense_prompt_embeddings, 
                    multi_object_mode=multi_object_mode
                )
            
            embeddings_downsize = self.embed_pooling(embeddings)
            masks_downsize = self.mask_pooling(masks.unsqueeze(1)).squeeze(1) # create artificial channel dim and convert to float tensor for interpolate, then convert back
            binary_masks_downsize = masks_downsize > self.mask_threshold

            masked_embeddings = self._index_embeddings(embeddings_downsize, binary_masks_downsize)

            all_masked_embeddings.append(masked_embeddings)
            all_ious.append(ious)
            all_sam_tokens.append(sam_tokens)

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

        self.model.set_image(target_image) # model expects HWC numpy input

        sparse_prompt_embeddings, dense_prompt_embeddings, multi_object_mode = self.model.get_point_embeddings(
            point_coords=self.trg_point_coords, 
            point_labels=self.trg_point_labels,
            normalize_coords=False
        )
        target_embeddings, masks, ious, sam_tokens = self.model.predict_embeddings(
            sparse_prompt_embeddings=sparse_prompt_embeddings, 
            dense_prompt_embeddings=dense_prompt_embeddings, 
            multi_object_mode=multi_object_mode
        )

        self.target_embedding = target_embeddings[0]

        # first frame embeddings are precomputed here as well, because the target image has been set, so the model knows what image dim to expect
        self.first_frame_sparse_embeddings, self.first_frame_dense_embeddings, self.first_frame_multi_object_mode = self.model.get_point_embeddings(
            point_coords=self.first_frame_src_point_coords, 
            point_labels=self.first_frame_src_point_labels,
            normalize_coords=False
        )
        
    def to(self, device: str) -> None:
        """
        TODO: this is hacky, because sam2 only lets you set device at model instantation
        """
        assert 'cuda' in device, "Error: cuda not found in device. Must be on gpu"
        if self.device != device:
            return SAM2RewardModel(device[-1], self.cfg_path, self.checkpoint_path)
        
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
        breakpoint()
        target_mean = target.mean(dim=0)
        source_means = torch.stack([source.mean(dim=0) for source in masked_sources])
        
        d = F.cosine_similarity(source_means, target_mean[None])
        return d
