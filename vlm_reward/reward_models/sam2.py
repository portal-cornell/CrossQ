from typing import Any, List
from PIL import Image

from jaxtyping import Float
from typing import Tuple, NewType, Any, Self


import torch
from torch import Tensor
from torchvision import transforms
import torch.nn.functional as F

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
from vlm_reward.reward_models.model_interface import RewardModel # this is the interface class

from huggingface_hub import hf_hub_download

def load_sam2_reward_model(rank: int, sam2_model_id: str, sam2_cfg_path:str,batch_size:int):
    return SAM2RewardModel(rank, sam2_model_id, sam2_cfg_path, batch_size)
    
class SAM2RewardModel(RewardModel):
    """

    Plan: 
    1. get the best mask prediction using argmax(iou_pred)
    2. downsize the image and embeddings to a size suitable for wasserstein ex. (16, 16)
    3. break the image into the predicted masks
    4. find the wasserstein distance between the predicted segments

    Why can't we just use mask_tokens_out? Because this should be some "prompt" that is sort of specific to the mask
    - i.e., it is some embedding for "humanoid", so that the model knows how to segment the humanoid in the future
    - the whole point is that, even if the humanoid is in different positions or part is occluded, it is still a humanoid and thus should be masked. This token allows for that
        - So basically, this will probably "undo" all the differences we care about (like position/rotation)

    later: find wasserstein distance between all the mask predictions (for when there is uncertainty in the prompts)
    """
    patch_resolution = (16, 16) # downsize the masks and embeddings to this so we can do wasserstein efficiently
    
    def __init__(self, rank: int, sam2_model_id: str, sam2_cfg_path:str, batch_size: int):
        """
        Initialize the SAM2-large model with the specified network type.
        
        :param sam2_cfg_path: the path to a config expected by build_sam2
        :param sam2_checkpoint_path: the path to a sam2 checkpoint
        :param batch_size: the size of batches on which to perform embedding calculations
        """
        self.device = f'cuda:{rank}'
        self.batch_size = batch_size
        self.cfg_path = sam2_cfg_path
        self.model = self._build_sam2_hf(sam2_model_id, sam2_cfg_path, self.device)
        #sam2 = build_sam2(sam2_cfg_path, sam2_checkpoint_path, device)
        #self.model = SAM2ImagePredictor(sam2)

        self.source_embedding = None
        self.target_embedding = None


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
        
        TODO: SAM gives dense embeddings (not a single vector). Use wasserstein here? Or extract from mask inference?

        Returns:
            A tensor containing the distance between the embeddings
        """
        if self.source_embedding is None or self.target_embedding is None:
            raise ValueError("Source and target embeddings must be set before prediction.")
        breakpoint()
        # TODO: implement wasserstein and mean feature here (well, make a new class for mean feature)
        pass 


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

    def _get_sam2_mask_predict_features(self, image_batch: Tensor, patch_dim: Tuple[int, int]):
        """
        For each image in the batch, get embeddings and masks, downsize them to patch_dim, and apply the masks
        Embedding is retrieved after applying the masked decoder (cross attention on the prompts)

        TODO: maybe set batch size to 1 and just iterate each time? This could make it easier to add previous prompts to next inference
        """
        self.model.reset_predictor()
        self.model.set_image_batch(image_batch)
 
        # ious are confidence in the masks, sam_tokens are stored in the memory and somehow inputs for the fturue
        all_embeddings, all_masks, all_ious, all_sam_tokens = self.model.predict_batch_embeddings()
        
        all_embeddings_downsize = F.interpolate(all_embeddings, patch_dim, mode='bilinear')
        all_masks_downsize = F.interpolate(all_masks.unsqueeze(1), patch_dim, mode='bilinear')[:, 0, ...] # create artificial channel dim for interpolate

        masked_embeddings = self._index_embeddings(all_embeddings_downsize, all_masks_downsize)
        return masked_embeddings

    def set_source_embeddings(self, image_batch: Tensor) -> None:
        """
        Set the source embeddings.
        
        :param image_batch: A batch of images as a tensor.
        """
        if image_batch.device != self.device: # because LPIPS doesn't have a device attribute
            image_batch = image_batch.to(self.device)
        if image_batch.dtype == torch.uint8:
            image_batch = image_batch.float()

        image_batch_split = torch.split(image_batch, self.batch_size)
        
        self.source_embedding = []
        # inference on all at once will cause an out of memory error
        for batch in image_batch_split:
            # UNCOMMENT BELOW for just using the raw embeddings (if you wanna average them or something)
            # self.source_embedding = self._get_all_sam2_encoder_features(batch)
            self.source_embedding += self._get_sam2_mask_predict_features(batch)
    
    def set_target_embedding(self, target_image: Tensor) -> None:
        """
        Set the target embedding.
        
        :param target_image: The target image as a tensor.
        """
        if target_image.device != self.device:
            target_image = target_image.to(self.device)
        
        if target_image.dtype == torch.uint8:
            target_image = target_image.float()
            
        if len(target_image.shape) == 3:
            target_image = target_image[None]
        self.target_embedding = self._get_all_sam2_features(target_image)[0]

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

    def _index_embeddings(self, embeddings: Float[Tensor, 'b c h w'], masks: Float[Tensor, 'b h w']) -> Float[Tensor, 'b positive_patches']:
        """
        embeddings: a set of dense embeddings for an image
        masks: a boolean mask for each embedding
        patch_dim: the dimension 

        Index the embeddings by the masks
        """
        b, c, h, w = embeddings.shape
        masks_flat = masks.view(b, h*w)
        embeddings_flat = embeddings.view(b,h*w, c)
        masked_embeddings = [emb[mask] for emb, mask in zip(embeddings_flat, masks_flat)]
        return masked_embeddings

    def torch_to_numpy(self, tensor):
        """
        safely convert any tensor to numpy
        """
        return tensor.float().detach().cpu().numpy()

    def eval(self):
        return self
