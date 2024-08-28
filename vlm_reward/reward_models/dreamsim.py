from abc import ABC, abstractmethod
from typing import Tuple, Any, List
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from dreamsim import dreamsim

from vlm_reward.reward_models.model_interface import RewardModel

def load_dreamsim_reward_model():
    return DreamSimRewardModel()

class DreamSimRewardModel(RewardModel):
    """
    Example of call:
    """
    def __init__(self):
        # Config: https://github.com/ssundaram21/dreamsim/blob/main/dreamsim/config.py
        # preprocess is used if we give image paths to RGB images as input
        # TODO: self.model instead of self.embed_module is better. But changed for now to
        # use the same name as in reward_main.dist_worker_compute_reward
        self.embed_module, self.preprocess = dreamsim(pretrained=True, cache_dir="/share/portal/wph52/CrossQ/models")
        self.source_embedding = None
        self.target_embedding = None
        self.device = 'cpu'

    def predict(self) -> torch.Tensor:
        """
        Calculates the DreamSim distance between the source and target embeddings.
        Assumes the embeddings are preprocessed images ready for the model.

        Note: another way would just be calling: self.embed_module(img_a, img_b)

        Returns:
            torch.Tensor: DreamSim distance score.
        """
        if self.source_embedding is None or self.target_embedding is None:
            raise ValueError("Source and target embeddings must be set before prediction.")
        # From the forward method of DreamSim:
        # https://github.com/ssundaram21/dreamsim/blob/99222ad4cd4512e975721665336fa8c795990ec3/dreamsim/model.py#L72
        distance = 1 - F.cosine_similarity(self.source_embedding, self.target_embedding, dim=-1)
        # The reward is the inverse of the distance
        similarity = 1.0 / (1.0 + distance)
        return similarity # reward

    def set_source_embeddings(self, image_batch: torch.Tensor) -> None:
        """
        Process and cache the embeddings of a batch of images.
        Assumes images are in the shape [b, c, h, w] where c=3, h=w=224.

        Args:
            image_batch (torch.Tensor): Batch of images. RGB images passed as 
            a (B, 3, 224, 224) tensor with values [0, 1] (preprocessed with get_tensor_from_image).
        """
        if image_batch.device != self.embed_module.device:
            image_batch = image_batch.to(self.embed_module.device)
        if image_batch.dtype == torch.uint8:
            image_batch = image_batch.float() / 255.0
        
        self.source_embedding = self.embed_module.embed(image_batch)

    def set_target_embedding(self, target_image: torch.Tensor) -> None:
        """
        Process and cache the embedding of the target image.
        Assumes image is in the shape [c, h, w] where c=3, h=w=224.

        Args:
            target_image (torch.Tensor): Target image. An RGB image passed as a (1, 3, 224, 224) 
            tensor with values [0, 1] (preprocessed with get_tensor_from_image).
        """
        if target_image.device != self.embed_module.device:
            target_image = target_image.to(self.embed_module.device)

        if target_image.dtype == torch.uint8:
            target_image = target_image.float() / 255.0

        if len(target_image.shape) == 3:
            target_image = target_image[None]

        self.target_embedding = self.embed_module.embed(target_image)[0]

    def to(self, device: str) -> None:
        """
        Send the model to a specified device.

        Args:
            device (str): The device to send the model to.
        """
        self.embed_module = self.embed_module.to(device)
        if self.source_embedding is not None:
            self.source_embedding = self.source_embedding.to(device)
        if self.target_embedding is not None:
            self.target_embedding = self.target_embedding.to(device)

    def cuda(self, rank: int) -> None:
        """
        Send the model to a CUDA device specified by rank.

        Args:
            rank (int): The rank of the CUDA device.
        """
        cuda_device = f'cuda:{rank}'
        self.device = cuda_device
        self.embed_module.to(self.device)
    
    def get_tensor_from_image(self, image_paths: List[str]) -> torch.Tensor:
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        processed_images = [self.preprocess(image) for image in images] # list of (1, 3, 224, 224)
        return torch.cat(processed_images, dim=0) # (B, 3, 224, 224)


# # Example of use
# image_paths = ["axis_exp/humanoid_kneeling_ref.png", "axis_exp/humanoid_kneeling_ref.png"]
# reward_model = DreamSimRewardModel()
# reward_model.cuda(0)

# # Convert images to tensor ()
# image_ref = reward_model.get_tensor_from_image([image_paths[0]])
# image_batch = reward_model.get_tensor_from_image(image_paths)

# # Set the source and target embeddings
# reward_model.set_source_embeddings(image_batch)
# reward_model.set_target_embedding(image_ref)

# # Calculate the reward
# reward = reward_model.predict()
# print(reward, reward.shape) # reward.shape: (B,)
