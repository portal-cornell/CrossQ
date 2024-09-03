from typing import Any, List
from abc import ABC, abstractmethod
from PIL import Image

import torch
from torch import Tensor
from torchvision import transforms
import lpips

from vlm_reward.reward_models.model_interface import RewardModel # this is the interface class

def load_lpips_reward_model():
    return LPIPSRewardModel()

class LPIPSRewardModel(RewardModel):
    def __init__(self, rank: int = 0, net_type: str = 'alex'):
        """
        Initialize the LPIPS model with the specified network type.
        
        :param net_type: Type of network ('alex' or 'vgg') used for computing LPIPS.
        """
        self.model = lpips.LPIPS(net=net_type) # loss fn alex
        self.source_embedding = None
        self.target_embedding = None
        # to get image between [-1, 1], which is what lpips takes
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0)
        ])
        self.device = 'cpu'

    def predict(self) -> Tensor:
        """
        Calculate the distance between the source and target embeddings using LPIPS.
        
        Returns:
            A tensor containing the LPIPS distance.
        """
        if self.source_embedding is None or self.target_embedding is None:
            raise ValueError("Source and target embeddings must be set before prediction.")
        
        distance = self.model(self.source_embedding, self.target_embedding).squeeze() # (B, 1, 1, 1) -> (B,)
        similarity = 1.0 / (1.0 + distance)
        return similarity # reward

    def set_source_embeddings(self, image_batch: Tensor) -> None:
        """
        Set the source embeddings.
        
        :param image_batch: A batch of images as a tensor.
        """
        if image_batch.device != self.device: # because LPIPS doesn't have a device attribute
            image_batch = image_batch.to(self.device)

        if image_batch.dtype == torch.uint8:
            image_batch = image_batch.float() / 255.0
        
        # LPIPS embeds the images at prediction time
        self.source_embedding = image_batch

    def set_target_embedding(self, target_image: Tensor) -> None:
        """
        Set the target embedding.
        
        :param target_image: The target image as a tensor.
        """
        if target_image.device != self.device:
            target_image = target_image.to(self.device)
        
        if target_image.dtype == torch.uint8:
            target_image = target_image.float() / 255.0
            
        self.target_embedding = target_image

    def to(self, device: str) -> None:
        """
        Send the model and cached embeddings to the specified device.
        
        :param device: Device to send the model to ('cpu', 'cuda:0', etc.).
        """
        self.model = self.model.to(device)
        if self.source_embedding is not None:
            self.source_embedding = self.source_embedding.to(device)
        if self.target_embedding is not None:
            self.target_embedding = self.target_embedding.to(device)
        return self


    def cuda(self, rank: int = 0) -> None:
        """
        Shortcut for sending the model to a CUDA device.
        
        :param rank: CUDA device rank.
        """
        cuda_device = f'cuda:{rank}'
        self.device = cuda_device
        self.model.to(cuda_device)
        return self


    def eval(self):
        self.model.eval()
        return self

    def get_tensor_from_image(self, image_paths: List[str]) -> torch.Tensor:
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        # needs unsqueeze to get (3, 224, 224) to (1, 3, 224, 224)
        processed_images = [self.transform(image).unsqueeze(dim=0) for image in images] # list of (1, 3, 224, 224)
        return torch.cat(processed_images, dim=0) # (B, 3, 224, 224)

# # Example of use
# image_paths = ["axis_exp/humanoid_kneeling_ref.png", "axis_exp/humanoid_kneeling_ref.png"]

# reward_model = LPIPSRewardModel(net_type='alex')
# reward_model.cuda(0)

# image_ref = reward_model.get_tensor_from_image([image_paths[0]])
# image_batch = reward_model.get_tensor_from_image(image_paths)

# reward_model.set_source_embeddings(image_batch)
# reward_model.set_target_embedding(image_ref)

# reward = reward_model.predict()
# print(reward, reward.shape) # reward.shape: (B,)
