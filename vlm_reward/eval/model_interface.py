
import torch 
from torch import Tensor
from jaxtyping import Float
from typing import Tuple, NewType, Any

from abc import abstractmethod, ABC

class RewardModel(ABC):
    """
    The following methods must be defined under these specifications for all reward models. 
    
    The exact implementations of the distance function and the image encoder are not defined here. Thus, the embedding type 
    cannot be known (it could be a tuple containing a mask, an array of hierarchical embeddings, etc.).
    
    """

    @abstractmethod
    def predict(self) -> Float[torch.Tensor, "n"]:
        """
        Call the reward model on the cached source and target embeddings.
        
        The target embedding is defined by set_target_embedding
        The source embedding is set by embed_module()

        Returns: rewards for the current source and target embeddings
        """
        pass 

    @abstractmethod
    def set_source_embeddings(self, image_batch: Float[torch.Tensor, "b c h w"]) -> None:
        """
        Embed a batch of images, caching the embeddings as instance variables
        
        The embeddings will have arbitrary shapes/types, but it is guaranteed that 
        self.predict uses them as inputs
        """
        pass

    @abstractmethod
    def set_target_embedding(self, target_image: Float[torch.Tensor, "c h w"]) -> None:
        """
        Cache an embedding of the target image
        """
        pass

    @abstractmethod    
    def to(self, device: str) -> None:
        """
        Send the model to device
        """
        pass

    @abstractmethod
    def cuda(self, rank: int) -> None:
        """
        Send the model to cuda
        """
        pass

