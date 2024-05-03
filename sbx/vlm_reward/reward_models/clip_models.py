from typing import List, Optional, Tuple, overload

import open_clip
import torch
import torch.distributed as dist
import torch.nn as nn

from torchvision.transforms import (
    Normalize,
    Compose,
    InterpolationMode,
    Resize,
    CenterCrop,
)

from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

def load_clip_reward_model(
    model_name, target_prompts, baseline_prompts, alpha, cache_dir: Optional[str] = None
):
    model_name_prefix, pretrained = model_name.split("/")
    model = open_clip.create_model(
        model_name=model_name_prefix, pretrained=pretrained, cache_dir=cache_dir
    )
    target_prompts = CLIPReward.tokenize_prompts(target_prompts)
    baseline_prompts = CLIPReward.tokenize_prompts(baseline_prompts)
    model = CLIPEmbed(model)
    model = CLIPReward(
        model=model,
        alpha=alpha,
        target_prompts=target_prompts,
        baseline_prompts=baseline_prompts,
    )
    return model.eval()


class CLIPEmbed(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        if isinstance(clip_model.visual.image_size, int):
            image_size = clip_model.visual.image_size
        else:
            image_size = clip_model.visual.image_size[0]
        self.transform = image_transform(image_size)

    @torch.inference_mode()
    def forward(self, x):
        if x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)

        with torch.no_grad(), torch.autocast("cuda", enabled=torch.cuda.is_available()):
            x = self.transform(x)
            x = self.clip_model.encode_image(x, normalize=True)
        return x


class CLIPReward(nn.Module):
    def __init__(
        self,
        *,
        model: CLIPEmbed,
        alpha: float,
        target_prompts: torch.Tensor,
        baseline_prompts: torch.Tensor,
    ) -> None:
        """CLIP Reward function that modifies the CLIP vector space by
        projecting all vectors onto the line spanned by the prompt and
        a baseline prompt. The alpha parameter controls the degree of
        projection. A value of 0.0 means that the reward function is
        equivalent to the CLIP reward function. A value of 1.0 means
        that the vector space is completely projected onto the line
        and becomes a 1D space. Any value in between is a linear
        interpolation between the two.

        Args:
            model (str): CLIP model.
            device (str): Device to use.
            alpha (float, optional): Coeefficient of projection.
            target_prompts (torch.Tensor): Tokenized prompts describing
                the target state.
            baseline_prompts (torch.Tensor): Tokenized prompts describing
                the baseline state.
        """
        super().__init__()
        self.embed_module = model
        target = self.embed_prompts(target_prompts).mean(dim=0, keepdim=True)
        baseline = self.embed_prompts(baseline_prompts).mean(dim=0, keepdim=True)
        direction = target - baseline
        # Register them as buffers so they are automatically moved around.
        self.register_buffer("target", target)
        self.register_buffer("baseline", baseline)
        self.register_buffer("direction", direction)

        self.alpha = alpha
        projection = self.compute_projection(alpha)
        self.register_buffer("projection", projection)
    
    def compute_projection(self, alpha: float) -> torch.Tensor:
        projection = self.direction.T @ self.direction / torch.norm(self.direction) ** 2
        identity = torch.diag(torch.ones(projection.shape[0])).to(projection.device)
        projection = alpha * projection + (1 - alpha) * identity
        return projection

    def update_alpha(self, alpha: float) -> None:
        self.alpha = alpha
        self.projection = self.compute_projection(alpha)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = 1 - (torch.norm((x - self.target) @ self.projection, dim=-1) ** 2) / 2
        return y

    @staticmethod
    def tokenize_prompts(x: List[str]) -> torch.Tensor:
        """Tokenize a list of prompts."""
        return open_clip.tokenize(x)

    def embed_prompts(self, x) -> torch.Tensor:
        """Embed a list of prompts."""
        with torch.no_grad():
            x = self.embed_module.clip_model.encode_text(x).float()
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def embed_images(self, x):
        return self.embed_module.forward(x)

def image_transform(
    image_size: int,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that
        # Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    normalize = Normalize(mean=mean, std=std)

    def convert_from_uint8_to_float(image: torch.Tensor) -> torch.Tensor:
        if image.dtype == torch.uint8:
            return image.to(torch.float32) / 255.0
        else:
            return image

    return Compose(
        [
            convert_from_uint8_to_float,
            Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            normalize,
        ]
    )