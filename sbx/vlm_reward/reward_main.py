from typing import List, Optional, Tuple, overload

import torch
import torch.distributed as dist

from loguru import logger

# TODO: finish all the imports needed here
# from sbx.vlm_reward.reward_models.dino_models import load_dino_reward_model
from sbx.vlm_reward.reward_models.clip_models import load_clip_reward_model

def load_reward_model(
                    model_name, 
                    model_config_dict):
    assert any([model_base_name in model_name.lower() for model_base_name in ["vit", "dino"]])

    # if "dino" in model_name.lower():
    #     reward_model = load_dino_reward_model(dino_model_name=model_name,
    #                                         metric=model_config_dict["metric"],
    #                                         human_seg_weight_path=model_config_dict["human_seg_weight_path"],
    #                                         target_human_threshold=model_config_dict["target_human_threshold"],
    #                                         tmp_dir=model_config_dict["tmp_dir"])
    if "vit" in model_name.lower():
        reward_model = load_clip_reward_model(model_name=model_name,
                                                target_prompts=model_config_dict["target_prompts"],
                                                baseline_prompts=model_config_dict["baseline_prompts"],
                                                alpha=model_config_dict["alpha"],
                                                cache_dir=model_config_dict["cache_dir"])

    return reward_model


def compute_rewards(
    model,
    frames: torch.Tensor,
    batch_size: int,
    num_workers: int,
    worker_frames_tensor=None,
) -> torch.Tensor:
    assert frames.device == torch.device("cpu")
    assert batch_size % num_workers == 0
    n_samples = len(frames)
    rewards = torch.zeros(n_samples, device=torch.device("cpu"))
    model = model.eval()
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            frames_batch = frames[i : i + batch_size]
            rewards_batch = dist_worker_compute_reward(
                rank=0,
                reward_model=model,
                render_dim=frames_batch.shape[1:],
                batch_size=batch_size // num_workers,
                num_workers=num_workers,
                frames=frames_batch,
                worker_frames_tensor=worker_frames_tensor,
            )
            rewards_batch = rewards_batch.cpu()
            rewards[i : i + batch_size] = rewards_batch 
    return rewards


@overload
def dist_worker_compute_reward(
    rank: int,
    reward_model,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames: torch.Tensor,
) -> torch.Tensor:
    ...


@overload
def dist_worker_compute_reward(
    rank: int,
    reward_model,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames: None = None,
) -> None:
    ...


def dist_worker_compute_reward(
    rank: int,
    reward_model,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames=None,
    worker_frames_tensor=None,
) -> Optional[torch.Tensor]:
    logger.info(f"[Worker {rank}] Computing reward...")
    if rank == 0:
        if frames is None:
            raise ValueError("Must pass render result on rank=0")
        if len(frames) != num_workers * batch_size:
            raise ValueError("Must pass render result with correct batch size")
        scatter_list = [t.cuda(rank) for t in torch.chunk(frames, num_workers, dim=0)]
    else:
        scatter_list = []

    worker_frames = worker_frames_tensor if worker_frames_tensor is not None else torch.zeros((batch_size, *render_dim), dtype=torch.uint8).cuda(rank)
    dist.scatter(worker_frames, scatter_list=scatter_list, src=0)
    with torch.no_grad():
        embeddings = reward_model.embed_module(worker_frames)
        rewards = reward_model(embeddings)

    def zero_t():
        return torch.zeros_like(rewards)

    recv_rewards = [zero_t() for _ in range(num_workers)] if rank == 0 else []
    dist.gather(rewards, gather_list=recv_rewards, dst=0)

    if rank == 0:
        return torch.cat(recv_rewards, dim=0).cuda(rank)
