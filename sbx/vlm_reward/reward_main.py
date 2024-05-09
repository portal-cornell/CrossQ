from typing import List, Optional, Tuple, overload

import torch
import torch.distributed as dist

from loguru import logger

# TODO: finish all the imports needed here
from sbx.vlm_reward.reward_models.dino_models import load_dino_reward_model
from sbx.vlm_reward.reward_models.clip_models import load_clip_reward_model

def load_reward_model(
                    rank,
                    batch_size,
                    model_name, 
                    model_config_dict):
    assert any([model_base_name in model_name.lower() for model_base_name in ["vit", "dino"]])

    if "dino" in model_name.lower():
        reward_model = load_dino_reward_model(rank=rank,
                                                batch_size=batch_size,
                                                model_name=model_name,
                                                image_metric=model_config_dict["image_metric"],
                                                human_seg_model_path=model_config_dict["human_seg_model_path"],
                                                ref_image_path_list=model_config_dict["ref_image_path"])

        logger.debug(f"Loaded DINO reward model. model_name={model_name}, ref_image={model_config_dict['ref_image_path']}")

    if (not ("dino" in model_name.lower())) and ("vit" in model_name.lower()):
        reward_model = load_clip_reward_model(model_name=model_name,
                                                target_prompts=model_config_dict["target_prompts"],
                                                baseline_prompts=model_config_dict["baseline_prompts"],
                                                alpha=model_config_dict["alpha"],
                                                cache_dir=model_config_dict["cache_dir"])

        logger.debug(f"Loaded CLIP reward model. model_name={model_name}, target_prompts={model_config_dict['target_prompts']}")

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
    logger.debug(f"compute_rewards: {n_samples=}, {batch_size=}")
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
        logger.debug(f"[Worker {rank}] before chunking {frames.size()=}")
        # total_size = frames.size()[0] if frames is not None else batch_size
        # rank_0_chunk_size = int(0.2 * total_size)
        # remaining_size = total_size - rank_0_chunk_size
        # remaining_chunk_size = remaining_size // (num_workers - 1)

        # chunk_list = [rank_0_chunk_size] + [remaining_chunk_size] * (num_workers - 1)
        # # TODO: Let's assume that the batch size is picked in a way that the remaining (num_workers - 1)
        # #   chunks of data all have the same size
        # # if remaining_size % (num_workers - 1) != 0:
        # #     last_chunk_size = remaining_size - remaining_chunk_size * (num_workers - 2)
        # #     chunk_list[-1] = last_chunk_size
        # logger.debug(f"[Worker {rank}] {chunk_list=}")

        # scatter_list = [t.cuda(rank) for t in torch.split(frames, split_size_or_sections=chunk_list, dim=0)]
        
        # scatter_list[0] = 
        # TODO: pad the first tensor
    else:
        scatter_list = []

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # worker_frames_size = rank_0_chunk_size if rank == 0 else batch_size
    # logger.debug(f"[Worker {rank}] does worker_frames_tensor exist: {worker_frames_tensor.size() if worker_frames_tensor is not None else None}")
    worker_frames = worker_frames_tensor if worker_frames_tensor is not None else torch.zeros((batch_size, *render_dim), dtype=torch.uint8).cuda(rank)
    logger.debug(f"[Worker {rank}] {worker_frames.size()=}, scatter_list={[x.size() for x in scatter_list]}")
    dist.scatter(worker_frames, scatter_list=scatter_list, src=0)
    with torch.no_grad():
        start_event.record()
        # logger.debug(f"[Worker {rank}] {worker_frames.size()=} allocated={round(torch.cuda.memory_allocated(rank)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(rank)/1024**3,1)}")
        embeddings = reward_model.embed_module(worker_frames)
        end_event.record()
        # if type(embeddings) == tuple:
        #     logger.debug(f"[Worker {rank}] {embeddings[0].size()= } allocated={round(torch.cuda.memory_allocated(rank)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(rank)/1024**3,1)}")
        # else:
        #     logger.debug(f"[Worker {rank}] {embeddings.size()= } allocated={round(torch.cuda.memory_allocated(rank)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(rank)/1024**3,1)}")
        
        ### timing
        torch.cuda.synchronize()
        execution_time = start_event.elapsed_time(end_event)
        logger.debug(f"Worker #{rank} - Image Embedding time: {execution_time / 1000} seconds")
        ####

        start_event.record()
        rewards = reward_model(embeddings)
        end_event.record()
        # logger.debug(f"[Worker {rank}] {rewards.size()=} allocated={round(torch.cuda.memory_allocated(rank)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(rank)/1024**3,1)}")


        ### timing
        torch.cuda.synchronize()
        execution_time = start_event.elapsed_time(end_event)
        logger.debug(f"Worker #{rank} - Reward Calculation time: {execution_time / 1000} seconds")
        ####

    def zero_t():
        return torch.zeros_like(rewards)

    recv_rewards = [zero_t() for _ in range(num_workers)] if rank == 0 else []
    dist.gather(rewards, gather_list=recv_rewards, dst=0)

    if rank == 0:
        consolidated_tensors = torch.cat(recv_rewards, dim=0).cuda(rank)
        logger.debug(f"[Worker {rank}] {consolidated_tensors=}")
        return torch.cat(recv_rewards, dim=0).cuda(rank)
