from typing import List, Optional, Tuple, overload

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.utils import save_image

from loguru import logger

# TODO: finish all the imports needed here
from sbx.vlm_reward.reward_models.dino_models import load_dino_reward_model
from sbx.vlm_reward.reward_models.clip_models import load_clip_reward_model



def load_reward_model(
                    rank,
                    worker_actual_batch_size,
                    model_name, 
                    model_config_dict):
    assert any([model_base_name in model_name.lower() for model_base_name in ["vit", "dino"]])

    if "dino" in model_name.lower():
        if "neg_image_path" in model_config_dict:
            neg_image_path_list=model_config_dict["neg_image_path"]
        else:
            neg_image_path_list = []

        if "baseline_image_path" in model_config_dict:
            baseline_image_path = model_config_dict["baseline_image_path"]
            baseline_mask_thresh = model_config_dict["baseline_mask_thresh"]
        else:
            baseline_image_path = None
            baseline_mask_thresh = None

        reward_model = load_dino_reward_model(rank=rank,
                                                batch_size=worker_actual_batch_size,
                                                model_name=model_name,
                                                image_metric=model_config_dict["image_metric"],
                                                human_seg_model_path=model_config_dict["human_seg_model_path"],
                                                pos_image_path_list=model_config_dict["pos_image_path"],
                                                neg_image_path_list=neg_image_path_list,
                                                source_mask_thresh=model_config_dict["source_mask_thresh"],
                                                target_mask_thresh=model_config_dict["target_mask_thresh"],
                                                baseline_image_path=baseline_image_path,
                                                baseline_mask_thresh=baseline_mask_thresh)

        logger.debug(f"Loaded DINO reward model. model_name={model_name}, pos_image={model_config_dict['pos_image_path']}, neg_image={neg_image_path_list}")

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
    rank0_batch_size_pct: float,
    batch_size: int, # Used to determine how the frames need to get splited up
    num_workers: int,
    worker_frames_tensor=None,
    dist=True,
) -> torch.Tensor:
    assert frames.device == torch.device("cpu")
    assert batch_size % num_workers == 0

    n_samples = len(frames)
    logger.debug(f"compute_rewards: {n_samples=}, {batch_size=}")
    rewards = torch.zeros(n_samples, device=torch.device("cpu"))
    model = model.eval()

    if not dist:

        for i in range(0, n_samples, batch_size):
            frames_batch = frames[i : i + batch_size]
            rewards_batch = compute_reward_nodist(frames_batch, model)
            rewards[i : i + batch_size] = rewards_batch

        return rewards

    if rank0_batch_size_pct < 1.0:
        rank_0_chunk_size = int(rank0_batch_size_pct * batch_size)
        remaining_size = batch_size - rank_0_chunk_size
        remaining_chunk_size = remaining_size // (num_workers - 1)
        rank_0_pad_size = remaining_chunk_size - rank_0_chunk_size
        
        total_batch_size = batch_size
    else:
        total_batch_size = batch_size // num_workers

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            frames_batch = frames[i : i + batch_size]
            print(f'--------shape: {frames_batch.shape}-----------')
            rewards_batch = dist_worker_compute_reward(
                rank=0,
                rank0_batch_size_pct=rank0_batch_size_pct,
                reward_model=model,
                render_dim=frames_batch.shape[1:],
                total_batch_size=total_batch_size,
                num_workers=num_workers,
                frames=frames_batch,
                worker_frames_tensor=worker_frames_tensor,
            )
            rewards_batch = rewards_batch.cpu()
            
            logger.debug(f"{rewards_batch.size()=}, {rewards_batch=}")

            if rank0_batch_size_pct < 1.0:
                # Remove the padding
                rewards_batch = rewards_batch[rank_0_pad_size:]
                logger.debug(f"Removed padding {rewards_batch.size()=}, {rewards_batch=}")
                
            rewards[i : i + batch_size] = rewards_batch

    
    return rewards


@overload
def dist_worker_compute_reward(
    rank: int,
    rank0_batch_size_pct: float,
    reward_model,
    render_dim: Tuple[int, int, int],
    total_batch_size: int,
    num_workers: int,
    frames: torch.Tensor,
) -> torch.Tensor:
    ...


@overload
def dist_worker_compute_reward(
    rank: int,
    rank0_batch_size_pct: float,
    reward_model,
    render_dim: Tuple[int, int, int],
    total_batch_size: int,
    num_workers: int,
    frames: None = None,
) -> None:
    ...


def dist_worker_compute_reward(
    rank: int,
    rank0_batch_size_pct: float,
    reward_model,
    render_dim: Tuple[int, int, int],
    total_batch_size: int,
    num_workers: int,
    frames=None,
    worker_frames_tensor=None,
) -> Optional[torch.Tensor]:
    logger.info(f"[Worker {rank} Computing reward...")
    if rank == 0:
        if frames is None:
            raise ValueError("Must pass render result on rank=0")


        elif rank0_batch_size_pct < 1.0:
            rank_0_chunk_size = int(rank0_batch_size_pct * total_batch_size) # .2 * 60 = 12 0 * 60 = 0

            remaining_size = total_batch_size - rank_0_chunk_size # 48  60
            remaining_chunk_size = remaining_size // (num_workers - 1) #48 60

            chunk_list = [rank_0_chunk_size] + [remaining_chunk_size] * (num_workers - 1) # [12, 48] [0, 60]

            ## only scatter if rank 0 (otherwise, tensors have already been scattered)
            scatter_list = [t.cuda(rank) for t in torch.split(frames, split_size_or_sections=chunk_list, dim=0)]
            

            rank_0_pad_size = remaining_chunk_size - rank_0_chunk_size # 48-12=26 60-0 = 60
            # Pad the left of the batch so all the chunks have the same size
            scatter_list[0] = F.pad(scatter_list[0], (0, 0, 0, 0, 0, 0, rank_0_pad_size, 0),
                                    "constant", 0)
        else:
            # Split evenly
            scatter_list = [t.cuda(rank) for t in torch.chunk(frames, num_workers, dim=0)]
    else:
        scatter_list = []

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # worker_frames_size = rank_0_chunk_size if rank == 0 else batch_size
    # logger.debug(f"[Worker {rank}] does worker_frames_tensor exist: {worker_frames_tensor.size() if worker_frames_tensor is not None else None}")
    if rank0_batch_size_pct == 1.0:
        remaining_chunk_size = total_batch_size 

    worker_frames = worker_frames_tensor if worker_frames_tensor is not None else torch.zeros((remaining_chunk_size, *render_dim), dtype=torch.uint8).cuda(rank)
    logger.debug(f"[Worker {rank}] {worker_frames.size()=}, scatter_list={[x.size() for x in scatter_list]}")
    dist.scatter(worker_frames, scatter_list=scatter_list, src=0) # this is where they get sent to other gpus
    with torch.no_grad():
        if rank == 0 and rank0_batch_size_pct == 0: # don't compute anything for device 0 if rank0 batch size is 0
            rewards = torch.zeros(rank_0_pad_size).to('cuda:0') # create empty rewards for now, but have the correct shape
        else:
            if rank == 0 and 0 < rank0_batch_size_pct < 1.0:
                worker_frames_to_compute = worker_frames[rank_0_pad_size:]
            else:
                worker_frames_to_compute = worker_frames

            start_event.record()
            # logger.debug(f"[Worker {rank}] {worker_frames_to_compute.size()=} allocated={round(torch.cuda.memory_allocated(rank)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(rank)/1024**3,1)}")
            embeddings = reward_model.embed_module(worker_frames_to_compute, reward_model.source_mask_thresh)
            end_event.record()
            # if type(embeddings) == tuple:
            #     logger.debug(f"[Worker {rank}] {embeddings[0].size()= } allocated={round(torch.cuda.memory_allocated(rank)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(rank)/1024**3,1)}")
            # else:
            #     logger.debug(f"[Worker {rank}] {embeddings.size()= } allocated={round(torch.cuda.memory_allocated(rank)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(rank)/1024**3,1)}")
            
            ### timing
            torch.cuda.synchronize()
            execution_time = start_event.elapsed_time(end_event)
            logger.info(f"Worker #{rank} - Image Embedding time: {execution_time / 1000} seconds")
            ####

            start_event.record()
            rewards = reward_model(embeddings)
            end_event.record()
            # logger.debug(f"[Worker {rank}] {rewards.size()=} allocated={round(torch.cuda.memory_allocated(rank)/1024**3,1)}, cached={round(torch.cuda.memory_reserved(rank)/1024**3,1)}")

            if rank == 0 and rank0_batch_size_pct < 1.0:
                rewards = F.pad(rewards, (rank_0_pad_size, 0))
            ### timing
            torch.cuda.synchronize()
            execution_time = start_event.elapsed_time(end_event)
            logger.info(f"Worker #{rank} - Reward Calculation time: {execution_time / 1000} seconds")
            ####

    def zero_t():
        return torch.zeros_like(rewards)

    recv_rewards = [zero_t() for _ in range(num_workers)] if rank == 0 else []
    dist.gather(rewards, gather_list=recv_rewards, dst=0)

    if rank == 0:
        consolidated_tensors = torch.cat(recv_rewards, dim=0).cuda(rank)
        return torch.cat(recv_rewards, dim=0).cuda(rank)


def compute_reward_nodist(frames, reward_model):

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start_event.record()
        embeddings = reward_model.embed_module(frames, reward_model.source_mask_thresh)
        end_event.record()
        
        ### timing
        torch.cuda.synchronize()
        execution_time = start_event.elapsed_time(end_event)
        logger.info(f"Worker - Image Embedding time: {execution_time / 1000} seconds")
        ####

        start_event.record()
        rewards = reward_model(embeddings)
        end_event.record()
        torch.cuda.synchronize()
        execution_time = start_event.elapsed_time(end_event)
        logger.info(f"Worker - Reward Calculation time: {execution_time / 1000} seconds")

    return rewards