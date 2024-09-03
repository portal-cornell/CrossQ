


import torch
import torch.distributed as dist
from typing import List

from vlm_reward.reward_models import RewardModel

def init_distributed(num_gpu_workers):
    """Initialize the distributed process group."""
    dist.init_process_group(backend='nccl', world_size=num_gpu_workers, rank=dist.get_rank())
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    return local_rank

def calculate_batch_sizes(frames, batch_size_percent, num_gpu_workers):
    """Calculate the batch sizes for each GPU based on the given percentage."""
    num_frames = len(frames)
    batch_sizes = [(int(num_frames * percent / 100)) for percent in batch_size_percent]
    
    # Adjust the last batch size to ensure all frames are used
    batch_sizes[-1] += num_frames - sum(batch_sizes)
    
    return batch_sizes

def scatter_data(frames, batch_size_percents: List[float], num_gpu_workers: int):
    """Scatter frames data across GPUs based on batch size percentage."""
    batch_sizes = calculate_batch_sizes(frames, batch_size_percents, num_gpu_workers)
    frames_chunks = torch.split(frames, batch_sizes)

    
    local_frames = torch.zeros_like(frames if frames_chunks is None else frames_chunks[0]).to(local_rank)
    dist.scatter(tensor=local_frames, scatter_list=frames_chunks if local_rank == 0 else [], src=0)
    return local_frames

def compute_local_rewards(model, local_frames, local_rank):
    """Compute rewards on the local GPU."""
    model.to(local_rank)
    
    with torch.no_grad():
        local_rewards = model(local_frames)
    return local_rewards

def gather_rewards(local_rewards, num_gpu_workers):
    """Gather rewards from all GPUs to the main process."""
    gathered_rewards = [torch.zeros_like(local_rewards) for _ in range(num_gpu_workers)]
    dist.all_gather(gathered_rewards, local_rewards)
    
    if dist.get_rank() == 0:
        all_rewards = torch.cat(gathered_rewards, dim=0)
        return all_rewards
    return None

def cleanup_distributed():
    """Clean up the distributed process group."""
    dist.destroy_process_group()

def compute_rewards(model, frames, num_gpu_workers, rank0_batch_size_percent=1.0):
    """Main function to compute rewards using distributed GPUs with batch size percentage."""    
    rank_nonzero_batch_size_percents = [(1 - rank0_batch_size_percent) / (num_gpu_workers - 1) for i in range(num_gpu_workers - 1)]
    batch_size_percents = [rank0_batch_size_percent] + rank_nonzero_batch_size_percents
    
    local_frames = scatter_data(frames, batch_size_percents, num_gpu_workers)
    
    local_rewards = compute_local_rewards(model, local_frames, local_rank)
    
    all_rewards = gather_rewards(local_rewards, num_gpu_workers)
    
    cleanup_distributed()
    
    return all_rewards