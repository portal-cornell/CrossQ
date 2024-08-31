import torch.distributed as dist
import torch
import os
import argparse
from multiprocess import start_processes

"""
A sanity check for torch.distributed
"""

def main(rank, stop_event):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    print(stop_event)
    print(f'initializing + {rank}')
    dist.init_process_group("nccl", rank=rank, world_size=2)
    print(f'initialized  + {rank}')
    if rank == 0:
        recv = [torch.randn(5).cuda(rank) for _ in range(2)]
    else:
        recv = None

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    tensor = torch.ones(5).cuda(rank) * rank
    end_event.record()
    torch.cuda.synchronize()

    print(f'gathering  + {rank}')
    dist.gather(tensor, gather_list=recv, dst=0)
    print(f'gathered  + {rank}')
    print(recv)

if __name__ == "__main__":
    start_processes(main, nprocs=2)