import os
import torch
import torch.distributed as dist

def dist_init(file_path):
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id%num_gpus)
    method = "file://" + file_path
    dist.init_process_group("nccl", init_method=method,
                            rank=proc_id, world_size=ntasks)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, proc_id%num_gpus
