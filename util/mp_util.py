import os
import platform
import torch

ROOT_PROC_RANK = 0

global_mp_device = None

def init(rank, num_procs, device, master_port):
    global global_mp_device

    assert(global_mp_device is None)
    global_mp_device = device

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    print("Using master port: {:d}".format(master_port))

    if (device == "cpu"):
        backend = "gloo"
    elif ("cuda" in device):
        backend = "nccl"
    else:
        assert False, "Unsupported multiprocessing device {:s}".format(device)
    
    os_platform = platform.system()
    if (backend == "nccl" and os_platform == "Windows"):
        print("Pytorch doesn't support NCCL on Windows, defaulting to gloo backend")
        backend = "gloo"

    torch.distributed.init_process_group(backend, rank=rank, world_size=num_procs)

    return

def get_num_procs():
    try:
        num_procs = torch.distributed.get_world_size()
    except:
        num_procs = 1
    return num_procs

def get_proc_rank():
    try:
        proc_rank = torch.distributed.get_rank()
    except:
        proc_rank = 0
    return proc_rank

def is_root_proc():
    rank = get_proc_rank()
    return rank == ROOT_PROC_RANK

def enable_mp():
    num_procs = get_num_procs()
    return num_procs > 1

def get_device():
    return global_mp_device

def broadcast(x):
    if (enable_mp()):
        data = x.clone()
        torch.distributed.broadcast(data, src=ROOT_PROC_RANK)
    else:
        data = x
    return data

def all_gather(x):
    n = get_num_procs()
    if (enable_mp()):
        data = [torch.empty_like(x) for i in range(n)]
        torch.distributed.all_gather(data, x)
    else:
        data = [x]
    return data

def reduce_sum(x):
    return reduce_all(x, torch.distributed.ReduceOp.SUM)

def reduce_prod(x):
    return reduce_all(x, torch.distributed.ReduceOp.PROD)

def reduce_mean(x):
    n = get_num_procs()
    sum_x = reduce_sum(x)
    mean_x = sum_x / n
    return mean_x

def reduce_min(x):
    return reduce_all(x, torch.distributed.ReduceOp.MIN)

def reduce_max(x):
    return reduce_all(x, torch.distributed.ReduceOp.MAX)

def reduce_all(x, op):
    if (enable_mp()):
        is_tensor = torch.is_tensor(x)
        if (is_tensor):
            buffer = x.clone()
        else:
            buffer = torch.tensor(x, device=get_device())

        torch.distributed.all_reduce(buffer, op=op)

        if (not is_tensor):
            buffer = buffer.item()
    else:
        buffer = x

    return buffer

def reduce_inplace_sum(x):
    reduce_inplace_all(x, torch.distributed.ReduceOp.SUM)
    return

def reduce_inplace_prod(x):
    reduce_inplace_all(x, torch.distributed.ReduceOp.PROD)
    return

def reduce_inplace_mean(x):
    n = get_num_procs()
    reduce_inplace_sum(x)
    x /= n
    return

def reduce_inplace_min(x):
    reduce_inplace_all(x, torch.distributed.ReduceOp.MIN)
    return

def reduce_inplace_max(x):
    reduce_inplace_all(x, torch.distributed.ReduceOp.MAX)
    return

def reduce_inplace_all(x, op):
    if (enable_mp()):
        torch.distributed.all_reduce(x, op=op)
    return
