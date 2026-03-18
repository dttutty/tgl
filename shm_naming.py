import time

import torch


def init_run_id(local_rank: int, src_rank: int = 0, provided_run_id: str | None = None) -> str:
    run_id = [None]
    if local_rank == src_rank:
        if provided_run_id:
            run_id[0] = provided_run_id
        else:
            run_id[0] = f"run_id_{time.time_ns()}"
    torch.distributed.broadcast_object_list(run_id, src=src_rank)
    return run_id[0]


def build_shm_namer(run_id: str):
    def shm_name(name: str) -> str:
        return f"{run_id}_{name}"

    return shm_name
