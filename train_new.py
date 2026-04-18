import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field


start_time = time.time()

parser=argparse.ArgumentParser(
    description='Distributed TGL trainer without train-step overlap. Direct invocation auto-launches via torchrun with nproc_per_node=num_gpus+1.'
)
parser.add_argument('--dataset', type=str, help='dataset name')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument(
    '--seed', nargs='?', type=int, const=42, default=None,
    help='set random seed; defaults to 42 if the flag is provided without a value'
)
parser.add_argument('--num_gpus', type=int, default=2, help='number of gpus to use')
parser.add_argument('--omp_num_threads', type=int, default=8)
parser.add_argument('--run_id', type=str, default=None, help='shared-memory namespace prefix for this run')
parser.add_argument(
    "--rnd_edim", type=int, default=128
)  # if your dataset has no edge features, set rnd_edim > 0 to use random edge features
parser.add_argument(
    "--rnd_ndim", type=int, default=0
)  # if your dataset has no node features, set rnd_ndim > 0 to use random node features
parser.add_argument('--tqdm', action='store_true', default=False, help='enable tqdm progress bars')
parser.add_argument('--strict_avoid_rc', action='store_true', default=False, help='serialize memory/mailbox writes across ranks to avoid race conditions')
parser.add_argument(
    '--compile',
    dest='compile_model',
    action='store_true',
    default=False,
    help='Compile pure-compute submodules with torch.compile before DDP wrapping.',
)
parser.add_argument(
    '--appendix-train-e2e',
    action='store_true',
    default=False,
    help='Emit structured train-minibatch timing JSON for appendix benchmarking.',
)
parser.add_argument(
    '--measure-start-epoch',
    type=int,
    default=1,
    help='Zero-based epoch index where train-minibatch timing starts.',
)
parser.add_argument(
    '--warmup-batches',
    type=int,
    default=20,
    help='Additional train minibatches to skip after measure-start-epoch.',
)
parser.add_argument(
    '--max-train-steps',
    type=int,
    default=0,
    help='Stop after this many host-dispatched train steps (0 means disabled).',
)
parser.add_argument(
    '--measure-batches',
    type=int,
    default=0,
    help='Cap appendix interval statistics to this many post-warmup train intervals (0 means disabled).',
)
args=parser.parse_args()
if args.measure_start_epoch < 0:
    raise ValueError('--measure-start-epoch must be >= 0')
if args.warmup_batches < 0:
    raise ValueError('--warmup-batches must be >= 0')
if args.max_train_steps < 0:
    raise ValueError('--max-train-steps must be >= 0')
if args.measure_batches < 0:
    raise ValueError('--measure-batches must be >= 0')


def _launched_via_torchrun() -> bool:
    return any(name in os.environ for name in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE'))


def _relaunch_with_torchrun(num_gpus: int) -> None:
    if num_gpus < 1:
        raise RuntimeError(f'--num_gpus must be >= 1, got {num_gpus}.')

    nproc_per_node = num_gpus + 1
    cmd = [
        sys.executable,
        '-m',
        'torch.distributed.run',
        '--standalone',
        f'--nproc_per_node={nproc_per_node}',
        sys.argv[0],
        *sys.argv[1:],
    ]
    print(
        f'Auto-launching distributed runner with nproc_per_node={nproc_per_node} '
        f'for --num_gpus={num_gpus}.'
    )
    os.execvpe(sys.executable, cmd, os.environ.copy())


if not _launched_via_torchrun():
    _relaunch_with_torchrun(args.num_gpus)

args.local_rank = 0
if 'LOCAL_RANK' in os.environ:
    args.local_rank = int(os.environ['LOCAL_RANK'])
if 'WORLD_SIZE' in os.environ:
    world_size = int(os.environ['WORLD_SIZE'])
    expected_world_size = args.num_gpus + 1
    if world_size != expected_world_size:
        raise RuntimeError(
            f'train_dist.py expects WORLD_SIZE={expected_world_size} for --num_gpus={args.num_gpus}, '
            f'but got WORLD_SIZE={world_size}. Launch with '
            f'`python -m torch.distributed.run --nproc_per_node={expected_world_size} ...` '
            'or run this script directly and let it auto-launch.'
        )

# assign GPUs from externally provided CUDA_VISIBLE_DEVICES
visible_devices_raw = os.environ.get('CUDA_VISIBLE_DEVICES', '')
visible_devices = [d.strip() for d in visible_devices_raw.split(',') if d.strip() != '']

if not visible_devices:
    # Default to the first `num_gpus` physical devices when the launcher did
    # not constrain visibility explicitly.
    visible_devices = [str(i) for i in range(args.num_gpus)]

if visible_devices_raw and args.num_gpus > 0 and len(visible_devices) < args.num_gpus:
    raise RuntimeError(
        f"Not enough CUDA devices in CUDA_VISIBLE_DEVICES='{visible_devices_raw}'. "
        f"Need at least {args.num_gpus}, but got {len(visible_devices)}."
    )

if args.local_rank < args.num_gpus:
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices[args.local_rank]
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = str(args.omp_num_threads)
os.environ['MKL_NUM_THREADS'] = str(args.omp_num_threads)

import torch
import dgl
import datetime
import math
import threading
import numpy as np
from tqdm import tqdm
from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array
from sklearn.metrics import average_precision_score, roc_auc_score
from modules import *
from sampler import *
from utils import *
from df_split import make_balance_plan
from shm_naming import init_run_id, build_shm_namer
import torch.distributed as dist # 确保已有或添加
from torch.profiler import profile, record_function, ProfilerActivity # 新增
from frost_data import (
    FrostBatchNegLinkSampler,
    compute_split_boundaries,
    load_dataset_counts,
    load_edges_df,
    load_feat as load_frost_feat,
    load_graph as load_frost_graph,
)


@dataclass(slots=True)
class AppendixTrainE2EBenchmark:
    enabled: bool
    system: str
    rank: int
    compile_enabled: bool
    batch_size: int | None
    measure_start_epoch: int
    warmup_batches: int
    measure_batches: int
    gpu_pairs: list[tuple[object, object]] = field(default_factory=list)
    wall_ms: list[float] = field(default_factory=list)
    reserved_mib: list[float] = field(default_factory=list)
    allocated_mib: list[float] = field(default_factory=list)
    train_batches_seen: int = 0
    _last_ev: object | None = None
    _last_wall_s: float | None = None

    def reset_chain(self) -> None:
        self._last_ev = None
        self._last_wall_s = None

    def on_train_batch_start(
        self,
        *,
        epoch: int,
        ev: object,
        wall_time_s: float,
    ) -> None:
        if not self.enabled:
            return
        if epoch < self.measure_start_epoch:
            self.reset_chain()
            return
        if self._last_ev is not None:
            self.gpu_pairs.append((self._last_ev, ev))
        if self._last_wall_s is not None:
            self.wall_ms.append((wall_time_s - self._last_wall_s) * 1000.0)
        self._last_ev = ev
        self._last_wall_s = wall_time_s
        self.train_batches_seen += 1

    def record_memory_snapshot(self) -> None:
        if not self.enabled or not torch.cuda.is_available():
            return
        self.reserved_mib.append(torch.cuda.memory_reserved() / (1024**2))
        self.allocated_mib.append(torch.cuda.memory_allocated() / (1024**2))

    def emit_summary(self) -> None:
        if not self.enabled:
            return

        torch.cuda.synchronize()
        skipped_intervals = min(self.warmup_batches, len(self.wall_ms), len(self.gpu_pairs))
        gpu_pairs = self.gpu_pairs[skipped_intervals:]
        wall_ms = self.wall_ms[skipped_intervals:]
        if self.measure_batches > 0:
            gpu_pairs = gpu_pairs[: self.measure_batches]
            wall_ms = wall_ms[: self.measure_batches]
        gpu_ms = [ev_prev.elapsed_time(ev_curr) for (ev_prev, ev_curr) in gpu_pairs]
        payload: dict[str, int | float | bool | str] = {
            "system": self.system,
            "rank": self.rank,
            "compile": self.compile_enabled,
            "measure_start_epoch": self.measure_start_epoch,
            "warmup_batches": self.warmup_batches,
            "measure_batches_target": self.measure_batches,
            "train_batches_seen": self.train_batches_seen,
            "interval_count_raw": len(self.wall_ms),
            "interval_count": len(wall_ms),
            "skipped_intervals": skipped_intervals,
        }
        if wall_ms:
            ordered_wall = sorted(wall_ms)
            n = len(ordered_wall)
            mean_wall = sum(ordered_wall) / n
            var_wall = (
                sum((value - mean_wall) ** 2 for value in ordered_wall) / n
                if n > 1
                else 0.0
            )
            payload.update(
                {
                    "avg_wall_ms": mean_wall,
                    "std_wall_ms": var_wall**0.5,
                    "p50_wall_ms": ordered_wall[int(0.50 * (n - 1))],
                    "p90_wall_ms": ordered_wall[int(0.90 * (n - 1))],
                    "p99_wall_ms": ordered_wall[int(0.99 * (n - 1))],
                }
            )
            if self.batch_size is not None and self.batch_size > 0 and mean_wall > 0:
                payload["events_per_sec_per_gpu"] = (
                    float(self.batch_size) * 1000.0 / mean_wall
                )
        if gpu_ms:
            ordered_gpu = sorted(gpu_ms)
            n_gpu = len(ordered_gpu)
            mean_gpu = sum(ordered_gpu) / n_gpu
            var_gpu = (
                sum((value - mean_gpu) ** 2 for value in ordered_gpu) / n_gpu
                if n_gpu > 1
                else 0.0
            )
            payload.update(
                {
                    "avg_gpu_ms": mean_gpu,
                    "std_gpu_ms": var_gpu**0.5,
                    "p50_gpu_ms": ordered_gpu[int(0.50 * (n_gpu - 1))],
                    "p90_gpu_ms": ordered_gpu[int(0.90 * (n_gpu - 1))],
                    "p99_gpu_ms": ordered_gpu[int(0.99 * (n_gpu - 1))],
                }
            )
        skipped_memory = min(self.warmup_batches, len(self.reserved_mib), len(self.allocated_mib))
        reserved_mib = self.reserved_mib[skipped_memory:]
        allocated_mib = self.allocated_mib[skipped_memory:]
        if self.measure_batches > 0:
            reserved_mib = reserved_mib[: self.measure_batches]
            allocated_mib = allocated_mib[: self.measure_batches]
        payload["memory_measured_batches"] = len(reserved_mib)
        if reserved_mib:
            ordered_reserved = sorted(reserved_mib)
            n_reserved = len(ordered_reserved)
            payload.update(
                {
                    "reserved_mib_mean": sum(ordered_reserved) / n_reserved,
                    "reserved_mib_p50": ordered_reserved[int(0.50 * (n_reserved - 1))],
                    "reserved_mib_p90": ordered_reserved[int(0.90 * (n_reserved - 1))],
                    "reserved_mib_max": ordered_reserved[-1],
                }
            )
        if allocated_mib:
            ordered_allocated = sorted(allocated_mib)
            n_allocated = len(ordered_allocated)
            payload.update(
                {
                    "allocated_mib_mean": sum(ordered_allocated) / n_allocated,
                    "allocated_mib_p50": ordered_allocated[int(0.50 * (n_allocated - 1))],
                    "allocated_mib_p90": ordered_allocated[int(0.90 * (n_allocated - 1))],
                    "allocated_mib_max": ordered_allocated[-1],
                }
            )
        print(f"APPENDIX_TRAIN_E2E_SUMMARY {json.dumps(payload, sort_keys=True)}")


def compile_tgl_model_submodules(model: torch.nn.Module) -> list[str]:
    compiled: list[str] = []
    if hasattr(model, 'memory_updater'):
        model.memory_updater = torch.compile(model.memory_updater)
        compiled.append('memory_updater')
    if hasattr(model, 'layers'):
        for name, layer in list(model.layers.items()):
            model.layers[name] = torch.compile(layer)
            compiled.append(f'layers.{name}')
    if hasattr(model, 'edge_predictor'):
        model.edge_predictor = torch.compile(model.edge_predictor)
        compiled.append('edge_predictor')
    if hasattr(model, 'combiner'):
        model.combiner = torch.compile(model.combiner)
        compiled.append('combiner')
    return compiled

if args.seed is not None:
    set_seed(args.seed)

sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
print_run_configuration(
    args,
    sample_param,
    memory_param,
    gnn_param,
    train_param,
    config_path=args.config,
    rank=args.local_rank,
)

torch.distributed.init_process_group(backend='gloo', timeout=datetime.timedelta(0, 3600))

run_id = init_run_id(args.local_rank, src_rank=0, provided_run_id=args.run_id)
shm_name = build_shm_namer(run_id)

nccl_group = None
if args.local_rank < args.num_gpus:
    nccl_group = torch.distributed.new_group(
        ranks=list(range(args.num_gpus)), backend='nccl'
    )

if args.local_rank == 0:
    _num_nodes, _num_edges = load_dataset_counts(args.dataset)
    _node_feats, _edge_feats = load_frost_feat(
        args.dataset,
        args.rnd_edim,
        args.rnd_ndim,
        num_nodes=_num_nodes,
        num_edges=_num_edges,
    )
dim_feats = [0, 0, 0, 0, 0, 0]
if args.local_rank == 0:
    if _node_feats is not None:
        _node_feats = _node_feats.float()
        dim_feats[0] = _node_feats.shape[0]
        dim_feats[1] = _node_feats.shape[1]
        dim_feats[2] = _node_feats.dtype
        node_feats = create_shared_mem_array(shm_name('node_feats'), _node_feats.shape, dtype=_node_feats.dtype)
        node_feats.copy_(_node_feats)
        del _node_feats
    else:
        node_feats = None
    if _edge_feats is not None:
        _edge_feats = _edge_feats.float()
        dim_feats[3] = _edge_feats.shape[0]
        dim_feats[4] = _edge_feats.shape[1]
        dim_feats[5] = _edge_feats.dtype
        edge_feats = create_shared_mem_array(shm_name('edge_feats'), _edge_feats.shape, dtype=_edge_feats.dtype)
        edge_feats.copy_(_edge_feats)
        del _edge_feats
    else: 
        edge_feats = None
torch.distributed.barrier()
torch.distributed.broadcast_object_list(dim_feats, src=0)
if args.local_rank > 0 and args.local_rank < args.num_gpus:
    node_feats = None
    edge_feats = None
    if dim_feats[0] > 0:
        node_feats = get_shared_mem_array(shm_name('node_feats'), (dim_feats[0], dim_feats[1]), dtype=dim_feats[2])
    if dim_feats[3] > 0:
        edge_feats = get_shared_mem_array(shm_name('edge_feats'), (dim_feats[3], dim_feats[4]), dtype=dim_feats[5])
orig_batch_size = train_param['batch_size']
if args.local_rank == 0:
    if not os.path.isdir('models'):
        os.mkdir('models')
    path_saver = ['models/{}_{}.pkl'.format(args.dataset, time.time())]
else:
    path_saver = [None]
torch.distributed.broadcast_object_list(path_saver, src=0)
path_saver = path_saver[0]

if args.local_rank == args.num_gpus:
    g = load_frost_graph(args.dataset)
    df = load_edges_df(args.dataset)
    num_nodes = [g['indptr'].shape[0] - 1]
else:
    num_nodes = [None]
torch.distributed.barrier()
torch.distributed.broadcast_object_list(num_nodes, src=args.num_gpus)
num_nodes = num_nodes[0]

mailbox = None
if memory_param['type'] != 'none':
    if args.local_rank == 0:
        node_memory = create_shared_mem_array(shm_name('node_memory'), torch.Size([num_nodes, memory_param['dim_out']]), dtype=torch.float32)
        node_memory_ts = create_shared_mem_array(shm_name('node_memory_ts'), torch.Size([num_nodes]), dtype=torch.int64)
        mails = create_shared_mem_array(shm_name('mails'), torch.Size([num_nodes, memory_param['mailbox_size'], 2 * memory_param['dim_out'] + dim_feats[4]]), dtype=torch.float32)
        mail_ts = create_shared_mem_array(shm_name('mails_ts'), torch.Size([num_nodes, memory_param['mailbox_size']]), dtype=torch.int64)
        next_mail_pos = create_shared_mem_array(shm_name('next_mail_pos'), torch.Size([num_nodes]), dtype=torch.long)
        update_mail_pos = create_shared_mem_array(shm_name('update_mail_pos'), torch.Size([num_nodes]), dtype=torch.int32)
        torch.distributed.barrier()
        node_memory.zero_()
        node_memory_ts.zero_()
        mails.zero_()
        mail_ts.zero_()
        next_mail_pos.zero_()
        update_mail_pos.zero_()
    else:
        torch.distributed.barrier()
        node_memory = get_shared_mem_array(shm_name('node_memory'), torch.Size([num_nodes, memory_param['dim_out']]), dtype=torch.float32)
        node_memory_ts = get_shared_mem_array(shm_name('node_memory_ts'), torch.Size([num_nodes]), dtype=torch.int64)
        mails = get_shared_mem_array(shm_name('mails'), torch.Size([num_nodes, memory_param['mailbox_size'], 2 * memory_param['dim_out'] + dim_feats[4]]), dtype=torch.float32)
        mail_ts = get_shared_mem_array(shm_name('mails_ts'), torch.Size([num_nodes, memory_param['mailbox_size']]), dtype=torch.int64)
        next_mail_pos = get_shared_mem_array(shm_name('next_mail_pos'), torch.Size([num_nodes]), dtype=torch.long)
        update_mail_pos = get_shared_mem_array(shm_name('update_mail_pos'), torch.Size([num_nodes]), dtype=torch.int32)
    mailbox = MailBox(memory_param, num_nodes, dim_feats[4], node_memory, node_memory_ts, mails, mail_ts, next_mail_pos, update_mail_pos, strict_avoid_rc=args.strict_avoid_rc, rank=args.local_rank, num_gpus=args.num_gpus, nccl_group=nccl_group)

class DataPipelineThread(threading.Thread):
    
    def __init__(self, my_mfgs, my_root, my_ts, my_eid, my_block, stream):
        super(DataPipelineThread, self).__init__()
        self.my_mfgs = my_mfgs
        self.my_root = my_root
        self.my_ts = my_ts
        self.my_eid = my_eid
        self.my_block = my_block
        self.stream = stream
        self.mfgs = None
        self.root = None
        self.ts = None
        self.eid = None
        self.block = None

    def run(self):
        with torch.cuda.stream(self.stream):
            # print(args.local_rank, 'start thread')
            nids, eids = get_ids(self.my_mfgs[0], node_feats, edge_feats)
            mfgs = mfgs_to_cuda(self.my_mfgs[0])
            prepare_input(mfgs, node_feats, edge_feats, pinned=True, nfeat_buffs=pinned_nfeat_buffs, efeat_buffs=pinned_efeat_buffs, nids=nids, eids=eids)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0], use_pinned_buffers=True)
                if memory_param['deliver_to'] == 'neighbors':
                    self.block = self.my_block[0]
            self.mfgs = mfgs
            self.root = self.my_root[0]
            self.ts = self.my_ts[0]
            self.eid = self.my_eid[0]
            # print(args.local_rank, 'finished')

    def get_stream(self):
        return self.stream

    def get_mfgs(self):
        return self.mfgs
    
    def get_root(self):
        return self.root
    
    def get_ts(self):
        return self.ts

    def get_eid(self):
        return self.eid

    def get_block(self):
        return self.block

start_time = time.time()

if args.local_rank < args.num_gpus:
    torch.cuda.set_device(0)
    # GPU worker process
    model = GeneralModel(dim_feats[1], dim_feats[4], sample_param, memory_param, gnn_param, train_param).cuda()
    compiled_submodules = []
    if args.compile_model:
        compiled_submodules = compile_tgl_model_submodules(model)
        print(
            f"[Rank {args.local_rank}] torch.compile enabled for submodules: "
            + ", ".join(compiled_submodules)
        )
    
    def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        return total, trainable
    
    total_params, trainable_params = count_parameters(model)
    print(f"[Rank {args.local_rank}] Model parameters: total={total_params} trainable={trainable_params}")
    find_unused_parameters = True if sample_param['history'] > 1 else False
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0], process_group=nccl_group, output_device=0, find_unused_parameters=find_unused_parameters)
    creterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])
    lr_scheduler = create_train_lr_scheduler(optimizer, train_param, default_monitor='val_ap')
    pinned_nfeat_buffs, pinned_efeat_buffs = get_pinned_buffers(sample_param, train_param['batch_size'], node_feats, edge_feats)
    if mailbox is not None:
        mailbox.allocate_pinned_memory_buffers(sample_param, train_param['batch_size'])
    tot_loss = 0
    prev_thread = None
    # Train start->start intervals (excluding val/test) for fair comparison
    train_s2s_pairs = []   # list[(cuda_event_prev, cuda_event_curr)] for GPU-stream intervals
    train_s2s_wall = []    # list[float] wall-clock intervals in seconds
    _last_train_ev = None
    _last_train_wall = None
    WARMUP_INTERVALS = 3   # drop first N intervals to skip CUDA JIT / allocator warmup
    
    
    
    # ================= [新增代码 START] =================
    # 计数器初始化
    cur_epoch = 0
    cur_step = 0
    target_profile_epoch = 1  # 第2个epoch (索引从0开始)
    profile_steps = 10        # 统计10个minibatch
    appendix_bench = AppendixTrainE2EBenchmark(
        enabled=args.appendix_train_e2e,
        system='tgl',
        rank=args.local_rank,
        compile_enabled=args.compile_model,
        batch_size=train_param['batch_size'],
        measure_start_epoch=args.measure_start_epoch,
        warmup_batches=args.warmup_batches,
        measure_batches=args.measure_batches,
    )
    
    
    def trace_handler(p):
        # 保存文件名带上 rank 和 epoch
        output_filename = f"trace_epoch{target_profile_epoch}_rank_{args.local_rank}.json"
        p.export_chrome_trace(output_filename)
        print(f"[Rank {args.local_rank}] Trace saved to {output_filename}")

    # 初始化 Profiler (schedule=None 表示手动控制)
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=None, 
        on_trace_ready=trace_handler,
        record_shapes=True,
        with_stack=True,
        with_flops=False,
    )

    def _record_train_start():
        global _last_train_ev, _last_train_wall
        ev = torch.cuda.Event(enable_timing=True)
        ev.record(torch.cuda.current_stream())
        now_wall = time.perf_counter()

        if _last_train_ev is not None:
            train_s2s_pairs.append((_last_train_ev, ev))
        if _last_train_wall is not None:
            train_s2s_wall.append(now_wall - _last_train_wall)

        _last_train_ev = ev
        _last_train_wall = now_wall
        appendix_bench.on_train_batch_start(
            epoch=cur_epoch,
            ev=ev,
            wall_time_s=now_wall,
        )

    def _reset_train_chain():
        global _last_train_ev, _last_train_wall
        _last_train_ev = None
        _last_train_wall = None
        appendix_bench.reset_chain()
    while True:
        my_model_state = [None]
        model_state = [None] * (args.num_gpus + 1)
        torch.distributed.scatter_object_list(my_model_state, model_state, src=args.num_gpus)
        if my_model_state[0] == -1:
            if len(train_s2s_pairs) >= 1:
                torch.cuda.synchronize()
                dt_ms = [
                    ev_prev.elapsed_time(ev_curr)
                    for (ev_prev, ev_curr) in train_s2s_pairs
                ][WARMUP_INTERVALS:]

                if dt_ms:
                    dt_ms_sorted = sorted(dt_ms)
                    n = len(dt_ms_sorted)
                    p50 = dt_ms_sorted[int(0.50*(n-1))]
                    p90 = dt_ms_sorted[int(0.90*(n-1))]
                    p99 = dt_ms_sorted[int(0.99*(n-1))]
                    avg = sum(dt_ms_sorted) / n

                    print(f"[rank {args.local_rank}] minibatch start->start (GPU stream): "
                          f"n={n} avg={avg:.3f}ms p50={p50:.3f}ms p90={p90:.3f}ms p99={p99:.3f}ms")

            # wall-clock summary (same boundaries as GPU-stream metric)
            if len(train_s2s_wall) >= 1:
                dt_s = list(train_s2s_wall)[WARMUP_INTERVALS:]

                if dt_s:
                    dt_s_sorted = sorted(dt_s)
                    n = len(dt_s_sorted)
                    p50 = dt_s_sorted[int(0.50*(n-1))]
                    p90 = dt_s_sorted[int(0.90*(n-1))]
                    p99 = dt_s_sorted[int(0.99*(n-1))]
                    avg = sum(dt_s_sorted) / n

                    print(f"[rank {args.local_rank}] minibatch start->start (wall): "
                          f"n={n} avg={avg*1000:.3f}ms p50={p50*1000:.3f}ms p90={p90*1000:.3f}ms p99={p99*1000:.3f}ms")
            appendix_bench.emit_summary()
            # --------------------------------------
            break
        elif my_model_state[0] == 4:
            continue
        elif my_model_state[0] == 2:
            torch.save(model.state_dict(), path_saver)
            continue
        elif my_model_state[0] == 3:
            model.load_state_dict(torch.load(path_saver, map_location=torch.device('cuda:0')))
            continue
        elif my_model_state[0] == 5:
            torch.distributed.gather_object(float(tot_loss), None, dst=args.num_gpus)
            tot_loss = 0
            
            # ================= [新增代码 START] =================
            print(f"[Rank {args.local_rank}] Finished Epoch {cur_epoch}, steps: {cur_step}")
            cur_epoch += 1 # 进入下一个 epoch
            cur_step = 0   # 重置 step
            # ================= [新增代码 END] =================
            
            
            continue
        elif my_model_state[0] == 6:
            scheduler_metrics = [None]
            torch.distributed.broadcast_object_list(scheduler_metrics, src=args.num_gpus)
            lr_scheduler_step = step_train_lr_scheduler(lr_scheduler, scheduler_metrics[0])
            if lr_scheduler_step is not None and args.local_rank == 0:
                print('\t{}'.format(format_lr_scheduler_step(lr_scheduler_step)))
            continue
        elif my_model_state[0] == 0:
            # The host is allowed to sample/scatter the next batch early, but
            # no worker may call mailbox.prep_input_mails() for batch n+1 until
            # every rank has finished update_mailbox()/update_memory() for batch
            # n. This barrier sits immediately before prep_input_mails() so the
            # dynamic state consumed by batch n+1 is globally fresh.
            my_mfgs = [None]
            multi_mfgs = [None] * (args.num_gpus + 1)
            my_root = [None]
            multi_root = [None] * (args.num_gpus + 1)
            my_ts = [None]
            multi_ts = [None] * (args.num_gpus + 1)
            my_eid = [None]
            multi_eid = [None] * (args.num_gpus + 1)
            my_block = [None]
            multi_block = [None] * (args.num_gpus + 1)
            torch.distributed.scatter_object_list(my_mfgs, multi_mfgs, src=args.num_gpus)
            if mailbox is not None:
                torch.distributed.scatter_object_list(my_root, multi_root, src=args.num_gpus)
                torch.distributed.scatter_object_list(my_ts, multi_ts, src=args.num_gpus)
                torch.distributed.scatter_object_list(my_eid, multi_eid, src=args.num_gpus)
                if memory_param['deliver_to'] == 'neighbors':
                    torch.distributed.scatter_object_list(my_block, multi_block, src=args.num_gpus)

            mfgs = mfgs_to_cuda(my_mfgs[0])
            prepare_input(
                mfgs,
                node_feats,
                edge_feats,
                pinned=True,
                nfeat_buffs=pinned_nfeat_buffs,
                efeat_buffs=pinned_efeat_buffs,
            )
            root_nodes = None
            ts = None
            eid = None
            block = None
            if mailbox is not None:
                # Cross-rank freshness gate: all ranks must have completed the
                # previous batch's mailbox/memory write-back before any rank can
                # read mailbox/memory for the next batch.
                torch.distributed.barrier()
                mailbox.prep_input_mails(mfgs[0], use_pinned_buffers=True)
                root_nodes = my_root[0]
                ts = my_ts[0]
                eid = my_eid[0]
                if memory_param['deliver_to'] == 'neighbors':
                    block = my_block[0]

            _record_train_start()
            should_prof = False

            if should_prof == True and cur_epoch == target_profile_epoch:
                if cur_step == 0:
                    prof.start()
                    print(f"[Rank {args.local_rank}] Profiler STARTED at epoch {cur_epoch}")
                if cur_step < profile_steps:
                    prof.step()

            model.train()
            optimizer.zero_grad()
            with record_function("model_step"):
                pred_pos, pred_neg = model(mfgs)
                loss = creterion(pred_pos, torch.ones_like(pred_pos))
                loss += creterion(pred_neg, torch.zeros_like(pred_neg))
                loss.backward()
                optimizer.step()

            if should_prof == True and cur_epoch == target_profile_epoch:
                if cur_step == profile_steps - 1:
                    prof.stop()
                    print(f"[Rank {args.local_rank}] Profiler STOPPED after {profile_steps} steps")

            cur_step += 1
            with torch.no_grad():
                tot_loss += float(loss)
            if mailbox is not None:
                with torch.no_grad():
                    mem_edge_feats = gather_feature_rows(edge_feats, eid) if edge_feats is not None else None
                    mailbox.update_mailbox(
                        model.module.memory_updater.last_updated_nid,
                        model.module.memory_updater.last_updated_memory,
                        root_nodes,
                        ts,
                        mem_edge_feats,
                        block,
                        peer_memory=getattr(model.module, 'last_mail_peer_memory', None),
                    )
                    mailbox.update_memory(
                        model.module.memory_updater.last_updated_nid,
                        model.module.memory_updater.last_updated_memory,
                        root_nodes,
                        model.module.memory_updater.last_updated_ts,
                    )
                    if memory_param['deliver_to'] == 'neighbors':
                        torch.distributed.barrier(group=nccl_group)
                        if args.local_rank == 0:
                            mailbox.update_next_mail_pos()
            appendix_bench.record_memory_snapshot()
        elif my_model_state[0] == 1:
            if prev_thread is not None:
                # finish last training mini-batch
                prev_thread.join()
                _record_train_start()
                mfgs = prev_thread.get_mfgs()
                model.train()
                optimizer.zero_grad()
                pred_pos, pred_neg = model(mfgs)
                loss = creterion(pred_pos, torch.ones_like(pred_pos))
                loss += creterion(pred_neg, torch.zeros_like(pred_neg))
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    tot_loss += float(loss)
                if mailbox is not None:
                    with torch.no_grad():
                        eid = prev_thread.get_eid()
                        mem_edge_feats = gather_feature_rows(edge_feats, eid) if edge_feats is not None else None
                        root_nodes = prev_thread.get_root()
                        ts = prev_thread.get_ts()
                        block = prev_thread.get_block()
                        mailbox.update_mailbox(
                            model.module.memory_updater.last_updated_nid,
                            model.module.memory_updater.last_updated_memory,
                            root_nodes,
                            ts,
                            mem_edge_feats,
                            block,
                            peer_memory=getattr(model.module, 'last_mail_peer_memory', None),
                        )
                        mailbox.update_memory(model.module.memory_updater.last_updated_nid, model.module.memory_updater.last_updated_memory, root_nodes, model.module.memory_updater.last_updated_ts)
                        if memory_param['deliver_to'] == 'neighbors':
                            torch.distributed.barrier(group=nccl_group)
                            if args.local_rank == 0:
                                mailbox.update_next_mail_pos()
                appendix_bench.record_memory_snapshot()
                prev_thread = None
            # Cut the train start->start chain so val/test time is excluded
            _reset_train_chain()
            my_mfgs = [None]
            multi_mfgs = [None] * (args.num_gpus + 1)
            torch.distributed.scatter_object_list(my_mfgs, multi_mfgs, src=args.num_gpus)
            mfgs = mfgs_to_cuda(my_mfgs[0])
            prepare_input(mfgs, node_feats, edge_feats, pinned=True, nfeat_buffs=pinned_nfeat_buffs, efeat_buffs=pinned_efeat_buffs)
            model.eval()
            with torch.no_grad():
                if mailbox is not None:
                    mailbox.prep_input_mails(mfgs[0], use_pinned_buffers=True)
                pred_pos, pred_neg = model(mfgs)
                if mailbox is not None:
                    my_root = [None]
                    multi_root = [None] * (args.num_gpus + 1)
                    my_ts = [None]
                    multi_ts = [None] * (args.num_gpus + 1)
                    my_eid = [None]
                    multi_eid = [None] * (args.num_gpus + 1)
                    torch.distributed.scatter_object_list(my_root, multi_root, src=args.num_gpus)
                    torch.distributed.scatter_object_list(my_ts, multi_ts, src=args.num_gpus)
                    torch.distributed.scatter_object_list(my_eid, multi_eid, src=args.num_gpus)
                    eid = my_eid[0]
                    mem_edge_feats = gather_feature_rows(edge_feats, eid) if edge_feats is not None else None
                    root_nodes = my_root[0]
                    ts = my_ts[0]
                    block = None
                    if memory_param['deliver_to'] == 'neighbors':
                        my_block = [None]
                        multi_block = [None] * (args.num_gpus + 1)
                        torch.distributed.scatter_object_list(my_block, multi_block, src=args.num_gpus)
                        block = my_block[0]
                    mailbox.update_mailbox(
                        model.module.memory_updater.last_updated_nid,
                        model.module.memory_updater.last_updated_memory,
                        root_nodes,
                        ts,
                        mem_edge_feats,
                        block,
                        peer_memory=getattr(model.module, 'last_mail_peer_memory', None),
                    )
                    mailbox.update_memory(model.module.memory_updater.last_updated_nid, model.module.memory_updater.last_updated_memory, root_nodes, model.module.memory_updater.last_updated_ts)
                    if memory_param['deliver_to'] == 'neighbors':
                        torch.distributed.barrier(group=nccl_group)
                        if args.local_rank == 0:
                            mailbox.update_next_mail_pos()
                y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
                y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)

                # ... (前向传播和计算逻辑不变)

                
                # 1. 计算局部指标 (为了复现作者的“错误”算法)
                ap = average_precision_score(y_true, y_pred)
                auc = roc_auc_score(y_true, y_pred)
                

                
                # 3. [新增] 发送原始预测值和标签 (为了计算“正确”算法)
                # 转为 numpy 以减小序列化开销，且 sklearn 需要 numpy
                torch.distributed.gather_object((y_pred.numpy(), y_true.numpy()), None, dst=args.num_gpus)
else:
    # hosting process
    train_edge_end, val_edge_end = compute_split_boundaries(df)
    sampler = None
    if not ('no_sample' in sample_param and sample_param['no_sample']):
        sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'],
                                  sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                                  sample_param['strategy']=='recent', sample_param['prop_time'],
                                  sample_param['history'], int(sample_param['duration']))
    neg_link_sampler = FrostBatchNegLinkSampler(
        dataset=args.dataset,
        n_nodes=g['indptr'].shape[0] - 1,
    )

    def eval(mode='val'):
        if mode == 'val':
            eval_df = df[train_edge_end:val_edge_end]
        elif mode == 'test':
            eval_df = df[val_edge_end:]
        elif mode == 'train':
            eval_df = df[:train_edge_end]
        raw_preds_tot = list()
        raw_trues_tot = list()
        # train_param['batch_size'] = orig_batch_size
        # Apply balanced data distribution plan
        df_out, step_ids, gpu_ids = make_balance_plan(eval_df, train_param['batch_size'], args.num_gpus)
        
        # Calculate total steps
        total_steps = max(step_ids) + 1 if step_ids else 0
        
        # Process each step (each step contains args.num_gpus batches)
        step_iter = tqdm(range(total_steps), desc='Eval batching') if args.tqdm else range(total_steps)
        for step in step_iter:
            multi_mfgs = list()
            multi_root = list()
            multi_ts = list()
            multi_eid = list()
            multi_block = list()
            
            # Process each GPU's batch in this step
            for gpu in range(args.num_gpus):
                # Get rows for this step and GPU
                mask = [(s == step and g == gpu) for s, g in zip(step_ids, gpu_ids)]
                rows = df_out[mask]
                
                if len(rows) > 0:
                    batch_src = rows.src.to_numpy(dtype=np.int64, copy=False)
                    batch_dst = rows.dst.to_numpy(dtype=np.int64, copy=False)
                    batch_eid = rows.eid.to_numpy(dtype=np.int64, copy=False)
                    batch_ts = rows.time.to_numpy(dtype=np.int64, copy=False)
                    neg_dst = neg_link_sampler.sample(batch_src, batch_eid)
                    root_nodes = np.concatenate([batch_src, batch_dst, neg_dst]).astype(np.int32)
                    ts = np.concatenate([batch_ts, batch_ts, batch_ts]).astype(np.int64)
                    if sampler is not None:
                        if 'no_neg' in sample_param and sample_param['no_neg']:
                            pos_root_end = root_nodes.shape[0] * 2 // 3
                            sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                        else:
                            sampler.sample(root_nodes, ts)
                        ret = sampler.get_ret()
                    if gnn_param['arch'] != 'identity':
                        mfgs = to_dgl_blocks(ret, sample_param['history'], cuda=False)
                    else:
                        mfgs = node_to_dgl_blocks(root_nodes, ts, cuda=False)
                    multi_mfgs.append(mfgs)
                    multi_root.append(root_nodes)
                    multi_ts.append(ts)
                    multi_eid.append(batch_eid)
                    if mailbox is not None and memory_param['deliver_to'] == 'neighbors':
                        multi_block.append(to_dgl_blocks(ret, sample_param['history'], reverse=True, cuda=False)[0][0])
            
            # Distribute to GPUs
            model_state = [1] * (args.num_gpus + 1)
            my_model_state = [None]
            torch.distributed.scatter_object_list(my_model_state, model_state, src=args.num_gpus)
            multi_mfgs.append(None)
            my_mfgs = [None]
            torch.distributed.scatter_object_list(my_mfgs, multi_mfgs, src=args.num_gpus)
            if mailbox is not None:
                multi_root.append(None)
                multi_ts.append(None)
                multi_eid.append(None)
                my_root = [None]
                my_ts = [None]
                my_eid = [None]
                torch.distributed.scatter_object_list(my_root, multi_root, src=args.num_gpus)
                torch.distributed.scatter_object_list(my_ts, multi_ts, src=args.num_gpus)
                torch.distributed.scatter_object_list(my_eid, multi_eid, src=args.num_gpus)
                if memory_param['deliver_to'] == 'neighbors':
                    multi_block.append(None)
                    my_block = [None]
                    torch.distributed.scatter_object_list(my_block, multi_block, src=args.num_gpus)
                        


                
                # [新增] 接收原始数据
                gathered_raw = [None] * (args.num_gpus + 1)
                # 这里的 src 不需要指定，因为大家都在 gather，主进程负责收
                torch.distributed.gather_object(None, gathered_raw, dst=args.num_gpus)
                
                # 提取并保存数据 (排除最后一个，因为那是主进程自己的占位符)
                for res in gathered_raw[:-1]:
                    if res is not None:
                        raw_preds_tot.append(res[0])
                        raw_trues_tot.append(res[1])
                

                multi_mfgs = list()
                multi_root = list()
                multi_ts = list()
                multi_eid = list()
                multi_block = list()


        
        # 2. 正确的算法 (Correct/Global)
        if len(raw_preds_tot) > 0:
            all_preds = np.concatenate(raw_preds_tot)
            all_trues = np.concatenate(raw_trues_tot)
            ap_correct = average_precision_score(all_trues, all_preds)
            auc_correct = roc_auc_score(all_trues, all_preds)
        else:
            ap_correct = 0.0
            auc_correct = 0.0

        print(f"[{mode.upper()}] Global AP: {ap_correct:.4f} AUC: {auc_correct:.4f}")
        
        # 这里你可以选择返回哪一个，通常建议返回正确的，或者两者都返回供记录
        return ap_correct, auc_correct
    best_ap = 0
    best_e = 0
    tap = 0
    tauc = 0
    global_train_steps = 0
    early_stop = False
    for e in range(train_param['epoch']):
        print('Epoch {:d}:'.format(e))
        time_sample = 0
        time_tot = 0
        if sampler is not None:
            sampler.reset()
        if mailbox is not None:
            mailbox.reset()
        # training
        # train_param['batch_size'] = orig_batch_size
        # itr_tot = train_edge_end // train_param['batch_size'] // args.num_gpus * args.num_gpus
        # train_param['batch_size'] = math.ceil(train_edge_end / itr_tot)
        
        multi_mfgs = list()
        multi_root = list()
        multi_ts = list()
        multi_eid = list()
        multi_block = list()
        train_df = df[:train_edge_end]
        
        
        # Apply balanced data distribution plan
        df_out, step_ids, gpu_ids = make_balance_plan(train_df, train_param['batch_size'], args.num_gpus)
        
        # Calculate total steps
        total_steps = max(step_ids) + 1 if step_ids else 0
        
        # Process each step (each step contains args.num_gpus batches)
        step_iter = tqdm(range(total_steps), desc='Train batching') if args.tqdm else range(total_steps)
        for step in step_iter:
            t_tot_s = time.time()
            multi_mfgs = list()
            multi_root = list()
            multi_ts = list()
            multi_eid = list()
            multi_block = list()
            
            # Process each GPU's batch in this step
            for gpu in range(args.num_gpus):
                # Get rows for this step and GPU
                mask = [(s == step and g == gpu) for s, g in zip(step_ids, gpu_ids)]
                rows = df_out[mask]
                
                if len(rows) > 0:
                    batch_src = rows.src.to_numpy(dtype=np.int64, copy=False)
                    batch_dst = rows.dst.to_numpy(dtype=np.int64, copy=False)
                    batch_eid = rows.eid.to_numpy(dtype=np.int64, copy=False)
                    batch_ts = rows.time.to_numpy(dtype=np.int64, copy=False)
                    neg_dst = neg_link_sampler.sample(batch_src, batch_eid)
                    root_nodes = np.concatenate([batch_src, batch_dst, neg_dst]).astype(np.int32)
                    ts = np.concatenate([batch_ts, batch_ts, batch_ts]).astype(np.int64)
                    if sampler is not None:
                        if 'no_neg' in sample_param and sample_param['no_neg']:
                            pos_root_end = root_nodes.shape[0] * 2 // 3
                            sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                        else:
                            sampler.sample(root_nodes, ts)
                        ret = sampler.get_ret()
                        time_sample += ret[0].sample_time()
                    if gnn_param['arch'] != 'identity':
                        mfgs = to_dgl_blocks(ret, sample_param['history'], cuda=False)
                    else:
                        mfgs = node_to_dgl_blocks(root_nodes, ts, cuda=False)
                    multi_mfgs.append(mfgs)
                    multi_root.append(root_nodes)
                    multi_ts.append(ts)
                    multi_eid.append(batch_eid)
                    if mailbox is not None and memory_param['deliver_to'] == 'neighbors':
                        multi_block.append(to_dgl_blocks(ret, sample_param['history'], reverse=True, cuda=False)[0][0])
            
            # Distribute to GPUs
            model_state = [0] * (args.num_gpus + 1)
            my_model_state = [None]
            torch.distributed.scatter_object_list(my_model_state, model_state, src=args.num_gpus)
            multi_mfgs.append(None)
            my_mfgs = [None]
            torch.distributed.scatter_object_list(my_mfgs, multi_mfgs, src=args.num_gpus)
            if mailbox is not None:
                multi_root.append(None)
                multi_ts.append(None)
                multi_eid.append(None)
                my_root = [None]
                my_ts = [None]
                my_eid = [None]
                torch.distributed.scatter_object_list(my_root, multi_root, src=args.num_gpus)
                torch.distributed.scatter_object_list(my_ts, multi_ts, src=args.num_gpus)
                torch.distributed.scatter_object_list(my_eid, multi_eid, src=args.num_gpus)
                if memory_param['deliver_to'] == 'neighbors':
                    multi_block.append(None)
                    my_block = [None]
                    torch.distributed.scatter_object_list(my_block, multi_block, src=args.num_gpus)
            # Wait only until workers have completed static H2D for this batch.
            # The worker-side barrier before mailbox.prep_input_mails() then
            # enforces the stronger guarantee we want:
            #   update_mailbox/update_memory(batch n)
            #     happens-before
            #   prep_input_mails(batch n+1)
            # while still allowing host-side sampling of batch n+1 to overlap.
            torch.distributed.barrier()
            time_tot += time.time() - t_tot_s
            time_tot += time.time() - t_tot_s
            global_train_steps += 1
            if args.max_train_steps > 0 and global_train_steps >= args.max_train_steps:
                print(
                    f"Reached max_train_steps={args.max_train_steps}; "
                    "stopping training loop early for throughput measurement."
                )
                early_stop = True
                break
        if early_stop:
            break
        model_state = [5] * (args.num_gpus + 1)
        my_model_state = [None]
        torch.distributed.scatter_object_list(my_model_state, model_state, src=args.num_gpus)
        gathered_loss = [None] * (args.num_gpus + 1)
        torch.distributed.gather_object(float(0), gathered_loss, dst=args.num_gpus)
        total_loss = np.sum(np.array(gathered_loss) * train_param['batch_size'])
        ap, auc = eval('val')
        model_state = [4] * (args.num_gpus + 1)
        model_state[0] = 2
        my_model_state = [None]
        torch.distributed.scatter_object_list(my_model_state, model_state, src=args.num_gpus)
            # for memory based models, testing after validation is faster
        tap_, tauc_ = eval('test')
        scheduler_metrics = {
            'train_loss': total_loss,
            'val_ap': ap,
            'val_auc': auc,
            'test_ap': tap_,
            'test_auc': tauc_,
            'test_score': tauc_,
        }
        if has_train_lr_scheduler(train_param):
            model_state = [6] * (args.num_gpus + 1)
            my_model_state = [None]
            torch.distributed.scatter_object_list(my_model_state, model_state, src=args.num_gpus)
            scheduler_payload = [scheduler_metrics]
            torch.distributed.broadcast_object_list(scheduler_payload, src=args.num_gpus)
        if ap > best_ap:
            best_e = e
            best_ap = ap
            tap = tap_
            tauc = tauc_
        print('\ttrain loss:{:.4f}  val ap:{:4f}  val auc:{:4f}'.format(total_loss, ap, auc))
        print('\ttotal time:{:.2f}s sample time:{:.2f}s'.format(time_tot, time_sample))

    if not early_stop:
        print('Best model at epoch {}.'.format(best_e))
        print('\ttest ap:{:4f}  test auc:{:4f}'.format(tap, tauc))
    else:
        print(
            "Skipped validation/test because run ended early at "
            f"max_train_steps={args.max_train_steps}."
        )

    # let all process exit
    model_state = [-1] * (args.num_gpus + 1)
    my_model_state = [None]
    torch.distributed.scatter_object_list(my_model_state, model_state, src=args.num_gpus)
    end = time.time()
    print(f"Elapsed time: {end - start_time:.2f} seconds")
