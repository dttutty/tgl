import argparse
import json
import os
import subprocess
from collections import deque
from dataclasses import dataclass, field

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="dataset name")
parser.add_argument("--config", type=str, help="path to config file")
parser.add_argument("--gpu", type=str, default="0", help="which GPU to use")
parser.add_argument("--model_name", type=str, default="", help="name of stored model")
parser.add_argument("--use_inductive", action="store_true")
parser.add_argument(
    "--seed",
    nargs="?",
    type=int,
    const=42,
    default=None,
    help="set random seed; defaults to 42 if the flag is provided without a value",
)
parser.add_argument(
    "--rand_edge_features", type=int, default=0, help="use random edge featrues"
)
parser.add_argument(
    "--rand_node_features", type=int, default=0, help="use random node featrues"
)
parser.add_argument(
    "--eval_neg_samples",
    type=int,
    default=1,
    help="how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!",
)
parser.add_argument(
    "--pin_memory",
    action="store_true",
    default=False,
    help="use pinned memory buffers for faster CPU->GPU feature transfer",
)
parser.add_argument(
    "--memory_update_delay_batches",
    type=int,
    default=0,
    help="delay mailbox/memory update by N minibatches during training",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=None,
    help="Batch size; if not set, use dataset default",
)
parser.add_argument(
    "--appendix-train-e2e",
    action="store_true",
    default=False,
    help="emit start-to-start train throughput summary for the measured window",
)
parser.add_argument(
    "--measure-start-epoch",
    type=int,
    default=0,
    help="first epoch whose train batches are eligible for throughput measurement",
)
parser.add_argument(
    "--warmup-batches",
    type=int,
    default=5,
    help="number of initial measured-train intervals to discard as warmup",
)
parser.add_argument(
    "--measure-batches",
    type=int,
    default=100,
    help="number of post-warmup train intervals to retain in the summary",
)
parser.add_argument(
    "--max-train-steps",
    type=int,
    default=0,
    help="stop training after this many train minibatches (0 disables early stop)",
)
args = parser.parse_args()

if args.memory_update_delay_batches < 0:
    raise ValueError("--memory_update_delay_batches must be >= 0")
if args.warmup_batches < 0:
    raise ValueError("--warmup-batches must be >= 0")
if args.measure_batches < 1:
    raise ValueError("--measure-batches must be >= 1")
if args.max_train_steps < 0:
    raise ValueError("--max-train-steps must be >= 0")

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import time
import dgl
import numpy as np
from modules import *
from sampler import *
from utils import *
from sklearn.metrics import average_precision_score, roc_auc_score
from frost_data import FrostBatchNegLinkSampler

# Auto set batch size from dataset defaults if not specified
if args.batch_size is None:
    dataset_defaults_sh = "/home/sqp17/Projects/frost/DATA/dataset_defaults.sh"
    if os.path.exists(dataset_defaults_sh):
        result = subprocess.run(
            [
                "bash",
                "-c",
                f'source "{dataset_defaults_sh}" && default_macro_batch_size "{args.data}"',
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            args.batch_size = int(result.stdout.strip())
            print(f"Using dataset default batch size: {args.batch_size}")
        else:
            args.batch_size = 4000
            print(
                f"Could not read dataset defaults, using default batch size: {args.batch_size}"
            )
    else:
        args.batch_size = 4000
        print(
            f"Dataset defaults file not found, using default batch size: {args.batch_size}"
        )


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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

        sync_cuda()
        skipped_intervals = min(
            self.warmup_batches,
            len(self.wall_ms),
            len(self.gpu_pairs),
        )
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


if args.seed is not None:
    set_seed(args.seed)

sample_param, memory_param, gnn_param, train_param = parse_config(args.config)

# Sync batch size: if config doesn't specify or user explicitly set via command line
if "batch_size" not in train_param or args.batch_size is not None:
    train_param["batch_size"] = args.batch_size

print_run_configuration(
    args,
    sample_param,
    memory_param,
    gnn_param,
    train_param,
    config_path=args.config,
)
node_feats, edge_feats = load_feat(
    args.data, args.rand_edge_features, args.rand_node_features
)
g, df = load_graph(args.data)
train_edge_end = df[df["default_split"].gt(0)].index[0]
val_edge_end = df[df["default_split"].gt(1)].index[0]


def get_inductive_links(df, train_edge_end, val_edge_end):
    train_df = df[:train_edge_end]
    test_df = df[val_edge_end:]

    total_node_set = set(np.unique(np.hstack([df["src"].values, df["dst"].values])))
    train_node_set = set(
        np.unique(np.hstack([train_df["src"].values, train_df["dst"].values]))
    )
    new_node_set = total_node_set - train_node_set

    del total_node_set, train_node_set

    inductive_inds = []
    for index, (_, row) in enumerate(test_df.iterrows()):
        if row.src in new_node_set or row.dst in new_node_set:
            inductive_inds.append(val_edge_end + index)

    print("Inductive links", len(inductive_inds), len(test_df))
    return [i for i in range(val_edge_end)] + inductive_inds


if args.use_inductive:
    inductive_inds = get_inductive_links(df, train_edge_end, val_edge_end)
    df = df.iloc[inductive_inds]

gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
combine_first = False
if "combine_neighs" in train_param and train_param["combine_neighs"]:
    combine_first = True
model = GeneralModel(
    gnn_dim_node,
    gnn_dim_edge,
    sample_param,
    memory_param,
    gnn_param,
    train_param,
    combined=combine_first,
).cuda()
mailbox = (
    MailBox(memory_param, g["indptr"].shape[0] - 1, gnn_dim_edge)
    if memory_param["type"] != "none"
    else None
)
creterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=train_param["lr"])
lr_scheduler = create_train_lr_scheduler(
    optimizer, train_param, default_monitor="val_ap"
)
sampler = None
if not ("no_sample" in sample_param and sample_param["no_sample"]):
    sampler = ParallelSampler(
        g["indptr"],
        g["indices"],
        g["eid"],
        g["ts"].astype(np.int64),
        sample_param["num_thread"],
        1,
        sample_param["layer"],
        sample_param["neighbor"],
        sample_param["strategy"] == "recent",
        sample_param["prop_time"],
        sample_param["history"],
        int(sample_param["duration"]),
    )

if args.use_inductive:
    test_df = df[val_edge_end:]
    inductive_nodes = set(test_df.src.values).union(test_df.src.values)
    print("inductive nodes", len(inductive_nodes))
    neg_link_sampler = NegLinkInductiveSampler(inductive_nodes)
else:
    neg_link_sampler = FrostBatchNegLinkSampler(
        dataset=args.data,
        n_nodes=g["indptr"].shape[0] - 1,
    )

if args.pin_memory:
    pinned_nfeat_buffs, pinned_efeat_buffs = get_pinned_buffers(
        sample_param, train_param["batch_size"], node_feats, edge_feats
    )
    if mailbox is not None:
        mailbox.allocate_pinned_memory_buffers(sample_param, train_param["batch_size"])
else:
    pinned_nfeat_buffs, pinned_efeat_buffs = None, None


def eval(mode="val"):
    neg_samples = 1
    model.eval()
    aps = list()
    aucs_mrrs = list()
    if mode == "val":
        eval_df = df[train_edge_end:val_edge_end]
    elif mode == "test":
        eval_df = df[val_edge_end:]
        neg_samples = args.eval_neg_samples
    elif mode == "train":
        eval_df = df[:train_edge_end]
    with torch.no_grad():
        total_loss = 0
        for _, rows in eval_df.groupby(eval_df.index // train_param["batch_size"]):
            batch_src = rows.src.values
            batch_eid = (
                rows["eid"].to_numpy(dtype=np.int64, copy=False)
                if "eid" in rows.columns
                else rows.index.to_numpy(dtype=np.int64, copy=False)
            )
            # Note: FrostBatchNegLinkSampler only supports 1 negative sample per edge
            neg_dst = neg_link_sampler.sample(batch_src, batch_eid)
            if neg_samples > 1:
                neg_dst = np.tile(neg_dst, neg_samples)
            root_nodes = np.concatenate(
                [
                    rows.src.values,
                    rows.dst.values,
                    neg_dst,
                ]
            ).astype(np.int32)
            ts = np.tile(rows.time.values, neg_samples + 2).astype(np.int64)
            if sampler is not None:
                if "no_neg" in sample_param and sample_param["no_neg"]:
                    pos_root_end = len(rows) * 2
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
            if gnn_param["arch"] != "identity":
                mfgs = to_dgl_blocks(ret, sample_param["history"], cuda=False)
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts, cuda=False)
            if args.pin_memory:
                nids, eids = get_ids(mfgs, node_feats, edge_feats)
            mfgs = mfgs_to_cuda(mfgs)
            prepare_input(
                mfgs,
                node_feats,
                edge_feats,
                combine_first=combine_first,
                pinned=args.pin_memory,
                nfeat_buffs=pinned_nfeat_buffs,
                efeat_buffs=pinned_efeat_buffs,
                nids=nids if args.pin_memory else None,
                eids=eids if args.pin_memory else None,
            )
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0], use_pinned_buffers=args.pin_memory)
            pred_pos, pred_neg = model(mfgs, neg_samples=neg_samples)
            total_loss += creterion(pred_pos, torch.ones_like(pred_pos))
            total_loss += creterion(pred_neg, torch.zeros_like(pred_neg))
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat(
                [torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0
            )
            aps.append(average_precision_score(y_true, y_pred))
            if neg_samples > 1:
                aucs_mrrs.append(
                    torch.reciprocal(
                        torch.sum(
                            pred_pos.squeeze()
                            < pred_neg.squeeze().reshape(neg_samples, -1),
                            dim=0,
                        )
                        + 1
                    ).type(torch.float)
                )
            else:
                aucs_mrrs.append(roc_auc_score(y_true, y_pred))
            if mailbox is not None:
                eid = (
                    rows["eid"].to_numpy(dtype=np.int64, copy=False)
                    if "eid" in rows.columns
                    else rows.index.to_numpy(dtype=np.int64, copy=False)
                )
                mem_edge_feats = (
                    gather_feature_rows(edge_feats, eid)
                    if edge_feats is not None
                    else None
                )
                block = None
                if memory_param["deliver_to"] == "neighbors":
                    block = to_dgl_blocks(ret, sample_param["history"], reverse=True)[
                        0
                    ][0]
                mailbox.update_mailbox(
                    model.memory_updater.last_updated_nid,
                    model.memory_updater.last_updated_memory,
                    root_nodes,
                    ts,
                    mem_edge_feats,
                    block,
                    neg_samples=neg_samples,
                    peer_memory=getattr(model, "last_mail_peer_memory", None),
                )
                mailbox.update_memory(
                    model.memory_updater.last_updated_nid,
                    model.memory_updater.last_updated_memory,
                    root_nodes,
                    model.memory_updater.last_updated_ts,
                    neg_samples=neg_samples,
                )
        if mode == "val":
            val_losses.append(float(total_loss))
    ap = float(torch.tensor(aps).mean())
    if neg_samples > 1:
        auc_mrr = float(torch.cat(aucs_mrrs).mean())
    else:
        auc_mrr = float(torch.tensor(aucs_mrrs).mean())
    return ap, auc_mrr


if not os.path.isdir("models"):
    os.mkdir("models")
if args.model_name == "":
    path_saver = "models/{}_{}.pkl".format(args.data, time.time())
else:
    path_saver = "models/{}.pkl".format(args.model_name)
best_ap = 0
best_e = 0
val_losses = list()
avg_time_sample = 0.0
avg_time_fetch_feature = 0.0
avg_time_fetch_memory = 0.0
avg_time_forward = 0.0
avg_time_backward = 0.0
avg_time_memory_update = 0.0
avg_time_tot = 0.0
appendix_bench = AppendixTrainE2EBenchmark(
    enabled=args.appendix_train_e2e,
    system="tgl_serial",
    rank=0,
    compile_enabled=False,
    batch_size=train_param["batch_size"],
    measure_start_epoch=args.measure_start_epoch,
    warmup_batches=args.warmup_batches,
    measure_batches=args.measure_batches,
)
global_train_steps = 0
early_stop = False
if mailbox is not None and args.memory_update_delay_batches > 0:
    print(
        "Delayed memory update enabled: {} minibatch(es)".format(
            args.memory_update_delay_batches
        )
    )

for e in range(train_param["epoch"]):
    print("Epoch {:d}:".format(e))
    time_sample = 0.0
    time_fetch_feature = 0.0
    time_fetch_memory = 0.0
    time_forward = 0.0
    time_backward = 0.0
    time_memory_update = 0.0
    time_tot = 0.0
    total_loss = 0.0
    appendix_bench.reset_chain()
    # training
    model.train()
    if sampler is not None:
        sampler.reset()
    if mailbox is not None:
        mailbox.reset()
        model.memory_updater.last_updated_nid = None
    memory_update_queue = deque()

    def flush_memory_update(task):
        if mailbox is None:
            return
        if task["nid"] is None or task["memory"] is None or task["updated_ts"] is None:
            return
        mailbox.update_mailbox(
            task["nid"],
            task["memory"],
            task["root_nodes"],
            task["ts"],
            task["mem_edge_feats"],
            task["block"],
            peer_memory=task.get("peer_memory"),
        )
        mailbox.update_memory(
            task["nid"], task["memory"], task["root_nodes"], task["updated_ts"]
        )

    for _, rows in df[:train_edge_end].groupby(
        df[:train_edge_end].index // train_param["batch_size"]
    ):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record(torch.cuda.current_stream())
        appendix_bench.on_train_batch_start(
            epoch=e,
            ev=ev,
            wall_time_s=time.perf_counter(),
        )
        t_tot_s = time.time()
        batch_src = rows.src.values
        batch_eid = (
            rows["eid"].to_numpy(dtype=np.int64, copy=False)
            if "eid" in rows.columns
            else rows.index.to_numpy(dtype=np.int64, copy=False)
        )
        root_nodes = np.concatenate(
            [batch_src, rows.dst.values, neg_link_sampler.sample(batch_src, batch_eid)]
        ).astype(np.int32)
        ts = np.concatenate(
            [rows.time.values, rows.time.values, rows.time.values]
        ).astype(np.int64)
        if sampler is not None:
            t_sample_s = time.perf_counter()
            if "no_neg" in sample_param and sample_param["no_neg"]:
                pos_root_end = root_nodes.shape[0] * 2 // 3
                sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
            else:
                sampler.sample(root_nodes, ts)
            ret = sampler.get_ret()
            time_sample += time.perf_counter() - t_sample_s
        t_fetch_feature_s = time.perf_counter()
        if gnn_param["arch"] != "identity":
            mfgs = to_dgl_blocks(ret, sample_param["history"], cuda=False)
        else:
            mfgs = node_to_dgl_blocks(root_nodes, ts, cuda=False)
        if args.pin_memory:
            nids, eids = get_ids(mfgs, node_feats, edge_feats)
        mfgs = mfgs_to_cuda(mfgs)
        prepare_input(
            mfgs,
            node_feats,
            edge_feats,
            combine_first=combine_first,
            pinned=args.pin_memory,
            nfeat_buffs=pinned_nfeat_buffs,
            efeat_buffs=pinned_efeat_buffs,
            nids=nids if args.pin_memory else None,
            eids=eids if args.pin_memory else None,
        )
        sync_cuda()
        time_fetch_feature += time.perf_counter() - t_fetch_feature_s
        if mailbox is not None:
            t_fetch_memory_s = time.perf_counter()
            mailbox.prep_input_mails(mfgs[0], use_pinned_buffers=args.pin_memory)
            sync_cuda()
            time_fetch_memory += time.perf_counter() - t_fetch_memory_s
        optimizer.zero_grad()
        sync_cuda()
        t_forward_s = time.perf_counter()
        pred_pos, pred_neg = model(mfgs)
        sync_cuda()
        time_forward += time.perf_counter() - t_forward_s
        loss = creterion(pred_pos, torch.ones_like(pred_pos))
        loss += creterion(pred_neg, torch.zeros_like(pred_neg))
        total_loss += float(loss) * train_param["batch_size"]
        sync_cuda()
        t_backward_s = time.perf_counter()
        loss.backward()
        optimizer.step()
        sync_cuda()
        time_backward += time.perf_counter() - t_backward_s
        if mailbox is not None:
            t_memory_update_s = time.perf_counter()
            eid = (
                rows["eid"].to_numpy(dtype=np.int64, copy=False)
                if "eid" in rows.columns
                else rows.index.to_numpy(dtype=np.int64, copy=False)
            )
            mem_edge_feats = (
                gather_feature_rows(edge_feats, eid) if edge_feats is not None else None
            )
            block = None
            if memory_param["deliver_to"] == "neighbors":
                block = to_dgl_blocks(ret, sample_param["history"], reverse=True)[0][0]

            memory_update_queue.append(
                {
                    "nid": model.memory_updater.last_updated_nid.detach().clone()
                    if model.memory_updater.last_updated_nid is not None
                    else None,
                    "memory": model.memory_updater.last_updated_memory.detach().clone()
                    if model.memory_updater.last_updated_memory is not None
                    else None,
                    "updated_ts": model.memory_updater.last_updated_ts.detach().clone()
                    if model.memory_updater.last_updated_ts is not None
                    else None,
                    "root_nodes": root_nodes.copy(),
                    "ts": ts.copy(),
                    "mem_edge_feats": mem_edge_feats.detach().clone()
                    if mem_edge_feats is not None
                    else None,
                    "block": block,
                    "peer_memory": model.last_mail_peer_memory.detach().clone()
                    if getattr(model, "last_mail_peer_memory", None) is not None
                    else None,
                }
            )

            if len(memory_update_queue) > args.memory_update_delay_batches:
                flush_memory_update(memory_update_queue.popleft())

            sync_cuda()
            time_memory_update += time.perf_counter() - t_memory_update_s
        appendix_bench.record_memory_snapshot()
        time_tot += time.time() - t_tot_s
        global_train_steps += 1
        if args.max_train_steps > 0 and global_train_steps >= args.max_train_steps:
            print(
                f"Reached max_train_steps={args.max_train_steps}; "
                "stopping training loop early for throughput measurement."
            )
            early_stop = True
            break

    if mailbox is not None and len(memory_update_queue) > 0:
        t_memory_update_s = time.perf_counter()
        while len(memory_update_queue) > 0:
            flush_memory_update(memory_update_queue.popleft())
        sync_cuda()
        time_memory_update += time.perf_counter() - t_memory_update_s

    if early_stop:
        appendix_bench.emit_summary()
        raise SystemExit(0)

    ap, auc = eval("val")
    test_ap, test_auc = eval("test")
    scheduler_metrics = {
        "train_loss": total_loss,
        "val_loss": val_losses[-1] if val_losses else None,
        "val_ap": ap,
        "val_auc": auc,
        "test_ap": test_ap,
        "test_score": test_auc,
    }
    if args.eval_neg_samples > 1:
        scheduler_metrics["test_mrr"] = test_auc
    else:
        scheduler_metrics["test_auc"] = test_auc
    lr_scheduler_step = step_train_lr_scheduler(lr_scheduler, scheduler_metrics)

    if e > 2 and ap > best_ap:
        best_e = e
        best_ap = ap
        best_test_ap = test_ap
        best_test_auc = test_auc
    test_metric_name = "test mrr" if args.eval_neg_samples > 1 else "test auc"
    print(
        "\ttrain loss:{:.4f}  val ap:{:4f}  val auc:{:4f}  test ap:{:4f}  {}:{:4f}".format(
            total_loss,
            ap,
            auc,
            test_ap,
            test_metric_name,
            test_auc,
        )
    )
    if lr_scheduler_step is not None:
        print("\t{}".format(format_lr_scheduler_step(lr_scheduler_step)))
    print(
        "\ttotal time:{:.2f}s sample:{:.2f}s fetch feature:{:.2f}s fetch memory:{:.2f}s forward:{:.2f}s backward:{:.2f}s memory update:{:.2f}s".format(
            time_tot,
            time_sample,
            time_fetch_feature,
            time_fetch_memory,
            time_forward,
            time_backward,
            time_memory_update,
        )
    )
    avg_time_tot += time_tot
    avg_time_sample += time_sample
    avg_time_fetch_feature += time_fetch_feature
    avg_time_fetch_memory += time_fetch_memory
    avg_time_forward += time_forward
    avg_time_backward += time_backward
    avg_time_memory_update += time_memory_update

n_epochs = train_param["epoch"]
print("\nAverage over {} epochs:".format(n_epochs))
print(
    "\ttotal time:{:.2f}s sample:{:.2f}s fetch feature:{:.2f}s fetch memory:{:.2f}s forward:{:.2f}s backward:{:.2f}s memory update:{:.2f}s".format(
        avg_time_tot / n_epochs,
        avg_time_sample / n_epochs,
        avg_time_fetch_feature / n_epochs,
        avg_time_fetch_memory / n_epochs,
        avg_time_forward / n_epochs,
        avg_time_backward / n_epochs,
        avg_time_memory_update / n_epochs,
    )
)

print("Best model at epoch {} had val AP: {:.4f}".format(best_e, best_ap))
if args.eval_neg_samples > 1:
    print("\ttest AP:{:4f}  test MRR:{:4f}".format(best_test_ap, best_test_auc))
else:
    print("\ttest AP:{:4f}  test AUC:{:4f}".format(best_test_ap, best_test_auc))
