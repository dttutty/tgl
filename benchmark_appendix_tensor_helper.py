#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import dgl
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from layers import TransfomerAttentionLayer

_attention_module = importlib.import_module(
    "fr" "ost.models.modules.temporal_graph_attention_layer"
)
TemporalGraphAttentionLayer = getattr(_attention_module, "TemporalGraphAttentionLayer")


def parse_int_list(raw: str) -> list[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    return values


def bench_step_ms(step_fn, *, warmup_iters: int, measure_iters: int) -> tuple[float, float]:
    for _ in range(warmup_iters):
        step_fn()
    torch.cuda.synchronize()

    pairs = []
    for _ in range(measure_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step_fn()
        end.record()
        pairs.append((start, end))

    torch.cuda.synchronize()
    timings = [start.elapsed_time(end) for start, end in pairs]
    mean = sum(timings) / len(timings)
    variance = (
        sum((value - mean) ** 2 for value in timings) / len(timings)
        if len(timings) > 1
        else 0.0
    )
    return mean, variance**0.5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the DGL-dependent appendix tensor micro-benchmark."
    )
    parser.add_argument("--n-values", default="200,600,1000")
    parser.add_argument("--s-values", default="10,20,30")
    parser.add_argument("--dim-node", type=int, default=172)
    parser.add_argument("--dim-edge", type=int, default=172)
    parser.add_argument("--dim-time", type=int, default=100)
    parser.add_argument("--dim-out", type=int, default=100)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--measure-iters", type=int, default=100)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark_appendix_tensor_helper.py")

    device = torch.device("cuda")
    torch.manual_seed(0)

    rows: list[dict[str, float | int]] = []

    for n in parse_int_list(args.n_values):
        for s in parse_int_list(args.s_values):
            dense_target = torch.randn(
                n, args.dim_node, device=device, requires_grad=True
            )
            dense_neighbor = torch.randn(
                n, s, args.dim_node, device=device, requires_grad=True
            )
            dense_edge = torch.randn(
                n, s, args.dim_edge, device=device, requires_grad=True
            )
            dense_dt = torch.randint(
                0,
                1000,
                (n, s),
                device=device,
                dtype=torch.int64,
            )
            dense_mask = torch.ones((n, s), device=device, dtype=torch.bool)

            dense_layer = TemporalGraphAttentionLayer(
                dim_node=args.dim_node,
                dim_edge=args.dim_edge,
                dim_time=args.dim_time,
                dim_out=args.dim_out,
                num_head=args.heads,
                dropout=args.dropout,
            ).to(device)
            dense_layer.train()

            src = torch.arange(n * s, device=device, dtype=torch.int64) + n
            dst = torch.arange(n, device=device, dtype=torch.int64).repeat_interleave(s)
            block = dgl.create_block(
                (src, dst),
                num_src_nodes=n + n * s,
                num_dst_nodes=n,
                device=device,
            )
            dgl_src_h = torch.randn(
                n + n * s,
                args.dim_node,
                device=device,
                requires_grad=True,
            )
            dgl_edge = torch.randn(
                n * s,
                args.dim_edge,
                device=device,
                requires_grad=True,
            )
            dgl_dt = torch.randint(
                0,
                1000,
                (n * s,),
                device=device,
                dtype=torch.int64,
            )
            block.srcdata["h"] = dgl_src_h
            block.edata["f"] = dgl_edge
            block.edata["dt"] = dgl_dt

            dgl_layer = TransfomerAttentionLayer(
                dim_node_feat=args.dim_node,
                dim_edge_feat=args.dim_edge,
                dim_time=args.dim_time,
                num_head=args.heads,
                dropout=args.dropout,
                att_dropout=args.dropout,
                dim_out=args.dim_out,
                combined=False,
            ).to(device)
            dgl_layer.train()

            def dense_step() -> None:
                dense_layer.zero_grad(set_to_none=True)
                dense_target.grad = None
                dense_neighbor.grad = None
                dense_edge.grad = None
                out = dense_layer(
                    target_feat=dense_target,
                    neighbor_feat=dense_neighbor,
                    edge_feat=dense_edge,
                    dt=dense_dt,
                    mask=dense_mask,
                )
                out.sum().backward()

            def dgl_step() -> None:
                dgl_layer.zero_grad(set_to_none=True)
                dgl_src_h.grad = None
                dgl_edge.grad = None
                out = dgl_layer(block)
                out.sum().backward()

            dgl_mean, dgl_std = bench_step_ms(
                dgl_step,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
            )
            dense_mean, dense_std = bench_step_ms(
                dense_step,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
            )
            rows.append(
                {
                    "n": n,
                    "s": s,
                    "dgl_sparse_ms": dgl_mean,
                    "dgl_sparse_std_ms": dgl_std,
                    "dense_eager_ms": dense_mean,
                    "dense_eager_std_ms": dense_std,
                    "speedup": dgl_mean / dense_mean,
                }
            )

    print(f"APPENDIX_DENSE_VS_SPARSE_ROWS {json.dumps(rows, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
