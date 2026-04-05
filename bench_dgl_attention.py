"""Benchmark: DGL sparse message-passing temporal attention (TGL-style).

Reproduces TGL's TransfomerAttentionLayer on synthetic DGL bipartite blocks.
Measures forward + backward latency across varying (N, S) configurations.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
import time
import json

# ── TGL-style temporal attention (copied from layers.py, non-combined branch) ──

class TimeEncode(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Linear(1, dim)
        self.w.weight = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim, dtype=np.float32)).reshape(dim, -1)
        )
        self.w.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, t):
        return torch.cos(self.w(t.reshape(-1, 1)))


class DGLTemporalAttention(nn.Module):
    """TGL's TransfomerAttentionLayer (non-combined branch)."""

    def __init__(self, dim_node, dim_edge, dim_time, dim_out, num_head, dropout=0.1):
        super().__init__()
        self.num_head = num_head
        self.dim_out = dim_out
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_time = dim_time
        self.dropout = nn.Dropout(dropout)
        self.att_dropout = nn.Dropout(dropout)
        self.att_act = nn.LeakyReLU(0.2)
        self.time_enc = TimeEncode(dim_time) if dim_time > 0 else None
        self.w_q = nn.Linear(dim_node + dim_time, dim_out)
        self.w_k = nn.Linear(dim_node + dim_edge + dim_time, dim_out)
        self.w_v = nn.Linear(dim_node + dim_edge + dim_time, dim_out)
        self.w_out = nn.Linear(dim_node + dim_out, dim_out)
        self.layer_norm = nn.LayerNorm(dim_out)

    def forward(self, b):
        if b.num_edges() == 0:
            return torch.zeros((b.num_dst_nodes(), self.dim_out), device=b.device)
        device = b.device
        if self.dim_time > 0:
            time_feat = self.time_enc(b.edata['dt'])
            zero_time_feat = self.time_enc(torch.zeros(b.num_dst_nodes(), dtype=torch.float32, device=device))

        # Q from dst nodes, K/V from src (neighbor) nodes
        Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[b.edges()[1]]
        K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
        V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))

        Q = Q.reshape(Q.shape[0], self.num_head, -1)
        K = K.reshape(K.shape[0], self.num_head, -1)
        V = V.reshape(V.shape[0], self.num_head, -1)

        att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q * K, dim=2)))
        att = self.att_dropout(att)
        V = (V * att[:, :, None]).reshape(V.shape[0], -1)

        b.srcdata['v'] = torch.cat([torch.zeros((b.num_dst_nodes(), V.shape[1]), device=device), V], dim=0)
        b.update_all(fn.copy_u('v', 'm'), fn.sum('m', 'h'))

        rst = torch.cat([b.dstdata['h'], b.srcdata['h'][:b.num_dst_nodes()]], dim=1)
        rst = self.w_out(rst)
        rst = F.relu(self.dropout(rst))
        return self.layer_norm(rst)


def make_dgl_block(N, S, dim_node, dim_edge, device):
    """Build a DGL bipartite block mimicking TGL's MFG structure.

    dst nodes: [0, N)  (root/target nodes)
    src nodes: [0, N + N*S)  where [0, N) are dst copies and [N, N+N*S) are neighbors
    Each dst node i has S incoming edges from src nodes [N + i*S, N + (i+1)*S).
    """
    num_src = N + N * S  # dst copies + neighbor nodes
    num_dst = N

    # Build edge lists: each dst node i connects to S unique src neighbor nodes
    src_ids = []
    dst_ids = []
    for i in range(N):
        for j in range(S):
            src_ids.append(N + i * S + j)  # neighbor node index in src space
            dst_ids.append(i)
    src_ids = torch.tensor(src_ids, dtype=torch.int64, device=device)
    dst_ids = torch.tensor(dst_ids, dtype=torch.int64, device=device)

    block = dgl.create_block(
        (src_ids, dst_ids),
        num_src_nodes=num_src,
        num_dst_nodes=num_dst,
    ).to(device)

    # Node features: src includes dst copies (first N) + neighbor features
    block.srcdata['h'] = torch.randn(num_src, dim_node, device=device)
    # Edge features + time deltas
    block.edata['f'] = torch.randn(block.num_edges(), dim_edge, device=device)
    block.edata['dt'] = torch.rand(block.num_edges(), device=device)

    return block


def benchmark(N, S, dim_node, dim_edge, dim_time, dim_out, num_head,
              warmup=10, repeat=100, device='cuda'):
    model = DGLTemporalAttention(dim_node, dim_edge, dim_time, dim_out, num_head).to(device)
    model.train()

    # Pre-build blocks for benchmark
    blocks = [make_dgl_block(N, S, dim_node, dim_edge, device) for _ in range(repeat + warmup)]

    # Warmup
    for i in range(warmup):
        b = blocks[i]
        out = model(b)
        loss = out.sum()
        loss.backward()
    torch.cuda.synchronize()

    # Timed runs
    fwd_times = []
    bwd_times = []
    for i in range(repeat):
        b = blocks[warmup + i]
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        out = model(b)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        fwd_times.append(t1 - t0)
        bwd_times.append(t2 - t1)

    fwd_ms = np.array(fwd_times) * 1000
    bwd_ms = np.array(bwd_times) * 1000
    total_ms = fwd_ms + bwd_ms
    return {
        'N': N, 'S': S,
        'fwd_ms': float(np.mean(fwd_ms)),
        'bwd_ms': float(np.mean(bwd_ms)),
        'total_ms': float(np.mean(total_ms)),
        'fwd_std': float(np.std(fwd_ms)),
        'bwd_std': float(np.std(bwd_ms)),
        'total_std': float(np.std(total_ms)),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dim_node', type=int, default=172)
    parser.add_argument('--dim_edge', type=int, default=172)
    parser.add_argument('--dim_time', type=int, default=100)
    parser.add_argument('--dim_out', type=int, default=100)
    parser.add_argument('--num_head', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeat', type=int, default=100)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    configs = [
        (200, 10), (200, 20), (200, 30),
        (600, 10), (600, 20), (600, 30),
        (1000, 10), (1000, 20), (1000, 30),
    ]

    print(f"{'='*70}")
    print(f"DGL Sparse Message-Passing Temporal Attention Benchmark")
    print(f"dim_node={args.dim_node} dim_edge={args.dim_edge} dim_time={args.dim_time} dim_out={args.dim_out} heads={args.num_head}")
    print(f"{'='*70}")
    print(f"{'N':>6} {'S':>4} | {'Fwd (ms)':>10} {'Bwd (ms)':>10} {'Total (ms)':>12} {'± std':>8}")
    print(f"{'-'*70}")

    results = []
    for N, S in configs:
        r = benchmark(N, S, args.dim_node, args.dim_edge, args.dim_time,
                       args.dim_out, args.num_head, args.warmup, args.repeat, device)
        results.append(r)
        print(f"{r['N']:>6} {r['S']:>4} | {r['fwd_ms']:>10.3f} {r['bwd_ms']:>10.3f} {r['total_ms']:>12.3f} {r['total_std']:>8.3f}")

    with open('bench_dgl_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to bench_dgl_results.json")
