import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import torch.nn.functional as F
import argparse


parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
parser.add_argument('--use_inductive', action='store_true')
parser.add_argument('--rand_edge_features', type=int, default=0, help='use random edge featrues')
parser.add_argument('--rand_node_features', type=int, default=0, help='use random node featrues')
parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')
args=parser.parse_args()


def load_graph(d):
    df = pd.read_csv(f'/home/sqp17/Projects/tgl/DATA/{d}/edges.csv')
    g = np.load(f'/home/sqp17/Projects/tgl/DATA/{d}/ext_full.npz')
    # assumes g contains arrays 'row', 'col', 'ts'
    return g, df

def load_feat(d, rand_de=0, rand_dn=0):
    node_feats = None
    if os.path.exists('DATA/{}/node_features.pt'.format(d)):
        node_feats = torch.load('DATA/{}/node_features.pt'.format(d))
        if node_feats.dtype == torch.bool:
            node_feats = node_feats.type(torch.float32)
    edge_feats = None
    if os.path.exists('DATA/{}/edge_features.pt'.format(d)):
        edge_feats = torch.load('DATA/{}/edge_features.pt'.format(d))
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.float32)
    if rand_de > 0:
        if d == 'LASTFM':
            edge_feats = torch.randn(1293103, rand_de)
        elif d == 'MOOC':
            edge_feats = torch.randn(411749, rand_de)
    if rand_dn > 0:
        if d == 'LASTFM':
            node_feats = torch.randn(1980, rand_dn)
        elif d == 'MOOC':
            edge_feats = torch.randn(7144, rand_dn)
    return node_feats, edge_feats
  
def get_inductive_links(df, train_edge_end, val_edge_end):
    train_df = df[:train_edge_end]
    test_df = df[val_edge_end:]
    
    total_node_set = set(np.unique(np.hstack([df['src'].values, df['dst'].values])))
    train_node_set = set(np.unique(np.hstack([train_df['src'].values, train_df['dst'].values])))
    new_node_set = total_node_set - train_node_set
    
    del total_node_set, train_node_set

    inductive_inds = []
    for index, (_, row) in enumerate(test_df.iterrows()):
        if row.src in new_node_set or row.dst in new_node_set:
            inductive_inds.append(val_edge_end+index)
    
    print('Inductive links', len(inductive_inds), len(test_df))
    return [i for i in range(val_edge_end)] + inductive_inds

class LastFMTemporalDataset(Dataset):
    def __init__(self, name, rand_de=0, rand_dn=0):
        g, df = load_graph(name)
        train_edge_end = df[df['ext_roll'].gt(0)].index[0]
        val_edge_end = df[df['ext_roll'].gt(1)].index[0]
        
        if args.use_inductive:
            inductive_inds = get_inductive_links(df, train_edge_end, val_edge_end)
            df = df.iloc[inductive_inds]
            
        node_feats, edge_feats = load_feat(args.data, args.rand_edge_features, args.rand_node_features)
        gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
        gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

        combine_first = False
        if 'combine_neighs' in train_param and train_param['combine_neighs']:
            combine_first = True
    
        # extract arrays and sort by timestamp
        src = g['row']
        dst = g['col']
        ts  = g['ts']
        order = np.argsort(ts)
        self.src = src[order]
        self.dst = dst[order]
        self.ts  = ts[order].astype(np.float32)

        # load edge features (or zeros if none)
        _, edge_feats = load_feat(name, rand_de, rand_dn)
        if edge_feats is None:
            # no edge features: use zero-dim features
            self.edge_feats = np.zeros((len(self.src), 0), dtype=np.float32)
        else:
            # ensure numpy array
            self.edge_feats = edge_feats.numpy() if torch.is_tensor(edge_feats) else edge_feats

        # labels for link prediction
        self.labels = np.ones(len(self.src), dtype=np.float32)
        
    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return (
            int(self.src[idx]),
            int(self.dst[idx]),
            float(self.ts[idx]),
            self.edge_feats[idx],
            self.labels[idx]
        )

# ============== 1. Helper Modules ==============

class TimeEncoder(nn.Module):
    """
    Sinusoidal Time Encoder from the TGN paper.
    Input: (batch_size, 1) tensor of time differences
    Output: (batch_size, time_dim) encoding
    """
    def __init__(self, time_dim):
        super().__init__()
        self.time_dim = time_dim
        self.basis_freq = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim)).float()
        )
        self.phase = nn.Parameter(torch.zeros(time_dim).float())

    def forward(self, ts_diff):
        ts = ts_diff.float().to(self.basis_freq.device)
        proj = ts * self.basis_freq.view(1, -1) + self.phase.view(1, -1)
        return torch.cos(proj)


class MessageFunction(nn.Module):
    def __init__(self, memory_dim, time_dim, edge_feat_dim, message_dim):
        super().__init__()
        input_dim = 2 * memory_dim + time_dim + edge_feat_dim
        self.layer = nn.Sequential(
            nn.Linear(input_dim, message_dim),
            nn.ReLU(),
        )

    def forward(self, src_mem, dst_mem, time_enc, edge_feat):
        x = torch.cat([src_mem, dst_mem, time_enc, edge_feat], dim=1)
        return self.layer(x)


class MessageAggregator(nn.Module):
    def __init__(self, agg_method='mean'):
        super().__init__()
        if agg_method not in ['mean', 'last']:
            raise ValueError("agg_method must be 'mean' or 'last'")
        self.agg_method = agg_method

    def forward(self, node_ids, messages):
        unique_ids = torch.unique(node_ids)
        agg = []
        for nid in unique_ids:
            msgs = messages[node_ids == nid]
            if self.agg_method == 'mean':
                agg.append(msgs.mean(dim=0))
            else:
                agg.append(msgs[-1])
        if not agg:
            return unique_ids, torch.empty((0, messages.size(1)), device=messages.device)
        return unique_ids, torch.stack(agg, dim=0)


class MemoryUpdater(nn.Module):
    def __init__(self, memory_dim, message_dim, update_type='gru'):
        super().__init__()
        if update_type == 'gru':
            self.cell = nn.GRUCell(message_dim, memory_dim)
        else:
            self.cell = nn.RNNCell(message_dim, memory_dim)

    def forward(self, agg_message, prev_memory):
        return self.cell(agg_message, prev_memory)


class EmbeddingModule(nn.Module):
    def __init__(self, memory_dim, embedding_dim):
        super().__init__()
        self.project = nn.Linear(memory_dim, embedding_dim)

    def forward(self, memory):
        return self.project(memory)


# ============== 2. TGN Model ==============

class TGN(nn.Module):
    def __init__(self, num_nodes, edge_feat_dim,
                 memory_dim, time_dim, embedding_dim, message_dim,
                 agg_method='mean', mem_update_type='gru', dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes

        # state buffers (not learnable parameters)
        self.register_buffer('memory', torch.zeros(num_nodes, memory_dim))
        self.register_buffer('last_update_ts', torch.zeros(num_nodes))

        # modules
        self.time_encoder = TimeEncoder(time_dim)
        self.message_fn = MessageFunction(memory_dim, time_dim, edge_feat_dim, message_dim)
        self.aggregator = MessageAggregator(agg_method)
        self.memory_updater = MemoryUpdater(memory_dim, message_dim, mem_update_type)
        self.embedding_module = EmbeddingModule(memory_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def compute_embedding(self, node_ids):
        node_ids = node_ids.long()
        mem = self.memory[node_ids]
        return self.embedding_module(mem)

    def update_state(self, src, dst, ts, edge_feats):
        device = src.device

        # snapshot current state
        mem_snap = self.memory.clone()
        prev_src_mem = mem_snap[src]
        prev_dst_mem = mem_snap[dst]
        prev_src_ts = self.last_update_ts[src].view(-1, 1)
        prev_dst_ts = self.last_update_ts[dst].view(-1, 1)

        # time encoding
        ts = ts.float().view(-1, 1).to(device)
        tenc_src = self.time_encoder(ts - prev_src_ts)
        tenc_dst = self.time_encoder(ts - prev_dst_ts)

        # message computation
        msg_dst = self.message_fn(prev_src_mem, prev_dst_mem, tenc_dst, edge_feats)
        msg_src = self.message_fn(prev_dst_mem, prev_src_mem, tenc_src, edge_feats)
        msg_dst, msg_src = self.dropout(msg_dst), self.dropout(msg_src)

        # aggregate
        all_nodes = torch.cat([src, dst], dim=0)
        all_msgs  = torch.cat([msg_src, msg_dst], dim=0)
        update_ids, agg_msgs = self.aggregator(all_nodes, all_msgs)

        if update_ids.numel() == 0:
            return

        # update memory
        prev_mem = mem_snap[update_ids]
        new_mem = self.memory_updater(agg_msgs, prev_mem)
        new_mem = self.dropout(new_mem)

        # write back without grad
        with torch.no_grad():
            self.memory[update_ids] = new_mem

            # vectorized timestamp update using scatter_reduce (PyTorch ≥2.0)
            idx = torch.cat([src, dst], dim=0)
            ts_rep = ts.repeat(2, 1).view(-1)
            max_ts = torch.zeros_like(self.last_update_ts)
            max_ts = max_ts.scatter_reduce(0, idx, ts_rep, reduce='amax')
            self.last_update_ts = torch.maximum(self.last_update_ts, max_ts)

    def detach_memory(self):
        self.memory.detach_()
        self.last_update_ts.detach_()

    def reset_state(self):
        with torch.no_grad():
            self.memory.zero_()
            self.last_update_ts.zero_()


# ============== 3. Training Loop ==============

class DummyTemporalDataset(Dataset):
    def __init__(self, num_interactions=1000, num_nodes=100, edge_feat_dim=16):
        self.src = np.random.randint(0, num_nodes, num_interactions)
        self.dst = np.random.randint(0, num_nodes, num_interactions)
        for i in range(num_interactions):
            while self.dst[i] == self.src[i]:
                self.dst[i] = np.random.randint(0, num_nodes)
        self.ts = np.sort(np.random.rand(num_interactions) * num_interactions).astype(np.float32)
        self.edge_feats = np.random.randn(num_interactions, edge_feat_dim).astype(np.float32)
        self.labels = np.ones(num_interactions).astype(np.float32)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return (
            self.src[idx], self.dst[idx],
            self.ts[idx], self.edge_feats[idx], self.labels[idx]
        )


def collate_fn(batch):
    src, dst, ts, feats, labels = zip(*batch)
    return (
        torch.tensor(src, dtype=torch.long),
        torch.tensor(dst, dtype=torch.long),
        torch.tensor(ts, dtype=torch.float),
        torch.from_numpy(np.stack(feats, 0)),
        torch.tensor(labels, dtype=torch.float)
    )


def train_lastfm():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # instantiate model
    model = TGN(
        num_nodes=1980,            # adjust to LastFM node count
        edge_feat_dim=0,           # if edge_feats dim == 0
        memory_dim=32,
        time_dim=16,
        embedding_dim=32,
        message_dim=32
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.BCEWithLogitsLoss()

    # prepare dataset & loader
    dataset = LastFMTemporalDataset('LASTFM')
    loader  = DataLoader(
        dataset, batch_size=512, shuffle=False,
        collate_fn=collate_fn, num_workers=4
    )

    for epoch in range(5):
        model.train()
        model.reset_state()
        total_loss = 0.0

        for src, dst, ts, feats, labels in loader:
            src, dst, ts, feats, labels = [x.to(DEVICE) for x in (src, dst, ts, feats, labels)]

            optimizer.zero_grad()

            # include update_state in the graph
            model.update_state(src, dst, ts, feats)

            # compute positive embeddings
            pos_src_emb = model.compute_embedding(src)
            pos_dst_emb = model.compute_embedding(dst)

            # negative sampling
            neg_dst = torch.randint(0, model.num_nodes, (src.size(0),), device=DEVICE)
            neg_src_emb = model.compute_embedding(src)
            neg_dst_emb = model.compute_embedding(neg_dst)

            # scores & loss
            pos_score = (pos_src_emb * pos_dst_emb).sum(dim=1)
            neg_score = (neg_src_emb * neg_dst_emb).sum(dim=1)
            scores = torch.cat([pos_score, neg_score], dim=0)
            labels_cat = torch.cat([
                torch.ones_like(pos_score),
                torch.zeros_like(neg_score)
            ], dim=0)

            loss = criterion(scores, labels_cat)
            loss.backward()
            optimizer.step()

            # detach memory to avoid graph growth
            model.detach_memory()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(loader):.4f}")

    print("Training completed on LastFM dataset.")

if __name__ == "__main__":
    train_lastfm()
