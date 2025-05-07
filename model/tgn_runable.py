from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader, Dataset



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
    
class LastKAggregator(nn.Module):
    def __init__(self, model_ref, k=3):
        super().__init__()
        self.model_ref = model_ref  # 需要访问 TGN 中的 mailbox
        self.k = k

    def forward(self, node_ids):
        unique_ids = torch.unique(node_ids)
        agg = []
        valid_ids = []

        for nid in unique_ids.tolist():
            mailbox = self.model_ref.mailbox.get(nid, [])
            if len(mailbox) == 0:
                continue
            msgs = list(mailbox)[-self.k:]
            msgs = torch.stack(msgs).to(self.model_ref.memory.device)
            agg.append(msgs[-1])  # last-k 中的最后一条 (你也可以改成 mean(msgs) 等策略)
            valid_ids.append(nid)

        if not agg:
            return torch.tensor([], dtype=torch.long), torch.empty(0, self.model_ref.memory.size(1)).to(self.model_ref.memory.device)
        return torch.tensor(valid_ids, dtype=torch.long).to(self.model_ref.memory.device), torch.stack(agg)



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
    def __init__(self, memory_dim, embedding_dim, node_feat_dim=0):
        super().__init__()
        self.project = nn.Linear(memory_dim + node_feat_dim, embedding_dim)

    def forward(self, memory, node_feat=None):
        if node_feat is not None:
            x = torch.cat([memory, node_feat], dim=1)
        else:
            x = memory
        return self.project(x)


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
        self.mailbox = defaultdict(list)  # 或 torch.Tensor with size [num_nodes, mailbox_size, message_dim]
        self.mailbox_size = 3


    def compute_embedding(self, node_ids, node_feat=None):
        node_ids = node_ids.long()
        mem = self.memory[node_ids]
        return self.embedding_module(mem, node_feat)


    def compute_messages(self, src, dst, ts, edge_feats):
        device = src.device

        mem_snap = self.memory.clone()
        prev_src_mem = mem_snap[src]
        prev_dst_mem = mem_snap[dst]
        prev_src_ts = self.last_update_ts[src].view(-1, 1)
        prev_dst_ts = self.last_update_ts[dst].view(-1, 1)

        ts = ts.float().view(-1, 1).to(device)
        tenc_src = self.time_encoder(ts - prev_src_ts)
        tenc_dst = self.time_encoder(ts - prev_dst_ts)

        msg_dst = self.message_fn(prev_src_mem, prev_dst_mem, tenc_dst, edge_feats)
        msg_src = self.message_fn(prev_dst_mem, prev_src_mem, tenc_src, edge_feats)
        msg_dst, msg_src = self.dropout(msg_dst), self.dropout(msg_src)

        all_nodes = torch.cat([src, dst], dim=0)
        all_msgs  = torch.cat([msg_src, msg_dst], dim=0)

        return all_nodes, all_msgs, ts  # 返回给 batch 外部暂存
    
    @torch.no_grad()
    def update_memory_and_ts(self, update_ids, agg_msgs, ts, src, dst):
        if update_ids.numel() == 0:
            return

        prev_mem = self.memory[update_ids]
        new_mem = self.memory_updater(agg_msgs, prev_mem)  # GRU/RNN 执行仍然会跟踪梯度
        new_mem = self.dropout(new_mem)
        self.memory[update_ids] = new_mem

        idx = torch.cat([src, dst], dim=0)
        ts_rep = ts.repeat_interleave(2, dim=0).view(-1)
        max_ts = torch.zeros_like(self.last_update_ts)
        max_ts = max_ts.scatter_reduce(0, idx, ts_rep, reduce='amax')
        self.last_update_ts = torch.maximum(self.last_update_ts, max_ts)
        for nid, msg in zip(update_ids.tolist(), agg_msgs):
            self.mailbox[nid].append(msg.detach().cpu())


    def detach_memory(self):
        self.memory.detach_()
        self.last_update_ts.detach_()

    def reset_state(self):
        with torch.no_grad():
            self.memory.zero_()
            self.last_update_ts.zero_()
            self.mailbox = defaultdict(lambda: deque(maxlen=3))  # 每个节点保留最近 3 条消息



# ============== 3. Training Loop ==============

class DummyTemporalDataset(Dataset):
    def __init__(self, num_interactions=1000, num_nodes=100, edge_feat_dim=16, node_feat_dim=16):
        self.src = np.random.randint(0, num_nodes, num_interactions)
        self.dst = np.random.randint(0, num_nodes, num_interactions)
        for i in range(num_interactions):
            while self.dst[i] == self.src[i]:
                self.dst[i] = np.random.randint(0, num_nodes)
        self.ts = np.sort(np.random.rand(num_interactions) * num_interactions).astype(np.float32)
        self.edge_feats = np.random.randn(num_interactions, edge_feat_dim).astype(np.float32)
        self.node_feats = np.random.randn(num_nodes, node_feat_dim).astype(np.float32)
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


def train():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TGN(
        num_nodes=100, edge_feat_dim=16,
        memory_dim=32, time_dim=16,
        embedding_dim=32, message_dim=32
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.BCEWithLogitsLoss()

    dataset = DummyTemporalDataset(5000, 100, 16)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    for epoch in range(5):
        model.train()
        model.reset_state()
        total_loss = 0.0

        for src, dst, ts, feats, labels in loader:
            src, dst, ts, feats, labels = [x.to(DEVICE) for x in (src, dst, ts, feats, labels)]

            optimizer.zero_grad()

            # include update_state in autograd graph
            all_nodes, all_msgs, ts_snapshot = model.compute_messages(src, dst, ts, feats)
            update_ids, agg_msgs = model.aggregator(all_nodes, all_msgs)

            # compute embeddings and loss
            pos_src_emb = model.compute_embedding(src)
            pos_dst_emb = model.compute_embedding(dst)

            neg_dst = torch.randint(0, 100, (src.size(0),), device=DEVICE)
            neg_src_emb = model.compute_embedding(src)
            neg_dst_emb = model.compute_embedding(neg_dst)

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
            
            model.update_memory_and_ts(update_ids, agg_msgs, ts_snapshot, src, dst)

            # cut off history to avoid OOM
            model.detach_memory()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(loader):.4f}")

    print("Training complete.")

if __name__ == "__main__":
    train()
