import torch
import torch.nn as nn
import numpy as np
import math
import time
from torch.utils.data import DataLoader, Dataset # For training loop example

# ============== 1. Helper Modules ==============

class TimeEncoder(nn.Module):
    """
    Sinusoidal Time Encoder from the TGN paper.
    Input shape: (batch_size, 1) tensor of time differences
    Output shape: (batch_size, time_dim)
    """
    def __init__(self, time_dim):
        super().__init__()
        self.time_dim = time_dim
        # Basis frequency coefficient (adjust if needed)
        self.basis_freq = nn.Parameter(torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim)).float())
        self.phase = nn.Parameter(torch.zeros(time_dim).float())

    def forward(self, ts_diff):
        # ts_diff shape: [batch_size, 1]
        batch_size = ts_diff.size(0)
        # Ensure ts_diff is float and on the correct device
        ts_diff = ts_diff.float().to(self.basis_freq.device)

        map_ts = ts_diff * self.basis_freq.view(1, -1) + self.phase.view(1, -1) # [batch_size, time_dim]
        harmonic = torch.cos(map_ts)
        return harmonic

class MessageFunction(nn.Module):
    """
    Computes messages. Can be MLP, Identity, etc.
    Input: source_memory, dest_memory, time_encoding, edge_features
    Output: message vector
    """
    def __init__(self, memory_dim, time_dim, edge_feat_dim, message_dim):
        super().__init__()
        input_dim = 2 * memory_dim + time_dim + edge_feat_dim
        self.layer = nn.Sequential(
            nn.Linear(input_dim, message_dim),
            nn.ReLU()
            # Optionally add more layers
        )

    def forward(self, src_mem, dst_mem, time_enc, edge_feat):
        input_features = torch.cat([src_mem, dst_mem, time_enc, edge_feat], dim=1)
        message = self.layer(input_features)
        return message

class MessageAggregator(nn.Module):
    """
    Aggregates messages for a node. 'mean' or 'last'.
    Input: node_ids (for mapping), messages (associated with node_ids)
    Output: aggregated_messages (one per unique node_id)
    """
    def __init__(self, agg_method='mean'):
        super().__init__()
        if agg_method not in ['mean', 'last']:
            raise ValueError("Aggregation method must be 'mean' or 'last'")
        self.agg_method = agg_method

    def forward(self, node_ids, messages):
        """
        Aggregates messages for the same node_id.
        Assumes messages for the *same* interaction event (e.g., u->v and v->u)
        are handled appropriately *before* calling this if only one message per
        node update is desired (often the case). This function aggregates
        messages potentially arriving *between* memory updates.

        Args:
            node_ids (Tensor): Shape [num_messages], IDs of nodes receiving messages.
            messages (Tensor): Shape [num_messages, message_dim].

        Returns:
            unique_node_ids (Tensor): Unique node IDs that received messages.
            agg_messages (Tensor): Aggregated message for each unique node ID.
                                     Shape [num_unique_nodes, message_dim].
        """
        unique_node_ids = torch.unique(node_ids)
        agg_messages = []

        # Ensure device consistency
        device = messages.device
        unique_node_ids = unique_node_ids.to(device)

        # This loop can be slow for large graphs; consider torch_scatter for optimization
        for nid in unique_node_ids:
            node_messages = messages[node_ids == nid]
            if self.agg_method == 'mean':
                agg_messages.append(node_messages.mean(dim=0))
            elif self.agg_method == 'last':
                # This assumes messages are chronologically ordered *for each node*
                # In many batch implementations, 'mean' is easier.
                # If using 'last', ensure messages are sorted by time *within* the batch processing
                # or handle message ordering carefully before aggregation.
                agg_messages.append(node_messages[-1])

        if not agg_messages:
             return unique_node_ids, torch.empty((0, messages.shape[1]), device=device) # Handle case with no messages

        agg_messages = torch.stack(agg_messages)
        return unique_node_ids, agg_messages


class MemoryUpdater(nn.Module):
    """
    Updates node memory using aggregated messages. GRU or RNN.
    Input: aggregated_message, previous_memory
    Output: new_memory
    """
    def __init__(self, memory_dim, message_dim, update_type='gru'):
        super().__init__()
        if update_type == 'gru':
            self.updater = nn.GRUCell(message_dim, memory_dim)
        elif update_type == 'rnn':
            self.updater = nn.RNNCell(message_dim, memory_dim)
        else:
            raise ValueError("Memory update type must be 'gru' or 'rnn'")

    def forward(self, agg_message, prev_memory):
        # GRUCell/RNNCell expect input [batch_size, input_size] and hidden [batch_size, hidden_size]
        # Here, batch_size corresponds to the number of nodes being updated.
        # Ensure prev_memory has gradients enabled if needed for backprop through time
        new_memory = self.updater(agg_message, prev_memory)
        return new_memory

class EmbeddingModule(nn.Module):
    """
    Computes temporal node embeddings using memory and recent messages/neighbors.
    This is a simplified version using just the memory.
    A full implementation might use graph attention over recent interactions.
    Input: node_ids, node_memory (at the time of embedding computation)
    Output: node_embeddings
    """
    def __init__(self, memory_dim, embedding_dim):
        super().__init__()
        # Simple projection from memory to embedding space
        # More complex versions (like Graph Attention) could be used here
        self.embedding_layer = nn.Linear(memory_dim, embedding_dim)

    def forward(self, node_memory):
        # node_memory shape: [num_nodes_to_embed, memory_dim]
        # If node_memory requires grad (because it resulted from previous updates
        # without detach), gradients will flow back through here to embedding_layer
        # AND back through node_memory itself.
        node_embeddings = self.embedding_layer(node_memory)
        return node_embeddings


# ============== 2. TGN Model ==============

class TGN(nn.Module):
    def __init__(self, num_nodes, edge_feat_dim,
                 memory_dim, time_dim, embedding_dim, message_dim,
                 agg_method='mean', mem_update_type='gru', dropout=0.1):
        super().__init__()

        self.num_nodes = num_nodes
        self.edge_feat_dim = edge_feat_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.embedding_dim = embedding_dim
        self.message_dim = message_dim

        # --- State (managed outside forward, updated by methods) ---
        # Node memory vectors - still requires_grad=False as we optimize modules, not the state tensor directly
        self.memory = nn.Parameter(torch.zeros((num_nodes, memory_dim)), requires_grad=False)
        # Last interaction timestamp for each node
        self.last_update_ts = nn.Parameter(torch.zeros(num_nodes), requires_grad=False)

        # --- Modules ---
        self.time_encoder = TimeEncoder(time_dim)
        self.message_fn = MessageFunction(memory_dim, time_dim, edge_feat_dim, message_dim)
        self.aggregator = MessageAggregator(agg_method)
        self.memory_updater = MemoryUpdater(memory_dim, message_dim, mem_update_type)
        self.embedding_module = EmbeddingModule(memory_dim, embedding_dim)

        self.dropout = nn.Dropout(dropout)

    def compute_embedding(self, node_ids):
        """
        Computes embeddings for given nodes *at the current memory state*.
        Args:
            node_ids (Tensor): LongTensor of node IDs.
        Returns:
            embeddings (Tensor): Computed embeddings. Shape [len(node_ids), embedding_dim].
        """
        node_ids = node_ids.long() # Ensure correct type

        # Fetch current memory for the nodes.
        # IMPORTANT: If memory has a computation graph attached (due to removed detach),
        # this operation keeps it attached. Gradients from the loss using the
        # resulting embeddings will flow back *through* this memory state.
        node_memory = self.memory[node_ids]

        # Compute embeddings using the embedding module
        embeddings = self.embedding_module(node_memory)
        return embeddings

    def update_state(self, src_nodes, dst_nodes, timestamps, edge_features):
        """
        Updates the memory and last_update_ts based on a batch of interactions.
        This function implements the core TGN logic: compute messages, aggregate, update memory.

        Args:
            src_nodes (Tensor): Source node IDs of interactions.
            dst_nodes (Tensor): Destination node IDs of interactions.
            timestamps (Tensor): Timestamps of interactions.
            edge_features (Tensor): Features of interactions. Shape [batch_size, edge_feat_dim].
        """
        batch_size = len(src_nodes)
        device = src_nodes.device # Get device from input tensors

        # Ensure inputs are LongTensors where needed
        src_nodes = src_nodes.long()
        dst_nodes = dst_nodes.long()

        # --- Get previous states ---
        # Clone memory. If self.memory has grad history, the clone will too.
        # This snapshot is used for message computation based on state *before* this batch's update.
        memory_snapshot = self.memory.clone()
        prev_src_mem = memory_snapshot[src_nodes]
        prev_dst_mem = memory_snapshot[dst_nodes]

        # Timestamps are parameters but treated as state, usually no grad needed back to them.
        prev_src_ts = self.last_update_ts[src_nodes].clone() # Clone for safety if needed elsewhere
        prev_dst_ts = self.last_update_ts[dst_nodes].clone()

        # --- Compute time differences and encodings ---
        # Timestamps tensor needs to be [batch_size, 1] for TimeEncoder
        ts_tensor = timestamps.float().view(-1, 1).to(device)
        time_diff_src = ts_tensor - prev_src_ts.view(-1, 1)
        time_diff_dst = ts_tensor - prev_dst_ts.view(-1, 1)

        # Time encoding computation will be part of the graph
        time_enc_src = self.time_encoder(time_diff_src)
        time_enc_dst = self.time_encoder(time_diff_dst)

        # --- Compute messages ---
        # These computations use modules whose parameters we want to train.
        msg_for_dst = self.message_fn(prev_src_mem, prev_dst_mem, time_enc_dst, edge_features)
        msg_for_src = self.message_fn(prev_dst_mem, prev_src_mem, time_enc_src, edge_features)

        # Apply dropout to messages
        msg_for_dst = self.dropout(msg_for_dst)
        msg_for_src = self.dropout(msg_for_src)

        # --- Aggregate messages per node ---
        # Combine all nodes involved and corresponding messages
        all_nodes = torch.cat([src_nodes, dst_nodes])
        all_messages = torch.cat([msg_for_src, msg_for_dst])

        # Aggregation computation will be part of the graph
        unique_update_nodes, aggregated_messages = self.aggregator(all_nodes, all_messages)

        if unique_update_nodes.nelement() == 0: # Handle case with no messages/updates
            print("Warning: No nodes to update.")
            # Update timestamps even if memory doesn't change? Optional.
            # self._update_timestamps(all_nodes, timestamps, src_nodes, dst_nodes) # Refactored timestamp update
            return # Nothing to update memory

        # --- Update memory for nodes that received messages ---
        # Fetch previous memory for the unique nodes needing updates FROM THE SNAPSHOT
        # because the GRU/RNN needs the state *before* the current update.
        prev_mem_to_update = memory_snapshot[unique_update_nodes]

        # Update memory using the updater module. This is a key part of the graph.
        new_node_memory = self.memory_updater(aggregated_messages, prev_mem_to_update)

        # Apply dropout to memory update
        new_node_memory = self.dropout(new_node_memory)

        # --- Store updated memory and timestamps ---

        # Store updated memory.
        # REMOVED torch.no_grad() here to allow gradients to flow back
        # from the use of self.memory in the *next* batch's compute_embedding
        # back through this assignment into new_node_memory, and thus to the
        # memory_updater, aggregator, message_fn, time_encoder.
        # This connects the computation graph across batches.
        # <--- MODIFICATION START
        self.memory[unique_update_nodes] = new_node_memory
        # <--- MODIFICATION END


        # Update timestamps - Keep this under no_grad as gradients typically aren't needed for timestamps
        # <--- MODIFICATION START
        with torch.no_grad():
            self._update_timestamps(all_nodes, timestamps, src_nodes, dst_nodes, device)
        # <--- MODIFICATION END

    # Helper method for timestamp update logic
    def _update_timestamps(self, all_nodes, timestamps, src_nodes, dst_nodes, device):
         # Find the latest timestamp for each node involved in this update batch
         # This is a simplification; assumes batch timestamps can update last_update_ts
         # A more precise way might involve iterating or using scatter_max
         max_ts_map = {}
         # Consider doing this on GPU if efficient scatter_max is available/worthwhile
         all_involved_nodes_unique = torch.unique(all_nodes)
         for nid_tensor in all_involved_nodes_unique: # Iterate over tensors if on GPU
             nid = nid_tensor.item() # Get Python number for dictionary key
             # Find relevant timestamps for this node ID within the current batch
             node_mask = (src_nodes == nid) | (dst_nodes == nid)
             if torch.any(node_mask):
                 node_timestamps = timestamps[node_mask]
                 max_ts_map[nid] = torch.max(node_timestamps)

         for nid, max_ts in max_ts_map.items():
             # Update only if the current batch time is later
             # Use index directly; nid is already a Python int
             if max_ts > self.last_update_ts[nid]:
                 # Ensure max_ts is moved to the correct device if not already
                 self.last_update_ts[nid] = max_ts.to(self.last_update_ts.device)


    # detach_memory is removed from the training loop, but might be useful
    # for specific evaluation scenarios or if implementing TBPTT manually.
    # Keep the method definition if needed elsewhere.
    def detach_memory(self):
        """Detaches the memory and last_update_ts from the computation graph."""
        self.memory.detach_()
        self.last_update_ts.detach_()

    def reset_state(self):
        """Resets memory and last update times to zero."""
        with torch.no_grad():
            self.memory.fill_(0.)
            self.last_update_ts.fill_(0.)

# ============== 3. Conceptual Training Loop ==============

# --- Dummy Data Generation ---
# (Dataset and Collate function remain the same)
class DummyTemporalDataset(Dataset):
    def __init__(self, num_interactions=1000, num_nodes=100, edge_feat_dim=16):
        self.num_interactions = num_interactions
        self.num_nodes = num_nodes
        self.edge_feat_dim = edge_feat_dim

        # Generate random interactions - MUST BE SORTED BY TIME for TGN
        self.src = np.random.randint(0, num_nodes, num_interactions)
        self.dst = np.random.randint(0, num_nodes, num_interactions)
        # Ensure src != dst for simplicity here
        for i in range(num_interactions):
            while self.dst[i] == self.src[i]:
                self.dst[i] = np.random.randint(0, num_nodes)

        # Generate increasing timestamps
        self.ts = np.sort(np.random.rand(num_interactions) * num_interactions).astype(np.float32)
        self.edge_feats = np.random.randn(num_interactions, edge_feat_dim).astype(np.float32)
        # Labels (e.g., 1 for existing edge) - needed for loss
        self.labels = np.ones(num_interactions).astype(np.float32)

    def __len__(self):
        return self.num_interactions

    def __getitem__(self, idx):
        return self.src[idx], self.dst[idx], self.ts[idx], self.edge_feats[idx], self.labels[idx]

def collate_fn(batch):
    src, dst, ts, edge_feats, labels = zip(*batch)
    return (torch.tensor(src, dtype=torch.long),
            torch.tensor(dst, dtype=torch.long),
            torch.tensor(ts, dtype=torch.float),
            torch.from_numpy(np.stack(edge_feats, axis=0)),
            torch.tensor(labels, dtype=torch.float))

# --- Training Setup ---
NUM_NODES = 100
EDGE_FEAT_DIM = 16
MEMORY_DIM = 32
TIME_DIM = 16
EMBEDDING_DIM = 32
MESSAGE_DIM = 32
BATCH_SIZE = 64 # Smaller batch size might be needed due to increased memory usage
LEARNING_RATE = 0.0003
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Instantiate Model, Optimizer, Loss ---
model = TGN(
    num_nodes=NUM_NODES,
    edge_feat_dim=EDGE_FEAT_DIM,
    memory_dim=MEMORY_DIM,
    time_dim=TIME_DIM,
    embedding_dim=EMBEDDING_DIM,
    message_dim=MESSAGE_DIM
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss() # For link prediction

# --- Create DataLoader ---
dataset = DummyTemporalDataset(num_interactions=5000, num_nodes=NUM_NODES, edge_feat_dim=EDGE_FEAT_DIM)
# shuffle=False is crucial for time order
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# --- Training Loop ---
print(f"Starting training on {DEVICE}...")
print("WARNING: Memory detachment per batch is disabled. Memory usage will increase.")
print("         Monitor memory consumption closely.")

for epoch in range(EPOCHS):
    start_epoch_time = time.time()
    model.train()
    # Reset memory at the start of each epoch (detaches history from previous epoch)
    model.reset_state()
    total_loss = 0

    for batch_idx, (src, dst, ts, edge_feats, labels) in enumerate(dataloader):
        optimizer.zero_grad()

        src, dst, ts, edge_feats, labels = src.to(DEVICE), dst.to(DEVICE), ts.to(DEVICE), edge_feats.to(DEVICE), labels.to(DEVICE)

        # --- TGN Core Logic for Link Prediction ---
        # 1. Compute embeddings for positive edges *using the memory state before this batch's update*
        #    Since detach is removed, this memory state potentially carries gradient history
        #    from the loss computation of the *previous* batch.
        pos_src_emb = model.compute_embedding(src)
        pos_dst_emb = model.compute_embedding(dst)

        # 2. Generate Negative Samples (Simple Random Sampling)
        num_neg_samples = 1 # Number of negative samples per positive one
        neg_dst = torch.randint(0, NUM_NODES, (len(src) * num_neg_samples,), device=DEVICE)
        # TODO: Ensure neg_dst != actual dst for corresponding src more robustly

        # Compute embeddings for negative samples using the same memory state
        neg_src_emb = model.compute_embedding(src.repeat_interleave(num_neg_samples))
        neg_dst_emb = model.compute_embedding(neg_dst)

        # 3. Calculate Scores
        pos_score = torch.sum(pos_src_emb * pos_dst_emb, dim=1)
        neg_score = torch.sum(neg_src_emb * neg_dst_emb, dim=1)

        # 4. Compute Loss
        scores = torch.cat([pos_score, neg_score])
        pos_labels = torch.ones_like(pos_score)
        neg_labels = torch.zeros_like(neg_score)
        all_labels = torch.cat([pos_labels, neg_labels])

        loss = criterion(scores, all_labels)

        # 5. Backpropagation
        # NOW, gradients will flow from 'loss' back through 'scores', 'embeddings',
        # 'embedding_module', AND crucially *through the 'self.memory'* state used
        # in compute_embedding, back to the operations in the *previous* call
        # to update_state that produced this memory state. This includes
        # memory_updater, aggregator, message_fn, time_encoder.
        loss.backward(retain_graph=True)
        optimizer.step()

        # --- TGN State Update ---
        # Update memory *after* loss computation based on the current batch's interactions.
        # This updated memory will be used in the *next* batch's compute_embedding call.
        # Since no_grad was removed from the memory assignment inside update_state,
        # the computation graph continues through this update.
        model.update_state(src, dst, ts, edge_feats)

        # --- Detach memory ---
        # model.detach_memory() # <--- MODIFICATION: REMOVED this call

        total_loss += loss.item()

        # Optional: Print progress
        if (batch_idx + 1) % 20 == 0:
             print(f'Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}')


    epoch_time = time.time() - start_epoch_time
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s')

print("Training finished.")

# --- Evaluation (Conceptual) ---
# model.eval()
# model.reset_state() # Reset for evaluation
# with torch.no_grad(): # Use no_grad for evaluation efficiency
#    for batch in eval_dataloader:
#        # Compute embeddings using model.compute_embedding
#        # Make predictions
#        # Update state using model.update_state to keep memory current for next eval batch
#        # Note: No detach needed here because of torch.no_grad() context manager
#        # Calculate metrics
