import numpy as np

# Given parameters
batch_size = 8
train_edge_end = 30
reorder = 4

# Generate base index
group_indexes = []
base_idx = np.arange(train_edge_end) // batch_size
group_indexes.append(base_idx)

# Generate random chunk scheduling offsets
group_idx = []
for i in range(reorder):
    group_idx += list(range(0 - i, reorder - i))
group_idx = np.repeat(np.array(group_idx), batch_size // reorder)
group_idx = np.tile(group_idx, train_edge_end // batch_size + 1)[:train_edge_end]
group_indexes.append(base_idx + group_idx)

# Generate reorder variants with prefix -1
for i in range(1, reorder):
    additional_idx = np.zeros(batch_size // reorder * i) - 1
    arr = np.concatenate([additional_idx, base_idx])[:train_edge_end]
    group_indexes.append(arr)

# Display the results
for idx, arr in enumerate(group_indexes):
    print(','.join(map(lambda x: str(int(x)) if x >= 0 else str(int(x)), arr)))
