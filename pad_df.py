
import numpy as np
import pandas as pd

def pad_df_to_gpu_multiple(
    df: pd.DataFrame,
    batch_size: int,
    n_gpu: int,
) -> tuple[pd.DataFrame, int, int]:
    """
    Pad df by repeating the last row so that:
      n_mini = ceil(len(df_padded) / batch_size)
    is divisible by n_gpu.

    Returns: (df_padded, pad_rows, n_mini_padded)
    """
    if n_gpu <= 0:
        raise ValueError(f"n_gpu must be >= 1, got {n_gpu}")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    n = len(df)
    if n == 0:
        return df.copy(), 0, 0

    n_mini = (n + batch_size - 1) // batch_size
    n_mini_pad = ((n_mini + n_gpu - 1) // n_gpu) * n_gpu

    # minimal N_pad such that ceil(N_pad / batch_size) == n_mini_pad
    n_pad = (n_mini_pad - 1) * batch_size + 1
    pad_rows = n_pad - n

    if pad_rows <= 0:
        return df, 0, n_mini

    last_row = df.iloc[[-1]]
    pad_block = pd.concat([last_row] * pad_rows, ignore_index=True)

    df_padded = pd.concat([df.reset_index(drop=True), pad_block], ignore_index=True)

    # sanity
    n2 = len(df_padded)
    n_mini2 = (n2 + batch_size - 1) // batch_size
    assert n_mini2 == n_mini_pad and (n_mini2 % n_gpu == 0)

    return df_padded, pad_rows, n_mini2


def make_groups(df: pd.DataFrame, batch_size: int):
    # robust to non-contiguous df.index
    assign = np.arange(len(df)) // batch_size
    return df.groupby(assign)