from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REQUIRED_EDGE_COLUMNS = {"eid", "src", "dst", "default_split"}
TIME_COLUMNS = ("time", "ts")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "DATA"
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from frost.sampling.neg_sampler import RandomNegLinkSampler as FrostNegLinkSampler


def resolve_dataset_dir(dataset: str) -> Path:
    direct = DATA_ROOT / dataset
    if direct.is_dir():
        return direct

    if DATA_ROOT.is_dir():
        for child in DATA_ROOT.iterdir():
            if child.is_dir() and child.name.lower() == dataset.lower():
                return child

    available = []
    if DATA_ROOT.is_dir():
        available = sorted(child.name for child in DATA_ROOT.iterdir() if child.is_dir())
    raise FileNotFoundError(
        f"Could not resolve dataset directory for {dataset!r} under {DATA_ROOT}. "
        f"Available datasets: {available}"
    )


def validate_strict_negative_mode(
    *,
    use_inductive: bool = False,
    eval_neg_samples: int = 1,
) -> None:
    if use_inductive:
        raise ValueError(
            "Current DATA/FROST compatibility mode does not support --use_inductive. "
            'It only supports the dst-partition random single-negative sampler.'
        )
    if eval_neg_samples != 1:
        raise ValueError(
            "Current DATA/FROST compatibility mode only supports --eval_neg_samples=1 "
            'because it uses the dst-partition random single-negative sampler.'
        )


def _normalize_time_column(df: pd.DataFrame) -> pd.DataFrame:
    if "time" in df.columns:
        return df
    if "ts" in df.columns:
        return df.rename(columns={"ts": "time"})
    raise ValueError(
        f"edges.csv must contain one of {TIME_COLUMNS}, found columns: {list(df.columns)}"
    )


def load_edges_df(dataset: str) -> pd.DataFrame:
    csv_path = resolve_dataset_dir(dataset) / "edges.csv"
    df = pd.read_csv(csv_path)
    df = _normalize_time_column(df)

    required_columns = REQUIRED_EDGE_COLUMNS | {"time"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset {dataset} is missing required edge columns: {sorted(missing_columns)}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.copy()
    for column in ("eid", "src", "dst", "time", "default_split"):
        df[column] = df[column].astype(np.int64)

    if not np.array_equal(df["eid"].to_numpy(), np.arange(1, len(df) + 1, dtype=np.int64)):
        raise ValueError(
            f"Dataset {dataset} must have contiguous 1-based eid values in edges.csv."
        )

    split_values = df["default_split"].to_numpy()
    if not np.all(split_values[:-1] <= split_values[1:]):
        raise ValueError(
            f"Dataset {dataset} must keep default_split contiguous by row order."
        )

    return df.reset_index(drop=True)


def compute_split_boundaries(df: pd.DataFrame) -> tuple[int, int]:
    split_values = df["default_split"].to_numpy()
    val_mask = split_values > 0
    test_mask = split_values > 1
    if not val_mask.any():
        raise ValueError("Could not find validation split boundary (default_split > 0).")
    if not test_mask.any():
        raise ValueError("Could not find test split boundary (default_split > 1).")
    train_edge_end = int(np.flatnonzero(val_mask)[0])
    val_edge_end = int(np.flatnonzero(test_mask)[0])
    if train_edge_end <= 0 or val_edge_end <= train_edge_end:
        raise ValueError(
            f"Invalid split boundaries derived from default_split: "
            f"train_edge_end={train_edge_end}, val_edge_end={val_edge_end}"
        )
    return train_edge_end, val_edge_end


def load_graph(dataset: str) -> dict[str, np.ndarray]:
    npz_path = resolve_dataset_dir(dataset) / "full_graph_with_reverse_edges.npz"
    with np.load(npz_path, allow_pickle=False) as npz:
        required_keys = {"indptr", "indices", "eid", "ts"}
        missing_keys = required_keys.difference(npz.files)
        if missing_keys:
            raise ValueError(
                f"Graph file {npz_path} is missing required arrays: {sorted(missing_keys)}"
            )
        return {
            "indptr": np.ascontiguousarray(npz["indptr"], dtype=np.int64),
            "indices": np.ascontiguousarray(npz["indices"], dtype=np.int64),
            "eid": np.ascontiguousarray(npz["eid"], dtype=np.int64),
            "ts": np.ascontiguousarray(npz["ts"], dtype=np.int64),
        }


def load_dataset_counts(dataset: str) -> tuple[int, int]:
    df = load_edges_df(dataset)
    graph = load_graph(dataset)
    num_nodes = graph["indptr"].shape[0] - 1
    num_edges = len(df)
    return num_nodes, num_edges


def _load_feature_tensor(path: Path) -> torch.Tensor:
    if path.suffix == ".npy":
        # Some dataset snapshots materialize symlinks as tiny text files whose
        # contents are the relative target path (for example "node_role.npy").
        # Follow those indirections before handing the file to np.load().
        seen_paths: set[Path] = set()
        while True:
            if path in seen_paths:
                raise ValueError(f"Detected cyclic feature-path indirection involving {path}")
            seen_paths.add(path)

            with path.open("rb") as f:
                prefix = f.read(6)

            if prefix == b"\x93NUMPY":
                break

            try:
                alias_text = path.read_text(encoding="utf-8").strip()
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"Feature file {path} is neither a valid NumPy file nor a UTF-8 alias."
                ) from exc

            if not alias_text:
                raise ValueError(f"Feature alias file {path} is empty.")

            alias_path = (path.parent / alias_text).resolve()
            if not alias_path.exists():
                raise FileNotFoundError(
                    f"Feature alias file {path} points to missing target {alias_path}."
                )
            path = alias_path

        arr = np.load(path, allow_pickle=False)
        tensor = torch.from_numpy(arr)
    elif path.suffix == ".pt":
        tensor = torch.load(path, map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor in {path}, got {type(tensor)!r}")
    else:
        raise ValueError(f"Unsupported feature suffix: {path.suffix}")

    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D feature tensor at {path}, got shape {tuple(tensor.shape)}")

    if tensor.dtype == torch.bool:
        tensor = tensor.to(torch.float32)
    return tensor


def _resolve_feature_path(dataset_dir: Path, stem: str) -> Path | None:
    for suffix in (".npy", ".pt"):
        path = dataset_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def _prepare_node_features(node_feats: torch.Tensor | None, num_nodes: int) -> torch.Tensor | None:
    if node_feats is None:
        return None
    if node_feats.shape[0] not in (num_nodes, num_nodes + 1):
        raise ValueError(
            f"Node feature row mismatch: expected {num_nodes} or {num_nodes + 1}, "
            f"got {node_feats.shape[0]}"
        )
    return node_feats


def _prepare_edge_features(edge_feats: torch.Tensor | None, num_edges: int) -> torch.Tensor | None:
    if edge_feats is None:
        return None
    if edge_feats.shape[0] == num_edges + 1:
        return edge_feats
    if edge_feats.shape[0] == num_edges:
        pad = torch.zeros((1, edge_feats.shape[1]), dtype=edge_feats.dtype)
        return torch.cat([pad, edge_feats], dim=0)
    raise ValueError(
        f"Edge feature row mismatch: expected {num_edges} or {num_edges + 1}, "
        f"got {edge_feats.shape[0]}"
    )


def load_feat(
    dataset: str,
    rand_de: int,
    rand_dn: int,
    *,
    num_nodes: int,
    num_edges: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    dataset_dir = resolve_dataset_dir(dataset)

    node_feat_path = _resolve_feature_path(dataset_dir, "node_features")
    edge_feat_path = _resolve_feature_path(dataset_dir, "edge_features")

    node_feats = (
        _prepare_node_features(_load_feature_tensor(node_feat_path), num_nodes)
        if node_feat_path is not None
        else None
    )
    edge_feats = (
        _prepare_edge_features(_load_feature_tensor(edge_feat_path), num_edges)
        if edge_feat_path is not None
        else None
    )

    if rand_dn > 0:
        node_feats = torch.randn((num_nodes, rand_dn), dtype=torch.float32)
    if rand_de > 0:
        edge_feats = torch.randn((num_edges + 1, rand_de), dtype=torch.float32)
        edge_feats[0].zero_()

    return node_feats, edge_feats


class FrostBatchNegLinkSampler:
    def __init__(self, *, dataset: str, n_nodes: int):
        self.inner = FrostNegLinkSampler(
            n_nodes=n_nodes,
            n_neg=1,
            dataset=dataset,
            data_root=DATA_ROOT,
            candidate_mode="dst_partition",
        )

    def sample(self, src_ids, edge_ids) -> np.ndarray:
        src_tensor = torch.tensor(np.asarray(src_ids), dtype=torch.long)
        edge_tensor = torch.tensor(np.asarray(edge_ids), dtype=torch.long)
        negatives = self.inner.sample(src_tensor, edge_tensor)
        return negatives.cpu().numpy()
