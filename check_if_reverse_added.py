import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


GRAPH_TYPES = ("int_train", "int_full", "ext_full")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check whether a generated graph matches gen_graph.py --add_reverse output."
    )
    parser.add_argument("--data", type=str, required=True, help="dataset name under DATA/")
    parser.add_argument(
        "--graph",
        type=str,
        default="ext_full",
        help="graph name (int_train/int_full/ext_full) or a path to a .npz graph file",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=5,
        help="max number of problematic edges to print",
    )
    return parser.parse_args()


def resolve_graph_path(data_name, graph_arg):
    graph_path = Path(graph_arg)
    if graph_path.suffix == ".npz" or graph_path.exists():
        if not graph_path.is_absolute():
            graph_path = Path.cwd() / graph_path
    else:
        graph_path = Path("DATA") / data_name / f"{graph_arg}.npz"

    graph_type = graph_path.stem
    if graph_type not in GRAPH_TYPES:
        raise ValueError(
            f"Cannot infer graph type from {graph_path}. Expected one of {GRAPH_TYPES}."
        )
    return graph_path, graph_type


def eligible_mask(edges_df, graph_type):
    int_roll = edges_df["int_roll"].to_numpy()
    if graph_type == "int_train":
        return int_roll == 0
    if graph_type == "int_full":
        return int_roll != 3
    if graph_type == "ext_full":
        return np.ones(len(edges_df), dtype=bool)
    raise ValueError(f"Unsupported graph type: {graph_type}")


def make_key_array(src, dst, ts, eid):
    keys = np.empty(
        len(eid),
        dtype=np.dtype(
            [("src", np.int64), ("dst", np.int64), ("ts", np.float64), ("eid", np.int64)]
        ),
    )
    keys["src"] = src.astype(np.int64, copy=False)
    keys["dst"] = dst.astype(np.int64, copy=False)
    keys["ts"] = ts.astype(np.float64, copy=False)
    keys["eid"] = eid.astype(np.int64, copy=False)
    return keys


def summarize_examples(edges_df, edge_ids, sample_limit):
    if len(edge_ids) == 0:
        return []
    rows = edges_df.iloc[edge_ids[:sample_limit]]
    examples = []
    for edge_id, row in rows.iterrows():
        examples.append(
            f"eid={edge_id}, src={int(row['src'])}, dst={int(row['dst'])}, time={row['time']}"
        )
    return examples


def main():
    args = parse_args()

    graph_path, graph_type = resolve_graph_path(args.data, args.graph)
    edges_path = Path("DATA") / args.data / "edges.csv"

    if not edges_path.exists():
        raise FileNotFoundError(f"edges.csv not found: {edges_path}")
    if not graph_path.exists():
        raise FileNotFoundError(f"graph file not found: {graph_path}")

    edges_df = pd.read_csv(edges_path)
    required_columns = {"src", "dst", "time", "int_roll"}
    missing_columns = required_columns - set(edges_df.columns)
    if missing_columns:
        raise ValueError(f"edges.csv is missing required columns: {sorted(missing_columns)}")

    mask = eligible_mask(edges_df, graph_type)
    eligible_edge_ids = np.flatnonzero(mask)
    edge_src = np.asarray(edges_df["src"], dtype=np.int64)
    edge_dst = np.asarray(edges_df["dst"], dtype=np.int64)
    edge_ts = np.asarray(edges_df["time"], dtype=np.float64)

    with np.load(graph_path) as graph_data:
        indptr = graph_data["indptr"]
        indices = graph_data["indices"]
        ts = graph_data["ts"]
        eid = graph_data["eid"]

    if not (len(indices) == len(ts) == len(eid)):
        raise ValueError("Graph arrays must have the same length: indices, ts, eid")
    if indptr.ndim != 1 or len(indptr) == 0:
        raise ValueError("indptr must be a non-empty 1D array")
    if indptr[-1] != len(indices):
        raise ValueError(
            f"CSR mismatch: indptr[-1]={indptr[-1]} but len(indices)={len(indices)}"
        )

    eid = eid.astype(np.int64, copy=False)
    if len(eid) > 0 and (eid.min() < 0 or eid.max() >= len(edges_df)):
        raise ValueError("Graph contains eid values outside the edges.csv row range")

    counts = np.bincount(eid, minlength=len(edges_df))
    eligible_counts = counts[eligible_edge_ids]
    ineligible_edge_ids = np.flatnonzero(~mask)
    ineligible_present = ineligible_edge_ids[counts[ineligible_edge_ids] > 0]

    graph_src = np.repeat(
        np.arange(len(indptr) - 1, dtype=np.int64),
        np.diff(indptr).astype(np.int64, copy=False),
    )
    graph_keys = make_key_array(
        graph_src,
        indices.astype(np.int64, copy=False),
        ts.astype(np.float64, copy=False),
        eid,
    )

    expected_forward = make_key_array(
        edge_src[mask],
        edge_dst[mask],
        edge_ts[mask],
        eligible_edge_ids,
    )
    expected_reverse = make_key_array(
        edge_dst[mask],
        edge_src[mask],
        edge_ts[mask],
        eligible_edge_ids,
    )

    forward_present = np.isin(expected_forward, graph_keys)
    reverse_present = np.isin(expected_reverse, graph_keys)

    all_single = np.all(eligible_counts == 1)
    all_double = np.all(eligible_counts == 2)
    forward_complete = np.all(forward_present)
    reverse_complete = np.all(reverse_present)
    total_expected_without_reverse = len(eligible_edge_ids)
    total_expected_with_reverse = len(eligible_edge_ids) * 2
    no_ineligible_entries = len(ineligible_present) == 0

    print(f"dataset: {args.data}")
    print(f"graph: {graph_path}")
    print(f"graph_type: {graph_type}")
    print(f"eligible_edges: {len(eligible_edge_ids)}")
    print(f"graph_entries: {len(eid)}")

    if (
        all_double
        and forward_complete
        and reverse_complete
        and len(eid) == total_expected_with_reverse
        and no_ineligible_entries
    ):
        print("reverse_added: yes")
        print("detail: every eligible edge appears once in the forward direction and once in the reverse direction with the same eid/time.")
        return 0

    if (
        all_single
        and forward_complete
        and len(eid) == total_expected_without_reverse
        and no_ineligible_entries
    ):
        print("reverse_added: no")
        print("detail: every eligible edge appears exactly once, which matches gen_graph.py without --add_reverse.")
        return 0

    print("reverse_added: inconsistent")
    print("detail: graph does not fully match either generation mode.")

    if not forward_complete:
        missing_forward_ids = eligible_edge_ids[~forward_present]
        print(f"missing_forward_edges: {len(missing_forward_ids)}")
        for example in summarize_examples(edges_df, missing_forward_ids, args.sample_limit):
            print(f"  {example}")

    if not reverse_complete:
        missing_reverse_ids = eligible_edge_ids[~reverse_present]
        print(f"missing_reverse_edges: {len(missing_reverse_ids)}")
        for example in summarize_examples(edges_df, missing_reverse_ids, args.sample_limit):
            print(f"  {example}")

    weird_count_ids = eligible_edge_ids[(eligible_counts != 1) & (eligible_counts != 2)]
    if len(weird_count_ids) > 0:
        print(f"edges_with_unexpected_eid_count: {len(weird_count_ids)}")
        for example in summarize_examples(edges_df, weird_count_ids, args.sample_limit):
            print(f"  {example}")

    mixed_count_ids = eligible_edge_ids[(eligible_counts == 1) | (eligible_counts == 2)]
    if 0 < len(mixed_count_ids) < len(eligible_edge_ids) and not (all_single or all_double):
        print("eid_count_mix: some eligible edges appear once and others appear twice")

    if len(ineligible_present) > 0:
        print(f"ineligible_edges_present: {len(ineligible_present)}")
        for example in summarize_examples(edges_df, ineligible_present, args.sample_limit):
            print(f"  {example}")

    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)