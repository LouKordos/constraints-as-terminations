#!/usr/bin/env python3
"""
Compare two PPO tensor–snapshot Zarr files chunk-wise and report any
numerical differences.

Example:
    python compare_tensor_snapshots.py run1/ run2/ --max-steps 1000 --chunk-size 2048
"""

import argparse
import glob
import os
import sys
from typing import List

import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import tqdm  # optional

DEFAULT_SIGNALS: List[str] = [
    "actions",
    "rewards",
    "obs",
    "obs_rms",
    "next_dones",
    "returns",
]

def find_zarr(path: str) -> str:
    """Return the (unique) training_data_*.zarr inside *path*/tensor_snapshots."""
    candidates = glob.glob(os.path.join(path, "tensor_snapshots", "training_data_*.zarr"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *.zarr store in {path}/tensor_snapshots, found {candidates}"
        )
    return candidates[0]

def compare_arrays(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> np.ndarray:
    """Return a boolean mask of elements that fail allclose."""
    return ~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk-wise comparison of PPO Zarr snapshots")
    parser.add_argument("run_a", help="First run directory")
    parser.add_argument("run_b", help="Second run directory")
    parser.add_argument(
        "--signals",
        nargs="+",
        default=DEFAULT_SIGNALS,
        help=f"Datasets to compare (default: {' '.join(DEFAULT_SIGNALS)})",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Compare up to this many training steps. NOTE: Since ppo.py records at an interval, the steps supplied to the argument are scaled by the append interval. If you want to compare until global training step 1000, this will internally be scaled to `1000/append_interval` steps.",
    )
    parser.add_argument("--chunk-size", type=int, default=1024, help="Time steps per read chunk")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for allclose")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance for allclose")
    parser.add_argument(
        "--max-print",
        type=int,
        default=10,
        help="Maximum mismatching indices to print per dataset (0 = unlimited)",
    )
    args = parser.parse_args()

    path_a = find_zarr(args.run_a)
    path_b = find_zarr(args.run_b)


    zarr_group_run_a = zarr.open_group(path_a, mode="r")
    print(zarr_group_run_a.attrs)
    zarr_group_run_b = zarr.open_group(path_b, mode="r")

    append_interval_a = zarr_group_run_a.attrs["append_interval"] # Might need to use attrs.asdict(), not tested yet
    append_interval_b = zarr_group_run_a.attrs["append_interval"]
    if append_interval_a != append_interval_b or append_interval_a is None or append_interval_b is None:
        print("append interval across runs needs to be identical and cannot be None, exiting.")
        sys.exit(1)

    exit_code = 0

    for dataset_name in args.signals:
        if dataset_name not in zarr_group_run_a or dataset_name not in zarr_group_run_b:
            print(f"[WARN] dataset {dataset_name!r} missing in one of the runs - skipping")
            continue

        array_run_a, array_run_b = zarr_group_run_a[dataset_name], zarr_group_run_b[dataset_name]
        if array_run_a.shape[1:] != array_run_b.shape[1:]:
            print(f"[ERROR] shape mismatch in {dataset_name}: {array_run_a.shape} vs {array_run_b.shape}")
            exit_code = 1
            continue

        steps = min(
            array_run_a.shape[0],
            array_run_b.shape[0],
            args.max_steps // append_interval_a if args.max_steps is not None else min(array_run_a.shape[0], array_run_b.shape[0]),
        )
        if steps == 0:
            print(f"[WARN] {dataset_name}: nothing to compare (zero steps)")
            continue

        print(f"Comparing {dataset_name} for {steps} time steps …")

        mismatches = 0
        printed = 0

        bar = tqdm(range(0, steps, args.chunk_size), desc=dataset_name, unit="chunk")
        for start in bar:
            end = min(start + args.chunk_size, steps)
            print("Fetching chunk from run a")
            chunk_a = array_run_a[start:end]
            print("Fetching chunk from run b")
            chunk_b = array_run_b[start:end]
            print("Comparing arrays")
            bad_mask = compare_arrays(chunk_a, chunk_b, args.rtol, args.atol)

            if bad_mask.any():
                idxs = np.argwhere(bad_mask)
                mismatches += len(idxs)
                if args.max_print == 0 or printed < args.max_print:
                    for rel_idx in idxs[: args.max_print - printed if args.max_print else None]:
                        global_idx = (start + rel_idx[0],) + tuple(rel_idx[1:])
                        val_a = chunk_a[tuple(rel_idx)]
                        val_b = chunk_b[tuple(rel_idx)]
                        print(
                            f"  MISMATCH {dataset_name} @ {global_idx}: "
                            f"runA={val_a!r} runB={val_b!r}"
                        )
                        printed += 1

        if mismatches:
            print(f"[FAIL] {dataset_name}: {mismatches} element(s) differ")
            exit_code = 1
        else:
            print(f"[OK]   {dataset_name}: allclose within rtol={args.rtol}, atol={args.atol}")

    sys.exit(exit_code)

if __name__ == "__main__":
    main()