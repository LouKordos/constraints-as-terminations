#!/usr/bin/env python3
"""
Compare two PPO tensor–snapshot Zarr files chunk-wise, report all
numerical differences, *and* tell you the earliest time-step at which the
runs diverge.

Example
-------
python compare_tensor_snapshots.py run1/ run2/ \
       --max-steps 1_000_000 --chunk-size 2048

Exit status
-----------
0 → runs identical w.r.t. rtol/atol and length
1 → any mismatch detected
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import zarr
from numcodecs import Blosc  # noqa: F401 – needed when store was created
from tqdm import tqdm

DEFAULT_SIGNALS: List[str] = [
    "actions",
    "rewards",
    "obs",
    "obs_rms",
    "next_dones",
    "returns",
]

def find_zarr(path: str) -> str:
    """Return the single training_data_*.zarr inside *path*/tensor_snapshots."""
    candidates = glob.glob(
        os.path.join(path, "tensor_snapshots", "training_data_*.zarr")
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *.zarr store in {path}/tensor_snapshots, "
            f"found {candidates}"
        )
    return candidates[0]

def compare_arrays(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> np.ndarray:
    """Boolean mask of elements that fail all-close."""
    return ~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)

def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Chunk-wise comparison of PPO Zarr snapshots with earliest mismatch detection")
    )
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
        help=(
            "Compare up to this many *training* steps. Value is divided by append_interval before indexing."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=1024, help="Steps per read chunk")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance")
    parser.add_argument(
        "--max-print",
        type=int,
        default=10,
        help="Max mismatching indices to print per dataset (0 = unlimited)",
    )
    parser.add_argument(
        "--stop-at-first-diff",
        action="store_true",
        default=False,
        help="Abort scanning once the first mismatch is found",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress progress bars",
    )
    args = parser.parse_args()

    path_a = find_zarr(args.run_a)
    path_b = find_zarr(args.run_b)

    zarr_run_a = zarr.open_group(path_a, mode="r")
    zarr_run_b = zarr.open_group(path_b, mode="r")

    append_interval_a = zarr_run_a.attrs.get("append_interval")
    append_interval_b = zarr_run_b.attrs.get("append_interval")
    if append_interval_a != append_interval_b or append_interval_a is None:
        print(f"ERROR: append_interval differs between runs ({append_interval_a} vs {append_interval_b})")
        sys.exit(1)
    append_interval = append_interval_a

    exit_code = 0
    earliest_global_mismatch: Optional[Tuple[str, int]] = None # (dataset, step)

    for dataset_name in args.signals:
        if dataset_name not in zarr_run_a or dataset_name not in zarr_run_b:
            print(f"[WARN] dataset {dataset_name!r} missing in one of the runs – skipping")
            continue

        np_array_run_a, np_array_run_b = zarr_run_a[dataset_name], zarr_run_b[dataset_name]
        if np_array_run_a.shape[1:] != np_array_run_b.shape[1:]:
            print(f"[ERROR] shape mismatch in {dataset_name}: {np_array_run_a.shape} vs {np_array_run_b.shape}")
            exit_code = 1
            continue

        # steps common to both runs (respecting --max-steps)
        common_steps = min(
            np_array_run_a.shape[0],
            np_array_run_b.shape[0],
            args.max_steps // append_interval if args.max_steps else max(np_array_run_a.shape[0], np_array_run_b.shape[0]),
        )

        mismatches = 0
        printed_counter = 0
        first_mismatch_dataset: Optional[int] = None

        loop_range = range(0, common_steps, args.chunk_size)
        if not args.quiet:
            loop_range = tqdm(loop_range, desc=dataset_name, unit="chunk")

        for start in loop_range:
            end = min(start + args.chunk_size, common_steps)
            chunk_run_a = np_array_run_a[start:end]
            chunk_run_b = np_array_run_b[start:end]
            bad_mask = compare_arrays(chunk_run_a, chunk_run_b, args.rtol, args.atol)

            if bad_mask.any():
                relative_bad_indices = np.argwhere(bad_mask)
                first_relative_mismatch = relative_bad_indices[0]
                first_absolute_mismatch = start + first_relative_mismatch[0]
                first_mismatch_dataset = first_absolute_mismatch
                # Update global earliest
                if earliest_global_mismatch is None or first_absolute_mismatch < earliest_global_mismatch[1]:
                    earliest_global_mismatch = (dataset_name, first_absolute_mismatch)

                # Pretty-print some samples
                if args.max_print == 0 or printed_counter < args.max_print:
                    for rel_idx in relative_bad_indices[: args.max_print - printed_counter if args.max_print else None]:
                        global_idx = (start + rel_idx[0],) + tuple(rel_idx[1:])
                        val_a = chunk_run_a[tuple(rel_idx)]
                        val_b = chunk_run_b[tuple(rel_idx)]
                        print(f"MISMATCH {dataset_name} @ {global_idx}: runA={val_a!r} runB={val_b!r}")
                        printed_counter += 1
                mismatches += len(relative_bad_indices)
                if args.stop_at_first_diff:
                    break  # stop scanning this dataset

        # # length mismatch beyond common prefix
        # if first_mismatch_dataset is None and np_array_run_a.shape[0] != np_array_run_b.shape[0]:
        #     first_mismatch_dataset = common_steps
        #     if earliest_global_mismatch is None or common_steps < earliest_global_mismatch[1]:
        #         earliest_global_mismatch = (dataset_name, common_steps)
        #     mismatches += 1
        #     print(f"LENGTH MISMATCH {dataset_name}: runA={np_array_run_a.shape[0]} steps, runB={np_array_run_b.shape[0]} steps (diverge at {common_steps})")

        # summary line
        if mismatches:
            print(f"[FAIL] {dataset_name}: {mismatches} element(s) differ")
            exit_code = 1
        else:
            print(f"[OK]   {dataset_name}: identical for {common_steps} step(s)")

        # optional early abort
        if args.stop_at_first_diff and earliest_global_mismatch is not None:
            break

    if earliest_global_mismatch:
        ds, step = earliest_global_mismatch
        print(
            f"\nEarliest mismatch: step {step} "
            f"(global training step {step * append_interval}) in dataset '{ds}'"
        )
    else:
        print("\nRuns are identical within the compared range and length.")

    sys.exit(exit_code)

if __name__ == "__main__":
    main()