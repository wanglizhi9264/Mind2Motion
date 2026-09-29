"""Read and audit the fixed train/validation/test split stored in the HDF5 file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


SPLIT_NAMES = ("train", "val", "test")


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    )


def load_splits(h5_path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Return fixed row indices and a leakage-audit summary.

    The release never creates a new random split. It uses the ``split`` column
    stored in the final V8 dataset so every run evaluates the same samples.
    """
    with h5py.File(h5_path, "r") as handle:
        split = decode(handle["split"][:])
        text_ids = decode(handle["text_id"][:])
        sample_names = decode(handle["sample_name"][:])

    unknown = sorted(set(split) - set(SPLIT_NAMES))
    if unknown:
        raise ValueError(f"Unknown split labels: {unknown}")

    indices = {name: np.flatnonzero(split == name) for name in SPLIT_NAMES}
    if sum(len(value) for value in indices.values()) != len(split):
        raise ValueError("Some samples were not assigned to a split")

    sample_sets = {
        name: set(sample_names[row_indices]) for name, row_indices in indices.items()
    }
    text_sets = {
        name: set(text_ids[row_indices]) for name, row_indices in indices.items()
    }
    overlap = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        sample_overlap = sample_sets[left] & sample_sets[right]
        text_overlap = text_sets[left] & text_sets[right]
        overlap[f"{left}_{right}"] = {
            "sample_names": len(sample_overlap),
            "text_ids": len(text_overlap),
        }
        if sample_overlap or text_overlap:
            raise ValueError(f"Split leakage detected between {left} and {right}")

    summary = {
        "dataset": "mixed_session_v8_without_jump_throw_8class",
        "rows": len(split),
        "counts": {name: len(indices[name]) for name in SPLIT_NAMES},
        "unique_text_ids": {name: len(text_sets[name]) for name in SPLIT_NAMES},
        "overlap": overlap,
        "note": (
            "The split is text-disjoint, but motion class and acquisition session "
            "remain confounded."
        ),
    }
    return indices, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output containing row indices and the audit summary.",
    )
    args = parser.parse_args()

    indices, summary = load_splits(args.h5)
    print(json.dumps(summary, indent=2))
    if args.output is not None:
        payload = {
            "summary": summary,
            "indices": {name: values.tolist() for name, values in indices.items()},
        }
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
