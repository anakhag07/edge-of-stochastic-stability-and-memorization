from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.data import generate_prototype_sets, prepare_dataset
from utils.input_prototypes import (
    build_input_prototype_metadata,
    build_input_subset_counts,
    save_input_prototype_package,
    select_input_prototype_subsets,
)
from utils.storage import create_plaintext_run_dir


def main():
    parser = argparse.ArgumentParser(description="Generate deterministic input-space prototype sets.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num-data", type=int, required=True)
    parser.add_argument("--classes", type=int, nargs="+", required=True)
    parser.add_argument("--dataset-seed", type=int, default=888)
    parser.add_argument("--input-boundary", type=int, default=None, help="Boundary prototypes per class")
    parser.add_argument("--input-inliers", type=int, default=None, help="Inlier prototypes per class")
    parser.add_argument("--input-x-outliers", type=int, default=None, help="X-outlier prototypes per class")
    parser.add_argument("--input-y-outliers", type=int, default=None, help="Y-outlier prototypes per class")
    args = parser.parse_args()

    if len(args.classes) != 2:
        raise ValueError("Input prototypes require exactly two classes via --classes")

    counts_by_subset = build_input_subset_counts(args)
    if not counts_by_subset:
        raise ValueError("Provide at least one of --input-boundary, --input-inliers, --input-x-outliers, or --input-y-outliers")

    for flag_name in ("input_boundary", "input_inliers", "input_x_outliers", "input_y_outliers"):
        flag_value = getattr(args, flag_name)
        if flag_value is not None and flag_value < 1:
            raise ValueError(f"--{flag_name.replace('_', '-')} must be >= 1 when provided")

    if "DATASETS" not in os.environ:
        raise ValueError("Please set the environment variable 'DATASETS'. Use 'export DATASETS=/path/to/datasets'")
    if "RESULTS" not in os.environ:
        raise ValueError("Please set the environment variable 'RESULTS'. Use 'export RESULTS=/path/to/results'")

    dataset_root = Path(os.environ.get("DATASETS"))
    results_root = Path(os.environ.get("RESULTS"))

    train_x, train_y, _, _ = prepare_dataset(
        args.dataset,
        dataset_root,
        args.num_data,
        args.classes,
        args.dataset_seed,
        loss_type="mse",
    )

    proto_classes = (0, 1) if args.dataset == "cifar10_2cls" else tuple(args.classes)
    n_boundary = counts_by_subset.get("boundary")
    n_inlier_pool = (
        counts_by_subset.get("inliers", 0)
        + counts_by_subset.get("x_outlier", 0)
        + counts_by_subset.get("y_outlier", 0)
    ) or None
    prototypes, indices = generate_prototype_sets(
        train_x,
        train_y,
        proto_classes,
        n_prototype=max(n_boundary or 0, n_inlier_pool or 0),
        n_boundary=n_boundary,
        n_inlier=n_inlier_pool,
        return_indices=True,
    )
    # Package only stores the raw boundary and inlier pools. training.py
    # rebuilds x_outlier / y_outlier at run time by partitioning the inlier
    # pool into disjoint slices and extrapolating x_outlier from the
    # post-holdout centroid; the helper lives in training.py next to the
    # other injection logic.
    saved_counts = {"boundary": n_boundary} if n_boundary else {}
    if n_inlier_pool:
        saved_counts["inliers"] = n_inlier_pool
    prototypes = {k: v for k, v in prototypes.items() if k in saved_counts}
    prototypes, indices = select_input_prototype_subsets(
        prototypes,
        indices,
        classes=proto_classes,
        counts_by_subset=saved_counts,
    )

    metadata = build_input_prototype_metadata(
        dataset=args.dataset,
        classes=args.classes,
        dataset_seed=args.dataset_seed,
        num_data=args.num_data,
        loss_type="mse",
        counts_by_subset=saved_counts,
    )

    run_dir = create_plaintext_run_dir(results_root, args.dataset, "geometric", tag="input_prototypes")
    target_dir = save_input_prototype_package(run_dir, prototypes, indices=indices, metadata=metadata)
    print(f"Saved input prototypes to {target_dir}")


if __name__ == "__main__":
    main()
