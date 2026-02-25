from __future__ import annotations

import argparse
import os
from pathlib import Path

from utils.data import prepare_dataset, generate_prototype_sets, trim_prototype_sets
from utils.storage import create_plaintext_run_dir
from utils.input_prototypes import save_input_prototype_package


def _build_counts(args):
    counts = {}
    if args.input_prototypes_boundary_count is not None:
        counts["boundary"] = args.input_prototypes_boundary_count
    if args.input_prototypes_inliers_count is not None:
        counts["inliers"] = args.input_prototypes_inliers_count
    if args.input_prototypes_x_outlier_count is not None:
        counts["x_outlier"] = args.input_prototypes_x_outlier_count
    if args.input_prototypes_y_outlier_count is not None:
        counts["y_outlier"] = args.input_prototypes_y_outlier_count
    return counts


def main():
    parser = argparse.ArgumentParser(description="Generate deterministic input-space prototype sets.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True,
                        help="Model name used for results folder naming (does not affect prototypes).")
    parser.add_argument("--num-data", type=int, required=True)
    parser.add_argument("--classes", type=int, nargs="+", required=True)
    parser.add_argument("--dataset-seed", type=int, default=888)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "ce"])
    parser.add_argument("--input-prototypes-frac", type=float, default=None)
    parser.add_argument("--input-prototypes-count", type=int, default=None)
    parser.add_argument("--input-prototypes-boundary-count", type=int, default=None)
    parser.add_argument("--input-prototypes-inliers-count", type=int, default=None)
    parser.add_argument("--input-prototypes-x-outlier-count", type=int, default=None)
    parser.add_argument("--input-prototypes-y-outlier-count", type=int, default=None)
    parser.add_argument("--tag", type=str, default="input_prototypes",
                        help="Suffix appended to the run folder name.")
    args = parser.parse_args()

    if len(args.classes) != 2:
        raise ValueError("Input prototypes require exactly two classes via --classes")

    if args.input_prototypes_count is not None and args.input_prototypes_count < 1:
        raise ValueError("--input-prototypes-count must be >= 1 when provided")
    if args.input_prototypes_frac is not None:
        if args.input_prototypes_frac <= 0 or args.input_prototypes_frac >= 1:
            raise ValueError("--input-prototypes-frac must be in (0, 1)")
    if args.input_prototypes_count is not None and args.input_prototypes_frac is not None:
        raise ValueError("Provide only one of --input-prototypes-count or --input-prototypes-frac")

    for flag_name in (
        "input_prototypes_boundary_count",
        "input_prototypes_inliers_count",
        "input_prototypes_x_outlier_count",
        "input_prototypes_y_outlier_count",
    ):
        flag_value = getattr(args, flag_name)
        if flag_value is not None and flag_value < 1:
            raise ValueError(f"--{flag_name.replace('_', '-')} must be >= 1 when provided")

    if "DATASETS" not in os.environ:
        raise ValueError("Please set the environment variable 'DATASETS'. Use 'export DATASETS=/path/to/datasets'")
    if "RESULTS" not in os.environ:
        raise ValueError("Please set the environment variable 'RESULTS'. Use 'export RESULTS=/path/to/results'")

    dataset_root = Path(os.environ.get("DATASETS"))
    results_root = Path(os.environ.get("RESULTS"))

    data = prepare_dataset(
        args.dataset,
        dataset_root,
        args.num_data,
        args.classes,
        args.dataset_seed,
        loss_type=args.loss,
    )
    train_x, train_y, _, _ = data

    if args.input_prototypes_count is not None:
        n_proto = args.input_prototypes_count
    elif args.input_prototypes_frac is not None:
        n_proto = max(1, int(round(train_x.shape[0] * args.input_prototypes_frac)))
    else:
        n_proto = max(1, int(round(train_x.shape[0] * 0.05)))
    
    proto_classes = (0, 1) if args.dataset == 'cifar10_2cls' else tuple(args.classes)
    prototypes, indices = generate_prototype_sets(
        train_x,
        train_y,
        proto_classes,
        n_prototype=n_proto,
        return_indices=True,
    )

    counts_by_subset = _build_counts(args)
    prototypes, indices = trim_prototype_sets(
        prototypes,
        proto_classes,
        counts_by_subset,
        indices,
    )

    metadata = {
        "dataset": args.dataset,
        "classes": list(args.classes),
        "dataset_seed": args.dataset_seed,
        "num_data": args.num_data,
        "loss_type": args.loss,
        "sizing": {
            "base_count": int(n_proto),
            "counts_by_subset": counts_by_subset,
        },
    }

    run_dir = create_plaintext_run_dir(results_root, args.dataset, args.model, tag=args.tag)
    target_dir = save_input_prototype_package(run_dir, prototypes, indices=indices, metadata=metadata)
    print(f"Saved input prototypes to {target_dir}")


if __name__ == "__main__":
    main()
