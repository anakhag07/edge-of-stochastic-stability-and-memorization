import torch as T
import torch
import torch.nn as nn
import os
from einops import rearrange, repeat
from torch import linalg as LA
import numpy as np
from copy import deepcopy
from pathlib import Path
import math
import random
import argparse
from typing import Dict, List, Tuple, Optional

import torch.nn.functional as F
import time
import torch.optim as optim

from utils.data import (
    prepare_dataset,
    get_dataset_presets,
    generate_prototype_sets,
    EXTRAPOLATION_FACTOR,
    _select_indices_by_class,
)
from utils.nets import SquaredLoss, MLP, CNN, prepare_net, initialize_net, prepare_optimizer, get_model_presets
from utils.nets import ResNet
from utils.storage import initialize_folders
from utils.input_prototypes import (
    build_input_subset_counts,
    parse_input_prototype_source,
    resolve_input_prototype_path,
    load_input_prototype_package,
    select_input_prototype_subsets,
)
from utils.wandb_utils import (
    init_wandb,
    log_metrics,
    save_checkpoint_wandb,
    find_closest_checkpoint_wandb,
    load_checkpoint_wandb,
    get_checkpoint_dir_for_run,
    is_wandb_available,
    generate_run_id,

)

from utils.noise import gd_with_noise, GradStorage, sde_integration
from utils.measure import *
from utils.measure import (
    compute_train_test_gap_from_tensors,
    compute_subset_metrics_from_tensors,
    compute_subset_metrics,
)
from utils.frequency import frequency_calculator, MeasurementContext
from utils.quadratic import QuadraticApproximation, flatten_params, set_model_params, unflatten_params
from utils.training_cli import build_parser as cli_build_parser, parse_args_with_config as cli_parse_args_with_config

from torch.autograd import grad
import json

DATASET_FOLDER = Path(os.environ['DATASETS']) if 'DATASETS' in os.environ else None
# export RESULTS=/scratch/gpfs/andreyev/eoss/results
RES_FOLDER = Path(os.environ['RESULTS']) if 'RESULTS' in os.environ else None

KNN_TRACKING_METRICS = ["full_loss", "accuracy", "lambda_max", "grad_hessian_grad", "batch_sharpness","grad_vmax_cos2", "grad_norm"]
TRAIN_OUTLIER_TRACKING_METRICS = [
    "per_example_loss_mean",
    "per_example_loss_std",
    "lambda_max",
    "grad_hessian_grad",
    "batch_sharpness",
]


def _refresh_runtime_paths() -> None:
    global DATASET_FOLDER, RES_FOLDER

    if 'DATASETS' not in os.environ:
        raise ValueError("Please set the environment variable 'DATASETS'. Use 'export DATASETS=/path/to/datasets'")
    if 'RESULTS' not in os.environ:
        raise ValueError("Please set the environment variable 'RESULTS'. Use 'export RESULTS=/path/to/results'")

    DATASET_FOLDER = Path(os.environ['DATASETS'])
    RES_FOLDER = Path(os.environ['RESULTS'])


def _is_config_comment_key(key: str) -> bool:
    return key.startswith("__comment") or key.startswith("_comment")


def _flatten_config_mapping(config_value, *, source_path: Path, flat_config: Optional[Dict[str, object]] = None):
    if flat_config is None:
        flat_config = {}

    if not isinstance(config_value, dict):
        raise ValueError(f"Config file must contain a JSON object at the top level: {source_path}")

    for key, value in config_value.items():
        if _is_config_comment_key(str(key)):
            continue
        if isinstance(value, dict):
            _flatten_config_mapping(value, source_path=source_path, flat_config=flat_config)
            continue
        if key in flat_config:
            raise ValueError(f"Duplicate config key '{key}' found in {source_path}")
        flat_config[key] = value

    return flat_config


def _load_json_config_defaults(parser: argparse.ArgumentParser, config_path: str) -> Dict[str, object]:
    path = Path(config_path)
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    flat_config = _flatten_config_mapping(payload, source_path=path)
    valid_dests = {action.dest for action in parser._actions}
    unknown_keys = sorted(key for key in flat_config if key not in valid_dests)
    if unknown_keys:
        raise ValueError(
            f"Unknown config key(s) in {path}: {', '.join(unknown_keys)}"
        )
    return flat_config


def _extract_config_path(argv: Optional[List[str]] = None) -> Optional[str]:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, default=None)
    config_args, _ = config_parser.parse_known_args(argv)
    return config_args.config


def parse_args_with_config(parser: argparse.ArgumentParser, argv: Optional[List[str]] = None):
    config_path = _extract_config_path(argv)
    if config_path:
        parser.set_defaults(**_load_json_config_defaults(parser, config_path))
    return parser.parse_args(argv)



def _build_input_prototype_counts(args) -> Dict[str, int]:
    return build_input_subset_counts(args)


def _generation_pool_sizes(counts_by_subset: Dict[str, int]) -> tuple[int | None, int | None, int | None]:
    if not counts_by_subset:
        return None, None, None

    n_boundary = counts_by_subset.get("boundary")
    n_inlier_pool = (
        counts_by_subset.get("inliers", 0)
        + counts_by_subset.get("x_outlier", 0)
        + counts_by_subset.get("y_outlier", 0)
    )
    n_inlier = n_inlier_pool or None
    n_prototype = max(counts_by_subset.values())
    return n_prototype, n_boundary, n_inlier


def _validate_nonempty_prototype_subsets(
    prototype_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    *,
    context: str,
    require_nonempty: bool = False,
):
    if not prototype_data:
        if require_nonempty:
            raise ValueError(f"{context}: no prototype subsets were provided")
        return
    for name, tensors in prototype_data.items():
        if name is None:
            raise ValueError(f"{context}: prototype subset name is None")
        if tensors is None:
            raise ValueError(f"{context}: prototype subset '{name}' is missing tensors")
        if not isinstance(tensors, (tuple, list)) or len(tensors) != 2:
            raise ValueError(
                f"{context}: prototype subset '{name}' must be a (X, Y) tuple"
            )
        X_p, Y_p = tensors
        if X_p is None or Y_p is None:
            raise ValueError(f"{context}: prototype subset '{name}' has missing tensors")
        if not torch.is_tensor(X_p) or not torch.is_tensor(Y_p):
            raise ValueError(
                f"{context}: prototype subset '{name}' must use torch tensors"
            )
        if X_p.shape[0] == 0 or Y_p.shape[0] == 0:
            raise ValueError(
                f"{context}: prototype subset '{name}' is empty; provide nonzero counts"
            )
        if X_p.shape[0] != Y_p.shape[0]:
            raise ValueError(
                f"{context}: prototype subset '{name}' has mismatched X/Y sizes"
            )


def _split_prototype_subset_by_class(
    X: torch.Tensor,
    Y: torch.Tensor,
    classes: Tuple[int, int],
    holdout_count: int,
    seed: int,
    subset_name: str,
):
    if holdout_count <= 0:
        empty = (X[:0], Y[:0])
        full_idx = torch.arange(X.shape[0], dtype=torch.long, device=X.device)
        empty_idx = torch.empty((0,), dtype=torch.long, device=X.device)
        return (X, Y), empty, full_idx, empty_idx

    labels = Y
    if labels.ndim > 1:
        labels = labels.argmax(dim=1)
    labels = labels.to(dtype=torch.long).cpu()

    rng = np.random.default_rng(seed)
    selected = []
    for class_id in classes:
        class_idx = (labels == class_id).nonzero(as_tuple=False).view(-1).cpu().numpy()
        if len(class_idx) == 0:
            print(
                f"Warning: input prototype subset '{subset_name}' has no samples for class {class_id}; "
                "skipping holdout for this class."
            )
            continue
        if len(class_idx) < holdout_count:
            raise ValueError(
                f"Holdout count {holdout_count} for '{subset_name}' exceeds available "
                f"samples ({len(class_idx)}) in class {class_id}."
            )
        chosen = rng.choice(class_idx, holdout_count, replace=False)
        selected.append(torch.tensor(chosen, dtype=torch.long))

    if not selected:
        empty = (X[:0], Y[:0])
        full_idx = torch.arange(X.shape[0], dtype=torch.long, device=X.device)
        empty_idx = torch.empty((0,), dtype=torch.long, device=X.device)
        return (X, Y), empty, full_idx, empty_idx

    holdout_idx = torch.cat(selected, dim=0)
    holdout_idx = torch.unique(holdout_idx, sorted=False).to(device=X.device)

    mask = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
    mask[holdout_idx] = False
    train_idx = mask.nonzero(as_tuple=False).view(-1)

    train_subset = (X.index_select(0, train_idx), Y.index_select(0, train_idx))
    holdout_subset = (X.index_select(0, holdout_idx), Y.index_select(0, holdout_idx))
    return train_subset, holdout_subset, train_idx, holdout_idx


def _split_input_prototype_sets(
    prototypes: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    indices: Optional[Dict[str, torch.Tensor]],
    *,
    classes: Tuple[int, int],
    holdout_counts: Dict[str, int],
    seed: int,
):
    if not prototypes:
        return {}, {}, indices, None

    missing = [name for name in holdout_counts.keys() if name not in prototypes]
    if missing:
        print(
            "Warning: holdout counts specified for subsets missing from prototypes: "
            f"{', '.join(sorted(missing))}."
        )

    train_sets = {}
    holdout_sets = {}
    train_indices = {} if indices is not None else None
    holdout_indices = {} if indices is not None else None

    for idx, (name, (X, Y)) in enumerate(prototypes.items()):
        if X is None or Y is None:
            continue
        holdout_count = holdout_counts.get(name, 0)
        train_subset, holdout_subset, train_idx, holdout_idx = _split_prototype_subset_by_class(
            X,
            Y,
            classes,
            holdout_count,
            seed + idx * 17,
            name,
        )
        train_sets[name] = train_subset
        holdout_sets[name] = holdout_subset

        if indices is not None and name in indices:
            idx_tensor = indices[name]
            if not torch.is_tensor(idx_tensor):
                idx_tensor = torch.tensor(idx_tensor, dtype=torch.long)
            idx_tensor = idx_tensor.to(dtype=torch.long)

            if holdout_idx.numel() == 0:
                train_indices[name] = idx_tensor
                holdout_indices[name] = idx_tensor[:0]
            else:
                holdout_indices[name] = idx_tensor.index_select(0, holdout_idx.to(device=idx_tensor.device))
                train_indices[name] = idx_tensor.index_select(0, train_idx.to(device=idx_tensor.device))

    return train_sets, holdout_sets, train_indices, holdout_indices


def _validate_input_prototype_metadata(metadata: dict, *, dataset: str, classes: List[int], dataset_seed: int, num_data: int):
    if not metadata:
        return
    expected_classes = metadata.get("classes")
    if expected_classes is not None and list(expected_classes) != list(classes):
        raise ValueError(
            f"Input prototype package classes {expected_classes} do not match requested classes {classes}."
        )
    expected_dataset = metadata.get("dataset")
    if expected_dataset is not None and expected_dataset != dataset:
        raise ValueError(
            f"Input prototype package dataset {expected_dataset} does not match requested dataset {dataset}."
        )
    expected_seed = metadata.get("dataset_seed")
    if expected_seed is not None and dataset_seed is not None and int(expected_seed) != int(dataset_seed):
        raise ValueError(
            f"Input prototype package dataset_seed {expected_seed} does not match requested seed {dataset_seed}."
        )
    expected_num_data = metadata.get("num_data")
    if expected_num_data is not None and num_data is not None and int(expected_num_data) != int(num_data):
        raise ValueError(
            f"Input prototype package num_data {expected_num_data} does not match requested num_data {num_data}."
        )


INPUT_PROTOTYPE_TRACKING_MAP = {
    'injected_x_outlier': 'input_space_prototypes/injected_x_outlier',
    'injected_y_outlier': 'input_space_prototypes/injected_y_outlier',
    'injected_inliers':   'input_space_prototypes/injected_inliers',
    'injected_boundary':  'input_space_prototypes/injected_boundary',
    'heldout_x_outlier': 'input_space_prototypes/heldout_x_outlier',
    'heldout_y_outlier': 'input_space_prototypes/heldout_y_outlier',
    'heldout_inliers':   'input_space_prototypes/heldout_inliers',
    'heldout_boundary':  'input_space_prototypes/heldout_boundary',
}


def _coerce_labels_like(ref_labels: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if ref_labels.ndim > 1:
        num_classes = ref_labels.shape[1]
        return F.one_hot(labels.long(), num_classes=num_classes).to(dtype=ref_labels.dtype)
    return labels.to(dtype=ref_labels.dtype)


def _select_outlier_subset_by_class(
    X: torch.Tensor,
    Y: torch.Tensor,
    classes: Tuple[int, int],
    count_per_class: int,
    seed: int,
    subset_name: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if count_per_class < 1:
        raise ValueError(f"{subset_name} count per class must be >= 1")

    labels = Y
    if labels.ndim > 1:
        labels = labels.argmax(dim=1)
    labels = labels.to(dtype=torch.long)

    rng = np.random.default_rng(seed)
    chosen_indices = []
    for class_id in classes:
        class_indices = (labels == class_id).nonzero(as_tuple=False).view(-1).cpu().numpy()
        if len(class_indices) < count_per_class:
            raise ValueError(
                f"{subset_name} only has {len(class_indices)} samples for class {class_id}; "
                f"cannot select {count_per_class}."
            )
        picked = rng.choice(class_indices, count_per_class, replace=False)
        chosen_indices.append(torch.tensor(picked, dtype=torch.long, device=X.device))

    indices = torch.cat(chosen_indices, dim=0)
    return X.index_select(0, indices), Y.index_select(0, indices)


def _drop_indices_from_dataset(
    X: torch.Tensor,
    Y: torch.Tensor,
    indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if indices is None:
        return X, Y
    if not torch.is_tensor(indices):
        indices = torch.tensor(indices, dtype=torch.long)
    indices = indices.to(dtype=torch.long)
    if indices.numel() == 0:
        return X, Y
    unique_idx = torch.unique(indices)
    if unique_idx.numel() == 0:
        return X, Y
    mask = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
    mask[unique_idx.to(device=X.device)] = False
    return X[mask], Y[mask]


def _partition_pool_by_class(X, Y, classes, *slice_sizes):
    labels = Y.argmax(dim=1) if Y.ndim > 1 else Y.long()
    slices = [[] for _ in slice_sizes]
    slice_labels = [[] for _ in slice_sizes]
    for cls in classes:
        cls_indices = (labels == cls).nonzero(as_tuple=False).view(-1)
        offset = 0
        for i, sz in enumerate(slice_sizes):
            if sz > 0:
                sel = cls_indices[offset:offset + sz]
                slices[i].append(X[sel])
                slice_labels[i].append(Y[sel])
                offset += sz
    result = []
    for i, sz in enumerate(slice_sizes):
        if sz > 0 and slices[i]:
            result.append((torch.cat(slices[i], dim=0), torch.cat(slice_labels[i], dim=0)))
        else:
            result.append(None)
    return result


def _build_prototype_injection_subsets(
    *,
    all_prototype_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    all_prototype_indices: Optional[Dict[str, torch.Tensor]],
    input_proto_counts: Dict[str, int],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    classes: Tuple[int, int],
    x_outlier_mode: str = "coherent",
    random_direction_seed: int = 42,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    List[Tuple[torch.Tensor, torch.Tensor]],
    Dict[str, Tuple[torch.Tensor, torch.Tensor]],
]:
    """Hold out source samples and rebuild the per-subset injection tensors.

    Returns the post-holdout training tensors, the augmentation list
    (label format already coerced to match ``train_y``), and a tracking
    dict with per-subset tensors in their native label format.
    """
    if len(classes) != 2:
        raise ValueError("Prototype injection requires exactly two classes.")

    n_inlier_inject = input_proto_counts.get("inliers", 0) or 0
    n_y_outlier = input_proto_counts.get("y_outlier", 0) or 0
    n_x_outlier = input_proto_counts.get("x_outlier", 0) or 0
    n_boundary_inject = input_proto_counts.get("boundary", 0) or 0

    pool_indices = all_prototype_indices or {}
    X_inlier_pool, Y_inlier_pool = all_prototype_data["inliers"]
    X_boundary_pool, Y_boundary_pool = all_prototype_data["boundary"]

    inlier_labels = Y_inlier_pool.argmax(dim=1) if Y_inlier_pool.ndim > 1 else Y_inlier_pool.long()
    boundary_labels = Y_boundary_pool.argmax(dim=1) if Y_boundary_pool.ndim > 1 else Y_boundary_pool.long()

    inlier_needed_per_class = n_inlier_inject + n_y_outlier + n_x_outlier
    for cls in classes:
        n_avail_inlier = int((inlier_labels == cls).sum().item())
        if n_avail_inlier < inlier_needed_per_class:
            raise ValueError(
                f"Inlier pool has {n_avail_inlier} samples for class {cls} but injection "
                f"needs {inlier_needed_per_class} disjoint samples (inliers={n_inlier_inject} "
                f"+ y_outlier={n_y_outlier} + x_outlier={n_x_outlier}). Regenerate the prototype "
                f"package with per-class counts matching the requested injection."
            )
        n_avail_boundary = int((boundary_labels == cls).sum().item())
        if n_avail_boundary < n_boundary_inject:
            raise ValueError(
                f"Boundary pool has {n_avail_boundary} samples for class {cls} but injection "
                f"needs {n_boundary_inject}."
            )

    # Hold out source samples from train_x before computing the centroid so that
    # v_diff for x_outlier extrapolation reflects the post-holdout class means.
    holdout_tensors = []
    boundary_holdout_idx = pool_indices.get("boundary") if pool_indices else None
    inlier_holdout_idx = pool_indices.get("inliers") if pool_indices else None
    if boundary_holdout_idx is not None and n_boundary_inject > 0:
        per_class = _select_indices_by_class(Y_boundary_pool, classes, n_boundary_inject)
        holdout_tensors.append(boundary_holdout_idx.index_select(0, per_class.cpu()))
    if inlier_holdout_idx is not None and inlier_needed_per_class > 0:
        per_class = _select_indices_by_class(Y_inlier_pool, classes, inlier_needed_per_class)
        holdout_tensors.append(inlier_holdout_idx.index_select(0, per_class.cpu()))
    if holdout_tensors:
        holdout_indices = torch.unique(torch.cat(holdout_tensors, dim=0))
        if holdout_indices.numel() > 0:
            orig_n = train_x.shape[0]
            train_x, train_y = _drop_indices_from_dataset(train_x, train_y, holdout_indices)
            removed = orig_n - train_x.shape[0]
            print(
                f"[holdout] removed {removed} samples from training set for input prototype "
                f"injection ({holdout_indices.numel()} unique indices)"
            )

    inlier_slice, y_outlier_source, x_outlier_source = _partition_pool_by_class(
        X_inlier_pool, Y_inlier_pool, classes,
        n_inlier_inject, n_y_outlier, n_x_outlier,
    )

    outlier_augments: List[Tuple[torch.Tensor, torch.Tensor]] = []
    train_outlier_tracking: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    if inlier_slice is not None:
        X_in, Y_in = inlier_slice
        train_outlier_tracking["inliers"] = (X_in, Y_in)
        outlier_augments.append((X_in, _coerce_labels_like(train_y, Y_in)))

    if y_outlier_source is not None:
        X_ysrc, Y_ysrc = y_outlier_source
        y_labels = Y_ysrc.argmax(dim=1) if Y_ysrc.ndim > 1 else Y_ysrc.long()
        flipped = torch.where(y_labels == classes[0], classes[1], classes[0])
        train_outlier_tracking["y_outlier"] = (X_ysrc, flipped)
        outlier_augments.append((X_ysrc, _coerce_labels_like(train_y, flipped)))

    if x_outlier_source is not None:
        X_xsrc, Y_xsrc = x_outlier_source
        x_labels = Y_xsrc.argmax(dim=1) if Y_xsrc.ndim > 1 else Y_xsrc.long()
        labels_all = train_y.argmax(dim=1) if train_y.ndim > 1 else train_y.long()
        mask_0 = labels_all == classes[0]
        mask_1 = labels_all == classes[1]
        c0 = train_x[mask_0].view(int(mask_0.sum()), -1).mean(dim=0, keepdim=True)
        c1 = train_x[mask_1].view(int(mask_1.sum()), -1).mean(dim=0, keepdim=True)
        v_diff = c1 - c0
        X_flat = X_xsrc.view(X_xsrc.shape[0], -1)
        extrapolated = torch.zeros_like(X_flat)
        displacement_norm = EXTRAPOLATION_FACTOR * v_diff.norm().item()

        if x_outlier_mode == "random_direction":
            rng = torch.Generator()
            rng.manual_seed(random_direction_seed)
            D = X_flat.shape[1]
            v_unit = v_diff.view(-1) / v_diff.norm()
            for i in range(X_flat.shape[0]):
                z = torch.randn(D, generator=rng)
                z = z - (z @ v_unit) * v_unit
                z = z / z.norm()
                extrapolated[i] = X_flat[i] + displacement_norm * z
        else:
            for i in range(X_flat.shape[0]):
                if x_labels[i] == classes[0]:
                    extrapolated[i] = X_flat[i] - EXTRAPOLATION_FACTOR * v_diff
                else:
                    extrapolated[i] = X_flat[i] + EXTRAPOLATION_FACTOR * v_diff
        X_x_out = extrapolated.view_as(X_xsrc)
        Y_x_out = x_labels.clone()
        train_outlier_tracking["x_outlier"] = (X_x_out, Y_x_out)
        outlier_augments.append((X_x_out, _coerce_labels_like(train_y, Y_x_out)))

    if n_boundary_inject > 0:
        boundary_slices = _partition_pool_by_class(
            X_boundary_pool, Y_boundary_pool, classes, n_boundary_inject,
        )
        if boundary_slices[0] is not None:
            X_b, Y_b = boundary_slices[0]
            train_outlier_tracking["boundary"] = (X_b, Y_b)
            outlier_augments.append((X_b, _coerce_labels_like(train_y, Y_b)))

    return train_x, train_y, outlier_augments, train_outlier_tracking


def prepare_train_outlier_subset_configs(
    outlier_data: dict,
    base_batch_size: int,
) -> List[dict]:
    if not outlier_data:
        return []

    configs = []
    for name, tensors in outlier_data.items():
        if name not in ("x_outlier", "y_outlier"):
            continue
        X_p, Y_p = tensors
        if X_p is None or Y_p is None:
            continue
        log_prefix = INPUT_PROTOTYPE_TRACKING_MAP.get(name, f"train_outlier/{name}")
        X_cpu = X_p.detach().cpu()
        Y_cpu = Y_p.detach().cpu()
        if X_cpu.numel() == 0:
            continue
        batch_size = min(base_batch_size, X_cpu.shape[0]) if base_batch_size else X_cpu.shape[0]
        configs.append({
            "enabled": True,
            "subsets": [{
                "name": name,
                "class_id": None,
                "X_tensor": X_cpu,
                "Y_tensor": Y_cpu,
            }],
            "metrics": list(TRAIN_OUTLIER_TRACKING_METRICS),
            "metric_kwargs": {
                "batch_sharpness": {
                    "batch_size": batch_size,
                    "n_estimates": 1,
                    "min_estimates": 1,
                    "eps": 1.0,
                }
            },
            "log_prefix": log_prefix,
        })
    return configs



def prepare_prototype_subset_configs(prototype_data: dict, base_batch_size: int) -> List[dict]:
    """
    Reuse the subset-tracking machinery to log prototype metrics instead of relying on
    the legacy compute_prototype_metrics helper.
    """
    if not prototype_data:
        return []

    configs = []
    for name, tensors in prototype_data.items():
        if name is None or tensors is None:
            continue
        if name.startswith("feature_"):
            # Feature-space prototypes are tracked separately with their own prefixes.
            continue
        log_prefix = INPUT_PROTOTYPE_TRACKING_MAP.get(name)
        if log_prefix is None:
            log_prefix = f"prototype/{name}"
        X_p, Y_p = tensors
        if X_p is None or Y_p is None:
            continue
        X_cpu = X_p.detach().cpu()
        Y_cpu = Y_p.detach().cpu()
        if X_cpu.numel() == 0:
            continue
        batch_size = min(base_batch_size, X_cpu.shape[0]) if base_batch_size else X_cpu.shape[0]
        metrics = list(KNN_TRACKING_METRICS)
        if "batch_sharpness" not in metrics:
            metrics.append("batch_sharpness")
        configs.append({
            "enabled": True,
            "subsets": [{
                "name": name,
                "class_id": None,
                "X_tensor": X_cpu,
                "Y_tensor": Y_cpu,
            }],
            "metrics": metrics,
            "metric_kwargs": {
                "batch_sharpness": {
                    "batch_size": batch_size,
                    "n_estimates": 1,
                    "min_estimates": 1,
                    "eps": 1.0,
                }
            },
            "log_prefix": log_prefix,
        })
    return configs



"""
# -------------------------------------
# NEW: Sharpness for Prototypes (legacy helper, superseded by subset tracking)
# ------------------------------------

def compute_prototype_metrics(net, loss_fn, prototype_data, device, base_batch_size=32):
    \"\"\"
    Legacy helper for logging prototype metrics directly. Subset tracking now
    handles prototype logging to keep behavior consistent across tracked sets.
    \"\"\"
    metrics = {}

    for name, (X_p, Y_p) in prototype_data.items():
        X_p = X_p.to(device)
        Y_p = Y_p.to(device)
        n = X_p.shape[0]
        batch_size = min(base_batch_size, n)

        with torch.no_grad():
            logits = net(X_p)
            loss_val = loss_fn(logits, Y_p).item()
        metrics[f\"prototype/{name}/loss\"] = loss_val

        proto_batch_sharp = calculate_averaged_grad_H_grad(
            net=net,
            X=X_p,
            Y=Y_p,
            loss_fn=loss_fn,
            batch_size=batch_size,
            n_estimates=1,
            min_estimates=1,
            eps=1.0,
            expectation_inside=False,
            with_replacement=False,
            return_confidence_interval=False,
        )
        metrics[f\"prototype/{name}/batch_sharpness\"] = float(proto_batch_sharp)

    return metrics
"""



def _format_duration(seconds):
    seconds = max(int(seconds), 0)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"



def _trainable_params(net):
    return [p for p in net.parameters() if p.requires_grad]

def _flatten_grads_like_params(grads, params):
    flats = []
    for g, p in zip(grads, params):
        if g is None:
            flats.append(torch.zeros_like(p).reshape(-1))
        else:
            flats.append(g.reshape(-1))
    return torch.cat(flats) if flats else torch.tensor([])

@torch.no_grad()
def _adam_pinv_sqrt_flat(optimizer, params, bias_correction=True):
    """
    Build diagonal P^{-1/2} for Adam/AdamW, matching denom = sqrt(v_hat) + eps.
      P = sqrt(v_hat) + eps
      P^{-1/2} = 1 / sqrt(P)
    """
    p2group = {}
    for pg in optimizer.param_groups:
        beta2 = pg.get("betas", (0.9, 0.999))[1]
        eps = pg.get("eps", 1e-8)
        for p in pg["params"]:
            p2group[p] = (beta2, eps)

    chunks = []
    for p in params:
        beta2, eps = p2group.get(p, (0.999, 1e-8))
        st = optimizer.state.get(p, {})
        v = st.get("exp_avg_sq", None)
        if v is None:
            chunks.append(torch.ones(p.numel(), device=p.device, dtype=p.dtype))
            continue

        v_hat = v
        if bias_correction:
            step = st.get("step", 0)
            if isinstance(step, torch.Tensor):
                step = int(step.item())
            step = max(int(step), 1)
            v_hat = v / (1.0 - (beta2 ** step))

        P_diag = v_hat.sqrt().add(eps)
        chunks.append(P_diag.rsqrt().reshape(-1))

    return torch.cat(chunks)


def _make_hvp(loss, params):
    """
    Hessian-vector product H @ v (flattened), using autograd.
    """
    grads = torch.autograd.grad(
        loss, params, create_graph=True, retain_graph=True, allow_unused=True
    )
    g_flat = _flatten_grads_like_params(grads, params)

    def hvp(v_flat):
        dot = (g_flat * v_flat).sum()
        hv = torch.autograd.grad(
            dot, params, retain_graph=True, allow_unused=True
        )
        return _flatten_grads_like_params(hv, params).detach()

    return hvp

def _power_iter_top_eig(operator, dim, iters=20, tol=1e-4, device=None, dtype=None, init_v=None):
    device = device or "cpu"
    dtype = dtype or torch.float32

    if init_v is None:
        v = torch.randn(dim, device=device, dtype=dtype)
    else:
        v = init_v.detach()
        if v.numel() != dim:
            v = torch.randn(dim, device=device, dtype=dtype)
        else:
            v = v.to(device=device, dtype=dtype)

    v = v / (v.norm() + 1e-12)

    last = None
    for _ in range(iters):
        w = operator(v)
        v = w / (w.norm() + 1e-12)

        Av = operator(v)
        lam = (v * Av).sum().item()

        if last is not None and abs(lam - last) / (abs(last) + 1e-12) < tol:
            last = lam
            break
        last = lam

    return last, v.detach()

def compute_adam_precond_lmax(
    net, optimizer, loss_fn, X_probe, Y_probe, *,
    power_iters=50, prev_vec=None, bias_correction=True, clamp_pinv_sqrt_max=1e3,
):
    params = _trainable_params(net)
    pinv_sqrt = _adam_pinv_sqrt_flat(optimizer, params, bias_correction=bias_correction)
    if clamp_pinv_sqrt_max is not None:
        pinv_sqrt = torch.clamp(pinv_sqrt, max=float(clamp_pinv_sqrt_max))

    was_training = net.training
    net.eval()

    out = net(X_probe)
    loss_probe = loss_fn(out, Y_probe)

    hvp = _make_hvp(loss_probe, params)

    def A(u):
        u2 = pinv_sqrt * u
        return pinv_sqrt * hvp(u2)

    lam, vec = _power_iter_top_eig(
        A, dim=pinv_sqrt.numel(),
        iters=power_iters,
        device=pinv_sqrt.device,
        dtype=pinv_sqrt.dtype,
        init_v=prev_vec,
    )

    if was_training: net.train()
    return float(lam), vec




# -------------------------------------
# Section: Measurement Runner
# -------------------------------------

class MeasurementRunner:
    """Centralized measurement orchestration for the training loop."""
    _FIRST_FEW_FULL_LOSS_STEPS = 32

    def __init__(
        self,
        *,
        net,
        loss_fn,
        full_inputs,
        test_inputs = None,
        measurements,
        device,
        batch_size,
        save_dir,
        eigenvector_cache,
        num_eigenvalues,
        use_power_iteration,
        precise_plots,
        rare_measure,
        param_reference,
        step_to_start,
        sde_enabled,
        gd_noise,
        proj_switch_step,
        quad_approx,
        # NEW:
        prototype_data,
        full_inputs_test=None,
        subset_tracking_cfgs=None,
        per_sample_cfg=None,
        log_every_step: bool = False,
        dense_window: tuple = None,
        precond_pi_vec = None,
    ):
        self.net = net
        self.loss_fn = loss_fn
        self.X, self.Y = full_inputs

        # NEW: fixed probe batch for preconditioned Adam sharpness
        self._precond_probe_size = 4096
        n = len(self.X)
        g = torch.Generator(device="cpu")
        g.manual_seed(12345)
        m = min(self._precond_probe_size, n)
        self._precond_probe_idx = torch.randperm(n, generator=g)[:m]

        # New: Test inputs handling
        if test_inputs is not None:
            self.X_test, self.Y_test = test_inputs
        else:
            self.X_test, self.Y_test = None, None

        self.measurements = measurements
        self.device = device
        self.batch_size = batch_size
        self.eigenvector_cache = eigenvector_cache
        self.num_eigenvalues = num_eigenvalues
        self.use_power_iteration = use_power_iteration
        self.precise_plots = precise_plots
        self.rare_measure = rare_measure
        self.param_reference = param_reference
        self.step_to_start = step_to_start
        self.sde_enabled = sde_enabled
        self.gd_noise = gd_noise
        self.proj_switch_step = proj_switch_step
        self.quad_approx = quad_approx
        self._precond_pi_vec = precond_pi_vec

        self.full_inputs_test = full_inputs_test
        self.per_sample_cfg = per_sample_cfg
        self.log_every_step = log_every_step
        self.dense_window = dense_window

        # NEW: Per-sample initialization logic
        if per_sample_cfg is not None and per_sample_cfg.get('enabled', False):
            self.per_sample_histograms = PerSampleHistograms(
                min_log10=per_sample_cfg['hist_min_log10'],
                max_log10=per_sample_cfg['hist_max_log10'],
                bins=per_sample_cfg['hist_bins'],
                metrics=per_sample_cfg.get('metrics', per_sample_cfg.get('hist_metrics', ['loss'])))
            # Initialize bin edges for log10 histogram
            self.bin_edges = np.linspace(
                per_sample_cfg['hist_min_log10'],
                per_sample_cfg['hist_max_log10'],
                per_sample_cfg['hist_bins'] + 1
            )
            # Initialize directories for saving
            self.ps_dir = save_dir / 'per_sample_histograms'
            self.frames_dir = self.ps_dir / 'frames'
            _ensure_dir(self.ps_dir)
            if not per_sample_cfg.get('no_frames', False):
                _ensure_dir(self.frames_dir)
        else:
            self.per_sample_histograms = None
            self.bin_edges = None
            self.ps_dir = None
            self.frames_dir = None

        self.subset_trackers = []
        if subset_tracking_cfgs:
            for cfg in subset_tracking_cfgs:
                if not cfg or not cfg.get('enabled', False):
                    continue
                tracked_subsets = []
                for subset in cfg.get('subsets', []):
                    indices = subset.get('indices')
                    X_tensor = subset.get('X_tensor')
                    Y_tensor = subset.get('Y_tensor')

                    if indices is None and (X_tensor is None or Y_tensor is None):
                        continue

                    entry = {
                        "name": subset.get('name', f"class_{subset.get('class_id', 'unknown')}"),
                        "class_id": subset.get('class_id'),
                    }

                    if indices is not None:
                        if not torch.is_tensor(indices):
                            indices = torch.tensor(indices, dtype=torch.long, device=self.X.device)
                        else:
                            indices = indices.to(device=self.X.device, dtype=torch.long)
                        if indices.numel() == 0:
                            continue
                        entry["indices"] = indices
                    else:
                        entry["X_tensor"] = X_tensor.detach().clone()
                        entry["Y_tensor"] = Y_tensor.detach().clone()

                    tracked_subsets.append(entry)

                if tracked_subsets:
                    self.subset_trackers.append({
                        "subsets": tracked_subsets,
                        "metrics": cfg.get('metrics', ["full_loss", "accuracy", "lambda_max"]),
                        "metric_kwargs": cfg.get('metric_kwargs', {}),
                        "log_prefix": cfg.get('log_prefix', "subset"),
                    })

        self.eigenvalues_log = []
        if 'lmax' in measurements and num_eigenvalues > 1:
            eigenvalues_path = save_dir / 'eigenvalues.json'
            self.eigenvalues_file = open(eigenvalues_path, 'w')
            self.eigenvalues_file.write('[\n')
        else:
            self.eigenvalues_file = None
        self.prototype_data = prototype_data #NEW
    def close(self):
        if self.eigenvalues_file is not None:
            self.eigenvalues_file.write('\n]')
            self.eigenvalues_file.close()

    def collect(
        self,
        *,
        ctx,
        optimizer,
        X_batch,
        Y_batch,
        epoch,
        step_in_epoch,
        step_number,
    ):
        metrics = {
            'batch_lmax': np.nan,
            'step_sharpness': np.nan,
            'batch_sharpness': np.nan,
            'batch_sharpness_exp_inside': np.nan,
            'fisher_batch_eigenval': np.nan,
            'fisher_total_eigenval': np.nan,
            'full_accuracy': np.nan,
            'full_gHg': np.nan,
            'full_loss': np.nan,
            'gni': np.nan,
            'grad_projections': None,
            'one_step_loss_change': np.nan,
            'param_distance': np.nan,
            'gradient_norm_squared': np.nan,
            'lmax': np.nan,
            'all_eigenvalues': None,
            'quadratic_loss': None,
            'quadratic_loss_gn': None,
            'proj_grad_ratio': None,
            'hessian_trace': np.nan,
            'train_acc': np.nan, # NEW
            'test_acc': np.nan,  # NEW
            'train_test_gap': np.nan, # NEW
            'lmax_precond_adam': np.nan,
            'adam_edge_threshold': np.nan,
            'adam_edge_ratio': np.nan,

        }

        if self.log_every_step:
            ctx.log_all_measurements = True

        epoch_loss_update = None


        # ----- Batch sharpness (expected Rayleigh quotient) -----
        if 'batch_sharpness' in self.measurements:
            if frequency_calculator.should_measure('batch_sharpness', ctx):
                metrics['batch_sharpness'] = calculate_averaged_grad_H_grad_step(
                    self.net,
                    self.X,
                    self.Y,
                    self.loss_fn,
                    batch_size=self.batch_size,
                    n_estimates=1000,
                    min_estimates=20,
                    eps=0.005,
                )
        # ----- Instantaneous step sharpness (current-batch Rayleigh quotient) -----
        if 'step_sharpness' in self.measurements:
            if frequency_calculator.should_measure('step_sharpness', ctx):
                self.net.zero_grad()
                preds = self.net(X_batch).squeeze(dim=-1)
                loss = self.loss_fn(preds, Y_batch)
                metrics['step_sharpness'] = compute_grad_H_grad(loss, self.net).item()

        # ----- Eigenvalues/Lambda max (full batch) -----
        lmax_now = False
        if 'lmax' in self.measurements:
            measurement_type = 'full_batch_lambda_max'
            lmax_now = frequency_calculator.should_measure(measurement_type, ctx)

        if lmax_now:
            if str(self.device).startswith('cuda'):
                torch.cuda.empty_cache()
            optimizer.zero_grad()

            lmax_max_size = 4096
            if str(self.device).startswith('cuda'):
                total_memory = torch.cuda.get_device_properties(0).total_memory
                if total_memory < 20 * 1024**3:
                    if isinstance(self.net, CNN):
                        lmax_max_size = 2048 + 512
                    if isinstance(self.net, ResNet):
                        lmax_max_size = 512

            if len(self.X) > lmax_max_size:
                print(f"Warning: Computing eigenvalues on subset of {lmax_max_size} samples instead of full dataset ({len(self.X)} samples) due to memory/time constraints. Most of the time it is fine, but should be corrected")
                idx = gimme_random_subset_idx(len(self.X), lmax_max_size)
                X_subset = self.X[idx]
                Y_subset = self.Y[idx]
            else:
                X_subset = self.X
                Y_subset = self.Y

            preds = self.net(X_subset).squeeze(dim=-1)
            loss = self.loss_fn(preds, Y_subset)
            if 'lmax_precond_adam' in self.measurements:
                if isinstance(optimizer, (optim.Adam, optim.AdamW)):

                    # NEW: measure on fixed probe batch (not X_subset / random)
                    was_training = self.net.training
                    self.net.eval()

                    idxp = self._precond_probe_idx.to(self.X.device)
                    Xp = self.X[idxp]
                    Yp = self.Y[idxp]

                    preds_p = self.net(Xp).squeeze(dim=-1)
                    loss_p = self.loss_fn(preds_p, Yp)

                    lam, v = compute_adam_precond_lmax(
                        net=self.net,
                        optimizer=optimizer,
                        loss_fn=self.loss_fn,
                        X_probe=Xp,
                        Y_probe=Yp,
                        power_iters=50,
                        prev_vec=self._precond_pi_vec,
                    )

                    metrics["lmax_precond_adam"] = float(lam)
                    self._precond_pi_vec = v

                    beta1 = float(optimizer.param_groups[0].get("betas", (0.9, 0.999))[0])
                    c = 2.0 * (1.0 + beta1) / (1.0 - beta1)  
                    lr = float(optimizer.param_groups[0]["lr"])
                    metrics["adam_edge_threshold"] = c / max(lr, 1e-30)
                    metrics["adam_edge_ratio"] = (lr * float(lam)) / c

                    if was_training:
                        self.net.train()



            if self.eigenvector_cache is not None:
                max_iterations = 100 if not self.use_power_iteration else 1000
                tolerance = 0.005 if self.num_eigenvalues < 6 else 0.03
                if self.precise_plots:
                    max_iterations = 300 if not self.use_power_iteration else 3000
                    tolerance = 0.001 if self.num_eigenvalues < 6 else 0.01

                eigenvalues, eigenvectors = compute_eigenvalues(
                    loss,
                    self.net,
                    k=self.num_eigenvalues,
                    max_iterations=max_iterations,
                    reltol=tolerance,
                    eigenvector_cache=self.eigenvector_cache,
                    return_eigenvectors=True,
                    use_power_iteration=self.use_power_iteration,
                )

                if self.num_eigenvalues == 1:
                    self.eigenvector_cache.store_eigenvector(eigenvectors, eigenvalues.item())
                    lmax_value = eigenvalues
                else:
                    self.eigenvector_cache.store_eigenvectors(
                        [eigenvectors[:, i] for i in range(eigenvectors.shape[1])],
                        eigenvalues.tolist(),
                    )
                    lmax_value = eigenvalues[0]
            else:
                eigenvalues = compute_eigenvalues(
                    loss,
                    self.net,
                    k=self.num_eigenvalues,
                    max_iterations=200,
                    reltol=0.03,
                    use_power_iteration=self.use_power_iteration,
                )
                if self.num_eigenvalues == 1:
                    lmax_value = eigenvalues
                else:
                    lmax_value = eigenvalues[0]

            if self.num_eigenvalues > 1:
                metrics['all_eigenvalues'] = eigenvalues
                if self.eigenvalues_file is not None:
                    eigenvalues_data = {
                        'epoch': epoch,
                        'step': step_number,
                        'eigenvalues': eigenvalues.tolist()
                        if isinstance(eigenvalues, torch.Tensor)
                        else [eigenvalues],
                    }
                    self.eigenvalues_log.append(eigenvalues_data)
                    if len(self.eigenvalues_log) > 1:
                        self.eigenvalues_file.write(',\n')
                    json.dump(eigenvalues_data, self.eigenvalues_file)
                    self.eigenvalues_file.flush()

            metrics['lmax'] = lmax_value.item()
            
            if isinstance(optimizer, (optim.Adam, optim.AdamW)):
                eos_threshold = 38.0 / optimizer.param_groups[0]['lr']
                sharpness_metric = 'lmax_precond_adam'
            elif self.batch_size is not None and self.batch_size < self.X.shape[0]:
                eos_threshold = 2.0 / optimizer.param_groups[0]['lr']
                sharpness_metric = 'batch_sharpness'
            else:
                eos_threshold = 2.0 / optimizer.param_groups[0]['lr']
                sharpness_metric = 'lmax'
            
            if not getattr(self, '_t_star_logged', False) and step_number > 50 and metrics.get(sharpness_metric, float('nan')) >= eos_threshold:
                self._t_star_logged = True
                metrics['t_star'] = step_number
                try:
                    wandb.run.summary["t_star"] = step_number
                    wandb.run.summary["lmax_at_t_star"] = metrics.get(sharpness_metric, float('nan'))
                    wandb.run.summary["eos_threshold"] = eos_threshold
                    wandb.run.summary["sharpness_metric"] = sharpness_metric
                except:
                    pass

            
            metrics['full_loss'] = loss.item()
            metrics['full_accuracy'] = calculate_accuracy(preds, Y_subset)

            epoch_loss_update = metrics['full_loss']

            print(
                f"Epoch {epoch + 1}, Step {step_in_epoch}: Total lambda max = {metrics['lmax']}, "
                f"Loss = {metrics['full_loss']} !!!"
            )

        if 'hessian_trace' in self.measurements:
            if frequency_calculator.should_measure('hessian_trace', ctx):
                metrics['hessian_trace'] = estimate_hessian_trace(
                    self.net,
                    self.X,
                    self.Y,
                    self.loss_fn,
                    max_estimates=256,
                    min_estimates=20,
                    eps=0.01,
                )


        # ----- NEW: Train/Test Gap -----
        if 'train_test_gap' in self.measurements and self.X_test is not None:
            if frequency_calculator.should_measure('train_test_gap',ctx):
                vals = compute_train_test_gap_from_tensors(
                    self.net, self.X, self.Y, self.X_test, self.Y_test
                )
                metrics['train_acc'] = vals['train_acc']
                metrics['test_acc'] = vals['test_acc']
                metrics['train_test_gap'] = vals['gap']

        if self.subset_trackers and frequency_calculator.should_measure('subset_metrics', ctx):
            for tracker in self.subset_trackers:
                metric_kwargs = tracker.get('metric_kwargs', {})
                for subset in tracker['subsets']:
                    if 'indices' in subset:
                        subset_results = compute_subset_metrics(
                            net=self.net,
                            loss_fn=self.loss_fn,
                            X=self.X,
                            Y=self.Y,
                            indices=subset['indices'],
                            metrics=tracker['metrics'],
                            eigenvector_cache=self.eigenvector_cache,
                            num_eigenvalues=self.num_eigenvalues,
                            use_power_iteration=self.use_power_iteration,
                            metric_kwargs=metric_kwargs,
                        )
                    else:
                        subset_results = compute_subset_metrics_from_tensors(
                            net=self.net,
                            loss_fn=self.loss_fn,
                            X_subset=subset['X_tensor'],
                            Y_subset=subset['Y_tensor'],
                            metrics=tracker['metrics'],
                            eigenvector_cache=self.eigenvector_cache,
                            num_eigenvalues=self.num_eigenvalues,
                            use_power_iteration=self.use_power_iteration,
                            metric_kwargs=metric_kwargs,
                        )

                    subset_name = subset.get('name')
                    if subset_name:
                        prefix = f"{tracker['log_prefix']}/{subset_name}"
                    else:
                        prefix = tracker['log_prefix']
                    for key, value in subset_results.items():
                        metrics[f"{prefix}/{key}"] = value


        # ----- Gradient-noise interaction (GNI) -----
        gni_now = False
        if 'gni' in self.measurements:
            gni_now = frequency_calculator.should_measure('gni', ctx)

        if gni_now:
            metrics['gni'] = calculate_gni(
                net=self.net,
                X=self.X,
                Y=self.Y,
                loss_fn=self.loss_fn,
                batch_size=self.batch_size,
                n_estimates=500,
                tolerance=0.05,
            )


        
        # ----- Fisher eigenvalues (total and batch) -----
        if 'fisher' in self.measurements:
            if frequency_calculator.should_measure('fisher_total', ctx):
                metrics['fisher_total_eigenval'] = compute_fisher_eigenvalues(self.net, self.X).item()
            if frequency_calculator.should_measure('fisher_batch', ctx):
                metrics['fisher_batch_eigenval'] = compute_fisher_eigenvalues(self.net, X_batch).item()

        # ----- Parameter distance from reference -----
        if 'param_distance' in self.measurements:
            if self.param_reference is None:
                raise ValueError('Parameter reference must be provided for param_distance measurement')
            if frequency_calculator.should_measure('param_distance', ctx):
                metrics['param_distance'] = calculate_param_distance(self.net, self.param_reference).item()

        # ----- Gradient norm squared estimate -----
        if 'gradient_norm' in self.measurements:
            if frequency_calculator.should_measure('gradient_norm_squared', ctx):
                metrics['gradient_norm_squared'] = calculate_gradient_norm_squared_mc(
                    self.net,
                    self.X,
                    self.Y,
                    self.loss_fn,
                    batch_size=self.batch_size,
                    n_estimates=200,
                    min_estimates=10,
                    eps=0.01,
                )

        # ----- Expected one-step loss change -----
        if 'one_step_loss_change' in self.measurements:
            if frequency_calculator.should_measure('one_step_loss_change', ctx):
                metrics['one_step_loss_change'] = calculate_expected_one_step_full_loss_change(
                    self.net,
                    self.X,
                    self.Y,
                    self.loss_fn,
                    optimizer,
                    batch_size=self.batch_size,
                    n_estimates=1000,
                    min_estimates=10,
                    eps=0.01,
                    use_subset_of_data=2048,
                )

        # ----- Quadratic approximation diagnostics -----
        if self.quad_approx is not None and self.quad_approx.is_active:
            metrics['quadratic_loss'] = self.quad_approx.compute_quadratic_loss_for_logging(self.X, self.Y)

        # ----- Gradient projection diagnostics -----
        grad_projection_now = False
        if 'grad_projection' in self.measurements:
            if self.sde_enabled or self.gd_noise:
                raise Exception('Gradient projection not implemented for SDE or GD with noise')
            grad_projection_now = frequency_calculator.should_measure('grad_projection', ctx)

        if self.proj_switch_step is not None and step_number >= self.proj_switch_step:
            grad_projection_now = True

        if grad_projection_now:
            if not (self.quad_approx is not None and self.quad_approx.is_active):
                if (
                    self.eigenvector_cache is not None
                    and hasattr(self.eigenvector_cache, 'eigenvectors')
                    and len(self.eigenvector_cache.eigenvectors) > 0
                ):
                    params = [p for p in self.net.parameters() if p.requires_grad]
                    full_preds = self.net(self.X).squeeze(dim=-1)
                    full_loss_for_grad = self.loss_fn(full_preds, self.Y)
                    grad_list = torch.autograd.grad(full_loss_for_grad, params, create_graph=False, retain_graph=False)
                    grad_flat = torch.cat([g.reshape(-1) for g in grad_list]).detach()

                    cached_vecs = torch.stack(self.eigenvector_cache.eigenvectors, dim=1).to(grad_flat.device)
                    cached_vals = getattr(self.eigenvector_cache, 'eigenvalues', None)

                    max_k = min(20, self.num_eigenvalues)
                    metrics['grad_projections'] = compute_gradient_projection_ratios(
                        grad_vector=grad_flat,
                        eigvecs=cached_vecs,
                        max_k=max_k,
                        eigenvalues=cached_vals,
                    )

        # ----- batch sharpness (but with the expectation inside) -----
        if 'batch_sharpness_exp_inside' in self.measurements:
            if frequency_calculator.should_measure('batch_sharpness_exp_inside', ctx):
                metrics['batch_sharpness_exp_inside'] = calculate_averaged_grad_H_grad(
                    self.net,
                    self.X,
                    self.Y,
                    self.loss_fn,
                    batch_size=self.batch_size,
                    n_estimates=1000,
                    min_estimates=20,
                    eps=0.005,
                )


        # ----- Batch lambda max -----
        batch_lmax_now = False
        if 'batch_lmax' in self.measurements:
            if self.gd_noise is None:
                batch_lmax_now = frequency_calculator.should_measure('batch_lambda_max', ctx)
            else:
                raise ValueError('Batch lambda max not implemented for GD noise')

        if batch_lmax_now:
            optimizer.zero_grad()
            preds = self.net(X_batch).squeeze(dim=-1)
            loss = self.loss_fn(preds, Y_batch)
            batch_lmax = compute_eigenvalues(loss, self.net, k=1, max_iterations=50, reltol=1e-3)
            metrics['batch_lmax'] = batch_lmax.item()
            print(
                f"Epoch {epoch + 1}, Step {step_in_epoch}: Batch Lambda Max = {metrics['batch_lmax']}, "
                f"Loss = {loss.item()}"
            )


        metrics['epoch_loss_update'] = epoch_loss_update
        return metrics
 

# -------------------------------------
# Section: Training Function
# -------------------------------------


def train(
            net,
            optimizer,
            data, # tuple of X_train, Y_train, X_test, Y_test
            max_epochs,
            max_steps,
            batch_size,
            save_to, #folder
            device,
            verbose=True,
            loss_fn=nn.CrossEntropyLoss(),
            permute=True,
            stop_loss=None,
            epoch_to_start=0,
            step_to_start=0,
            gd_noise=False,
            noise_magnitude=None,
            results_rarely: bool = False,
            measurements: set = {},
            param_reference = None,  # reference weights to measure distance from during training
            cache_eigenvectors: bool = True,  # use eigenvector caching for warm starts
            sde_enabled: bool = False,  # enable SDE integration
            sde_h: float = 0.01,  # SDE integration time step
            sde_eta: float = None,  # SDE learning rate (uses optimizer lr if None)
            sde_seed: int = 888,  # SDE random seed
            use_power_iteration: bool = False,  # Use power iteration for eigenvalue computation
            num_eigenvalues: int = 1,  # Number of eigenvalues to compute
            checkpoint_every_n_steps: int = None,  # Checkpoint frequency 
            quad_switch_step: int = None,  # Step to switch to quadratic approximation
            use_gauss_newton: bool = False,  # Use Gauss-Newton instead of Hessian
            quad_switch_lr: float = None,  # lr to use after switching to quadratic approximation
            precise_plots: bool = False,  # Enable more frequent measurements for precise plotting
            rare_measure: bool = False,  # Make expensive measurements rarer
            # Gradient projection configuration
            proj_switch_step: int = None,  # Step to start projecting minibatch gradients
            proj_top_l: int = None,        # Number of top eigendirections to use for projection
            proj_to_residual: bool = False, # If True, project to orthogonal complement of top-l eigenspace
            wandb_run=None,
            wandb_enabled: bool = False,
            wandb_run_id: str = None,
            subset_tracking_cfgs=None,
            prototype_data=None,
            log_every_step: bool = False,
            dense_window: tuple = None,
            lmax_decay: bool = False,
            lmax_decay_target_lr: float = 0.001,
            lmax_decay_steps: int = 10000,
            lmax_decay_initial_lr: float = None,
            lmax_drop: bool = False,
            lmax_drop_mult: float = 0.5,
            lmax_drop_target_lr: float = None,
            lr_drop_at_step: int = None,
            lr_drop_to: float = None,
    ):
    
    # -------------------------------------
    # Section: Setup
    # -------------------------------------
    start_time = time.time()
    print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    wandb_finished = False

    def _log_train_time():
        nonlocal wandb_finished, wandb_run
        if wandb_run is None or wandb_finished:
            return
        end_time = time.time()
        train_time_s = end_time - start_time
        print(f"Training finished at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Total training time: {train_time_s:.2f} seconds")
        wandb_run.summary["train_time_s"] = float(train_time_s)

    # ----- Checkpoint Frequency Defaults -----
    NET_SAVES_PER_TRAINING = 200

    assert max_epochs is not None or max_steps is not None
    if max_epochs is None:
        # Set very high epoch limit if only max_steps is used
        max_epochs = 100000

    # ----- Dataset Wiring -----
    X_train, Y_train, X_test, Y_test = data

    X, Y = X_train, Y_train

    # ----- LR Drop Setup -----
    lmax_drop_applied = False
    lmax_drop_threshold = 2.0 / optimizer.param_groups[0]['lr']

    # ----- Device Alignment -----
    net = net.to(device)
    net.train()
    net.float()
    
    X = X.to(device)
    Y = Y.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)


    # ----- Storage Preparation -----
    save_to.mkdir(parents=True, exist_ok=True)

    model_save_path = save_to / 'checkpoints'
    results_file = save_to / 'results.txt'
    if device == 'cpu':
        # No buffering on CPU to ensure writes happen immediately
        results_file = open(results_file, 'a', buffering=1)
        torch.set_num_threads(40)
    else:
        # Use buffering on GPU for better performance
        results_file = open(results_file, 'a', buffering=1_000)

    # ----- State Initialization -----
    step_number = -1 if step_to_start == 0 else step_to_start

    if gd_noise is not None:
        grad_storage = GradStorage(net, recalculate_every=30)
    
    # ----- Stochastic Dynamics Setup -----
    if sde_enabled:
        if sde_eta is None:
            sde_eta = optimizer.param_groups[0]['lr']  # Use optimizer's learning rate
        sde_rng = T.Generator(device=device)
        sde_rng.manual_seed(sde_seed)  # Use the provided SDE seed

    # ----- Checkpoint Interval Selection -----
    if checkpoint_every_n_steps is None:
        checkpoint_every_n_steps = max(max_steps // NET_SAVES_PER_TRAINING, 1)
    print(f"Will save checkpoints every {checkpoint_every_n_steps} steps")

    # ----- Quadratic Approximation Setup -----
    quad_approx = None
    if quad_switch_step is not None:
        quad_approx = QuadraticApproximation(net, loss_fn, device, quad_switch_step, use_gauss_newton)
        
        # Handle continuation case - if we're starting after the switch step,
        # we need to initialize the anchor immediately
        if step_to_start >= quad_switch_step:
            print(f"Initializing quadratic approximation at continuation step {step_to_start} (switch was at {quad_switch_step})")
            # Initialize with current model state as anchor
            quad_approx.anchor_params = flatten_params(net).detach().clone()
            quad_approx.anchor_loss = 0.0  # Will be computed on first batch
            quad_approx.delta = torch.zeros_like(quad_approx.anchor_params)
            quad_approx.is_active = True
            print("Quadratic approximation initialized as active for continuation")
        else:
            print(f"Quadratic approximation will switch at step {quad_switch_step}")
    # ----- Training State Trackers -----
    epoch_loss = float('+inf')
    stop_training = False

    # -------------------------------------
    # Section: Measurements
    # -------------------------------------
    # ----- Eigenvector Cache Setup -----
    eigenvector_cache = None
    # Also create cache if gradient projection is enabled, since it relies on cached eigenvectors
    if (('lmax' in measurements or 'grad_projection' in measurements) or (proj_switch_step is not None)) and cache_eigenvectors:
        max_cache = 5
        if num_eigenvalues is not None:
            max_cache = max(max_cache, num_eigenvalues)
        if proj_top_l is not None:
            max_cache = max(max_cache, proj_top_l)
        eigenvector_cache = EigenvectorCache(max_eigenvectors=max_cache)
    
    # ----- Measurement Runner Wiring -----
    measurement_runner = MeasurementRunner(
        net=net,
        loss_fn=loss_fn,
        full_inputs=(X, Y),
        test_inputs=(X_test, Y_test),
        measurements=measurements,
        device=device,
        batch_size=batch_size,
        save_dir=save_to,
        eigenvector_cache=eigenvector_cache,
        num_eigenvalues=num_eigenvalues,
        use_power_iteration=use_power_iteration,
        precise_plots=precise_plots,
        rare_measure=rare_measure,
        param_reference=param_reference,
        step_to_start=step_to_start,
        sde_enabled=sde_enabled,
        gd_noise=gd_noise,
        proj_switch_step=proj_switch_step,
        quad_approx=quad_approx,
        subset_tracking_cfgs=subset_tracking_cfgs,
        prototype_data=prototype_data,    
        full_inputs_test=None,
        log_every_step=log_every_step,
        dense_window=dense_window,
    )
    # ----- Run Identification -----
    run_id = wandb_run_id or generate_run_id()

    # If resuming at/after projection switch step, precompute eigendirections immediately
    if proj_switch_step is not None and step_to_start >= proj_switch_step:
        raise ValueError("Start of grad projection has to be after restart step, not at or before it")

    # -------------------------------------
    # Section: Training Step
    # -------------------------------------
    for epoch in range(epoch_to_start, max_epochs):
        if step_number >= max_steps:
            print(f"Reached max steps {max_steps}, stopping the training")
            results_file.flush()
            results_file.close()
            if wandb_run is not None and not wandb_finished:
                _log_train_time()
                wandb_run.finish()
                wandb_finished = True
            break

        # --- Epoch Data Preparation ---
        shuffle = T.randperm(len(X))
        if permute:
            X_shuffled = X[shuffle]
            Y_shuffled = Y[shuffle]
        else:
            X_shuffled = X
            Y_shuffled = Y

        # Checkpoint saving happens in the step loop now, based on step number

        losses_in_epoch = []
        if stop_training:
            break

        # --- Minibatch Iteration ---
        for i in range(0, len(X) // batch_size): # i runs over steps in a epoch
            step_number += 1

            msg = f"{epoch:03d}, {step_number:05d}, "
            # --- Measurement Context and Sampling ---
            ctx = MeasurementContext(
                step_number=step_number,
                batch_size=batch_size,
                epoch=epoch,
                device=str(device),
                lr=optimizer.param_groups[0]['lr'],
                precise_plots=precise_plots,
                rare_measure=rare_measure,
                log_all_measurements=log_every_step,
                dense_window=measurement_runner.dense_window,
            )

            X_batch = X_shuffled[i*batch_size : (i+1)*batch_size]
            Y_batch = Y_shuffled[i*batch_size : (i+1)*batch_size]

            if permute:
                batch_indices = shuffle[i*batch_size : (i+1)*batch_size]
            else:
                batch_indices = torch.arange(i*batch_size, (i+1)*batch_size, device=device)

            # -------------------------------------
            # Section: Measurements
            # -------------------------------------
            metrics = measurement_runner.collect(
                ctx=ctx,
                optimizer=optimizer,
                X_batch=X_batch,
                Y_batch=Y_batch,
                epoch=epoch,
                step_in_epoch=i,
                step_number=step_number,
            )
            if lmax_drop:
                if args.adam and ('adam_edge_ratio' in metrics):
                    r = metrics.get('adam_edge_ratio', float('nan'))
                    if (not lmax_drop_applied) and math.isfinite(r) and (r >= 1.0):
                        lmax_drop_applied = True
                        old_lr = optimizer.param_groups[0]['lr']
                        new_lr = old_lr * float(lmax_drop_mult)
                        if lmax_drop_target_lr is not None:
                            new_lr = max(new_lr, float(lmax_drop_target_lr))
                        for pg in optimizer.param_groups:
                            pg['lr'] = new_lr
                        print(f"[ADAM LR DROP] step={step_number} edge_ratio={r:.3f} lr: {old_lr:g} -> {new_lr:g}")
                else:
                    lmax_value = metrics.get('lmax', float('nan'))
                    if (not lmax_drop_applied) and math.isfinite(lmax_value) and (lmax_value >= lmax_drop_threshold):
                        lmax_drop_applied = True
                        old_lr = optimizer.param_groups[0]['lr']
                        new_lr = old_lr * float(lmax_drop_mult)
                        if lmax_drop_target_lr is not None:
                            new_lr = max(new_lr, float(lmax_drop_target_lr))
                        for pg in optimizer.param_groups:
                            pg['lr'] = new_lr
                        print(f"[LR DROP] step={step_number} lmax={lmax_value:.4f} thresh={lmax_drop_threshold:.4f} lr: {old_lr:g} -> {new_lr:g}")


            # --- Lmax-based Learning Rate Decay Logic ---
            if lmax_decay:
                lmax_value = metrics.get('lmax', float('nan'))
                if not decay_active and math.isfinite(lmax_value):
                    if lmax_value >= lmax_decay_threshold:
                        decay_active = True
                        decay_start_step = step_number
                        decay_start_lr = optimizer.param_groups[0]['lr']
                        print(
                            f"Lmax decay triggered at step {step_number}: "
                            f"lmax={lmax_value:.4f} >= {lmax_decay_threshold:.4f}"
                        )

                if decay_active:
                    steps_elapsed = step_number - decay_start_step
                    t = min(steps_elapsed / max(lmax_decay_steps, 1), 1.0)
                    new_lr = decay_start_lr + t * (lmax_decay_target_lr - decay_start_lr)
                    for pg in optimizer.param_groups:
                        pg['lr'] = new_lr

            if lmax_drop and lmax_decay:
                raise ValueError("Use either --lmax-drop or --lmax-decay, not both.")

            # --- Scheduled LR drop (for fork runs: exact t* alignment) ---
            if lr_drop_at_step is not None and lr_drop_to is not None:
                if step_number == lr_drop_at_step:
                    old_lr = optimizer.param_groups[0]['lr']
                    for pg in optimizer.param_groups:
                        pg['lr'] = lr_drop_to
                    print(f"[SCHEDULED LR DROP] step={step_number} lr: {old_lr:g} -> {lr_drop_to:g}")

            # --- Epoch-Level Loss Tracking ---
            if metrics['epoch_loss_update'] is not None:
                if math.isnan(metrics['epoch_loss_update']):
                    print('Full loss is NaN, the network prolly diverged, stopping the training')
                    results_file.flush()
                    results_file.close()
                    if wandb_run is not None and not wandb_finished:
                        _log_train_time()
                        wandb_run.finish()
                        wandb_finished = True
                    measurement_runner.close()
                    return
                epoch_loss = metrics['epoch_loss_update']

            if stop_loss is not None and epoch_loss < stop_loss:
                print(f"Loss {epoch_loss} is below the stop loss {stop_loss}, stopping the training")
                stop_training = True
                break

            # -------------------------------------
            # Section: Training Step (Update)
            # -------------------------------------
            optimizer.zero_grad()

            if sde_enabled:
                # SDE integration step - uses full dataset X, Y
                # integrates it for the time [0, eta]
                loss = sde_integration(net=net, X=X, Y=Y, loss_fn=loss_fn, 
                                     batch_size=batch_size, h=sde_h, eta=sde_eta, 
                                     rng=sde_rng)
                
                if math.isinf(loss) or math.isnan(loss):
                    results_file.flush()
                    results_file.close()
                    if wandb_run is not None and not wandb_finished:
                        _log_train_time()
                        wandb_run.finish()
                        wandb_finished = True
                    raise ValueError("Loss is inf or NaN, stopping the training")
                    
                    
            elif gd_noise:
                # this is the GD with noise
                # the whole thing is done in the function, including updating the weights
                loss = gd_with_noise(net=net, X = X, Y=Y, loss_fn=loss_fn, noise_type=gd_noise, 
                                     optimizer=optimizer, batch_size=batch_size, step_number=step_number, 
                                     grad_storage=grad_storage, noise_magnitude=noise_magnitude)
            
            elif quad_approx is not None and quad_approx.is_active:
                # Quadratic approximation dynamics
                quad_gradient = quad_approx.compute_quadratic_gradient(X_batch, Y_batch, batch_indices)
                
                # Get current learning rate from optimizer
                current_lr = optimizer.param_groups[0]['lr']
                if quad_switch_lr is not None:
                    current_lr = quad_switch_lr

                # Update delta using quadratic gradient
                quad_approx.update_delta(current_lr, quad_gradient)
                
                # Set model parameters to current quadratic position
                current_params = quad_approx.get_current_params()
                set_model_params(net, current_params)
                
                # Compute loss for logging (using current model state)
                preds = net(X_batch).squeeze(dim=-1)
                loss = loss_fn(preds, Y_batch)
                
            elif proj_switch_step is not None and step_number >= proj_switch_step:
                # Gradient projection step (only for plain SGD, no momentum/Adam
                    
                # Recompute top-l eigendirections at the requested cadence (default: every step)
                if frequency_calculator.should_measure('proj_eigens_refresh', ctx):
                    ##### TEMP! 
                    full_preds = net(X).squeeze(dim=-1)
                    full_loss_for_eigs = loss_fn(full_preds, Y)
                    _eigvals, eigvecs_block = compute_eigenvalues(
                        full_loss_for_eigs, net,
                        k=proj_top_l if proj_top_l is not None else 1,
                        max_iterations=500,
                        reltol=0.005,
                        eigenvector_cache=eigenvector_cache,
                        return_eigenvectors=True,
                        use_power_iteration=False
                    )

                else:
                    # Use cached eigenvectors
                    if eigenvector_cache is not None and hasattr(eigenvector_cache, 'eigenvectors') and len(eigenvector_cache.eigenvectors) > 0:
                        eigvecs_block = torch.stack(eigenvector_cache.eigenvectors, dim=1).to(device)
                    else:
                        eigvecs_block = None

    

                from utils.lobpcg import _maybe_orthonormalize
                V = eigvecs_block.clone()
                if V.dim() == 1:
                    V = V.unsqueeze(1)
                V = _maybe_orthonormalize(V, assume_ortho=True)

                # --- Projected Step Solve ---
                optimizer.zero_grad()
                params_before_step = flatten_params(net).detach().clone()

                # calculate the step
                preds = net(X_batch).squeeze(dim=-1)

                loss = loss_fn(preds, Y_batch)

                if math.isinf(loss.item()) or math.isnan(loss.item()):
                    results_file.flush()
                    results_file.close()
                    if wandb_run is not None and not wandb_finished:
                        _log_train_time()
                        wandb_run.finish()
                        wandb_finished = True
                    raise ValueError("Loss is inf or NaN, stopping the training")

                # Backward pass for minibatch gradient
                loss.backward()

                optimizer.step()
                
                params_after_step = flatten_params(net).clone()

                # --- Gradient Projection Adjustment ---
                with torch.no_grad():
                    update = params_after_step - params_before_step

                    coeffs = V.T @ update
                    update_in_top = V @ coeffs

                    if proj_to_residual:
                        update_proj = update - update_in_top
                    else:
                        update_proj = update_in_top

                    new_params = params_before_step + update_proj
                    set_model_params(net, new_params)

                    # # Flatten current gradient for logging
                    # params = [p for p in net.parameters() if p.requires_grad]
                    # with torch.no_grad():
                    #     grad_list = [p.grad.reshape(-1) if p.grad is not None else torch.zeros_like(p.reshape(-1)) for p in params]
                    #     g_flat = torch.cat(grad_list).detach().clone()

                    #     coeffs = V.T @ g_flat
                    #     g_in_top = V @ coeffs

                    #     if proj_to_residual:
                    #         g_proj = g_flat - g_in_top
                    #     else:
                    #         g_proj = g_in_top

                    #     denom = torch.linalg.vector_norm(g_flat)
                    #     numer = torch.linalg.vector_norm(g_proj)
                    #     proj_grad_ratio = (numer / (denom + 1e-12)).item()

                    #     projection_basis = V
                    #     params_before_step = flatten_params(net).detach().clone()

            else:
                # Standard SGD step
                preds = net(X_batch).squeeze(dim=-1)

                loss = loss_fn(preds, Y_batch)

                if math.isinf(loss.item()) or math.isnan(loss.item()):
                    results_file.flush()
                    results_file.close()
                    if wandb_run is not None and not wandb_finished:
                        _log_train_time()
                        wandb_run.finish()
                        wandb_finished = True
                    raise ValueError("Loss is inf or NaN, stopping the training")

                # Check if we should initialize quadratic approximation
                if quad_approx is not None:
                    full_dataset = (X, Y) if use_gauss_newton else None
                    quad_approx.initialize_anchor(step_number, loss.item(), full_dataset)

                # Backward pass for minibatch gradient
                loss.backward()

                optimizer.step()


            # Handle loss value (SDE returns float, others return tensor)
            batch_loss = loss if isinstance(loss, float) else loss.item()
            losses_in_epoch.append(batch_loss)

            # --- Checkpoint Handling ---
            # Save checkpoint using wandb system
            ckpt_start_ts = time.time()
            ckpt_start_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ckpt_start_ts))
            checkpoint_path = save_checkpoint_wandb(
                model=net,
                optimizer=optimizer,
                step=step_number,
                epoch=epoch,
                loss=batch_loss,
                run_id=run_id,
                save_every_n_steps=checkpoint_every_n_steps
            )
            ckpt_end_ts = time.time()
            ckpt_end_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ckpt_end_ts))
            if checkpoint_path:
                print(f"Checkpoint start at step {step_number}: {ckpt_start_str}")
                print(f"Checkpoint end at step {step_number}: {ckpt_end_str} (elapsed {ckpt_end_ts - ckpt_start_ts:.2f}s)")
                print(f"Checkpoint saved at step {step_number}: {checkpoint_path}")
                if max_steps is not None and step_number > step_to_start:
                    elapsed = time.time() - start_time
                    steps_done = step_number - step_to_start
                    avg_step_time = elapsed / steps_done
                    remaining_steps = max(max_steps - step_number, 0)
                    eta_seconds = remaining_steps * avg_step_time
                    eta_str = _format_duration(eta_seconds)
                    finish_ts = time.time() + eta_seconds
                    finish_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(finish_ts))
                    print(f"  Estimated time remaining: {eta_str} (finish ~ {finish_str})")

            # -------------------------------------
            # Section: Logging (Step)
            # -------------------------------------
            if True: # not results_rarely or (results_rarely and ghg_now):
                # (0) epoch, (1) step, (2) batch loss, (3) full loss, (4) lambda max, (5) step sharpness, (6) batch sharpness, (7) Gradient-Noise Interaction, (8) total accuracy"""
                # Log metrics   
                msg += (
                    f"{batch_loss:7.6f}, {metrics['full_loss']:7.6f}, {metrics['lmax']:6.2f}, "
                    f"{metrics['step_sharpness']:6.1f}, {metrics['batch_sharpness']:6.1f}, "
                    f"{metrics['gni']:6.2f}, {metrics['full_accuracy']:6.2f},"
                    f"{metrics['train_test_gap']:6.3f}" 
                )
                results_file.write(msg + "\n")
                
                if wandb_enabled:
                    wandb_metrics = metrics.copy()
                    wandb_metrics.update({
                        "epoch": epoch,
                        "step": step_number,
                        "batch_loss": batch_loss,
                        "lr": optimizer.param_groups[0]["lr"],
                    })
                    rename_map = {
                        "batch_lmax": "batch_lambda_max",
                        "lmax": "lambda_max",
                        "batch_sharpness": "batch_sharpn",
                        "full_gHg": "grad_H_grad",
                        "fisher_batch_eigenval": "batch_fisher_eigenval",
                        "fisher_total_eigenval": "total_fisher_eigenval",
                        "gni": "GNI",
                        "full_accuracy": "accuracy",
                        "lmax_precond_adam": "lambda_max_precond_adam",
                        "adam_edge_threshold": "adam_edge_threshold",
                        "adam_edge_ratio": "adam_edge_ratio",

                    }
                    for old_key, new_key in rename_map.items():
                        if old_key in wandb_metrics:
                            wandb_metrics[new_key] = wandb_metrics.pop(old_key)
                    wandb_metrics.pop("epoch_loss_update", None)
                    log_metrics(wandb_metrics)

        
        # --- Epoch Finalization ---
        epoch_loss = np.mean(losses_in_epoch)
        
        results_file.flush()

        
    # -------------------------------------
    # Section: Logging
    # -------------------------------------
    # ----- Final Checkpoint Save -----
    # Save final checkpoint
    final_checkpoint_path = save_checkpoint_wandb(
        model=net,
        optimizer=optimizer,
        step=step_number,
        epoch=epoch,
        loss=batch_loss,
        run_id=run_id,
        save_every_n_steps=1  # Always save final checkpoint
    )
    print(f"Final checkpoint saved: {final_checkpoint_path}")

    results_file.close()

    measurement_runner.close()

    # ----- Final Reporting -----
    if wandb_run is not None and not wandb_finished:
        _log_train_time()
        wandb_run.finish()
        wandb_finished = True


    # ----- Optional Final Measurements -----
    if 'final' in measurements:
        final_file = save_to / 'final.json'
        final_file = open(final_file, 'w') 

        # do the final measurements here - depending on what is needed
    




def build_parser() -> argparse.ArgumentParser:
    return cli_build_parser(EXTRAPOLATION_FACTOR)


if __name__ == '__main__':
    _refresh_runtime_paths()
    # -------------------------------------
    # Section: Runtime Setup
    # -------------------------------------
    # ----- Reproducibility Seeds -----
    # seed = 88881
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = True
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    
    # -------------------------------------
    # Section: Argument Parser
    # -------------------------------------
    parser = cli_build_parser(EXTRAPOLATION_FACTOR)
    args = cli_parse_args_with_config(parser)


    # -------------------------------------
    # Section: Runtime Setup
    # -------------------------------------
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
    # ----- wandb Availability Check -----
    wandb_installed = is_wandb_available()
    if not wandb_installed and not args.disable_wandb:
        print("wandb is not installed; proceeding with logging disabled. Re-run with --disable-wandb to silence this message.")
        args.disable_wandb = True
    elif args.disable_wandb:
        print("wandb logging disabled by flag (--disable-wandb).")

    wandb_enabled = wandb_installed and not args.disable_wandb


    # -------------------------------------
    # Section: Experiment Setup
    # -------------------------------------
    # ----- Argument Post-processing -----
    # --- Parameter Extraction ---
    full_batch_requested = args.batch == 'full'
    batch_size = None if full_batch_requested else args.batch
    dataset = args.dataset
    if args.cpu:
        if T.cuda.is_available():
            print('CUDA is available but running on CPU due to --cpu flag.')
        device = 'cpu'
    else:
        device = T.device('cuda') if T.cuda.is_available() else 'cpu'

    if args.momentum is not None and args.adam:
        raise ValueError("You should provide either momentum or adam, not both")

    if args.momentum is not None and args.momentum < 1e-4 and not args.adam:
        args.momentum = None  # if momentum is too small, just use SGD without momentum

    
    # --- Argument Validation ---
    if args.final:
        raise ValueError("--final needs to be re-implemented")

    if args.param_distance:
        raise NotImplementedError("--param-distance needs to be re-implemented")

    if args.steps is not None and args.epochs is not None:
        raise ValueError("You should provide either epochs or steps, not both")
    if batch_size is not None and batch_size < 1:
        raise ValueError("--batch must be >= 1 when provided as an integer")
    for flag_name in (
        "input_x_outliers",
        "input_y_outliers",
        "input_inliers",
        "input_boundary",
    ):
        flag_value = getattr(args, flag_name)
        if flag_value is not None and flag_value < 1:
            raise ValueError(f"--{flag_name.replace('_', '-')} must be >= 1 when provided")

    input_proto_counts = _build_input_prototype_counts(args)
    input_proto_source = parse_input_prototype_source(args.input_prototype_source)
    if input_proto_counts and input_proto_source["mode"] is None:
        raise ValueError("Input prototype subset counts require --input-prototype-source.")
    if input_proto_source["mode"] not in (None, "generate", "from", "none"):
        raise ValueError("Unsupported input prototype source mode.")
    if input_proto_source["mode"] == "none" and input_proto_counts:
        raise ValueError("--input-prototype-source none cannot be combined with input prototype subset counts.")
    if input_proto_source["mode"] in ("generate", "from") and not input_proto_counts:
        raise ValueError(
            "Provide at least one of --input-boundary, --input-inliers, --input-x-outliers, or --input-y-outliers."
        )

    # Validate gradient projection feature flags and conflicts
    if (args.proj_switch_step is not None) or (args.proj_top_l is not None) or args.proj_to_residual:
        if args.proj_switch_step is None or args.proj_top_l is None:
            raise ValueError("Gradient projection requires both --proj-switch-step and --proj-top-l")
        if args.proj_top_l < 1:
            raise ValueError("--proj-top-l must be a positive integer")
        if args.adam or (args.momentum is not None and args.momentum != 0):
            raise ValueError("Gradient projection currently supports only plain SGD (no momentum/Adam)")
    
    # Validate wandb continuation arguments
    if (args.cont_run_id is not None) != (args.cont_step is not None):
        raise ValueError("Both --cont-run-id and --cont-step must be provided together for wandb continuation")

    # Validate scheduled LR drop arguments
    if (args.lr_drop_at_step is not None) != (args.lr_drop_to is not None):
        raise ValueError("Both --lr-drop-at-step and --lr-drop-to must be provided together")

    # Check for mutually exclusive training modes
    exclusive_modes = []
    if args.proj_switch_step is not None:
        exclusive_modes.append("gradient projection (--proj-switch-step)")
    if args.sde:
        exclusive_modes.append("SDE dynamics (--sde)")
    if args.gd_noise is not None:
        exclusive_modes.append("GD with noise (--gd-noise)")
    if args.quad_switch_step is not None:
        exclusive_modes.append("quadratic approximation (--quad-switch-step)")
    
    if len(exclusive_modes) > 1:
        raise ValueError(f"Cannot use multiple training modes simultaneously: {', '.join(exclusive_modes)}. Please choose only one.")

    # ----- Measurement Selection -----
    measurements = {name for name, enabled in [
    ('lmax', args.lambdamax),
    ('batch_lmax', args.batch_lambdamax),
    ('step_sharpness', args.step_sharpness),
    ('batch_sharpness', args.batch_sharpness),
    ('batch_sharpness_exp_inside', args.batch_sharpness_exp_inside),
    ('grad_projection', args.grad_projection),
    ('gradient_norm', args.gradient_norm),
    ('one_step_loss_change', args.one_step_loss_change),
    ('gni', args.gni),
    ('fisher', args.fisher),
    ('final', args.final),
    ('param_distance', args.param_distance),
    ('hessian_trace', args.hessian_trace),
    ('train_test_gap', args.train_test_gap),
    ('lmax_precond_adam', args.precond_lmax),
    ] if enabled}

    # ----- Result Storage Setup -----
    RES_FOLDER.mkdir(parents=True, exist_ok=True)
    run_folder = initialize_folders(args, RES_FOLDER)
    step_to_start = 0
    
    # ----- Loss Function Selection -----
    if args.loss == 'mse':
        loss_fn = SquaredLoss()
    elif args.loss == 'ce':
        loss_fn = nn.CrossEntropyLoss()

    # ----- Dataset and Model Presets -----
    dataset_presets = get_dataset_presets()
    model_presets = get_model_presets()

    # --- Dataset Preparation ---
    data = prepare_dataset(dataset, DATASET_FOLDER, args.num_data, args.classes, args.dataset_seed, loss_type=args.loss)

    # --- Unpack dataset and build tuple_data ---
    train_x, train_y, test_x, test_y = data  
    tuple_data = (train_x, train_y, test_x, test_y)
    original_train_len = int(train_x.shape[0])


    input_proto_counts = _build_input_prototype_counts(args)
    input_proto_source = parse_input_prototype_source(args.input_prototype_source)
    proto_classes = (0, 1) if dataset == 'cifar10_2cls' else tuple(args.classes)
    selected_prototype_data = {}
    selected_prototype_indices = None
    all_prototype_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    all_prototype_indices: Optional[Dict[str, torch.Tensor]] = None
    if input_proto_source["mode"] == "from":
        proto_path = resolve_input_prototype_path(
            input_proto_source["value"],
            results_root=RES_FOLDER,
            dataset=dataset,
            model=args.model,
        )
        all_prototype_data, all_prototype_indices, proto_meta = load_input_prototype_package(proto_path)
        _validate_input_prototype_metadata(
            proto_meta,
            dataset=dataset,
            classes=args.classes,
            dataset_seed=args.dataset_seed,
            num_data=args.num_data,
        )
        # Only trim subsets actually present in the loaded package. The
        # injection helper rebuilds x_outlier / y_outlier at run time from
        # the inlier pool, so packages only need to store boundary + inliers.
        available_counts = {
            k: v for k, v in input_proto_counts.items() if k in all_prototype_data
        }
        selected_prototype_data, selected_prototype_indices = select_input_prototype_subsets(
            all_prototype_data,
            all_prototype_indices,
            classes=proto_classes,
            counts_by_subset=available_counts,
        )
        print(f"Loaded input prototypes from {proto_path}")
    elif input_proto_source["mode"] == "generate":
        n_prototype, n_boundary, n_inlier = _generation_pool_sizes(input_proto_counts)
        all_prototype_data, all_prototype_indices = generate_prototype_sets(
            train_x,
            train_y,
            proto_classes,
            n_prototype=n_prototype,
            n_boundary=n_boundary,
            n_inlier=n_inlier,
            return_indices=True,
        )
        available_counts = {
            k: v for k, v in input_proto_counts.items() if k in all_prototype_data
        }
        selected_prototype_data, selected_prototype_indices = select_input_prototype_subsets(
            all_prototype_data,
            all_prototype_indices,
            classes=proto_classes,
            counts_by_subset=available_counts,
        )
    input_proto_mode = args.input_prototypes_mode
    prototype_data = dict(selected_prototype_data)
    train_outlier_tracking: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    if prototype_data and input_proto_mode in ("train", "val"):
        train_x, train_y, outlier_augments, train_outlier_tracking = _build_prototype_injection_subsets(
            all_prototype_data=all_prototype_data,
            all_prototype_indices=all_prototype_indices,
            input_proto_counts=input_proto_counts,
            train_x=train_x,
            train_y=train_y,
            classes=proto_classes,
            x_outlier_mode=args.x_outlier_mode,
            random_direction_seed=args.random_direction_seed,
        )
        if input_proto_mode == "train" and outlier_augments:
            aug_X = [train_x] + [payload[0] for payload in outlier_augments]
            aug_Y = [train_y] + [payload[1] for payload in outlier_augments]
            train_x = torch.cat(aug_X, dim=0)
            train_y = torch.cat(aug_Y, dim=0)
        data = (train_x, train_y, test_x, test_y)
        tuple_data = data

    _validate_nonempty_prototype_subsets(
        prototype_data,
        context=f"prototype data ({input_proto_mode} mode)",
        require_nonempty=(input_proto_mode == "val"),
    )

    effective_train_len = int(train_x.shape[0])
    if effective_train_len == 0:
        raise ValueError("Training set is empty after holdout/augmentation.")
    if full_batch_requested:
        batch_size = effective_train_len
    elif batch_size > effective_train_len:
        print(f"[batch] reducing batch size from {batch_size} to {effective_train_len} to match training set size")
        batch_size = effective_train_len

    # --- Model Construction ---
    name = args.model
    params = model_presets[name]['params']
    params['input_dim'] = dataset_presets[dataset]['input_dim']
    params['output_dim'] = dataset_presets[dataset]['output_dim']
    net = prepare_net(
        model_type=model_presets[name]['type'], 
        params=params
        )

    # --- Model Initialization ---
    if not args.no_init:
        initialize_net(net, scale=args.init_scale, seed=args.init_seed)

    # ----- Checkpoint Continuation Handling -----
    wandb_checkpoint_loaded = False
    epoch_to_start = 0
    if args.cont_run_id is not None and args.cont_step is not None:
        # Continue from wandb checkpoint
        checkpoint_dir = get_checkpoint_dir_for_run(args.cont_run_id)
        if checkpoint_dir is None:
            raise FileNotFoundError(f"Cannot find checkpoint directory for run ID: {args.cont_run_id}")
        
        checkpoint_info = find_closest_checkpoint_wandb(args.cont_step, checkpoint_dir=checkpoint_dir)
        if checkpoint_info is None:
            raise FileNotFoundError(f"No suitable checkpoint found for step {args.cont_step} in run {args.cont_run_id}")
        
        
        if args.adam:
            # loaded_data = load_checkpoint_wandb(checkpoint_info, net, optimizer)
            raise ValueError("Cannot continue from wandb checkpoint with Adam optimizer (only SGD is supported). With Adam need to also keep the state, which is not implemented yet")
        
        loaded_data = load_checkpoint_wandb(checkpoint_info, net)
        step_to_start = loaded_data['step']
        epoch_to_start = loaded_data['epoch']
        wandb_checkpoint_loaded = True
        
        print(f"Loaded checkpoint from step {loaded_data['step']} (epoch {loaded_data['epoch']}) from run {args.cont_run_id}")
        print(f"Closest checkpoint to requested step {args.cont_step}: actual step {loaded_data['step']}")
        
        # Handle quadratic approximation continuation
        if args.quad_switch_step is not None and loaded_data['step'] >= args.quad_switch_step:
            print(f"Warning: Continuing from step {loaded_data['step']} which is at or after quad_switch_step {args.quad_switch_step}")
            print("Quadratic approximation will be initialized as active from the start.")

    # ----- wandb Initialization -----
    wandb_run = None
    wandb_run_id = None
    if wandb_enabled:
        wandb_run = init_wandb(args, step_to_start)
        wandb_run_id = wandb_run.id
    else:
        wandb_run_id = generate_run_id()

    # ----- Reference Parameter Handling -----
    # Load the reference parameters to calculate the distance from (if provided)
    param_reference = None
    if args.param_distance:
        if args.param_file is None:
            # Create a zero vector as a reference if no param_file is provided
            print("No parameter file provided. Creating a zero vector as reference.")
            param_reference = []
            for param in net.parameters():
                param_reference.append(torch.zeros_like(param).flatten())
            param_reference = torch.cat(param_reference)
    if args.param_file is not None:
        param_reference = T.load(args.param_file, map_location=device)
        # param_reference = param_reference['model_state_dict']
        # param_reference = {k: v.to(device) for k, v in param_reference.items()} 

    # ----- Optimizer Preparation -----
    optimizer = prepare_optimizer(net, args.lr, args.momentum, args.adam, args.weight_decay)

    # ----- Checkpoint Cadence Determination -----
    if args.checkpoint_every is not None:
        checkpoint_every_n_steps = args.checkpoint_every
    else:
        checkpoint_every_n_steps = max(args.steps // 200, 1) if args.steps else None
    
    subset_tracking_cfgs = []
    if train_outlier_tracking:
        tracking_prefix = "injected" if input_proto_mode == "train" else "heldout"
        prefixed_tracking = {
            f"{tracking_prefix}_{name}": tensors
            for name, tensors in train_outlier_tracking.items()
        }
        subset_tracking_cfgs.extend(prepare_prototype_subset_configs(prefixed_tracking, base_batch_size=batch_size))


    # ----- Training Invocation -----
    train(
        net=net,
        optimizer=optimizer,
        data=tuple_data,
        max_epochs=args.epochs,
        max_steps=args.steps,
        batch_size=batch_size,
        save_to=run_folder,
        device=device,
        loss_fn=loss_fn,
        verbose=True,
        stop_loss = args.stop_loss,
        epoch_to_start=epoch_to_start,
        step_to_start=step_to_start,
        gd_noise=args.gd_noise,
        noise_magnitude=args.noise_mag,
        results_rarely=args.results_rarely,
        measurements=measurements,
        param_reference=param_reference,
        cache_eigenvectors = not args.disable_cache_eigenvectors,
        sde_enabled=args.sde,
        sde_h=args.sde_h,
        sde_eta=args.sde_eta,
        sde_seed=args.sde_seed,
        use_power_iteration=args.use_power_iteration,
        num_eigenvalues=args.num_eigenvalues,
        checkpoint_every_n_steps=checkpoint_every_n_steps,
        quad_switch_step=args.quad_switch_step,
        quad_switch_lr=args.quad_switch_lr,
        use_gauss_newton=args.use_gauss_newton,
        precise_plots=args.precise_plots,
        rare_measure=args.rare_measure,
        proj_switch_step=args.proj_switch_step,
        proj_top_l=args.proj_top_l,
        proj_to_residual=args.proj_to_residual,
        wandb_run=wandb_run,
        wandb_enabled=wandb_enabled,
        wandb_run_id=wandb_run_id,
        #NEW
        subset_tracking_cfgs=subset_tracking_cfgs,
        prototype_data=prototype_data,
        log_every_step=args.log_every_step,
        dense_window=tuple(getattr(args, "dense_window", ())) if getattr(args, "dense_window", None) else None,
        lmax_decay=getattr(args, "lmax_decay", False),
        lmax_decay_target_lr=getattr(args, "lmax_decay_target_lr", None),
        lmax_decay_steps=getattr(args, "lmax_decay_steps", 10000),
        lmax_decay_initial_lr=args.lr,
        lmax_drop=args.lmax_drop,
        lmax_drop_mult=args.lmax_drop_mult,
        lmax_drop_target_lr=args.lmax_drop_target_lr,
        lr_drop_at_step=args.lr_drop_at_step,
        lr_drop_to=args.lr_drop_to,
    )
