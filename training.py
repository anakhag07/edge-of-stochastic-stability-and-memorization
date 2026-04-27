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
from typing import Dict, List, Tuple

import torch.nn.functional as F
import time
import torch.optim as optim

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.data import (
    prepare_dataset,
    get_dataset_presets,
    generate_prototype_sets,
    generate_feature_space_prototype_sets,
    EXTRAPOLATION_FACTOR,
    trim_prototype_sets,
)
from utils.nets import SquaredLoss, MLP, CNN, prepare_net, initialize_net, prepare_optimizer, get_model_presets
from utils.nets import ResNet
from utils.storage import initialize_folders
from utils.input_prototypes import (
    resolve_input_prototype_path,
    load_input_prototype_package,
)
from utils.wandb_utils import (
    init_wandb,
    log_metrics,
    log_knn_outlier_results,
    save_checkpoint_wandb,
    find_closest_checkpoint_wandb,
    load_checkpoint_wandb,
    get_checkpoint_dir_for_run,
    is_wandb_available,
    generate_run_id,

)

from utils.noise import gd_with_noise, GradStorage, sde_integration
from utils.measure import *
from utils.measure import compute_train_test_gap_from_tensors
from utils.frequency import frequency_calculator, MeasurementContext
from utils.quadratic import QuadraticApproximation, flatten_params, set_model_params, unflatten_params

from torch.autograd import grad
import json

if 'DATASETS' not in os.environ:
    raise ValueError("Please set the environment variable 'DATASETS'. Use 'export DATASETS=/path/to/datasets'")
if 'RESULTS' not in os.environ:
    raise ValueError("Please set the environment variable 'RESULTS'. Use 'export RESULTS=/path/to/results'")

DATASET_FOLDER = Path(os.environ.get('DATASETS'))
# export RESULTS=/scratch/gpfs/andreyev/eoss/results
RES_FOLDER = Path(os.environ.get('RESULTS'))

KNN_TRACKING_METRICS = ["full_loss", "accuracy", "lambda_max", "grad_hessian_grad", "batch_sharpness","grad_vmax_cos2", "grad_norm"]
TRAIN_OUTLIER_TRACKING_METRICS = [
    "per_example_loss_mean",
    "per_example_loss_std",
    "lambda_max",
    "grad_hessian_grad",
    "batch_sharpness",
]


def _load_reference_knn_indices(dataset_name: str, model_name: str, run_name: str) -> Dict[int, List[int]]:
    plaintext_root = RES_FOLDER / 'plaintext' / f"{dataset_name}_{model_name}"
    ref_run_dir = plaintext_root / run_name
    indices_path = ref_run_dir / 'knn_outlier_indices.json'
    if not indices_path.exists():
        raise FileNotFoundError(
            f"Cannot find knn_outlier_indices.json at {indices_path}. "
            "Ensure the reference run name is correct."
        )

    with open(indices_path, 'r') as f:
        indices_payload = json.load(f)

    per_class_indices = indices_payload.get('per_class_indices', {})
    if not per_class_indices:
        raise ValueError(f"No per-class indices found in {indices_path}")

    cleaned = {}
    for class_key, idx_list in per_class_indices.items():
        class_id = int(class_key)
        cleaned[class_id] = [int(idx) for idx in idx_list]
    return cleaned


def _build_tracked_subsets(per_class_indices: Dict[int, List[int]], track_top: int):
    tracked_subsets = []
    trimmed_by_class: Dict[int, List[int]] = {}
    for class_id, idx_list in per_class_indices.items():
        trimmed = [int(idx) for idx in idx_list[:track_top]]
        if not trimmed:
            continue
        trimmed_by_class[class_id] = trimmed
        tracked_subsets.append({
            "name": f"class_{class_id}",
            "class_id": class_id,
            "indices": trimmed,
        })
    return tracked_subsets, trimmed_by_class


def _sample_inlier_subsets(
    labels: torch.Tensor,
    excluded_by_class: Dict[int, List[int]],
    seed: int,
) -> List[dict]:
    if labels is None or not excluded_by_class:
        return []

    label_tensor = labels.detach().cpu()
    if label_tensor.ndim > 1:
        label_tensor = label_tensor.argmax(dim=1)
    label_tensor = label_tensor.to(dtype=torch.long)
    rng = random.Random(seed)

    subsets = []
    for class_id, excluded in excluded_by_class.items():
        class_mask = (label_tensor == class_id)
        class_indices = class_mask.nonzero(as_tuple=False).view(-1).tolist()
        if not class_indices:
            continue
        excluded_set = set(excluded)
        candidates = [idx for idx in class_indices if idx not in excluded_set]
        if not candidates:
            continue
        desired = min(len(excluded), len(candidates))
        if desired <= 0:
            continue
        if len(candidates) > desired:
            sampled = rng.sample(candidates, desired)
        else:
            sampled = candidates
        subsets.append({
            "name": f"class_{class_id}",
            "class_id": class_id,
            "indices": sampled,
        })
    return subsets


def _parse_input_prototype_source(raw: str):
    if raw is None:
        return {"mode": None, "value": None}

    value = raw.strip()
    lowered = value.lower()
    if lowered in ("none", "off", "false"):
        return {"mode": "none", "value": None}
    if lowered in ("generate", "gen"):
        return {"mode": "generate", "value": None}
    if value.startswith("from:"):
        return {"mode": "from", "value": value[5:]}
    if value.startswith("run:"):
        return {"mode": "from", "value": value[4:]}
    return {"mode": "from", "value": value}


def _build_input_prototype_counts(args) -> Dict[str, int]:
    counts = {}
    if args.input_prototypes_boundary_count is not None:
        counts["boundary"] = args.input_prototypes_boundary_count
    if args.input_prototypes_inliers_count is not None:
        counts["inliers"] = args.input_prototypes_inliers_count
    return counts


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


FEATURE_PROTOTYPE_TRACKING_MAP = {
    'feature_boundary': 'feature_space_prototypes/knn_outlier',
    'feature_inliers': 'feature_space_prototypes/knn_inlier',
    'feature_x_outlier': 'feature_space_prototypes/synthetic_x_outlier',
    'feature_y_outlier': 'feature_space_prototypes/synthetic_y_outlier',
}

INPUT_PROTOTYPE_TRACKING_MAP = {
    'boundary': 'input_space_prototypes/boundary_points',
    'inliers': 'input_space_prototypes/inlier_points',
    'x_outlier': 'input_space_prototypes/synthetic_x_outlier',
    'y_outlier': 'input_space_prototypes/synthetic_y_outlier',
    'injected_x_outlier': 'input_space_prototypes/injected_x_outlier',
    'injected_y_outlier': 'input_space_prototypes/injected_y_outlier',
    'injected_inliers':   'input_space_prototypes/injected_inliers',
    'injected_boundary':  'input_space_prototypes/injected_boundary',
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


def prepare_feature_prototype_subset_configs(prototype_data: dict) -> List[dict]:
    if not prototype_data:
        return []

    configs = []
    for proto_key, log_prefix in FEATURE_PROTOTYPE_TRACKING_MAP.items():
        if proto_key not in prototype_data:
            continue
        X_p, Y_p = prototype_data[proto_key]
        configs.append({
            "enabled": True,
            "subsets": [{
                "name": None,
                "class_id": None,
                "X_tensor": X_p.detach().cpu(),
                "Y_tensor": Y_p.detach().cpu(),
            }],
            "metrics": KNN_TRACKING_METRICS,
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


def _feature_prototype_dir(dataset_name: str, model_name: str, run_name: str) -> Path:
    plaintext_root = RES_FOLDER / 'plaintext' / f"{dataset_name}_{model_name}"
    return plaintext_root / run_name / 'feature_prototypes'


def _metadata_to_cpu(payload):
    if torch.is_tensor(payload):
        return payload.detach().cpu()
    if isinstance(payload, dict):
        return {k: _metadata_to_cpu(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_metadata_to_cpu(v) for v in payload]
    return payload


def _save_feature_prototype_package(
    save_dir: Path,
    prototypes: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    metadata: Dict[str, dict],
):
    target_dir = save_dir / 'feature_prototypes'
    target_dir.mkdir(parents=True, exist_ok=True)

    package = {
        "prototypes": {
            name: {
                "inputs": X.detach().cpu(),
                "labels": Y.detach().cpu(),
            }
            for name, (X, Y) in prototypes.items()
        },
        "metadata": _metadata_to_cpu(metadata),
    }

    torch.save(package, target_dir / 'prototypes.pt')

    summary = {
        "classes": metadata.get("classes"),
        "counts": {name: int(payload["inputs"].shape[0]) for name, payload in package["prototypes"].items()},
    }
    indices = metadata.get("indices")
    if indices:
        summary["indices"] = {name: [int(idx) for idx in tensor.tolist()] for name, tensor in indices.items()}

    with open(target_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


def _load_reference_feature_prototypes(dataset_name: str, model_name: str, run_name: str):
    proto_dir = _feature_prototype_dir(dataset_name, model_name, run_name)
    proto_path = proto_dir / 'prototypes.pt'
    if not proto_path.exists():
        raise FileNotFoundError(
            f"Cannot find feature prototype package at {proto_path}. "
            "Ensure the reference run computed feature-space prototypes."
        )

    package = torch.load(proto_path, map_location='cpu')
    proto_payload = package.get("prototypes", {})
    if not proto_payload:
        raise ValueError(f"No prototype tensors stored in {proto_path}")

    prototypes = {}
    for name, payload in proto_payload.items():
        inputs = payload.get("inputs")
        labels = payload.get("labels")
        if inputs is None or labels is None:
            continue
        prototypes[name] = (inputs, labels)

    if not prototypes:
        raise ValueError(f"Feature prototype file {proto_path} did not contain usable tensors.")

    metadata = package.get("metadata", {})
    return prototypes, metadata


def prepare_knn_subset_tracking_configs(args, dataset_name: str, model_name: str, data) -> List[dict]:
    if not args.track_knn_outliers_from:
        return []

    per_class_indices = _load_reference_knn_indices(dataset_name, model_name, args.track_knn_outliers_from)
    allowed_classes = set(args.classes or [])
    if allowed_classes:
        per_class_indices = {cid: idxs for cid, idxs in per_class_indices.items() if cid in allowed_classes}
        if not per_class_indices:
            raise ValueError(
                f"No reference KNN indices for requested classes {sorted(allowed_classes)} "
                f"in run {args.track_knn_outliers_from}"
            )

    track_top = max(1, args.track_knn_topk)

    tracked_subsets, trimmed_by_class = _build_tracked_subsets(per_class_indices, track_top)
    if not tracked_subsets:
        raise ValueError(
            f"No indices remained after applying --track-knn-topk={track_top} "
            f"for run {args.track_knn_outliers_from}"
        )

    configs = [{
        "enabled": True,
        "subsets": tracked_subsets,
        "metrics": KNN_TRACKING_METRICS,
        "log_prefix": f"knn_outlier/{args.track_knn_outliers_from}",
    }]

    _, Y_train, _, _ = data
    inlier_seed = (args.dataset_seed or 0) + 1337
    inlier_subsets = _sample_inlier_subsets(Y_train, trimmed_by_class, seed=inlier_seed)
    if inlier_subsets:
        configs.append({
            "enabled": True,
            "subsets": inlier_subsets,
            "metrics": KNN_TRACKING_METRICS,
            "log_prefix": f"knn_inlier/{args.track_knn_outliers_from}",
        })
    else:
        print("Warning: Unable to sample knn_inlier subsets; insufficient inlier candidates.")

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



# -------------------------------------
# NEW: Per-sample histograms setup
# -------------------------------------
class PerSampleHistograms:
    def __init__(self, min_log10, max_log10, bins, metrics):
        self.min_log10 = min_log10
        self.max_log10 = max_log10
        self.bins = bins
        self.metrics = metrics
        self.histograms = {metric: np.zeros((bins,)) for metric in metrics}
        self.counts = {metric: 0 for metric in metrics}
        self.quantiles = {metric: np.zeros((bins,)) for metric in metrics}

# @torch.no_grad()

def _per_sample_stats(net, loss_fn, X, Y, loss_type='ce', batch_size=1024, device='cuda'):
    """
    Return dict of numpy arrays: loss, resid_norm, kappa, grad_norm for dataset (X, Y).
    loss_type: 'ce' (cross-entropy) or 'mse' (SquaredLoss: 0.5 * ||y - yhat||^2).
    """
    was_training = net.training
    net.eval()

    out_loss, out_resid, out_kappa, out_grad_norm = [], [], [], []

    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device)
        yb = Y[i:i + batch_size].to(device)

        grad_norms_batch = []

        for j in range(xb.shape[0]):
            xb_j = xb[j:j + 1]  # [1, ...]
            yb_j = yb[j:j + 1]  # [1] or [1, C]

            net.zero_grad()

            z_j = net(xb_j)

            if loss_type == 'ce':
                # Cross-entropy for a single sample
                loss_j = F.cross_entropy(z_j, yb_j.long(), reduction='sum')
            else:
                # MSE: 0.5 * ||y - yhat||^2
                zf = z_j
                yf = yb_j

                if zf.ndim > 1 and zf.size(-1) == 1:
                    zf = zf.squeeze(-1)
                if yf.ndim > 1 and yf.size(-1) == 1:
                    yf = yf.squeeze(-1)

                loss_j = 0.5 * (zf - yf).pow(2).sum()

            loss_j.backward()

            grads = []
            for param in net.parameters():
                if param.grad is not None:
                    grads.append(param.grad.view(-1))

            if grads:
                g = torch.cat(grads)
                grad_norms_batch.append(torch.linalg.norm(g).item())
            else:
                grad_norms_batch.append(0.0)

        out_grad_norm.extend(grad_norms_batch)

        # ----------------------------------------------------
        # Loss / resid_norm / kappa
        # ----------------------------------------------------
        with torch.no_grad():
            z = net(xb)

            if loss_type == 'ce':
                # Per-sample CE loss
                loss = F.cross_entropy(z, yb.long(), reduction='none')

                # Residual wrt logits: p - y_onehot
                p = torch.softmax(z, dim=1)
                y1 = F.one_hot(yb.long(), num_classes=z.size(1)).float()
                resid = p - y1
                resid_norm = resid.norm(dim=1)

                # Curvature proxy: Frobenius norm of softmax Hessian
                C = p.size(1)
                I = torch.eye(C, device=p.device).unsqueeze(0)  # [1, C, C]
                Hout = I * p.unsqueeze(2) - p.unsqueeze(2) * p.unsqueeze(1)
                kappa = torch.linalg.norm(Hout, dim=(1, 2))

            else:
                # MSE: SquaredLoss 0.5 * ||y - yhat||^2
                zf = z
                yf = yb

                if zf.ndim > 1 and zf.size(-1) == 1:
                    zf = zf.squeeze(-1)
                if yf.ndim > 1 and yf.size(-1) == 1:
                    yf = yf.squeeze(-1)
                if yf.ndim == 1 and zf.ndim == 2:
                    yf = F.one_hot(yf.long(), num_classes=zf.size(-1)).float()

                diff = zf - yf                       # [B] or [B, D]
                loss = 0.5 * (diff ** 2)
                if loss.ndim > 1:
                    loss = loss.sum(dim=1)

                if diff.ndim > 1:
                    resid_norm = diff.norm(dim=1)
                else:
                    resid_norm = diff.abs()

                # For plain MSE, output-space curvature is constant
                kappa = torch.ones_like(loss)

        out_loss.append(loss.cpu())
        out_resid.append(resid_norm.cpu())
        out_kappa.append(kappa.cpu())

    if was_training:
        net.train()

    return {
        'loss': torch.cat(out_loss).numpy(),
        'resid': torch.cat(out_resid).numpy(),
        'kappa': torch.cat(out_kappa).numpy(),
        'grad_norm': np.asarray(out_grad_norm),
    }

def _hist_log10(values, bin_edges):
    v = np.asarray(values)
    v = np.clip(v, a_min=np.finfo(float).tiny, a_max=None)  # avoid log of 0
    lv = np.log10(v)
    counts, _ = np.histogram(lv, bins=bin_edges)
    return counts

def _quantiles(values, qs=(0.1,0.5,0.9,0.99)):
    v = np.asarray(values)
    return np.quantile(v, qs)

def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def _format_duration(seconds):
    seconds = max(int(seconds), 0)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"

def _render_frame(bin_edges, counts_train, counts_test, title, out_png_path):
    plt.figure(figsize=(6,4))
    centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    plt.step(centers, counts_train, where='mid', label='train', alpha=0.8)
    plt.step(centers, counts_test, where='mid', label='test', alpha=0.6)
    plt.yscale('log') # log applied
    plt.xlabel('log10(value)')
    plt.ylabel('log10(count)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=140)
    plt.close()



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
        memorization_outlier_frac: float,
        # NEW:
        prototype_data,
        full_inputs_test=None,
        per_sample_cfg=None,
        subset_tracking_cfgs=None,
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
        self.memorization_outlier_frac = memorization_outlier_frac
        self._precond_pi_vec = precond_pi_vec

        # NEW: Per-sample config
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
                        "log_prefix": cfg.get('log_prefix', "knn_outlier"),
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
            'memorization_hessian_outliers': None,
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
            elif self.batch_size < self.X.shape[0]:
                eos_threshold = 2.0 / optimizer.param_groups[0]['lr']
                sharpness_metric = 'batch_sharpness'
            else:
                eos_threshold = 2.0 / optimizer.param_groups[0]['lr']
                sharpness_metric = 'lmax'
            
            if not getattr(self, '_t_star_logged', False) and metrics.get(sharpness_metric, float('nan')) >= eos_threshold:
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

        # ----- Memorization via Hessian outliers -----
        if 'memorization_hessian_outliers' in self.measurements:
            if frequency_calculator.should_measure('memorization_hessian_outliers', ctx):
                optimizer.zero_grad()
                mem_stats = compute_outlier_vs_bulk_stats_hessian(
                    net=self.net,
                    X_train=self.X,
                    Y_train=self.Y,
                    loss_fn=self.loss_fn,
                    optimizer=optimizer,
                    frac=self.memorization_outlier_frac,
                )
                if mem_stats:
                    metrics.update({f"memorization_hessian_outliers/{k}": v for k, v in mem_stats.items()})
        
        # ----- NEW: Train/Test Gap -----
        if 'train_test_gap' in self.measurements and self.X_test is not None:
            if frequency_calculator.should_measure('train_test_gap',ctx):
                vals = compute_train_test_gap_from_tensors(
                    self.net, self.X, self.Y, self.X_test, self.Y_test
                )
                metrics['train_acc'] = vals['train_acc']
                metrics['test_acc'] = vals['test_acc']
                metrics['train_test_gap'] = vals['gap']

        if self.subset_trackers and frequency_calculator.should_measure('knn_outlier_metrics', ctx):
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

        # ----- NEW: Per-sample histograms -----
        if self.per_sample_cfg and self.per_sample_cfg['enabled']:
            every = self.per_sample_cfg['every']
            if step_number % every == 0:
                loss_type = 'ce' if isinstance(self.loss_fn, nn.CrossEntropyLoss) else 'mse'
                # train
                stats_tr = _per_sample_stats(self.net, self.loss_fn, self.X, self.Y, loss_type=loss_type, batch_size=self.batch_size, device=self.device)
                # test (if present)
                if (self.X_test is not None) and (len(self.X_test) > 0):
                    stats_te = _per_sample_stats(self.net, self.loss_fn, self.X_test, self.Y_test, loss_type=loss_type, batch_size=self.batch_size, device=self.device)
                else:
                    stats_te = {k: np.zeros_like(v) for k, v in stats_tr.items()}

                # collect metrics
                for metric in self.per_sample_histograms.metrics:
                    # Quantiles (log to wandb)
                    quantiles_tr = _quantiles(stats_tr[metric])
                    quantiles_te = _quantiles(stats_te[metric])
                    
                    for q, name in zip((0.1, 0.5, 0.9, 0.99), ('q10','q50','q90','q99')):
                        metrics[f'ps_quantiles/{metric}_train_{name}'] = np.quantile(stats_tr[metric], q)
                        metrics[f'ps_quantiles/{metric}_test_{name}'] = np.quantile(stats_te[metric], q)

                    # Histograms (save to file)
                    counts_tr = _hist_log10(stats_tr[metric], self.bin_edges)
                    counts_te = _hist_log10(stats_te[metric], self.bin_edges)

                    # Store for saving to file later
                    np.savez(
                        self.ps_dir / f'step_{step_number:05d}_{metric}.npz',
                        bin_edges=self.bin_edges,
                        counts_train=counts_tr,
                        counts_test=counts_te,
                        quantiles_train=quantiles_tr,
                        quantiles_test=quantiles_te
                    )

                    # Render frame
                    if not self.per_sample_cfg.get('no_frames', False):
                        title = f'Step {step_number}: log10({metric})'
                        out_path = self.frames_dir / f'{step_number:05d}_{metric}.png'
                        _render_frame(self.bin_edges, counts_tr, counts_te, title, out_path)



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


        # ----- NEW: Prototype per-sample stats over time -----
        if self.prototype_data is not None and self.per_sample_cfg and self.per_sample_cfg['enabled']:
            proto_every = self.per_sample_cfg["every"]
            if step_number % proto_every == 0:
                loss_type = 'ce' if isinstance(self.loss_fn, nn.CrossEntropyLoss) else 'mse'

                proto_dir = self.ps_dir / "prototypes"
                _ensure_dir(proto_dir)

                for name, (X_p, Y_p) in self.prototype_data.items():
                    X_p = X_p.to(self.device)
                    Y_p = Y_p.to(self.device)

                    # Per-sample stats
                    stats = _per_sample_stats(
                        self.net,
                        self.loss_fn,
                        X_p,
                        Y_p,
                        loss_type=loss_type,
                        batch_size=len(X_p),
                        device=self.device,
                    )
                    
                    # NEW: Add batch sharpness for this prototype set
                    # Compute full loss on this prototype set
                    self.net.zero_grad()
                    preds = self.net(X_p).squeeze(dim=-1)
                    proto_loss = self.loss_fn(preds, Y_p)
                    
                    # Compute batch sharpness (gHg/g²)
                    batch_sharpness = compute_grad_H_grad(proto_loss, self.net).item()
                    
                    # Add to stats
                    stats['batch_sharpness'] = batch_sharpness
                    stats['mean_loss'] = np.mean(stats['loss'])
                    
                    out_path = proto_dir / f"step_{step_number:05d}_{name}.npz"
                    np.savez(out_path, **stats)



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
            memorization_outlier_frac: float = 0.05,  # Fraction of samples marked as outliers for memorization metric
            # Gradient projection configuration
            proj_switch_step: int = None,  # Step to start projecting minibatch gradients
            proj_top_l: int = None,        # Number of top eigendirections to use for projection
            proj_to_residual: bool = False, # If True, project to orthogonal complement of top-l eigenspace
            wandb_run=None,
            wandb_enabled: bool = False,
            wandb_run_id: str = None,
            per_sample_cfg=None, #NEW 
            knn_outlier_cfg=None,
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

    # ----- Lmax Decay Setup -----
    decay_active = False
    decay_start_step = None
    decay_start_lr = None
    if lmax_decay_initial_lr is None:
        lmax_decay_initial_lr = optimizer.param_groups[0]['lr']
    lmax_decay_threshold = 2.0 / lmax_decay_initial_lr

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
        memorization_outlier_frac=memorization_outlier_frac,
        per_sample_cfg=per_sample_cfg,
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
                # Adam: trigger off AEoS ratio ~ 1
                if args.adam and ('adam_edge_ratio' in metrics):
                    r = metrics.get('adam_edge_ratio', float('nan'))
                    if (not decay_active) and math.isfinite(r) and (r >= 1.0):
                        decay_active = True
                        old_lr = optimizer.param_groups[0]['lr']
                        new_lr = old_lr * float(lmax_drop_mult)
                        if lmax_drop_target_lr is not None:
                            new_lr = max(new_lr, float(lmax_drop_target_lr))
                        for pg in optimizer.param_groups:
                            pg['lr'] = new_lr
                        print(f"[ADAM LR DROP] step={step_number} edge_ratio={r:.3f} lr: {old_lr:g} -> {new_lr:g}")
                else:
                    lmax_value = metrics.get('lmax', float('nan'))
                    if (not decay_active) and math.isfinite(lmax_value) and (lmax_value >= lmax_decay_threshold):
                        decay_active = True  # reuse as "already dropped"
                        old_lr = optimizer.param_groups[0]['lr']
                        new_lr = old_lr * float(lmax_drop_mult)
                        if lmax_drop_target_lr is not None:
                            new_lr = max(new_lr, float(lmax_drop_target_lr))
                        for pg in optimizer.param_groups:
                            pg['lr'] = new_lr
                        print(f"[LR DROP] step={step_number} lmax={lmax_value:.4f} thresh={lmax_decay_threshold:.4f} lr: {old_lr:g} -> {new_lr:g}")

                
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

    # ----- NEW: per-sample stats on prototype sets -----
    if prototype_data is not None:
        print("Computing per-sample metrics for prototype sets...")
        loss_type = 'ce' if isinstance(loss_fn, nn.CrossEntropyLoss) else 'mse'

        proto_dir = save_to / "prototype_final"
        _ensure_dir(proto_dir)

        for name, (X_p, Y_p) in prototype_data.items():
            X_p = X_p.to(device)
            Y_p = Y_p.to(device)

            stats = _per_sample_stats(
                net,
                loss_fn,
                X_p,
                Y_p,
                loss_type=loss_type,
                batch_size=batch_size,
                device=device,
            )
            out_path = proto_dir / f"final_{name}.npz"
            np.savez(out_path, **stats)
            print(f"  saved {name} -> {out_path}")

    results_file.close()

    measurement_runner.close()

    # ----- Final Reporting -----
    if wandb_run is not None and not wandb_finished:
        _log_train_time()
        wandb_run.finish()
        wandb_finished = True

    if knn_outlier_cfg and knn_outlier_cfg.get('enabled', False):
        print("Computing k-NN neighbor-mix outliers...")
        feature_batch = knn_outlier_cfg.get('feature_batch_size', 512)
        k_neighbors = knn_outlier_cfg.get('k_neighbors', 32)
        top_k_per_class = knn_outlier_cfg.get('top_k_per_class', 5)
        balance_target = knn_outlier_cfg.get('balance_target', 0.5)
        chunk_size = knn_outlier_cfg.get('chunk_size', 1024)
        normalize = knn_outlier_cfg.get('normalize', True)
        return_indices = knn_outlier_cfg.get('return_neighbor_indices', True)

        features = extract_feature_matrix(
            net,
            X,
            batch_size=feature_batch,
            flatten_outputs=True,
        )
        label_tensor = Y.detach().cpu()

        outlier_summary = identify_knn_outliers_by_neighbor_mix(
            features,
            label_tensor,
            k_neighbors=k_neighbors,
            top_k_per_class=top_k_per_class,
            balance_target=balance_target,
            chunk_size=chunk_size,
            normalize=normalize,
            return_neighbor_indices=return_indices,
        )

        per_class_indices = {}
        per_class_stats = {}
        flattened_rows = []
        global_balance_dev = []
        global_entropy = []
        global_gap = []

        for class_id, entries in outlier_summary.get("outliers", {}).items():
            if not entries:
                continue
            per_class_indices[int(class_id)] = [int(e["dataset_index"]) for e in entries]

            ratios = np.array([e["same_class_ratio"] for e in entries], dtype=np.float32)
            devs = np.array([e["balance_deviation"] for e in entries], dtype=np.float32)
            entropies = np.array([e["neighbor_entropy"] for e in entries], dtype=np.float32)
            gaps = np.array([e["top_two_gap"] for e in entries], dtype=np.float32)

            per_class_stats[int(class_id)] = {
                "count": len(entries),
                "mean_same_class_ratio": float(ratios.mean()),
                "mean_balance_deviation": float(devs.mean()),
                "mean_neighbor_entropy": float(entropies.mean()),
                "mean_top_two_gap": float(gaps.mean()),
            }

            global_balance_dev.extend(devs.tolist())
            global_entropy.extend(entropies.tolist())
            global_gap.extend(gaps.tolist())

            for entry in entries:
                flattened_rows.append({
                    "dataset_index": int(entry["dataset_index"]),
                    "class_id": int(class_id),
                    "same_class_ratio": float(entry["same_class_ratio"]),
                    "balance_deviation": float(entry["balance_deviation"]),
                    "neighbor_entropy": float(entry["neighbor_entropy"]),
                    "top_two_gap": float(entry["top_two_gap"]),
                    "avg_neighbor_distance": float(entry["avg_neighbor_distance"]),
                })

        outlier_path = save_to / 'knn_outliers.json'
        with open(outlier_path, 'w') as f:
            json.dump(outlier_summary, f, indent=2)

        indices_payload = {
            "k_neighbors": k_neighbors,
            "top_k_per_class": top_k_per_class,
            "class_ids": outlier_summary.get("class_ids", []),
            "per_class_indices": {str(k): v for k, v in per_class_indices.items()},
            "flat_indices": sorted({idx for v in per_class_indices.values() for idx in v}),
        }
        with open(save_to / 'knn_outlier_indices.json', 'w') as f:
            json.dump(indices_payload, f, indent=2)

        if wandb_enabled and wandb_run is not None:
            total_flagged = len(flattened_rows)
            wandb_metrics = {
                "knn_outliers/total_flagged": total_flagged,
                "knn_outliers/mean_balance_deviation": float(np.mean(global_balance_dev)) if global_balance_dev else float('nan'),
                "knn_outliers/mean_neighbor_entropy": float(np.mean(global_entropy)) if global_entropy else float('nan'),
                "knn_outliers/mean_top_two_gap": float(np.mean(global_gap)) if global_gap else float('nan'),
            }
            for class_id, stats in per_class_stats.items():
                prefix = f"knn_outliers/class_{class_id}"
                for name, value in stats.items():
                    wandb_metrics[f"{prefix}/{name}"] = value

            log_knn_outlier_results(wandb_metrics, flattened_rows)

        print(f"k-NN outlier summary saved to {outlier_path}")


    # ----- Optional Final Measurements -----
    if 'final' in measurements:
        final_file = save_to / 'final.json'
        final_file = open(final_file, 'w') 

        # do the final measurements here - depending on what is needed
    




if __name__ == '__main__':
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
    parser = argparse.ArgumentParser(description='Training script')
    # --- Seed ---
    parser.add_argument("--seed", type=int, default=88881)

    # --- Training Parameters ---
    parser.add_argument('--batch', type=int, default=64, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--steps', type=int, default=10000, help='Number of steps to train. Either epochs or steps should be provided')
    parser.add_argument('--cpu', action='store_true', help='Force training to run on CPU even if CUDA is available')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')

    # --- Lmax LR Decay Configuration ---
    parser.add_argument('--lmax-decay', action='store_true',
                        help='Enable linear lr decay once lambda_max >= 2/initial_lr')
    parser.add_argument('--lmax-decay-target-lr', type=float, default=0.0001,
                        help='Target lr for linear decay after lmax trigger')
    parser.add_argument('--lmax-decay-steps', type=int, default=10000,
                        help='Number of steps to linearly decay to target lr')

    # --- LR Drop Configuration ---
    parser.add_argument('--lr-drop-at-step', type=int, default=None,
                        help='Exact step at which to drop the learning rate (for fork runs). '
                             'Use with --lr-drop-to. The run starts at --lr and switches at this step.')
    parser.add_argument('--lr-drop-to', type=float, default=None,
                        help='Target LR after --lr-drop-at-step fires.')
    parser.add_argument('--lmax-drop', action='store_true',
                        help='One-time LR drop once lambda_max exceeds threshold (2/initial_lr).')
    parser.add_argument('--lmax-drop-mult', type=float, default=0.5,
                        help='Multiply LR by this factor on trigger. (0.5 = 50%% drop, 0.8 = 20%% drop)')
    parser.add_argument('--lmax-drop-target-lr', type=float, default=None,
                        help='Optional floor: LR after drop is max(LR*mult, target).')

    # --- Loss Configuration ---
    parser.add_argument('--stop-loss', '--stop_loss', type=float, default=None, help='Stop training if loss goes below this value')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'ce'], help='Loss function to use (mse or ce)')

    # --- Dataset Configuration ---
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use for training')
    parser.add_argument('--classes', type=int, nargs=2, default=[1, 9], help='Two class labels to use for training. Default is [1, 9], as being probably the most difficult classes to separate')
    parser.add_argument('--num-data', '--num_data', type=int, default=1024, help='Number of datapoints to train on')

    # --- Model Configuration ---
    parser.add_argument('--model', type=str, default='mlp', help='Network architecture to use for training')
    parser.add_argument('--init-scale', '--init_scale', type=float, default=0.2, help='Initialization scale for network weights')
    parser.add_argument('--no-init', '--no_init', action='store_true', help='If set, do not initialize network weights')

    # --- wandb Continuation Options ---
    parser.add_argument('--cont-run-id', '--cont_run_id', type=str, default=None, help='Wandb run ID to continue training from')
    parser.add_argument('--cont-step', '--cont_step', type=int, default=None, help='Step to continue training from (uses closest available checkpoint)')
    parser.add_argument('--checkpoint-every', '--checkpoint_every', type=int, default=None, help='Save checkpoint every N steps (default: auto-calculated based on total steps)')

    # --- Optimizer Variants ---
    parser.add_argument('--momentum', type=float, default=None, help='Momentum for SGD optimizer')
    parser.add_argument('--adam', action='store_true', help='If set, use Adam optimizer instead of SGD')
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--precond-lmax', action='store_true',
        help='Log Adam-preconditioned top Hessian eigenvalue (AEoS sharpness).')

    # --- Measurement Flags (Primary) ---
    # parser.add_argument('--fullbs', action='store_true', help='If set, compute the lambda_max, aka FullBS')
    parser.add_argument('--lambdamax', '--lmax', action='store_true', help='If set, compute the lambda_max, aka FullBS')
    parser.add_argument('--batch-sharpness', '--batch-sharpness-step', '--bs', action='store_true', dest='batch_sharpness',
                        help='If set, compute the batch sharpness: E[gHg/g²] with the expectation taken across mini-batches. Use --batch-sharpness-step for backward compatibility.')
    parser.add_argument('--step-sharpness', action='store_true', dest='step_sharpness',
                        help='If set, compute the step sharpness: the current-mini-batch Rayleigh quotient g·Hg/g². Average across steps to recover the traditional batch sharpness.')
    parser.add_argument('--gni', action='store_true', help='If set, compute the Gradient-Noise Interaction quantity.')

    # --- Measurement Flags (Secondary, aka still useful) ---
    parser.add_argument('--hessian-trace', action='store_true', help='Estimate the trace of the full-batch loss Hessian via a Hutchinson-style estimator')
    parser.add_argument('--grad-projection', action='store_true', help='Compute grad_projection_i: fraction of full-batch gradient lying in span of top-i cached Hessian eigenvectors (i up to 20); uses cached eigenvectors only; only for plain SGD')
    parser.add_argument('--one-step-loss-change', action='store_true', help='If set, compute the expected one-step change in loss using Monte Carlo estimation')
    parser.add_argument('--gradient-norm', action='store_true', help='If set, compute the Monte Carlo estimate of squared norm of mini-batch gradients')
    parser.add_argument('--final', action='store_true', help='If set, compute the lambda_max and step sharpness at the end')
    
    # --- Measurement Flags (Tertiary, aka almost completely useless) ---
    parser.add_argument('--batch-sharpness-exp-inside', action='store_true', help='If set, compute the batch sharpness using E[gHg]/E[g²], where the expectation is inside the ratio. Compare with step-sharpness, where the expectation stays outside the ratio.')
    parser.add_argument('--batch-lambdamax','--batchlmax', action='store_true', help='If set, compute the batch lambda_max(H_B), aka batch lambda max')
    parser.add_argument('--fisher', action='store_true', help='If set, compute Fisher information matrix eigenvalue. Currently only works with one-dim output')
    parser.add_argument('--param-distance', '--param_distance', action='store_true', help='If set, compute the distance from the reference weights')
    parser.add_argument('--param-file', '--param_file', type=str, default=None, help='Path to reference parameters for computing parameter distance')
    parser.add_argument('--log-every-step', action='store_true', help='Force all configured measurements to log every training step, bypassing frequency rules.')
    parser.add_argument('--dense-window', nargs=3, type=int, metavar=('START', 'END', 'EVERY'),
                        default=None,
                        help='Dense measurement window: measure every EVERY steps in [START, END]. '
                             'E.g. --dense-window 400 3000 16 measures every 16 steps from step 400 to 3000.')

    # --- Measurement Configuration ---
    parser.add_argument('--disable-cache-eigenvectors', '--disable_cache_eigenvectors', action='store_true', help='If set, disable eigenvector caching for warm starts to improve eigenvalue computation performance')
    parser.add_argument('--use-power-iteration', '--use_power_iteration', action='store_true', help='If set, use power iteration method instead of LOBPCG for eigenvalue computation')
    parser.add_argument('--num-eigenvalues', '--num_eigenvalues', '--k', type=int, default=1, help='Number of eigenvalues to compute when computing lambda_max (default: 1)')

    parser.add_argument('--results-rarely', '--results_rarely', action='store_true', help='If set, results will be recorded less frequently')
    parser.add_argument('--precise-plots', action='store_true', help='Enable more frequent measurements for precise plotting')
    parser.add_argument('--rare-measure', dest='rare_measure', action='store_true', help='Activate regime where expensive measurements are performed rarely')

    # --- Noise Configuration ---
    parser.add_argument('--gd-noise', '--gd_noise', type=str, default=None, help='Do noisy GD, to simulate SGD. Supported noises: sgd, diag, iso, const')
    parser.add_argument('--noise-mag', '--noise_mag', type=float, default=None, help='The noise magnitude for the constant noise')
    
    # --- SDE Configuration ---
    parser.add_argument('--sde', action='store_true', help='Simulate the SDE dynamics (the one that correspond to the SGD). It integrates the SDE using the Euler-Maruyama method')
    parser.add_argument('--sde-h', '--sde_h', type=float, default=0.01, help='SDE *integration* time step size (default: 0.01)')
    parser.add_argument('--sde-eta', '--sde_eta', type=float, default=None, help='Learning rate for SDE (uses --lr if not specified)')
    parser.add_argument('--sde-seed', '--sde_seed', type=int, default=888, help='Random seed for SDE noise generation (default: 888)')

    # --- Quadratic Approximation Configuration ---
    parser.add_argument('--quad-switch-step', '--quad_switch_step', type=int, default=None, help='Step at which to switch from true NN dynamics to quadratic Taylor approximation dynamics')
    parser.add_argument('--use-gauss-newton', '--use_gauss_newton', action='store_true', help='Use Gauss-Newton matrix instead of Hessian for quadratic approximation')
    parser.add_argument('--quad-switch-lr', '--quad_switch_lr', type=float, default=None, help='lr to use after switching, used to test explosion')

    # --- Gradient Projection Configuration ---
    parser.add_argument('--proj-switch-step', dest='proj_switch_step', type=int, default=None,
                        help='Step number to start projecting minibatch gradient onto top-l Hessian eigendirections (full batch)')
    parser.add_argument('--proj-top-l', dest='proj_top_l', type=int, default=None,
                        help='Number of top Hessian eigendirections to project against/onto after switch step')
    parser.add_argument('--proj-to-residual', dest='proj_to_residual', action='store_true',
                        help='After --proj-switch-step, apply gradient projected to orthogonal complement of top-l eigenspace')

    # --- Randomness Settings ---
    parser.add_argument('--dataset-seed', '--dataset_seed', type=int, default=888, help='Random seed for dataset preparation')
    parser.add_argument('--init-seed', '--init_seed', type=int, default=8888, help='Random seed for network initialization')

    # --- wandb Settings ---
    parser.add_argument('--wandb-tag', type=str, default=None, help='Tag to add to the wandb run')
    parser.add_argument('--wandb-name', type=str, default=None, help='Optional suffix appended to default wandb run name (sanitized)')
    parser.add_argument('--wandb-notes', type=str, default=None, help='Optional notes/description attached to the wandb run')
    parser.add_argument('--disable-wandb', action='store_true', help='Disable Weights & Biases logging for debugging/testing')

    # --- New Measurement Flag ---
    parser.add_argument('--train-test-gap', action='store_true', help='If set, compute the training and testing accuracy and gap (heavy, runs rarely)')

    # --- NEW: Per-Sample Histogram Configuration ---

    parser.add_argument('--per-sample', action='store_true',
                        help='Track per-sample loss/residual/curvature histograms over time and save frames')
    parser.add_argument('--per-sample-every', type=int, default=100,
                        help='Snapshot cadence in steps for per-sample histograms (default: 100)')
    parser.add_argument('--hist-min-log10', type=float, default=-6.0,
                        help='Left edge for log10 binning (default: -6)')
    parser.add_argument('--hist-max-log10', type=float, default=2.0,
                        help='Right edge for log10 binning (default: 2)')
    parser.add_argument('--hist-bins', type=int, default=80,
                        help='Number of bins for log10 histograms (default: 80)')
    parser.add_argument('--per-sample-metrics', type=str, nargs='+',
                        default=['loss','resid','kappa'],
                        choices=['loss','resid','kappa'],
                        help='Which metrics to histogram (default: loss resid kappa)')
    parser.add_argument('--no-frames', action='store_true',
                        help='Only save counts/quantiles as .npz; do not render PNG frames')


    # --- NEW: Memorization via Outliers identified by Alignment with Top Hessian Eigenvector ---
    parser.add_argument('--memorization-hessian-outliers', action='store_true', help='Compute memorization stats based on alignment with top Hessian eigenvector (heavy; runs rarely)')
    parser.add_argument('--memorization-outlier-frac', type=float, default=0.05, help='Fraction of examples treated as outliers for Hessian-alignment memorization stats')

    # --- NEW: Outlier Mining Configuration ---
    parser.add_argument('--knn-outliers', action='store_true',
                        help='After training, run a k-NN pass to flag ambiguous samples')
    parser.add_argument('--knn-neighbors', type=int, default=32,
                        help='Number of neighbors used for the ambiguity score')
    parser.add_argument('--knn-top-per-class', type=int, default=10,
                        help='How many outliers to keep per class')
    parser.add_argument('--knn-feature-batch', type=int, default=512,
                        help='Batch size used during the post-hoc embedding pass')
    parser.add_argument('--knn-chunk-size', type=int, default=1024,
                        help='Chunk size used while computing the k-NN graph')
    parser.add_argument('--knn-no-normalize', action='store_true',
                        help='Disable L2-normalization of feature vectors before k-NN')
    parser.add_argument('--track-knn-outliers-from', type=str, default=None,
                        help='Existing plaintext run folder name (e.g., 20251124_0820_35_lr0.01000_b8) whose knn_outlier_indices.json should be tracked during training')
    parser.add_argument('--track-knn-topk', type=int, default=5,
                        help='Number of stored outliers per class to track from the reference run')

    # --- NEW: Feature-space Prototype Tracking ---
    parser.add_argument('--feature-prototypes', action='store_true',
                        help='After training, export feature-space prototype sets for reuse')
    parser.add_argument('--feature-prototype-batch', type=int, default=512,
                        help='Batch size for feature extraction when exporting feature prototypes')
    parser.add_argument('--feature-prototype-topk', type=int, default=50,
                        help='Maximum prototypes per class to store from feature space')
    parser.add_argument('--feature-prototype-kneighbors', type=int, default=32,
                        help='k-NN neighborhood size when identifying boundary points in feature space')
    parser.add_argument('--feature-prototype-no-normalize', action='store_true',
                        help='Disable L2 normalization before computing feature-space prototypes')
    parser.add_argument('--feature-prototype-extrapolation', type=float, default=EXTRAPOLATION_FACTOR,
                        help='Extrapolation factor used when building feature-space x-outliers')
    parser.add_argument('--track-feature-prototypes-from', type=str, default=None,
                        help='Existing plaintext run folder whose feature-space prototype sets should be tracked during training')
    parser.add_argument('--train-input-prototypes', type=str, default=None,
                        help='Input prototype source for training run: generate | from:<path or run> | none')
    parser.add_argument('--test-input-prototypes', type=str, default=None,
                        help='Input prototype source for logging/eval: from:<path or run> | none (defaults to train-input-prototypes when unset)')
    parser.add_argument('--input-prototypes-frac', type=float, default=None,
                        help='Fraction of training set (per class) to use for input prototypes when generating')
    parser.add_argument('--input-prototypes-count', type=int, default=None,
                        help='Count per class to use for all input prototype subsets when generating')
    parser.add_argument('--input-prototypes-boundary-count', type=int, default=None,
                        help='Override count per class for boundary prototypes')
    parser.add_argument('--input-prototypes-inliers-count', type=int, default=None,
                        help='Override count per class for inlier prototypes')
    parser.add_argument('--input-prototypes-x-outlier-count', type=int, default=None,
                        help='Override count per class for x-outlier prototypes')
    parser.add_argument('--input-prototypes-y-outlier-count', type=int, default=None,
                        help='Override count per class for y-outlier prototypes')
    parser.add_argument('--input-prototypes-holdout', type=str, default='auto',
                        choices=['auto', 'boundary_inliers', 'none'],
                        help='Hold out boundary/inlier indices from training when using train-input-prototypes (auto => boundary_inliers for new flags)')
    parser.add_argument('--track-input-prototypes', action='store_true',
                        help='DEPRECATED: Track/log input-space prototype subsets (boundary/inliers/synthetic outliers) on wandb')
    parser.add_argument('--input-prototype-holdout-per-class', type=int, default=None,
                        help='DEPRECATED: hold out this many boundary/inlier points per class from training')
    parser.add_argument('--input-prototype-holdout-frac', type=float, default=None,
                        help='DEPRECATED: hold out this fraction of the training set per class (used to size boundary/inlier prototypes)')
    parser.add_argument('--train-input-x-outliers', type=int, default=None,
                        help='Augment the training set with this many input-space x-outliers per class; '
                             'also logs input-space prototype subsets')
    parser.add_argument('--x-outlier-mode', choices=['coherent', 'random_direction'], default='coherent',
                        help='X-outlier displacement mode: "coherent" (along v_diff, default) or '
                             '"random_direction" (random orthogonal directions, same displacement magnitude)')
    parser.add_argument('--random-direction-seed', type=int, default=42,
                        help='RNG seed for random_direction x-outlier mode (default 42)')
    parser.add_argument('--input-extrapolation-factor', type=float, default=EXTRAPOLATION_FACTOR,
                        help=f'Override extrapolation factor alpha for input-space x-outlier generation '
                             f'(default {EXTRAPOLATION_FACTOR}).')
    parser.add_argument('--train-input-y-outliers', type=int, default=None,
                        help='Augment the training set with this many input-space y-outliers per class; '
                             'also logs input-space prototype subsets')
    parser.add_argument('--train-input-inliers', type=int, default=None,
                        help='Augment the training set with this many input inliers per class; '
                             'also logs input-space prototype subsets')
    parser.add_argument('--train-input-boundary', type=int, default=None,
                        help='Augment the training set with this many input boundary points per class; '
                             'also logs input-space prototype subsets')
    # ----- Argument Parsing -----
    args = parser.parse_args()


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
    batch_size = args.batch
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
    # Allow tracking input prototypes even when training with input outliers.

    for flag_name in ("train_input_x_outliers", "train_input_y_outliers"):
        flag_value = getattr(args, flag_name)
        if flag_value is not None and flag_value < 1:
            raise ValueError(f"--{flag_name.replace('_', '-')} must be >= 1 when provided")

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

    if args.input_prototype_holdout_per_class is not None and args.input_prototype_holdout_per_class < 1:
        raise ValueError("--input-prototype-holdout-per-class must be >= 1 when provided")
    if args.input_prototype_holdout_frac is not None:
        if args.input_prototype_holdout_frac <= 0 or args.input_prototype_holdout_frac >= 1:
            raise ValueError("--input-prototype-holdout-frac must be in (0, 1)")
    if (
        args.input_prototype_holdout_per_class is not None
        or args.input_prototype_holdout_frac is not None
    ) and not args.track_input_prototypes and args.train_input_prototypes is None:
        print("Warning: input prototype holdout sizing was provided without input prototype tracking; ignoring holdout.")

    if args.memorization_outlier_frac <= 0 or args.memorization_outlier_frac >= 1:
        raise ValueError("--memorization-outlier-frac must be in (0, 1)")

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

    if args.feature_prototypes and (args.classes is None or len(args.classes) != 2):
        raise ValueError("--feature-prototypes requires specifying exactly two classes via --classes")

    if args.lmax_decay_steps < 1:
        raise ValueError("--lmax-decay-steps must be >= 1")
    if args.lmax_decay_target_lr < 0:
        raise ValueError("--lmax-decay-target-lr must be >= 0")
    if args.lmax_decay and not args.lambdamax:
        print("--lmax-decay requires --lambdamax; enabling lambda_max measurement.")
        args.lambdamax = True
    
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
    ('memorization_hessian_outliers', args.memorization_hessian_outliers),
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


    use_new_proto_flags = args.train_input_prototypes is not None or args.test_input_prototypes is not None
    train_proto_source = _parse_input_prototype_source(args.train_input_prototypes)
    test_proto_source = _parse_input_prototype_source(args.test_input_prototypes)

    if train_proto_source["mode"] is None and args.track_input_prototypes:
        train_proto_source = {"mode": "generate", "value": None}

    if test_proto_source["mode"] is None and train_proto_source["mode"] not in (None, "none"):
        test_proto_source = {"mode": train_proto_source["mode"], "value": train_proto_source["value"]}

    input_proto_counts = _build_input_prototype_counts(args)

    def _base_proto_count():
        if args.input_prototypes_count is not None:
            return args.input_prototypes_count
        if args.input_prototypes_frac is not None:
            return max(1, int(round(train_x.shape[0] * args.input_prototypes_frac)))
        if not use_new_proto_flags:
            if args.input_prototype_holdout_per_class is not None:
                return args.input_prototype_holdout_per_class
            if args.input_prototype_holdout_frac is not None:
                return max(1, int(round(train_x.shape[0] * args.input_prototype_holdout_frac)))
        return max(1, int(round(train_x.shape[0] * 0.05)))

    n_proto = _base_proto_count()
    proto_classes = (0, 1) if dataset == 'cifar10_2cls' else tuple(args.classes)
    # Inlier pool must be large enough for inlier + y-outlier + x-outlier injection
    n_inlier_needed = (
        (args.train_input_inliers or 0)
        + (args.train_input_y_outliers or 0)
        + (args.train_input_x_outliers or 0)
    )
    n_boundary_needed = args.train_input_boundary or 0
    if n_inlier_needed > 0 or n_boundary_needed > 0:
        n_proto = max(n_proto, n_inlier_needed, n_boundary_needed)

    train_prototype_data = None
    train_prototype_indices = None
    if train_proto_source["mode"] == "from":
        proto_path = resolve_input_prototype_path(
            train_proto_source["value"],
            results_root=RES_FOLDER,
            dataset=dataset,
            model=args.model,
        )
        train_prototype_data, train_prototype_indices, proto_meta = load_input_prototype_package(proto_path)
        _validate_input_prototype_metadata(
            proto_meta,
            dataset=dataset,
            classes=args.classes,
            dataset_seed=args.dataset_seed,
            num_data=args.num_data,
        )
        train_prototype_data, train_prototype_indices = trim_prototype_sets(
            train_prototype_data,
            tuple(args.classes),
            input_proto_counts,
            train_prototype_indices,
        )
        print(f"Loaded input prototypes from {proto_path}")
    elif train_proto_source["mode"] == "generate":
        train_prototype_data, train_prototype_indices = generate_prototype_sets(
            train_x, train_y, proto_classes, n_prototype=n_proto,
            n_boundary=n_boundary_needed if n_boundary_needed > 0 else None,
            n_inlier=n_inlier_needed if n_inlier_needed > 0 else None,
            return_indices=True,
        )
        train_prototype_data, train_prototype_indices = trim_prototype_sets(
            train_prototype_data,
            proto_classes,
            input_proto_counts,
            train_prototype_indices,
        )

    log_prototype_data = None
    if test_proto_source["mode"] == "from":
        proto_path = resolve_input_prototype_path(
            test_proto_source["value"],
            results_root=RES_FOLDER,
            dataset=dataset,
            model=args.model,
        )
        log_prototype_data, _, proto_meta = load_input_prototype_package(proto_path)
        _validate_input_prototype_metadata(
            proto_meta,
            dataset=dataset,
            classes=args.classes,
            dataset_seed=args.dataset_seed,
            num_data=args.num_data,
        )
        log_prototype_data, _ = trim_prototype_sets(
            log_prototype_data,
            tuple(args.classes),
            input_proto_counts,
            None,
        )
        print(f"Loaded test input prototypes from {proto_path}")
    elif test_proto_source["mode"] == "generate":
        log_prototype_data, _ = generate_prototype_sets(
            train_x, train_y, proto_classes, n_prototype=n_proto, return_indices=True
        )
        log_prototype_data, _ = trim_prototype_sets(
            log_prototype_data,
            proto_classes,
            input_proto_counts,
            None,
        )

    prototype_data = log_prototype_data or train_prototype_data or {}
    combined_prototype_data = dict(prototype_data)
    if args.track_feature_prototypes_from:
        tracked_feature_prototypes, _ = _load_reference_feature_prototypes(
            dataset,
            args.model,
            args.track_feature_prototypes_from,
        )
        for name, tensors in tracked_feature_prototypes.items():
            combined_prototype_data[name] = tensors
        print(
            f"Loaded feature-space prototypes from run {args.track_feature_prototypes_from} "
            f"({len(tracked_feature_prototypes)} subsets)"
        )

    prototype_data = combined_prototype_data
    prototype_data_for_aug = train_prototype_data

    holdout_mode = args.input_prototypes_holdout
    if holdout_mode == "auto":
        if use_new_proto_flags and train_proto_source["mode"] not in (None, "none"):
            holdout_mode = "boundary_inliers"
        elif args.track_input_prototypes and (
            args.input_prototype_holdout_per_class is not None or args.input_prototype_holdout_frac is not None
        ):
            holdout_mode = "boundary_inliers"
        else:
            holdout_mode = "none"

    if holdout_mode == "boundary_inliers" and train_prototype_indices:
        holdout_tensors = []
        boundary_idx = train_prototype_indices.get("boundary") if train_prototype_indices else None
        inlier_idx = train_prototype_indices.get("inliers") if train_prototype_indices else None
        if boundary_idx is not None:
            holdout_tensors.append(boundary_idx)
        if inlier_idx is not None:
            holdout_tensors.append(inlier_idx)
        if holdout_tensors:
            holdout_indices = torch.unique(torch.cat(holdout_tensors, dim=0))
            if holdout_indices.numel() > 0:
                orig_n = train_x.shape[0]
                train_x, train_y = _drop_indices_from_dataset(train_x, train_y, holdout_indices)
                removed = orig_n - train_x.shape[0]
                print(
                    f"[holdout] removed {removed} samples from training set for input prototypes "
                    f"(boundary+inliers, {holdout_indices.numel()} unique indices)"
                )
                data = (train_x, train_y, test_x, test_y)
                tuple_data = data
    elif holdout_mode == "boundary_inliers" and train_proto_source["mode"] not in (None, "none"):
        print("Warning: input prototype holdout requested but no indices were available; skipping holdout.")

    has_injection = (args.train_input_x_outliers is not None or args.train_input_y_outliers is not None
                     or args.train_input_inliers is not None or args.train_input_boundary is not None)

    if has_injection and train_prototype_data is None:
        raise ValueError("Training input injection requires train-input-prototypes.")

    train_outlier_tracking = {}
    if has_injection:
        classes = proto_classes
        if len(classes) != 2:
            raise ValueError("Training input injection requires exactly two classes.")

        # --- Partition the held-out pools for injection ---
        # The saved prototype package contains:
        #   boundary pool: all held-out boundary points
        #   inliers pool:  all held-out inlier points (mutually exclusive with boundary)
        #
        # Injection partitions the inlier pool into three disjoint slices:
        #   slice 0 → injected as inliers (correct label)
        #   slice 1 → injected as y-outliers (flipped label)
        #   slice 2 → injected as x-outliers (extrapolated, correct label)
        # Boundary pool is injected directly.
        # All partitioning uses deterministic per-class slicing (first N per class).

        X_inlier_pool, Y_inlier_pool = prototype_data_for_aug["inliers"]
        X_boundary_pool, Y_boundary_pool = prototype_data_for_aug["boundary"]

        # Determine per-class counts needed from the inlier pool
        n_inlier_inject = args.train_input_inliers or 0
        n_y_outlier = args.train_input_y_outliers or 0
        n_x_outlier = args.train_input_x_outliers or 0
        n_boundary_inject = args.train_input_boundary or 0

        # Validate pool sizes
        inlier_labels = Y_inlier_pool.argmax(dim=1) if Y_inlier_pool.ndim > 1 else Y_inlier_pool.long()
        boundary_labels = Y_boundary_pool.argmax(dim=1) if Y_boundary_pool.ndim > 1 else Y_boundary_pool.long()

        inlier_needed_per_class = n_inlier_inject + n_y_outlier + n_x_outlier
        for cls in classes:
            n_avail_inlier = (inlier_labels == cls).sum().item()
            if n_avail_inlier < inlier_needed_per_class:
                raise ValueError(
                    f"Inlier pool has {n_avail_inlier} samples for class {cls} but injection "
                    f"needs {inlier_needed_per_class} (inlier={n_inlier_inject} + "
                    f"y_outlier={n_y_outlier} + x_outlier={n_x_outlier}). "
                    f"Increase the inlier pool size in the prototype package."
                )
            n_avail_boundary = (boundary_labels == cls).sum().item()
            if n_avail_boundary < n_boundary_inject:
                raise ValueError(
                    f"Boundary pool has {n_avail_boundary} samples for class {cls} but "
                    f"injection needs {n_boundary_inject}."
                )

        # Partition inlier pool by class into disjoint slices
        outlier_augments = []

        def _partition_pool_by_class(X, Y, classes, *slice_sizes):
            """Split a pool into disjoint per-class slices.

            Returns a list of (X_slice, Y_slice) tuples, one per slice_size entry.
            """
            labels = Y.argmax(dim=1) if Y.ndim > 1 else Y.long()
            slices = [[] for _ in slice_sizes]
            slice_labels = [[] for _ in slice_sizes]
            for cls in classes:
                cls_mask = labels == cls
                cls_indices = cls_mask.nonzero(as_tuple=False).view(-1)
                offset = 0
                for i, sz in enumerate(slice_sizes):
                    if sz > 0:
                        sel = cls_indices[offset:offset + sz]
                        slices[i].append(X[sel])
                        slice_labels[i].append(Y[sel])
                        offset += sz
            result = []
            for i in range(len(slice_sizes)):
                if slice_sizes[i] > 0 and slices[i]:
                    result.append((torch.cat(slices[i], dim=0), torch.cat(slice_labels[i], dim=0)))
                else:
                    result.append(None)
            return result

        inlier_slices = _partition_pool_by_class(
            X_inlier_pool, Y_inlier_pool, classes,
            n_inlier_inject, n_y_outlier, n_x_outlier,
        )
        inlier_slice, y_outlier_source, x_outlier_source = inlier_slices

        # Inject inliers (correct labels, as-is)
        if inlier_slice is not None:
            X_in, Y_in = inlier_slice
            train_outlier_tracking["inliers"] = (X_in, Y_in)
            outlier_augments.append((X_in, _coerce_labels_like(train_y, Y_in)))

        # Inject y-outliers (same images, flipped labels)
        if y_outlier_source is not None:
            X_ysrc, Y_ysrc = y_outlier_source
            # Flip labels: class_0 ↔ class_1
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
            c0 = train_x[mask_0].view(mask_0.sum(), -1).mean(dim=0, keepdim=True)
            c1 = train_x[mask_1].view(mask_1.sum(), -1).mean(dim=0, keepdim=True)
            v_diff = c1 - c0

            X_flat = X_xsrc.view(X_xsrc.shape[0], -1)
            extrapolated = torch.zeros_like(X_flat)
            alpha = float(getattr(args, "input_extrapolation_factor", EXTRAPOLATION_FACTOR))
            displacement_norm = alpha * v_diff.norm().item()

            if args.x_outlier_mode == "random_direction":
                rng = torch.Generator()
                rng.manual_seed(args.random_direction_seed)
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
                        extrapolated[i] = X_flat[i] - alpha * v_diff
                    else:
                        extrapolated[i] = X_flat[i] + alpha * v_diff

            X_x_out = extrapolated.view_as(X_xsrc)
            Y_x_out = x_labels.clone()
            train_outlier_tracking["x_outlier"] = (X_x_out, Y_x_out)
            outlier_augments.append((X_x_out, _coerce_labels_like(train_y, Y_x_out)))

        # Inject boundary (as-is from boundary pool)
        if n_boundary_inject > 0:
            boundary_slices = _partition_pool_by_class(
                X_boundary_pool, Y_boundary_pool, classes,
                n_boundary_inject,
            )
            boundary_slice = boundary_slices[0]
            if boundary_slice is not None:
                X_b, Y_b = boundary_slice
                train_outlier_tracking["boundary"] = (X_b, Y_b)
                outlier_augments.append((X_b, _coerce_labels_like(train_y, Y_b)))

        if outlier_augments:
            aug_X = [train_x] + [payload[0] for payload in outlier_augments]
            aug_Y = [train_y] + [payload[1] for payload in outlier_augments]
            train_x = torch.cat(aug_X, dim=0)
            train_y = torch.cat(aug_Y, dim=0)
            data = (train_x, train_y, test_x, test_y)
            tuple_data = data

    effective_train_len = int(train_x.shape[0])
    if effective_train_len == 0:
        raise ValueError("Training set is empty after holdout/augmentation.")
    if batch_size > effective_train_len:
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
    
    knn_outlier_cfg = None
    if args.knn_outliers:
        if args.knn_neighbors < 2:
            raise ValueError("--knn-neighbors must be >= 2 when --knn-outliers is set")
        if args.knn_top_per_class < 1:
            raise ValueError("--knn-top-per-class must be >= 1 when --knn-outliers is set")

        knn_outlier_cfg = {
            "enabled": True,
            "k_neighbors": args.knn_neighbors,
            "top_k_per_class": args.knn_top_per_class,
            "feature_batch_size": args.knn_feature_batch,
            "chunk_size": args.knn_chunk_size,
            "normalize": not args.knn_no_normalize,
            "return_neighbor_indices": True,
        }

    subset_tracking_cfgs = prepare_knn_subset_tracking_configs(args, dataset, args.model, data) if args.track_knn_outliers_from else []
    subset_tracking_cfgs.extend(prepare_feature_prototype_subset_configs(prototype_data))
    track_input_metrics = bool(prototype_data) or bool(train_outlier_tracking)
    if track_input_metrics:
        subset_tracking_cfgs.extend(prepare_prototype_subset_configs(prototype_data, base_batch_size=batch_size))
    
    
    if train_outlier_tracking:
        injected_tracking = {
            f"injected_{name}": tensors
            for name, tensors in train_outlier_tracking.items()
        }
        subset_tracking_cfgs.extend(prepare_prototype_subset_configs(injected_tracking, base_batch_size=batch_size))


    per_sample_cfg = None
    if args.per_sample:
        per_sample_cfg = {
            'enabled': True,
            'every': max(1, int(args.per_sample_every)),
            'hist_min_log10': args.hist_min_log10,
            'hist_max_log10': args.hist_max_log10,
            'hist_bins': args.hist_bins,
            'metrics': args.per_sample_metrics,   # ['loss','resid','kappa']
            'no_frames': args.no_frames,
        }

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
        memorization_outlier_frac=args.memorization_outlier_frac,
        proj_switch_step=args.proj_switch_step,
        proj_top_l=args.proj_top_l,
        proj_to_residual=args.proj_to_residual,
        wandb_run=wandb_run,
        wandb_enabled=wandb_enabled,
        wandb_run_id=wandb_run_id,
        #NEW
        per_sample_cfg=per_sample_cfg,
        knn_outlier_cfg=knn_outlier_cfg,
        subset_tracking_cfgs=subset_tracking_cfgs,
        prototype_data=prototype_data,
        log_every_step=args.log_every_step,
        dense_window=tuple(args.dense_window) if args.dense_window else None,
        lmax_decay=args.lmax_decay,
        lmax_decay_target_lr=args.lmax_decay_target_lr,
        lmax_decay_steps=args.lmax_decay_steps,
        lmax_decay_initial_lr=args.lr,
        lmax_drop=args.lmax_drop,
        lmax_drop_mult=args.lmax_drop_mult,
        lmax_drop_target_lr=args.lmax_drop_target_lr,
        lr_drop_at_step=args.lr_drop_at_step,
        lr_drop_to=args.lr_drop_to,
    )

    if args.feature_prototypes:
        print("Computing feature-space prototypes for export...")
        feature_batch = args.feature_prototype_batch
        features = extract_feature_matrix(
            net=net,
            inputs=train_x,
            batch_size=feature_batch,
            flatten_outputs=True,
        )
        proto_sets, proto_meta = generate_feature_space_prototype_sets(
            net=None,
            Y_train=train_y,
            classes=tuple(args.classes),
            precomputed_features=features,
            original_inputs=train_x,
            normalize_features=not args.feature_prototype_no_normalize,
            k_neighbors=args.feature_prototype_kneighbors,
            prototypes_per_class=args.feature_prototype_topk,
            extrapolation_factor=args.feature_prototype_extrapolation,
        )
        _save_feature_prototype_package(run_folder, proto_sets, proto_meta)
        print(f"Feature-space prototypes saved to {run_folder / 'feature_prototypes'}")
