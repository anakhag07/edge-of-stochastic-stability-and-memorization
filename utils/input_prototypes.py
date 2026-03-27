from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from utils.data import trim_prototype_sets


INPUT_PROTOTYPE_DIRNAME = "input_prototypes"
INPUT_PROTOTYPE_SUBSETS = ("boundary", "inliers", "x_outlier", "y_outlier")


def parse_input_prototype_source(raw: Optional[str]) -> Dict[str, Optional[str]]:
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


def build_input_subset_counts(args) -> Dict[str, int]:
    counts = {}
    if getattr(args, "input_boundary", None) is not None:
        counts["boundary"] = args.input_boundary
    if getattr(args, "input_inliers", None) is not None:
        counts["inliers"] = args.input_inliers
    if getattr(args, "input_x_outliers", None) is not None:
        counts["x_outlier"] = args.input_x_outliers
    if getattr(args, "input_y_outliers", None) is not None:
        counts["y_outlier"] = args.input_y_outliers
    return counts


def select_input_prototype_subsets(
    prototypes: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    indices: Optional[Dict[str, torch.Tensor]],
    *,
    classes: Tuple[int, int],
    counts_by_subset: Dict[str, int],
) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
    if not counts_by_subset:
        return {}, {} if indices is not None else None

    trimmed, trimmed_indices = trim_prototype_sets(
        prototypes,
        classes,
        counts_by_subset,
        indices,
    )

    selected = {}
    selected_indices = {} if trimmed_indices is not None else None
    for name in counts_by_subset:
        if name not in trimmed:
            raise ValueError(f"Requested input prototype subset '{name}' was not available.")
        selected[name] = trimmed[name]
        if selected_indices is not None and trimmed_indices and name in trimmed_indices:
            selected_indices[name] = trimmed_indices[name]

    return selected, selected_indices


def build_input_prototype_metadata(
    *,
    dataset: str,
    classes,
    dataset_seed: int,
    num_data: int,
    loss_type: str,
    counts_by_subset: Dict[str, int],
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "classes": list(classes),
        "dataset_seed": dataset_seed,
        "num_data": num_data,
        "loss_type": loss_type,
        "sizing": {
            "counts_by_subset": dict(counts_by_subset),
        },
    }


def _to_cpu(payload):
    if torch.is_tensor(payload):
        return payload.detach().cpu()
    if isinstance(payload, dict):
        return {k: _to_cpu(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_to_cpu(v) for v in payload]
    return payload


def save_input_prototype_package(
    save_dir: Path,
    prototypes: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    *,
    indices: Optional[Dict[str, torch.Tensor]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    target_dir = save_dir / INPUT_PROTOTYPE_DIRNAME
    target_dir.mkdir(parents=True, exist_ok=True)

    package = {
        "prototypes": {
            name: {
                "inputs": X.detach().cpu(),
                "labels": Y.detach().cpu(),
            }
            for name, (X, Y) in prototypes.items()
        },
        "metadata": _to_cpu(metadata or {}),
    }
    if indices:
        package["indices"] = _to_cpu(indices)

    package_path = target_dir / "prototypes.pt"
    torch.save(package, package_path)

    summary = {
        "counts": {name: int(payload["inputs"].shape[0]) for name, payload in package["prototypes"].items()},
    }
    if metadata:
        for key in ("dataset", "classes", "dataset_seed", "num_data", "loss_type", "sizing"):
            if key in metadata:
                summary[key] = metadata[key]
    if indices:
        summary["indices"] = {
            name: [int(idx) for idx in tensor.tolist()] for name, tensor in _to_cpu(indices).items()
        }

    summary_path = target_dir / "summary.json"
    with open(summary_path, "w") as f:
        import json

        json.dump(summary, f, indent=2)

    return target_dir


def resolve_input_prototype_path(
    value: str,
    *,
    results_root: Path,
    dataset: str,
    model: str,
) -> Path:
    raw = value.strip()
    if raw.startswith("from:"):
        raw = raw[5:]
    if raw.startswith("run:"):
        raw = raw[4:]

    candidate = Path(raw).expanduser()
    if candidate.exists():
        if candidate.is_dir():
            if (candidate / "prototypes.pt").exists():
                return candidate / "prototypes.pt"
            if (candidate / INPUT_PROTOTYPE_DIRNAME / "prototypes.pt").exists():
                return candidate / INPUT_PROTOTYPE_DIRNAME / "prototypes.pt"
        return candidate

    run_dir = results_root / "plaintext" / f"{dataset}_{model}" / raw
    return run_dir / INPUT_PROTOTYPE_DIRNAME / "prototypes.pt"


def load_input_prototype_package(path: Path):
    if path.is_dir():
        path = path / "prototypes.pt"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find input prototype package at {path}.")

    package = torch.load(path, map_location="cpu")
    proto_payload = package.get("prototypes", {})
    if not proto_payload:
        raise ValueError(f"No prototype tensors stored in {path}")

    prototypes = {}
    for name, payload in proto_payload.items():
        inputs = payload.get("inputs")
        labels = payload.get("labels")
        if inputs is None or labels is None:
            continue
        prototypes[name] = (inputs, labels)

    if not prototypes:
        raise ValueError(f"Input prototype file {path} did not contain usable tensors.")

    indices_payload = package.get("indices", {})
    indices = {}
    for name, idx in indices_payload.items():
        if idx is None:
            continue
        if torch.is_tensor(idx):
            indices[name] = idx.to(dtype=torch.long)
        else:
            indices[name] = torch.tensor(idx, dtype=torch.long)

    metadata = package.get("metadata", {})
    return prototypes, (indices or None), metadata
