from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch


INPUT_PROTOTYPE_DIRNAME = "input_prototypes"


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
