import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Helpers to find latest run
# -----------------------------
def get_results_root() -> Path:
    results_env = os.environ.get("RESULTS")
    if results_env is None:
        raise RuntimeError("Environment variable RESULTS is not set.")
    return Path(results_env)


def get_latest_run(results_root: Path) -> Path:
    """
    Find the most recently modified run folder under
    $RESULTS/plaintext/cifar10_mlp/*
    """
    base = results_root / "plaintext" / "cifar10_mlp"
    if not base.exists():
        raise FileNotFoundError(f"{base} does not exist")

    candidates = [p for p in base.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {base}")

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def extract_step_from_name(fname: str) -> int | None:
    """
    Expect filenames like 'step_00000_boundary.npz'
    """
    m = re.search(r"step_(\d+)_", fname)
    return int(m.group(1)) if m else None


# -----------------------------
# Main
# -----------------------------
def main():
    results_root = get_results_root()
    run_dir = get_latest_run(results_root)
    print(f"[INFO] Using run directory: {run_dir}")

    proto_dir = run_dir / "per_sample_histograms" / "prototypes"
    if not proto_dir.exists():
        raise FileNotFoundError(f"Prototype directory not found: {proto_dir}")

    print(f"[INFO] Prototype directory: {proto_dir}")

    # The four prototype names used in your code
    prototype_names = ["boundary", "x_outlier", "y_outlier", "inliers"]

    # time_series[name] = { "step": [], "loss": [], "curv": [] }
    time_series: dict[str, dict[str, list[float]]] = {
        name: {"step": [], "loss": [], "curv": []} for name in prototype_names
    }

    # -----------------------------
    # Load all .npz files
    # -----------------------------
    npz_files = sorted(proto_dir.glob("step_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No step_*.npz files found in {proto_dir}")

    for f in npz_files:
        step = extract_step_from_name(f.name)
        if step is None:
            continue

        # Expect filename pattern: step_00000_boundary.npz
        # -> last token before .npz is the prototype name
        name_part = f.stem.split("_", 2)[-1]
        if name_part not in prototype_names:
            # Skip anything unexpected
            continue

        data = np.load(f)

        # From your training code, each file has:
        # 'loss', 'resid', 'kappa', 'grad_norm', 'batch_sharpness', 'mean_loss'
        mean_loss = float(data["mean_loss"])
        batch_sharpness = float(data["batch_sharpness"])

        ts = time_series[name_part]
        ts["step"].append(step)
        ts["loss"].append(mean_loss)
        ts["curv"].append(batch_sharpness)

    # Sort each prototype’s timeseries by step (in case file order is weird)
    for name in prototype_names:
        ts = time_series[name]
        if not ts["step"]:
            print(f"[WARN] No data collected for prototype '{name}'")
            continue
        # sort by step
        order = np.argsort(ts["step"])
        ts["step"] = np.array(ts["step"])[order]
        ts["loss"] = np.array(ts["loss"])[order]
        ts["curv"] = np.array(ts["curv"])[order]

    # -----------------------------
    # Plot: step vs curvature
    # -----------------------------
    plt.figure(figsize=(7, 5))
    for name in prototype_names:
        ts = time_series[name]
        if len(ts["step"]) == 0:
            continue
        plt.plot(ts["step"], ts["curv"], label=name)
    plt.xlabel("Training step")
    plt.ylabel("Batch sharpness (gᵀHg / ||g||²)")
    plt.title("Curvature vs Step for Prototype Sets")
    plt.legend()
    plt.tight_layout()

    curv_out = run_dir / "prototype_curvature_vs_step.png"
    plt.savefig(curv_out, dpi=200)
    print(f"[INFO] Saved curvature plot to {curv_out}")

    # -----------------------------
    # Plot: step vs mean loss
    # -----------------------------
    plt.figure(figsize=(7, 5))
    for name in prototype_names:
        ts = time_series[name]
        if len(ts["step"]) == 0:
            continue
        plt.plot(ts["step"], ts["loss"], label=name)
    plt.xlabel("Training step")
    plt.ylabel("Mean loss over prototype set")
    plt.title("Loss vs Step for Prototype Sets")
    plt.legend()
    plt.tight_layout()

    loss_out = run_dir / "prototype_loss_vs_step.png"
    plt.savefig(loss_out, dpi=200)
    print(f"[INFO] Saved loss plot to {loss_out}")

    print("[DONE] Plots written to:", run_dir)


if __name__ == "__main__":
    main()

