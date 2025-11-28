import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio


COLORS = {
    "x_outlier": "red",
    "y_outlier": "blue",
    "boundary": "green",
    "inliers": "purple",
}


PROTO_NAMES = ["x_outlier", "y_outlier", "boundary", "inliers"]


def make_proto_multi_gif(proto_dir, metric, out_path, fps=4):
    proto_dir = Path(proto_dir)
    out_path = Path(out_path)

    print(f"[proto-multi-gif] proto_dir = {proto_dir.resolve()}")
    print(f"[proto-multi-gif] metric    = {metric}")

    sample_files = sorted(proto_dir.glob(f"step_*_{PROTO_NAMES[0]}.npz"))
    if not sample_files:
        print("[proto-multi-gif] No prototype files found.")
        return

    # Extract steps: step_00100_x_outlier → "00100"
    steps = [f.stem.split("_")[1] for f in sample_files]

    frames = []
    tmp_dir = proto_dir / f"_gif_tmp_all_{metric}"
    tmp_dir.mkdir(exist_ok=True)

    for step_str in steps:
        plt.figure(figsize=(6, 5))

        for proto in PROTO_NAMES:
            f = proto_dir / f"step_{step_str}_{proto}.npz"
            if not f.exists():
                continue
            d = np.load(f)
            if metric not in d:
                raise KeyError(
                    f"Metric '{metric}' missing from {f.name}. Keys: {list(d.keys())}"
                )
            vals = d[metric]
            plt.hist(
                vals,
                bins=30,
                alpha=0.5,
                label=proto,
                color=COLORS.get(proto, None),
            )

        plt.title(f"All prototypes – {metric} – step {int(step_str)}")
        plt.xlabel(metric)
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()

        png_path = tmp_dir / f"{step_str}.png"
        plt.savefig(png_path, dpi=120)
        plt.close()
        frames.append(imageio.imread(png_path))

    imageio.mimsave(out_path, frames, fps=fps)
    print(f"[proto-multi-gif] Saved GIF → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto-dir", required=True)
    parser.add_argument("--metric", required=True,
                        choices=["loss", "resid", "kappa", "grad_norm"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args()

    make_proto_multi_gif(
        proto_dir=args.proto_dir,
        metric=args.metric,
        out_path=args.out,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()

