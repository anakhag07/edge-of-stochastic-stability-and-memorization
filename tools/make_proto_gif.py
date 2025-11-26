import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def make_proto_gif(proto_dir, proto_name, metric, out_path, fps=4):
    proto_dir = Path(proto_dir)

    print(f"[proto-gif] proto_dir = {proto_dir.resolve()}")
    print(f"[proto-gif] searching for: step_*_{proto_name}.npz")

    files = sorted(proto_dir.glob(f"step_*_{proto_name}.npz"))
    if not files:
        print(f"[proto-gif] No files found for proto '{proto_name}'")
        return

    print(f"[proto-gif] Found {len(files)} files")

    tmp_dir = proto_dir / f"_gif_tmp_{proto_name}_{metric}"
    tmp_dir.mkdir(exist_ok=True)

    frames = []

    for f in files:
        d = np.load(f)
        if metric not in d:
            raise KeyError(
                f"Metric '{metric}' not in {f.name}. Available keys: {list(d.keys())}"
            )

        vals = d[metric]
        step_str = f.stem.split("_")[1]  # step_00000_x_outlier → "00000"
        step = int(step_str)

        plt.figure(figsize=(5, 4))
        plt.hist(vals, bins=30, alpha=0.85)
        plt.title(f"{proto_name} – {metric} at step {step}")
        plt.xlabel(metric)
        plt.ylabel("count")
        plt.tight_layout()

        png_path = tmp_dir / f"{step_str}.png"
        plt.savefig(png_path, dpi=120)
        plt.close()

        frames.append(imageio.imread(png_path))

    imageio.mimsave(out_path, frames, fps=fps)
    print(f"[proto-gif] Saved GIF → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto-dir", required=True)
    parser.add_argument("--proto", required=True)
    parser.add_argument("--metric", required=True,
                        choices=["loss", "resid", "kappa", "grad_norm"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args()

    make_proto_gif(
        proto_dir=args.proto_dir,
        proto_name=args.proto,
        metric=args.metric,
        out_path=args.out,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()

