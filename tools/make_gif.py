import imageio.v2 as imageio
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--frames-dir', required=True)
parser.add_argument('--metric', required=True, choices=['loss','resid','kappa'])
parser.add_argument('--out', required=True)
parser.add_argument('--fps', type=int, default=6)
args = parser.parse_args()

frames = sorted(Path(args.frames_dir).glob(f"*_{args.metric}.png"))
images = [imageio.imread(str(p)) for p in frames]
imageio.mimsave(args.out, images, fps=args.fps)
print(f'Wrote {args.out} with {len(images)} frames')



