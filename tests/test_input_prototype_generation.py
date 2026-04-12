from pathlib import Path

import torch

from utils.data import generate_prototype_sets


def _make_two_class_dataset(samples_per_class: int = 8):
    class0 = torch.stack(
        [torch.tensor([float(i), float(i) * 0.1]) for i in range(samples_per_class)],
        dim=0,
    )
    class1 = torch.stack(
        [torch.tensor([10.0 + float(i), 1.0 + float(i) * 0.1]) for i in range(samples_per_class)],
        dim=0,
    )
    X = torch.cat([class0, class1], dim=0)
    Y = torch.cat(
        [
            torch.zeros(samples_per_class, dtype=torch.long),
            torch.ones(samples_per_class, dtype=torch.long),
        ],
        dim=0,
    )
    return X, Y


def test_generate_prototype_sets_excludes_boundary_from_inliers():
    X, Y = _make_two_class_dataset()

    prototypes, indices = generate_prototype_sets(
        X,
        Y,
        (0, 1),
        n_prototype=3,
        n_boundary=2,
        n_inlier=3,
        return_indices=True,
    )

    assert set(prototypes) == {"boundary", "inliers", "x_outlier", "y_outlier"}
    assert set(indices) == {"boundary", "inliers"}

    boundary_idx = set(indices["boundary"].tolist())
    inlier_idx = set(indices["inliers"].tolist())
    assert boundary_idx.isdisjoint(inlier_idx)


def test_generate_prototype_sets_respects_requested_pool_sizes():
    X, Y = _make_two_class_dataset()

    prototypes, indices = generate_prototype_sets(
        X,
        Y,
        (0, 1),
        n_prototype=4,
        n_boundary=1,
        n_inlier=3,
        return_indices=True,
    )

    assert prototypes["boundary"][0].shape[0] == 2
    assert prototypes["inliers"][0].shape[0] == 6
    assert prototypes["x_outlier"][0].shape[0] == 6
    assert prototypes["y_outlier"][0].shape[0] == 6
    assert indices["boundary"].shape[0] == 2
    assert indices["inliers"].shape[0] == 6


def test_fork_script_uses_main_input_prototype_flags():
    repo_root = Path(__file__).resolve().parents[1]
    script = (repo_root / "training_scripts" / "fork_intervention_v2proto.slurm").read_text()

    assert "--input-prototype-source from:$PROTO" in script
    assert "--input-prototypes-mode train" in script
    assert "--input-boundary  25" in script
    assert "--input-inliers   25" in script
    assert "--input-x-outliers 25" in script
    assert "--input-y-outliers 25" in script
    assert "--train-input-prototypes" not in script
    assert "--train-input-boundary" not in script
    assert "--train-input-inliers" not in script
    assert "--train-input-x-outliers" not in script
    assert "--train-input-y-outliers" not in script
