import math

import pytest
import torch
import torch.nn as nn

try:
    from utils.measure import compute_subset_metrics
    _EINOPS_AVAILABLE = True
except ModuleNotFoundError as exc:
    if exc.name == "einops":
        compute_subset_metrics = None
        _EINOPS_AVAILABLE = False
    else:
        raise


@pytest.mark.skipif(not _EINOPS_AVAILABLE, reason="einops is required for measure utilities")
def test_compute_subset_metrics_reports_batch_sharpness():
    assert compute_subset_metrics is not None
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Linear(3, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )
    loss_fn = nn.MSELoss(reduction="mean")

    X = torch.randn(12, 3)
    Y = torch.randn(12)
    indices = torch.arange(6)

    metrics = ["batch_sharpness"]
    metric_kwargs = {
        "batch_sharpness": {
            "batch_size": 4,
            "n_estimates": 1,
            "min_estimates": 1,
            "eps": 1.0,
        }
    }

    results = compute_subset_metrics(
        net=net,
        loss_fn=loss_fn,
        X=X,
        Y=Y,
        indices=indices,
        metrics=metrics,
        metric_kwargs=metric_kwargs,
    )

    assert "batch_sharpness" in results
    assert isinstance(results["batch_sharpness"], float)
    assert math.isfinite(results["batch_sharpness"])
