import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


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
        raise ValueError(f"Unknown config key(s) in {path}: {', '.join(unknown_keys)}")
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


def build_parser(extrapolation_factor: float) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--config', type=str, default=None, help='Path to a JSON config file. CLI flags override config values.')
    parser.add_argument("--seed", type=int, default=88881)
    parser.add_argument('--batch', type=int, default=64, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--steps', type=int, default=10000, help='Number of steps to train. Either epochs or steps should be provided')
    parser.add_argument('--cpu', action='store_true', help='Force training to run on CPU even if CUDA is available')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--lmax-decay', action='store_true', help='Enable linear lr decay once lambda_max >= 2/initial_lr')
    parser.add_argument('--lmax-decay-target-lr', type=float, default=0.0001, help='Target lr for linear decay after lmax trigger')
    parser.add_argument('--lmax-decay-steps', type=int, default=10000, help='Number of steps to linearly decay to target lr')
    parser.add_argument('--lmax-drop', action='store_true', help='One-time LR drop once lambda_max exceeds threshold (2/initial_lr).')
    parser.add_argument('--lmax-drop-mult', type=float, default=0.5, help='Multiply LR by this factor on trigger. (0.5 = 50%% drop, 0.8 = 20%% drop)')
    parser.add_argument('--lmax-drop-target-lr', type=float, default=None, help='Optional floor: LR after drop is max(LR*mult, target).')
    parser.add_argument('--stop-loss', '--stop_loss', type=float, default=None, help='Stop training if loss goes below this value')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'ce'], help='Loss function to use (mse or ce)')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use for training')
    parser.add_argument('--classes', type=int, nargs=2, default=[1, 9], help='Two class labels to use for training. Default is [1, 9], as being probably the most difficult classes to separate')
    parser.add_argument('--num-data', '--num_data', type=int, default=1024, help='Number of datapoints to train on')
    parser.add_argument('--model', type=str, default='mlp', help='Network architecture to use for training')
    parser.add_argument('--init-scale', '--init_scale', type=float, default=0.2, help='Initialization scale for network weights')
    parser.add_argument('--no-init', '--no_init', action='store_true', help='If set, do not initialize network weights')
    parser.add_argument('--cont-run-id', '--cont_run_id', type=str, default=None, help='Wandb run ID to continue training from')
    parser.add_argument('--cont-step', '--cont_step', type=int, default=None, help='Step to continue training from (uses closest available checkpoint)')
    parser.add_argument('--checkpoint-every', '--checkpoint_every', type=int, default=None, help='Save checkpoint every N steps (default: auto-calculated based on total steps)')
    parser.add_argument('--momentum', type=float, default=None, help='Momentum for SGD optimizer')
    parser.add_argument('--adam', action='store_true', help='If set, use Adam optimizer instead of SGD')
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--precond-lmax', action='store_true', help='Log Adam-preconditioned top Hessian eigenvalue (AEoS sharpness).')
    parser.add_argument('--lambdamax', '--lmax', action='store_true', help='If set, compute the lambda_max, aka FullBS')
    parser.add_argument('--batch-sharpness', '--batch-sharpness-step', '--bs', action='store_true', dest='batch_sharpness', help='If set, compute the batch sharpness: E[gHg/g²] with the expectation taken across mini-batches. Use --batch-sharpness-step for backward compatibility.')
    parser.add_argument('--step-sharpness', action='store_true', dest='step_sharpness', help='If set, compute the step sharpness: the current-mini-batch Rayleigh quotient g·Hg/g². Average across steps to recover the traditional batch sharpness.')
    parser.add_argument('--gni', action='store_true', help='If set, compute the Gradient-Noise Interaction quantity.')
    parser.add_argument('--hessian-trace', action='store_true', help='Estimate the trace of the full-batch loss Hessian via a Hutchinson-style estimator')
    parser.add_argument('--grad-projection', action='store_true', help='Compute grad_projection_i: fraction of full-batch gradient lying in span of top-i cached Hessian eigenvectors (i up to 20); uses cached eigenvectors only; only for plain SGD')
    parser.add_argument('--one-step-loss-change', action='store_true', help='If set, compute the expected one-step change in loss using Monte Carlo estimation')
    parser.add_argument('--gradient-norm', action='store_true', help='If set, compute the Monte Carlo estimate of squared norm of mini-batch gradients')
    parser.add_argument('--final', action='store_true', help='If set, compute the lambda_max and step sharpness at the end')
    parser.add_argument('--batch-sharpness-exp-inside', action='store_true', help='If set, compute the batch sharpness using E[gHg]/E[g²], where the expectation is inside the ratio. Compare with step-sharpness, where the expectation stays outside the ratio.')
    parser.add_argument('--batch-lambdamax', '--batchlmax', action='store_true', help='If set, compute the batch lambda_max(H_B), aka batch lambda max')
    parser.add_argument('--fisher', action='store_true', help='If set, compute Fisher information matrix eigenvalue. Currently only works with one-dim output')
    parser.add_argument('--param-distance', '--param_distance', action='store_true', help='If set, compute the distance from the reference weights')
    parser.add_argument('--param-file', '--param_file', type=str, default=None, help='Path to reference parameters for computing parameter distance')
    parser.add_argument('--log-every-step', action='store_true', help='Force all configured measurements to log every training step, bypassing frequency rules.')
    parser.add_argument('--disable-cache-eigenvectors', '--disable_cache_eigenvectors', action='store_true', help='If set, disable eigenvector caching for warm starts to improve eigenvalue computation performance')
    parser.add_argument('--use-power-iteration', '--use_power_iteration', action='store_true', help='If set, use power iteration method instead of LOBPCG for eigenvalue computation')
    parser.add_argument('--num-eigenvalues', '--num_eigenvalues', '--k', type=int, default=1, help='Number of eigenvalues to compute when computing lambda_max (default: 1)')
    parser.add_argument('--results-rarely', '--results_rarely', action='store_true', help='If set, results will be recorded less frequently')
    parser.add_argument('--precise-plots', action='store_true', help='Enable more frequent measurements for precise plotting')
    parser.add_argument('--rare-measure', dest='rare_measure', action='store_true', help='Activate regime where expensive measurements are performed rarely')
    parser.add_argument('--gd-noise', '--gd_noise', type=str, default=None, help='Do noisy GD, to simulate SGD. Supported noises: sgd, diag, iso, const')
    parser.add_argument('--noise-mag', '--noise_mag', type=float, default=None, help='The noise magnitude for the constant noise')
    parser.add_argument('--sde', action='store_true', help='Simulate the SDE dynamics (the one that correspond to the SGD). It integrates the SDE using the Euler-Maruyama method')
    parser.add_argument('--sde-h', '--sde_h', type=float, default=0.01, help='SDE *integration* time step size (default: 0.01)')
    parser.add_argument('--sde-eta', '--sde_eta', type=float, default=None, help='Learning rate for SDE (uses --lr if not specified)')
    parser.add_argument('--sde-seed', '--sde_seed', type=int, default=888, help='Random seed for SDE noise generation (default: 888)')
    parser.add_argument('--quad-switch-step', '--quad_switch_step', type=int, default=None, help='Step at which to switch from true NN dynamics to quadratic Taylor approximation dynamics')
    parser.add_argument('--use-gauss-newton', '--use_gauss_newton', action='store_true', help='Use Gauss-Newton matrix instead of Hessian for quadratic approximation')
    parser.add_argument('--quad-switch-lr', '--quad_switch_lr', type=float, default=None, help='lr to use after switching, used to test explosion')
    parser.add_argument('--proj-switch-step', dest='proj_switch_step', type=int, default=None, help='Step number to start projecting minibatch gradient onto top-l Hessian eigendirections (full batch)')
    parser.add_argument('--proj-top-l', dest='proj_top_l', type=int, default=None, help='Number of top Hessian eigendirections to project against/onto after switch step')
    parser.add_argument('--proj-to-residual', dest='proj_to_residual', action='store_true', help='After --proj-switch-step, apply gradient projected to orthogonal complement of top-l eigenspace')
    parser.add_argument('--dataset-seed', '--dataset_seed', type=int, default=888, help='Random seed for dataset preparation')
    parser.add_argument('--init-seed', '--init_seed', type=int, default=8888, help='Random seed for network initialization')
    parser.add_argument('--wandb-tag', type=str, default=None, help='Tag to add to the wandb run')
    parser.add_argument('--wandb-name', type=str, default=None, help='Optional suffix appended to default wandb run name (sanitized)')
    parser.add_argument('--wandb-notes', type=str, default=None, help='Optional notes/description attached to the wandb run')
    parser.add_argument('--disable-wandb', action='store_true', help='Disable Weights & Biases logging for debugging/testing')
    parser.add_argument('--train-test-gap', action='store_true', help='If set, compute the training and testing accuracy and gap (heavy, runs rarely)')
    parser.add_argument('--per-sample', action='store_true', help='Track per-sample loss/residual/curvature histograms over time and save frames')
    parser.add_argument('--per-sample-every', type=int, default=100, help='Snapshot cadence in steps for per-sample histograms (default: 100)')
    parser.add_argument('--hist-min-log10', type=float, default=-6.0, help='Left edge for log10 binning (default: -6)')
    parser.add_argument('--hist-max-log10', type=float, default=2.0, help='Right edge for log10 binning (default: 2)')
    parser.add_argument('--hist-bins', type=int, default=80, help='Number of bins for log10 histograms (default: 80)')
    parser.add_argument('--per-sample-metrics', type=str, nargs='+', default=['loss', 'resid', 'kappa'], choices=['loss', 'resid', 'kappa'], help='Which metrics to histogram (default: loss resid kappa)')
    parser.add_argument('--no-frames', action='store_true', help='Only save counts/quantiles as .npz; do not render PNG frames')
    parser.add_argument('--memorization-hessian-outliers', action='store_true', help='Compute memorization stats based on alignment with top Hessian eigenvector (heavy; runs rarely)')
    parser.add_argument('--memorization-outlier-frac', type=float, default=0.05, help='Fraction of examples treated as outliers for Hessian-alignment memorization stats')
    parser.add_argument('--knn-outliers', action='store_true', help='After training, run a k-NN pass to flag ambiguous samples')
    parser.add_argument('--knn-neighbors', type=int, default=32, help='Number of neighbors used for the ambiguity score')
    parser.add_argument('--knn-top-per-class', type=int, default=10, help='How many outliers to keep per class')
    parser.add_argument('--knn-feature-batch', type=int, default=512, help='Batch size used during the post-hoc embedding pass')
    parser.add_argument('--knn-chunk-size', type=int, default=1024, help='Chunk size used while computing the k-NN graph')
    parser.add_argument('--knn-no-normalize', action='store_true', help='Disable L2-normalization of feature vectors before k-NN')
    parser.add_argument('--track-knn-outliers-from', type=str, default=None, help='Existing plaintext run folder name (e.g., 20251124_0820_35_lr0.01000_b8) whose knn_outlier_indices.json should be tracked during training')
    parser.add_argument('--track-knn-topk', type=int, default=5, help='Number of stored outliers per class to track from the reference run')
    parser.add_argument('--feature-prototypes', action='store_true', help='After training, export feature-space prototype sets for reuse')
    parser.add_argument('--feature-prototype-batch', type=int, default=512, help='Batch size for feature extraction when exporting feature prototypes')
    parser.add_argument('--feature-prototype-topk', type=int, default=50, help='Maximum prototypes per class to store from feature space')
    parser.add_argument('--feature-prototype-kneighbors', type=int, default=32, help='k-NN neighborhood size when identifying boundary points in feature space')
    parser.add_argument('--feature-prototype-no-normalize', action='store_true', help='Disable L2 normalization before computing feature-space prototypes')
    parser.add_argument('--feature-prototype-extrapolation', type=float, default=extrapolation_factor, help='Extrapolation factor used when building feature-space x-outliers')
    parser.add_argument('--track-feature-prototypes-from', type=str, default=None, help='Existing plaintext run folder whose feature-space prototype sets should be tracked during training')
    parser.add_argument('--input-prototypes-mode', type=str, default=None, choices=['train', 'val'], help='Unified input-prototype mode: train (include all prototypes in training) or val (hold out subsets and log metrics on held-out sets). When unset, legacy behavior applies.')
    parser.add_argument('--train-input-prototypes', type=str, default=None, help='Input prototype source for training run: generate | from:<path or run> | none')
    parser.add_argument('--test-input-prototypes', type=str, default=None, help='Input prototype source for logging/eval: from:<path or run> | none (defaults to train-input-prototypes when unset)')
    parser.add_argument('--input-prototypes-frac', type=float, default=None, help='Fraction of training set (per class) to use for input prototypes when generating')
    parser.add_argument('--input-prototypes-count', type=int, default=None, help='Count per class to use for all input prototype subsets when generating')
    parser.add_argument('--input-prototypes-boundary-count', type=int, default=None, help='Override count per class for boundary prototypes')
    parser.add_argument('--input-prototypes-inliers-count', type=int, default=None, help='Override count per class for inlier prototypes')
    parser.add_argument('--input-prototypes-x-outlier-count', type=int, default=None, help='Override count per class for x-outlier prototypes')
    parser.add_argument('--input-prototypes-y-outlier-count', type=int, default=None, help='Override count per class for y-outlier prototypes')
    parser.add_argument('--input-prototypes-holdout-boundary-count', type=int, default=None, help='Holdout count per class for boundary prototypes in validation mode')
    parser.add_argument('--input-prototypes-holdout-inliers-count', type=int, default=None, help='Holdout count per class for inlier prototypes in validation mode')
    parser.add_argument('--input-prototypes-holdout-x-outlier-count', type=int, default=None, help='Holdout count per class for x-outlier prototypes in validation mode')
    parser.add_argument('--input-prototypes-holdout-y-outlier-count', type=int, default=None, help='Holdout count per class for y-outlier prototypes in validation mode')
    parser.add_argument('--input-prototypes-holdout', type=str, default='auto', choices=['auto', 'boundary_inliers', 'none'], help='[DEPRECATED] Hold out boundary/inlier indices from training when using train-input-prototypes (auto => boundary_inliers for legacy flags)')
    parser.add_argument('--track-input-prototypes', action='store_true', help='[DEPRECATED] Track/log input-space prototype subsets (boundary/inliers/synthetic outliers) on wandb')
    parser.add_argument('--input-prototype-holdout-per-class', type=int, default=None, help='[DEPRECATED] Hold out this many boundary/inlier points per class from training')
    parser.add_argument('--input-prototype-holdout-frac', type=float, default=None, help='[DEPRECATED] Hold out this fraction of the training set per class (used to size boundary/inlier prototypes)')
    parser.add_argument('--train-input-x-outliers', type=int, default=None, help='Augment the training set with this many input-space x-outliers per class; also logs input-space prototype subsets')
    parser.add_argument('--train-input-y-outliers', type=int, default=None, help='Augment the training set with this many input-space y-outliers per class; also logs input-space prototype subsets')
    parser.add_argument('--train-input-inliers', type=int, default=None, help='Augment the training set with this many input inliers per class; also logs input-space prototype subsets')
    parser.add_argument('--train-input-boundary', type=int, default=None, help='Augment the training set with this many input boundary points per class; also logs input-space prototype subsets')
    return parser
