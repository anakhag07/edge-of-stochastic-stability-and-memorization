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


def _parse_batch_value(raw):
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value == 'full':
            return 'full'
        try:
            return int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("--batch must be a positive integer or 'full'") from exc
    raise argparse.ArgumentTypeError("--batch must be a positive integer or 'full'")


def build_parser(extrapolation_factor: float) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Training script')

    # --- Config Loading ---
    parser.add_argument('--config', type=str, default=None, help='Path to a JSON config file. CLI flags override config values.')

    # --- Core Training Setup ---
    parser.add_argument("--seed", type=int, default=88881)
    parser.add_argument('--batch', type=_parse_batch_value, default=64, help="Input batch size for training, or 'full' for full-batch GD")
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--steps', type=int, default=10000, help='Number of steps to train. Either epochs or steps should be provided')
    parser.add_argument('--cpu', action='store_true', help='Force training to run on CPU even if CUDA is available')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')

    # --- LR Schedule Controls ---
    parser.add_argument('--lmax-drop', action='store_true', help='One-time LR drop once lambda_max exceeds threshold (2/initial_lr).')
    parser.add_argument('--lmax-drop-mult', type=float, default=0.5, help='Multiply LR by this factor on trigger. (0.5 = 50%% drop, 0.8 = 20%% drop)')
    parser.add_argument('--lmax-drop-target-lr', type=float, default=None, help='Optional floor: LR after drop is max(LR*mult, target).')
    parser.add_argument('--lr-drop-at-step', '--lr_drop_at_step', type=int, default=None, help='Apply a one-time scheduled LR change when global step reaches this value.')
    parser.add_argument('--lr-drop-to', '--lr_drop_to', type=float, default=None, help='Learning rate to set when --lr-drop-at-step is reached.')
    parser.add_argument('--stop-loss', '--stop_loss', type=float, default=None, help='Stop training if loss goes below this value')

    # --- Dataset and Model Selection ---
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'ce'], help='Loss function to use (mse or ce)')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use for training')
    parser.add_argument('--classes', type=int, nargs=2, default=[1, 9], help='Two class labels to use for training. Default is [1, 9], as being probably the most difficult classes to separate')
    parser.add_argument('--num-data', '--num_data', type=int, default=1024, help='Number of datapoints to train on')
    parser.add_argument('--model', type=str, default='mlp', help='Network architecture to use for training')
    parser.add_argument('--init-scale', '--init_scale', type=float, default=0.2, help='Initialization scale for network weights')
    parser.add_argument('--no-init', '--no_init', action='store_true', help='If set, do not initialize network weights')

    # --- Checkpoint Continuation ---
    parser.add_argument('--cont-run-id', '--cont_run_id', type=str, default=None, help='Wandb run ID to continue training from')
    parser.add_argument('--cont-step', '--cont_step', type=int, default=None, help='Step to continue training from (uses closest available checkpoint)')
    parser.add_argument('--checkpoint-every', '--checkpoint_every', type=int, default=None, help='Save checkpoint every N steps (default: auto-calculated based on total steps)')

    # --- Optimizer Selection ---
    parser.add_argument('--momentum', type=float, default=None, help='Momentum for SGD optimizer')
    parser.add_argument('--adam', action='store_true', help='If set, use Adam optimizer instead of SGD')
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--precond-lmax', action='store_true', help='Log Adam-preconditioned top Hessian eigenvalue (AEoS sharpness).')

    # --- Primary Measurements ---
    parser.add_argument('--lambdamax', '--lmax', action='store_true', help='If set, compute the lambda_max, aka FullBS')
    parser.add_argument('--batch-sharpness', '--batch-sharpness-step', '--bs', action='store_true', dest='batch_sharpness', help='If set, compute the batch sharpness: E[gHg/g²] with the expectation taken across mini-batches. Use --batch-sharpness-step for backward compatibility.')
    parser.add_argument('--step-sharpness', action='store_true', dest='step_sharpness', help='If set, compute the step sharpness: the current-mini-batch Rayleigh quotient g·Hg/g². Average across steps to recover the traditional batch sharpness.')
    parser.add_argument('--gni', action='store_true', help='If set, compute the Gradient-Noise Interaction quantity.')

    # --- Additional Measurements ---
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

    # --- Measurement Configuration ---
    parser.add_argument('--disable-cache-eigenvectors', '--disable_cache_eigenvectors', action='store_true', help='If set, disable eigenvector caching for warm starts to improve eigenvalue computation performance')
    parser.add_argument('--use-power-iteration', '--use_power_iteration', action='store_true', help='If set, use power iteration method instead of LOBPCG for eigenvalue computation')
    parser.add_argument('--num-eigenvalues', '--num_eigenvalues', '--k', type=int, default=1, help='Number of eigenvalues to compute when computing lambda_max (default: 1)')
    parser.add_argument('--results-rarely', '--results_rarely', action='store_true', help='If set, results will be recorded less frequently')
    parser.add_argument('--precise-plots', action='store_true', help='Enable more frequent measurements for precise plotting')
    parser.add_argument('--rare-measure', dest='rare_measure', action='store_true', help='Activate regime where expensive measurements are performed rarely')

    # --- Noise and SDE Controls ---
    parser.add_argument('--gd-noise', '--gd_noise', type=str, default=None, help='Do noisy GD, to simulate SGD. Supported noises: sgd, diag, iso, const')
    parser.add_argument('--noise-mag', '--noise_mag', type=float, default=None, help='The noise magnitude for the constant noise')
    parser.add_argument('--sde', action='store_true', help='Simulate the SDE dynamics (the one that correspond to the SGD). It integrates the SDE using the Euler-Maruyama method')
    parser.add_argument('--sde-h', '--sde_h', type=float, default=0.01, help='SDE *integration* time step size (default: 0.01)')
    parser.add_argument('--sde-eta', '--sde_eta', type=float, default=None, help='Learning rate for SDE (uses --lr if not specified)')
    parser.add_argument('--sde-seed', '--sde_seed', type=int, default=888, help='Random seed for SDE noise generation (default: 888)')

    # --- Quadratic / Projection Controls ---
    parser.add_argument('--quad-switch-step', '--quad_switch_step', type=int, default=None, help='Step at which to switch from true NN dynamics to quadratic Taylor approximation dynamics')
    parser.add_argument('--use-gauss-newton', '--use_gauss_newton', action='store_true', help='Use Gauss-Newton matrix instead of Hessian for quadratic approximation')
    parser.add_argument('--quad-switch-lr', '--quad_switch_lr', type=float, default=None, help='lr to use after switching, used to test explosion')
    parser.add_argument('--proj-switch-step', dest='proj_switch_step', type=int, default=None, help='Step number to start projecting minibatch gradient onto top-l Hessian eigendirections (full batch)')
    parser.add_argument('--proj-top-l', dest='proj_top_l', type=int, default=None, help='Number of top Hessian eigendirections to project against/onto after switch step')
    parser.add_argument('--proj-to-residual', dest='proj_to_residual', action='store_true', help='After --proj-switch-step, apply gradient projected to orthogonal complement of top-l eigenspace')

    # --- Determinism and Logging ---
    parser.add_argument('--dataset-seed', '--dataset_seed', type=int, default=888, help='Random seed for dataset preparation')
    parser.add_argument('--init-seed', '--init_seed', type=int, default=8888, help='Random seed for network initialization')
    parser.add_argument('--wandb-tag', type=str, default=None, help='Tag to add to the wandb run')
    parser.add_argument('--wandb-name', type=str, default=None, help='Optional suffix appended to default wandb run name (sanitized)')
    parser.add_argument('--wandb-notes', type=str, default=None, help='Optional notes/description attached to the wandb run')
    parser.add_argument('--disable-wandb', action='store_true', help='Disable Weights & Biases logging for debugging/testing')
    parser.add_argument('--train-test-gap', action='store_true', help='If set, compute the training and testing accuracy and gap (heavy, runs rarely)')

    # --- Input-space Prototypes ---
    parser.add_argument('--input-prototypes-mode', type=str, default='train', choices=['train', 'val'], help='Input prototype mode: train includes the selected prototype subsets in training; val excludes real-data subsets from training and tracks them as held-out evaluation sets.')
    parser.add_argument('--input-prototype-source', type=str, default=None, help='Input prototype source: generate | from:<path or run> | none')
    parser.add_argument('--input-x-outliers', type=int, default=None, help='Number of input-space x-outlier prototypes per class to include/track')
    parser.add_argument('--input-y-outliers', type=int, default=None, help='Number of input-space y-outlier prototypes per class to include/track')
    parser.add_argument('--input-inliers', type=int, default=None, help='Number of input-space inlier prototypes per class to include/track')
    parser.add_argument('--input-boundary', type=int, default=None, help='Number of input-space boundary prototypes per class to include/track')
    return parser
