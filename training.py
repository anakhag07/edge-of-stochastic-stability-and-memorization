import torch as T
import torch
import torch.nn as nn
import os
from einops import rearrange, repeat
from torch import linalg as LA
import numpy as np
from copy import deepcopy
from pathlib import Path
import math
import random
import argparse

import time

from utils.data import prepare_dataset, get_dataset_presets
from utils.nets import SquaredLoss, MLP, CNN, prepare_net, initialize_net, prepare_optimizer, get_model_presets
from utils.nets import ResNet
from utils.storage import initialize_folders
from utils.wandb_utils import (
    init_wandb,
    log_metrics,
    save_checkpoint_wandb,
    find_closest_checkpoint_wandb,
    load_checkpoint_wandb,
    get_checkpoint_dir_for_run,
    is_wandb_available,
    generate_run_id,
)

from utils.noise import gd_with_noise, GradStorage, sde_integration
from utils.measure import *

from torch.autograd import grad
import json


if 'DATASETS' not in os.environ:
    raise ValueError("Please set the environment variable 'DATASETS'. Use 'export DATASETS=/path/to/datasets'")
if 'RESULTS' not in os.environ:
    raise ValueError("Please set the environment variable 'RESULTS'. Use 'export RESULTS=/path/to/results'")

DATASET_FOLDER = Path(os.environ.get('DATASETS'))
RES_FOLDER = Path(os.environ.get('RESULTS'))


# -------------------------------------
# NEW: Per-sample histograms setup
# -------------------------------------
class PerSampleHistograms:
    def __init__(self, min_log10, max_log10, bins, metrics):
        self.min_log10 = min_log10
        self.max_log10 = max_log10
        self.bins = bins
        self.metrics = metrics
        self.histograms = {metric: np.zeros((bins,)) for metric in metrics}
        self.counts = {metric: 0 for metric in metrics}
        self.quantiles = {metric: np.zeros((bins,)) for metric in metrics}

@torch.no_grad()
def _per_sample_stats(net, loss_fn, X, Y, loss_type='ce', batch_size=1024, device='cuda'):
    """Return dict of numpy arrays: loss, resid_norm, kappa for dataset (X,Y)."""
    net.eval()
    out_loss, out_resid, out_kappa = [], [], []
    for i in range(0, len(X), batch_size):
        xb = X[i:i+batch_size].to(device)
        yb = Y[i:i+batch_size].to(device)
        z = net(xb)  # logits for CE; prediction for MSE
        if loss_type == 'ce':
            # losses per-sample
            loss = torch.nn.functional.cross_entropy(z, yb, reduction='none')
            # residual wrt logits: p - y_onehot
            p = torch.softmax(z, dim=1)
            y1 = torch.nn.functional.one_hot(yb, num_classes=z.size(1)).float()
            resid = p - y1                      # dL/dz
            resid_norm = resid.norm(dim=1)
            # curvature proxy: Frobenius norm of softmax Hessian
            # ||diag(p) - p p^T||_F
            I = torch.eye(p.size(1), device=p.device).unsqueeze(0)   # [1, C, C]
            Hout = I * p.unsqueeze(2) - p.unsqueeze(2) * p.unsqueeze(1)
            kappa = torch.linalg.norm(Hout, dim=(1,2))
        else:
            # MSE (your SquaredLoss is 0.5*||y - yhat||^2); match that here per-sample
            if z.ndim == 1 or z.size(-1) == 1:
                z = z.squeeze(-1)
            # If Y is class index for 2-class, convert upstream; here assume Y already numeric
            diff = (z - yb)
            loss = 0.5*(diff**2)
            if loss.ndim > 1:
                loss = loss.sum(dim=1)
            resid_norm = diff if diff.ndim==1 else diff.norm(dim=1)
            kappa = torch.ones_like(loss)  # output-space curvature is constant for MSE
        out_loss.append(loss.detach().cpu())
        out_resid.append(resid_norm.detach().cpu())
        out_kappa.append(kappa.detach().cpu())
    net.train()
    return {
        'loss': torch.cat(out_loss).numpy(),
        'resid': torch.cat(out_resid).numpy(),
        'kappa': torch.cat(out_kappa).numpy(),
    }

def _hist_log10(values, bin_edges):
    v = np.asarray(values)
    v = np.clip(v, a_min=np.finfo(float).tiny, a_max=None)  # avoid log of 0
    lv = np.log10(v)
    counts, _ = np.histogram(lv, bins=bin_edges)
    return counts

def _quantiles(values, qs=(0.1,0.5,0.9,0.99)):
    v = np.asarray(values)
    return np.quantile(v, qs)

def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def _render_frame(bin_edges, counts_train, counts_test, title, out_png_path):
    plt.figure(figsize=(6,4))
    centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    plt.step(centers, counts_train, where='mid', label='train', alpha=0.8)
    plt.step(centers, counts_test, where='mid', label='test', alpha=0.6)
    plt.yscale('log') # log applied
    plt.xlabel('log10(value)')
    plt.ylabel('log10(count)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=140)
    plt.close()

# -------------------------------------
# Section: Measurement Runner
# -------------------------------------

class MeasurementRunner:
    """Centralized measurement orchestration for the training loop."""
    _FIRST_FEW_FULL_LOSS_STEPS = 32

    def __init__(
        self,
        *,
        net,
        loss_fn,
        full_inputs,
        test_inputs = None,
        measurements,
        device,
        batch_size,
        save_dir,
        eigenvector_cache,
        num_eigenvalues,
        use_power_iteration,
        precise_plots,
        rare_measure,
        param_reference,
        step_to_start,
        sde_enabled,
        gd_noise,
        proj_switch_step,
        quad_approx,
        memorization_outlier_frac: float,
        # NEW:
        full_inputs_test=None,
        per_sample_cfg=None,
    ):
        self.net = net
        self.loss_fn = loss_fn
        self.X, self.Y = full_inputs

        # New: Test inputs handling
        if test_inputs is not None:
            self.X_test, self.Y_test = test_inputs
        else:
            self.X_test, self.Y_test = None, None

        self.measurements = measurements
        self.device = device
        self.batch_size = batch_size
        self.eigenvector_cache = eigenvector_cache
        self.num_eigenvalues = num_eigenvalues
        self.use_power_iteration = use_power_iteration
        self.precise_plots = precise_plots
        self.rare_measure = rare_measure
        self.param_reference = param_reference
        self.step_to_start = step_to_start
        self.sde_enabled = sde_enabled
        self.gd_noise = gd_noise
        self.proj_switch_step = proj_switch_step
        self.quad_approx = quad_approx
        self.memorization_outlier_frac = memorization_outlier_frac
        
        # NEW: Per-sample config
        self.full_inputs_test = full_inputs_test
        self.per_sample_cfg = per_sample_cfg
        
        # NEW: Per-sample initialization logic
        if per_sample_cfg is not None and per_sample_cfg.get('enabled', False):
            self.per_sample_histograms = PerSampleHistograms(
                min_log10=per_sample_cfg['hist_min_log10'],
                max_log10=per_sample_cfg['hist_max_log10'],
                bins=per_sample_cfg['hist_bins'],
                metrics=per_sample_cfg.get('metrics', per_sample_cfg.get('hist_metrics', ['loss'])))
            # Initialize bin edges for log10 histogram
            self.bin_edges = np.linspace(
                per_sample_cfg['hist_min_log10'],
                per_sample_cfg['hist_max_log10'],
                per_sample_cfg['hist_bins'] + 1
            )
            # Initialize directories for saving
            self.ps_dir = save_dir / 'per_sample_histograms'
            self.frames_dir = self.ps_dir / 'frames'
            _ensure_dir(self.ps_dir)
            if not per_sample_cfg.get('no_frames', False):
                _ensure_dir(self.frames_dir)
        else:
            self.per_sample_histograms = None
            self.bin_edges = None
            self.ps_dir = None       
        # NEW: Per-sample config
        self.full_inputs_test = full_inputs_test
        self.per_sample_cfg = per_sample_cfg
        
        # NEW: Per-sample initialization logic
        if per_sample_cfg is not None and per_sample_cfg.get('enabled', False):
            self.per_sample_histograms = PerSampleHistograms(
                min_log10=per_sample_cfg['hist_min_log10'],
                max_log10=per_sample_cfg['hist_max_log10'],
                bins=per_sample_cfg['hist_bins'],
                metrics=per_sample_cfg.get('metrics', per_sample_cfg.get('hist_metrics', ['loss'])))
            # Initialize bin edges for log10 histogram
            self.bin_edges = np.linspace(
                per_sample_cfg['hist_min_log10'],
                per_sample_cfg['hist_max_log10'],
                per_sample_cfg['hist_bins'] + 1
            )
            # Initialize directories for saving
            self.ps_dir = save_dir / 'per_sample_histograms'
            self.frames_dir = self.ps_dir / 'frames'
            _ensure_dir(self.ps_dir)
            if not per_sample_cfg.get('no_frames', False):
                _ensure_dir(self.frames_dir)
        else:
            self.per_sample_histograms = None
            self.bin_edges = None
            self.ps_dir = None
            self.frames_dir = None

        self.eigenvalues_log = []
        if 'lmax' in measurements and num_eigenvalues > 1:
            eigenvalues_path = save_dir / 'eigenvalues.json'
            self.eigenvalues_file = open(eigenvalues_path, 'w')
            self.eigenvalues_file.write('[\n')
        else:
            self.eigenvalues_file = None

    def close(self):
        if self.eigenvalues_file is not None:
            self.eigenvalues_file.write('\n]')
            self.eigenvalues_file.close()

    def collect(
        self,
        *,
        ctx,
        optimizer,
        X_batch,
        Y_batch,
        epoch,
        step_in_epoch,
        step_number,
    ):
        metrics = {
            'batch_lmax': np.nan,
            'step_sharpness': np.nan,
            'batch_sharpness': np.nan,
            'batch_sharpness_exp_inside': np.nan,
            'fisher_batch_eigenval': np.nan,
            'fisher_total_eigenval': np.nan,
            'full_accuracy': np.nan,
            'full_gHg': np.nan,
            'full_loss': np.nan,
            'gni': np.nan,
            'grad_projections': None,
            'one_step_loss_change': np.nan,
            'param_distance': np.nan,
            'gradient_norm_squared': np.nan,
            'lmax': np.nan,
            'all_eigenvalues': None,
            'quadratic_loss': None,
            'quadratic_loss_gn': None,
            'proj_grad_ratio': None,
            'hessian_trace': np.nan,
            'memorization_hessian_outliers': None,
            'train_acc': np.nan, # NEW
            'test_acc': np.nan,  # NEW
            'train_test_gap': np.nan, # NEW
        }

        epoch_loss_update = None


        # ----- Batch sharpness (expected Rayleigh quotient) -----
        if 'batch_sharpness' in self.measurements:
            if frequency_calculator.should_measure('batch_sharpness', ctx):
                metrics['batch_sharpness'] = calculate_averaged_grad_H_grad_step(
                    self.net,
                    self.X,
                    self.Y,
                    self.loss_fn,
                    batch_size=self.batch_size,
                    n_estimates=1000,
                    min_estimates=20,
                    eps=0.005,
                )
        # ----- Instantaneous step sharpness (current-batch Rayleigh quotient) -----
        if 'step_sharpness' in self.measurements:
            if frequency_calculator.should_measure('step_sharpness', ctx):
                self.net.zero_grad()
                preds = self.net(X_batch).squeeze(dim=-1)
                loss = self.loss_fn(preds, Y_batch)
                metrics['step_sharpness'] = compute_grad_H_grad(loss, self.net).item()

        # ----- Eigenvalues/Lambda max (full batch) -----
        lmax_now = False
        if 'lmax' in self.measurements:
            measurement_type = 'full_batch_lambda_max'
            lmax_now = frequency_calculator.should_measure(measurement_type, ctx)

        if lmax_now:
            if str(self.device).startswith('cuda'):
                torch.cuda.empty_cache()
            optimizer.zero_grad()

            lmax_max_size = 4096
            if str(self.device).startswith('cuda'):
                total_memory = torch.cuda.get_device_properties(0).total_memory
                if total_memory < 20 * 1024**3:
                    if isinstance(self.net, CNN):
                        lmax_max_size = 2048 + 512
                    if isinstance(self.net, ResNet):
                        lmax_max_size = 512

            if len(self.X) > lmax_max_size:
                print(f"Warning: Computing eigenvalues on subset of {lmax_max_size} samples instead of full dataset ({len(self.X)} samples) due to memory/time constraints. Most of the time it is fine, but should be corrected")
                idx = gimme_random_subset_idx(len(self.X), lmax_max_size)
                X_subset = self.X[idx]
                Y_subset = self.Y[idx]
            else:
                X_subset = self.X
                Y_subset = self.Y

            preds = self.net(X_subset).squeeze(dim=-1)
            loss = self.loss_fn(preds, Y_subset)

            if self.eigenvector_cache is not None:
                max_iterations = 100 if not self.use_power_iteration else 1000
                tolerance = 0.005 if self.num_eigenvalues < 6 else 0.03
                if self.precise_plots:
                    max_iterations = 300 if not self.use_power_iteration else 3000
                    tolerance = 0.001 if self.num_eigenvalues < 6 else 0.01

                eigenvalues, eigenvectors = compute_eigenvalues(
                    loss,
                    self.net,
                    k=self.num_eigenvalues,
                    max_iterations=max_iterations,
                    reltol=tolerance,
                    eigenvector_cache=self.eigenvector_cache,
                    return_eigenvectors=True,
                    use_power_iteration=self.use_power_iteration,
                )

                if self.num_eigenvalues == 1:
                    self.eigenvector_cache.store_eigenvector(eigenvectors, eigenvalues.item())
                    lmax_value = eigenvalues
                else:
                    self.eigenvector_cache.store_eigenvectors(
                        [eigenvectors[:, i] for i in range(eigenvectors.shape[1])],
                        eigenvalues.tolist(),
                    )
                    lmax_value = eigenvalues[0]
            else:
                eigenvalues = compute_eigenvalues(
                    loss,
                    self.net,
                    k=self.num_eigenvalues,
                    max_iterations=200,
                    reltol=0.03,
                    use_power_iteration=self.use_power_iteration,
                )
                if self.num_eigenvalues == 1:
                    lmax_value = eigenvalues
                else:
                    lmax_value = eigenvalues[0]

            if self.num_eigenvalues > 1:
                metrics['all_eigenvalues'] = eigenvalues
                if self.eigenvalues_file is not None:
                    eigenvalues_data = {
                        'epoch': epoch,
                        'step': step_number,
                        'eigenvalues': eigenvalues.tolist()
                        if isinstance(eigenvalues, torch.Tensor)
                        else [eigenvalues],
                    }
                    self.eigenvalues_log.append(eigenvalues_data)
                    if len(self.eigenvalues_log) > 1:
                        self.eigenvalues_file.write(',\n')
                    json.dump(eigenvalues_data, self.eigenvalues_file)
                    self.eigenvalues_file.flush()

            metrics['lmax'] = lmax_value.item()
            metrics['full_loss'] = loss.item()
            metrics['full_accuracy'] = calculate_accuracy(preds, Y_subset)

            epoch_loss_update = metrics['full_loss']

            print(
                f"Epoch {epoch + 1}, Step {step_in_epoch}: Total lambda max = {metrics['lmax']}, "
                f"Loss = {metrics['full_loss']} !!!"
            )

        

        if 'hessian_trace' in self.measurements:
            if frequency_calculator.should_measure('hessian_trace', ctx):
                metrics['hessian_trace'] = estimate_hessian_trace(
                    self.net,
                    self.X,
                    self.Y,
                    self.loss_fn,
                    max_estimates=256,
                    min_estimates=20,
                    eps=0.01,
                )

        # ----- Memorization via Hessian outliers -----
        if 'memorization_hessian_outliers' in self.measurements:
            if frequency_calculator.should_measure('memorization_hessian_outliers', ctx):
                optimizer.zero_grad()
                mem_stats = compute_outlier_vs_bulk_stats_hessian(
                    net=self.net,
                    X_train=self.X,
                    Y_train=self.Y,
                    loss_fn=self.loss_fn,
                    optimizer=optimizer,
                    frac=self.memorization_outlier_frac,
                )
                if mem_stats:
                    metrics.update({f"memorization_hessian_outliers/{k}": v for k, v in mem_stats.items()})
        
        # ----- NEW: Train/Test Gap -----
        if 'train_test_gap' in self.measurements and self.X_test is not None:
            if frequency_calculator.should_measure('train_test_gap',ctx):
                vals = compute_train_test_gap_from_tensors(
                    self.net, self.X, self.Y, self.X_test, self.Y_test
                )
                metrics['train_acc'] = vals['train_acc']
                metrics['test_acc'] = vals['test_acc']
                metrics['train_test_gap'] = vals['gap']

        # ----- NEW: Per-sample histograms -----
        if self.per_sample_cfg and self.per_sample_cfg['enabled']:
            every = self.per_sample_cfg['every']
            if step_number % every == 0:
                loss_type = 'ce' if isinstance(self.loss_fn, nn.CrossEntropyLoss) else 'mse'
                # train
                stats_tr = _per_sample_stats(self.net, self.loss_fn, self.X, self.Y, loss_type=loss_type, batch_size=self.batch_size, device=self.device)
                # test (if present)
                if (self.X_test is not None) and (len(self.X_test) > 0):
                    stats_te = _per_sample_stats(self.net, self.loss_fn, self.X_test, self.Y_test, loss_type=loss_type, batch_size=self.batch_size, device=self.device)
                else:
                    stats_te = {k: np.zeros_like(v) for k, v in stats_tr.items()}

                # collect metrics
                for metric in self.per_sample_histograms.metrics:
                    # Quantiles (log to wandb)
                    quantiles_tr = _quantiles(stats_tr[metric])
                    quantiles_te = _quantiles(stats_te[metric])
                    
                    for q, name in zip((0.1, 0.5, 0.9, 0.99), ('q10','q50','q90','q99')):
                        metrics[f'ps_quantiles/{metric}_train_{name}'] = np.quantile(stats_tr[metric], q)
                        metrics[f'ps_quantiles/{metric}_test_{name}'] = np.quantile(stats_te[metric], q)

                    # Histograms (save to file)
                    counts_tr = _hist_log10(stats_tr[metric], self.bin_edges)
                    counts_te = _hist_log10(stats_te[metric], self.bin_edges)

                    # Store for saving to file later
                    np.savez(
                        self.ps_dir / f'step_{step_number:05d}_{metric}.npz',
                        bin_edges=self.bin_edges,
                        counts_train=counts_tr,
                        counts_test=counts_te,
                        quantiles_train=quantiles_tr,
                        quantiles_test=quantiles_te
                    )

                    # Render frame
                    if not self.per_sample_cfg.get('no_frames', False):
                        title = f'Step {step_number}: log10({metric})'
                        out_path = self.frames_dir / f'{step_number:05d}_{metric}.png'
                        _render_frame(self.bin_edges, counts_tr, counts_te, title, out_path)



        # ----- Gradient-noise interaction (GNI) -----
        gni_now = False
        if 'gni' in self.measurements:
            gni_now = frequency_calculator.should_measure('gni', ctx)

        if gni_now:
            metrics['gni'] = calculate_gni(
                net=self.net,
                X=self.X,
                Y=self.Y,
                loss_fn=self.loss_fn,
                batch_size=self.batch_size,
                n_estimates=500,
                tolerance=0.05,
            )


        
        # ----- Fisher eigenvalues (total and batch) -----
        if 'fisher' in self.measurements:
            if frequency_calculator.should_measure('fisher_total', ctx):
                metrics['fisher_total_eigenval'] = compute_fisher_eigenvalues(self.net, self.X).item()
            if frequency_calculator.should_measure('fisher_batch', ctx):
                metrics['fisher_batch_eigenval'] = compute_fisher_eigenvalues(self.net, X_batch).item()

        # ----- Parameter distance from reference -----
        if 'param_distance' in self.measurements:
            if self.param_reference is None:
                raise ValueError('Parameter reference must be provided for param_distance measurement')
            if frequency_calculator.should_measure('param_distance', ctx):
                metrics['param_distance'] = calculate_param_distance(self.net, self.param_reference).item()

        # ----- Gradient norm squared estimate -----
        if 'gradient_norm' in self.measurements:
            if frequency_calculator.should_measure('gradient_norm_squared', ctx):
                metrics['gradient_norm_squared'] = calculate_gradient_norm_squared_mc(
                    self.net,
                    self.X,
                    self.Y,
                    self.loss_fn,
                    batch_size=self.batch_size,
                    n_estimates=200,
                    min_estimates=10,
                    eps=0.01,
                )

        # ----- Expected one-step loss change -----
        if 'one_step_loss_change' in self.measurements:
            if frequency_calculator.should_measure('one_step_loss_change', ctx):
                metrics['one_step_loss_change'] = calculate_expected_one_step_full_loss_change(
                    self.net,
                    self.X,
                    self.Y,
                    self.loss_fn,
                    optimizer,
                    batch_size=self.batch_size,
                    n_estimates=1000,
                    min_estimates=10,
                    eps=0.01,
                    use_subset_of_data=2048,
                )

        # ----- Quadratic approximation diagnostics -----
        if self.quad_approx is not None and self.quad_approx.is_active:
            metrics['quadratic_loss'] = self.quad_approx.compute_quadratic_loss_for_logging(self.X, self.Y)

        # ----- Gradient projection diagnostics -----
        grad_projection_now = False
        if 'grad_projection' in self.measurements:
            if self.sde_enabled or self.gd_noise:
                raise Exception('Gradient projection not implemented for SDE or GD with noise')
            grad_projection_now = frequency_calculator.should_measure('grad_projection', ctx)

        if self.proj_switch_step is not None and step_number >= self.proj_switch_step:
            grad_projection_now = True

        if grad_projection_now:
            if not (self.quad_approx is not None and self.quad_approx.is_active):
                if (
                    self.eigenvector_cache is not None
                    and hasattr(self.eigenvector_cache, 'eigenvectors')
                    and len(self.eigenvector_cache.eigenvectors) > 0
                ):
                    params = [p for p in self.net.parameters() if p.requires_grad]
                    full_preds = self.net(self.X).squeeze(dim=-1)
                    full_loss_for_grad = self.loss_fn(full_preds, self.Y)
                    grad_list = torch.autograd.grad(full_loss_for_grad, params, create_graph=False, retain_graph=False)
                    grad_flat = torch.cat([g.reshape(-1) for g in grad_list]).detach()

                    cached_vecs = torch.stack(self.eigenvector_cache.eigenvectors, dim=1).to(grad_flat.device)
                    cached_vals = getattr(self.eigenvector_cache, 'eigenvalues', None)

                    max_k = min(20, self.num_eigenvalues)
                    metrics['grad_projections'] = compute_gradient_projection_ratios(
                        grad_vector=grad_flat,
                        eigvecs=cached_vecs,
                        max_k=max_k,
                        eigenvalues=cached_vals,
                    )

        # ----- batch sharpness (but with the expectation inside) -----
        if 'batch_sharpness_exp_inside' in self.measurements:
            if frequency_calculator.should_measure('batch_sharpness_exp_inside', ctx):
                metrics['batch_sharpness_exp_inside'] = calculate_averaged_grad_H_grad(
                    self.net,
                    self.X,
                    self.Y,
                    self.loss_fn,
                    batch_size=self.batch_size,
                    n_estimates=1000,
                    min_estimates=20,
                    eps=0.005,
                )


        # ----- Batch lambda max -----
        batch_lmax_now = False
        if 'batch_lmax' in self.measurements:
            if self.gd_noise is None:
                batch_lmax_now = frequency_calculator.should_measure('batch_lambda_max', ctx)
            else:
                raise ValueError('Batch lambda max not implemented for GD noise')

        if batch_lmax_now:
            optimizer.zero_grad()
            preds = self.net(X_batch).squeeze(dim=-1)
            loss = self.loss_fn(preds, Y_batch)
            batch_lmax = compute_eigenvalues(loss, self.net, k=1, max_iterations=50, reltol=1e-3)
            metrics['batch_lmax'] = batch_lmax.item()
            print(
                f"Epoch {epoch + 1}, Step {step_in_epoch}: Batch Lambda Max = {metrics['batch_lmax']}, "
                f"Loss = {loss.item()}"
            )
    

        metrics['epoch_loss_update'] = epoch_loss_update
        return metrics
    
        


# -------------------------------------
# Section: Training Function
# -------------------------------------


def train(
            net,
            optimizer,
            data, # tuple of X_train, Y_train, X_test, Y_test
            max_epochs,
            max_steps,
            batch_size,
            save_to, #folder
            device,
            verbose=True,
            loss_fn=nn.MSELoss(),
            permute=True,
            stop_loss=None,
            initial_sharpness=0,
            epoch_to_start=0,
            step_to_start=0,
            sharpness_every=None,
            gd_noise=False,
            noise_magnitude=None,
            results_rarely: bool = False,
            measurements: set = {},
            param_reference = None,  # reference weights to measure distance from during training
            cache_eigenvectors: bool = True,  # use eigenvector caching for warm starts
            sde_enabled: bool = False,  # enable SDE integration
            sde_h: float = 0.01,  # SDE integration time step
            sde_eta: float = None,  # SDE learning rate (uses optimizer lr if None)
            sde_seed: int = 888,  # SDE random seed
            use_power_iteration: bool = False,  # Use power iteration for eigenvalue computation
            num_eigenvalues: int = 1,  # Number of eigenvalues to compute
            checkpoint_every_n_steps: int = None,  # Checkpoint frequency 
            quad_switch_step: int = None,  # Step to switch to quadratic approximation
            use_gauss_newton: bool = False,  # Use Gauss-Newton instead of Hessian
            quad_switch_lr: float = None,  # lr to use after switching to quadratic approximation
            precise_plots: bool = False,  # Enable more frequent measurements for precise plotting
            rare_measure: bool = False,  # Make expensive measurements rarer
            memorization_outlier_frac: float = 0.05,  # Fraction of samples marked as outliers for memorization metric
            # Gradient projection configuration
            proj_switch_step: int = None,  # Step to start projecting minibatch gradients
            proj_top_l: int = None,        # Number of top eigendirections to use for projection
            proj_to_residual: bool = False, # If True, project to orthogonal complement of top-l eigenspace
            wandb_run=None,
            wandb_enabled: bool = False,
            wandb_run_id: str = None,
            per_sample_cfg=None, #NEW 
    ):
    
    # -------------------------------------
    # Section: Setup
    # -------------------------------------
    start_time = time.time()
    print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")


    # COMPUTE_GHG = True

    NET_SAVES_PER_TRAINING = 200


    # if epochs is None:
    #     epochs = steps // (len(data[0]) // batch_size) + 1

    assert max_epochs is not None or max_steps is not None
    if max_epochs is None:
        max_epochs = 1000000

    X_train, Y_train, X_test, Y_test = data

    X, Y = X_train, Y_train

    net = net.to(device)
    net.train()
    net.float()

    RECORD_EVERY = 20

    # Create the save_to directory if it doesn't exist
    save_to.mkdir(parents=True, exist_ok=True)

    # paths
    model_save_path = save_to / 'checkpoints'

    results_file = save_to / 'results.txt'
    if device == 'cpu':
        # No buffering on CPU to ensure writes happen immediately
        results_file = open(results_file, 'a', buffering=1)
        torch.set_num_threads(40)
    else:
        # Use buffering on GPU for better performance
        # TO CHANGE
        results_file = open(results_file, 'a', buffering=1_00)

    final_file = save_to / 'final.json'
    final_file = open(final_file, 'w')  


    X = X.to(device)
    Y = Y.to(device)

    step_number = -1 if step_to_start == 0 else step_to_start

    if gd_noise is not None:
        grad_storage = GradStorage(net, recalculate_every=10)

    # calculate epochs to save - total we want at most 100 saves for the whole training
    
    steps_in_epoch = len(X) // batch_size
    epochs_expected = max_steps // steps_in_epoch
    save_net_every = max(epochs_expected // NET_SAVES_PER_TRAINING, 1)



    epoch_loss = float('+inf')
    stop_training = False

    # -------------------------------------
    # Section: Measurements
    # -------------------------------------
    # ----- Eigenvector Cache Setup -----
    eigenvector_cache = None
    # Also create cache if gradient projection is enabled, since it relies on cached eigenvectors
    if (('lmax' in measurements or 'grad_projection' in measurements) or (proj_switch_step is not None)) and cache_eigenvectors:
        max_cache = 5
        if num_eigenvalues is not None:
            max_cache = max(max_cache, num_eigenvalues)
        if proj_top_l is not None:
            max_cache = max(max_cache, proj_top_l)
        eigenvector_cache = EigenvectorCache(max_eigenvectors=max_cache)
    
    # ----- Measurement Runner Wiring -----
    measurement_runner = MeasurementRunner(
        net=net,
        loss_fn=loss_fn,
        full_inputs=(X, Y),
        test_inputs=(X_test, Y_test),
        measurements=measurements,
        device=device,
        batch_size=batch_size,
        save_dir=save_to,
        eigenvector_cache=eigenvector_cache,
        num_eigenvalues=num_eigenvalues,
        use_power_iteration=use_power_iteration,
        precise_plots=precise_plots,
        rare_measure=rare_measure,
        param_reference=param_reference,
        step_to_start=step_to_start,
        sde_enabled=sde_enabled,
        gd_noise=gd_noise,
        proj_switch_step=proj_switch_step,
        quad_approx=quad_approx,
        memorization_outlier_frac=memorization_outlier_frac,
    )
    # ----- Run Identification -----
    run_id = wandb_run_id or generate_run_id()

    # If resuming at/after projection switch step, precompute eigendirections immediately
    if proj_switch_step is not None and step_to_start >= proj_switch_step:
        raise ValueError("Start of grad projection has to be after restart step, not at or before it")

    # -------------------------------------
    # Section: Training Step
    # -------------------------------------
    for epoch in range(epoch_to_start, max_epochs):
        if step_number >= max_steps:
            print(f"Reached max steps {max_steps}, stopping the training")
            results_file.flush()
            results_file.close()
            break

        shuffle = T.randperm(len(X))
        if permute:
            X_shuffled = X[shuffle]
            Y_shuffled = Y[shuffle]
        else:
            X_shuffled = X
            Y_shuffled = Y

        # save the model
        model_save_fle = model_save_path / f'net_{epoch}.pt'
        if save_net_every == 0 or epoch % save_net_every == 0:
            T.save(net.state_dict(), model_save_file)


        losses_in_epoch = []
        if stop_training:
            break

        for i in range(0, len(X) // batch_size): # i runs over steps in a epoch
            step_number += 1

            msg = f"{epoch:03d}, {step_number:05d}, "

            X_batch = X_shuffled[i*batch_size : (i+1)*batch_size]
            Y_batch = Y_shuffled[i*batch_size : (i+1)*batch_size]

            ##### LMBDA MAX WORK ####

            ### condition to calculate the lambda max ###
            TEMP_OVERRIDE = False
            if 'gni' in measurements: TEMP_OVERRIDE = True 
            lmax_now = False
            
            if 'lmax' in measurements:
                if TEMP_OVERRIDE:
                    def do_lmax():
                        FIRST_FEW = 256
                        FIRST_SUPER_FEW = 128
                        if step_number < FIRST_SUPER_FEW:
                            return True
                        
                        if step_number < FIRST_FEW:
                            return step_number % 4 == 0

                        how_often = 256
                        if step_number > 10000:
                            how_often = how_often * 2
                        if step_number > 20000:
                            how_often = how_often * 2
                        if step_number > 100_000:
                            how_often = how_often * 2
                        fullbs_now = step_number % how_often == 0
                        return fullbs_now
                    
                    lmax_now = do_lmax()

                else:
                    def do_lmax():
                        if batch_size <= 32:
                            how_often = 256
                        else:
                            how_often = 128
                        
                        if step_number > 10000:
                            how_often = how_often * 2
                        if step_number > 20000:
                            how_often = how_often * 2
                        if step_number > 100_000:
                            how_often = how_often * 2
                        return step_number % how_often == 0
                    
                    lmax_now = do_lmax()

            if lmax_now:
                # Clear CUDA cache if we're using GPU
                if str(device).startswith('cuda'):
                    torch.cuda.empty_cache()
                optimizer.zero_grad()

                # # TODO
                # LMAX_MAX_SIZE = 8192
                # # Check available CUDA memory before full batch computation
                # if str(device).startswith('cuda'):
                #     total_memory = torch.cuda.get_device_properties(0).total_memory
                #     if total_memory < 20 * 1024**3:  # Less than 20GB
                #         if isinstance(net, CNN):
                #             LMAX_MAX_SIZE = 2048 + 512
                #         if isinstance(net, ResNet):
                #             LMAX_MAX_SIZE = 512
                LMAX_MAX_SIZE = 1_000_000 # a placeholder originally used not to compute the lambda max on the whole dataset

                    
                if len(X) > LMAX_MAX_SIZE:
                    # If the dataset is too large, take a random subset
                    subset_indices = np.random.choice(len(X), LMAX_MAX_SIZE, replace=False)
                    X_subset = X[subset_indices]
                    Y_subset = Y[subset_indices]
                else:
                    # Use the whole dataset
                    X_subset = X
                    Y_subset = Y

                preds = net(X_subset).squeeze(dim=-1)

                loss = loss_fn(preds, Y_subset)
                lmax = compute_lambdamax(loss, net, max_iterations=100, 
                                              epsilon=1e-4)
                
                # if COMPUTE_GHG:
                #     sharpness, gHg = sharpness
                #     gHg = gHg.item()
                # else:
                #     gHg = np.nan

                lmax = lmax.item()
                full_gHg = np.nan

                print(f"Epoch {epoch+1}, Step {i}: Total lambda max = {lmax}, Loss = {loss.item()} !!!")
                # total_lmax = lmax
                # total_gHg = gHg
                full_loss = loss.item()

                full_accuracy = calculate_accuracy(preds, Y_subset)



                if math.isnan(full_loss):
                    print("Full loss is NaN, the network prolly diverged, stopping the training")
                    results_file.flush()
                    results_file.close()
                    return
                
                epoch_loss = full_loss

            else:
                lmax = np.nan
                # total_lmax = np.nan
                full_loss = np.nan
                full_gHg = np.nan
                # total_gHg = np.nan
                full_accuracy = np.nan
            
            if stop_loss is not None and epoch_loss < stop_loss:
                print(f"Loss {epoch_loss} is below the stop loss {stop_loss}, stopping the training")
                stop_training = True
                break


            ###### BATCH LAMBDAMAX WORK  ######
            batch_lmax_now = False
            if 'batch_lmax' in measurements:
                if gd_noise is None:
                    how_often = 16
                    if step_number > 50_000:
                        how_often = 32
                    if step_number > 100_000:
                        how_often = 64
                    
                    batch_lmax_now = step_number % how_often == 0
                    
                else:
                    raise ValueError("There should be some value here, but it is not implemented yet")

            if batch_lmax_now:
                optimizer.zero_grad()
                preds = net(X_batch).squeeze(dim=-1)

                loss = loss_fn(preds, Y_batch)
                batch_lmax = compute_lambdamax(loss, net, max_iterations=50, 
                                                    epsilon=1e-3)
                batch_lmax = batch_lmax.item()

                #     gHg = gHg.item()
                #     batch_gHg = gHg
                # else:
                # batch_gHg = np.nan
                # batch_sharpness = batch_sharpness.item()

                if i % RECORD_EVERY == 0:
                    print(f"Epoch {epoch+1}, Step {i}: Batch Lambda Max = {batch_lmax}, Loss = {loss.item()}")
                
            else:
                batch_lmax = np.nan
                # batch_gHg = np.nan

            ###### BATCH SHARPNESS WORK ######
            batch_sharpness_now = False
            if 'batch_sharpness' in measurements:
                how_often = 8
                batch_sharpness_now = step_number % how_often == 0
            
            if batch_sharpness_now:
                net.zero_grad()
                preds = net(X_batch).squeeze(dim=-1)
                loss = loss_fn(preds, Y_batch)
                batch_sharpness = compute_grad_H_grad(loss,
                                    net)
                batch_sharpness = batch_sharpness.item()
            
            else:
                batch_sharpness = np.nan
            
            ####### STATIC BATCH SHARPNESS WORK #######
            # frequency to calculate it
            batch_sharpness_static_now = False
            if 'batch_sharpness_static' in measurements:
                if batch_size < 32:
                    how_often = 128
                else:
                    how_often = 64
                if step_number > 5000:
                    how_often = how_often * 2
                if step_number > 50000:
                    how_often = how_often * 2
                
                batch_sharpness_static_now = step_number % how_often == 0


            batch_sharpness_static = np.nan
            if batch_sharpness_static_now:
                batch_sharpness_static = calculate_averaged_grad_H_grad(net,
                                                  X,
                                                  Y,
                                                  loss_fn,
                                                  batch_size=batch_size,
                                                  n_estimates=600,
                                                  tolerance = 0.01
                )
            

            ##### FISHER WORK #####

            fisher_total_eigenval = np.nan
            fisher_batch_eigenval = np.nan

            if 'fisher' in measurements:
                if calculate_fisher_total_condition(i, step_number, batch_size, initial_sharpness, sharpness_every):
                    fisher_total_eigenval = compute_fisher_eigenvalues(net, X).item()
                
                if calculate_fisher_batch_condition(i, step_number, batch_size, initial_sharpness, sharpness_every):
                    fisher_batch_eigenval = compute_fisher_eigenvalues(net, X_batch).item()
            

            ##### GNI work ####
            # frequency to calculate it
            gni_now = False
            if 'gni' in measurements:
                if batch_size < 32:
                    how_often = 256
                else:
                    how_often = 64
                
                if step_number > 5000:
                    pass
                    # how_often = how_often * 2
                
                gni_now = step_number % how_often == 0

                if step_number - step_to_start < 8:
                    gni_now = True
            
            gni = np.nan
            if gni_now:
                gni = calculate_averaged_gni(
                    net=net,
                    X=X,
                    Y=Y,
                    loss_fn=loss_fn,
                    batch_size=batch_size,
                    n_estimates=500,
                    tolerance=0.01
                )
            
            ##### WEIGHT DISTANCE WORK ####

            param_distance = np.nan

            param_distance_now = False
            if 'param_distance' in measurements:
                if param_reference is None:
                    raise ValueError("You should provide a reference weights to measure distance from")
                if batch_size < 32:
                    how_often = 1
                else:
                    how_often = 1
                
                param_distance_now = step_number % how_often == 0

            if param_distance_now:
                # calculate the distance from the reference weights
                param_distance = calculate_param_distance(net, param_reference)
                param_distance = param_distance.item()



            
            # now calculate the total loss for GNI
            FIRST_FEW = 32
            full_loss_now = False
            if 'gni' in measurements:
                if step_number - step_to_start < FIRST_FEW:
                    full_loss_now = True
                
                how_often = 32
                if step_number % how_often == 0:
                    full_loss_now = True


                if full_loss_now:
                    X_subset = X
                    Y_subset = Y
                    preds = net(X_subset).squeeze(dim=-1)

                    loss = loss_fn(preds, Y_subset)
                    full_loss = loss.item()


            ######## SGD STEP #######
            optimizer.zero_grad()

            if not gd_noise:
                preds = net(X_batch).squeeze(dim=-1)

                loss = loss_fn(preds, Y_batch)

                if math.isinf(loss.item()) or math.isnan(loss.item()):
                    results_file.flush()
                    results_file.close()
                    raise ValueError("Loss is inf or NaN, stopping the training")

                # Backward pass
                loss.backward()
                
                optimizer.step()

            else:
                # this is the GD with noise
                # the whole thing is done in the function, including updating the weights
                loss = gd_with_noise(net=net, X = X, Y=Y, loss_fn=loss_fn, noise_type=gd_noise, 
                                     optimizer=optimizer, batch_size=batch_size, step_number=step_number, 
                                     grad_storage=grad_storage, noise_magnitude=noise_magnitude)
            

            batch_loss = loss.item()
            losses_in_epoch.append(batch_loss)

            ############## RECORD THE RESULTS ##############
            if True: # not results_rarely or (results_rarely and ghg_now):
                msg += f"{batch_loss:7.6f}, {full_loss:7.6f}, {batch_lmax:6.2f}, {lmax:6.2f}, {batch_sharpness:6.1f}, {full_gHg:6.1f}, {fisher_batch_eigenval:6.2f}, {fisher_total_eigenval:6.1f}, {batch_sharpness_static:6.2f}, {gni:6.2f}, {full_accuracy:6.2f}, {param_distance:.7e}"
                results_file.write(msg + "\n")

        
        # end of epoch
        epoch_loss = np.mean(losses_in_epoch)
        
        results_file.flush()

        
    
    T.save(net.state_dict(), model_save_path / f'net_final.pt')

    results_file.close()

    end_time = time.time()
    print(f"Training finished at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total training time: {end_time - start_time:.2f} seconds")



    # ----- Optional Final Measurements -----
    if 'final' in measurements:
        final_file = save_to / 'final.json'
        final_file = open(final_file, 'w') 

        # do the final measurements here - depending on what is needed
    




if __name__ == '__main__':
    seed = 888
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    

    # Command line arguments
    parser = argparse.ArgumentParser(description='Training script')
    # Training parameters
    parser.add_argument('--batch', type=int, default=64, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--steps', type=int, default=10000, help='Number of steps to train. Either epochs or steps should be provided')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--stop_loss', type=float, default=None, help='Stop training if loss goes below this value')
    # Loss function
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'ce'], help='Loss function to use (mse or ce)')

    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use for training')
    parser.add_argument('--classes', type=int, nargs=2, default=[1, 9], help='Two class labels to use for training. Default is [1, 9], as being probably the most difficult classes to separate')
    parser.add_argument('--num_data', type=int, default=1024, help='Number of datapoints to train on')

    # Model configuration
    parser.add_argument('--model', type=str, default='mlp', help='Network architecture to use for training')
    parser.add_argument('--init_scale', type=float, default=None, help='Initialization scale for network weights')
    parser.add_argument('--no_init', action='store_true', help='If set, do not initialize network weights')

    # Continuation options
    parser.add_argument('--cont_folder', type=str, default=None, help='Folder to continue training from')
    parser.add_argument('--cont_epoch', type=int, default=0, help='Epoch to continue training from')
    parser.add_argument('--cont_last', action='store_true', help='If set, continue training from the last run that fits the parameters')

    # Momentum and adam
    parser.add_argument('--momentum', type=float, default=None, help='Momentum for SGD optimizer')
    parser.add_argument('--adam', action='store_true', help='If set, use Adam optimizer instead of SGD')

    # Measurement settings
    parser.add_argument('--sharp_every', type=int, help='Frequency of sharpness computation')
    parser.add_argument('--init_sharp', type=int, default=0, help='Compute total sharpness for the first n steps')
    
    # Measurement settings (main)
    # parser.add_argument('--fullbs', action='store_true', help='If set, compute the lambda_max, aka FullBS')
    parser.add_argument('--lambdamax', '--lmax', action='store_true', help='If set, compute the lambda_max, aka FullBS')
    parser.add_argument('--batch-sharpness', '--batch-sharpness-step', '--bs', action='store_true', dest='batch_sharpness',
                        help='If set, compute the batch sharpness: E[gHg/g²] with the expectation taken across mini-batches. Use --batch-sharpness-step for backward compatibility.')
    parser.add_argument('--step-sharpness', action='store_true', dest='step_sharpness',
                        help='If set, compute the step sharpness: the current-mini-batch Rayleigh quotient g·Hg/g². Average across steps to recover the traditional batch sharpness.')
    parser.add_argument('--gni', action='store_true', help='If set, compute the Gradient-Noise Interaction quantity.')

    # --- Measurement Flags (Secondary, aka still useful) ---
    parser.add_argument('--hessian-trace', action='store_true', help='Estimate the trace of the full-batch loss Hessian via a Hutchinson-style estimator')
    parser.add_argument('--grad-projection', action='store_true', help='Compute grad_projection_i: fraction of full-batch gradient lying in span of top-i cached Hessian eigenvectors (i up to 20); uses cached eigenvectors only; only for plain SGD')
    parser.add_argument('--one-step-loss-change', action='store_true', help='If set, compute the expected one-step change in loss using Monte Carlo estimation')
    parser.add_argument('--gradient-norm', action='store_true', help='If set, compute the Monte Carlo estimate of squared norm of mini-batch gradients')
    parser.add_argument('--final', action='store_true', help='If set, compute the lambda_max and step sharpness at the end')
    parser.add_argument('--memorization-hessian-outliers', action='store_true', help='Compute memorization stats based on alignment with top Hessian eigenvector (heavy; runs rarely)')
    parser.add_argument('--memorization-outlier-frac', type=float, default=0.05, help='Fraction of examples treated as outliers for Hessian-alignment memorization stats')

    
    # --- Measurement Flags (Tertiary, aka almost completely useless) ---
    parser.add_argument('--batch-sharpness-exp-inside', action='store_true', help='If set, compute the batch sharpness using E[gHg]/E[g²], where the expectation is inside the ratio. Compare with step-sharpness, where the expectation stays outside the ratio.')
    parser.add_argument('--batch-lambdamax','--batchlmax', action='store_true', help='If set, compute the batch lambda_max(H_B), aka batch lambda max')
    parser.add_argument('--batch-sharpness-static', action='store_true', help='If set, compute the ')
    parser.add_argument('--gni', action='store_true', help='If set, compute the Gradient-Noise Interaction quantity')

    # Measurement settings (secondary)
    parser.add_argument('--fisher', action='store_true', help='If set, compute Fisher information matrix eigenvalue. Currently only works with one-dim output')
    parser.add_argument('--param_distance', action='store_true', help='If set, compute the distance from the reference weights')
    parser.add_argument('--param_file', type=str, default=None, help='Path to reference parameters for computing parameter distance')
    parser.add_argument('--final', action='store_true', help='If set, compute the lambda_max and batch sharpness at the end')


    parser.add_argument('--results_rarely', action='store_true', help='If set, results will be recorded less frequently')

    # Noise configuration
    parser.add_argument('--gd_noise', type=str, default=None, help='Do GD, but with gaussian noise to simulate SGD. Supoprted noises: sgd, diag, iso, const')
    parser.add_argument('--noise_mag', type=float, default=None, help='The noise magnitude for the constant noise')

    # Randomness settings
    parser.add_argument('--dataset_seed', type=int, default=888, help='Random seed for dataset preparation')
    parser.add_argument('--init_seed', type=int, default=8888, help='Random seed for network initialization')

    # --- wandb Settings ---
    parser.add_argument('--wandb-tag', type=str, default=None, help='Tag to add to the wandb run')
    parser.add_argument('--wandb-name', type=str, default=None, help='Optional suffix appended to default wandb run name (sanitized)')
    parser.add_argument('--wandb-notes', type=str, default=None, help='Optional notes/description attached to the wandb run')
    parser.add_argument('--disable-wandb', action='store_true', help='Disable Weights & Biases logging for debugging/testing')

    # --- New Measurement Flag ---
    parser.add_argument('--train-test-gap', action='store_true', help='If set, compute the training and testing accuracy and gap (heavy, runs rarely)')

    # --- NEW: Per-Sample Histogram Configuration ---
    parser.add_argument('--per-sample', action='store_true', help='If set, compute per-sample histograms (heavy; runs rarely)')
    parser.add_argument('--per-sample-freq', type=float, default=None, help='Frequency of per-sample histograms, as fraction of max_steps (default: 0.01 = every 100 steps for 10k max_steps)')
    parser.add_argument('--per-sample-min-log10', type=float, default=-8, help='Min log10 value for log10 histograms (default: -8)')
    parser.add_argument('--per-sample-max-log10', type=float, default=0, help='Max log10 value for log10 histograms (default: 0)')
    parser.add_argument('--per-sample-bins', type=int, default=80, help='Number of bins for log10 histograms (default: 80)')
    parser.add_argument('--per-sample-metrics', type=str, nargs='+', default=['loss','resid','kappa'], choices=['loss','resid','kappa'], help='Which metrics to histogram (default: loss resid kappa)')
    parser.add_argument('--no-frames', action='store_true', help='Only save counts/quantiles as .npz; do not render PNG frames')

    # ----- Argument Parsing -----
    args = parser.parse_args()


    #### deal with all the arguments
    # set the parameters
    batch_size = args.batch
    dataset = args.dataset
    device = (T.device('cuda') if T.cuda.is_available() else 'cpu')
    
    
    if args.steps is not None and args.epochs is not None:
        raise ValueError("You should provide either epochs or steps, not both")
    
    ### set which values to compute ####
    measurements = {name for name, enabled in [
    ('lmax', args.lambdamax),
    ('batch_lmax', args.batch_lambdamax),
    ('batch_sharpness', args.batch_sharpness),
    ('batch_sharpness_static', args.batch_sharpness_static),
    ('gni', args.gni),
    ('fisher', args.fisher),
    ('final', args.final),
    ('param_distance', args.param_distance),
    ] if enabled}

    #### result storage ####
    RES_FOLDER.mkdir(parents=True, exist_ok=True)
    run_folder = initialize_folders(args, RES_FOLDER)

    if args.cont_folder is not None:
        run_folder, step_to_start = run_folder
    else:
        step_to_start = 0

    #### prepare loss #####
    if args.loss == 'mse':
        loss_fn = SquaredLoss()
    elif args.loss == 'ce':
        loss_fn = nn.CrossEntropyLoss()

    ##### Prepare dataset and model ####
    dataset_presets = get_dataset_presets()
    model_presets = get_model_presets()

    ### Prepare dataset ###
    data = prepare_dataset(dataset, DATASET_FOLDER, args.num_data, args.classes, args.dataset_seed, loss_type=args.loss)

    ### Prepare model ###
    name = args.model
    params = model_presets[name]['params']
    params['input_dim'] = dataset_presets[dataset]['input_dim']
    params['output_dim'] = dataset_presets[dataset]['output_dim']
    net = prepare_net(
        model_type=model_presets[name]['type'], 
        params=params
        )

    #### Initialize net #####
    if not args.no_init:
        initialize_net(net, scale=args.init_scale, seed=args.init_seed)

    #### Load the model if continuing ####
    if args.cont_folder is not None:
        cont_folder = Path(RES_FOLDER / args.cont_folder)
        state_file = cont_folder / 'checkpoints' / f'net_{args.cont_epoch}.pt'
        net.load_state_dict(T.load(state_file, map_location=device))

    
    #### Distance from the reference weights ####
    # Load the reference parameters to calculate the distance from (if provided)
    param_reference = None
    if args.param_distance:
        if args.param_file is None:
            # Create a zero vector as a reference if no param_file is provided
            print("No parameter file provided. Creating a zero vector as reference.")
            param_reference = []
            for param in net.parameters():
                param_reference.append(torch.zeros_like(param).flatten())
            param_reference = torch.cat(param_reference)
    if args.param_file is not None:
        param_reference = T.load(args.param_file, map_location=device)
        # param_reference = param_reference['model_state_dict']
        # param_reference = {k: v.to(device) for k, v in param_reference.items()}

    ### Prepare optimizer ###
    optimizer = prepare_optimizer(net, args.lr, args.momentum, args.adam)

    # ----- Checkpoint Cadence Determination -----
    if args.checkpoint_every is not None:
        checkpoint_every_n_steps = args.checkpoint_every
    else:
        checkpoint_every_n_steps = max(args.steps // 200, 1) if args.steps else None
    
    # ----- Training Invocation -----
    train(
        net=net,
        optimizer=optimizer,
        data=data,
        max_epochs=args.epochs,
        max_steps=args.steps,
        batch_size=args.batch,
        save_to=run_folder,
        device=device,
        loss_fn=loss_fn,
        verbose=True,
        stop_loss = args.stop_loss,
        initial_sharpness = args.init_sharp,
        epoch_to_start=args.cont_epoch,
        step_to_start = step_to_start,
        sharpness_every=args.sharp_every,
        gd_noise=args.gd_noise,
        noise_magnitude=args.noise_mag,
        results_rarely=args.results_rarely,
        measurements=measurements,
        param_reference=param_reference,
        cache_eigenvectors = not args.disable_cache_eigenvectors,
        sde_enabled=args.sde,
        sde_h=args.sde_h,
        sde_eta=args.sde_eta,
        sde_seed=args.sde_seed,
        use_power_iteration=args.use_power_iteration,
        num_eigenvalues=args.num_eigenvalues,
        checkpoint_every_n_steps=checkpoint_every_n_steps,
        quad_switch_step=args.quad_switch_step,
        quad_switch_lr=args.quad_switch_lr,
        use_gauss_newton=args.use_gauss_newton,
        precise_plots=args.precise_plots,
        rare_measure=args.rare_measure,
        memorization_outlier_frac=args.memorization_outlier_frac,
        proj_switch_step=args.proj_switch_step,
        proj_top_l=args.proj_top_l,
        proj_to_residual=args.proj_to_residual,
        wandb_run=wandb_run,
        wandb_enabled=wandb_enabled,
        wandb_run_id=wandb_run_id,
        #NEW
        per_sample_cfg=None,

    )
