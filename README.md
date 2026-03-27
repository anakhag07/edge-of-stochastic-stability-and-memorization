# edge-of-stochastic-stability-and-memorization
This repository builds on edge-of-stochastic-stability repo to quantify memorization during training and relationship to edge of stochastic stability and progressive sharpening. The information on the original repository follows here. 

## Tracking Outlier Metrics (Input Prototypes)

Input-space prototypes now use one source flag, one mode flag, and four per-subset count flags. All four counts are per class. Base configs should leave `input_prototype_source` unset unless they also request at least one subset count.

- `--input-prototype-source`: `generate` or `from:<path-or-run>`
- `--input-prototypes-mode`: `train` or `val`
- `--input-boundary`, `--input-inliers`, `--input-x-outliers`, `--input-y-outliers`: selected prototypes per class

In `train` mode, the selected prototype subsets are appended to the training set and tracked during training.

In `val` mode, the selected prototype subsets are tracked but not trained on. Real-data subsets (`boundary`, `inliers`) are removed from the base training set before optimization; synthetic subsets (`x_outlier`, `y_outlier`) remain track-only.

Example (train mode):
```bash
python training.py --dataset cifar10_2cls --model mlp --loss ce \
  --batch 10000 --lr 0.01 --steps 150000 --num-data 10000 --init-scale 0.2 \
  --dataset-seed 888 --init-seed 8312 --lambdamax --batch-sharpness --classes 3 5 \
  --input-prototype-source from:$PROTO --input-prototypes-mode train \
  --input-x-outliers 25 --input-y-outliers 25 --input-inliers 25 --input-boundary 25
```

Example (validation mode):
```bash
python training.py --dataset cifar10_2cls --model mlp --loss ce \
  --batch 10000 --lr 0.01 --steps 150000 --num-data 10000 --init-scale 0.2 \
  --dataset-seed 888 --init-seed 8312 --lambdamax --batch-sharpness --classes 3 5 \
  --input-prototype-source from:$PROTO --input-prototypes-mode val \
  --input-x-outliers 25 --input-y-outliers 25 --input-inliers 25 --input-boundary 25
```

Full-GD batch behavior follows the effective training set size after prototype processing. Use `--batch full` to request explicit full-batch GD; `train` mode grows the batch to include injected prototypes and `val` mode shrinks it to match the held-out-adjusted training set.

Reusable prototype packages can be generated with:
```bash
python tools/generate_input_prototypes.py --dataset cifar10_2cls --model mlp \
  --classes 3 5 --num-data 10000 --loss ce --dataset-seed 888 \
  --input-boundary 50 --input-inliers 50 --input-x-outliers 50 --input-y-outliers 50
```

___




## Example Usage for New Metrics
Args: 
--

## Edge of Stochastic Stability (EoSS)

This repository accompanies the paper [Edge of Stochastic Stability: Revisiting the Edge of Stability for SGD](https://arxiv.org/abs/2412.20553) by Arseniy Andreyev and Pierfrancesco Beneventano. Feel free to reuse it in any way - if you ended up using this code, please consider citing it by citing our paper.

## Key Capabilities
- Run training for MLP/CNN/ResNet on CIFAR‑10/Fashion‑MNIST/SVHN with SGD, SGDM, or Adam.
- Measure core quantities: Batch Sharpness, $\lambda_{\max}$ (and other top eigenvalues of the Hessian of the loss), GNI, and Hessian trace
- Log to Weights & Biases for visualization, sweeps, and plots
- Easily extend with new measurements (see “Adding new measurements” checklist).
- Restart runs from a checkpoint and change hyper‑parameters mid‑training.

## Quick Start
- **Setup** (virtual environment highly recommended):
  ```bash
  conda create -n eoss python=3.12
  conda activate eoss
  pip install -r requirements.txt
  ```
  OR
  ```bash
  python3 -m venv eoss
  source eoss/bin/activate
  pip install -r requirements.txt
  ```
  note on conda: it seems to be deprecated by the "community" - although the code still works, although feel free to use an alternative
- **Point to data/results roots**:
  ```bash
  export DATASETS="/path/to/datasets"
  export RESULTS="/path/to/results"
  ```
  It is recommended to make those persistent -- e.g. by adding them into your .bashrc
- **Download the datasets**
```bash
  python setup/download_datasets.py
  ```
  If you are running on MacOS, this might fail because the python process doesn't have disk access. A *potential* solution is to enable full disk access in System Settings.
- **Sanity run (CPU-friendly)**:
  ```bash
  python training.py --dataset cifar10 --model mlp --batch 4 --lr 0.05 \
  --steps 1000 --num-data 64 \
  --lambdamax --batch-sharpness --disable-wandb \
  --cpu
  ```
  Runs training, writes results to a legacy `results.txt` under `$RESULTS`. Should take less than a minute to run.
- **Sanity run from JSON config**:
  ```bash
  python training.py --config configs/smoke_train.json
  ```
  JSON config files may use nested groups and `__comment` keys for section labels. CLI flags still work and override config values, e.g. `python training.py --config configs/smoke_train.json --steps 40`.
- **Inspect the latest run**:
  ```bash
  python visualization/plot_results.py
  ```
  Produces quick plots from the most recent `results.txt` into `visualization/img/`.



## Standard Training Run

- **Setting-up wandb**:
  The code uses wandb for logging the measurements during the runs. In particular, the rudimentary results.txt only supports very basic measurements.
  Therefore, setting up wandb is highly recommended. 
  Go to wandb.ai, create an account (they have edu discounts), and grab your API key.
  Add the API key and the project name (e.g. "eoss") into your env (e.g. into your .bashrc):
  ```bash
  export WANDB_MODE=offline           # store runs locally 
  export WANDB_PROJECT=eoss         # project name, change if you want
  export WANDB_DIR=$RESULTS     # where to store the wandb data

  export WANDB_API_KEY=<your api key here>
  ```
  This, respectively: stores results locally (without uploading to wandb servers); give project name (change if you want); selects where to store the wandb data; gives the api key;
  While we are at it, go ahead and email wandb to enable [run forking](https://docs.wandb.ai/guides/runs/forking/) on your "wandb entity" - it takes a couple of days for them to do it, and you would need it to continue runs.

- **Launching training**
  ```bash
  python training.py --dataset cifar10 --model mlp --batch 8 --lr 0.01 \
    --steps 150000 --num-data 8192 \
    --init-scale 0.2 --dataset-seed 111 --init-seed 8312 \
    --stop-loss 0.00001 
    --lambdamax --batch-sharpness
  ```
  Equivalent config-first invocation:
  ```bash
  python training.py --config path/to/experiment.json
  ```
  with optional overrides such as:
  ```bash
  python training.py --config path/to/experiment.json --lr 0.02 --batch 16
  ```
  Note: this is too computationally demanding to run on CPU - recommended to run on GPU, e.g. through slurm
  - 150k-step CIFAR-10 run with SGD on an MLP using batch size 8 and learning rate 0.01.
  - Seeds control dataset shuffling (`--dataset-seed`) and parameter initialization (`--init-seed`). Stops when the loss is at 1e-5.
  - This tracks two most important measurements: lambda_max and batch sharpness.
  - The code automatically detects if there is cuda present, and runs on it. Use an explicit --cpu switch if you really want to run on cpu.

- **Uploading results to wandb**
  ```bash
  cd $WANDB_DIR
  wandb sync --sync-all --include-offline --mark-synced --no-include-synced
  ```
  If you were running with `WANDB_MODE=offline`, you know need to upload the results to wandb (the reason why we did it is often-times slurm jobs with GPUs don't have access to the internet). This command uploads it. You can use tmux to run this command repeatedly (I know this is a hack, I don't care)

  You can now go to your wandb.ai panel to view the results!


## Visualization
Use `visualization/template.ipynb` for visualizing runs. This notebook should be extended for more concrete plots. One can either specify the wandb ids to plot (can be taken e.g. from the run's url in the web interface), or pull all the runs from one tag (convenient for sweeps, see below)

## Varying hyperparameters
- You can start with changing the learning rate/step size: notice how the 2/η threshold moves.
- Changing batch size: notice how the level of stabilization of lambda_max changes (the bigger the batch size, the higher the level).
- If you set learning rate as too low (relative to the "difficulty" of the problem), you might not enter EoSS regime. That is, you are going to converge before progressive sharpening brings you to regime of instability, and batch sharpness stabilizes below 2/η.
- If you set batch size too small (for the given step size), you might actually not converge. That is, you will enter a sort of EoSS-like instability regime, where batch sharpness is around 2/η, but there is no loss reduction happening. The reason behind this is usually ill-conditioned landscape, and is the exact situation of when in training one needs to increase the batch size/reduce the learning rate to "continue" training - i.e. indicating that there is no direction in the landscape which will continue training without causing progressive sharpening (which cannot happen because batch sharpness is already at 2/η).

## Things to be added at some point
- Transformer support - ViT (easier), for language (harder).
- Dataloader support (rather than loading the whole dataset into the memory). Not too diffcult - just make sure you are not making slow operations like the unparallelized imagine transformation of PyTorch. When I first did it, it took most of the computation time.
- Batches in the Hessian-vector product. Right now when computting the full-batch lambda_max, we might run out of memory on bigger models. It can be computed with smaller chunks of samples (Hessian is additive), but then you kinda lose the current speed-up where you keep one of the forward passes.
- Froward-mode AD + batching of vectors in the hvp. Honestly, unsure about this one. It is supposed to speed-up computation/use less memory, but in my experiments I didn' notice any improvements.
- Deterministic batches (determined by the epoch) - necessary for the restarts to see the exact same batches (easy).
- Tracking of distance between weights in different runs. I would recommend using a random projection, but it would be good to track this, the style of "Break-even point.." by Jastrzebski et al. and "Edge of Stability" by Cohen et al. One should use a projection to smaller random subspace (there is a way to do that efficiently).
- CI tests
I would also appreciate pull requests with any of those!


## Notebook runs
Use `notebooks/view_landscape.ipynb` for an example of a notebook that loads a specified checkpoint in a given run (see `RUN_ID` and `TARGET_STEP`). One can evalute the loss at that point, or e.g. continue traing. This particular notebook showcases the noise-induced oscillations present from the beginning of the training.
(more to come)


## wandb Logging
- wandb initializes automatically; set `WANDB_PROJECT`/`WANDB_MODE` if you need custom routing. Using wandb is recommended because `results.txt` only preserves the basic measurement trio for backward compatibility.

- Tag and annotate runs for later comparison:
  - `--wandb-tag batch-size-sweep` (useful for sweeps).
  - `--wandb-name mlp_b8_lr1e-2`.
  - `--wandb-notes "increase lr after 40k steps"`.
- Additional wandb details are in the Appendix.

## Continuing a run
You need forking functionality to be enabled in wandb for this to work (see above). 
Resume a run witch changed hyperparameters (e.g. reducing learning rate)
  ```bash
python training.py --dataset cifar10 --model mlp --batch 8 --lr 0.01 \
  --steps 150000 --num-data 8192 \
  --init-scale 0.2 --dataset-seed 111 --init-seed 8312 \
  --cont-run-id <run_id> --cont-step <approx_step> \
  ```


## How to run hyperparameter sweeps
There is a somewhat rudimentary way to run hyperparameter sweeps. Basically, you launch off a bunch of trainings (e.g. with slurm) with the same `--wandb-tag`, e.g. `--wandb-tag lr-sweep`, and then you can visualize them all together in the wandb.


## Adding new measurements

1. Implement the computation in `utils/measure.py` (or wherever).
2. Add a CLI flag and wire it in the loop in `training.py`.
3. Add/update frequency rule in `utils/frequency.py`.
4. Add wandb logging: define metric and extend `log_metrics(...)` in `utils/wandb_utils.py`.
5. Add a minimal test in `tests/` if you want.

## Repo map (only useful files)
- `training.py` – CLI entry point; sets up datasets/models, orchestrates SGD, and wires every measurement and checkpoint hook.
- `utils/measure.py` – all measurement kernels (batch sharpness, λ_max eigens, GNI, gradient norms, etc.).
- `utils/frequency.py` – schedules when measurements/checkpoints fire based on step counts and flags.
- `utils/wandb_utils.py` – W&B init, logging, checkpoint save/restore, run naming, continuation helpers.
- `utils/nets.py` – model zoo (MLP/CNN/ResNet variants), init schemes, and optimizer presets.
- `tests/` – regression and measurement sanity tests covering eigensolvers, sharpness estimators, wandb naming, and checkpoint restarts.
- `visualization/template.ipynb` – quick-start notebook to pull logged runs (W&B IDs or tags) and generate diagnostic plots.


------------------------

# Appendix

## Measurements reference
(might be unmainted - check the arg parser in training.py)
- `--lambdamax` / `--lmax`: full-dataset Hessian eigensolves via `utils.measure.compute_eigenvalues`. Uses lobpcg. See `--num-eigenvalues` for computing multiple highest eigenvalues. Supports eigenvalue caching for warm restarts, and power-iteration as a fallback, logs `lambda_max`, optionally larger igencalues, and refreshes `full_loss`/`full_accuracy` on the sampled set (currently only computes it on a subset). 
- `--step-sharpness`: Evaluates the Rayleigh quotient g·Hg/g² on the live mini-batch (`training.py:259`) through `compute_grad_H_grad`, giving the single-step sharpness proxy.
- `--batch-sharpness`: Estimates E[g·Hg/g²] over fresh batches via `calculate_averaged_grad_H_grad_step` with up to 1000 Monte Carlo samples and a 0.5% relative-error stop-rule (`training.py:267`); logs to wandb as `batch_sharpn`.
- `--gni`: Computes the gradient–noise interaction ratio (`training.py:242`) using 500 Monte Carlo draws with 5% tolerance in `calculate_gni`.


- `--batch-sharpness-exp-inside`: Switches the estimator to E[g·Hg]/E[g²] (`training.py:281`), sharing the same Monte Carlo controls to compare expectation placement.
- `--grad-projection`: Requires cached top Hessian eigenvectors (from `--lambdamax` or projection refresh) and plain SGD; records the cumulative gradient mass captured by the top-`l` directions via `compute_gradient_projection_ratios` (`training.py:344`).
- `--gradient-norm`: Uses `calculate_gradient_norm_squared_mc` to Monte Carlo E[‖∇f_B‖²] with relative-error 1% (`training.py:309`); useful for scaling sharpness diagnostics.
- `--hessian-trace`: Runs a Hutchinson-style estimator of the full-batch Hessian trace using repeated Hessian-vector products (`utils.measure.estimate_hessian_trace`), logging to wandb as `hessian_trace`; cadence is sparse because it requires full-batch graph construction.
- `--one-step-loss-change`: Samples mini-batches, performs a temporary optimizer step, evaluates full loss, restores parameters, and normalizes by η‖g‖² (`training.py:323`) through `calculate_expected_one_step_full_loss_change` (1000 max trials, adaptive stop at 1%).
- `--fisher`: Logs the top eigenvalue of the empirical Fisher on the entire dataset and on the current batch (`training.py:295`) by reusing the NTK-based solver in `compute_fisher_eigenvalues` (single-output models only).
- `--final`: Reserved hook for end-of-run sweeps (`training.py:908`); currently writes a placeholder `final.json` if enabled.
- `--param-distance`: Measures the L₂ gap to a reference parameter vector each time cadence allows (`training.py:302`); build it from `--param-file` or fall back to the zero vector when none is supplied.

## On Initialization of NNs
- The size of initialization really matters for the experiments. That's why the default initalization is low - so that initial landscape sharpness is low. Feel free to edit `--init-scale`.

## On Tests
We don't have CI implemented yet. The `tests/` folder just contains a number of random tests.

## Adding New Models, Datasets and Losses

**Models**
1. Implement the module in `utils/nets.py` (or a helper) as a `torch.nn.Module`; expect `params['input_dim']`/`params['output_dim']` to be filled in by the dataset preset.
2. Register the preset in `utils/nets.py:get_model_presets()` with a unique `type` key plus default hyperparameters.
3. Extend `utils/nets.py:prepare_net()` to instantiate the new `type` and update `initialize_net()` so seeding/`--init-scale` reuse still works.
4. If the model needs custom optimizer flags or smoke coverage, wire them in `training.py` and drop a quick regression in `tests/`.

**Datasets**
1. Add a loader in `utils/data.py` that returns `(X_train, Y_train, X_test, Y_test)` tensors, handles normalization, and accepts `num_data`, `dataset_seed`, `loss_type`, and optional `classes`.
2. Insert the dataset entry into `utils/data.py:get_dataset_presets()` with the correct `input_dim`/`output_dim` so the model factory can size itself.
3. Add a branch in `utils/data.py:prepare_dataset()` that points to the loader, respects `DATASETS` folder layout, and forwards `loss_type` so labels are transformed correctly.
4. Document any extra CLI knobs and, if the dataset needs assets, add a sanity check in `tests/` or a notebook before large runs.

**Losses**
1. Implement the loss (matching `loss_fn(preds, targets, sampling_vector=None)`) in `utils/nets.py` or a dedicated helper so measurements that pass masks still work.
2. Extend the `--loss` parser choices and instantiation block in `training.py` to map the new CLI string to your loss object.
3. Update each dataset branch in `utils/data.py` to handle the new `loss_type` (e.g., emit one-hot targets for squared losses, integer labels for CE-style losses) so shapes stay compatible.
4. Add a lightweight test (or reuse an existing one) that runs a forward/backward step with the new loss to catch shape/logging regressions early.

## wandb Additional Details

- Checkpoints: Saved under `wandb_checkpoints/<run_id>/` as `checkpoint_step_<step>.pt` plus an index `checkpoint_metadata.json`. See `utils/wandb_utils.py:save_checkpoint_wandb` and `find_closest_checkpoint_wandb`. Default cadence is ~200 saves per run (computed as `steps//200`, min 1); override with `--checkpoint-every N`. Final checkpoint always saved.
  - Location honors `WANDB_DIR` (defaults to `.`): checkpoints go to `$WANDB_DIR/wandb_checkpoints/<run_id>/`.
  - Contents: model/optimizer state dicts, `step`, `epoch`, `loss`, `run_id` (`utils/wandb_utils.py:save_checkpoint_wandb`).

- Run naming: Name is `"{dataset}_{model}_b{batch}_lr{lr}"` with an optional sanitized suffix from `--wandb-name`. See `utils/naming.py:compose_run_name` and `sanitize_run_name_part` (collapses invalid chars, trims, max 128). You can add `--wandb-tag` and `--wandb-notes` (`training.py:1026`–`1029`).

- Offline by default: `utils/wandb_utils.py:init_wandb` sets `mode=os.getenv("WANDB_MODE", "offline")`. Cluster GPUs often lack internet; offline mode logs locally under `wandb/` (or `$WANDB_DIR/wandb/`). To sync later: `wandb sync <offline-run-dir>` or set `WANDB_MODE=online` and `WANDB_API_KEY` to stream live.

- Continuation (step-based): Use `--cont-run-id <run_id> --cont-step <approx_step>`.
  - Loader picks the closest checkpoint ≤ target (`utils/wandb_utils.py:find_closest_checkpoint_wandb`), then restores model (SGD only; Adam continuation is blocked in `training.py:1161`).
  - The new run is initialized as a fork of the original using `fork_from=f"<run_id>?_step=<loaded_step>"` with `fork_from` and `fork_step` also stored in config (`utils/wandb_utils.py:init_wandb`).
  - Checkpoint cadence in training loop: saved every `checkpoint_every_n_steps` (derived in `training.py:1215`–`1220`) and at the very end (`training.py:884`).

- Fork a run (wandb): The code relies on W&B “fork a run” to start a new run that points back to the original at a given step (`init_wandb(..., fork_from=...)`). This feature may be gated; contact W&B support to enable it for your account/org and see W&B docs for “fork a run”.

- If you are running with `WANDB_MODE=offline`, you have to manually sync your runs (see the wandb section in the body). You can do it consistenetly in the background in the following hacky way using tmux:
```bash
tmux new -d -s wandb-sync 'bash -lc '\''conda activate dl; cd $WANDB_DIR; while true; do wandb sync --sync-all --include-offline --mark-synced --no-include-synced; sleep 300; done'\'''
```
To check on the tmux session:
```bash
tmux attach -t wandb-sync
```

## Dataset & Results Folders
Set `DATASETS` to the root containing subfolders expected by loaders:
- CIFAR-10: `$DATASETS/cifar10/` (torchvision format)
- Fashion-MNIST: `$DATASETS/fmnist/` (torchvision format)
- SVHN: `$DATASETS/svhn/` with `train_32x32.mat` and `test_32x32.mat`
- ImageNet-32: `$DATASETS/imagenet32/` with `train_data_batch_*` and `val_data`


Example structure:
```
$DATASETS/
├── cifar10/
└── other_datasets/

$RESULTS/plaintext/
├── cifar10_mlp/
│   └── 20250625_0640_14_lr0.00800_b8/
│       ├── results.txt            # legacy log (deprecated)
│       └── checkpoints/           # see wandb_checkpoints as well
└── other_experiments/
```

## Repo map

```
eoss/
├── training.py                    # main CLI: dataset/model setup, SGD loop, measurements
├── batch_sharpness_scaling.py     # offline W&B run loader to sweep batch sharpness w/ CI
├── compute_finals.py              # legacy experiment driver aggregating sharpness metrics
├── sharpness_gap.py               # script for sharpness-gap investigations on ResNets
├── setup/
│   └── download_datasets.py       # helper to fetch datasets into $DATASETS
├── utils/
│   ├── __init__.py
│   ├── data.py                    # dataset presets, tensor loaders, caching helpers
│   ├── nets.py                    # model zoo (MLP/CNN/ResNet) + init/optimizer presets
│   ├── measure.py                 # batch sharpness, λ_max eigens, GNI, gradient norms
│   ├── frequency.py               # cadence rules for measurements & checkpoints
│   ├── wandb_utils.py             # W&B init/logging, checkpoint I/O, continuation logic
│   ├── naming.py                  # run-name, tags, and note sanitization helpers
│   ├── noise.py                   # stochastic dynamics helpers (SDE, noisy GD)
│   ├── quadratic.py               # quadratic approximations and projection utilities
│   ├── lobpcg.py                  # Hessian eigen solver (LOBPCG) w/ caching tweaks
│   ├── resnet*.py                 # ResNet definitions with/without BatchNorm variants
│   ├── storage.py                 # lightweight persistence utilities
│   └── examples/                  # sample configs/snippets referenced in docs
├── tests/
│   ├── test_batch_sharpness_confidence_interval.py
│   ├── test_compute_eigenvalues.py
│   ├── test_gradient_projection.py
│   ├── test_hessian_trace.py
│   ├── test_wandb_naming.py
│   ├── ...                        # broader coverage for eigensolvers, frequency, checkpoints
│   └── cmd_test.sh                # convenience wrapper to launch targeted pytest runs
├── visualization/
│   ├── img/                       # generated PNGs from visualization scripts
│   ├── plot_results.py            # quick plots for legacy `results.txt` flows
│   ├── template.ipynb             # notebook to pull W&B runs & plot sharpness dynamics
│   └── vis_utils.py               # helpers shared across visualization notebooks
├── notebook_runs/
│   ├── template.ipynb             # canonical notebook for measurement sweeps
│   ├── wandb_checkpoint_resume.ipynb
│   ├── ...                        # exploratory notebooks for sweeps, landscapes, resumes
│   └── toy_models/                # toy setups for analytical checks
├── data_vis/                      # legacy visualization notebooks + polished figures
├── data_vis_ps/                   # paper-ready plots created from sweeps
├── docs/
│   ├── quadratic_approx.md        # notes on quadratic modeling assumptions
│   └── WANDB_CHECKPOINTING_GUIDE.md # wandb checkpoint deep dive and workflows
├── slurm_scripts/                 # example SLURM launchers and helper wrappers
└── slurm/                         # archived SLURM stdout/err logs (cluster provenance)
```

## License

Apache License 2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
