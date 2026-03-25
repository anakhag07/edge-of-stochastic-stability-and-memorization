# Agent Instructions

Project context: edge-of-stochastic-stability and memorization experiments. Primary entry point is `training.py`.

## Core Working Rules

- Prefer small, focused changes with clear doc updates.
- Keep simulation logic deterministic when a seed is set.
- Do not rely on prior chat context as source of truth; verify behavior from the repository state.

## Session Workflow

### Start of Session
Before editing code:

1. Read `AGENTS.md`.
2. Read `README.md`.
3. Inspect relevant entry points and neighboring files.
4. Check current test coverage in `tests/` for the feature area.
5. Confirm branch state and task scope.

### End of Session
Before finishing a build session:

1. Run relevant tests.
2. Run the deterministic `training.py` smoke test when behavior changes.
3. Update `README.md` if behavior, structure, configuration, or usage changed.
4. Update `AGENTS.md` if durable workflow or organization knowledge changed.
5. Leave a concise handoff summary.

## Modes

### Plan Mode
- Do not edit code.
- Propose implementation approach, target files, and unit-test structure.
- Ask whether the proposed test structure is appropriate before implementing tests.
- Call out expected `README.md` and `AGENTS.md` updates.

### Build Mode
- Make focused code changes.
- Keep docs in sync with implementation.
- Run validation commands after changes.

## Testing Strategy

- Primary test is a short, deterministic `training.py` smoke run.
- Use explicit seeds, CPU, and `--disable-wandb` to keep runs fast and repeatable.
- Avoid plotting, filesystem I/O, or long-running simulations.
- Add or update small, focused tests in the existing flat `tests/` layout.
- Prefer deterministic tests with explicit seeds.

## Required Maintenance

- Update `README.md` and `requirements.txt` when necessary after changes.
- Re-export public APIs in package `__init__.py` files when modules are added or moved.

## Lessons Learned

- Write down lessons from mistakes made to avoid repeating them.

## Validation

Run the smoke test after changes:

```bash
conda activate eoss
python training.py --dataset cifar10 --model mlp --loss ce \
  --batch 4 --lr 0.05 --steps 20 --num-data 64 \
  --dataset-seed 1 --init-seed 1 --disable-wandb --cpu
```

Config-based equivalent:

```bash
conda activate eoss
python training.py --config configs/smoke_train.json
```
