## Conventions

- Prefer small, focused changes with clear doc updates.
- Keep the simulation logic deterministic when a seed is set.

## Testing Strategy

- Primary test is a short, deterministic `training.py` smoke run.
- Use explicit seeds, CPU, and `--disable-wandb` to keep runs fast and repeatable.
- Avoid plotting, filesystem I/O, or long-running simulations.
- If adding unit tests, keep them small and in the existing flat `tests/` layout.

## Plan Mode

- When in plan mode, propose the unit-test structure and ask whether it is appropriate before implementing tests.

## Required Maintenance

- Update `README.md` and `requirements.txt` if necessary after any changes.
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
