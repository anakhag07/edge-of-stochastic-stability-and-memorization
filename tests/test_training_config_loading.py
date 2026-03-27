import json
import tempfile
from pathlib import Path

import pytest

from utils.training_cli import build_parser, parse_args_with_config


def _write_config(payload) -> str:
    handle = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
    with handle:
        json.dump(payload, handle)
    return handle.name


def test_json_config_sets_parser_defaults():
    config_path = _write_config(
        {
            "__comment": "ignored",
            "training": {"batch": 7, "lr": 0.125, "steps": 11},
            "data": {"dataset": "cifar10", "num_data": 21},
            "run": {"cpu": True, "disable_wandb": True},
        }
    )

    args = parse_args_with_config(build_parser(0.5), ['--config', config_path])

    assert args.batch == 7
    assert args.lr == 0.125
    assert args.steps == 11
    assert args.num_data == 21
    assert args.cpu is True
    assert args.disable_wandb is True


def test_cli_overrides_json_config_values():
    config_path = _write_config(
        {
            "training": {"batch": 7, "lr": 0.125},
            "run": {"cpu": True},
        }
    )

    args = parse_args_with_config(build_parser(0.5), ['--config', config_path, '--batch', '9', '--lr', '0.5'])

    assert args.batch == 9
    assert args.lr == 0.5
    assert args.cpu is True


def test_cli_batch_full_overrides_json_config_value():
    config_path = _write_config(
        {
            "training": {"batch": 7, "lr": 0.125},
        }
    )

    args = parse_args_with_config(build_parser(0.5), ['--config', config_path, '--batch', 'full'])

    assert args.batch == 'full'
    assert args.lr == 0.125


def test_unknown_json_key_raises_helpful_error():
    config_path = _write_config({"training": {"not_a_real_flag": 1}})

    with pytest.raises(ValueError, match="Unknown config key"):
        parse_args_with_config(build_parser(0.5), ['--config', config_path])


def test_invalid_batch_string_is_rejected():
    with pytest.raises(SystemExit):
        parse_args_with_config(build_parser(0.5), ['--batch', 'giant'])


def test_legacy_cli_still_works_without_config():
    args = parse_args_with_config(build_parser(0.5), ['--batch', '5', '--lr', '0.02', '--cpu'])

    assert args.batch == 5
    assert args.lr == 0.02
    assert args.cpu is True


def test_batch_full_parses_without_config():
    args = parse_args_with_config(build_parser(0.5), ['--batch', 'full', '--lr', '0.02'])

    assert args.batch == 'full'
    assert args.lr == 0.02


def test_batch_full_can_come_from_json_config():
    config_path = _write_config(
        {
            "training": {"batch": "full", "lr": 0.125},
        }
    )

    args = parse_args_with_config(build_parser(0.5), ['--config', config_path])

    assert args.batch == 'full'
    assert args.lr == 0.125


def test_sample_smoke_config_parses():
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / 'configs' / 'smoke_train.json'

    args = parse_args_with_config(build_parser(0.5), ['--config', str(config_path)])

    assert args.dataset == 'cifar10'
    assert args.model == 'mlp'
    assert args.loss == 'ce'
    assert args.batch == 4
    assert args.steps == 20
    assert args.lambdamax is True
    assert args.batch_sharpness is True
