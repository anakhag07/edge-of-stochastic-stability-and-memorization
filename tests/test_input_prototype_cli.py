import pytest

from utils.training_cli import build_parser, parse_args_with_config


def test_parser_accepts_simplified_input_prototype_flags():
    args = parse_args_with_config(
        build_parser(0.5),
        [
            "--input-prototype-source",
            "generate",
            "--input-prototypes-mode",
            "val",
            "--input-boundary",
            "5",
            "--input-inliers",
            "4",
            "--input-x-outliers",
            "3",
            "--input-y-outliers",
            "2",
        ],
    )

    assert args.input_prototype_source == "generate"
    assert args.input_prototypes_mode == "val"
    assert args.input_boundary == 5
    assert args.input_inliers == 4
    assert args.input_x_outliers == 3
    assert args.input_y_outliers == 2


def test_input_prototype_mode_defaults_to_train():
    args = parse_args_with_config(build_parser(0.5), [])

    assert args.input_prototypes_mode == "train"


@pytest.mark.parametrize(
    "legacy_flag",
    [
        "--train-input-prototypes",
        "--test-input-prototypes",
        "--track-input-prototypes",
        "--input-prototypes-holdout-boundary-count",
        "--input-prototype-holdout-per-class",
        "--train-input-x-outliers",
        "--per-sample",
        "--knn-outliers",
        "--feature-prototypes",
        "--track-feature-prototypes-from",
        "--track-knn-outliers-from",
        "--lmax-decay",
    ],
)
def test_legacy_input_prototype_flags_are_removed(legacy_flag):
    parser = build_parser(0.5)
    with pytest.raises(SystemExit):
        parser.parse_args([legacy_flag, "1"])
