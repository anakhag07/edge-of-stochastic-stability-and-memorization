import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = REPO_ROOT / "training.py"


class TestInputPrototypeHoldoutFlags(unittest.TestCase):
    def test_removed_string_holdout_flag(self):
        content = TRAINING_SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("--input-prototypes-holdout-count", content)
        self.assertNotIn("input_prototypes_holdout_count", content)

    def test_explicit_holdout_flags_present(self):
        content = TRAINING_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--input-prototypes-holdout-boundary-count", content)
        self.assertIn("--input-prototypes-holdout-inliers-count", content)
        self.assertIn("--input-prototypes-holdout-x-outlier-count", content)
        self.assertIn("--input-prototypes-holdout-y-outlier-count", content)


if __name__ == "__main__":
    unittest.main()
