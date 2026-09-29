from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import factory, world_semantics


class Run:
    def __init__(self, review, warning=False):
        self.files = {world_semantics.REVIEW_ARTIFACT: review}
        if warning:
            self.files[world_semantics.WARNING_ARTIFACT] = {"status": "warning"}
    def has(self, name):
        return name in self.files
    def read(self, name):
        return self.files[name]


class DisclosureWorldReviewRefreshTests(unittest.TestCase):
    def test_exact_execution_warning_still_requires_fresh_review(self):
        run = Run({"status": "error"}, warning=True)
        with patch.object(world_semantics, "validate_review", return_value=[]):
            self.assertFalse(factory._release_current_world_review(run, {}, object(), {}))

    def test_current_passed_review_can_be_reused(self):
        run = Run({"status": "passed"})
        with patch.object(world_semantics, "validate_review", return_value=[]):
            self.assertTrue(factory._release_current_world_review(run, {}, object(), {}))

    def test_invalid_or_errored_review_requires_refresh(self):
        for review, errors in (({"status": "error"}, []), ({"status": "passed"}, [{"code": "stale"}])):
            with self.subTest(review=review, errors=errors), \
                    patch.object(world_semantics, "validate_review", return_value=errors):
                self.assertFalse(factory._release_current_world_review(Run(review), {}, object(), {}))


if __name__ == "__main__":
    unittest.main()
