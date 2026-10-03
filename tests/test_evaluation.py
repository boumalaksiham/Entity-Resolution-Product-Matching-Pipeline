import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from train_evaluate import evaluate


class EvaluationTests(unittest.TestCase):
    def test_calibration_is_disjoint_and_guard_scores_are_used(self):
        frame = pd.DataFrame({"title1": [str(i) for i in range(40)],
                              "title2": ["other"] * 40,
                              "label": [i % 2 for i in range(40)]})
        class FakeModel:
            def predict_batch(self, pairs):
                return [{"raw_similarity": 0.9 if int(a) % 2 else 0.8,
                         "final_score": 0.9 if int(a) % 2 else 0.1}
                        for a, _ in pairs]
        seen = []
        def calibrate(scores, labels):
            seen.append(len(scores))
            return 0.5
        previous = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                with patch("train_evaluate.find_optimal_threshold", side_effect=calibrate):
                    result = evaluate(FakeModel(), frame)
                report = result["report"]
                self.assertEqual(seen, [24, 24])
                self.assertFalse(set(report["calibration_indices"]) & set(report["test_indices"]))
                self.assertEqual(report["models"]["attribute_guard"]["classification_report"]["accuracy"], 1.0)
                self.assertLess(report["models"]["semantic_baseline"]["classification_report"]["accuracy"], 1.0)
                self.assertEqual(json.loads(Path("results/evaluation.json").read_text()), report)
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
