"""
Training & Evaluation Script
=============================
This script:
    1. Loads the product pair dataset
    2. Generates embeddings using the Sentence Transformer
    3. Calibrates thresholds on a separate stratified split
    4. Evaluates baseline and attribute guard with frozen thresholds
    5. Saves the best threshold to disk

Run this script first before using the demo.

Usage:
    python train_evaluate.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Add project root to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_builder import build_dataframe


def find_optimal_threshold(similarities: np.ndarray, labels: np.ndarray) -> float:
    """
    Tries many threshold values and picks the one that gives the best F1 score.
    
    Why do we need this?
        If we just use 0.5 as threshold, we might miss many matches.
        Different datasets need different thresholds.
        This function finds the best one automatically.
    """
    best_threshold = 0.5
    best_f1 = 0.0

    # Try thresholds from 0.3 to 0.95 in small steps
    for threshold in np.arange(0.3, 0.95, 0.01):
        predictions = (similarities >= threshold).astype(int)

        # True Positives, False Positives, False Negatives
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return round(float(best_threshold), 2)


def evaluate(model, df: pd.DataFrame, threshold: float = 0.75) -> dict:
    """Calibrate on one split; report frozen thresholds on held-out pairs.

    This small pair-level split is a demonstration, not product-disjoint validation.
    """
    calibration, test = train_test_split(
        df, test_size=0.4, random_state=42, stratify=df["label"]
    )
    def scores(frame):
        pairs = list(zip(frame["title1"], frame["title2"]))
        results = model.predict_batch(pairs)
        return {key: np.array([r[key] for r in results])
                for key in ("raw_similarity", "final_score")}

    calibration_scores = scores(calibration)
    frozen = {key: find_optimal_threshold(values, calibration["label"].to_numpy())
              for key, values in calibration_scores.items()}
    test_scores = scores(test)
    labels = test["label"].to_numpy()
    report = {"random_state": 42, "calibration_pairs": len(calibration),
              "test_pairs": len(test), "calibration_indices": calibration.index.tolist(),
              "test_indices": test.index.tolist(), "models": {}}
    for name, key in (("semantic_baseline", "raw_similarity"),
                      ("attribute_guard", "final_score")):
        predictions = (test_scores[key] >= frozen[key]).astype(int)
        metrics = classification_report(labels, predictions, labels=[0, 1],
                                        target_names=["NO MATCH", "MATCH"],
                                        output_dict=True, zero_division=0)
        report["models"][name] = {
            "threshold": frozen[key], "classification_report": metrics,
            "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
            "auc_roc": float(roc_auc_score(labels, test_scores[key]))}
        print(f"{name}: threshold calibrated on {len(calibration)} pairs; "
              f"evaluated on {len(test)} held-out pairs")
        print(classification_report(labels, predictions, labels=[0, 1],
                                    target_names=["NO MATCH", "MATCH"], zero_division=0))
    os.makedirs("results", exist_ok=True)
    with open("results/evaluation.json", "w") as handle:
        json.dump(report, handle, indent=2)
    return {"optimal_threshold": frozen["final_score"],
            "auc_roc": report["models"]["attribute_guard"]["auc_roc"],
            "report": report}


def run_example_predictions(model, threshold: float):
    """
    Shows the model working on brand new examples it has never seen.
    This is what you'd demo in an interview.
    """
    print("\n" + "="*60)
    print("LIVE PREDICTIONS ON NEW EXAMPLES")
    print("="*60)

    new_pairs = [
        # Should be MATCH
        ("Nike Air Force 1 Low White Men's Size 9", "nike air force1 low white sz9 mens"),
        ("Apple MacBook Pro 14 M3 Pro 512GB Space Black", "MacBook Pro 14-inch M3 Pro chip 512GB Space Black"),
        ("Instant Pot Duo 7-in-1 Electric Pressure Cooker 6Qt", "Instant Pot 6 Quart 7-in-1 Duo Pressure Cooker"),

        # Should be NO MATCH
        ("Nike Air Force 1 Low White Men's Size 9", "Nike Air Force 1 High White Men's Size 9"),
        ("Apple MacBook Pro 14 M3 Pro 512GB Space Black", "Apple MacBook Air 15 M3 512GB Space Gray"),
        ("Instant Pot Duo 7-in-1 Electric Pressure Cooker 6Qt", "Ninja Foodi 9-in-1 Pressure Cooker 6.5Qt"),
    ]

    model.threshold = threshold
    results = model.predict_batch(new_pairs)

    for r in results:
        emoji = "✅" if r["prediction"] == "MATCH" else "❌"
        print(f"\n{emoji} {r['prediction']}  (similarity: {r['similarity']})")
        print(f"   Title 1: {r['title1']}")
        print(f"   Title 2: {r['title2']}")


def main():
    # 1. Build dataset
    print("Building dataset...")
    df = build_dataframe()
    print(f"Dataset: {len(df)} pairs ({df['label'].sum()} matches, {(df['label']==0).sum()} non-matches)")

    from models.siamese_model import EntityResolutionModel

    # 2. Load model (downloads ~90MB model on first run)
    model = EntityResolutionModel(threshold=0.75)

    # 3. Evaluate and find optimal threshold
    results = evaluate(model, df, threshold=0.75)
    optimal_threshold = results["optimal_threshold"]

    # 4. Save optimal threshold so demo.py can use it
    os.makedirs("models", exist_ok=True)
    with open("models/config.json", "w") as f:
        json.dump({"optimal_threshold": optimal_threshold}, f)
    print(f"\nSaved optimal threshold ({optimal_threshold}) to models/config.json")

    # 5. Show live predictions
    run_example_predictions(model, optimal_threshold)

    print("\n✅ Training & evaluation complete!")
    print("   Now run: python demo.py  to try your own product titles.")


if __name__ == "__main__":
    main()
