"""
Training & Evaluation Script
=============================
This script:
    1. Loads the product pair dataset
    2. Generates embeddings using the Sentence Transformer
    3. Evaluates the model using standard ML metrics
    4. Finds the OPTIMAL threshold (the cutoff score for MATCH vs NO MATCH)
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
from models.siamese_model import EntityResolutionModel


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


def evaluate(model: EntityResolutionModel, df: pd.DataFrame, threshold: float) -> dict:
    """
    Runs model predictions on the full dataset and prints a performance report.
    
    Metrics explained:
        Precision: Of all pairs we predicted as MATCH, how many truly were?
        Recall:    Of all true matches, how many did we catch?
        F1 Score:  Harmonic mean of precision and recall (balanced metric)
        AUC-ROC:   Overall ability to separate matches from non-matches (0.5=random, 1.0=perfect)
    """
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)

    # Step 1: Encode all titles
    print("Generating embeddings...")
    emb1 = model.encode(df["title1"].tolist())
    emb2 = model.encode(df["title2"].tolist())

    # Step 2: Compute similarity scores for every pair
    similarities = np.array([
        cosine_similarity([emb1[i]], [emb2[i]])[0][0]
        for i in range(len(df))
    ])

    labels = df["label"].values

    # Step 3: Find optimal threshold
    optimal_threshold = find_optimal_threshold(similarities, labels)
    print(f"\nOptimal threshold found: {optimal_threshold}")
    print(f"(Using this instead of default {threshold})")

    # Step 4: Make predictions with optimal threshold
    predictions = (similarities >= optimal_threshold).astype(int)

    # Step 5: Print full report
    print("\nClassification Report:")
    print("-" * 40)
    print(classification_report(labels, predictions, target_names=["NO MATCH", "MATCH"]))

    print("Confusion Matrix:")
    print("-" * 40)
    cm = confusion_matrix(labels, predictions)
    print(f"              Predicted NO MATCH  |  Predicted MATCH")
    print(f"Actual NO MATCH:      {cm[0][0]:>4}        |      {cm[0][1]:>4}")
    print(f"Actual MATCH:         {cm[1][0]:>4}        |      {cm[1][1]:>4}")

    auc = roc_auc_score(labels, similarities)
    print(f"\nAUC-ROC Score: {auc:.4f}  (1.0 = perfect, 0.5 = random guessing)")

    return {"optimal_threshold": optimal_threshold, "auc_roc": auc}


def run_example_predictions(model: EntityResolutionModel, threshold: float):
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
