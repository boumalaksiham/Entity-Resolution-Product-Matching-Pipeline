"""
Entity Resolution Model v2 — Siamese Sentence Transformer + Attribute Guard
=============================================================================
IMPROVEMENT OVER V1:
    V1 problem: "Sony XM5" vs "Sony XM4" scored 0.97 (WRONG MATCH)
                "5Qt" vs "3.5Qt" scored 0.97 (WRONG MATCH)

    Root cause: Sentence transformers encode holistic meaning.
                A single digit difference gets drowned out by all the
                shared words around it.

    Fix: Two-stage pipeline
        Stage 1: Sentence transformer computes semantic similarity (same as before)
        Stage 2: Attribute Guard extracts key attributes (model numbers, sizes,
                 storage, etc.) and PENALIZES the score if critical ones differ

    Final score = semantic_similarity * attribute_penalty
                  (penalty = 1.0 if all attributes match, < 1.0 if they conflict)
"""

import re
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class AttributeExtractor:
    """
    Extracts key product attributes from a title using regex patterns.

    Why regex and not ML here?
        For numbers, model codes, and sizes — regex is faster, more
        reliable, and easier to explain in an interview than a neural net.
        ML is used where meaning matters; regex where structure matters.
    """

    STORAGE_PATTERN    = re.compile(r'\b(\d+(?:\.\d+)?)\s*(gb|tb)\b', re.IGNORECASE)
    SIZE_PATTERN       = re.compile(r'\b(\d+(?:\.\d+)?)\s*(qt|oz|l|liter|litre|quart)\b', re.IGNORECASE)
    SCREEN_PATTERN     = re.compile(r'\b(\d+(?:\.\d+)?)\s*(?:in|inch|"|\'\'|-inch)\b', re.IGNORECASE)
    MODEL_CODE_PATTERN = re.compile(r'\b([A-Z]{1,4}\d{1,4}(?:[A-Z]{0,3}\d{0,3})?)\b')
    SHOE_SIZE_PATTERN  = re.compile(r'\b(?:size|sz)\s*(\d+(?:\.\d+)?)\b', re.IGNORECASE)
    GEN_PATTERN        = re.compile(r'\b(\d+)(?:st|nd|rd|th)?\s*gen(?:eration)?\b', re.IGNORECASE)

    def extract(self, title: str) -> dict:
        title_upper = title.upper()
        return {
            "storage":     [m.group(1) + m.group(2).upper() for m in self.STORAGE_PATTERN.finditer(title)],
            "sizes":       [m.group(1) + m.group(2).lower() for m in self.SIZE_PATTERN.finditer(title)],
            "screens":     [m.group(1) for m in self.SCREEN_PATTERN.finditer(title)],
            "model_codes": self.MODEL_CODE_PATTERN.findall(title_upper),
            "shoe_sizes":  [m.group(1) for m in self.SHOE_SIZE_PATTERN.finditer(title)],
            "generations": [m.group(1) for m in self.GEN_PATTERN.finditer(title)],
        }


class AttributePenaltyCalculator:
    """
    Compares extracted attributes from two titles.
    Returns a penalty multiplier between 0.0 and 1.0.

        1.0 = no conflicts found → don't penalize
        0.5 = one critical conflict → cut score in half
        0.3 = multiple conflicts → heavily penalize
    """

    PENALTIES = {
        "storage":     0.40,
        "sizes":       0.40,
        "model_codes": 0.50,
        "shoe_sizes":  0.45,
        "screens":     0.35,
        "generations": 0.45,
    }

    def compute_penalty(self, attrs1: dict, attrs2: dict) -> tuple:
        penalty = 1.0
        conflicts = []

        for attr_type, penalty_amount in self.PENALTIES.items():
            vals1 = set(attrs1.get(attr_type, []))
            vals2 = set(attrs2.get(attr_type, []))

            if vals1 and vals2 and vals1.isdisjoint(vals2):
                penalty *= penalty_amount
                conflicts.append(f"{attr_type}: {vals1} vs {vals2}")

        return round(penalty, 4), conflicts


class EntityResolutionModel:
    """
    V2: Sentence Transformer + Attribute Guard

    Pipeline:
        title1, title2
            → [Sentence Transformer]  → raw semantic similarity
            → [Attribute Extractor]   → attrs1, attrs2
            → [Penalty Calculator]    → penalty
            → final_score = raw_similarity * penalty
            → MATCH if final_score >= threshold
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.75):
        print(f"Loading model: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.extractor = AttributeExtractor()
        self.penalty_calc = AttributePenaltyCalculator()
        print("Model loaded!")

    def encode(self, titles: list) -> np.ndarray:
        return self.model.encode(titles, convert_to_numpy=True)

    def predict_pair(self, title1: str, title2: str) -> dict:
        # Stage 1: semantic similarity
        embeddings = self.encode([title1, title2])
        raw_sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])

        # Stage 2: attribute penalty
        attrs1 = self.extractor.extract(title1)
        attrs2 = self.extractor.extract(title2)
        penalty, conflicts = self.penalty_calc.compute_penalty(attrs1, attrs2)

        final_score = raw_sim * penalty
        prediction = "MATCH" if final_score >= self.threshold else "NO MATCH"

        return {
            "title1": title1,
            "title2": title2,
            "raw_similarity": round(raw_sim, 4),
            "penalty": penalty,
            "conflicts": conflicts,
            "final_score": round(final_score, 4),
            "prediction": prediction,
            "threshold_used": self.threshold,
        }

    def predict_batch(self, pairs: list) -> list:
        titles1 = [p[0] for p in pairs]
        titles2 = [p[1] for p in pairs]
        emb1 = self.encode(titles1)
        emb2 = self.encode(titles2)

        results = []
        for i, (t1, t2) in enumerate(pairs):
            raw_sim = float(cosine_similarity([emb1[i]], [emb2[i]])[0][0])
            attrs1 = self.extractor.extract(t1)
            attrs2 = self.extractor.extract(t2)
            penalty, conflicts = self.penalty_calc.compute_penalty(attrs1, attrs2)
            final_score = raw_sim * penalty

            results.append({
                "title1": t1,
                "title2": t2,
                "raw_similarity": round(raw_sim, 4),
                "penalty": penalty,
                "conflicts": conflicts,
                "final_score": round(final_score, 4),
                "prediction": "MATCH" if final_score >= self.threshold else "NO MATCH",
                "threshold_used": self.threshold,
            })

        return results
