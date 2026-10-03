# Entity Resolution — Product Matching Pipeline

## Evaluation scope

The historical 78% to 92%+ claim below is not a verified held-out result. The original evaluation tuned the threshold on the same 40 curated pairs it scored and evaluated raw similarity without applying the attribute guard. The revised script calibrates thresholds on a separate split, evaluates both raw and guarded scores on held-out pairs, and writes a machine-readable report. Run it to obtain new results; the historical numbers must not be used as the revised experiment's results. Pair-level splitting still allows related products in both splits, so a larger product-disjoint benchmark remains necessary.

### Semantic Similarity + Attribute Conflict Detection

> Built as part of an e-commerce ML portfolio targeting applied research roles at companies like eBay, Amazon, and Shopify.

---

## Overview

This project solves a core problem in e-commerce: **millions of sellers list the same product with different titles, abbreviations, and formats**. Entity Resolution is the task of identifying which listings refer to the same real-world product.

| Title 1 | Title 2 | Prediction |
|---|---|---|
| `Nike Air Max 90 White Men's Size 10` | `nike airmax90 white sz10 mens` | ✅ MATCH |
| `Sony WH-1000XM5 Headphones Black` | `Sony WH-1000XM4 Headphones Black` | ❌ NO MATCH |
| `Apple iPhone 14 Pro 256GB Deep Purple` | `iphone 14 pro 256gb deep purple unlocked` | ✅ MATCH |
| `KitchenAid Artisan Stand Mixer 5Qt Red` | `KitchenAid Artisan Stand Mixer 3.5Qt Red` | ❌ NO MATCH |

---

## The Problem with Naive Approaches

**String matching fails** on abbreviations, typos, and word order:
- `"airmax90"` ≠ `"Air Max 90"` by string comparison
- `"sz10"` ≠ `"Size 10"` by string comparison

**Vanilla sentence transformers fail** on single-attribute differences:
- `"Sony XM5 Headphones Black"` vs `"Sony XM4 Headphones Black"` → cosine similarity: **0.97** (incorrectly flagged as MATCH)
- `"KitchenAid 5Qt"` vs `"KitchenAid 3.5Qt"` → cosine similarity: **0.97** (incorrectly flagged as MATCH)

The root cause: transformers encode *holistic meaning*. A single digit difference (XM**5** vs XM**4**) gets drowned out by all the surrounding shared words.

---

## Solution — Two-Stage Pipeline

```
Title 1 ──────────────────────────────────────────────────────┐
                                                               ▼
                                              [Sentence Transformer]
                                               all-MiniLM-L6-v2
                                              [Sentence Transformer]
                                                               │
Title 2 ──────────────────────────────────────────────────────┘
                                                               │
                                                    raw_similarity (0–1)
                                                               │
                                                               ▼
Title 1 ──► [Attribute Extractor] ──► attrs1 ──┐
                                               ├──► [Penalty Calculator]
Title 2 ──► [Attribute Extractor] ──► attrs2 ──┘          │
                                                     penalty (0–1)
                                                           │
                                                           ▼
                                        final_score = raw_similarity × penalty
                                                           │
                                               ┌───────────┴───────────┐
                                          score ≥ 0.75             score < 0.75
                                               │                       │
                                           ✅ MATCH               ❌ NO MATCH
```

### Stage 1 — Semantic Similarity (Sentence Transformer)
- Encodes both titles into 384-dimensional dense vectors
- Computes cosine similarity between the two vectors
- Captures semantic meaning: `"sz10"` ≈ `"Size 10"`, `"hoover"` ≈ `"vacuum"`
- Model: `all-MiniLM-L6-v2` — 6-layer transformer, 22M parameters, CPU-friendly

### Stage 2 — Attribute Guard (Regex + Penalty)
- Extracts structured attributes from each title using regex patterns
- Attributes tracked: storage (GB/TB), volume/size (Qt/oz), screen size (in), model codes (XM5, V15), shoe sizes, generation numbers
- If both titles contain the **same attribute type** with **different values** → conflict detected → score penalized
- Penalty is multiplicative: one conflict = ×0.40–0.50, multiple conflicts compound

**Why regex for attributes and not ML?**
For structured numerical attributes, regex is faster, more reliable, and perfectly accurate. ML is reserved for semantic understanding where context matters. Using the right tool for each sub-problem is a deliberate design decision.

---

## Results

| Test Case | Raw Similarity | Penalty | Final Score | Prediction | Correct? |
|---|---|---|---|---|---|
| Nike Air Max 90 vs nike airmax90 | 0.84 | 1.00 | 0.84 | ✅ MATCH | ✅ |
| Sony XM5 vs Sony XM4 | 0.97 | 0.50 | 0.49 | ❌ NO MATCH | ✅ |
| KitchenAid 5Qt vs 3.5Qt | 0.97 | 0.40 | 0.39 | ❌ NO MATCH | ✅ |
| iPhone 14 Pro 256GB vs 512GB | 0.98 | 0.40 | 0.39 | ❌ NO MATCH | ✅ |
| AirPods Pro 2nd Gen vs airpods pro 2 | 0.88 | 1.00 | 0.88 | ✅ MATCH | ✅ |
| Dyson V15 vs dyson v15 detect hoover | 0.93 | 1.00 | 0.93 | ✅ MATCH | ✅ |

**Historical demonstration claim (not independently validated): 78% baseline → 92%+ with attribute guard. See Evaluation scope above; rerun the revised script for held-out results.**

---

## Project Structure

```
entity_resolution/
├── data/
│   ├── dataset_builder.py     ← 40 labeled product pairs (20 match, 20 no-match)
│   └── __init__.py
├── models/
│   ├── siamese_model.py       ← Core model: SentenceTransformer + AttributeExtractor
│   │                             + AttributePenaltyCalculator + EntityResolutionModel
│   ├── config.json            ← Saved optimal threshold from training
│   └── __init__.py
├── train_evaluate.py          ← Evaluation script: metrics, confusion matrix, AUC-ROC
├── demo.py                    ← Interactive CLI demo with full prediction breakdown
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### Requirements
- Python 3.10+
- No GPU required — fully CPU compatible

### Installation
```bash
# Clone and navigate to project
git clone https://github.com/boumalaksiham/Entity-Resolution-Product-Matching-Pipeline.git
cd Entity-Resolution-Product-Matching-Pipeline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# Install dependencies (pin numpy<2 for torch compatibility)
pip install "numpy<2" torch==2.2.2 transformers==4.40.0 \
    sentence-transformers==2.7.0 pandas==2.2.2 \
    scikit-learn==1.4.2 matplotlib seaborn
```

### Run Evaluation
```bash
python train_evaluate.py
```
Calibrates baseline and guarded-score thresholds on 24 pairs and evaluates frozen thresholds on 16 held-out pairs. Writes results/evaluation.json with metrics and split indices. This small pair-level split is not a product-disjoint benchmark.

### Run Interactive Demo
```bash
python demo.py
```
Enter any two product titles and see the full prediction breakdown — raw similarity, conflicts detected, penalty applied, and final score.

---

## Key Design Decisions

**1. Shared encoder, not two separate models**
Both titles are encoded by the same sentence transformer. This ensures the vector space is consistent — distances are meaningful because both embeddings live in the same semantic space.

**2. Multiplicative penalty, not hard rules**
The attribute penalty multiplies the similarity score rather than overriding it. This means a conflict *lowers* confidence but doesn't override strong semantic evidence. Edge cases (e.g. size mentioned in one title but not the other) are handled gracefully — we only penalize when *both* titles have conflicting values for the same attribute.

**3. Threshold optimization**
Rather than using a fixed threshold, `train_evaluate.py` sweeps thresholds from 0.30 to 0.95 and selects the one maximizing F1 score on the evaluation set.

**4. Separation of concerns**
`AttributeExtractor`, `AttributePenaltyCalculator`, and `EntityResolutionModel` are separate classes. This makes each component independently testable and easy to extend — for example, adding a new attribute type only requires a new regex in `AttributeExtractor`.

---

## Limitations & Future Work

**Current limitations:**
- The sentence transformer is not fine-tuned on product data — it uses general-purpose language understanding. A model fine-tuned on (title1, title2, label) triplets from real eBay/Amazon data would improve recall significantly.
- Collector slang and brand-specific terminology (e.g. "chicago" = Jordan 1 Black/White colorway) is not handled.
- Model code regex may over-match on non-model tokens in some edge cases.

**Extensions for production scale:**
- **Fine-tuning**: Use `sentence-transformers` `fit()` with labeled product pairs to adapt embeddings to e-commerce vocabulary
- **FAISS indexing**: Replace brute-force pairwise comparison with approximate nearest neighbor search for sub-millisecond lookup across millions of listings
- **Alias dictionary**: Maintain a lookup table for brand-specific slang (colorway names, product nicknames) as a preprocessing step
- **Confidence calibration**: Platt scaling to convert raw scores to calibrated probabilities for downstream use

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `sentence-transformers` | 2.7.0 | Semantic title encoding |
| `torch` | 2.2.2 | Transformer backend |
| `transformers` | 4.40.0 | HuggingFace model hub |
| `scikit-learn` | 1.4.2 | Evaluation metrics |
| `pandas` | 2.2.2 | Dataset management |
| `numpy` | <2.0 | Numerical operations |

---

## Related Work

This pipeline is inspired by real-world entity resolution systems described in:
- *"Product Matching in E-commerce using Deep Learning"* — common industry approach
- Facebook AI's **FAISS** library for scalable similarity search
- The **WDC Product Matching** benchmark dataset (webdatacommons.org)

For a production-grade version of this system, the next steps would be training on the WDC dataset (~26M product pairs) and deploying with FAISS for real-time matching.

---

*This project is part of a 4-project ML portfolio covering Entity Resolution, Hierarchical Taxonomy Classification, Named Entity Recognition, and Image Quality Scoring for e-commerce applications.*
## Evaluation regression check

Run `python -m unittest discover -s tests -v` from the repository root. The check uses a fake scorer to verify split separation and application of guarded scores; it does not download a model or claim model performance.
