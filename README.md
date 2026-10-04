# Entity Resolution — Product Matching

A product-title matching prototype that combines sentence embeddings with explicit checks for conflicting product attributes. Similar wording alone can hide differences such as model number, capacity, storage, or generation.

## A concrete matching problem

`Apple iPhone 14 Pro 256GB Deep Purple` and `Apple iPhone 14 Pro 512GB Deep Purple` describe closely related products, but different storage variants. The interactive demo includes this pair to expose a limitation of semantic similarity alone.

The guard checks attributes only when both titles provide values. An explicit storage conflict multiplies the score by **0.40**; other detected conflicts can reduce it further. Missing attributes are not proof of agreement. Inspect the raw similarity, extracted conflicts, multiplier, and final score together when reviewing a decision.

**Design choice:** retain a pretrained semantic encoder for wording differences and add inspectable rules for selected identifiers. This makes the source of a score reduction visible, while introducing dependence on regex coverage and manually chosen penalties.

## Method

1. Encode both titles with `all-MiniLM-L6-v2`.
2. Compute cosine similarity.
3. Extract storage, volume, screen size, model codes, shoe sizes, and generations with regular expressions.
4. Apply multiplicative penalties when both titles contain conflicting values.
5. Compare the guarded score with a calibrated threshold.

The score is a similarity/heuristic combination, **not a calibrated match probability**. The pretrained encoder is not fine-tuned by `train_evaluate.py`.

## Data and evaluation

[data/dataset_builder.py](data/dataset_builder.py) constructs **40 curated pairs** with binary match labels. The evaluator uses a reproducible stratified pair split: **24 calibration pairs and 16 evaluation pairs**, with seed 42. Baseline and guarded-score thresholds are selected independently on calibration data, then frozen for evaluation.

The historical 78% to 92%+ statement is not retained as a verified result: the earlier evaluator reused calibration data and bypassed the guard. Run the current script to generate current metrics. Product-related pairs may appear in both splits; this is not product-disjoint validation or evidence of production performance.

## Setup

Use Python 3.10 or 3.11 for the repository's pinned PyTorch 2.2.2 environment. Run commands from the repository root. The first model load downloads pretrained weights; an internet connection is needed unless the model cache is populated. A GPU is not required by the current scripts. Runtime depends on hardware and is not benchmarked here.

```bash
git clone https://github.com/boumalaksiham/Entity-Resolution-Product-Matching-Pipeline.git
cd Entity-Resolution-Product-Matching-Pipeline
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. If installation reports an unsupported wheel, check Python version and architecture before changing the pinned environment.

## Run

```bash
python train_evaluate.py
python demo.py
```

The evaluator prints classification reports and writes:

| Artifact | Contents |
|---|---|
| `results/evaluation.json` | Split indices, frozen thresholds, classification reports, confusion matrices, and ROC-AUC for both scorers |
| `models/config.json` | Guarded-score threshold used by the interactive demo |

Running evaluation overwrites these generated files. Preserve outputs with the commit, environment, and run date if citing results.

## Repository map

| File | Purpose |
|---|---|
| [data/dataset_builder.py](data/dataset_builder.py) | Curated labeled pairs |
| [models/siamese_model.py](models/siamese_model.py) | Encoder wrapper, attribute extraction, penalties, and prediction |
| [train_evaluate.py](train_evaluate.py) | Calibration and held-out evaluation |
| [demo.py](demo.py) | Interactive predictions and conflict breakdown |
| [tests/test_evaluation.py](tests/test_evaluation.py) | Evaluation regression check |

## Verification

```bash
python -m unittest discover -s tests -v
```

The regression check uses a fake scorer to verify split separation, guarded-score use, and JSON export. It does not measure encoder performance or download model weights.

## Limitations and next steps

Regex patterns may overmatch identifiers and miss aliases or equivalent units. Multiplicative penalties are manually specified. The dataset is small and curated, and thresholds may not transfer to another catalog. Next steps are product-disjoint evaluation, larger labeled data, unit normalization, precision/recall tradeoff analysis, and score calibration.
