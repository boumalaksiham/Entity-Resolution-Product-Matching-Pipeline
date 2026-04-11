"""
Demo v2 — Entity Resolution with Attribute Guard
==================================================
Shows raw similarity, penalty applied, conflicts found, and final score.

Usage:
    python demo.py
"""

import os, sys, json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.siamese_model import EntityResolutionModel


def load_threshold(default=0.75):
    config_path = "models/config.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f).get("optimal_threshold", default)
    return default


def print_result(result: dict):
    match = result["prediction"] == "MATCH"
    bar_length = int(result["final_score"] * 40)
    bar = "█" * bar_length + "░" * (40 - bar_length)

    print("\n" + "─" * 62)
    print(f"  Title 1       : {result['title1']}")
    print(f"  Title 2       : {result['title2']}")
    print(f"  Raw Similarity: {result['raw_similarity']:.4f}")

    if result["conflicts"]:
        print(f"  ⚠️  Conflicts  : {', '.join(result['conflicts'])}")
        print(f"  Penalty       : {result['penalty']} (score reduced)")
    else:
        print(f"  ✅ No conflicts found — no penalty applied")

    print(f"  Final Score   : [{bar}] {result['final_score']:.4f}")
    print(f"  Result        : {'✅ SAME PRODUCT (MATCH)' if match else '❌ DIFFERENT PRODUCT (NO MATCH)'}")
    print("─" * 62)


def run_preset_examples(model):
    print("\n" + "=" * 62)
    print("  PRESET EXAMPLES — showing V2 improvement")
    print("=" * 62)

    presets = [
        # These were WRONG in V1 — should now be fixed
        ("Sony WH-1000XM5 Wireless Headphones Black",
         "Sony WH-1000XM4 Wireless Headphones Black"),
        ("KitchenAid Artisan Stand Mixer 5Qt Empire Red",
         "KitchenAid Artisan Stand Mixer 3.5Qt Empire Red"),
        ("Apple iPhone 14 Pro 256GB Deep Purple",
         "Apple iPhone 14 Pro 512GB Deep Purple"),

        # These should still be correct MATCH
        ("Nike Air Max 90 White Size 10", "nike airmax90 white sz10"),
        ("Sony WH-1000XM5 Wireless Headphones Black",
         "sony wh1000xm5 headphone black bluetooth"),
        ("Dyson V15 Detect Cordless Vacuum Yellow",
         "dyson v15 detect absolute yellow cordless hoover"),
    ]

    for t1, t2 in presets:
        result = model.predict_pair(t1, t2)
        print_result(result)


def interactive_mode(model):
    print("\n" + "=" * 62)
    print("  INTERACTIVE MODE — Type your own product titles")
    print("  (Press Ctrl+C to quit)")
    print("=" * 62)

    while True:
        try:
            print()
            title1 = input("Enter Product Title 1: ").strip()
            if not title1:
                continue
            title2 = input("Enter Product Title 2: ").strip()
            if not title2:
                continue
            result = model.predict_pair(title1, title2)
            print_result(result)
        except KeyboardInterrupt:
            print("\n\nExiting. Goodbye!")
            break


def main():
    print("=" * 62)
    print("  ENTITY RESOLUTION DEMO v2")
    print("  Sentence Transformer + Attribute Guard")
    print("=" * 62)

    threshold = load_threshold()
    model = EntityResolutionModel(threshold=threshold)

    run_preset_examples(model)
    interactive_mode(model)


if __name__ == "__main__":
    main()
