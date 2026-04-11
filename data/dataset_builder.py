"""
Dataset Builder
===============
Creates a labeled dataset of product title pairs for training and evaluation.

Each row has:
    - title1: first product listing title
    - title2: second product listing title  
    - label:  1 = same product, 0 = different product

In a real job scenario, this data would come from eBay's internal database.
Here we build a realistic synthetic + structured dataset to demonstrate the concept.
"""

import pandas as pd
import random

# ── POSITIVE PAIRS (label = 1): Same product, written differently ────────────
# This simulates how different sellers write the same product title

POSITIVE_PAIRS = [
    # Nike shoes — same shoe, different formatting
    ("Nike Air Max 90 White Men's Size 10", "nike airmax 90 white sz10 mens"),
    ("Nike Air Max 90 White Men's Size 10", "Nike AM90 – White – Men's US 10"),
    ("Apple iPhone 14 Pro 256GB Deep Purple Unlocked", "iPhone 14 Pro 256 GB Deep Purple Factory Unlocked"),
    ("Apple iPhone 14 Pro 256GB Deep Purple Unlocked", "APPLE iphone14pro 256gb deeppurple unlocked"),
    ("Samsung Galaxy S23 Ultra 5G 512GB Phantom Black", "Samsung S23 Ultra 512GB 5G Black Unlocked"),
    ("Samsung Galaxy S23 Ultra 5G 512GB Phantom Black", "Galaxy S23 Ultra Phantom Black 512 GB 5G"),
    ("Sony WH-1000XM5 Wireless Noise Cancelling Headphones Black", "Sony WH1000XM5 BT Headphone Black Noise Canceling"),
    ("Sony WH-1000XM5 Wireless Noise Cancelling Headphones Black", "Sony wh-1000xm5 wireless headphones (black)"),
    ("Canon EOS R5 Mirrorless Camera Body Only", "Canon EOS R5 Body – Mirrorless Digital Camera"),
    ("Canon EOS R5 Mirrorless Camera Body Only", "canon eosr5 mirrorless camera body"),
    ("Levi's 501 Original Fit Jeans Men's 32x32 Dark Stonewash", "Levis 501 jeans mens dark stonewash 32 32"),
    ("Levi's 501 Original Fit Jeans Men's 32x32 Dark Stonewash", "Levi Strauss 501 Original 32x32 Dark Stone Wash"),
    ("Vitamix 5200 Blender Professional-Grade 64oz Black", "Vitamix 5200 64 oz Professional Blender Black"),
    ("Vitamix 5200 Blender Professional-Grade 64oz Black", "vitamix 5200 blender 64oz black professional grade"),
    ("LEGO Star Wars Millennium Falcon 75192 Building Set", "LEGO 75192 Millennium Falcon Star Wars Building Kit"),
    ("LEGO Star Wars Millennium Falcon 75192 Building Set", "Lego star wars 75192 millennium falcon set"),
    ("Dyson V15 Detect Absolute Cordless Vacuum Yellow", "Dyson V15 Detect Vacuum Cordless Yellow/Iron"),
    ("Dyson V15 Detect Absolute Cordless Vacuum Yellow", "dyson v15 detect absolute cordless hoover yellow"),
    ("KitchenAid Artisan Stand Mixer 5Qt Empire Red KSM150PSER", "KitchenAid KSM150PSER Artisan 5-Quart Stand Mixer Red"),
    ("KitchenAid Artisan Stand Mixer 5Qt Empire Red KSM150PSER", "kitchenaid 5 qt artisan stand mixer empire red ksm150"),
]

# ── NEGATIVE PAIRS (label = 0): Different products ──────────────────────────
# These are genuinely different products that the model must NOT confuse

NEGATIVE_PAIRS = [
    # Different shoe sizes — looks similar but NOT the same
    ("Nike Air Max 90 White Men's Size 10", "Nike Air Max 90 White Men's Size 11"),
    # Different storage — common trap
    ("Apple iPhone 14 Pro 256GB Deep Purple Unlocked", "Apple iPhone 14 Pro 512GB Deep Purple Unlocked"),
    # Different colors
    ("Samsung Galaxy S23 Ultra 5G 512GB Phantom Black", "Samsung Galaxy S23 Ultra 5G 512GB Green"),
    # Different models
    ("Sony WH-1000XM5 Wireless Noise Cancelling Headphones Black", "Sony WH-1000XM4 Wireless Noise Cancelling Headphones Black"),
    # Canon body vs kit
    ("Canon EOS R5 Mirrorless Camera Body Only", "Canon EOS R5 Mirrorless Camera with 24-105mm Lens Kit"),
    # Different jeans sizes
    ("Levi's 501 Original Fit Jeans Men's 32x32 Dark Stonewash", "Levi's 501 Original Fit Jeans Men's 34x30 Dark Stonewash"),
    # Different blender sizes
    ("Vitamix 5200 Blender Professional-Grade 64oz Black", "Vitamix 5200 Blender Professional-Grade 32oz Black"),
    # Completely different products
    ("Nike Air Max 90 White Men's Size 10", "Adidas Ultra Boost 22 Running Shoes White Men's 10"),
    ("Apple iPhone 14 Pro 256GB Deep Purple Unlocked", "Samsung Galaxy S23 Ultra 5G 256GB Phantom Black"),
    ("Sony WH-1000XM5 Wireless Noise Cancelling Headphones Black", "Bose QuietComfort 45 Bluetooth Headphones Black"),
    ("Canon EOS R5 Mirrorless Camera Body Only", "Nikon Z9 Mirrorless Camera Body Only"),
    ("Levi's 501 Original Fit Jeans Men's 32x32 Dark Stonewash", "Wrangler Cowboy Cut Jeans Men's 32x32 Dark Wash"),
    ("Vitamix 5200 Blender Professional-Grade 64oz Black", "Ninja BN701 Professional Plus Blender 72oz Black"),
    ("LEGO Star Wars Millennium Falcon 75192 Building Set", "LEGO Technic Bugatti Chiron 42083 Building Kit"),
    ("Dyson V15 Detect Absolute Cordless Vacuum Yellow", "Dyson V11 Torque Drive Cordless Vacuum Blue"),
    ("KitchenAid Artisan Stand Mixer 5Qt Empire Red KSM150PSER", "KitchenAid Professional 600 Stand Mixer 6Qt Silver"),
    # Trap: same brand, very similar name — different product
    ("Samsung Galaxy S23 Ultra 5G 512GB Phantom Black", "Samsung Galaxy S23 Plus 5G 512GB Phantom Black"),
    ("Apple iPhone 14 Pro 256GB Deep Purple Unlocked", "Apple iPhone 14 256GB Deep Purple Unlocked"),
    ("Dyson V15 Detect Absolute Cordless Vacuum Yellow", "Dyson V12 Detect Slim Absolute Cordless Vacuum Yellow"),
    ("Sony WH-1000XM5 Wireless Noise Cancelling Headphones Black", "Sony WH-CH720N Wireless Noise Cancelling Headphones Black"),
]


def build_dataframe() -> pd.DataFrame:
    """
    Combines positive and negative pairs into a single labeled DataFrame.

    Returns a DataFrame with columns: title1, title2, label
        label = 1 → same product (MATCH)
        label = 0 → different product (NO MATCH)
    """
    rows = []

    for t1, t2 in POSITIVE_PAIRS:
        rows.append({"title1": t1, "title2": t2, "label": 1})

    for t1, t2 in NEGATIVE_PAIRS:
        rows.append({"title1": t1, "title2": t2, "label": 0})

    df = pd.DataFrame(rows)

    # Shuffle the rows so positives and negatives are mixed
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def save_dataset(path: str = "data/product_pairs.csv"):
    """Builds and saves the dataset to a CSV file."""
    df = build_dataframe()
    df.to_csv(path, index=False)
    print(f"Dataset saved to {path}")
    print(f"  Total pairs : {len(df)}")
    print(f"  Match (1)   : {df['label'].sum()}")
    print(f"  No Match (0): {(df['label'] == 0).sum()}")
    return df


if __name__ == "__main__":
    save_dataset()
