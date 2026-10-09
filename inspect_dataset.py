from pathlib import Path
import pandas as pd

# Find the project folder and dataset files
BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
EVAL_PATH = BASE_DIR / "data" / "eval.csv"

# Inspect both datasets
for name, path in [("TRAINING", TRAIN_PATH), ("EVALUATION", EVAL_PATH)]:
    print(f"\n{'=' * 15} {name} DATASET {'=' * 15}")

    if not path.exists():
        print(f"File not found: {path}")
        continue

    df = pd.read_csv(path)

    print("\nDataset size (rows, columns):", df.shape)
    print("\nColumn names:", df.columns.tolist())

    print("\nMissing values per column:")
    print(df.isna().sum())

    # Check category labels if the column exists
    if "category" in df.columns:
        print("\nNumber of unique categories:", df["category"].nunique())
        print("\nRecords per category:")
        print(df["category"].value_counts().to_string())

    # Show a short sample of resume text
    if "text" in df.columns and not df.empty:
        print("\nFirst resume sample (up to 500 characters):")
        print(str(df["text"].iloc[0])[:500])
