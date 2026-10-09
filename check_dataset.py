from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

train = pd.read_csv(BASE_DIR / "data" / "train.csv")
evaluation = pd.read_csv(BASE_DIR / "data" / "eval.csv")

# Compare the categories
train_categories = set(train["category"].unique())
eval_categories = set(evaluation["category"].unique())

print("Categories in training:", len(train_categories))
print("Categories in evaluation:", len(eval_categories))

print("\nEvaluation categories:")
for category in sorted(eval_categories):
    print(f"{category}: present in training = {category in train_categories}")

# Check for duplicate resume text
print("\nDuplicate texts in training:", train["text"].duplicated().sum())
print("Duplicate texts in evaluation:", evaluation["text"].duplicated().sum())

# Check for overlap between training and evaluation
train_texts = set(train["text"].astype(str))
eval_texts = set(evaluation["text"].astype(str))

print("Texts appearing in both datasets:", len(train_texts & eval_texts))
