from pathlib import Path
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
MODEL_PATH = BASE_DIR / "resume_classifier.joblib"

# Load and clean the dataset
df = pd.read_csv(DATA_PATH)
df = df[["text", "category"]].dropna()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]
df = df.drop_duplicates(subset="text")

X = df["text"]
y = df["category"]

# Create the selected model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True
    )),
    ("classifier", LinearSVC(
        class_weight="balanced",
        random_state=42
    ))
])

# Train using all available training records
print("Training the final Linear SVC model...")
model.fit(X, y)

# Save the trained pipeline
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved successfully to: {MODEL_PATH}")
print("Number of training resumes:", len(df))
print("Number of supported categories:", len(model.classes_))
print("Categories:", ", ".join(model.classes_))
