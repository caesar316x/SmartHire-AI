
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
MODEL_PATH = BASE_DIR / "resume_classifier.joblib"

# Load the training dataset
df = pd.read_csv(DATA_PATH)

# Keep only the columns needed for text classification
df = df[["text", "category"]].dropna()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

# Remove exact duplicate resume texts
df = df.drop_duplicates(subset="text")

X = df["text"]
y = df["category"]

# Split the data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Build the text classification pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

# Train the model
print("Training the resume classifier...")
model.fit(X_train, y_train)

# Evaluate it on the held-out test set
predictions = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save the trained model
joblib.dump(model, MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)
print("Training complete!")
