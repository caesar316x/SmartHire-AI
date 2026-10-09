from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train.csv"

# Load and clean the dataset
df = pd.read_csv(DATA_PATH)
df = df[["text", "category"]].dropna()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]
df = df.drop_duplicates(subset="text")

X = df["text"]
y = df["category"]

# Use the same split for both models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),
    "Linear SVC": LinearSVC(
        class_weight="balanced",
        random_state=42
    )
}


for name, classifier in models.items():
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True
        )),
        ("classifier", classifier)
    ])

    print(f"\n{'=' * 15} {name} {'=' * 15}")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Accuracy:", round(
        accuracy_score(y_test, predictions) * 100, 2
    ), "%")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    print(
        "Macro F1 score:",
        round(f1_score(y_test, predictions, average="macro"), 3)
    )

    print("\nConfusion matrix:")
    labels = sorted(y_test.unique())
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    print(pd.DataFrame(
        matrix,
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels]
    ))

    
