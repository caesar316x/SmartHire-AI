
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "resume_classifier.joblib"

# Load the model once when this file is imported
model = joblib.load(MODEL_PATH)


def predict_resume_category(resume_text):
    """
    Predict a job category from extracted resume text.
    Returns the predicted category as a string.
    """
    if not isinstance(resume_text, str) or not resume_text.strip():
        raise ValueError("Resume text cannot be empty.")

    prediction = model.predict([resume_text])[0]
    return prediction


if __name__ == "__main__":
    sample_text = """
    Software developer with experience in Java, Python,
    SQL, database management, and software development.
    """

    category = predict_resume_category(sample_text)
    print("Predicted job category:", category)

    try:
        predict_resume_category("")
    except ValueError as error:
        print("Empty input test:", error)
