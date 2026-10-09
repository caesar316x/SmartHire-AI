
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "resume_classifier.joblib"

# Load the saved model
model = joblib.load(MODEL_PATH)

sample_resumes = [
    """
    Software developer with experience in Java, Python,
    SQL, database management, software testing, and web development.
    """,
    """
    Accountant with experience in financial statements,
    bookkeeping, tax preparation, auditing, and budgeting.
    """,
    """
    Registered nurse with experience in patient care,
    clinical assessments, medical records, and healthcare services.
    """,
    """
    Graphic designer experienced in logo design, branding,
    Adobe Photoshop, illustration, and visual communication.
    """,
    """
    Experienced teacher specializing in classroom instruction,
    lesson planning, student assessment, and curriculum development.
    """
]

for i, resume in enumerate(sample_resumes, start=1):
    # Predict the most likely category
    prediction = model.predict([resume])[0]

    # Get scores for all possible categories
    scores = model.decision_function([resume])[0]

    # Find the three categories with the highest scores
    top_indices = scores.argsort()[-3:][::-1]
    categories = model.classes_

    print(f"\nSample Resume {i}")
    print("Predicted category:", prediction)
    print("Top three candidate categories:")

    for index in top_indices:
        print(f"  {categories[index]}: {scores[index]:.3f}")
