from predict_resume import predict_resume_category

def analyze_resume(resume_text):
    category = predict_resume_category(resume_text)

    return {
        "predicted_category": category
    }


if __name__ == "__main__":
    sample_resume = """
    Software developer with experience in Java, Python,
    SQL, database management, and software development.
    """

    result = analyze_resume(sample_resume)

    print("Integration test result:")
    print(result)
