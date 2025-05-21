

from evaluation.evaluation import Evaluation
from datetime import datetime

def run_unit_test(evaluation):
    today = datetime.now().strftime("%d-%m-%Y")
    test_path = "evaluation/tests/test_sample_question_answer.json"
    log_path = f"evaluation/logs/{today}.json"

    matched, unmatched, results = evaluation.evaluate_model_performance(test_path, log_path)

    print(f"\nUnit Test Summary: {matched} matched / {matched + unmatched} total\n")
    for q, expected, actual, score in results:
        print(f"Q: {q}")
        print(f"Expected: {expected}")
        print(f"Actual: {actual}")
        print(f"Score: {score:.2f}\n")

    evaluation.visualize_results(matched, unmatched)