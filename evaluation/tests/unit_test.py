

from evaluation.evaluation import Evaluation
from datetime import datetime

def run_unit_test(evaluation):
    today = datetime.now().strftime("%d-%m-%Y")
    test_path = "evaluation/tests/test_sample_question_answer.json"
    log_path = f"evaluation/logs/{today}.json"

    matched, unmatched, results = evaluation.evaluate_model_performance2(test_path, log_path)

    print(f"\nUnit Test Summary: {matched} matched / {matched + unmatched} total\n")

    for question, expected, actual, f1, rouge, bert_f1 in results:
        print(f"Q: {question}")
        print(f"Expected: {expected}")
        print(f"Actual: {actual}")
        print(f"Token-level F1 Score: {f1:.2f}")
        print(f"ROUGE-L Score:        {rouge:.2f}")
        print(f"BERTScore F1:         {bert_f1:.2f}\n")

    evaluation.visualize_results(matched, unmatched)

