import os
import json
from datetime import datetime
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_score, recall_score, f1_score
import re
from collections import Counter
from rouge_score import rouge_scorer

model = SentenceTransformer("all-MiniLM-L6-v2")

class Evaluation:

    def __init__(self):
        pass

    def run_evaluation(self, retriever, generator):
        """
        This evaluation runs over test questions and logs the output to a date specific JSON file.
        """

        test_file = "evaluation/tests/test_sample_question_answer.json"
        log_dir = "evaluation/logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, datetime.now().strftime("%d-%m-%Y") + ".json")

        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                log_entries = json.load(f)
        else:
            log_entries = []

        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        for item in test_data:
            question = item["question"]
            task = item.get("task", "qa") 
            options = item.get("options", None)

            if "context" not in item:
                if task in ["qa", "classification"]:
                    retrieved_chunks, _ = retriever.query(question, k=1)
                    context = "\n\n".join(retrieved_chunks)
                else:
                    retrieved_chunks = []
                    context = ""
            else:
                context = item["context"]
                retrieved_chunks = [context]


            prompt = generator.build_prompt(
                context=context,
                task_input=question,
                mode=task,
                options=options
            )
            answer = generator.generate_answer(prompt, mode=task, options=options)
            if task == "classification":
                answer = answer.strip().lower()
                if "offensive" in answer:
                    answer = "Offensive"
                elif "non-offensive" in answer:
                    answer = "Non-offensive"
                else:
                    answer = "Unclear"

            if task == "mcq":
                valid_letters = [chr(97+i) for i in range(len(options))] 
                answer = answer.strip().lower()
                if answer not in valid_letters:

                    for letter in valid_letters:
                        if letter in answer:
                            answer = letter
                            break
                    else:
                        answer = "invalid"


            log_entry = {
                "question": question,
                "task": task,
                "retrieved_chunks": retrieved_chunks,
                "prompt": prompt,
                "generated_answer": answer,
                "timestamp": datetime.now().isoformat(timespec='seconds'),
                "group_id": "Team NNN"
            }

            log_entries.append(log_entry)

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_entries, f, indent=4)

        print(f"Evaluation complete. Log written to {log_file}")

    
    def semantic_similarity(self, expected: str, actual: str) -> float:
        """
        Compute cosine similarity between expected and actual answer embeddings.
        Returns a float between 0 and 1.
        """
        embeddings = model.encode([expected, actual], convert_to_tensor=True)
        return float(util.cos_sim(embeddings[0], embeddings[1]))


    def evaluate_model_performance(self, test_file_path: str, log_file_path: str, threshold: float = 0.3):
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        with open(test_file_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        with open(log_file_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)

        matched = 0
        unmatched = 0
        detailed_results = []

        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_rouge = 0
        f1_count = 0

        for test_item in test_data:
            question = test_item["question"]
            expected_answer = test_item["expected_answer"]

            matching_logs = [log for log in log_data if log["question"] == question]
            if not matching_logs:
                unmatched += 1
                detailed_results.append((question, expected_answer, None, 0, 0))
                continue

            generated_answer = matching_logs[-1]["generated_answer"]

            precision, recall, f1 = self.compute_f1_score(expected_answer, generated_answer)

            rouge_score = scorer.score(expected_answer, generated_answer)["rougeL"].fmeasure

            if rouge_score >= threshold:
                matched += 1
            else:
                unmatched += 1

            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_rouge += rouge_score
            f1_count += 1

            detailed_results.append((question, expected_answer, generated_answer, f1, rouge_score))

        avg_precision = total_precision / f1_count if f1_count else 0
        avg_recall = total_recall / f1_count if f1_count else 0
        avg_f1 = total_f1 / f1_count if f1_count else 0
        avg_rouge = total_rouge / f1_count if f1_count else 0

        print(f"\nToken-level Evaluation Metrics:")
        print(f"Average Precision: {avg_precision:.2f}")
        print(f"Average Recall:    {avg_recall:.2f}")
        print(f"Average F1 Score:  {avg_f1:.2f}")
        print(f"Average ROUGE-L:   {avg_rouge:.2f}")

        return matched, unmatched, detailed_results

    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            return re.sub(r'[^\w\s]', '', text)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))


    def compute_f1_score(self, expected, predicted):
        """Compute token-level precision, recall, and F1 score."""
        expected_tokens = self.normalize_answer(expected).split()
        predicted_tokens = self.normalize_answer(predicted).split()

        common = Counter(expected_tokens) & Counter(predicted_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0, 0.0, 0.0

        precision = num_same / len(predicted_tokens)
        recall = num_same / len(expected_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    def visualize_results(self, matched: int, unmatched: int):
        labels = [f"Matched ({matched})", f"Unmatched ({unmatched})"]
        counts = [matched, unmatched]
        colors = ["green", "red"]

        plt.figure(figsize=(6, 6))
        plt.pie(
            counts,
            labels=labels,
            autopct='%1.1f%%',
            startangle=140,
            colors=colors,
            textprops={'fontsize': 12}
        )
        plt.title("LLM Evaluation")
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

