import os
import json
from datetime import datetime
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
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
            task = item.get("task", "qa")  # default to 'qa'
            options = item.get("options", None)

            retrieved_chunks, _ = retriever.query(question, k=1)
            context = "\n\n".join(retrieved_chunks)

            prompt = generator.build_prompt(
                context=context,
                task_input=question,
                mode=task,
                options=options
            )
            answer = generator.generate_answer(prompt)
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


    def evaluate_model_performance(self, test_file_path: str, log_file_path: str, threshold: float = 0.5):
        """Compare generated answers from the log file with expected answers from test file."""
        with open(test_file_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        with open(log_file_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)

        matched = 0
        unmatched = 0
        detailed_results = []

        for test_item in test_data:
            question = test_item["question"]
            expected_answer = test_item["expected_answer"]

            matching_logs = [log for log in log_data if log["question"] == question]
            if not matching_logs:
                unmatched += 1
                detailed_results.append((question, expected_answer, None, 0))
                continue

            generated_answer = matching_logs[-1]["generated_answer"]  # Use the latest if duplicates
            score = self.semantic_similarity(expected_answer, generated_answer)
            if score >= threshold:
                matched += 1
            else:
                unmatched += 1
            detailed_results.append((question, expected_answer, generated_answer, score))

        return matched, unmatched, detailed_results


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

