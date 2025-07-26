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
from bert_score import score as bert_score

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
                answer = 'a'


            log_entry = {
                "question": question,
                "task": task,
                "retrieved_chunks": retrieved_chunks,
                "prompt": prompt,
                "context": context,
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
    
    def evaluate_bert_score(expected_list, predicted_list, lang='en', model_type='microsoft/deberta-xlarge-mnli'):
        """
        Compute BERTScore between lists of expected and predicted answers.
        
        Args:
            expected_list (List[str]): Ground truth answers.
            predicted_list (List[str]): Model-generated answers.
            lang (str): Language code (default: 'en').
            model_type (str): Model to use for BERTScore (default: DeBERTa MNLI, good for English).

        Returns:
            avg_precision, avg_recall, avg_f1, all_f1s: Averages and per-sample F1s.
        """
        assert len(expected_list) == len(predicted_list), "Expected and predicted lists must be the same length."

        P, R, F1 = bert_score(predicted_list, expected_list, lang=lang, model_type=model_type, verbose=True)
        
        avg_precision = P.mean().item()
        avg_recall    = R.mean().item()
        avg_f1        = F1.mean().item()

        return avg_precision, avg_recall, avg_f1, F1.tolist()


    def evaluate_model_performance(self, test_file_path: str, log_file_path: str, threshold: float = 0.60):
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

            if test_item["task"] == "summarization":
                context = test_item.get("context", "").strip()
                matching_logs = [log for log in log_data if log["task"] == "summarization" and log.get("context", "").strip() == context]
            else:
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
    

    def evaluate_model_performance2(self, test_file_path: str, log_file_path: str, threshold: float = 0.60):
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        with open(test_file_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        with open(log_file_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)

        detailed_results = []

        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_rouge = 0
        f1_count = 0

        matched = 0
        unmatched = 0

        # For BERTScore calculation
        question_pairs = []
        bert_expected = []
        bert_predicted = []

        for test_item in test_data:
            question = test_item["question"]
            expected_answer = test_item["expected_answer"]

            if test_item["task"] == "summarization":
                context = test_item.get("context", "").strip()
                matching_logs = [log for log in log_data if log["task"] == "summarization" and log.get("context", "").strip() == context]
            else:
                matching_logs = [log for log in log_data if log["question"] == question]

            if not matching_logs:
                unmatched += 1
                detailed_results.append((question, expected_answer, None, 0, 0, 0))  # F1, ROUGE, BERT F1
                continue

            generated_answer = matching_logs[-1]["generated_answer"]

            precision, recall, f1 = self.compute_f1_score(expected_answer, generated_answer)
            rouge_score = scorer.score(expected_answer, generated_answer)["rougeL"].fmeasure

            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_rouge += rouge_score
            f1_count += 1

            bert_expected.append(expected_answer)
            bert_predicted.append(generated_answer)
            question_pairs.append((question, expected_answer, generated_answer, f1, rouge_score))  # BERT to be appended later

        # Calculate BERTScore once
        if bert_expected and bert_predicted:
            P, R, F1 = bert_score(bert_predicted, bert_expected, lang='en', model_type='microsoft/deberta-xlarge-mnli', verbose=True)
            bert_f1_list = F1.tolist()
        else:
            bert_f1_list = []

        for i, (question, expected, actual, f1, rouge) in enumerate(question_pairs):
            bert_f1 = bert_f1_list[i]
            if bert_f1 >= threshold:
                matched += 1
            else:
                unmatched += 1
            detailed_results.append((question, expected, actual, f1, rouge, bert_f1))

        avg_precision = total_precision / f1_count if f1_count else 0
        avg_recall = total_recall / f1_count if f1_count else 0
        avg_f1 = total_f1 / f1_count if f1_count else 0
        avg_rouge = total_rouge / f1_count if f1_count else 0
        avg_bert_f1 = sum(bert_f1_list) / len(bert_f1_list) if bert_f1_list else 0

        print(f"\nToken-level Evaluation Metrics:")
        print(f"Average Precision:    {avg_precision:.2f}")
        print(f"Average Recall:       {avg_recall:.2f}")
        print(f"Average F1 Score:     {avg_f1:.2f}")
        print(f"Average ROUGE-L:      {avg_rouge:.2f}")
        print(f"Average BERTScore F1: {avg_bert_f1:.2f}")

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
        expected_tokens = self.normalize_answer(expected).split() # F1 is a token-level metric, so we compare word-by-word, not characters or full sentences
        predicted_tokens = self.normalize_answer(predicted).split()

        common = Counter(expected_tokens) & Counter(predicted_tokens)
        num_same = sum(common.values()) # num_same counts how many tokens the model got right.

        if num_same == 0:
            return 0.0, 0.0, 0.0

        precision = num_same / len(predicted_tokens) # how much was correct
        recall = num_same / len(expected_tokens) # how much did it cover
        f1 = 2 * precision * recall / (precision + recall) # Harmonic mean balances precision and recall; high only if both are high
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
    def evaluate_single_prediction(self, question, expected_answer, generated_answer, threshold=0.25):
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        precision, recall, f1 = self.compute_f1_score(expected_answer, generated_answer)

        rouge = scorer.score(expected_answer, generated_answer)["rougeL"].fmeasure

        P, R, F1 = bert_score([generated_answer], [expected_answer],
                            lang='en', model_type='microsoft/deberta-xlarge-mnli', verbose=False)
        bert_f1 = F1.tolist()[0]

        is_match = bert_f1 >= threshold

        return {
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "rouge_l": rouge,
            "bert_f1": bert_f1,
            "bert_match_threshold": threshold,
            "is_match": is_match
        }

