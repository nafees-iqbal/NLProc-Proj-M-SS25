import os
import json
from datetime import datetime

class Evaluation:

    def __init__(self):
        pass

    def run_evaluation(self, retriever, generator):
        """
        this evaluation using test questions and logs the output to a date specific JSON file.

        Parameters:
        retriever: an instance of Retriever class
        generator: an instance of Generator class
        """
        test_file = "evaluation/tests/test_generated_answer.json"
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
            retrieved_chunks, _ = retriever.query(question, k=1)
            context = "\n\n".join(retrieved_chunks)

            prompt = generator.build_prompt(context=context, question=question)
            answer = generator.generate_answer(prompt)

            log_entry = {
                "question": question,
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

