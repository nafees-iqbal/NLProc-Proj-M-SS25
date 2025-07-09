# pipeline.py
import os
import sys
import json
from datetime import datetime
import re

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from baseline.retriever.retreiver import Retriever
from baseline.generator.generator import Generator

retriever = Retriever()
generator = Generator()

index_path = "retriever_index"
index_file = f"{index_path}.faiss"
text_file = f"{index_path}_texts.pkl"
courses_folder = "baseline/data/uni-bamberg-courses/dsg-dsam-m"

log_dir = "evaluation/logs"
os.makedirs(log_dir, exist_ok=True)

class Pipeline:
    def __init__(self):
        pass

    def setup_index(self):
        if os.path.exists(index_file) and os.path.exists(text_file):
            print("Loading existing FAISS index and text chunks...")
            retriever.load(index_path)
        else:
            print("Index not found. Building from scratch...")
            retriever.add_documents(courses_folder)
            retriever.save(index_path)
            print("Index built and saved.")

    def process_question(self, question, task="summarization", options=None, context=None, group_id="Team NNN"):
        """
        This runs the pipeline like run_evaluation but for a single question.
        Returns dict with answer, context, prompt and writes to log.
        """
        if '?' in question:
            task = "qa"
        elif re.search(r"\bsummariz(e|ation)\b", question, re.IGNORECASE):
            task = "summarization"
        elif "options" in question.lower():
            task = "mcq"
        else:
            task = "classification"

        if not context:
            if task in ["qa", "classification"]:
                retrieved_chunks, _ = retriever.query(question, k=3)
                context = "\n\n".join(retrieved_chunks)
            else:
                retrieved_chunks = []
                context = ""
        else:
            retrieved_chunks = [context]

        options = []
        if task == "mcq":
            matches = re.findall(r'[a-dA-D]\)\s*([^\n]+)', question)
            options = [match.strip() for match in matches]

        prompt = generator.build_prompt(
            context=context,
            task_input=question,
            mode=task,
            options=options
        )
    
        answer = generator.generate_answer(prompt, mode=task, options=options)

        print(answer)

        if task == "mcq":
            valid_letters = [chr(97+i) for i in range(len(options or []))]
            answer = answer.strip().lower()
            if answer not in valid_letters:
                for letter in valid_letters:
                    if letter in answer:
                        answer = letter
                        break
                else:
                    answer = "invalid"

        log_file = os.path.join(log_dir, datetime.now().strftime("%d-%m-%Y") + ".json")
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                log_entries = json.load(f)
        else:
            log_entries = []

        log_entry = {
            "question": question,
            "task": task,
            "retrieved_chunks": retrieved_chunks,
            "prompt": prompt,
            "context": context,
            "generated_answer": answer,
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "group_id": group_id
        }
        '''log_entries.append(log_entry)

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_entries, f, indent=4)'''

        return {
            "answer": answer,
            "context": context,
            "prompt": prompt
        }

    def run_batch_evaluation(self, evaluation):
        evaluation.run_evaluation(retriever, generator)
