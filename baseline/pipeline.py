# pipeline.py

import os
import sys
from retriever.retreiver import Retriever
from generator.generator import Generator

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from evaluation.evaluation import Evaluation
from evaluation.tests.unit_test import run_unit_test

retriever = Retriever()
generator = Generator()
evaluation = Evaluation()


index_path = "retriever_index"
index_file = f"{index_path}.faiss"
text_file = f"{index_path}_texts.pkl"
courses_folder = "baseline/data/uni-bamberg-courses/dsg-dsam-m"


if os.path.exists(index_file) and os.path.exists(text_file):
    print("Loading existing FAISS index and text chunks...")
    retriever.load(index_path)
else:
    print("Index not found. Building from scratch...")
    retriever.add_documents(courses_folder)
    retriever.save(index_path)
    print("Index built and saved.")


evaluation.run_evaluation(retriever, generator)
run_unit_test(evaluation)
