# pipeline.py

import os
from retriever.retreiver import Retriever
from generator.generator import Generator

retriever = Retriever()
generator = Generator()

index_path = "retriever_index"
index_file = f"{index_path}.faiss"
text_file = f"{index_path}_texts.pkl"
courses_folder = "baseline/data/uni-bamberg-courses/dsg-dsam-m"

example_context = (
    "In a distributed system, nodes communicate over a network and do not share memory. "
    "Consistency and fault tolerance are major concerns in such systems."
)

example_question = (
    "What are the key challenges in maintaining consistency in distributed systems?"
)


if os.path.exists(index_file) and os.path.exists(text_file):
    print("Loading existing FAISS index and text chunks...")
    retriever.load(index_path)
else:
    print("Index not found. Building from scratch...")
    retriever.add_documents(courses_folder)
    retriever.save(index_path)
    print("Index built and saved.")


def run_rag_pipeline(query, k=3):
    print(f"\nQuery: {query}")

    retrieved_chunks, _ = retriever.query(query, k=k)
    combined_context = "\n\n".join(retrieved_chunks)

    prompt = generator.build_prompt(
        context=combined_context,
        task=query,
        example_context=example_context,
        example_output=example_question
    )
    answer = generator.generate_answer(prompt)

    print("Generated Answer:\n", answer)

# 🔍 Run query
run_rag_pipeline(
    query = "What is middleware?",
    k=1
)
