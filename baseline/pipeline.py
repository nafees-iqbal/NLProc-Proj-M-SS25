# pipeline.py

import os
from retriever.retreiver import Retriever

retriever = Retriever()

index_path = "retriever_index"
index_file = f"{index_path}.faiss"
text_file = f"{index_path}_texts.pkl"

if os.path.exists(index_file) and os.path.exists(text_file):
    print("Loading existing FAISS index and text chunks...")
    retriever.load(index_path)
else:
    print("Index not found. Building from scratch...")
    retriever.add_documents("baseline/data/text-samples")
    retriever.save(index_path)
    print("Index built and saved.")

# -------------------------------------------------------------------
# Function to test retriever output
def test_retriever_accuracy(retriever, query, expected_keywords, k=3):
    """
    Runs a query and checks whether the top-k retrieved chunks contain any expected keywords.

    Parameters:
    - retriever: the Retriever instance
    - query (str): the input query string
    - expected_keywords (list of str): keywords we expect to find in the top-k results
    - k (int): number of top results to check
    """
    results, _ = retriever.query(query, k=k)
    print(f"Query: '{query}'")
    match_found = False
    matched_keywords = set()

    for i, chunk in enumerate(results):
        print(f"\nResult {i+1}:\n{chunk[:300]}...\n")
        for keyword in expected_keywords:
            if keyword.lower() in chunk.lower():
                matched_keywords.add(keyword.lower())

    if matched_keywords:
        print(f"Test Passed: Found keyword(s): {', '.join(matched_keywords)}")
        match_found = True
    else:
        print("Test Failed: None of the expected keywords were found.")
        print(f"Expected one of: {', '.join(expected_keywords)}")

    return match_found

test_retriever_accuracy(
    retriever,
    query="a princess who dances with magical shoes",
    expected_keywords=["twelve", "princess", "dance"],
    k=3
)
