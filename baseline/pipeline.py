# put the code that calls all the different components here

from retriever.retreiver import Retriever


# Create retriever instance
retriever = Retriever()

retriever.add_documents("baseline/data/text-samples")

def test_retriever_accuracy(retriever, query, expected_keywords, k=1):
    """
    Runs a query and checks whether the top k retrieved chunks contain any expected keywords.

    Parameters:
    - retriever: the Retriever instance
    - query (str): the input query
    - expected_keywords: words we expect to see in retrieved text
    - k (int): top k chunks to retrieve
    """
    results, _ = retriever.query(query, k=k)
    print(f"Query: '{query}'")

    match_found = False
    for i, chunk in enumerate(results):
        print(f"\nResult {i+1}:")
        print(chunk[:300] + '...')
        if any(keyword.lower() in chunk.lower() for keyword in expected_keywords):
            match_found = True

    if match_found:
        print("Test Passed: Expected content found in retrieved chunks.")
    else:
        print("Test Failed: Expected keywords not found.")

test_retriever_accuracy(
    retriever,
    query="a princess who dances with magical shoes",
    expected_keywords=["twelve", "princess", "dance"]
)
